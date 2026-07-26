import os
import json
import shutil
import subprocess
import threading
import zipfile
import io
import re
import time
import logging
import gc
import uuid
from script_repair import EXPLICIT_SILENCE_MS
from verbalization import (SET_APART_HINT, VERBALIZE, classify,
                           extract_delivery_cues, is_pictographic_kana,
                           split_bracketed_spans, strip_emoji_dividers)
from utils import (atomic_json_write, safe_load_json, is_oom_failure,
                   get_app_config_path, is_nonverbal_text)
from config_settings import load_app_config
from tts import (
    TTSEngine,
    combine_audio_with_pauses,
    compute_timeline,
    sanitize_filename,
    voice_category,
    DEFAULT_PAUSE_MS,
    SAME_SPEAKER_PAUSE_MS
)
from pydub import AudioSegment

MAX_CHUNK_CHARS = 500


def _new_chunk_uid():
    """Stable per-chunk id for audio filenames.

    Unlike chunk['id'] (which is the list position and gets renumbered on every
    insert/delete), this is assigned once and never changes, so a chunk's audio
    filename can't collide with a neighbour's after the list shifts.
    """
    return uuid.uuid4().hex[:12]




def get_speaker(entry):
    """Get speaker from entry, checking both 'speaker' and 'type' fields."""
    return entry.get("speaker") or entry.get("type") or ""


def _is_structural_text(text):
    """Check if text is a title, chapter heading, dedication, or other structural fragment."""
    stripped = text.strip()
    if not stripped:
        return True
    # Very short and not a full sentence (no sentence-ending punctuation)
    if len(stripped) < 80 and not stripped[-1] in '.!?':
        return True
    return False


def _make_chunk(speaker, text, instruct, pause_after=None):
    """Build a chunk dict, omitting pause_after when None for clean JSON."""
    chunk = {"speaker": speaker, "text": text, "instruct": instruct}
    if pause_after is not None:
        chunk["pause_after"] = pause_after
    return chunk


def split_on_unspeakable(entry, scene_break_pause_ms):
    """Split one entry into speakable parts, handling each glyph by class.

    A scene break inside prose becomes a boundary carrying pause_after rather
    than a character the narrator reads aloud. Verbalized symbols become words.
    Delivery cues move into instruct. Anything unmapped is left in place and
    reported, so an unknown glyph stays visible instead of being silently voiced.

    Returns (parts, review_chars). Never mutates the entry it is given.
    """
    # A run of repeated emoji is a section divider, the same role as a row of
    # box-drawing characters, so it is folded to a scene-break glyph before
    # classification. Done first because each emoji is category So on its own
    # and would otherwise be reported for review one character at a time.
    text, _dividers = strip_emoji_dividers(str(entry.get("text") or ""))
    review = []
    for position, char in enumerate(text):
        neighbours = text[max(0, position - 2):position] + text[position + 1:position + 3]
        if classify(char) == "review" or is_pictographic_kana(char, neighbours):
            review.append(char)
    segments, current = [], []
    for char in text:
        kind = classify(char)
        if kind == "scene_break":
            segments.append("".join(current))
            current = []
        elif kind == "verbalize":
            current.append(VERBALIZE[char])
        else:
            current.append(char)
    segments.append("".join(current))

    parts = []
    for index, segment in enumerate(segments):
        segment_parts = []
        # A bracketed span is delivered differently from the prose around it,
        # so it becomes its own part rather than inheriting the entry's
        # instruct. These boundaries are NOT scene breaks and carry no pause.
        for fragment, set_apart in split_bracketed_spans(segment):
            cleaned, hints = extract_delivery_cues(fragment)
            if set_apart:
                hints = hints + [SET_APART_HINT]
            cleaned = " ".join(cleaned.split()) if cleaned.strip() else ""
            if not cleaned:
                continue
            part = dict(entry)
            part["text"] = cleaned
            if hints:
                existing = str(part.get("instruct") or "").strip()
                part["instruct"] = " ".join(filter(None, [existing] + hints))
            segment_parts.append(part)

        if not segment_parts:
            # A scene break with nothing before it has no audio anchor to
            # attach a pause to, mirroring the leading-nonverbal rule below.
            if parts:
                parts[-1]["pause_after"] = max(
                    int(parts[-1].get("pause_after") or 0), scene_break_pause_ms)
            continue
        # The gap between segments is the scene break, so the pause goes on the
        # last part of every segment except the final one.
        if index < len(segments) - 1:
            segment_parts[-1]["pause_after"] = max(
                int(segment_parts[-1].get("pause_after") or 0),
                scene_break_pause_ms)
        parts.extend(segment_parts)

    return parts, review


def log_review_characters(review):
    """Warn about glyphs no rule could classify, grouped by character.

    split_on_unspeakable leaves these in the text on purpose - guessing at an
    unknown symbol is worse than flagging it - but the report previously went
    nowhere, so "flagged" meant the narrator read it aloud with nobody told.
    """
    if not review:
        return
    counts = {}
    for item in review:
        counts.setdefault(item["character"], []).append(item["entry_index"])
    summary = ", ".join(
        f"{character!r} x{len(indexes)} (first at entry {min(indexes)})"
        for character, indexes in sorted(counts.items()))
    logger.warning(
        "Unmapped symbols left in TTS text and needing review: %s", summary)


def get_speakable_entries(script_entries, review_sink=None):
    """Return copied TTS entries, converting nonverbal marks into a pause.

    Punctuation-only and block-glyph dialogue remains in annotated_script.json
    for fidelity, but sending it to TTS produces noise or failures. A nonverbal
    entry extends the preceding speakable entry's pause without mutating caller
    data. Leading nonverbal marks have no audio anchor and are omitted.
    """
    speakable = []
    for entry_index, source_entry in enumerate(script_entries):
        entry = dict(source_entry)
        if is_nonverbal_text(entry.get("text")):
            if not speakable:
                continue
            previous = speakable[-1]
            previous["pause_after"] = max(
                int(previous.get("pause_after") or 0), DEFAULT_PAUSE_MS)
        else:
            parts, review = split_on_unspeakable(entry, EXPLICIT_SILENCE_MS)
            if review_sink is not None:
                review_sink.extend({"entry_index": entry_index,
                                    "character": character}
                                   for character in review)
            speakable.extend(parts)
    return speakable


def group_into_chunks(script_entries, max_chars=MAX_CHUNK_CHARS,
                      review_sink=None):
    """Group consecutive entries by same speaker into chunks up to max_chars"""
    script_entries = get_speakable_entries(script_entries,
                                           review_sink=review_sink)
    if not script_entries:
        return []

    chunks = []
    current_speaker = get_speaker(script_entries[0])
    current_text = script_entries[0].get("text", "")
    current_instruct = script_entries[0].get("instruct", "")
    current_pause_after = script_entries[0].get("pause_after")

    for entry in script_entries[1:]:
        speaker = get_speaker(entry)
        text = entry.get("text", "")
        instruct = entry.get("instruct", "")

        # Don't merge structural text (titles, chapter headings, dedications),
        # and never merge past a pending pause: pause_after belongs to the
        # boundary after current_text, so absorbing the next entry moves that
        # silence to the end of the combined text. A scene break inside a
        # paragraph then played its pause after both sentences instead of
        # between them, which is not audible as a break at all.
        if (speaker == current_speaker and instruct == current_instruct
                and not current_pause_after
                and not _is_structural_text(current_text)
                and not _is_structural_text(text)):
            combined = current_text + " " + text
            if len(combined) <= max_chars:
                current_text = combined
                # Last merged entry's pause_after wins
                current_pause_after = entry.get("pause_after", current_pause_after)
            else:
                chunks.append(_make_chunk(current_speaker, current_text, current_instruct, current_pause_after))
                current_text = text
                current_instruct = instruct
                current_pause_after = entry.get("pause_after")
        else:
            chunks.append(_make_chunk(current_speaker, current_text, current_instruct, current_pause_after))
            current_speaker = speaker
            current_text = text
            current_instruct = instruct
            current_pause_after = entry.get("pause_after")

    # Don't forget the last chunk
    chunks.append(_make_chunk(current_speaker, current_text, current_instruct, current_pause_after))

    return chunks

logger = logging.getLogger(__name__)

class ProjectManager:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.script_path = os.path.join(root_dir, "annotated_script.json")
        self.chunks_path = os.path.join(root_dir, "chunks.json")
        self.voicelines_dir = os.path.join(root_dir, "voicelines")
        self.voice_config_path = os.path.join(root_dir, "voice_config.json")
        # config.json placement must match app.py's get_app_config_path exactly:
        # legacy app/config.json when the data dir is the repo root, else
        # <data_dir>/config.json. root_dir here IS the data dir (ProjectManager is
        # constructed with DATA_DIR), so resolve against this file's own repo
        # root/app dir — NOT root_dir/app, which points at <data_dir>/app and
        # silently misses the config when ALEXANDRIA_DATA_DIR relocates data.
        _app_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = get_app_config_path(root_dir, os.path.dirname(_app_dir), _app_dir)

        # Ensure voicelines dir exists
        os.makedirs(self.voicelines_dir, exist_ok=True)

        self.engine = None
        self._chunks_lock = threading.Lock()  # Thread-safe chunks.json writes
        self._uids_backfilled = False  # see _read_chunks
        self._config_cache = None
        self._config_mtime = None
        self._config_lock = threading.Lock()

    def invalidate_config_cache(self):
        """Force the next config read to reload config.json from disk."""
        with self._config_lock:
            self._config_cache = None
            self._config_mtime = None

    def _read_config(self):
        """Load config.json, caching by mtime to avoid repeated disk reads/parses
        across hot-path calls (get_engine, _load_tts_config, ...)."""
        with self._config_lock:
            try:
                mtime = os.path.getmtime(self.config_path)
            except OSError:
                mtime = None
            if self._config_cache is None or mtime != self._config_mtime:
                self._config_cache = load_app_config(self.config_path)
                self._config_mtime = mtime
            return self._config_cache

    def get_engine(self):
        if self.engine:
            return self.engine

        config = self._read_config()

        try:
            self.engine = TTSEngine(config)
            print(f"TTS engine initialized (mode={self.engine.mode})")
            return self.engine
        except Exception as e:
            print(f"Failed to initialize TTS engine: {e}")
            return None

    def _load_tts_config(self):
        """Load TTS config section from config.json for pause defaults."""
        return self._read_config().get("tts", {})

    def load_chunks(self):
        """Load chunks from disk, regenerating if missing or corrupted.
        
        Uses _chunks_lock to prevent race conditions with concurrent saves.
        """
        with self._chunks_lock:
            return self._read_chunks()

    def _read_chunks(self):
        """Read chunks.json, regenerating from script if missing or corrupted.

        Internal method - callers must hold _chunks_lock. Always returns a
        list (possibly empty), never None.
        """
        chunks = safe_load_json(self.chunks_path)
        if chunks is not None:
            # Backfill stable uids for chunks saved before uid-based filenames.
            # Assigned once and persisted so a chunk keeps the same audio file
            # name across later inserts/deletes. _modify_chunk (the primitive
            # behind every per-chunk status update during generation) calls
            # _read_chunks() fresh on every single call, so without caching
            # that the backfill already ran, this O(N) scan would repeat on
            # every status transition of every chunk - O(N^2) total work for
            # a full generation pass. insert_chunk/restore_chunk already
            # assign a uid at creation time, so once every persisted chunk
            # has one, nothing in this app can ever produce a uid-less chunk
            # again - safe to check once per instance lifetime.
            if not self._uids_backfilled:
                changed = False
                for chunk in chunks:
                    if isinstance(chunk, dict) and not chunk.get("uid"):
                        chunk["uid"] = _new_chunk_uid()
                        changed = True
                if changed:
                    atomic_json_write(chunks, self.chunks_path)
                self._uids_backfilled = True
            return chunks
        if os.path.exists(self.chunks_path):
            # Back the corrupted file up rather than deleting it - regenerating
            # resets every chunk to pending/no-audio, so keep a copy in case the
            # user's generation progress can be salvaged from it.
            backup = self.chunks_path + ".corrupt"
            logger.warning("chunks.json is corrupted; backing it up to %s and regenerating.", backup)
            try:
                os.replace(self.chunks_path, backup)
            except OSError:
                try:
                    os.remove(self.chunks_path)
                except OSError:
                    pass

        # If no chunks (or corrupted), generate from script
        if os.path.exists(self.script_path):
            try:
                with open(self.script_path, "r", encoding="utf-8") as f:
                    script = json.load(f)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"annotated_script.json is also corrupted ({e}). Starting with empty chunks.")
                return []

            review = []
            chunks = group_into_chunks(script, review_sink=review)
            log_review_characters(review)

            # Initialize chunk status
            for i, chunk in enumerate(chunks):
                chunk["id"] = i
                chunk["uid"] = _new_chunk_uid()
                chunk["status"] = "pending"  # pending, generating, done, error
                chunk["audio_path"] = None

            atomic_json_write(chunks, self.chunks_path)
            return chunks

        return []

    def _resolve_alias(self, speaker, voice_config):
        """Resolve speaker aliases by following `alias_of` chain in voice_config.

        Prevent infinite loops by limiting chain length.
        Returns the canonical speaker name (string).
        """
        if not speaker:
            return speaker
        name = speaker
        seen = set()
        for _ in range(16):  # Increased from 8 to handle longer alias chains
            if name in seen:
                logger.warning(f"Alias cycle detected for speaker '{speaker}': chain visited {seen}")
                break
            seen.add(name)
            entry = voice_config.get(name, {})
            alias = entry.get('alias_of') or entry.get('alias')
            if not alias:
                break
            # If alias is empty or same, stop
            if not isinstance(alias, str) or alias.strip() == '' or alias == name:
                break
            # Continue resolution
            name = alias
        else:
            # Loop completed without break - chain exceeded limit
            logger.warning(f"Alias chain for '{speaker}' exceeded 16 iterations; using last resolved name '{name}'")
        return name

    def save_chunks(self, chunks):
        with self._chunks_lock:
            atomic_json_write(chunks, self.chunks_path)

    def _modify_chunk(self, index, mutator):
        """Atomically apply `mutator(chunk)` to a single chunk (thread-safe read-modify-write).

        Unlike load_chunks() + modify + save_chunks(), this holds the lock for the
        entire read-modify-write cycle, preventing concurrent threads from
        overwriting each other's updates.
        """
        with self._chunks_lock:
            chunks = self._read_chunks()
            if not (0 <= index < len(chunks)):
                return None
            chunk = chunks[index]
            mutator(chunk)
            atomic_json_write(chunks, self.chunks_path)
            return chunk

    def insert_chunk(self, after_index):
        """Insert an empty chunk after the given index. Returns the new chunk list."""
        with self._chunks_lock:
            chunks = self._read_chunks()
            if not (0 <= after_index < len(chunks)):
                return None

            # Copy speaker from the row we're splitting from
            source = chunks[after_index]
            new_chunk = {
                "id": after_index + 1,
                "uid": _new_chunk_uid(),
                "speaker": source.get("speaker", "NARRATOR"),
                "text": "",
                "instruct": "",
                "status": "pending",
                "audio_path": None
            }
            chunks.insert(after_index + 1, new_chunk)

            # Re-number all IDs
            for i, chunk in enumerate(chunks):
                chunk["id"] = i

            atomic_json_write(chunks, self.chunks_path)
            return chunks

    def delete_chunk(self, index):
        """Delete a chunk at the given index. Returns (deleted_chunk, updated_chunks) or None."""
        with self._chunks_lock:
            chunks = self._read_chunks()
            if not (0 <= index < len(chunks)):
                return None
            if len(chunks) <= 1:
                return None  # don't allow deleting the last chunk

            deleted = chunks.pop(index)

            # Re-number all IDs
            for i, chunk in enumerate(chunks):
                chunk["id"] = i

            atomic_json_write(chunks, self.chunks_path)
            return deleted, chunks

    def restore_chunk(self, at_index, chunk_data):
        """Re-insert a chunk at a specific index. Returns the updated chunk list."""
        with self._chunks_lock:
            chunks = self._read_chunks()

            at_index = max(0, min(at_index, len(chunks)))
            if isinstance(chunk_data, dict):
                chunk_data.setdefault("uid", _new_chunk_uid())
            chunks.insert(at_index, chunk_data)

            # Re-number all IDs
            for i, chunk in enumerate(chunks):
                chunk["id"] = i

            atomic_json_write(chunks, self.chunks_path)
            return chunks

    def update_chunk(self, index, data):
        """Thread-safe chunk update. See _modify_chunk."""
        def mutator(chunk):
            if "text" in data: chunk["text"] = data["text"]
            if "instruct" in data: chunk["instruct"] = data["instruct"]
            if "speaker" in data: chunk["speaker"] = data["speaker"]

            # pause_after: set or clear (None removes the key)
            if "pause_after" in data:
                if data["pause_after"] is not None:
                    chunk["pause_after"] = max(0, int(data["pause_after"]))
                else:
                    chunk.pop("pause_after", None)

            # If text/instruct/speaker changed, reset status (but keep old audio until regen)
            if "text" in data or "instruct" in data or "speaker" in data:
                chunk["status"] = "pending"

        return self._modify_chunk(index, mutator)

    def _update_chunk_fields(self, index, **kwargs):
        """Atomically set arbitrary fields on a chunk (status, audio_path, error, etc.).

        Unlike update_chunk, this applies the given fields directly with no
        special-casing or status side-effects. See _modify_chunk.
        """
        def mutator(chunk):
            chunk.update(kwargs)

        return self._modify_chunk(index, mutator)

    def generate_chunk_audio(self, index):
        chunks = self.load_chunks()
        if not (0 <= index < len(chunks)):
            return False, "Invalid chunk index"

        chunk = chunks[index]
        self._update_chunk_fields(index, status="generating")

        temp_path = None
        try:
            engine = self.get_engine()
            if not engine:
                self._update_chunk_fields(index, status="error")
                return False, "TTS engine not initialized"

            # atomic_json_write's write-temp+rename makes plain reads safe even
            # while app.py's voice_library endpoints hold file_lock(voice_config_path)
            # for a read-modify-write — no extra locking needed here.
            voice_config = safe_load_json(self.voice_config_path, default={})

            speaker = chunk["speaker"]
            # Resolve aliases to canonical speaker used for TTS
            canonical_speaker = self._resolve_alias(speaker, voice_config)
            if canonical_speaker != speaker:
                print(f"Resolving alias: '{speaker}' -> '{canonical_speaker}'")
            speaker_to_use = canonical_speaker
            text = chunk["text"]
            instruct = chunk.get("instruct", "")

            print(f"Generating chunk {index}: speaker={speaker}, instruct='{instruct}', text='{text[:50]}...'")

            # Generate to temp file (unique per chunk for parallel processing)
            import tempfile
            fd, temp_path = tempfile.mkstemp(prefix=f"chunk_{index}_", suffix=".wav", dir=self.root_dir)
            os.close(fd)  # Close the file descriptor, we'll write via TTS engine

            # Pass canonical speaker to the TTS engine so it uses the aliased config
            success = engine.generate_voice(text, instruct, speaker_to_use, voice_config, temp_path)

            if success:
                # Check file size
                if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                     self._update_chunk_fields(index, status="error")
                     return False, "Generated audio file is missing or empty"

                print(f"Generated WAV size: {os.path.getsize(temp_path)} bytes")

                # Try to convert to mp3, fallback to wav if ffmpeg missing.
                # Name by stable uid (not list position) so a later insert/delete
                # can't make this file collide with another chunk's audio.
                filename_base = f"voiceline_{chunk.get('uid') or f'{index+1:04d}'}_{sanitize_filename(speaker_to_use)}"

                # Shared MP3-export-with-WAV-fallback (raises on 0-duration audio,
                # caught by the outer handler below).
                audio_path = self._export_chunk_audio(temp_path, filename_base)
                self._update_chunk_fields(index, status="done", audio_path=audio_path)

                return True, audio_path
            else:
                self._update_chunk_fields(index, status="error")
                return False, "Generation failed"

        except Exception as e:
            try:
                self._update_chunk_fields(index, status="error")
            except Exception as update_err:
                print(f"Warning: Failed to update chunk {index} status to error: {update_err}")
            return False, str(e)
        finally:
            # Cleanup runs on every exit path - success, early validation returns,
            # and exceptions - so the temp wav never leaks.
            if temp_path:
                self._remove_temp_file(temp_path)

    def _load_pause_defaults(self):
        """Return (pause_between_speakers_ms, pause_same_speaker_ms) from config."""
        tts_cfg = self._load_tts_config()
        return (
            tts_cfg.get("pause_between_speakers_ms", DEFAULT_PAUSE_MS),
            tts_cfg.get("pause_same_speaker_ms", SAME_SPEAKER_PAUSE_MS),
        )

    def _load_chunks_with_audio(self):
        """Load chunks and pair each with its AudioSegment.

        Returns (result, skipped_count): result is the list of (chunk, segment)
        pairs that loaded successfully; skipped_count is how many chunks were
        dropped (missing audio_path, missing file, or a failed audio load) so
        callers can report a partial export instead of a silent blanket success.
        """
        chunks = self.load_chunks()
        result = []
        skipped = 0
        for chunk in chunks:
            path = chunk.get("audio_path")
            if not path:
                skipped += 1
                continue
            full_path = os.path.join(self.root_dir, path)
            if not os.path.exists(full_path):
                skipped += 1
                continue
            try:
                segment = AudioSegment.from_file(full_path)
                result.append((chunk, segment))
            except Exception as e:
                print(f"Error loading audio segment {path}: {e}")
                skipped += 1
        return result, skipped

    def merge_audio(self):
        chunks_with_audio, skipped = self._load_chunks_with_audio()
        if not chunks_with_audio:
            return False, "No audio segments found"

        pause_ms, same_speaker_pause_ms = self._load_pause_defaults()
        timeline = compute_timeline(chunks_with_audio, pause_ms, same_speaker_pause_ms)

        # Build final audio from timeline
        audio_segments = [seg for _, seg, _ in timeline]
        speakers = [chunk["speaker"] for chunk, _, _ in timeline]
        pause_overrides = [chunk.get("pause_after") for chunk, _, _ in timeline]

        final_audio = combine_audio_with_pauses(
            audio_segments, speakers, pause_ms, same_speaker_pause_ms, pause_overrides
        )
        output_filename = "cloned_audiobook.mp3"
        output_path = os.path.join(self.root_dir, output_filename)
        final_audio.export(output_path, format="mp3")

        if skipped:
            return True, f"{output_filename} ({skipped} chunk(s) skipped — missing/corrupt audio)"
        return True, output_filename

    def export_audacity(self):
        """Export project as an Audacity-compatible zip with per-speaker WAV tracks,
        a LOF file for auto-import, and a labels file for chunk annotations."""
        chunks_with_audio, skipped = self._load_chunks_with_audio()
        if not chunks_with_audio:
            return False, "No audio segments found"

        # Phase 1 — Compute timeline
        pause_ms, same_speaker_pause_ms = self._load_pause_defaults()
        timeline = compute_timeline(chunks_with_audio, pause_ms, same_speaker_pause_ms)

        if not timeline:
            return False, "No audio segments found"

        # Total duration = last chunk's start + its length
        _, last_seg, last_start = timeline[-1]
        total_duration_ms = last_start + len(last_seg)

        # Phase 2 — Build per-speaker WAV tracks
        speakers_ordered = []
        seen = set()
        for chunk, _, _ in timeline:
            if chunk["speaker"] not in seen:
                speakers_ordered.append(chunk["speaker"])
                seen.add(chunk["speaker"])

        speaker_tracks = {}
        for speaker in speakers_ordered:
            track_cursor = 0
            track = AudioSegment.empty()

            for chunk, segment, start_ms in timeline:
                if chunk["speaker"] != speaker:
                    continue
                # Insert silence gap from current track position to this chunk's start
                gap = start_ms - track_cursor
                if gap > 0:
                    track += AudioSegment.silent(duration=gap)
                track += segment
                track_cursor = start_ms + len(segment)

            # Pad to total duration so all tracks are equal length
            remaining = total_duration_ms - track_cursor
            if remaining > 0:
                track += AudioSegment.silent(duration=remaining)

            speaker_tracks[speaker] = track

        # Phase 3 — Build LOF and labels content
        lof_lines = []
        for speaker in speakers_ordered:
            safe_name = sanitize_filename(speaker)
            lof_lines.append(f'file "{safe_name}.wav"')
        lof_content = "\n".join(lof_lines) + "\n"

        label_lines = []
        for chunk, segment, start_ms in timeline:
            start_sec = start_ms / 1000.0
            end_sec = (start_ms + len(segment)) / 1000.0
            text_preview = chunk.get("text", "")[:80]
            label = f"[{chunk['speaker']}] {text_preview}"
            # Audacity labels are tab-separated, one per line - a tab or newline
            # inside the text would add a phantom column or split the row.
            label = label.replace("\t", " ").replace("\r", " ").replace("\n", " ")
            label_lines.append(f"{start_sec:.6f}\t{end_sec:.6f}\t{label}")
        labels_content = "\n".join(label_lines) + "\n"

        # Phase 4 — Zip everything to a temp path, then atomically replace, so a
        # failure mid-write can't leave a truncated zip the download route serves
        # as valid (or clobber a previous good export with garbage).
        zip_path = os.path.join(self.root_dir, "audacity_export.zip")
        tmp_zip = zip_path + ".tmp"
        try:
            with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("project.lof", lof_content)
                zf.writestr("labels.txt", labels_content)

                for speaker in speakers_ordered:
                    safe_name = sanitize_filename(speaker)
                    wav_buffer = io.BytesIO()
                    speaker_tracks[speaker].export(wav_buffer, format="wav")
                    zf.writestr(f"{safe_name}.wav", wav_buffer.getvalue())
            os.replace(tmp_zip, zip_path)
        except BaseException:
            try:
                if os.path.exists(tmp_zip):
                    os.remove(tmp_zip)
            except OSError:
                pass
            raise

        if skipped:
            return True, f"{zip_path} ({skipped} chunk(s) skipped — missing/corrupt audio)"
        return True, zip_path

    def merge_m4b(self, per_chunk_chapters=False, metadata=None):
        """Merge audio chunks into an M4B audiobook with chapter markers.

        Args:
            per_chunk_chapters: If True, each chunk is a chapter. If False,
                detect chapter headings and group chunks into sections.
            metadata: Optional dict with keys: title, author, narrator, year,
                description, cover_path (absolute path to cover image).

        Returns:
            tuple: (success: bool, message: str)
        """
        metadata = metadata or {}
        chunks_with_audio, skipped = self._load_chunks_with_audio()
        if not chunks_with_audio:
            return False, "No audio segments found"

        # Phase 1 — Compute timeline
        pause_ms, same_speaker_pause_ms = self._load_pause_defaults()
        timeline = compute_timeline(chunks_with_audio, pause_ms, same_speaker_pause_ms)

        if not timeline:
            return False, "No audio segments found"

        # Phase 2 — Build chapters
        chapters = self._build_m4b_chapters(timeline, per_chunk_chapters)
        print(f"  M4B: {len(chapters)} chapters")

        # Phase 3 — Combine audio and export to temp WAV
        audio_segments = [seg for _, seg, _ in timeline]
        speakers = [chunk["speaker"] for chunk, _, _ in timeline]
        pause_overrides = [chunk.get("pause_after") for chunk, _, _ in timeline]
        final_audio = combine_audio_with_pauses(
            audio_segments, speakers, pause_ms, same_speaker_pause_ms, pause_overrides
        )

        temp_wav = os.path.join(self.root_dir, "temp_m4b_combined.wav")
        meta_path = os.path.join(self.root_dir, "temp_m4b_meta.txt")
        output_path = os.path.join(self.root_dir, "audiobook.m4b")

        encode_ok = False
        try:
            final_audio.export(temp_wav, format="wav")

            # Phase 4 — Write FFmpeg metadata file with book metadata
            meta_lines = [";FFMETADATA1"]
            meta_lines.append(f"title={self._escape_ffmeta(metadata.get('title') or 'Audiobook')}")
            meta_lines.append(f"artist={self._escape_ffmeta(metadata.get('author') or '')}")
            meta_lines.append(f"album_artist={self._escape_ffmeta(metadata.get('narrator') or '')}")
            meta_lines.append(f"date={self._escape_ffmeta(metadata.get('year') or '')}")
            meta_lines.append(f"comment={self._escape_ffmeta(metadata.get('description') or '')}")
            meta_lines.append("genre=Audiobook")
            meta_lines.append("")
            for title, start_ms, end_ms in chapters:
                safe_title = self._escape_ffmeta(title)
                meta_lines.append("[CHAPTER]")
                meta_lines.append("TIMEBASE=1/1000")
                meta_lines.append(f"START={start_ms}")
                meta_lines.append(f"END={end_ms}")
                meta_lines.append(f"title={safe_title}")
                meta_lines.append("")

            with open(meta_path, "w", encoding="utf-8") as f:
                f.write("\n".join(meta_lines))

            # Phase 5 — FFmpeg: WAV + chapters → M4B (AAC)
            cover_path = metadata.get("cover_path") or ""
            has_cover = cover_path and os.path.exists(cover_path)

            cmd = ["ffmpeg", "-y", "-i", temp_wav]
            if has_cover:
                cmd += ["-i", cover_path]
            cmd += ["-i", meta_path, "-map_metadata", "2" if has_cover else "1"]
            # Map audio stream
            cmd += ["-map", "0:a"]
            if has_cover:
                # Map cover as attached picture
                cmd += ["-map", "1:v", "-c:v", "copy", "-disposition:v:0", "attached_pic"]
            cmd += [
                "-c:a", "aac",
                "-b:a", "128k",
                "-movflags", "+faststart",
                output_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                print(f"FFmpeg stderr: {result.stderr[-500:]}")
                return False, f"FFmpeg failed (exit {result.returncode})"
            encode_ok = True

        finally:
            # On a timed-out/failed encode also delete the partial output .m4b —
            # ffmpeg writes incrementally and the download route serves the file
            # purely on existence, so a partial file would be handed out as valid.
            cleanup = [temp_wav, meta_path]
            if not encode_ok:
                cleanup.append(output_path)
            for tmp in cleanup:
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass

        if skipped:
            return True, f"audiobook.m4b ({skipped} chunk(s) skipped — missing/corrupt audio)"
        return True, "audiobook.m4b"

    @staticmethod
    def _escape_ffmeta(text):
        """Escape special characters for FFmpeg metadata format."""
        text = text.replace("\\", "\\\\")
        text = text.replace("=", "\\=")
        text = text.replace(";", "\\;")
        text = text.replace("#", "\\#")
        text = text.replace("\n", " ")
        text = text.replace("\r", " ")  # CRLF (e.g. pasted description) would else corrupt the key=value line
        return text

    # Regex for detecting chapter/section headings in chunk text
    _HEADING_RE = re.compile(
        r'^(chapter|part|book|volume|prologue|epilogue|introduction|conclusion|act|section)\b',
        re.IGNORECASE
    )

    def _build_m4b_chapters(self, timeline, per_chunk_chapters):
        """Build chapter list from timeline entries.

        Returns:
            list of (title, start_ms, end_ms) tuples
        """
        if per_chunk_chapters:
            chapters = []
            for chunk, segment, start_ms in timeline:
                end_ms = start_ms + len(segment)
                text_preview = chunk.get("text", "")[:80]
                title = f"[{chunk['speaker']}] {text_preview}"
                chapters.append((title, start_ms, end_ms))
            return chapters

        # Smart grouping: detect chapter headings
        heading_indices = []
        for i, (chunk, _, _) in enumerate(timeline):
            text = chunk.get("text", "").strip()
            # Starts with a heading keyword, or short structural text in its
            # own right (likely a stylized chapter title with no keyword,
            # e.g. "The Awakening") - the second check used to also require
            # self._HEADING_RE.search(text), which is redundant with (and
            # since _HEADING_RE is ^-anchored, can never succeed when) the
            # first check already failed, making this branch unreachable.
            if self._HEADING_RE.match(text):
                heading_indices.append(i)
            elif len(text) < 80 and '"' not in text and text:
                heading_indices.append(i)

        # If no headings detected, fall back to per-chunk
        if not heading_indices:
            print("  M4B: No chapter headings detected, falling back to per-chunk chapters")
            return self._build_m4b_chapters(timeline, per_chunk_chapters=True)

        chapters = []

        # Pre-heading chunks → "Introduction"
        if heading_indices[0] > 0:
            start_ms = timeline[0][2]
            last_before = heading_indices[0] - 1
            end_ms = timeline[last_before][2] + len(timeline[last_before][1])
            chapters.append(("Introduction", start_ms, end_ms))

        # Each heading starts a chapter that runs until the next heading
        for idx, head_i in enumerate(heading_indices):
            title = timeline[head_i][0].get("text", "").strip()
            # Truncate long titles
            if len(title) > 120:
                title = title[:117] + "..."

            start_ms = timeline[head_i][2]

            # End = start of next heading, or end of last chunk
            if idx + 1 < len(heading_indices):
                next_head_i = heading_indices[idx + 1]
                last_in_group = next_head_i - 1
            else:
                last_in_group = len(timeline) - 1

            end_ms = timeline[last_in_group][2] + len(timeline[last_in_group][1])
            chapters.append((title, start_ms, end_ms))

        return chapters

    def generate_chunks_parallel(self, indices, max_workers=2, progress_callback=None,
                                  cancel_check=None):
        """Generate multiple chunks in parallel using ThreadPoolExecutor.

        Uses individual TTS API calls with per-speaker voice settings.

        Args:
            indices: List of chunk indices to generate
            max_workers: Number of concurrent TTS workers
            progress_callback: Optional callback(completed, failed, total) for progress updates
            cancel_check: Optional callable returning True when cancellation is requested

        Returns:
            dict with 'completed', 'failed', and 'cancelled' keys
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import gc

        results = {"completed": [], "failed": [], "cancelled": 0}

        # Filter out empty-text chunks
        chunks = self.load_chunks()
        if chunks:
            indices = [i for i in indices if 0 <= i < len(chunks) and chunks[i].get("text", "").strip()]

        total = len(indices)

        if total == 0:
            return results

        all_indices = list(indices)

        def _run_round(round_indices, workers):
            """One pass at `workers` concurrency.

            Returns (completed, oom_failed, hard_failed, cancelled). OOM failures
            are kept separate so the caller can step concurrency down and retry
            just those, while hard failures (bad text, missing engine) are final.
            """
            completed, oom_failed, hard_failed = [], [], []
            was_cancelled = False
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(self.generate_chunk_audio, idx): idx for idx in round_indices}
                for future in as_completed(futures):
                    if cancel_check and cancel_check() and not was_cancelled:
                        was_cancelled = True
                        print("[CANCEL] Cancellation requested — stopping parallel generation")
                        for pending_future in futures:
                            if pending_future is not future:
                                pending_future.cancel()
                    idx = futures[future]
                    if future.cancelled():
                        continue
                    try:
                        success, msg = future.result()
                        if success:
                            completed.append(idx)
                            print(f"Chunk {idx} completed: {msg}")
                        elif is_oom_failure(msg):
                            oom_failed.append((idx, msg))
                            print(f"Chunk {idx} failed (VRAM): {msg}")
                        else:
                            hard_failed.append((idx, msg))
                            print(f"Chunk {idx} failed: {msg}")
                    except Exception as e:
                        (oom_failed if is_oom_failure(e) else hard_failed).append((idx, str(e)))
                        print(f"Chunk {idx} error: {e}")
                    if progress_callback:
                        # oom_failed chunks aren't final yet (they get retried
                        # next round at a lower worker count), but they ARE
                        # accounted for right now - count them alongside
                        # completed/hard_failed so the running total this
                        # round doesn't look like it's stalled or losing track
                        # of in-flight work.
                        progress_callback(len(results["completed"]) + len(completed),
                                          len(results["failed"]) + len(hard_failed) + len(oom_failed), total)
            return completed, oom_failed, hard_failed, was_cancelled

        # Start at the configured worker count and step down one at a time only
        # when a round hits VRAM OOM, retrying just the OOM-failed chunks, until
        # they succeed or we're down to a single worker.
        workers = max(1, max_workers)
        pending = list(all_indices)
        cancelled = False
        print(f"Starting parallel generation of {total} chunks with {workers} workers...")
        while pending:
            completed, oom_failed, hard_failed, was_cancelled = _run_round(pending, workers)
            results["completed"].extend(completed)
            results["failed"].extend(hard_failed)
            if was_cancelled:
                cancelled = True
                break
            if oom_failed and workers > 1:
                workers -= 1
                pending = [idx for idx, _ in oom_failed]
                print(f"[VRAM] Out-of-memory on {len(pending)} chunk(s) — stepping TTS "
                      f"workers down to {workers} and retrying.")
                gc.collect()
                continue
            # workers == 1 (or no OOM left): remaining OOM failures are now final.
            results["failed"].extend(oom_failed)
            break

        # Reset remaining "generating" chunks to "pending" on cancel.
        if cancelled:
            done_indices = set(results["completed"]) | {idx for idx, _ in results["failed"]}
            chunks = self.load_chunks()
            if chunks:
                for idx in all_indices:
                    if idx not in done_indices and 0 <= idx < len(chunks) and chunks[idx].get("status") == "generating":
                        chunks[idx]["status"] = "pending"
                        results["cancelled"] += 1
                self.save_chunks(chunks)

        print(f"Parallel generation complete: {len(results['completed'])} succeeded, "
              f"{len(results['failed'])} failed, {results['cancelled']} cancelled")
        return results

    def _group_indices_by_voice_type(self, indices, chunks, voice_config):
        """Reorder indices so chunks with the same voice type are contiguous.

        Grouping key matches how tts.py routes batches:
        - "custom" for custom voices (all batched together)
        - "clone:{speaker}" for clone voices (batched per speaker)
        - "lora:{adapter}" for LoRA voices (batched per adapter)
        - "design" for voice design (always sequential)

        Within each group, original order is preserved.
        """
        from collections import OrderedDict
        groups = OrderedDict()

        for idx in indices:
            if not (0 <= idx < len(chunks)):
                groups.setdefault("custom", []).append(idx)
                continue
            speaker = chunks[idx].get("speaker", "")
            # Resolve alias before grouping so alias groups collate with their canonical speaker
            canonical = self._resolve_alias(speaker, voice_config)
            voice_data = voice_config.get(canonical, {})
            category = voice_category(voice_data)

            if category == "clone":
                key = f"clone:{canonical}"
            elif category == "lora":
                adapter_id = voice_data.get("adapter_id", "")
                key = f"lora:{adapter_id}"
            elif category == "design":
                key = "design"
            else:
                key = "custom"

            groups.setdefault(key, []).append(idx)

        reordered = []
        for key, group_indices in groups.items():
            print(f"  Voice group '{key}': {len(group_indices)} chunks")
            reordered.extend(group_indices)

        return reordered

    def _export_chunk_audio(self, temp_path, filename_base):
        """Convert a temp WAV to MP3, falling back to a copied WAV when ffmpeg
        lacks an MP3 encoder. Returns the relative audio_path (under voicelines/).
        Raises ValueError if the source audio has zero duration.
        """
        segment = AudioSegment.from_file(temp_path)
        if len(segment) == 0:
            raise ValueError("Audio has 0 duration")

        try:
            mp3_filename = f"{filename_base}.mp3"
            mp3_filepath = os.path.join(self.voicelines_dir, mp3_filename)
            segment.export(mp3_filepath, format="mp3")

            # Validate: conda ffmpeg often lacks libmp3lame, producing a tiny
            # (~428 byte) header-only file without raising an error.
            mp3_size = os.path.getsize(mp3_filepath) if os.path.exists(mp3_filepath) else 0
            if mp3_size < 1024:
                print(f"MP3 export produced invalid file ({mp3_size} bytes) — ffmpeg likely "
                      f"lacks MP3 encoder (libmp3lame). Falling back to WAV.")
                os.remove(mp3_filepath)
                raise RuntimeError("MP3 export produced invalid file")
            return f"voicelines/{mp3_filename}"

        except Exception as e:
            if "invalid file" not in str(e).lower():
                print(f"MP3 conversion failed: {e}")
            wav_filename = f"{filename_base}.wav"
            wav_filepath = os.path.join(self.voicelines_dir, wav_filename)
            shutil.copy(temp_path, wav_filepath)
            return f"voicelines/{wav_filename}"

    def _remove_temp_file(self, temp_path):
        """Best-effort delete of a temp batch WAV, retrying briefly on transient locks."""
        if not os.path.exists(temp_path):
            return
        for attempt in range(3):
            try:
                os.remove(temp_path)
                return
            except OSError:
                if attempt < 2:
                    time.sleep(0.1 * (attempt + 1))
                else:
                    print(f"Warning: Could not delete temp file {temp_path}")

    def _finalize_completed_chunk(self, idx, chunks):
        """Convert one completed chunk's temp audio to its final file and
        update its status in `chunks` (in place - chunks is this whole
        batch's load-mutate-save accumulator, saved once by the caller).

        Returns ("completed", idx, audio_path) or ("failed", idx, error_msg)
        for the caller to fold into its own results dict.
        """
        if not (0 <= idx < len(chunks)):
            print(f"Chunk {idx} skipped: index out of range (chunks changed during generation?)")
            return "failed", idx, "Index out of range after reload"

        temp_path = os.path.join(self.root_dir, f"temp_batch_{idx}.wav")
        if not os.path.exists(temp_path):
            chunks[idx]["status"] = "error"
            return "failed", idx, "Temp audio file not found"

        try:
            chunk = chunks[idx]
            speaker = chunk.get("speaker", "unknown")
            # Stable uid (not list position) so the file can't collide with a
            # neighbour's after an insert/delete shifts indices.
            filename_base = f"voiceline_{chunk.get('uid') or f'{idx+1:04d}'}_{sanitize_filename(speaker)}"
            chunks[idx]["audio_path"] = self._export_chunk_audio(temp_path, filename_base)
            chunks[idx]["status"] = "done"
            print(f"Chunk {idx} completed: {chunks[idx]['audio_path']}")
            self._remove_temp_file(temp_path)
            return "completed", idx, chunks[idx]["audio_path"]
        except Exception as e:
            print(f"Error processing chunk {idx}: {e}")
            chunks[idx]["status"] = "error"
            return "failed", idx, str(e)

    def _record_batch_failures(self, batch_failed, chunks, current_batch_size):
        """Split a batch's failures into retryable OOM indices and hard
        failures (status=error, set in `chunks` in place - see
        _finalize_completed_chunk's docstring).

        Returns (oom_failed_indices, hard_failure_tuples) for the caller to
        retry/fold into its own results dict.
        """
        oom_failed = []
        hard_failures = []
        for idx, error in batch_failed:
            if is_oom_failure(error) and current_batch_size > 1:
                # Retryable at a smaller size - don't mark as error yet.
                oom_failed.append(idx)
                continue
            if 0 <= idx < len(chunks):
                chunks[idx]["status"] = "error"
            hard_failures.append((idx, error))
        return oom_failed, hard_failures

    def generate_chunks_batch(self, indices, batch_seed=-1, batch_size=4, progress_callback=None,
                               batch_group_by_type=False, cancel_check=None):
        """Generate multiple chunks using batch TTS API with a single seed.

        Args:
            indices: List of chunk indices to generate
            batch_seed: Single seed for all generations (-1 for random)
            batch_size: Number of chunks per batch request
            progress_callback: Optional callback(completed, failed, total) for progress updates
            batch_group_by_type: Group indices by voice type before batching for
                GPU efficiency. When False, indices are batched in sequential order.
            cancel_check: Optional callable returning True when cancellation is requested

        Returns:
            dict with 'completed', 'failed', and 'cancelled' keys
        """
        results = {"completed": [], "failed": [], "cancelled": 0}

        # Load chunks and voice config
        chunks = self.load_chunks()

        # Filter out empty-text chunks
        if chunks:
            indices = [i for i in indices if 0 <= i < len(chunks) and chunks[i].get("text", "").strip()]

        total = len(indices)

        if total == 0:
            return results

        print(f"Starting batch generation of {total} chunks (batch_size={batch_size}, seed={batch_seed}, "
              f"group_by_type={batch_group_by_type})...")
        
        # atomic_json_write's write-temp+rename makes plain reads safe even
        # while app.py's voice_library endpoints hold file_lock(voice_config_path)
        # for a read-modify-write — no extra locking needed here.
        voice_config = safe_load_json(self.voice_config_path, default={})

        # Get TTS engine
        engine = self.get_engine()
        if not engine:
            for idx in indices:
                results["failed"].append((idx, "TTS engine not initialized"))
            return results

        # Mark all chunks as generating
        for idx in indices:
            if 0 <= idx < len(chunks):
                chunks[idx]["status"] = "generating"
        self.save_chunks(chunks)

        # Optionally reorder indices so same voice-type chunks are contiguous.
        # This produces larger homogeneous batches (e.g. all custom voices
        # together) instead of fragmenting each batch across voice types.
        if batch_group_by_type:
            indices = self._group_indices_by_voice_type(indices, chunks, voice_config)

        # Process indices in batches, starting at the configured size and
        # stepping the size down only when a batch hits VRAM OOM (retrying just
        # the OOM-failed chunks at the smaller size), down to 1.
        pending = list(indices)
        current_batch_size = max(1, batch_size)
        batch_num = 0
        while pending:
            if cancel_check and cancel_check():
                print(f"[CANCEL] Cancellation requested before batch {batch_num + 1}")
                break

            batch_indices = pending[:current_batch_size]
            pending = pending[current_batch_size:]
            batch_num += 1
            print(f"Batch {batch_num} ({len(batch_indices)} chunks, size={current_batch_size}, {len(pending)} queued)")

            # Build batch request data
            batch_chunks = []
            for idx in batch_indices:
                if 0 <= idx < len(chunks):
                    chunk = chunks[idx]
                    # Resolve aliases so batch uses canonical speaker config
                    speaker = chunk.get("speaker", "")
                    canonical = self._resolve_alias(speaker, voice_config)
                    batch_chunks.append({
                        "index": idx,
                        "text": chunk.get("text", ""),
                        "instruct": chunk.get("instruct", ""),
                        "speaker": canonical
                    })

            # Call batch TTS with single seed
            batch_results = engine.generate_batch(batch_chunks, voice_config, self.root_dir, batch_seed)

            # Process completed chunks - convert to MP3 and update status
            chunks = self.load_chunks()  # Reload for each batch

            for idx in batch_results["completed"]:
                outcome, out_idx, payload = self._finalize_completed_chunk(idx, chunks)
                if outcome == "completed":
                    results["completed"].append(out_idx)
                else:
                    results["failed"].append((out_idx, payload))

            oom_failed, hard_failures = self._record_batch_failures(
                batch_results["failed"], chunks, current_batch_size)
            results["failed"].extend(hard_failures)

            self.save_chunks(chunks)

            if oom_failed:
                gc.collect()
                current_batch_size = max(1, current_batch_size - 1)
                pending = oom_failed + pending  # retry the OOM chunks first
                print(f"[VRAM] Out-of-memory on {len(oom_failed)} chunk(s) — stepping TTS "
                      f"batch size down to {current_batch_size} and retrying.")

            if progress_callback:
                progress_callback(len(results["completed"]), len(results["failed"]), total)

        # Reset remaining "generating" chunks to "pending" on cancel or completion
        done_indices = set(results["completed"]) | {idx for idx, _ in results["failed"]}
        chunks = self.load_chunks()
        if chunks:
            for idx in indices:
                if idx not in done_indices and 0 <= idx < len(chunks) and chunks[idx].get("status") == "generating":
                    chunks[idx]["status"] = "pending"
                    results["cancelled"] += 1
            if results["cancelled"]:
                self.save_chunks(chunks)

        print(f"Batch generation complete: {len(results['completed'])} succeeded, "
              f"{len(results['failed'])} failed, {results['cancelled']} cancelled")
        return results
