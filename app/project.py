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
from utils import atomic_json_write
from tts import (
    TTSEngine,
    combine_audio_with_pauses,
    compute_timeline,
    sanitize_filename,
    DEFAULT_PAUSE_MS,
    SAME_SPEAKER_PAUSE_MS
)
from pydub import AudioSegment

MAX_CHUNK_CHARS = 500

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


def group_into_chunks(script_entries, max_chars=MAX_CHUNK_CHARS):
    """Group consecutive entries by same speaker into chunks up to max_chars"""
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

        # Don't merge structural text (titles, chapter headings, dedications)
        if (speaker == current_speaker and instruct == current_instruct
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
        self.config_path = os.path.join(root_dir, "app", "config.json")

        # Ensure voicelines dir exists
        os.makedirs(self.voicelines_dir, exist_ok=True)

        self.engine = None
        self._chunks_lock = threading.Lock()  # Thread-safe file writes

    def get_engine(self):
        if self.engine:
            return self.engine

        # Load config
        config = {}
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
            except Exception:
                pass

        try:
            self.engine = TTSEngine(config)
            print(f"TTS engine initialized (mode={self.engine.mode})")
            return self.engine
        except Exception as e:
            print(f"Failed to initialize TTS engine: {e}")
            return None

    def _load_tts_config(self):
        """Load TTS config section from config.json for pause defaults."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f).get("tts", {})
        except Exception:
            return {}

    def load_chunks(self):
        if os.path.exists(self.chunks_path):
            try:
                with open(self.chunks_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"WARNING: chunks.json is corrupted ({e}). Regenerating from script...")
                os.remove(self.chunks_path)

        # If no chunks (or corrupted), generate from script
        if os.path.exists(self.script_path):
            try:
                with open(self.script_path, "r", encoding="utf-8") as f:
                    script = json.load(f)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"WARNING: annotated_script.json is also corrupted ({e}). Starting with empty chunks.")
                return []

            chunks = group_into_chunks(script)

            # Initialize chunk status
            for i, chunk in enumerate(chunks):
                chunk["id"] = i
                chunk["status"] = "pending" # pending, generating, done, error
                chunk["audio_path"] = None

            self.save_chunks(chunks)
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
        for _ in range(8):
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
        return name

    def save_chunks(self, chunks):
        with self._chunks_lock:
            atomic_json_write(chunks, self.chunks_path)

    def _update_chunk_fields(self, index, **fields):
        """Atomically update fields on a single chunk (thread-safe read-modify-write).

        Unlike load_chunks() + modify + save_chunks(), this holds the lock for the
        entire read-modify-write cycle, preventing concurrent threads from
        overwriting each other's updates.
        """
        with self._chunks_lock:
            if not os.path.exists(self.chunks_path):
                return None
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            if not (0 <= index < len(chunks)):
                return None
            chunks[index].update(fields)
            atomic_json_write(chunks, self.chunks_path)
            return chunks[index]

    def insert_chunk(self, after_index):
        """Insert an empty chunk after the given index. Returns the new chunk list."""
        with self._chunks_lock:
            if not os.path.exists(self.chunks_path):
                return None
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            if not (0 <= after_index < len(chunks)):
                return None

            # Copy speaker from the row we're splitting from
            source = chunks[after_index]
            new_chunk = {
                "id": after_index + 1,
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
            if not os.path.exists(self.chunks_path):
                return None
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
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
            if not os.path.exists(self.chunks_path):
                return None
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            at_index = max(0, min(at_index, len(chunks)))
            chunks.insert(at_index, chunk_data)

            # Re-number all IDs
            for i, chunk in enumerate(chunks):
                chunk["id"] = i

            atomic_json_write(chunks, self.chunks_path)
            return chunks

    def update_chunk(self, index, data):
        chunks = self.load_chunks()
        if 0 <= index < len(chunks):
            chunk = chunks[index]
            # Update fields
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

            self.save_chunks(chunks)
            return chunk
        return None

    def generate_chunk_audio(self, index):
        chunks = self.load_chunks()
        if not (0 <= index < len(chunks)):
            return False, "Invalid chunk index"

        chunk = chunks[index]
        self._update_chunk_fields(index, status="generating")

        try:
            engine = self.get_engine()
            if not engine:
                self._update_chunk_fields(index, status="error")
                return False, "TTS engine not initialized"

            # Load voice config
            voice_config = {}
            if os.path.exists(self.voice_config_path):
                with open(self.voice_config_path, "r", encoding="utf-8") as f:
                    voice_config = json.load(f)

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
            temp_path = os.path.join(self.root_dir, f"temp_chunk_{index}.wav")

            # Pass canonical speaker to the TTS engine so it uses the aliased config
            success = engine.generate_voice(text, instruct, speaker_to_use, voice_config, temp_path)

            if success:
                # Check file size
                if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                     self._update_chunk_fields(index, status="error")
                     return False, "Generated audio file is missing or empty"

                print(f"Generated WAV size: {os.path.getsize(temp_path)} bytes")

                # Try to convert to mp3, fallback to wav if ffmpeg missing
                filename_base = f"voiceline_{index+1:04d}_{sanitize_filename(speaker_to_use)}"
                audio_path = None

                try:
                    segment = AudioSegment.from_wav(temp_path)

                    if len(segment) == 0:
                         self._update_chunk_fields(index, status="error")
                         return False, "Generated audio has 0 duration"

                    mp3_filename = f"{filename_base}.mp3"
                    mp3_filepath = os.path.join(self.voicelines_dir, mp3_filename)

                    # This might fail if ffmpeg is missing or lacks MP3 encoder
                    segment.export(mp3_filepath, format="mp3")

                    # Validate: conda ffmpeg often lacks libmp3lame, producing
                    # a tiny (~428 byte) header-only file without raising an error
                    mp3_size = os.path.getsize(mp3_filepath) if os.path.exists(mp3_filepath) else 0
                    if mp3_size < 1024:
                        print(f"MP3 export produced invalid file ({mp3_size} bytes) — ffmpeg likely lacks MP3 encoder (libmp3lame). Falling back to WAV.")
                        os.remove(mp3_filepath)
                        raise RuntimeError("MP3 export produced invalid file")

                    audio_path = f"voicelines/{mp3_filename}"

                except Exception as e:
                    if "invalid file" not in str(e).lower():
                        print(f"MP3 conversion failed (ffmpeg missing?): {e}")
                    # Fallback: copy WAV
                    wav_filename = f"{filename_base}.wav"
                    wav_filepath = os.path.join(self.voicelines_dir, wav_filename)
                    shutil.copy(temp_path, wav_filepath)

                    audio_path = f"voicelines/{wav_filename}"

                self._update_chunk_fields(index, status="done", audio_path=audio_path)

                # Cleanup with retry (may be locked by pydub/ffmpeg on Windows)
                if os.path.exists(temp_path):
                    for attempt in range(3):
                        try:
                            os.remove(temp_path)
                            break
                        except OSError:
                            if attempt < 2:
                                time.sleep(0.1 * (attempt + 1))
                            else:
                                print(f"Warning: Could not delete temp file {temp_path}")

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

    def _load_pause_defaults(self):
        """Return (pause_between_speakers_ms, pause_same_speaker_ms) from config."""
        tts_cfg = self._load_tts_config()
        return (
            tts_cfg.get("pause_between_speakers_ms", DEFAULT_PAUSE_MS),
            tts_cfg.get("pause_same_speaker_ms", SAME_SPEAKER_PAUSE_MS),
        )

    def _load_chunks_with_audio(self):
        """Load chunks and pair each with its AudioSegment. Returns list of (chunk, segment)."""
        chunks = self.load_chunks()
        result = []
        for chunk in chunks:
            path = chunk.get("audio_path")
            if not path:
                continue
            full_path = os.path.join(self.root_dir, path)
            if not os.path.exists(full_path):
                continue
            try:
                segment = AudioSegment.from_file(full_path)
                result.append((chunk, segment))
            except Exception as e:
                print(f"Error loading audio segment {path}: {e}")
        return result

    def merge_audio(self):
        chunks_with_audio = self._load_chunks_with_audio()
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

        return True, output_filename

    def export_audacity(self):
        """Export project as an Audacity-compatible zip with per-speaker WAV tracks,
        a LOF file for auto-import, and a labels file for chunk annotations."""
        chunks_with_audio = self._load_chunks_with_audio()
        if not chunks_with_audio:
            return False, "No audio segments found"

        # Phase 1 — Compute timeline
        pause_ms, same_speaker_pause_ms = self._load_pause_defaults()
        timeline = compute_timeline(chunks_with_audio, pause_ms, same_speaker_pause_ms)

        if not timeline:
            return False, "No audio segments found"

        # Total duration = last chunk's start + its length
        last_chunk, last_seg, last_start = timeline[-1]
        total_duration_ms = last_start + len(last_seg)

        # Phase 2 — Build per-speaker WAV tracks
        speakers_ordered = []
        seen = set()
        for chunk, segment, start_ms in timeline:
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
            label_lines.append(f"{start_sec:.6f}\t{end_sec:.6f}\t{label}")
        labels_content = "\n".join(label_lines) + "\n"

        # Phase 4 — Zip everything
        zip_path = os.path.join(self.root_dir, "audacity_export.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("project.lof", lof_content)
            zf.writestr("labels.txt", labels_content)

            for speaker in speakers_ordered:
                safe_name = sanitize_filename(speaker)
                wav_buffer = io.BytesIO()
                speaker_tracks[speaker].export(wav_buffer, format="wav")
                zf.writestr(f"{safe_name}.wav", wav_buffer.getvalue())

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
        chunks_with_audio = self._load_chunks_with_audio()
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

        finally:
            for tmp in [temp_wav, meta_path]:
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass

        return True, "audiobook.m4b"

    @staticmethod
    def _escape_ffmeta(text):
        """Escape special characters for FFmpeg metadata format."""
        text = text.replace("\\", "\\\\")
        text = text.replace("=", "\\=")
        text = text.replace(";", "\\;")
        text = text.replace("#", "\\#")
        text = text.replace("\n", " ")
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
        for i, (chunk, segment, start_ms) in enumerate(timeline):
            text = chunk.get("text", "").strip()
            # Short structural text (likely a heading) or starts with heading keyword
            if self._HEADING_RE.match(text):
                heading_indices.append(i)
            elif len(text) < 80 and '"' not in text and text and self._HEADING_RE.search(text):
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

        results = {"completed": [], "failed": [], "cancelled": 0}

        # Filter out empty-text chunks
        chunks = self.load_chunks()
        if chunks:
            indices = [i for i in indices if 0 <= i < len(chunks) and chunks[i].get("text", "").strip()]

        total = len(indices)

        if total == 0:
            return results

        print(f"Starting parallel generation of {total} chunks with {max_workers} workers...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.generate_chunk_audio, idx): idx
                for idx in indices
            }

            cancelled = False
            for future in as_completed(futures):
                if cancel_check and cancel_check():
                    cancelled = True
                    print("[CANCEL] Cancellation requested — stopping parallel generation")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

                idx = futures[future]
                try:
                    success, msg = future.result()
                    if success:
                        results["completed"].append(idx)
                        print(f"Chunk {idx} completed: {msg}")
                    else:
                        results["failed"].append((idx, msg))
                        print(f"Chunk {idx} failed: {msg}")
                except Exception as e:
                    results["failed"].append((idx, str(e)))
                    print(f"Chunk {idx} error: {e}")

                if progress_callback:
                    progress_callback(len(results["completed"]), len(results["failed"]), total)

            # Reset remaining "generating" chunks to "pending"
            if cancelled:
                done_indices = set(results["completed"]) | {idx for idx, _ in results["failed"]}
                chunks = self.load_chunks()
                if chunks:
                    for idx in indices:
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
            voice_type = voice_data.get("type", "custom")

            if voice_type == "clone":
                key = f"clone:{canonical}"
            elif voice_type in ("lora", "builtin_lora"):
                adapter_id = voice_data.get("adapter_id", "")
                key = f"lora:{adapter_id}"
            elif voice_type == "design":
                key = "design"
            else:
                key = "custom"

            groups.setdefault(key, []).append(idx)

        reordered = []
        for key, group_indices in groups.items():
            print(f"  Voice group '{key}': {len(group_indices)} chunks")
            reordered.extend(group_indices)

        return reordered

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
        voice_config = {}
        if os.path.exists(self.voice_config_path):
            with open(self.voice_config_path, "r", encoding="utf-8") as f:
                voice_config = json.load(f)

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

        # Split indices into batches
        batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
        print(f"Processing {len(batches)} batches...")

        cancelled = False
        for batch_num, batch_indices in enumerate(batches):
            if cancel_check and cancel_check():
                cancelled = True
                print(f"[CANCEL] Cancellation requested before batch {batch_num + 1}")
                break

            print(f"Batch {batch_num + 1}/{len(batches)}: {len(batch_indices)} chunks")

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
                if not (0 <= idx < len(chunks)):
                    print(f"Chunk {idx} skipped: index out of range (chunks changed during generation?)")
                    results["failed"].append((idx, "Index out of range after reload"))
                    continue

                temp_path = os.path.join(self.root_dir, f"temp_batch_{idx}.wav")

                if not os.path.exists(temp_path):
                    results["failed"].append((idx, "Temp audio file not found"))
                    chunks[idx]["status"] = "error"
                    continue

                try:
                    chunk = chunks[idx]
                    speaker = chunk.get("speaker", "unknown")
                    filename_base = f"voiceline_{idx+1:04d}_{sanitize_filename(speaker)}"

                    try:
                        segment = AudioSegment.from_file(temp_path)
                        if len(segment) == 0:
                            results["failed"].append((idx, "Audio has 0 duration"))
                            chunks[idx]["status"] = "error"
                            continue

                        mp3_filename = f"{filename_base}.mp3"
                        mp3_filepath = os.path.join(self.voicelines_dir, mp3_filename)
                        segment.export(mp3_filepath, format="mp3")

                        # Validate: conda ffmpeg often lacks libmp3lame, producing
                        # a tiny (~428 byte) header-only file without raising an error
                        mp3_size = os.path.getsize(mp3_filepath) if os.path.exists(mp3_filepath) else 0
                        if mp3_size < 1024:
                            print(f"MP3 export produced invalid file ({mp3_size} bytes) for chunk {idx} — ffmpeg likely lacks MP3 encoder (libmp3lame). Falling back to WAV.")
                            os.remove(mp3_filepath)
                            raise RuntimeError("MP3 export produced invalid file")

                        chunks[idx]["audio_path"] = f"voicelines/{mp3_filename}"

                    except Exception as e:
                        if "invalid file" not in str(e).lower():
                            print(f"MP3 conversion failed for chunk {idx}: {e}")
                        wav_filename = f"{filename_base}.wav"
                        wav_filepath = os.path.join(self.voicelines_dir, wav_filename)
                        shutil.copy(temp_path, wav_filepath)
                        chunks[idx]["audio_path"] = f"voicelines/{wav_filename}"

                    chunks[idx]["status"] = "done"
                    results["completed"].append(idx)
                    print(f"Chunk {idx} completed: {chunks[idx]['audio_path']}")

                    if os.path.exists(temp_path):
                        for attempt in range(3):
                            try:
                                os.remove(temp_path)
                                break
                            except OSError:
                                if attempt < 2:
                                    time.sleep(0.1 * (attempt + 1))
                                else:
                                    print(f"Warning: Could not delete temp file {temp_path}")

                except Exception as e:
                    print(f"Error processing chunk {idx}: {e}")
                    results["failed"].append((idx, str(e)))
                    chunks[idx]["status"] = "error"

            for idx, error in batch_results["failed"]:
                if 0 <= idx < len(chunks):
                    chunks[idx]["status"] = "error"
                results["failed"].append((idx, error))

            self.save_chunks(chunks)

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
