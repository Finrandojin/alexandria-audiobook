import os
import json
import shutil
import threading
import zipfile
import io
import time
from tts import (
    TTSEngine,
    combine_audio_with_pauses,
    sanitize_filename,
    DEFAULT_PAUSE_MS,
    SAME_SPEAKER_PAUSE_MS
)
from pydub import AudioSegment

MAX_CHUNK_CHARS = 500

def get_speaker(entry):
    """Get speaker from entry, checking both 'speaker' and 'type' fields."""
    return entry.get("speaker") or entry.get("type") or ""


def group_into_chunks(script_entries, max_chars=MAX_CHUNK_CHARS):
    """Group consecutive entries by same speaker into chunks up to max_chars"""
    if not script_entries:
        return []

    chunks = []
    current_speaker = get_speaker(script_entries[0])
    current_text = script_entries[0].get("text", "")
    current_instruct = script_entries[0].get("instruct", "")

    for entry in script_entries[1:]:
        speaker = get_speaker(entry)
        text = entry.get("text", "")
        instruct = entry.get("instruct", "")

        if speaker == current_speaker:
            combined = current_text + " " + text
            if len(combined) <= max_chars:
                current_text = combined
                # Keep the more specific instruct if available
                if instruct and not current_instruct:
                    current_instruct = instruct
            else:
                chunks.append({
                    "speaker": current_speaker,
                    "text": current_text,
                    "instruct": current_instruct,
                })
                current_text = text
                current_instruct = instruct
        else:
            chunks.append({
                "speaker": current_speaker,
                "text": current_text,
                "instruct": current_instruct,
            })
            current_speaker = speaker
            current_text = text
            current_instruct = instruct

    # Don't forget the last chunk
    chunks.append({
        "speaker": current_speaker,
        "text": current_text,
        "instruct": current_instruct,
    })

    return chunks

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
            except: pass

        try:
            self.engine = TTSEngine(config)
            print(f"TTS engine initialized (mode={self.engine.mode})")
            return self.engine
        except Exception as e:
            print(f"Failed to initialize TTS engine: {e}")
            return None

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
            with open(self.script_path, "r", encoding="utf-8") as f:
                script = json.load(f)
            chunks = group_into_chunks(script)

            # Initialize chunk status
            for i, chunk in enumerate(chunks):
                chunk["id"] = i
                chunk["status"] = "pending" # pending, generating, done, error
                chunk["audio_path"] = None

            self.save_chunks(chunks)
            return chunks

        return []

    def _atomic_json_write(self, data, target_path, max_retries=5):
        """Atomically write JSON data with retry logic for Windows file locking."""
        tmp_path = target_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        for attempt in range(max_retries):
            try:
                os.replace(tmp_path, target_path)
                return
            except OSError as e:
                if attempt < max_retries - 1 and (
                    e.errno == 5 or "Access is denied" in str(e) or "being used by another process" in str(e)
                ):
                    delay = 0.05 * (2 ** attempt)
                    time.sleep(delay)
                    continue
                raise

    def save_chunks(self, chunks):
        with self._chunks_lock:
            self._atomic_json_write(chunks, self.chunks_path)

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
            self._atomic_json_write(chunks, self.chunks_path)
            return chunks[index]

    def update_chunk(self, index, data):
        chunks = self.load_chunks()
        if 0 <= index < len(chunks):
            chunk = chunks[index]
            # Update fields
            if "text" in data: chunk["text"] = data["text"]
            if "instruct" in data: chunk["instruct"] = data["instruct"]
            if "speaker" in data: chunk["speaker"] = data["speaker"]

            # If text/instruct/speaker changed, reset status (but keep old audio until regen)
            if "text" in data or "instruct" in data or "speaker" in data:
                chunk["status"] = "pending"

            print(f"update_chunk({index}): instruct='{chunk.get('instruct', '')}', speaker='{chunk.get('speaker', '')}'")
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
            text = chunk["text"]
            instruct = chunk.get("instruct", "")

            print(f"Generating chunk {index}: speaker={speaker}, instruct='{instruct}', text='{text[:50]}...'")

            # Generate to temp file (unique per chunk for parallel processing)
            temp_path = os.path.join(self.root_dir, f"temp_chunk_{index}.wav")

            success = engine.generate_voice(text, instruct, speaker, voice_config, temp_path)

            if success:
                # Check file size
                if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                     self._update_chunk_fields(index, status="error")
                     return False, "Generated audio file is missing or empty"

                print(f"Generated WAV size: {os.path.getsize(temp_path)} bytes")

                # Try to convert to mp3, fallback to wav if ffmpeg missing
                filename_base = f"voiceline_{index+1:04d}_{sanitize_filename(speaker)}"
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

    def merge_audio(self):
        chunks = self.load_chunks()
        audio_segments = []
        speakers = []

        for chunk in chunks:
            path = chunk.get("audio_path")
            if path:
                full_path = os.path.join(self.root_dir, path)
                if os.path.exists(full_path):
                    try:
                        # Auto-detect format (mp3 or wav)
                        segment = AudioSegment.from_file(full_path)
                        audio_segments.append(segment)
                        speakers.append(chunk["speaker"])
                    except Exception as e:
                        print(f"Error loading audio segment {path}: {e}")

        if not audio_segments:
            return False, "No audio segments found"

        final_audio = combine_audio_with_pauses(audio_segments, speakers)
        output_filename = "cloned_audiobook.mp3"
        output_path = os.path.join(self.root_dir, output_filename)
        final_audio.export(output_path, format="mp3")

        return True, output_filename

    def export_audacity(self):
        """Export project as an Audacity-compatible zip with per-speaker WAV tracks,
        a LOF file for auto-import, and a labels file for chunk annotations."""
        chunks = self.load_chunks()

        # Phase 1 — Compute timeline (matching merge_audio pause logic exactly)
        timeline = []  # list of (chunk, segment, abs_start_ms)
        prev_speaker = None
        cursor_ms = 0

        for chunk in chunks:
            path = chunk.get("audio_path")
            if not path:
                continue
            full_path = os.path.join(self.root_dir, path)
            if not os.path.exists(full_path):
                continue
            try:
                segment = AudioSegment.from_file(full_path)
            except Exception as e:
                print(f"Error loading audio for Audacity export {path}: {e}")
                continue

            speaker = chunk["speaker"]
            if prev_speaker is not None:
                if speaker == prev_speaker:
                    cursor_ms += SAME_SPEAKER_PAUSE_MS
                else:
                    cursor_ms += DEFAULT_PAUSE_MS

            timeline.append((chunk, segment, cursor_ms))
            cursor_ms += len(segment)
            prev_speaker = speaker

        if not timeline:
            return False, "No audio segments found"

        total_duration_ms = cursor_ms

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

    def generate_chunks_parallel(self, indices, max_workers=2, progress_callback=None):
        """Generate multiple chunks in parallel using ThreadPoolExecutor.

        Uses individual TTS API calls with per-speaker voice settings.

        Args:
            indices: List of chunk indices to generate
            max_workers: Number of concurrent TTS workers
            progress_callback: Optional callback(completed, failed, total) for progress updates

        Returns:
            dict with 'completed' and 'failed' lists
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = {"completed": [], "failed": []}
        total = len(indices)

        if total == 0:
            return results

        print(f"Starting parallel generation of {total} chunks with {max_workers} workers...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.generate_chunk_audio, idx): idx
                for idx in indices
            }

            for future in as_completed(futures):
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

        print(f"Parallel generation complete: {len(results['completed'])} succeeded, {len(results['failed'])} failed")
        return results

    def generate_chunks_batch(self, indices, batch_seed=-1, batch_size=4, progress_callback=None):
        """Generate multiple chunks using batch TTS API with a single seed.

        Faster than parallel but uses same seed for all voices. Only works with
        custom voices (clone voices will be skipped).

        Args:
            indices: List of chunk indices to generate
            batch_seed: Single seed for all generations (-1 for random)
            batch_size: Number of chunks per batch request
            progress_callback: Optional callback(completed, failed, total) for progress updates

        Returns:
            dict with 'completed' and 'failed' lists
        """
        results = {"completed": [], "failed": []}
        total = len(indices)

        if total == 0:
            return results

        print(f"Starting batch generation of {total} chunks (batch_size={batch_size}, seed={batch_seed})...")

        # Load chunks and voice config
        chunks = self.load_chunks()
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

        # Split indices into batches
        batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
        print(f"Processing {len(batches)} batches...")

        for batch_num, batch_indices in enumerate(batches):
            print(f"Batch {batch_num + 1}/{len(batches)}: {len(batch_indices)} chunks")

            # Build batch request data
            batch_chunks = []
            for idx in batch_indices:
                if 0 <= idx < len(chunks):
                    chunk = chunks[idx]
                    batch_chunks.append({
                        "index": idx,
                        "text": chunk.get("text", ""),
                        "instruct": chunk.get("instruct", ""),
                        "speaker": chunk.get("speaker", "")
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

        print(f"Batch generation complete: {len(results['completed'])} succeeded, {len(results['failed'])} failed")
        return results
