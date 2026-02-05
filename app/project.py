import os
import json
import shutil
import threading
from tts import (
    generate_voice,
    generate_batch,
    combine_audio_with_pauses,
    sanitize_filename,
    DEFAULT_PAUSE_MS,
    SAME_SPEAKER_PAUSE_MS
)
from pydub import AudioSegment
from gradio_client import Client

MAX_CHUNK_CHARS = 500

def group_into_chunks(script_entries, max_chars=MAX_CHUNK_CHARS):
    """Group consecutive entries by same speaker into chunks up to max_chars"""
    if not script_entries:
        return []

    chunks = []
    current_speaker = script_entries[0].get("speaker")
    current_text = script_entries[0].get("text", "")
    current_style = script_entries[0].get("style", "")

    for entry in script_entries[1:]:
        speaker = entry.get("speaker")
        text = entry.get("text", "")
        style = entry.get("style", "")

        if speaker == current_speaker:
            combined = current_text + " " + text
            if len(combined) <= max_chars:
                current_text = combined
                # Keep the more specific style if available
                if style and not current_style:
                    current_style = style
            else:
                chunks.append({
                    "speaker": current_speaker,
                    "text": current_text,
                    "style": current_style
                })
                current_text = text
                current_style = style
        else:
            chunks.append({
                "speaker": current_speaker,
                "text": current_text,
                "style": current_style
            })
            current_speaker = speaker
            current_text = text
            current_style = style

    # Don't forget the last chunk
    chunks.append({
        "speaker": current_speaker,
        "text": current_text,
        "style": current_style
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

        self.client = None
        self._chunks_lock = threading.Lock()  # Thread-safe file writes

    def get_client(self):
        if self.client:
            return self.client

        # Load config to get URL
        url = "http://127.0.0.1:7860"
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    config = json.load(f)
                    url = config.get("tts", {}).get("url", url)
            except: pass

        print(f"Connecting to TTS server at {url}...")
        try:
            self.client = Client(url)
            return self.client
        except Exception as e:
            print(f"Failed to connect to TTS: {e}")
            return None

    def load_chunks(self):
        if os.path.exists(self.chunks_path):
            with open(self.chunks_path, "r") as f:
                return json.load(f)

        # If no chunks, generate from script
        if os.path.exists(self.script_path):
            with open(self.script_path, "r") as f:
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

    def save_chunks(self, chunks):
        with self._chunks_lock:
            with open(self.chunks_path, "w") as f:
                json.dump(chunks, f, indent=2)

    def update_chunk(self, index, data):
        chunks = self.load_chunks()
        if 0 <= index < len(chunks):
            chunk = chunks[index]
            old_style = chunk.get("style", "")

            # Update fields
            if "text" in data: chunk["text"] = data["text"]
            if "style" in data: chunk["style"] = data["style"]
            if "speaker" in data: chunk["speaker"] = data["speaker"]

            # If text/style/speaker changed, reset status (but keep old audio until regen)
            if "text" in data or "style" in data or "speaker" in data:
                chunk["status"] = "pending"

            print(f"update_chunk({index}): style changed from '{old_style}' to '{chunk.get('style', '')}'")
            self.save_chunks(chunks)
            return chunk
        return None

    def generate_chunk_audio(self, index):
        chunks = self.load_chunks()
        if not (0 <= index < len(chunks)):
            return False, "Invalid chunk index"

        chunk = chunks[index]
        chunk["status"] = "generating"
        self.save_chunks(chunks)

        try:
            client = self.get_client()
            if not client:
                chunk["status"] = "error"
                self.save_chunks(chunks)
                return False, "TTS Client not connected"

            # Load voice config
            voice_config = {}
            if os.path.exists(self.voice_config_path):
                with open(self.voice_config_path, "r") as f:
                    voice_config = json.load(f)

            speaker = chunk["speaker"]
            text = chunk["text"]
            style = chunk.get("style", "")

            print(f"Generating chunk {index}: speaker={speaker}, style='{style}', text='{text[:50]}...'")

            # Generate to temp file (unique per chunk for parallel processing)
            temp_path = os.path.join(self.root_dir, f"temp_chunk_{index}.wav")

            success = generate_voice(text, style, speaker, voice_config, temp_path, client)

            if success:
                # Check file size
                if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                     chunk["status"] = "error"
                     self.save_chunks(chunks)
                     return False, "Generated audio file is missing or empty"

                print(f"Generated WAV size: {os.path.getsize(temp_path)} bytes")

                # Try to convert to mp3, fallback to wav if ffmpeg missing
                filename_base = f"voiceline_{index+1:04d}_{sanitize_filename(speaker)}"

                try:
                    segment = AudioSegment.from_wav(temp_path)

                    if len(segment) == 0:
                         chunk["status"] = "error"
                         self.save_chunks(chunks)
                         return False, "Generated audio has 0 duration"

                    mp3_filename = f"{filename_base}.mp3"
                    mp3_filepath = os.path.join(self.voicelines_dir, mp3_filename)

                    # This might fail if ffmpeg is missing
                    segment.export(mp3_filepath, format="mp3")

                    chunk["audio_path"] = f"voicelines/{mp3_filename}"

                except Exception as e:
                    print(f"MP3 conversion failed (ffmpeg missing?): {e}")
                    # Fallback: copy WAV
                    wav_filename = f"{filename_base}.wav"
                    wav_filepath = os.path.join(self.voicelines_dir, wav_filename)
                    shutil.copy(temp_path, wav_filepath)

                    chunk["audio_path"] = f"voicelines/{wav_filename}"

                chunk["status"] = "done"
                self.save_chunks(chunks)

                # Cleanup
                if os.path.exists(temp_path):
                    os.remove(temp_path)

                return True, chunk["audio_path"]
            else:
                chunk["status"] = "error"
                self.save_chunks(chunks)
                return False, "Generation failed"

        except Exception as e:
            chunk["status"] = "error"
            self.save_chunks(chunks)
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
            with open(self.voice_config_path, "r") as f:
                voice_config = json.load(f)

        # Get TTS client
        client = self.get_client()
        if not client:
            for idx in indices:
                results["failed"].append((idx, "TTS Client not connected"))
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
                        "style": chunk.get("style", ""),
                        "speaker": chunk.get("speaker", "")
                    })

            # Call batch TTS with single seed
            batch_results = generate_batch(batch_chunks, voice_config, self.root_dir, client, batch_seed)

            # Process completed chunks - convert to MP3 and update status
            chunks = self.load_chunks()  # Reload for each batch

            for idx in batch_results["completed"]:
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
                        chunks[idx]["audio_path"] = f"voicelines/{mp3_filename}"

                    except Exception as e:
                        print(f"MP3 conversion failed for chunk {idx}: {e}")
                        wav_filename = f"{filename_base}.wav"
                        wav_filepath = os.path.join(self.voicelines_dir, wav_filename)
                        shutil.copy(temp_path, wav_filepath)
                        chunks[idx]["audio_path"] = f"voicelines/{wav_filename}"

                    chunks[idx]["status"] = "done"
                    results["completed"].append(idx)
                    print(f"Chunk {idx} completed: {chunks[idx]['audio_path']}")

                    if os.path.exists(temp_path):
                        os.remove(temp_path)

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
