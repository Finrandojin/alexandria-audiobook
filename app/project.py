import os
import json
import shutil
from tts import (
    generate_voice,
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
        with open(self.chunks_path, "w") as f:
            json.dump(chunks, f, indent=2)

    def update_chunk(self, index, data):
        chunks = self.load_chunks()
        if 0 <= index < len(chunks):
            chunk = chunks[index]
            # Update fields
            if "text" in data: chunk["text"] = data["text"]
            if "style" in data: chunk["style"] = data["style"]
            if "speaker" in data: chunk["speaker"] = data["speaker"]

            # If text/style/speaker changed, reset status (but keep old audio until regen)
            if "text" in data or "style" in data or "speaker" in data:
                chunk["status"] = "pending"

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
            style = chunk["style"]

            # Generate to temp file
            temp_path = os.path.join(self.root_dir, "temp_chunk.wav")

            success = generate_voice(text, style, speaker, voice_config, temp_path, client)

            if success:
                # Convert to mp3 and save to voicelines
                segment = AudioSegment.from_wav(temp_path)

                if len(segment) == 0:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    chunk["status"] = "error"
                    self.save_chunks(chunks)
                    return False, "Generated audio has 0 duration"

                filename = f"voiceline_{index+1:04d}_{sanitize_filename(speaker)}.mp3"
                filepath = os.path.join(self.voicelines_dir, filename)
                segment.export(filepath, format="mp3")

                chunk["status"] = "done"
                # Store relative path for frontend
                chunk["audio_path"] = f"voicelines/{filename}"
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
                        segment = AudioSegment.from_mp3(full_path)
                        audio_segments.append(segment)
                        speakers.append(chunk["speaker"])
                    except:
                        pass # Skip bad files

        if not audio_segments:
            return False, "No audio segments found"

        final_audio = combine_audio_with_pauses(audio_segments, speakers)
        output_filename = "cloned_audiobook.mp3"
        output_path = os.path.join(self.root_dir, output_filename)
        final_audio.export(output_path, format="mp3")

        return True, output_filename
