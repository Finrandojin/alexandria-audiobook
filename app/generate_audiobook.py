import os
import re
import json
from pydub import AudioSegment
from gradio_client import Client, handle_file
import shutil
from tts import (
    sanitize_filename,
    preprocess_text_for_tts,
    test_tts_connection,
    generate_custom_voice,
    generate_clone_voice,
    generate_voice,
    combine_audio_with_pauses,
    DEFAULT_PAUSE_MS,
    SAME_SPEAKER_PAUSE_MS
)

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

def main():
    # Load configurations
    config = {}
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
    except:
        print("Warning: config.json not found or invalid. Using defaults.")

    voice_config = {}
    try:
        with open("../voice_config.json", "r") as f:
            voice_config = json.load(f)
    except:
        pass

    tts_url = config.get("tts", {}).get("url", "http://127.0.0.1:7860")
    if not tts_url:
        print("Error: TTS URL not found in config.json")
        return

    # Test TTS connection
    if not test_tts_connection(tts_url, voice_config):
        print("\nAborting: TTS connection test failed.")
        return

    print(f"\nConnecting to TTS server at {tts_url}...")
    client = Client(tts_url)

    # Read the JSON script
    with open("../annotated_script.json", "r", encoding="utf-8") as f:
        script_entries = json.load(f)

    # Group into chunks
    chunks = group_into_chunks(script_entries, MAX_CHUNK_CHARS)

    print(f"Loaded {len(script_entries)} script entries, grouped into {len(chunks)} chunks\n")

    audio_segments = []
    chunk_speakers = []

    temp_dir = "output_audio_cloned"
    os.makedirs(temp_dir, exist_ok=True)

    voicelines_dir = "../voicelines"
    os.makedirs(voicelines_dir, exist_ok=True)

    successful = 0
    failed = 0

    for i, chunk in enumerate(chunks):
        speaker = chunk["speaker"]
        text = chunk["text"]
        style = chunk["style"]

        temp_path = os.path.join(temp_dir, f"chunk_{i}.wav")
        preview = text[:60] + "..." if len(text) > 60 else text
        style_preview = f" [{style}]" if style else ""
        print(f"[{i+1}/{len(chunks)}] {speaker}{style_preview} ({len(text)} chars): '{preview}'")

        if generate_voice(text, style, speaker, voice_config, temp_path, client):
            try:
                segment = AudioSegment.from_wav(temp_path)
                audio_segments.append(segment)
                chunk_speakers.append(speaker)

                # Export individual voiceline
                voiceline_filename = f"voiceline_{i+1:04d}_{sanitize_filename(speaker)}.mp3"
                voiceline_path = os.path.join(voicelines_dir, voiceline_filename)
                segment.export(voiceline_path, format="mp3")

                successful += 1
            except Exception as e:
                print(f"  Could not process audio file: {e}")
                failed += 1
        else:
            failed += 1

    print(f"\n--- Generation Complete ---")
    print(f"Successful: {successful}, Failed: {failed}")

    if not audio_segments:
        print("No audio segments were generated. Exiting.")
        return

    unique_speakers = sorted(set(chunk_speakers))
    print(f"\nSpeakers ({len(unique_speakers)}): {', '.join(unique_speakers)}")
    print(f"Individual voicelines saved to: {os.path.abspath(voicelines_dir)}/")

    print(f"\nCombining {len(audio_segments)} audio segments with pauses...")
    print(f"  Pause between speakers: {DEFAULT_PAUSE_MS}ms")
    print(f"  Pause within same speaker: {SAME_SPEAKER_PAUSE_MS}ms")

    final_audio = combine_audio_with_pauses(audio_segments, chunk_speakers)
    output_filename = "../cloned_audiobook.mp3"
    final_audio.export(output_filename, format="mp3")
    print(f"Combined audiobook saved as {output_filename}")


if __name__ == '__main__':
    main()
