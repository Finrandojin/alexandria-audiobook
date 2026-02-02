import os
import json
from pydub import AudioSegment
import gradio_client
import shutil

def generate_cloned_voice_line(line_text, speaker, voice_config, tts_url, output_path):
    try:
        voice_data = voice_config.get(speaker)
        if not voice_data:
            print(f"Warning: No voice configuration found for speaker '{speaker}'. Skipping line.")
            return False

        ref_audio_path = voice_data.get("ref_audio")
        ref_text = voice_data.get("ref_text")

        if not ref_audio_path or not ref_text:
            print(f"Warning: Incomplete voice configuration for speaker '{speaker}'. Skipping line.")
            return False
            
        client = gradio_client.Client(tts_url)
        
        result = client.predict(
            ref_audio_path,
            ref_text,
            line_text, # target_text
            "Auto", # language
            False, # use_xvector_only
            "1.7B", # model_size
            200, # max_chunk_chars
            0, # chunk_gap
            -1, # seed
            api_name="/generate_voice_clone"
        )
        
        generated_audio_filepath = result[0]
        if not os.path.exists(generated_audio_filepath):
            print(f"Error: Gradio client did not return a valid file path for line: '{line_text}'")
            return False

        shutil.copy(generated_audio_filepath, output_path)
        return True

    except Exception as e:
        print(f"Error generating voice for line: '{line_text}' with speaker '{speaker}'. Error: {e}")
        return False

def main():
    # Load configurations
    with open("config.json", "r") as f:
        config = json.load(f)
    
    with open("voice_config.json", "r") as f:
        voice_config = json.load(f)

    tts_url = config.get("tts", {}).get("url")
    if not tts_url:
        print("Error: TTS URL not found in config.json")
        return

    # Read the annotated script
    with open("annotated_script.txt", "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    audio_segments = []
    output_dir = "output_audio_cloned"
    os.makedirs(output_dir, exist_ok=True)

    for i, line in enumerate(lines):
        if ':' not in line:
            print(f"Skipping line without speaker: '{line}'")
            continue

        speaker, text_to_speak = line.split(':', 1)
        speaker = speaker.strip().replace('*', '').replace('#', '')
        text_to_speak = text_to_speak.strip()

        if not text_to_speak or not speaker:
            continue

        output_path = os.path.join(output_dir, f"line_{i}.wav")
        print(f"Generating voice for '{speaker}': '{text_to_speak[:50]}...'")
        
        if generate_cloned_voice_line(text_to_speak, speaker, voice_config, tts_url, output_path):
            try:
                audio_segments.append(AudioSegment.from_wav(output_path))
            except Exception as e:
                print(f"Could not process audio file {output_path}: {e}")

    if not audio_segments:
        print("No audio segments were generated. Exiting.")
        return

    final_audio = sum(audio_segments)
    output_filename = "cloned_audiobook.mp3"
    final_audio.export(output_filename, format="mp3")
    print(f"

Audiobook saved as {output_filename}")


if __name__ == '__main__':
    main()
