import argparse
import os
import json
import google.generativeai as genai
from pydub import AudioSegment
import requests # for TTS API
import gradio_client # New import for Gradio API interaction
import shutil # New import for file operations

def generate_voice_line(line, output_path, tts_url):
    try:
        # Separate the speaker/label from the text to be spoken
        text_to_speak = line
        if ':' in line:
            text_to_speak = line.split(':', 1)[1]

        # Clean the text to be spoken
        clean_text = text_to_speak.replace('*', '').replace('#', '').strip()
        
        if not clean_text:
            print(f"Skipping empty line after parsing: '{line}'")
            return False

        client = gradio_client.Client(tts_url)
        
        # Using the /generate_voice_design API name as per documentation
        result = client.predict(
            clean_text,
            "Auto",
            "A clear, neutral voice reading a book.",
            -1,
            api_name="/generate_voice_design"
        )
        
        generated_audio_filepath = result[0]
        if not os.path.exists(generated_audio_filepath):
            print(f"Error: Gradio client did not return a valid file path for line: '{clean_text}'")
            return False

        shutil.copy(generated_audio_filepath, output_path)

        return True # Signal success

    except Exception as e:
        print(f"Error generating voice for line: '{line}'. Error: {e}")
        return False # Signal failure

def get_annotated_script(model, chunk):
    response = model.generate_content(
        "You are a script writer. Your task is to convert a book chapter into a script format. The script should be annotated with NARRATOR: for narrative parts and CHARACTER_NAME: for dialogue. Make sure to properly attribute the dialogue to the correct character.\n\n" + chunk
    )
    return response.text

def read_book_in_chunks(file_path, chunk_size=4096):
    with open(file_path, 'r') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk

def main():
    parser = argparse.ArgumentParser(description='Process a book.')
    parser.add_argument('--file', type=str, required=True, help='The path to the book file.')
    args = parser.parse_args()

    print(f"Processing book: {args.file}")

    with open("config.json", "r") as f:
        config = json.load(f)

    genai.configure(api_key="AIzaSyCJCok2v54_L1wL8e21SREAfWIWdL1Ilwc") # TODO: Remove this hardcoded key and use environment variable if publishing to GitHub
    model = genai.GenerativeModel(config["llm"]["model_name"])

    annotated_script = ""
    for chunk in read_book_in_chunks(args.file):
        print(f"Processing chunk of size: {len(chunk)}")
        annotated_script += get_annotated_script(model, chunk)

    print("\n\nAnnotated Script:\n")
    print(annotated_script)

    lines = [line.strip() for line in annotated_script.split("\n") if line.strip()]
    
    audio_segments = []
    output_dir = "output_audio"
    os.makedirs(output_dir, exist_ok=True)

    for i, line in enumerate(lines):
        output_path = os.path.join(output_dir, f"line_{i}.mp3")
        print(f"Generating voice for: {line}")
        if generate_voice_line(line, output_path, config["tts"]["url"]):
            try:
                audio_segments.append(AudioSegment.from_wav(output_path))
            except Exception as e:
                print(f"Could not process audio file {output_path}: {e}")

    final_audio = sum(audio_segments)
    output_filename = os.path.splitext(os.path.basename(args.file))[0] + ".mp3"
    final_audio.export(output_filename, format="mp3")
    print(f"\n\nAudiobook saved as {output_filename}")


if __name__ == '__main__':
    main()
