import argparse
import os
import json
import google.generativeai as genai

def get_annotated_script(model, chunk):
    response = model.generate_content(
        "You are a script writer. Your task is to convert a book chapter into a script format. The script should be annotated with NARRATOR: for narrative parts and CHARACTER_NAME: for dialogue. Make sure to properly attribute the dialogue to the correct character.

" + chunk
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
    parser = argparse.ArgumentParser(description='Generate an annotated script from a book file.')
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

    # Save the script to a file in the parent directory
    output_path = os.path.join("..", "annotated_script.txt")
    with open(output_path, 'w') as f:
        f.write(annotated_script)
    
    print(f"

Annotated script saved to {output_path}")


if __name__ == '__main__':
    main()
