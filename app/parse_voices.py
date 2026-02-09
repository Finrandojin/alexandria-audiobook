import json
import os

def main():
    input_path = "../annotated_script.json"
    output_path = "../voices.json"

    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Please generate the script first.")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        script_data = json.load(f)

    # Extract unique speakers (check both "speaker" and "type" fields)
    voices = set()
    for entry in script_data:
        speaker = (entry.get("speaker") or entry.get("type") or "").strip()
        if speaker:
            voices.add(speaker.upper())

    voice_list = sorted(list(voices))

    with open(output_path, 'w', encoding="utf-8") as f:
        json.dump(voice_list, f, indent=2, ensure_ascii=False)

    print(f"Found {len(voice_list)} unique voices: {', '.join(voice_list)}")
    print(f"Saved voice list to {output_path}")

if __name__ == '__main__':
    main()
