import json
import os
import re

def main():
    input_path = "annotated_script.txt"
    output_path = "voices.json"
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Please generate the script first.")
        return

    voices = set()
    
    # Regex to capture the speaker part before the colon
    speaker_regex = re.compile(r'^\s*([a-zA-Z0-9\s_]+):')

    with open(input_path, 'r') as f:
        for line in f:
            match = speaker_regex.match(line)
            if match:
                # Extract the speaker, strip extra whitespace and markdown chars
                speaker = match.group(1).strip().replace('*', '').replace('#', '')
                if speaker:
                    voices.add(speaker)

    # Convert set to list for JSON serialization
    voice_list = sorted(list(voices))
    
    with open(output_path, 'w') as f:
        json.dump(voice_list, f, indent=4)
        
    print(f"Found {len(voice_list)} unique voices: {voice_list}")
    print(f"Saved voice list to {output_path}")

if __name__ == '__main__':
    main()
