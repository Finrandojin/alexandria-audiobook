import os
import sys
import json
import re
from openai import OpenAI

SYSTEM_PROMPT = """You are a script writer converting books/novels into audioplay scripts. Output ONLY valid JSON arrays, no markdown, no explanations.

OUTPUT FORMAT:
[
  {"speaker": "NARRATOR", "text": "Description text here.", "style": "tone direction"},
  {"speaker": "CHARACTER", "text": "Dialogue here.", "style": "emotional direction"}
]

FIELDS:
- "speaker": Character name in UPPERCASE (use "NARRATOR" only for third-person descriptions)
- "text": The spoken text, with bracketed non-verbal sounds where appropriate
- "style": Brief delivery direction (2-5 words like "warm, nostalgic" or "cold, threatening")

NON-VERBAL SOUNDS - Include where emotionally appropriate:
[sighs], [laughs], [chuckles], [giggles], [scoffs], [gasps], [groans], [moans],
[whimpers], [sobs], [cries], [sniffs], [whispers], [shouts], [screams],
[clears throat], [coughs], [pauses], [hesitates], [stammers], [gulps]

RULES:
1. FIRST-PERSON vs THIRD-PERSON:
   - "I", "my", "me" (first-person) = use CHARACTER'S NAME, NOT "NARRATOR"
   - "He", "She", "The" (third-person) = use "NARRATOR"
2. Break long passages into chunks under 400 characters each
3. SPLIT ON TONE CHANGES: Create separate entries when emotional tone shifts
4. Always output COMPLETE sentences
5. Output ONLY valid JSON array - no markdown, no code blocks"""

USER_PROMPT_TEMPLATE = """Convert this text into an audioplay script JSON array:

{context}
{chunk}"""

def split_into_chunks(text, max_size=3000):
    """Split text into chunks at paragraph/sentence boundaries."""
    paragraphs = re.split(r'\n\s*\n', text)

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) + 2 > max_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            if len(para) > max_size:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 1 > max_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence
                    else:
                        current_chunk += " " + sentence if current_chunk else sentence
            else:
                current_chunk = para
        else:
            current_chunk += "\n\n" + para if current_chunk else para

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def process_chunk(client, model_name, chunk, chunk_num, total_chunks, previous_entries=None):
    """Process a text chunk and return JSON script entries"""
    context_parts = []

    if chunk_num == 1:
        context_parts.append("(Beginning of text)")
    elif chunk_num == total_chunks:
        context_parts.append("(End of text)")
    else:
        context_parts.append(f"(Part {chunk_num} of {total_chunks})")

    if previous_entries and len(previous_entries) > 0:
        last_speaker = previous_entries[-1].get("speaker", "UNKNOWN")
        speaker_counts = {}
        for entry in previous_entries:
            s = entry.get("speaker", "")
            if s and s != "NARRATOR":
                speaker_counts[s] = speaker_counts.get(s, 0) + 1

        if speaker_counts:
            main_char = max(speaker_counts, key=speaker_counts.get)
            context_parts.append(f"Main character: {main_char}. Last speaker: {last_speaker}.")
            context_parts.append(f"First-person text ('I', 'my') is {main_char} speaking.")

    context = "\n".join(context_parts)

    user_prompt = USER_PROMPT_TEMPLATE.format(context=context, chunk=chunk)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=4096
        )

        text = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling LLM API: {e}")
        return []

    # Clean up markdown code blocks if present
    if text.startswith("```"):
        lines = text.split("\n")
        end_idx = -1
        for i, line in enumerate(lines[1:], 1):
            if line.strip().startswith("```"):
                end_idx = i
                break
        if end_idx > 0:
            text = "\n".join(lines[1:end_idx])
        else:
            text = "\n".join(lines[1:])

    # Try to find JSON array in response
    if not text.startswith("["):
        # Look for JSON array in the response
        match = re.search(r'\[[\s\S]*\]', text)
        if match:
            text = match.group(0)

    try:
        entries = json.loads(text)
        if isinstance(entries, list):
            return entries
    except json.JSONDecodeError as e:
        print(f"Warning: Could not parse chunk {chunk_num} response as JSON: {e}")
        print(f"Response preview: {text[:300]}...")

        # Try to salvage partial JSON
        try:
            last_complete = text.rfind('},')
            if last_complete > 0:
                salvaged = text[:last_complete+1] + ']'
                entries = json.loads(salvaged)
                print(f"Salvaged {len(entries)} entries from partial response")
                return entries
        except:
            pass

    return []

def main():
    if len(sys.argv) < 2:
        print("Error: No input file path provided.")
        print("Usage: python generate_script.py <input_file_path>")
        sys.exit(1)

    input_file_path = sys.argv[1]
    print(f"Processing book from: {input_file_path}")

    if not os.path.exists(input_file_path):
        print(f"Error: Input file not found: {input_file_path}")
        sys.exit(1)

    with open(input_file_path, 'r', encoding='utf-8') as f:
        book_content = f.read()

    print(f"Read {len(book_content)} characters")

    # Load LLM config
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    if not os.path.exists(config_path):
        print("Error: config.json not found. Please run configure.js first.")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = json.load(f)

    llm_config = config.get("llm", {})
    base_url = llm_config.get("base_url", "http://localhost:1234/v1")
    api_key = llm_config.get("api_key", "local")
    model_name = llm_config.get("model_name", "local-model")

    print(f"Connecting to: {base_url}")
    print(f"Using model: {model_name}")

    # Create OpenAI client with custom base URL
    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )

    # Split into chunks at natural boundaries
    chunks = split_into_chunks(book_content, max_size=3000)
    total_chunks = len(chunks)

    print(f"Split into {total_chunks} chunks at paragraph/sentence boundaries")

    all_entries = []
    for i, chunk in enumerate(chunks, 1):
        print(f"Processing chunk {i}/{total_chunks} ({len(chunk)} chars)...")

        previous = all_entries if len(all_entries) > 0 else None
        entries = process_chunk(client, model_name, chunk, i, total_chunks, previous_entries=previous)
        all_entries.extend(entries)
        print(f"  Got {len(entries)} entries")

    if not all_entries:
        print("Error: No script entries generated")
        sys.exit(1)

    # Save as JSON
    output_path = os.path.join("..", "annotated_script.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_entries, f, indent=2, ensure_ascii=False)

    # Summary
    speakers = set(entry.get("speaker", "UNKNOWN") for entry in all_entries)
    print(f"\nGenerated {len(all_entries)} script entries")
    print(f"Speakers found: {', '.join(sorted(speakers))}")
    print(f"Output saved to: {output_path}")


if __name__ == '__main__':
    main()
