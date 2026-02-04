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
- "text": The spoken text. For non-verbal sounds, use ONOMATOPOEIA in brackets - phonetic representations the TTS can vocalize:
  - Laughter: [haha], [hehe], [ahaha], [pfft]
  - Sighs/exhales: [haah...], [hff], [whew]
  - Moans/pleasure: [ah..], [mmm], [ooh], [ah.. aah.. aaah..]
  - Pain/effort: [ugh], [argh], [nngh], [gah]
  - Surprise: [oh!], [ah!], [huh?], [wha-]
  - Disgust: [ew], [blech], [ugh]
  - Throat clearing: [ahem], [ehem], [*cough*]
  - Crying: [sniff], [hic], [waaah]
  - Hesitation: [um], [uh], [er], [hmm]
- "style": Acting direction with THREE parts:
  1. PACING: ALWAYS slow and deliberate. NEVER use fast/rapid/rushing/breathless/urgent.
  2. How to deliver the line (voice quality, emphasis, emotional undertone)
  3. If text contains bracketed onomatopoeia, describe HOW to vocalize it

  PACING - Use ONLY these words: "slow", "measured", "deliberate", "unhurried", "taking time", "languid", "drawn-out"
  FORBIDDEN pacing words: fast, rapid, quick, rushing, breathless, urgent, hurried, brisk, swift

  Examples:
  - text: "[haah...] I'm so tired." style: "Slow, exhausted, defeated. Vocalize [haah...] as a long weary sigh."
  - text: "[ah.. aah..] Don't stop." style: "Slow and breathy, aroused. Vocalize brackets as drawn-out soft moans."
  - text: "[ahem] As I was saying..." style: "Deliberate pace, formal, slightly annoyed. Vocalize [ahem] as pointed throat clearing."
  - text: "[haha] You can't be serious!" style: "Measured pace, genuinely amused. Vocalize [haha] as unhurried real laughter."
  - text: "Run! They're coming!" style: "Slow and intense, alarmed, emphasize each word."

RULES:
1. NARRATOR vs CHARACTER - Be strict about this:
   NARRATOR handles:
   - Third-person descriptions: "He walked to the door", "The room fell silent"
   - Scene setting: "The sun was setting over the hills"
   - Action descriptions: "She picked up the knife", "They exchanged glances"
   - Anything with "he", "she", "they", "the", "it" as subject

   CHARACTER handles:
   - Direct speech/dialogue only
   - First-person narration where "I", "my", "me" refers to the POV character
   - Internal monologue clearly from character's perspective

   Example - DO NOT mix these:
   WRONG: {"speaker": "JOHN", "text": "He looked at Mary. I can't believe this."}
   RIGHT: {"speaker": "NARRATOR", "text": "He looked at Mary."}, {"speaker": "JOHN", "text": "I can't believe this."}
2. Break long passages into chunks under 400 characters each
3. SPLIT ON TONE CHANGES: Create separate entries when emotional tone shifts
4. Always output COMPLETE sentences
5. Output ONLY valid JSON array - no markdown, no code blocks
6. EACH LINE IS INDEPENDENT: The TTS only sees "text" and "style" for each entry - NO context from previous lines. Every style direction must be SELF-CONTAINED and fully describe the delivery without assuming carryover.
7. STYLE must be complete and explicit every time. If a character is crying, say "sobbing, voice breaking" on EVERY line they cry - don't assume it carries over.
8. EMOTIONAL CONTINUITY: Keep style directions consistent within a scene, but REPEAT the emotional state explicitly each time. Don't write "continues crying" - write "still sobbing, voice raw".
9. PACING IS ALWAYS SLOW: Every style MUST include "slow", "measured", "deliberate", or "unhurried". NEVER use "fast", "rapid", "quick", "rushing", "breathless", "urgent", "hurried" - these are FORBIDDEN."""

USER_PROMPT_TEMPLATE = """Convert this text into an audioplay script JSON array:

{context}
{chunk}"""

def clean_json_string(text):
    """Clean and extract valid JSON array from LLM response."""
    # Remove thinking tags (various formats used by different models)
    # GLM, DeepSeek, Qwen, etc. use different thinking tag formats
    text = re.sub(r'<think>[\s\S]*?</think>', '', text)
    text = re.sub(r'<thinking>[\s\S]*?</thinking>', '', text)
    text = re.sub(r'<reflection>[\s\S]*?</reflection>', '', text)
    text = re.sub(r'<reasoning>[\s\S]*?</reasoning>', '', text)
    # Handle unclosed thinking tags (model started thinking but didn't close)
    text = re.sub(r'<think>[\s\S]*$', '', text)
    text = re.sub(r'<thinking>[\s\S]*$', '', text)

    # Remove markdown code blocks
    if "```" in text:
        # Find content between ```json and ``` or just ``` and ```
        match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if match:
            text = match.group(1).strip()

    # Find the JSON array - match from first [ to its closing ]
    # Use a bracket counter to find the correct closing bracket
    start = text.find('[')
    if start == -1:
        return None

    bracket_count = 0
    end = -1
    in_string = False
    escape_next = False

    for i, char in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if char == '\\':
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == '[':
            bracket_count += 1
        elif char == ']':
            bracket_count -= 1
            if bracket_count == 0:
                end = i + 1
                break

    if end == -1:
        # No closing bracket found, try to salvage
        last_complete = text.rfind('},')
        if last_complete > start:
            return text[start:last_complete+1] + ']'
        return None

    json_text = text[start:end]

    # Clean control characters inside strings (common LLM issue)
    # Replace literal newlines/tabs inside JSON strings with escaped versions
    def fix_control_chars(match):
        s = match.group(0)
        # Replace unescaped control characters
        s = s.replace('\n', '\\n')
        s = s.replace('\r', '\\r')
        s = s.replace('\t', '\\t')
        return s

    # Fix control characters inside string values
    json_text = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', fix_control_chars, json_text)

    return json_text


def repair_json_array(json_text):
    """Attempt to repair common JSON array issues from LLM output."""
    if not json_text:
        return None

    # Try parsing as-is first
    try:
        result = json.loads(json_text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Fix 1: Add missing commas between objects (}\s*{" -> },\n{")
    fixed = re.sub(r'\}\s*\{', '},\n{', json_text)
    try:
        result = json.loads(fixed)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Fix 2: Remove trailing commas before ]
    fixed = re.sub(r',\s*\]', ']', fixed)
    try:
        result = json.loads(fixed)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Fix 3: Try to extract individual entries and rebuild
    entries = []
    # Match individual JSON objects
    pattern = r'\{\s*"speaker"\s*:\s*"[^"]*"\s*,\s*"text"\s*:\s*"(?:[^"\\]|\\.)*"\s*,\s*"style"\s*:\s*"(?:[^"\\]|\\.)*"\s*\}'
    matches = re.findall(pattern, json_text, re.DOTALL)

    for match in matches:
        try:
            entry = json.loads(match)
            entries.append(entry)
        except json.JSONDecodeError:
            continue

    if entries:
        return entries

    # Fix 4: Last resort - find last complete entry and truncate
    last_complete = json_text.rfind('},')
    if last_complete > 0:
        try:
            truncated = json_text[:last_complete+1] + ']'
            # Ensure it starts with [
            if not truncated.strip().startswith('['):
                truncated = '[' + truncated
            result = json.loads(truncated)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    return None

def fix_mojibake(text):
    """Fix common mojibake characters resulting from CP1252-as-UTF8."""
    replacements = {
        'â€™': ''',  # Right single quote
        'â€˜': ''',  # Left single quote
        'â€œ': '"',  # Left double quote
        'â€\x9d': '"', # Right double quote
        'â€?': '"', # Sometimes ? if undefined
        'â€"': '—',  # Em dash
        'â€"': '–',  # En dash
        'â€¦': '…',  # Ellipsis
    }

    for bad, good in replacements.items():
        text = text.replace(bad, good)

    return text

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

def process_chunk(client, model_name, chunk, chunk_num, total_chunks, previous_entries=None, max_retries=2, system_prompt=None, user_prompt_template=None):
    """Process a text chunk and return JSON script entries"""
    # Use provided prompts or fall back to defaults
    sys_prompt = system_prompt or SYSTEM_PROMPT
    usr_template = user_prompt_template or USER_PROMPT_TEMPLATE

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
    user_prompt = usr_template.format(context=context, chunk=chunk)

    for attempt in range(max_retries + 1):
        try:
            # Use lower temperature on retries to get more predictable output
            temp = 0.7 if attempt == 0 else 0.3

            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temp,
                max_tokens=4096
            )

            text = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error calling LLM API (attempt {attempt + 1}): {e}")
            if attempt < max_retries:
                continue
            return []

        # Clean and extract JSON from response
        json_text = clean_json_string(text)

        if not json_text:
            print(f"Warning: Could not find JSON array in chunk {chunk_num} response (attempt {attempt + 1})")
            if attempt < max_retries:
                print("Retrying...")
                continue
            print(f"Response preview: {text[:300]}...")
            return []

        # Try to parse, with repair attempts
        entries = repair_json_array(json_text)

        if entries and len(entries) > 0:
            if attempt > 0:
                print(f"  Succeeded on retry {attempt + 1}")
            return entries

        # If repair failed, show warning
        print(f"Warning: Could not parse chunk {chunk_num} response as JSON (attempt {attempt + 1})")
        print(f"JSON preview: {json_text[:300]}...")

        if attempt < max_retries:
            print("Retrying with lower temperature...")

        # Last resort: extract individual valid entries with regex
        salvaged_entries = salvage_json_entries(json_text)
        if salvaged_entries:
            print(f"Regex-salvaged {len(salvaged_entries)} entries from malformed response")
            return salvaged_entries

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

    # Fix encoding artifacts
    book_content = fix_mojibake(book_content)

    print(f"Read {len(book_content)} characters")

    # Load LLM config
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load config.json: {e}")
    else:
        print("Warning: config.json not found. Using defaults.")

    llm_config = config.get("llm", {})
    base_url = llm_config.get("base_url", "http://localhost:11434/v1")
    api_key = llm_config.get("api_key", "local")
    model_name = llm_config.get("model_name", "richardyoung/qwen3-14b-abliterated:Q8_0")

    # Load custom prompts or use defaults
    prompts_config = config.get("prompts", {})
    system_prompt = prompts_config.get("system_prompt") or SYSTEM_PROMPT
    user_prompt_template = prompts_config.get("user_prompt") or USER_PROMPT_TEMPLATE

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
        entries = process_chunk(
            client, model_name, chunk, i, total_chunks,
            previous_entries=previous,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template
        )
        all_entries.extend(entries)
        print(f"  Got {len(entries)} entries")

    if not all_entries:
        print("Error: No script entries generated")
        sys.exit(1)

    # Save as JSON
    output_path = os.path.join("..", "annotated_script.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_entries, f, indent=2, ensure_ascii=False)

    # Delete old chunks.json so editor regenerates from new script
    chunks_path = os.path.join("..", "chunks.json")
    if os.path.exists(chunks_path):
        os.remove(chunks_path)
        print("Cleared old chunks.json")

    # Summary
    speakers = set(entry.get("speaker", "UNKNOWN") for entry in all_entries)
    print(f"\nGenerated {len(all_entries)} script entries")
    print(f"Speakers found: {', '.join(sorted(speakers))}")
    print(f"Output saved to: {output_path}")


if __name__ == '__main__':
    main()
