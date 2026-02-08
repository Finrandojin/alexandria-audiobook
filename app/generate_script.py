import os
import sys
import json
import re
from openai import OpenAI

SYSTEM_PROMPT = """You are a script writer converting books/novels into audiobook scripts that are read by an advanced TTS system. Output ONLY valid JSON arrays, no markdown, no explanations.

OUTPUT FORMAT:
[
  {"speaker": "NARRATOR", "text": "The coals had grown dim, just a little bit of orange that shone faintly onto Sion's face from underneath, making him look like he was going to tell a ghost story.", "instruct": "Neutral, even narration."},
  {"speaker": "SION", "text": "Steamshield is the city of the future.", "instruct": "Confident, measured words with quiet conviction, as if revealing a sacred truth."},
  {"speaker": "BRIN", "text": "Really.", "instruct": "Flat, skeptical delivery, understated disbelief."},
  {"speaker": "NARRATOR", "text": "Brin could not quite keep the skepticism out of his voice. His experience in this world was like living in the past in most ways. Sure, it was a magical and wonderful version of the past, but still archaic.", "instruct": "Neutral, even narration."}
]
Notice: Brin's spoken word is CHARACTER. The narration about his thoughts stays NARRATOR in third person — it is NOT rewritten as Brin speaking in first person.

FIELDS:
- "speaker": Character name in UPPERCASE. Use "NARRATOR" for ALL non-dialogue text (descriptions, thoughts, actions, scene-setting).
- "text": The spoken text exactly as TTS should say it.
  - PRESERVE THE AUTHOR'S WORDS. Do not change person, tense, or wording. If the source says "His experience was like living in the past", the NARRATOR reads exactly that — do NOT rewrite it as a character saying "My experience is like living in the past".
  - Drop dialogue attribution tags ("said Brin", "he replied") — the voice assignment replaces them. But keep any descriptive action from the attribution as NARRATOR text, using the character's name (not a pronoun) so listeners can connect the voice to the name.
    Source: '"Really," said Brin, not quite able to keep the skepticism out of his voice.'
    → {"speaker": "BRIN", "text": "Really.", ...}
    → {"speaker": "NARRATOR", "text": "Brin could not quite keep the skepticism out of his voice.", ...}
  - Write all sounds as real words the voice can say. NEVER use bracket tags like [gasps], [sighs], <sigh>, <cry> — TTS cannot vocalize these.
  - Prefer merging sounds into dialogue rather than standalone sound entries. Pure sound-only entries cause TTS to loop.
  - WRONG: {"text": "Ahh!"} — too short, will loop. Merge into dialogue instead.
  - Non-human characters should NOT have speaking lines — describe their actions through NARRATOR.
  - The TTS reads text LITERALLY — it cannot interpret abbreviations, Roman numerals, or symbols. Convert anything that would sound wrong when read character-by-character:
    "Chapter I" → "Chapter One", "Chapter IV" → "Chapter Four", "Dr." → "Doctor", "Mr." → "Mister", "St." → "Saint" or "Street" (context), "3rd" → "third", "&" → "and"
- "instruct": Short TTS voice direction (one emotion + one vocal modifier, ~3-8 words). The instruct goes directly to the TTS engine — describe how the VOICE sounds, not what the body does.
  BE DIRECT: No weak qualifiers ("slightly", "a bit", "somewhat") — they get ignored or bleed unpredictably.
  NARRATOR: Always "Neutral, even narration." — nothing else. The TTS finds natural emphasis from the text itself. Do NOT add mood, pace, tone, stress, or pause directions.
  CHARACTER: Read the emotional grain of the line, then amplify it. One clear emotion + one vocal quality modifier. The instruct cannot override the text's natural tone — it can only lean into it.
  BEST PATTERN: "Quietly [emotion]" — e.g. "Quietly menacing.", "Quietly furious.", "Quietly devastated."
  GOOD: "Seething, low and dangerous." / "Cold fury, dangerously quiet." / "Exhausted, drained of everything but this last demand." / "Analytical, dissecting with contempt." / "Venomous contempt, low and controlled."
  WRONG: "Trembling with shock." — physical action, TTS cannot enact it.
  WRONG: "Sneering contempt, dripping with disdain." — synonym stacking dilutes both concepts.
  WRONG: "Noble sacrifice, accepting with dignity." — conflicting emotions collapse into the weaker one.
  DESCRIBE THE VOICE not the body: voice cracking, hollow, seething, flat, low, cold, numb, drained — these work. Trembling, choking, gritting, shaking, gasping — these are ignored.
  PACING words that work: "ponderously orating", "measured words", "deliberate intonation"
  AVOID: "fast", "rapid", "rushing", "breathless", "urgent", "slow" (taken literally — causes robotic pauses)

RULES:
1. NARRATOR vs CHARACTER — The most important rule. Be strict:
   CHARACTER: ONLY words spoken aloud by a character. Dialogue, verbal replies, shouted orders — if another character in the scene would hear it, it is CHARACTER.
   NARRATOR: EVERYTHING else — descriptions, actions, thoughts, internal monologue, scene-setting, reflections. If no one else in the scene would hear it, it is NARRATOR.
   - Thoughts and internal reflections are ALWAYS NARRATOR, even when they concern a specific character.
   - NEVER convert narration into dialogue. If the author wrote it as third-person narration, it stays third-person narration read by NARRATOR.
   WRONG: Source says 'His experience was like living in the past' → {"speaker": "BRIN", "text": "My experience is like living in the past"}
   RIGHT: Source says 'His experience was like living in the past' → {"speaker": "NARRATOR", "text": "His experience was like living in the past."}
   WRONG: {"speaker": "JOHN", "text": "He looked at Mary. I can't believe this."}
   RIGHT: {"speaker": "NARRATOR", "text": "He looked at Mary."}, {"speaker": "JOHN", "text": "I can't believe this."}
   - INTERLEAVED PARAGRAPHS: When a source paragraph mixes narration with quoted speech, you MUST split it into separate entries. Carefully track quotation marks to determine what is spoken vs narrated.
     Source: '"You may be interested in this." He threw over a sheet of pink note-paper. "It came by the last post," said he. "Read it aloud."'
     → {"speaker": "HOLMES", "text": "You may be interested in this."}
     → {"speaker": "NARRATOR", "text": "He threw over a sheet of pink note-paper."}
     → {"speaker": "HOLMES", "text": "It came by the last post. Read it aloud."}
     Source: 'He took down a heavy brown volume from his shelves. "Here we are, Egria. It is in Bohemia."'
     → {"speaker": "NARRATOR", "text": "He took down a heavy brown volume from his shelves."}
     → {"speaker": "HOLMES", "text": "Here we are, Egria. It is in Bohemia."}
2. NEVER remove narration. All narration from the source must appear in the output.
3. PRESERVE THE AUTHOR'S TEXT. Do not change tense, person, or wording.
4. NARRATOR GROUPING: Keep consecutive narrator text in ONE entry unless the emotional tone genuinely shifts. Do NOT split narration into individual sentences — short narrator lines cause inconsistent TTS delivery. Only split when a real tone change occurs (e.g. reflective passage → sudden action).
5. Output ONLY valid JSON array — no markdown, no code blocks.
6. The TTS processes every line in isolation. It does not know what came before or after. Give the instruct field all the context it needs to deliver the line correctly.
7. EMOTIONAL CONTINUITY: Keep instruct directions consistent within a scene. Repeat the emotional state explicitly — don't write "still crying", write "Sad, quiet sobbing."
8. INSTRUCT MUST MATCH THE SCENE: Read the context of the scene before choosing an instruct. A character excitedly showing off deductions is not "seething" — they are "triumphant" or "quietly excited". Match the actual emotional tone, not a random emotion.
9. INSTRUCT VOCABULARY: Describe the VOICE only. No physical actions ("leaning forward", "gritting teeth"), no narrative descriptions ("trying to sound", "offering a guess"), no weak qualifiers ("slightly", "a bit", "faintly"). Write what the voice does: "Hesitant.", "Quietly triumphant.", "Uncertain."
"""

USER_PROMPT_TEMPLATE = """{context}

Remember: if another character in the scene would hear the words, it is CHARACTER dialogue. Everything else is NARRATOR. Preserve the author's original wording, person, and tense exactly.

SOURCE TEXT:
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
    pattern = r'\{\s*"speaker"\s*:\s*"[^"]*"\s*,\s*"text"\s*:\s*"(?:[^"\\]|\\.)*"\s*,\s*"instruct"\s*:\s*"(?:[^"\\]|\\.)*"\s*\}'
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

def salvage_json_entries(json_text):
    """Last resort: extract individual valid entries with regex."""
    entries = []
    # Match individual JSON objects with speaker, text, instruct fields
    pattern = r'\{\s*"speaker"\s*:\s*"([^"]*)"\s*,\s*"text"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"instruct"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}'
    matches = re.finditer(pattern, json_text, re.DOTALL)

    for match in matches:
        try:
            entry = {
                "speaker": match.group(1),
                "text": match.group(2).replace('\\"', '"').replace('\\n', '\n'),
                "instruct": match.group(3).replace('\\"', '"').replace('\\n', '\n')
            }
            entries.append(entry)
        except Exception:
            continue

    return entries if entries else None


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

def process_chunk(client, model_name, chunk, chunk_num, total_chunks, previous_entries=None, max_retries=2, system_prompt=None, user_prompt_template=None, max_tokens=4096, temperature=0.6, top_p=0.8, top_k=20, min_p=0, presence_penalty=0.0, banned_tokens=None):
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
        # Build character roster for name consistency across chunks
        characters_seen = sorted(set(
            entry.get("speaker", "") for entry in previous_entries
            if entry.get("speaker", "") and entry.get("speaker", "") != "NARRATOR"
        ))
        if characters_seen:
            context_parts.append(f"Characters in this book: {', '.join(characters_seen)}")

        # Include last few entries so the model can maintain style and tone continuity
        tail = previous_entries[-3:]
        context_parts.append("\nPrevious section ended with:")
        for entry in tail:
            context_parts.append(json.dumps(entry, ensure_ascii=False))

    context = "\n".join(context_parts)
    user_prompt = usr_template.format(context=context, chunk=chunk)

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                max_tokens=max_tokens,
                extra_body={
                    k: v for k, v in {
                        "top_k": top_k,
                        "min_p": min_p,
                        "banned_tokens": banned_tokens if banned_tokens else None,
                    }.items() if v is not None
                }
            )

            choice = response.choices[0]
            text = choice.message.content.strip()
            finish_reason = choice.finish_reason
            usage = getattr(response, 'usage', None)

            # Log raw response for debugging
            log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "llm_responses.log")
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(f"\n{'='*80}\n")
                lf.write(f"CHUNK {chunk_num}/{total_chunks} | attempt {attempt + 1} | finish_reason={finish_reason}\n")
                if usage:
                    lf.write(f"tokens: prompt={getattr(usage, 'prompt_tokens', '?')} completion={getattr(usage, 'completion_tokens', '?')}\n")
                lf.write(f"{'─'*80}\n")
                lf.write(text)
                lf.write(f"\n{'='*80}\n")

            print(f"  finish_reason={finish_reason}", end="")
            if usage:
                print(f" | tokens: prompt={getattr(usage, 'prompt_tokens', '?')} completion={getattr(usage, 'completion_tokens', '?')}", end="")
            print()

            if finish_reason == "length":
                print(f"  WARNING: Response was truncated (hit max_tokens={max_tokens}). Consider increasing max_tokens.")

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

    # Load generation settings
    generation_config = config.get("generation", {})
    chunk_size = generation_config.get("chunk_size", 3000)
    max_tokens = generation_config.get("max_tokens", 4096)
    temperature = generation_config.get("temperature", 0.6)
    top_p = generation_config.get("top_p", 0.8)
    top_k = generation_config.get("top_k", 20)
    min_p = generation_config.get("min_p", 0)
    presence_penalty = generation_config.get("presence_penalty", 0.0)
    banned_tokens = generation_config.get("banned_tokens", [])

    print(f"Connecting to: {base_url}")
    print(f"Using model: {model_name}")
    print(f"Chunk size: {chunk_size} chars, Max tokens: {max_tokens}")
    if banned_tokens:
        print(f"Banned tokens: {banned_tokens}")

    # Create OpenAI client with custom base URL
    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )

    # Split into chunks at natural boundaries
    chunks = split_into_chunks(book_content, max_size=chunk_size)
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
            user_prompt_template=user_prompt_template,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            banned_tokens=banned_tokens
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

    # Summary (check both "speaker" and "type" fields)
    speakers = set(entry.get("speaker") or entry.get("type") or "UNKNOWN" for entry in all_entries)
    print(f"\nGenerated {len(all_entries)} script entries")
    print(f"Speakers found: {', '.join(sorted(speakers))}")
    print(f"Output saved to: {output_path}")


if __name__ == '__main__':
    main()
