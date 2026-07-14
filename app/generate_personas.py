import os
import sys
import json
import time
import re
import argparse
import shutil
import tempfile
from openai import OpenAI

from tts import TTSEngine, sanitize_filename
from utils import atomic_json_write as _atomic_json_write
from persona_prompts import PERSONA_SYSTEM_PROMPT, PERSONA_USER_PROMPT, PERSONA_ADVANCED_PROMPT


def extract_json_object(text):
    # Find first JSON object in text
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    end = None
    for i, ch in enumerate(text[start:], start):
        if esc:
            esc = False
            continue
        if ch == '\\':
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        return None
    obj_text = text[start:end]
    try:
        return json.loads(obj_text)
    except Exception:
        return None


def normalize_speaker_name(name):
    if not isinstance(name, str):
        return ""
    s = name.strip().lower()
    # Remove common honorifics and punctuation for alias heuristics
    s = re.sub(r'^(mr|mrs|ms|miss|dr|prof|sir|lady|lord)\.?\s+', '', s)
    s = re.sub(r'[^a-z0-9\s]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _token_jaccard(a: str, b: str) -> float:
    """Jaccard similarity on normalized name tokens."""
    norm_a = normalize_speaker_name(a)
    norm_b = normalize_speaker_name(b)
    if not norm_a or not norm_b:
        return 0.0
    tokens_a = set(norm_a.split())
    tokens_b = set(norm_b.split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a.intersection(tokens_b)
    union = tokens_a.union(tokens_b)
    return len(intersection) / len(union)


def _resolve_to_canonical(raw_name: str, allowed: list, threshold=0.4) -> str | None:
    """Map a raw name to the closest canonical label, or None.

    Tries:
    1. Exact match after normalization.
    2. Substring match after normalization.
    3. Token Jaccard similarity.
    """
    if not raw_name:
        return None

    norm_raw = normalize_speaker_name(raw_name)
    if not norm_raw:
        return None

    # Step 1: Exact match after normalization
    for name in allowed:
        if normalize_speaker_name(name) == norm_raw:
            return name

    # Step 2: Substring match (either raw is in name, or name is in raw)
    for name in allowed:
        norm_name = normalize_speaker_name(name)
        if norm_name and (norm_name in norm_raw or norm_raw in norm_name):
            return name

    # Step 3: Token Jaccard similarity
    best_name = None
    best_score = 0.0
    for name in allowed:
        score = _token_jaccard(raw_name, name)
        if score > best_score:
            best_score = score
            best_name = name

    if best_score >= threshold:
        return best_name

    return None


_NARRATOR_LABELS = frozenset({"NARRATOR", "NARRATION", "NARRATIVE"})


def _collect_narrator_context(script, speaker, window=4):
    """Gather unique narrator lines within `window` entries (before and after) of any appearance.

    - Scans both before and after each appearance.
    - Looks at all appearances, not just the first.
    - Accepts any speaker labels in _NARRATOR_LABELS.
    """
    context_lines = []
    seen_lines = set()
    window = max(1, int(window or 4))

    # Find all indices of the speaker's appearances in the script
    speaker_indices = []
    for i, entry in enumerate(script):
        if _entry_speaker(entry) == speaker:
            speaker_indices.append(i)

    # For each appearance, look at the window around it
    for idx in speaker_indices:
        # Check before and after
        start_idx = max(0, idx - window)
        end_idx = min(len(script), idx + window + 1)
        for j in range(start_idx, end_idx):
            if j == idx:
                continue
            entry = script[j]
            entry_speaker = _entry_speaker(entry).upper()
            entry_text = _entry_text(entry)
            if entry_speaker in _NARRATOR_LABELS and entry_text:
                if entry_text not in seen_lines:
                    seen_lines.add(entry_text)
                    context_lines.append(entry_text)
                    if len(context_lines) >= window:
                        return context_lines

    return context_lines


def _resolve_aliases_batch(client, model_name, speakers_info, existing_names):
    """Resolve aliases for all speakers in a single one-shot LLM call.

    speakers_info is a dict:
    {
      "SPEAKER_NAME": {
         "sample_lines": [...],
         "narrator_context": [...]
      }
    }

    Returns a dict mapping each raw speaker label to its canonical group leader name.
    """
    if not speakers_info:
        return {}

    prompt_items = []
    for speaker, info in speakers_info.items():
        samples = "\n".join(f"  - {line}" for line in info["sample_lines"][:3])
        context = "\n".join(f"  - {line}" for line in info["narrator_context"][:2])
        prompt_items.append(
            f"Speaker label: '{speaker}'\n"
            f"Nearby narrator context:\n{context or '  (None)'}\n"
            f"Sample spoken lines:\n{samples or '  (None)'}"
        )

    formatted_speakers = "\n\n---\n\n".join(prompt_items)
    candidates = "\n".join(f"- {name}" for name in existing_names) if existing_names else "(none)"

    prompt = (
        "You are an expert audiobook production assistant specializing in character identification.\n"
        "Below is a list of speaker labels found in a script, along with their sample lines and surrounding narrator context.\n\n"
        "Your task is to analyze these speakers globally and identify which labels represent the same character (aliases/variants) "
        "and which are truly distinct characters. Group them under a single canonical name (preferably the most complete or common name).\n\n"
        "Rules:\n"
        "1. Identify duplicates, minor spelling variations, honorific variations (e.g., 'Mr. Darcy' and 'Darcy'), and nickname vs full name relationships.\n"
        "2. For each input speaker label, specify its resolved canonical name.\n"
        "3. If a speaker is unique and has no other aliases, its canonical name should just be itself.\n"
        "4. You can also map a speaker label to one of the existing configured characters listed below if it is a match.\n\n"
        f"Existing configured characters:\n{candidates}\n\n"
        "Return ONLY one JSON object where keys are the original input speaker labels, and values are their resolved canonical names.\n"
        "Example shape:\n"
        "{\n"
        "  \"DARCY\": \"MR. DARCY\",\n"
        "  \"ELIZABETH BENNET\": \"ELIZABETH BENNET\",\n"
        "  \"LIZZY\": \"ELIZABETH BENNET\"\n"
        "}\n\n"
        f"Speakers to analyze:\n\n{formatted_speakers}"
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a precise casting director. You output ONLY valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=max(1500, len(speakers_info) * 80),
        )
        result = extract_json_object(response.choices[0].message.content.strip())
        if isinstance(result, dict):
            # Normalize keys and values to match exact input names casing
            resolved = {}
            for k, v in result.items():
                if isinstance(k, str) and isinstance(v, str):
                    resolved[k.strip()] = v.strip()
            return resolved
    except Exception as e:
        print(f"Warning: Batch alias resolution failed: {e}")

    return {}


def pick_ref_text(lines):
    for ln in lines:
        if ln and len(ln.strip()) >= 12:
            return ln.strip()
    return next((ln.strip() for ln in lines if ln and ln.strip()), "")


def parse_alias_decision(text):
    parsed = extract_json_object(text)
    if not parsed:
        return {
            "is_alias": False,
            "alias_of": "",
            "description": "",
            "ref_text": "",
            "reason": "unparseable"
        }
    return {
        "is_alias": bool(parsed.get("is_alias", False)),
        "alias_of": str(parsed.get("alias_of", "") or "").strip(),
        "description": str(parsed.get("description", "") or "").strip(),
        "ref_text": str(parsed.get("ref_text", "") or "").strip(),
        "reason": str(parsed.get("reason", "") or "").strip()
    }


def _as_list(value):
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _unique_extend(existing, values, limit=80):
    seen = {str(v).strip().lower() for v in existing if str(v).strip()}
    for value in _as_list(values):
        key = value.lower()
        if key not in seen:
            existing.append(value)
            seen.add(key)
        if len(existing) >= limit:
            break
    return existing


def _entry_speaker(entry):
    return (entry.get("speaker") or entry.get("type") or "").strip()


def _entry_text(entry):
    return (entry.get("text") or "").strip()


def _batch_entries(script, batch_size):
    batch_size = max(1, int(batch_size or 40))
    for start in range(0, len(script), batch_size):
        yield start, script[start:start + batch_size]


def _json_preview(data, max_chars=12000):
    text = json.dumps(data, ensure_ascii=False, indent=2)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...TRUNCATED..."


def _character_ref_path(ref_dir, speaker):
    safe = sanitize_filename(speaker or "unknown")
    return os.path.join(ref_dir, f"{safe}.json")


def _load_character_ref(ref_dir, speaker):
    path = _character_ref_path(ref_dir, speaker)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "name": speaker,
        "aliases": [],
        "features": [],
        "personality": [],
        "voice_clues": [],
        "relationships": [],
        "sample_lines": [],
        "observations": [],
    }


def _append_character_ref(ref_dir, speaker, batch_number, character_data):
    ref = _load_character_ref(ref_dir, speaker)
    ref["name"] = ref.get("name") or speaker
    ref["aliases"] = _unique_extend(ref.get("aliases", []), character_data.get("aliases", []))
    ref["features"] = _unique_extend(ref.get("features", []), character_data.get("features", []), limit=120)
    ref["personality"] = _unique_extend(ref.get("personality", []), character_data.get("personality", []), limit=120)
    ref["voice_clues"] = _unique_extend(ref.get("voice_clues", []), character_data.get("voice_clues", []), limit=120)
    ref["relationships"] = _unique_extend(ref.get("relationships", []), character_data.get("relationships", []), limit=120)
    ref["sample_lines"] = _unique_extend(ref.get("sample_lines", []), character_data.get("sample_lines", []), limit=50)

    observation = {
        "batch": batch_number,
        "features": _as_list(character_data.get("features", [])),
        "personality": _as_list(character_data.get("personality", [])),
        "voice_clues": _as_list(character_data.get("voice_clues", [])),
        "relationships": _as_list(character_data.get("relationships", [])),
        "evidence": character_data.get("evidence", []),
        "sample_lines": _as_list(character_data.get("sample_lines", [])),
    }
    ref.setdefault("observations", []).append(observation)
    ref["updated_at"] = int(time.time())
    _atomic_json_write(ref, _character_ref_path(ref_dir, speaker))
    return ref


def _build_batch_discovery_prompt(batch_start, batch, allowed_speakers):
    lines = []
    for offset, entry in enumerate(batch):
        speaker = _entry_speaker(entry)
        text = _entry_text(entry)
        if not speaker and not text:
            continue
        lines.append(f"[{batch_start + offset}] {speaker}: {text}")

    allowed = "\n".join(f"- {name}" for name in allowed_speakers) if allowed_speakers else "(none)"
    batch_text = "\n".join(lines)
    return (
        "You are building character reference files for an audiobook voice generator.\n"
        "Read this batch of script entries and do only two things:\n"
        "1. discover which characters/speakers are present from the allowed labels;\n"
        "2. describe observed character features, personality, and voice-relevant clues.\n\n"
        "The ONLY valid character keys are listed below. Do not invent new names.\n"
        "Return ONLY one JSON object where each key is EXACTLY one of the allowed labels:\n"
        "{\n"
        "  \"SPEAKER_LABEL\": {\n"
        "    \"aliases\": [\"optional alternate names\"],\n"
        "    \"features\": [\"physical/social/role facts supported by this batch\"],\n"
        "    \"personality\": [\"personality traits supported by this batch\"],\n"
        "    \"voice_clues\": [\"age, gender, accent, timbre, pace, tone, delivery clues\"],\n"
        "    \"relationships\": [\"relationships to other characters if explicit\"],\n"
        "    \"evidence\": [{\"entry_index\": 0, \"quote\": \"short quote\"}],\n"
        "    \"sample_lines\": [\"good spoken sample lines for this character\"]\n"
        "  }\n"
        "}\n\n"
        f"Allowed speaker labels:\n{allowed}\n\n"
        f"Script batch:\n{batch_text}"
    )


def _fallback_batch_characters(batch):
    by_speaker = {}
    for offset, entry in enumerate(batch):
        speaker = _entry_speaker(entry)
        text = _entry_text(entry)
        if not speaker:
            continue
        data = by_speaker.setdefault(speaker, {
            "name": speaker,
            "aliases": [],
            "features": [],
            "personality": [],
            "voice_clues": [],
            "relationships": [],
            "evidence": [],
            "sample_lines": [],
        })
        if text and len(data["sample_lines"]) < 3:
            data["sample_lines"].append(text)
        if text and len(data["evidence"]) < 3:
            data["evidence"].append({"entry_index": offset, "quote": text[:240]})
    return list(by_speaker.values())


def _compile_character_prompt(character_ref, prompt_template=None):
    compact = {
        "name": character_ref.get("name", ""),
        "aliases": character_ref.get("aliases", [])[:20],
        "features": character_ref.get("features", [])[:80],
        "personality": character_ref.get("personality", [])[:80],
        "voice_clues": character_ref.get("voice_clues", [])[:80],
        "relationships": character_ref.get("relationships", [])[:60],
        "sample_lines": character_ref.get("sample_lines", [])[:30],
        "observations": character_ref.get("observations", [])[-30:],
    }
    if prompt_template:
        return prompt_template.format(character_ref=_json_preview(compact))
    return (
        "You are compiling an audiobook character reference into a final TTS voice persona.\n"
        "Use only supported observations. The final description should be practical for voice design.\n"
        "Return ONLY one JSON object with keys:\n"
        "- description: 2-4 sentences covering apparent age/gender if inferable, timbre, accent/dialect, pace, emotional baseline, personality, and delivery guidance.\n"
        "- ref_text: 1-2 representative spoken sentences from the character, or the best available sample line.\n\n"
        f"Character reference:\n{_json_preview(compact)}"
    )


def _fallback_compiled_persona(character_ref):
    name = character_ref.get("name", "Character")
    parts = []
    for key in ("voice_clues", "personality", "features"):
        parts.extend(character_ref.get(key, [])[:5])
    description = f"{name} has a clear, natural audiobook voice."
    if parts:
        description = f"{name} should sound like: " + "; ".join(parts[:10]) + "."
    ref_text = pick_ref_text(character_ref.get("sample_lines", []))
    return description, ref_text


def _save_generated_preview(root, engine, voice_config, speaker, description, ref_text):
    try:
        wav_path, sr = engine.generate_voice_design(description=description, sample_text=ref_text)
        dest_dir = os.path.join(root, "designed_voices")
        os.makedirs(dest_dir, exist_ok=True)
        safe = sanitize_filename(speaker)
        voice_id = f"{safe}_{int(time.time_ns())}"
        dest_filename = f"{voice_id}_preview.wav"
        dest_path = os.path.join(dest_dir, dest_filename)
        try:
            shutil.copy2(wav_path, dest_path)
        except Exception as e:
            print(f"Warning: Could not copy preview for {speaker}: {e}")

        voice_entry = voice_config.get(speaker, {})
        voice_entry.update({
            "type": "clone",
            "ref_audio": os.path.relpath(dest_path, root).replace('\\\\', '/'),
            "ref_text": ref_text,
            "description": description,
            "character_style": description,
            "seed": -1
        })
        voice_config[speaker] = voice_entry

        meta_path = os.path.join(dest_dir, f"{voice_id}_meta.json")
        _atomic_json_write({"description": description, "ref_text": ref_text, "preview": os.path.relpath(dest_path, root)}, meta_path)

        try:
            manifest_path = os.path.join(dest_dir, 'manifest.json')
            manifest = []
            if os.path.exists(manifest_path):
                try:
                    with open(manifest_path, 'r', encoding='utf-8') as mf:
                        manifest = json.load(mf)
                except Exception:
                    manifest = []

            stale_entries = [entry for entry in manifest if entry.get("name") == speaker]
            manifest = [entry for entry in manifest if entry.get("name") != speaker]
            for stale in stale_entries:
                stale_filename = stale.get("filename") or ""
                if stale_filename:
                    stale_path = os.path.join(dest_dir, stale_filename)
                    if os.path.exists(stale_path) and os.path.abspath(stale_path) != os.path.abspath(dest_path):
                        try:
                            os.remove(stale_path)
                        except OSError:
                            pass
                stale_meta = stale.get("id")
                if stale_meta:
                    stale_meta_path = os.path.join(dest_dir, f"{stale_meta}_meta.json")
                    if os.path.exists(stale_meta_path):
                        try:
                            os.remove(stale_meta_path)
                        except OSError:
                            pass

            manifest.append({
                "id": voice_id,
                "name": speaker,
                "description": description,
                "sample_text": ref_text,
                "filename": os.path.basename(dest_path)
            })
            _atomic_json_write(manifest, manifest_path)
        except Exception as e:
            print(f"Warning: could not update manifest for {speaker}: {e}")

        print(f"Persona generated and preview saved for {speaker}: {dest_path}")
        return True
    except Exception as e:
        print(f"Error generating voice preview for {speaker}: {e}")
        voice_entry = voice_config.get(speaker, {})
        voice_entry.update({"type": "design", "description": description, "ref_text": ref_text})
        voice_config[speaker] = voice_entry
        return False


def run_advanced_persona_generation(script, selected_speakers, samples, voice_config, client, model_name, engine, root, args, system_prompt=None, advanced_prompt=None):
    ref_dir = os.path.join(root, "persona_refs")
    os.makedirs(ref_dir, exist_ok=True)

    selected_set = set(selected_speakers)
    batches = list(_batch_entries(script, args.batch_size))
    print(f"Advanced persona generation enabled.")
    print(f"Writing per-character reference files to: {ref_dir}")
    print(f"Processing {len(script)} script entries in {len(batches)} batches of up to {max(1, int(args.batch_size or 40))}")

    for batch_number, (batch_start, batch) in enumerate(batches, start=1):
        prompt = _build_batch_discovery_prompt(batch_start, batch, selected_speakers)
        print(f"Advanced discovery batch {batch_number}/{len(batches)} ({len(batch)} entries)")
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You produce concise JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=4000,
            )
            raw_content = response.choices[0].message.content.strip()
            parsed = extract_json_object(raw_content)
            characters = []
            if isinstance(parsed, dict):
                if "characters" in parsed and isinstance(parsed["characters"], list):
                    characters = parsed["characters"]
                else:
                    for key, val in parsed.items():
                        if isinstance(val, dict):
                            val["name"] = key
                            characters.append(val)
            if not characters:
                print(f"Warning: discovery batch {batch_number} returned no parseable characters; using speaker fallback.")
                print(f"  LLM response (first 500 chars): {raw_content[:500]}")
                characters = _fallback_batch_characters(batch)
        except Exception as e:
            print(f"Warning: discovery batch {batch_number} failed: {e}; using speaker fallback.")
            characters = _fallback_batch_characters(batch)

        for character in characters:
            if not isinstance(character, dict):
                continue
            speaker = str(character.get("name") or character.get("speaker") or character.get("speaker_label") or "").strip()
            if not speaker:
                continue
            
            # Map raw/fuzzy name to allowed canonical speaker labels
            canonical_speaker = _resolve_to_canonical(speaker, selected_speakers)
            if not canonical_speaker:
                continue
            
            character["name"] = canonical_speaker
            _append_character_ref(ref_dir, canonical_speaker, batch_number, character)

    print("Compiling character reference files into final voice personas.")
    for speaker in selected_speakers:
        ref = _load_character_ref(ref_dir, speaker)
        if not ref.get("sample_lines"):
            ref["sample_lines"] = [line for line in samples.get(speaker, [])[:8] if line]
            _atomic_json_write(ref, _character_ref_path(ref_dir, speaker))

        print(f"Compiling persona for: {speaker}")
        description = ""
        ref_text = ""
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt or "You produce concise JSON only."},
                    {"role": "user", "content": _compile_character_prompt(ref, advanced_prompt)}
                ],
                temperature=0.25,
                max_tokens=600,
            )
            parsed = extract_json_object(response.choices[0].message.content.strip())
            if parsed:
                description = str(parsed.get("description", "") or "").strip()
                ref_text = str(parsed.get("ref_text", "") or "").strip()
        except Exception as e:
            print(f"Warning: compile failed for {speaker}: {e}")

        if not description:
            description, ref_text = _fallback_compiled_persona(ref)
        if not ref_text:
            ref_text = pick_ref_text(samples.get(speaker, []))
        if not ref_text:
            ref_text = f"{speaker} speaks in a clear, natural voice."
        if not description:
            print(f"Warning: Empty compiled description for {speaker}, skipping")
            continue

        voice_entry = voice_config.get(speaker, {})
        voice_entry["persona_ref"] = os.path.relpath(_character_ref_path(ref_dir, speaker), root).replace('\\\\', '/')
        voice_config[speaker] = voice_entry
        _save_generated_preview(root, engine, voice_config, speaker, description, ref_text)
        time.sleep(0.5)


# _atomic_json_write imported from utils


def main():
    parser = argparse.ArgumentParser(description="Generate personas for speakers in annotated script")
    parser.add_argument("--new-only", action="store_true", help="Process only speakers missing from voice_config.json")
    parser.add_argument("--alias-check", action="store_true", help="Use LLM + heuristics to decide alias_of vs truly new character")
    parser.add_argument("--advanced", action="store_true", help="Batch the full script into per-character reference files before compiling voice personas")
    parser.add_argument("--batch-size", type=int, default=40, help="Script entries per advanced discovery batch")
    parser.add_argument("--speakers", default="", help="Optional comma-separated speaker allowlist")
    parser.add_argument("--narration-window", type=int, default=4, help="How many preceding narrator lines to include as intro context")
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_path = os.path.join(root, "annotated_script.json")
    voice_config_path = os.path.join(root, "voice_config.json")
    app_config_path = os.path.join(os.path.dirname(__file__), "config.json")

    if not os.path.exists(script_path):
        print(f"Error: {script_path} not found. Generate script first.")
        sys.exit(1)

    with open(script_path, "r", encoding="utf-8") as f:
        script = json.load(f)

    # Collect sample lines per speaker + first-appearance narrator context
    samples = {}
    first_index = {}
    for i, entry in enumerate(script):
        speaker = _entry_speaker(entry)
        if not speaker:
            continue
        samples.setdefault(speaker, []).append(entry.get("text", "").strip())
        if speaker not in first_index:
            first_index[speaker] = i

    narrator_context = {}
    window = max(1, int(args.narration_window or 4))
    for speaker in samples.keys():
        narrator_context[speaker] = _collect_narrator_context(script, speaker, window)

    # Load LLM config
    config = {}
    if os.path.exists(app_config_path):
        try:
            with open(app_config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load app/config.json: {e}")

    llm_cfg = config.get("llm", {})
    base_url = llm_cfg.get("base_url", "http://localhost:11434/v1")
    api_key = llm_cfg.get("api_key", "local")
    model_name = llm_cfg.get("model_name", "richardyoung/qwen3-14b-abliterated:Q8_0")

    client = OpenAI(base_url=base_url, api_key=api_key)

    # Load persona prompts from config, fall back to defaults
    prompts_cfg = config.get("prompts", {})
    persona_system = prompts_cfg.get("persona_system_prompt") or PERSONA_SYSTEM_PROMPT
    persona_user = prompts_cfg.get("persona_user_prompt") or PERSONA_USER_PROMPT
    persona_advanced = prompts_cfg.get("persona_advanced_prompt") or PERSONA_ADVANCED_PROMPT

    # Load existing voice_config (preserve other fields)
    voice_config = {}
    if os.path.exists(voice_config_path):
        try:
            with open(voice_config_path, "r", encoding="utf-8") as f:
                voice_config = json.load(f)
        except Exception:
            voice_config = {}

    # Disable compile_codec for persona previews: compilation overhead
    # outweighs benefit for single generations, and subprocess context
    # can trigger HIP kernel errors on ROCm.
    tts_cfg = dict(config.get("tts", {}))
    tts_cfg["compile_codec"] = False
    engine = TTSEngine({"tts": tts_cfg})

    selected_speakers = list(samples.keys())
    if args.new_only:
        selected_speakers = [s for s in selected_speakers if s not in voice_config]
    if args.speakers.strip():
        allow = {s.strip() for s in args.speakers.split(",") if s.strip()}
        selected_speakers = [s for s in selected_speakers if s in allow]

    if not selected_speakers:
        print("No speakers to process.")
        return

    if args.advanced:
        run_advanced_persona_generation(
            script=script,
            selected_speakers=selected_speakers,
            samples=samples,
            voice_config=voice_config,
            client=client,
            model_name=model_name,
            engine=engine,
            root=root,
            args=args,
            system_prompt=persona_system,
            advanced_prompt=persona_advanced,
        )
        try:
            _atomic_json_write(voice_config, voice_config_path)
            print(f"Updated voice_config saved to {voice_config_path}")
        except Exception as e:
            print(f"Failed to save voice_config.json: {e}")
        return

    print(f"Processing {len(selected_speakers)} speakers")

    # Step 1: Pre-process with exact heuristic + high-confidence fuzzy matching
    resolved_aliases = {}
    remaining_speakers = []

    for speaker in selected_speakers:
        existing_names = [n for n in voice_config.keys() if n != speaker]
        # Fast heuristic exact check
        norm_self = normalize_speaker_name(speaker)
        heuristic_alias = ""
        for candidate in existing_names:
            if normalize_speaker_name(candidate) == norm_self and norm_self:
                heuristic_alias = candidate
                break

        if not heuristic_alias and existing_names:
            # High-confidence fuzzy check
            heuristic_alias = _resolve_to_canonical(speaker, existing_names, threshold=0.8)

        if heuristic_alias:
            print(f"Fast heuristic/fuzzy alias detected: {speaker} -> {heuristic_alias}")
            resolved_aliases[speaker] = heuristic_alias
        else:
            remaining_speakers.append(speaker)

    # Step 2: One-shot batch alias resolution for the remaining speakers (if alias-check is enabled)
    if args.alias_check and remaining_speakers:
        print(f"Running one-shot batch alias resolution for {len(remaining_speakers)} candidates...")
        
        # Split remaining speakers into chunks of 25 to prevent context/output token exhaustion
        chunk_size = 25
        batch_mapping = {}
        
        for idx in range(0, len(remaining_speakers), chunk_size):
            chunk = remaining_speakers[idx:idx + chunk_size]
            speakers_info = {}
            for speaker in chunk:
                speakers_info[speaker] = {
                    "sample_lines": samples.get(speaker, []),
                    "narrator_context": narrator_context.get(speaker, [])
                }
            
            # Use current configured names plus any previously resolved canonical names as existing references
            existing_configured = list(voice_config.keys()) + list(batch_mapping.values())
            
            print(f"Resolving alias batch {idx//chunk_size + 1} ({len(chunk)} speakers)...")
            chunk_mapping = _resolve_aliases_batch(client, model_name, speakers_info, existing_configured)
            batch_mapping.update(chunk_mapping)

        # Build case-insensitive normalized lookup mapping to survive LLM key casing changes
        normalized_mapping = {}
        for k, v in batch_mapping.items():
            if k and v:
                normalized_mapping[normalize_speaker_name(k)] = v

        for speaker in remaining_speakers:
            norm_speaker = normalize_speaker_name(speaker)
            resolved_name = normalized_mapping.get(norm_speaker, speaker)
            if resolved_name != speaker:
                # LLM identified this as an alias!
                all_possible = list(voice_config.keys()) + remaining_speakers
                canonical_target = _resolve_to_canonical(resolved_name, all_possible, threshold=0.6)
                if canonical_target and canonical_target != speaker:
                    print(f"Batch LLM alias detected: {speaker} -> {canonical_target}")
                    resolved_aliases[speaker] = canonical_target
                else:
                    print(f"Batch LLM mapped '{speaker}' to '{resolved_name}' but couldn't resolve canonical spelling. Treating as new.")

    # Apply all resolved aliases to voice_config
    for speaker, alias_target in resolved_aliases.items():
        voice_entry = voice_config.get(speaker, {})
        voice_entry.update({
            "alias_of": alias_target,
            "seed": voice_entry.get("seed", -1),
        })
        voice_config[speaker] = voice_entry

    # Step 3: Generate personas for remaining truly unique speakers
    unique_speakers = [s for s in remaining_speakers if s not in resolved_aliases]
    print(f"Generating personas for {len(unique_speakers)} unique speakers...")

    for speaker in unique_speakers:
        lines = samples.get(speaker, [])
        try:
            print(f"Generating persona for: {speaker} ({len(lines)} lines samples)")

            sample_text = "\n".join(lines[:8])
            intro_ctx = narrator_context.get(speaker, [])
            intro_blob = "\n".join(intro_ctx) if intro_ctx else "(No nearby narrator intro lines found.)"

            user_prompt = persona_user.format(
                speaker=speaker,
                narrator_context=intro_blob,
                sample_lines=sample_text
            )

            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": persona_system},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=400,
            )

            text = response.choices[0].message.content.strip()
            parsed = extract_json_object(text)
            description = ""
            ref_text = ""
            if parsed:
                description = str(parsed.get("description", "") or "").strip()
                ref_text = str(parsed.get("ref_text", "") or "").strip()

            if not description:
                print(f"Warning: LLM did not return parseable JSON for {speaker}. Response preview:\n{text[:300]}")
                # Fallback to compiled fallback persona
                description, ref_text = _fallback_compiled_persona({
                    "name": speaker,
                    "sample_lines": lines
                })

            if not ref_text:
                ref_text = pick_ref_text(lines)

            # Generate and save voice preview
            _save_generated_preview(root, engine, voice_config, speaker, description, ref_text)

            time.sleep(0.5)

        except Exception as e:
            print(f"Unhandled error for {speaker}: {e}")

    # Persist voice_config
    try:
        _atomic_json_write(voice_config, voice_config_path)
        print(f"Updated voice_config saved to {voice_config_path}")
    except Exception as e:
        print(f"Failed to save voice_config.json: {e}")


if __name__ == '__main__':
    main()
