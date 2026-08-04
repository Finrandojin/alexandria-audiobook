import asyncio
import gc
import json
import logging
import os
import re
import signal
import sys
from typing import Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, model_validator
from config_settings import load_app_config

from core import (
    CAST_MAJOR_LINE_THRESHOLD,
    CONFIG_PATH,
    LLMConfigError,
    LORA_MODELS_MANIFEST,
    SCRIPT_PATH,
    VOICE_CONFIG_PATH,
    VOICE_LIBRARY_PATH,
    _load_builtin_lora_manifest,
    _load_manifest,
    _load_voice_library,
    _make_library_entry,
    _make_llm_client,
    _norm_name,
    _script_line_counts,
    _send_signal_tree,
    _warn_corrupted_json,
    check_global_gpu_lock,
    claim_gpu_task,
    get_active_book_id,
    get_cast_adapter_usage,
    get_cast_member_key,
    get_cast_storage_pool,
    get_trait_assignment_metadata,
    process_state,
    project_manager,
    run_process,
)
from lmstudio_settings import get_current_status, get_effective_max_tokens
from tts import voice_category
from utils import (
    atomic_json_write,
    atomic_json_write_pair,
    extract_json_object,
    file_lock,
    safe_load_json,
    secure_filename,
)


logger = logging.getLogger("AlexandriaUI")
router = APIRouter()


class VoiceConfigItem(BaseModel):
    type: str = "custom"
    voice: Optional[str] = "Ryan"
    character_style: Optional[str] = ""
    default_style: Optional[str] = ""  # backward compat, prefer character_style
    seed: Optional[str] = "-1"
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
    adapter_id: Optional[str] = None
    adapter_path: Optional[str] = None
    description: Optional[str] = ""  # voice description (for design type)
    members: Optional[List[str]] = None  # speaker names to voice at once (ensemble type)

    @model_validator(mode="after")
    def validate_ensemble_members(self):
        if self.type == "ensemble" and not self.members:
            raise ValueError("ensemble voices must include at least one member")
        if self.type == "ensemble" and any(not member.strip() for member in self.members):
            raise ValueError("ensemble member names cannot be blank")
        return self

class SuggestVoicesRequest(BaseModel):
    only_unset: bool = False  # only suggest for characters not already set to a lora/builtin_lora voice
    max_lines: int = 8        # how many sample dialogue lines per character to feed the matcher
    cast: Optional[str] = None

class VoiceSuggestionApplyRequest(BaseModel):
    character: str
    cast: Optional[str] = None
    suggestion: Dict

class VoiceSuggestionApplyBulkRequest(BaseModel):
    cast: Optional[str] = None
    suggestions: Dict[str, Dict]


class GeneratePersonasRequest(BaseModel):
    advanced: bool = False
    batch_size: int = 40


@router.get("/api/voices")
async def get_voices():
    # Parse voices directly from the current script (no stale cache)
    voices_list = []
    if os.path.exists(SCRIPT_PATH):
        try:
            with open(SCRIPT_PATH, "r", encoding="utf-8") as f:
                script_data = json.load(f)
            voices_set = set()
            for entry in script_data:
                speaker = (entry.get("speaker") or entry.get("type") or "").strip()
                if speaker:
                    voices_set.add(speaker)
            voices_list = sorted(voices_set)
        except (json.JSONDecodeError, ValueError) as e:
            _warn_corrupted_json("script", SCRIPT_PATH, "returning empty voice list", e)

    if not voices_list:
        return []

    # Combine with config
    voice_config = {}
    if os.path.exists(VOICE_CONFIG_PATH):
        try:
            with open(VOICE_CONFIG_PATH, "r", encoding="utf-8") as f:
                voice_config = json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            _warn_corrupted_json("voice config", VOICE_CONFIG_PATH, "ignoring", e)
            voice_config = {}

    missing_speakers = {voice_name for voice_name in voices_list if voice_name not in voice_config}

    result = []
    for voice_name in voices_list:
        config = voice_config.get(voice_name, {})
        result.append({
            "name": voice_name,
            "config": config,
            "persona_pending": voice_name in missing_speakers
        })
    return result


@router.post("/api/generate_personas")
async def generate_personas(background_tasks: BackgroundTasks, request: GeneratePersonasRequest = GeneratePersonasRequest()):
    """Generate LLM-derived voice persona descriptions and VoiceDesign previews.

    This runs `app/generate_personas.py` which:
    - reads `annotated_script.json`,
    - asks the configured LLM to produce a short `description` and `ref_text` for each character,
    - uses the VoiceDesign model to synthesize a preview and saves it,
    - updates `voice_config.json` with a clone-style reference for each character.
    """
    check_global_gpu_lock("persona")

    process_state["persona"]["cancel"] = False

    # Unload TTS engine to free GPU for the subprocess
    if project_manager.engine is not None:
        logger.info("Unloading TTS engine for persona generation...")
        project_manager.engine = None
        gc.collect()

    command = [sys.executable, "-u", "generate_personas.py"]
    if request.advanced:
        batch_size = max(1, min(int(request.batch_size or 40), 200))
        command.extend(["--advanced", "--batch-size", str(batch_size)])
    claim_gpu_task("persona")
    background_tasks.add_task(run_process, command, "persona")
    return {"status": "started", "advanced": request.advanced}


@router.post("/api/cancel_persona")
async def cancel_persona():
    if not process_state["persona"]["running"]:
        return {"status": "idle"}

    process_state["persona"]["cancel"] = True
    process_state["persona"]["logs"].append("[CANCEL] Cancellation requested")

    proc = process_state["persona"].get("process")
    if proc and proc.poll() is None:
        try:
            _send_signal_tree(proc, signal.SIGTERM)
        except (ProcessLookupError, OSError) as e:
            logger.warning(f"Failed to terminate persona process cleanly: {e}")

    return {"status": "cancelling"}

@router.post("/api/save_voice_config")
async def save_voice_config(config_data: Dict[str, VoiceConfigItem]):
    def _save():
        # Hold the lock across the read-modify-write so this can't race a batch
        # review's concurrent speaker-rename remap of the same file.
        with file_lock(VOICE_CONFIG_PATH):
            current_config = {}
            if os.path.exists(VOICE_CONFIG_PATH):
                with open(VOICE_CONFIG_PATH, "r", encoding="utf-8") as f:
                    try:
                        current_config = json.load(f)
                    except (json.JSONDecodeError, ValueError) as e:
                        _warn_corrupted_json("voice config", VOICE_CONFIG_PATH, "overwriting with new data", e)

            # Update current config with new data
            for voice_name, config in config_data.items():
                # Convert Pydantic model to dict
                current_config[voice_name] = config.model_dump()

            atomic_json_write(current_config, VOICE_CONFIG_PATH)

    # Offload to a worker thread so file_lock's wait loop can't block the event loop.
    try:
        await asyncio.to_thread(_save)
    except TimeoutError:
        raise HTTPException(status_code=503, detail="Voice config is busy (locked by another operation); please try again.")

    return {"status": "saved"}


# --- Auto-suggest best LoRA voice per character -------------------------------

def _infer_lora_gender(model):
    """Best-effort gender for a LoRA candidate: explicit field, then name suffix,
    then description keywords, then mean f0 from voice_features."""
    g = (model.get("gender") or "").strip().lower()
    if g in ("male", "female"):
        return g
    name_id = f"{model.get('name', '')} {model.get('id', '')}".lower()
    if re.search(r"(_|\b)f(\b|_|\d|emale)", name_id):
        return "female"
    if re.search(r"(_|\b)m(\b|_|\d|ale)", name_id):
        return "male"
    desc = (model.get("description") or model.get("voice_profile") or "").lower()
    if any(w in desc for w in ("alto", "soprano", "mezzo", "feminine", "woman", "girl")):
        return "female"
    if any(w in desc for w in ("baritone", "tenor", "bass", "masculine", "man", "boy")):
        return "male"
    f0 = (model.get("voice_features") or {}).get("mean_f0")
    if isinstance(f0, (int, float)) and f0 > 0:
        return "female" if f0 >= 165 else "male"
    return "unknown"


def _infer_character_gender(text):
    """Rough gender guess for a character from persona/style/sample text via pronoun counts."""
    t = (text or "").lower()
    male = len(re.findall(r"\b(he|him|his|himself|man|men|boy|male|sir|mr|lord|king|father)\b", t))
    female = len(re.findall(r"\b(she|her|hers|herself|woman|women|girl|female|lady|mrs|ms|miss|queen|mother)\b", t))
    if male > female and male > 0:
        return "male"
    if female > male and female > 0:
        return "female"
    return "unknown"


AGE_GROUPS = ("child", "teen", "young_adult", "adult", "middle_aged", "elderly")


def _age_group_from_years(age):
    age = int(age)
    if age <= 12:
        return "child"
    if age <= 19:
        return "teen"
    if age <= 29:
        return "young_adult"
    if age <= 39:
        return "adult"
    if age <= 59:
        return "middle_aged"
    return "elderly"


def _infer_age_group(text):
    """Best-effort normalized apparent age from names, profiles, or dialogue."""
    value = (text or "").lower().replace("-", " ").replace("_", " ")
    numeric = re.search(r"\b(?:aged?\s+(\d{1,3})|(\d{1,3})\s+years?\s+old)\b", value)
    if not numeric and re.fullmatch(r"\s*\d{1,3}\s*", value):
        numeric = re.match(r"\s*(\d{1,3})", value)
    if numeric:
        years = next(group for group in numeric.groups() if group is not None)
        if 1 <= int(years) <= 120:
            return _age_group_from_years(years)
    decade = re.search(r"\b([2-8])0s\b", value)
    if decade:
        return _age_group_from_years(int(decade.group(1)) * 10 + 5)
    patterns = (
        ("child", r"\b(child|kid|little boy|little girl|preteen|under ?1[0-2])\b"),
        ("teen", r"\b(teen|teenage|adolescent)\b"),
        ("young_adult", r"\b(young adult|young man|young woman|twent(?:y|ies))\b"),
        ("middle_aged", r"\b(middle aged|middle age|forties|fifties)\b"),
        ("elderly", r"\b(elderly|old man|old woman|senior|sixties|seventies|eighties)\b"),
        ("adult", r"\b(adult|grown man|grown woman|thirties)\b"),
    )
    return next((group for group, pattern in patterns if re.search(pattern, value)), "unknown")


def _infer_lora_age(model):
    explicit = str(model.get("age_group") or model.get("age") or "").strip().lower().replace("-", "_").replace(" ", "_")
    if explicit in AGE_GROUPS:
        return explicit
    evidence = " ".join(str(model.get(k) or "") for k in ("age", "name", "id", "description", "voice_profile"))
    return _infer_age_group(evidence)


def _infer_character_traits(name, profile, lines):
    """Infer traits with evidence priority: label, then persona.

    DIALOGUE IS NOT EVIDENCE ABOUT ITS SPEAKER. `_infer_character_gender`
    counts gendered words, and the words a character SPEAKS describe whoever
    they are talking about. Measured on the live book: Subaru's own 412 lines
    contain 46 feminine tokens against 27 masculine - "she" x20, "her" x11,
    "girl" x10 - because he spends the book talking to and about Emilia, Felt
    and Satella. He was classified female. So were ROM and Reinhard; Emilia
    came out male. The signal is inverted for exactly the characters who speak
    most, which is the ones that matter.

    It was rated "low" confidence, which correctly kept it out of the HARD
    gender filter - but `get_voice_allocation` still applies a soft penalty of
    3 against candidates whose gender differs from the inferred one, so a wrong
    guess actively pushed correct-gender voices down the ranking.

    Dropping the source leaves gender "unknown" when only dialogue is
    available, and "unknown" is handled correctly everywhere: no hard filter,
    no soft penalty. Determining gender properly needs coreference - which
    pronoun refers to whom - and that is BookNLP's or the LLM's job, not a
    regex over the wrong text.
    """
    sources = (("character label", name, "high"),
               ("existing persona/style", profile, "medium"))
    result = {"gender": "unknown", "gender_confidence": "unknown",
              "age_group": "unknown", "age_confidence": "unknown", "trait_evidence": ""}
    evidence = []
    for source, text, confidence in sources:
        if result["gender"] == "unknown":
            gender = _infer_character_gender(text)
            if gender != "unknown":
                result.update(gender=gender, gender_confidence=confidence)
                evidence.append(f"{source}: {gender}")
        if result["age_group"] == "unknown":
            age = _infer_age_group(text)
            if age != "unknown":
                result.update(age_group=age, age_confidence=confidence)
                evidence.append(f"{source}: {age.replace('_', ' ')}")
    result["trait_evidence"] = "; ".join(evidence) or "No explicit gender or age evidence"
    result["local_trait_evidence"] = result["trait_evidence"]
    result["llm_trait_evidence"] = ""
    return result


def _age_distance(character_age, voice_age):
    if character_age == "unknown" or voice_age == "unknown":
        return 2
    return abs(AGE_GROUPS.index(character_age) - AGE_GROUPS.index(voice_age))


def _is_authoritative_confidence(confidence):
    return confidence in ("high", "medium")


def _is_stronger_authoritative_confidence(current, proposed):
    confidence_rank = {"unknown": 0, "low": 1, "medium": 2, "high": 3}
    return (_is_authoritative_confidence(proposed)
            and confidence_rank.get(proposed, 0) > confidence_rank.get(current, 0))


def _build_lora_candidates():
    """Downloaded built-in + user-trained adapters with normalized fields for matching."""
    candidates = []
    for m in _load_builtin_lora_manifest():
        if not m.get("downloaded", False):
            continue
        candidates.append({
            "adapter_id": m["id"],
            "name": m.get("name") or m["id"],
            "type": "builtin_lora",
            "gender": _infer_lora_gender(m),
            "age_group": _infer_lora_age(m),
            "description": m.get("description") or m.get("voice_profile") or "",
        })
    for m in _load_manifest(LORA_MODELS_MANIFEST):
        candidates.append({
            "adapter_id": m["id"],
            "name": m.get("name") or m["id"],
            "type": "lora",
            "gender": _infer_lora_gender(m),
            "age_group": _infer_lora_age(m),
            "description": m.get("description") or m.get("voice_profile") or "",
        })
    return candidates


def _select_representative_lines(lines: List[str], limit: int) -> List[str]:
    """Sample dialogue across the whole book rather than only its beginning."""
    if len(lines) <= limit:
        return lines
    if limit <= 1:
        return [lines[0]]
    indices = [round(i * (len(lines) - 1) / (limit - 1)) for i in range(limit)]
    return [lines[i] for i in dict.fromkeys(indices)]


def _rank_heuristic_candidates(profile: str, candidates: List[dict], preferred_gender=None,
                               preferred_age="unknown", filter_gender=True) -> List[str]:
    gender = preferred_gender if preferred_gender in ("male", "female") else _infer_character_gender(profile)
    pool = candidates
    if filter_gender and gender != "unknown":
        pool = [c for c in candidates if c.get("gender") == gender] or candidates
    words = set(re.findall(r"[a-z]{4,}", profile.lower()))
    ranked = sorted(pool, key=lambda c: (
        _age_distance(preferred_age, c.get("age_group", "unknown")),
        -len(words & set(re.findall(r"[a-z]{4,}", c.get("description", "").lower()))),
        c["adapter_id"]))
    return [c["adapter_id"] for c in ranked]


def get_voice_allocation(profile, candidates, initial_ranked, traits,
                         existing_adapter, usage, priority):
    """Pure compatibility/reuse decision; the caller owns usage mutation."""
    cand_by_id = {c["adapter_id"]: c for c in candidates}
    hard_gender = (traits["gender"] != "unknown"
                   and _is_authoritative_confidence(traits["gender_confidence"]))
    hard_age = _is_authoritative_confidence(traits["age_confidence"])
    gender_matches = [c for c in candidates if c.get("gender") == traits["gender"]]
    gender_fallback = hard_gender and not gender_matches
    ranked = [adapter_id for adapter_id in initial_ranked if adapter_id in cand_by_id]
    for adapter_id in _rank_heuristic_candidates(
            profile, candidates, traits["gender"],
            traits["age_group"] if hard_age else "unknown",
            filter_gender=False):
        if adapter_id not in ranked:
            ranked.append(adapter_id)
    rank_order = {adapter_id: index for index, adapter_id in enumerate(ranked)}

    def allocation_score(adapter_id):
        candidate = cand_by_id[adapter_id]
        candidate_gender = candidate.get("gender", "unknown")
        hard_gender_tier = 0
        soft_gender_penalty = 0
        if hard_gender:
            if candidate_gender != traits["gender"]:
                hard_gender_tier = 1 if candidate_gender == "unknown" else 2
        elif traits["gender"] != "unknown" and candidate_gender != traits["gender"]:
            soft_gender_penalty = 1 if candidate_gender == "unknown" else 3
        distance = _age_distance(traits["age_group"], candidate.get("age_group", "unknown"))
        age_penalty = distance * (100 if hard_age else 1)
        reuse_penalty = 100 if priority == "major" else 2
        compatibility_and_reuse = (
            rank_order[adapter_id] * 10 + age_penalty + soft_gender_penalty
            + usage.get(adapter_id, {}).get("character_count", 0) * reuse_penalty)
        return hard_gender_tier, compatibility_and_reuse, adapter_id

    ranked.sort(key=allocation_score)
    if existing_adapter in cand_by_id:
        chosen_id, is_new_identity = existing_adapter, False
    else:
        chosen_id = ranked[0]
        is_new_identity = True
    chosen = cand_by_id[chosen_id]
    existing_trait_mismatch = bool(existing_adapter and (
        (_is_authoritative_confidence(traits["gender_confidence"])
         and traits["gender"] != "unknown"
         and chosen.get("gender") not in (traits["gender"], "unknown"))
        or (_is_authoritative_confidence(traits["age_confidence"])
            and _age_distance(traits["age_group"], chosen.get("age_group", "unknown")) >= 3)))
    return chosen_id, ranked, is_new_identity, gender_fallback, existing_trait_mismatch


@router.post("/api/suggest_voices")
async def suggest_voices(request: SuggestVoicesRequest = SuggestVoicesRequest()):
    """Suggest the best-matching downloaded LoRA voice for each character based on
    the character's dialogue + persona, ranked by the configured LLM (heuristic fallback).

    Offloaded to threadpool via asyncio.to_thread to avoid blocking the event loop."""
    # Reserve the GPU slot for the duration of the (local-LLM) suggestion so it
    # can't run concurrently with TTS/review and trigger a VRAM OOM. Released in
    # finally since this is a synchronous request, not a run_process task.
    claim_gpu_task("voices")
    try:
        return await asyncio.to_thread(_suggest_voices_impl, request)
    finally:
        process_state["voices"]["running"] = False


def _suggest_voices_impl(request: SuggestVoicesRequest):
    # Sync implementation that makes a blocking LLM call and file I/O.
    # Called via asyncio.to_thread from the async endpoint above.
    if not os.path.exists(SCRIPT_PATH):
        raise HTTPException(status_code=400, detail="No script found. Generate a script first.")

    try:
        with open(SCRIPT_PATH, "r", encoding="utf-8") as f:
            script = json.load(f)
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(status_code=400, detail="Script is not valid JSON.")

    # Collect every per-character dialogue line so counts are accurate; sample
    # representative lines across the book only when building the prompt.
    samples = {}
    for entry in script:
        speaker = (entry.get("speaker") or entry.get("type") or "").strip()
        text = (entry.get("text") or "").strip()
        if not speaker or not text:
            continue
        lines = samples.setdefault(speaker, [])
        if text not in lines:
            lines.append(text)
    if not samples:
        return {"method": "none", "suggestions": {}, "message": "No characters found in script."}

    # Existing config (for persona descriptions/styles + only_unset filtering)
    voice_config = {}
    if os.path.exists(VOICE_CONFIG_PATH):
        try:
            with open(VOICE_CONFIG_PATH, "r", encoding="utf-8") as f:
                voice_config = json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            _warn_corrupted_json("voice config", VOICE_CONFIG_PATH, "treating as empty", e)
            voice_config = {}

    candidates = _build_lora_candidates()
    if not candidates:
        raise HTTPException(status_code=400, detail="No downloaded LoRA voices available. Download a built-in voice or train an adapter first.")

    line_limit = max(1, min(int(request.max_lines or 8), 30))
    book_id = get_active_book_id()
    lib = _load_voice_library()
    cast_name = (request.cast or "").strip() or None
    if cast_name and cast_name not in lib["casts"]:
        raise HTTPException(status_code=404, detail=f"Cast '{cast_name}' not found.")
    usage = get_cast_adapter_usage(lib, cast_name)
    line_counts = _script_line_counts()

    # Build profiles in importance order: narrator, then most dialogue lines.
    characters = {}
    ordered_names = sorted(samples, key=lambda n: (0 if _norm_name(n) == "narrator" else 1, -len(samples[n]), _norm_name(n)))
    for speaker in ordered_names:
        lines = samples[speaker]
        if request.only_unset:
            existing = voice_config.get(speaker, {})
            if voice_category(existing) == "lora" and existing.get("adapter_id"):
                continue
        cfg = voice_config.get(speaker, {})
        persona_bits = [cfg.get("description") or "", cfg.get("character_style") or "", cfg.get("default_style") or ""]
        profile = " ".join(b for b in persona_bits if b)
        traits = _infer_character_traits(speaker, profile, lines)
        count = line_counts.get(speaker, len(lines))
        try:
            member_key = get_cast_member_key(speaker, book_id)
        except ValueError:
            member_key = None
        characters[speaker] = {
            "profile": profile,
            "lines": _select_representative_lines(lines, line_limit),
            "line_count": count,
            "priority": "major" if _norm_name(speaker) == "narrator" or count >= CAST_MAJOR_LINE_THRESHOLD else "minor",
            "member_key": member_key,
            **traits,
        }

    if not characters:
        return {"method": "none", "suggestions": {}, "message": "No characters to suggest (all already set)."}

    # When only filling unset roles, already-configured current-book roles are
    # fixed assignments and must contribute to reuse pressure unless the same
    # identity is already represented in the selected cast.
    if request.only_unset:
        cast_members = (lib.get("casts", {}).get(cast_name, {}).get("members", {})
                        if cast_name else {})
        for name, cfg in voice_config.items():
            adapter_id = (cfg or {}).get("adapter_id")
            if not adapter_id or voice_category(cfg) != "lora":
                continue
            try:
                key = get_cast_member_key(name, book_id)
            except ValueError:
                continue
            if key in cast_members:
                continue
            item = usage.setdefault(adapter_id, {"character_count": 0, "total_lines": 0, "characters": []})
            item["character_count"] += 1
            item["total_lines"] += line_counts.get(name, 0)
            item["characters"].append(name)

    cand_by_id = {c["adapter_id"]: c for c in candidates}
    suggestions = {}
    rankings = {}
    style_by_name = {}
    reason_by_name = {}
    method = "heuristic"
    llm_warning = None

    # --- Try LLM ranking first ---
    llm_ok = False
    try:
        # don't let a stuck model hang the worker thread forever
        client, model_name = _make_llm_client(timeout=120)

        voice_catalog = "\n".join(
            f'- id="{c["adapter_id"]}" | name="{c["name"][:50]}" | gender={c.get("gender", "unknown")} | age={c.get("age_group", "unknown")} | series_use={usage.get(c["adapter_id"], {}).get("character_count", 0)} | description: {(c["description"] or "(none)")[:80]}'
            for c in candidates
        )
        system_prompt = (
            "You are a casting director matching narrated audiobook characters to available LoRA TTS voices. "
            "For each character, rank up to three fitting voice ids and write concise TTS delivery guidance based only on the book text. "
            "The style should describe cadence, energy, formality, confidence, and supported emotion; do not invent biography or accent. "
            "Infer gender and broad apparent age only when supported by the supplied book evidence. "
            "Known character gender must match voice gender; prefer the closest available age group. "
            "Only use provided voice ids. Return every requested character in the structured response."
        )
        casting_schema = {
            "name": "audiobook_casting",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "characters": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "ranked_adapter_ids": {
                                    "type": "array", "items": {"type": "string"},
                                    "minItems": 1, "maxItems": 3,
                                },
                                "character_style": {"type": "string"},
                                "reason": {"type": "string"},
                                "character_gender": {"type": "string", "enum": ["male", "female", "unknown"]},
                                "age_group": {"type": "string", "enum": ["child", "teen", "young_adult", "adult", "middle_aged", "elderly", "unknown"]},
                                "trait_evidence": {"type": "string"},
                                "trait_confidence": {"type": "string", "enum": ["high", "medium", "low", "unknown"]},
                            },
                            "required": ["name", "ranked_adapter_ids", "character_style", "reason", "character_gender", "age_group", "trait_evidence", "trait_confidence"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["characters"],
                "additionalProperties": False,
            },
        }
        character_items = list(characters.items())
        full_cfg = load_app_config(CONFIG_PATH)
        llm_cfg = full_cfg.get("llm") or {}
        status = get_current_status(
            full_cfg.get("llm_mode", "local"), llm_cfg.get("base_url", ""),
            model_name, (full_cfg.get("llm_remote_ssh") or "").strip(),
            use_cache=True)
        for start in range(0, len(character_items), 2):
            batch = character_items[start:start + 2]
            char_block = "\n\n".join(
                f'CHARACTER: {name}\nLines: {info["line_count"]} ({info["priority"]})\nCurrent trait estimate: gender={info["gender"]}, age={info["age_group"]}\nPersona/style: {(info["profile"] or "(none)")[:200]}\nSample lines:\n'
                + "\n".join(f'  - "{ln[:140]}"' for ln in info["lines"])
                for name, info in batch
            )
            user_prompt = f"AVAILABLE VOICES:\n{voice_catalog}\n\nCHARACTERS:\n{char_block}"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            effective_max = get_effective_max_tokens(
                2600, status.get("context_length"), messages, hard_max=12000)
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format={"type": "json_schema", "json_schema": casting_schema},
                temperature=0.3,
                max_tokens=effective_max,
                timeout=120,
            )
            raw = response.choices[0].message.content or ""
            parsed = extract_json_object(raw)
            if parsed is None:
                finish_reason = response.choices[0].finish_reason
                logger.warning("Unparseable casting response (%s) preview: %s", finish_reason, raw[:500])
                raise ValueError(f"Could not parse a JSON object from casting batch ({len(raw)} chars)")
            parsed_items = parsed.get("characters", []) if isinstance(parsed, dict) else []
            parsed_by_name = {
                item.get("name"): item for item in parsed_items
                if isinstance(item, dict) and isinstance(item.get("name"), str)
            }
            for name, _info in batch:
                pick = parsed_by_name.get(name)
                if isinstance(pick, dict):
                    ranked = pick.get("ranked_adapter_ids") or ([pick.get("adapter_id")] if pick.get("adapter_id") else [])
                    rankings[name] = list(dict.fromkeys(i for i in ranked if i in cand_by_id))
                    style_by_name[name] = (pick.get("character_style") or "").strip()[:500]
                    reason_by_name[name] = (pick.get("reason") or "").strip()[:240]
                    info = characters[name]
                    llm_confidence = pick.get("trait_confidence", "unknown")
                    llm_gender = pick.get("character_gender")
                    llm_age = pick.get("age_group")
                    accepted_traits = []
                    rejected_conflict = False
                    if (llm_gender in ("male", "female")
                            and _is_stronger_authoritative_confidence(
                                info["gender_confidence"], llm_confidence)):
                        info["gender"] = llm_gender
                        info["gender_confidence"] = llm_confidence
                        accepted_traits.append(f"gender={llm_gender}")
                    elif llm_gender in ("male", "female") and llm_gender != info["gender"]:
                        rejected_conflict = True
                    if (llm_age in AGE_GROUPS
                            and _is_stronger_authoritative_confidence(
                                info["age_confidence"], llm_confidence)):
                        info["age_group"] = llm_age
                        info["age_confidence"] = llm_confidence
                        accepted_traits.append(f"age={llm_age.replace('_', ' ')}")
                    elif llm_age in AGE_GROUPS and llm_age != info["age_group"]:
                        rejected_conflict = True
                    info["llm_trait_evidence"] = (pick.get("trait_evidence") or "")[:300]
                    if accepted_traits and info["llm_trait_evidence"]:
                        llm_evidence = ("LM accepted " + ", ".join(accepted_traits)
                                        if rejected_conflict else f"LM: {info['llm_trait_evidence']}")
                        info["trait_evidence"] = (
                            f"Local: {info['local_trait_evidence']}; {llm_evidence}")[:300]
    except LLMConfigError as e:
        # Config issue (e.g. base_url rejected by _validate_local_llm_base_url) -
        # surface to the UI instead of silently falling back to heuristic.
        llm_warning = str(e)
    except Exception as e:
        logger.warning(f"LLM voice suggestion failed, falling back to heuristic: {e}")

    if rankings:
        llm_ok = True
        method = "llm"

    # Fill missing rankings/styles deterministically, then allocate in priority
    # order while updating reuse counts after every new distinct character.
    for name, info in characters.items():
        profile_text = " ".join([name, info["profile"]] + info["lines"])
        if not rankings.get(name):
            rankings[name] = _rank_heuristic_candidates(
                profile_text, candidates, info["gender"],
                info["age_group"] if _is_authoritative_confidence(info["age_confidence"]) else "unknown",
                filter_gender=_is_authoritative_confidence(info["gender_confidence"]))
        if not style_by_name.get(name):
            style_by_name[name] = info["profile"] or "Natural delivery matching the character's dialogue and role in this book."
        if not reason_by_name.get(name):
            reason_by_name[name] = "Deterministic compatibility and series-diversity ranking"

        existing_member = None
        if cast_name and info["member_key"]:
            existing_member = (lib["casts"][cast_name].get("members", {}).get(info["member_key"])
                               or lib.get("shared", {}).get(info["member_key"]))
        existing_adapter = ((existing_member or {}).get("config") or {}).get("adapter_id")
        (chosen_id, ranked, is_new_identity, gender_fallback,
         existing_trait_mismatch) = get_voice_allocation(
            profile_text, candidates, rankings[name], info, existing_adapter,
            usage, info["priority"])
        before = usage.get(chosen_id, {}).get("character_count", 0)
        if is_new_identity:
            usage.setdefault(chosen_id, {"character_count": 0, "total_lines": 0, "characters": []})
            usage[chosen_id]["character_count"] += 1
            usage[chosen_id]["total_lines"] += info["line_count"]
            usage[chosen_id]["characters"].append(name)
        chosen = cand_by_id[chosen_id]
        suggestions[name] = {
            "adapter_id": chosen_id, "adapter_name": chosen["name"], "type": chosen["type"],
            "character_style": style_by_name[name], "reason": reason_by_name[name],
            "line_count": info["line_count"], "priority": info["priority"], "book_id": book_id,
            "cast_member_key": info["member_key"], "reuse_count_before": before,
            "reuse_count_after": before + (1 if is_new_identity else 0),
            "reused": before > 0 and is_new_identity,
            "forced_reuse": info["priority"] == "major" and before > 0 and all(usage.get(i, {}).get("character_count", 0) > 0 for i in ranked),
            "character_gender": info["gender"], "character_age_group": info["age_group"],
            "voice_gender": chosen.get("gender", "unknown"),
            "voice_age_group": chosen.get("age_group", "unknown"),
            "trait_evidence": info["trait_evidence"],
            "local_trait_evidence": info["local_trait_evidence"],
            "llm_trait_evidence": info["llm_trait_evidence"],
            "gender_confidence": info["gender_confidence"],
            "age_confidence": info["age_confidence"],
            "gender_fallback": gender_fallback,
            "existing_trait_mismatch": existing_trait_mismatch,
        }

    if not llm_ok and suggestions:
        method = "heuristic"

    return {"method": method, "suggestions": suggestions, "candidate_count": len(candidates),
            "adapter_usage": usage, "book_id": book_id, "cast": cast_name,
            "major_line_threshold": CAST_MAJOR_LINE_THRESHOLD, "llm_warning": llm_warning}




def _apply_voice_suggestions(suggestions: Dict[str, dict], cast_name: Optional[str]) -> dict:
    candidates = {c["adapter_id"]: c for c in _build_lora_candidates()}
    counts = _script_line_counts()
    book_id = get_active_book_id()
    if cast_name and not book_id:
        raise HTTPException(status_code=400, detail="Active book identity is required to save suggestions to a cast.")

    with file_lock(VOICE_LIBRARY_PATH), file_lock(VOICE_CONFIG_PATH):
        voice_config = safe_load_json(VOICE_CONFIG_PATH, default={})
        lib = _load_voice_library()
        if cast_name and cast_name not in lib["casts"]:
            raise HTTPException(status_code=404, detail=f"Cast '{cast_name}' not found.")
        usage = get_cast_adapter_usage(lib, cast_name)
        applied = []
        for character, suggestion in suggestions.items():
            if character not in counts:
                continue
            suggestion_book_id = secure_filename(suggestion.get("book_id") or "")
            if suggestion_book_id != secure_filename(book_id or ""):
                raise HTTPException(status_code=409, detail=(
                    f"Suggestion for '{character}' belongs to a different book. Generate suggestions again."))
            adapter_id = suggestion.get("adapter_id")
            candidate = candidates.get(adapter_id)
            if not candidate:
                raise HTTPException(status_code=400, detail=f"Unknown or unavailable LoRA adapter: {adapter_id}")
            style = (suggestion.get("character_style") or "").strip()[:500]
            cfg = dict(voice_config.get(character) or {})
            cfg.update({
                "type": candidate["type"], "adapter_id": adapter_id,
                "adapter_path": (f"builtin_lora/{adapter_id}" if candidate["type"] == "builtin_lora"
                                 else f"lora_models/{adapter_id}"),
                "character_style": style, "seed": "-1",
                **get_trait_assignment_metadata(suggestion),
            })
            voice_config[character] = cfg

            if cast_name:
                try:
                    key = get_cast_member_key(character, book_id)
                except ValueError as e:
                    raise HTTPException(status_code=400, detail=str(e))
                members = get_cast_storage_pool(lib, cast_name, character)
                casting = {
                    "priority": suggestion.get("priority"),
                    "suggestion_reason": (suggestion.get("reason") or "")[:240],
                    "reuse_count_when_assigned": usage.get(adapter_id, {}).get("character_count", 0),
                    **get_trait_assignment_metadata(suggestion),
                }
                members[key] = _make_library_entry(
                    character, cfg, counts[character], book_id, casting, members.get(key))
                usage = get_cast_adapter_usage(lib, cast_name)
            applied.append(character)

        if cast_name:
            atomic_json_write_pair(voice_config, VOICE_CONFIG_PATH,
                                   lib, VOICE_LIBRARY_PATH)
        else:
            atomic_json_write(voice_config, VOICE_CONFIG_PATH)
    return {"applied": applied, "count": len(applied), "cast": cast_name,
            "book_id": book_id, "adapter_usage": get_cast_adapter_usage(lib, cast_name)}


@router.post("/api/suggest_voices/apply")
async def apply_voice_suggestion(request: VoiceSuggestionApplyRequest):
    return await asyncio.to_thread(
        _apply_voice_suggestions, {request.character: request.suggestion},
        (request.cast or "").strip() or None)

@router.post("/api/suggest_voices/apply_bulk")
async def apply_voice_suggestions_bulk(request: VoiceSuggestionApplyBulkRequest):
    return await asyncio.to_thread(
        _apply_voice_suggestions, request.suggestions,
        (request.cast or "").strip() or None)
