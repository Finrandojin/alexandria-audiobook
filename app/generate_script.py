"""Stage 2b: turn a book into an annotated audiobook script.

The LLM is an ANALYTICAL CLASSIFIER here, never a generator of book text.
Code tokenizes each chunk into spans (span_tokenizer), sends the LLM span ids
plus their text, and receives back ONLY labels --
``{"id", "speaker", "role", "instruct"}``. Code then reassembles the script
verbatim from the source string by span offsets. Consequences:

* Truncated or malformed LLM output costs LABELS, never PROSE. Any span the
  LLM did not label falls back to NARRATOR with its text intact.
* A chunk's entries always concatenate byte-for-byte back to the chunk. That
  is asserted in code (`_assert_chunk_verbatim`) -- a failure is a bug, not a
  warning.
* Degradation is never silent: it is counted, summarised, and surfaced as
  exit code 3 (see the exit site in main()).
* A misspelled schema KEY is recovered (_recover_label_keys), a misspelled
  speaker NAME is not (that is speaker_canon's job, under much tighter rules).
  Keys come from this module's own four-word vocabulary; names come from the
  book, where two near-identical spellings are routinely two people. "text" is
  never a recovery target, so key recovery can never admit model prose.
"""

import os
import sys
import json
import re
import argparse
from openai import OpenAI
from rapidfuzz.distance import Levenshtein
from default_prompts import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT
from span_tokenizer import tokenize, validate_spans, QUOTED, UNQUOTED
from speaker_canon import (
    UNATTESTED,
    UNVERIFIABLE,
    attest_speaker,
    canonicalize,
    contradicts_attribution,
    near_spellings,
    remember_in_roster,
    repair_speaker,
    resolve_against_roster,
    roster_key,
    source_word_index,
)

# Cap for single-speaker mode: entries at this size pass through
# group_into_chunks (MAX_CHUNK_CHARS=500) as-is without further splitting.
SINGLE_SPEAKER_MAX_CHARS = 500

# Canonical narrator label. Compared with `!= "NARRATOR"` in this file and in
# review_script.py, so casing is load-bearing.
NARRATOR = "NARRATOR"

# Fallback voice directions when a span carries no usable "instruct".
DEFAULT_NARRATOR_INSTRUCT = "Neutral, even narration."
DEFAULT_CHARACTER_INSTRUCT = "Natural, in-character delivery."

# Exit code used when the script was written but one or more chunks degraded.
EXIT_DEGRADED = 3

# Schema marker carried by BOTH halves of default_prompts.txt. A prompt saved
# in config.json before this stage existed asks the LLM to rewrite the book
# into speaker/text/instruct objects, which the span classifier cannot use --
# every span would fall back to NARRATOR. Requiring the marker makes that
# mismatch loud and self-healing instead of silent.
PROMPT_SCHEMA_MARKER = "span-labels-v1"

# How much of a previous entry's text to show as continuity context. Context is
# input, not output -- a snippet is enough and keeps the prompt cheap.
CONTEXT_SNIPPET_CHARS = 120

# Cap on how many character names build_context()'s roster block may list.
# Previously uncapped: a real 976-chunk run reached 577 names, a ~7,400-char
# block (~2,000 tokens) growing monotonically for the whole book and crowding
# out the span payload it exists to support.
#
# This is PROMPT HYGIENE, not a fix for speaker-name drift -- verified: the
# drift observed in that run happened at chunks 379/380/430/442, while the
# roster block was still well under any cap, so a smaller roster would not
# have prevented it (speaker_canon's roster resolution is what handles that).
# The justification is simply that an unbounded, monotonically growing block
# is a defect on any long book on any server.
#
# Policy: keep the MOST RECENTLY SPOKEN names (a backwards pass over the
# previous entries that stops at the cap), then present them alphabetically.
# Recency, not frequency: the roster's job is spelling continuity for the
# current scene's cast, and a character who last spoke 400 chunks ago is not
# the one about to be misspelled here; frequency would instead favour
# book-wide protagonists, the names least at risk of drift. Deterministic
# given the same entries. Default 50: larger than any realistic single-scene
# cast, ~650 chars. Override via generation.max_context_roster_names.
MAX_CONTEXT_ROSTER_NAMES = 50

# --- Silent prompt-truncation tripwire -------------------------------------
#
# A local server whose serving context window is smaller than the prompt does
# not error: it silently drops the overflow and answers from what is left. A
# real 976-chunk run spent its entire second half in this state (prompt_tokens
# pinned at exactly 2050 for 640 of 1,294 records -- Ollama's default
# num_ctx=2048), and the run summary was green throughout: the retry path
# quietly papered over it, firing 318 extra calls. Nothing detected it.
#
# Detection compares the reported prompt_tokens against a chars/4 estimate of
# the prompt actually sent. Warning only -- never a hard failure, never a
# change in control flow -- because the estimate is a heuristic.
PROMPT_CHARS_PER_TOKEN = 4

# Warn when the estimate exceeds reported prompt_tokens by more than this
# factor. The false-alarm direction is one-sided and understood: text with a
# HIGH token density (CJK source text tokenizes to far more than one token per
# 4 chars) pushes prompt_tokens UP relative to the estimate and therefore can
# never trip this. Only unusually LOW-density text could, so the factor leaves
# a 60% margin below the English baseline before saying anything.
SILENT_TRUNCATION_RATIO = 1.6

# Flatline detector thresholds (see detect_flatlined_prompt_tokens). The
# per-call ratio check above only fires once the prompt is far past the cap;
# a prompt sitting just above the window is truncated just as silently but
# looks unremarkable per call. Across a run it does not: prompt_tokens stops
# tracking prompt size and pins to one value. That is the production
# signature, and it is nearly free to check.
_FLATLINE_MIN_SAMPLES = 20
_FLATLINE_SHARE = 0.25
_FLATLINE_CHAR_SPREAD = 0.25


def estimate_prompt_tokens(text):
    """Rough token count for an assembled prompt. Deliberately crude."""
    return len(text or "") // PROMPT_CHARS_PER_TOKEN


def looks_silently_truncated(prompt_chars, prompt_tokens):
    """True when reported ``prompt_tokens`` is implausibly low for a prompt of
    ``prompt_chars`` characters, i.e. the server likely truncated it."""
    if not prompt_tokens or prompt_tokens <= 0 or not prompt_chars:
        return False
    return prompt_chars // PROMPT_CHARS_PER_TOKEN > SILENT_TRUNCATION_RATIO * prompt_tokens


def detect_flatlined_prompt_tokens(samples):
    """Spot a prompt_tokens flatline across a run. Returns ``(value, count,
    total)`` when one dominant value covers a large share of calls despite the
    prompts themselves varying substantially in size, else ``None``.

    ``samples`` is an iterable of ``(prompt_chars, prompt_tokens)`` pairs.
    """
    pairs = [(c, t) for c, t in samples if t and c]
    if len(pairs) < _FLATLINE_MIN_SAMPLES:
        return None

    counts = {}
    for _, tokens in pairs:
        counts[tokens] = counts.get(tokens, 0) + 1
    value, count = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))
    if count < _FLATLINE_SHARE * len(pairs):
        return None

    # If the prompts that produced that value were all the same size, a
    # repeated token count is simply correct, not evidence of a cap.
    sizes = [c for c, t in pairs if t == value]
    if max(sizes) - min(sizes) < _FLATLINE_CHAR_SPREAD * max(sizes):
        return None

    return value, count, len(pairs)


def _print_silent_truncation_warning(chunk_num, total_chunks, prompt_chars, prompt_tokens):
    """Loud, actionable warning. Diagnostic only; control flow is unchanged."""
    print(f"  {'!' * 60}")
    print(f"  PROMPT LIKELY TRUNCATED BY THE SERVER on chunk {chunk_num}/{total_chunks}: "
          f"sent {prompt_chars} chars (~{prompt_chars // PROMPT_CHARS_PER_TOKEN} "
          f"est. tokens) but the server reported prompt_tokens={prompt_tokens}.")
    print("  The model is being asked to classify spans it was never shown, so labels")
    print("  degrade and retries fire needlessly. Raise the SERVER's context window:")
    print("    - Ollama: OLLAMA_CONTEXT_LENGTH=8192 (env), or set generation.num_ctx")
    print("      in config.json, or bake num_ctx into the model's Modelfile")
    print("    - llama.cpp / vLLM: raise -c / --max-model-len")
    print("  Alternatively lower generation.chunk_size or "
          "generation.max_context_roster_names.")
    print(f"  {'!' * 60}")


# Placeholder speaker labels the model invents when it cannot identify a
# speaker, despite the prompt telling it to use NARRATOR. A production run
# produced 307 such entries ("SPEAKER 1", "SPEAKER 2", "VOICE 3", ...).
#
# The pattern is deliberately NARROW and language-neutral: one token followed
# by a bare number, which is what an enumerated placeholder always looks like
# in any language. It carries no word list, because this pipeline processes
# translated text and an English stoplist would be both wrong and unbounded --
# the lexical cases (SOMEONE, UNKNOWN, bare pronouns) are handled prompt-side
# in default_prompts.txt instead. A real character name ending in a bare
# number is vanishingly rare; such a label falls back to NARRATOR, which
# preserves the prose and costs only a voice assignment.
_PLACEHOLDER_LABEL_RE = re.compile(r'^\S+\s*\d+$')


def is_placeholder_speaker(name):
    """True for enumerated placeholder labels such as "SPEAKER 1"/"VOICE3"."""
    return bool(name) and bool(_PLACEHOLDER_LABEL_RE.match(name.strip()))


# Where raw LLM responses are logged. Production default; the test suite points
# ALEXANDRIA_LLM_LOG_DIR at a tempdir so fixture responses never pollute the
# real forensic log (it had accumulated 1200+ test records).
LLM_LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
LLM_LOG_DIR_ENV = "ALEXANDRIA_LLM_LOG_DIR"


def llm_log_dir():
    """Directory for llm_responses.log. Read per call so an override set after
    import (e.g. by a test harness) still takes effect."""
    return os.environ.get(LLM_LOG_DIR_ENV) or LLM_LOG_DIR

def strip_thinking_tags(text):
    """Remove reasoning-model thinking blocks from a raw response.

    Factored out of clean_json_string so the regex salvagers can reuse the
    exact same stripping. Reasoning text is full of id-like JSON fragments
    ("...span {"id": 12...") that a bare regex salvage happily mistakes for
    labels -- one live run salvaged 96 phantom ids out of a model's <think>
    block. Strip first, then salvage.
    """
    if not text:
        return text
    # Remove thinking tags (various formats used by different models)
    # GLM, DeepSeek, Qwen, etc. use different thinking tag formats
    text = re.sub(r'<think>[\s\S]*?</think>', '', text)
    text = re.sub(r'<thinking>[\s\S]*?</thinking>', '', text)
    text = re.sub(r'<reflection>[\s\S]*?</reflection>', '', text)
    text = re.sub(r'<reasoning>[\s\S]*?</reasoning>', '', text)
    # Handle unclosed thinking tags (model started thinking but didn't close)
    text = re.sub(r'<think>[\s\S]*$', '', text)
    text = re.sub(r'<thinking>[\s\S]*$', '', text)
    return text


def _balanced_array_end(text, start):
    """Index just past the ``]`` closing the array opened at ``start``.

    String-aware (brackets inside JSON strings do not count). Returns -1 when
    no balanced close exists, i.e. the array was truncated.
    """
    bracket_count = 0
    in_string = False
    escape_next = False

    for i, char in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if char == '\\':
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == '[':
            bracket_count += 1
        elif char == ']':
            bracket_count -= 1
            if bracket_count == 0:
                return i + 1
    return -1


def _escape_control_chars(json_text):
    """Escape literal newlines/tabs inside JSON string values (common LLM bug)."""
    def fix_control_chars(match):
        s = match.group(0)
        s = s.replace('\n', '\\n')
        s = s.replace('\r', '\\r')
        s = s.replace('\t', '\\t')
        return s

    return re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', fix_control_chars, json_text)


def clean_json_string(text):
    """Extract the JSON array from an LLM response.

    Tries EVERY ``[`` in the response, in order, and returns the first whose
    balanced slice actually parses as a list -- not merely the first ``[``
    character. A model that prefaces its array with prose containing brackets
    ("Sure! Here are the labels [see list below]:") used to hand back
    ``'[see list below]'``, a truthy string that then displaced the raw text
    in extract_labels' salvage step and cost the chunk EVERY label, even
    though the real array sat intact a few characters later.

    Falls back to the first syntactically-balanced candidate (or a truncated
    tail repaired with a closing bracket) when none of them parse, so callers
    that do their own repair -- review_script.py drives this same function --
    still receive what they used to.
    """
    text = strip_thinking_tags(text)

    # Remove markdown code blocks
    if "```" in text:
        # Find content between ```json and ``` or just ``` and ```
        match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if match:
            text = match.group(1).strip()

    fallback = None
    search_from = 0

    while True:
        start = text.find('[', search_from)
        if start == -1:
            break
        search_from = start + 1

        end = _balanced_array_end(text, start)
        if end == -1:
            # Truncated: repair by closing after the last complete object.
            last_complete = text.rfind('},')
            if last_complete > start and fallback is None:
                fallback = text[start:last_complete + 1] + ']'
            continue

        candidate = _escape_control_chars(text[start:end])
        if repair_json_array(candidate, quiet=True):
            return candidate
        if fallback is None:
            fallback = candidate

    return fallback


def repair_json_array(json_text, quiet=False):
    """Attempt to repair common JSON array issues from LLM output.

    ``quiet`` suppresses the dropped-entry warning. clean_json_string() uses
    this function to PROBE candidate slices, and a probe that rejects a
    bracketed fragment of prose must not report it as damage to the model's
    real array.
    """
    if not json_text:
        return None

    def _filter_entries(lst):
        """Keep only dict entries; LLMs sometimes emit bare strings in the array."""
        filtered = [e for e in lst if isinstance(e, dict)]
        if len(filtered) < len(lst) and not quiet:
            print(f"  Warning: Dropped {len(lst) - len(filtered)} non-object entries from LLM JSON array")
        return filtered if filtered else None

    # Try parsing as-is first
    try:
        result = json.loads(json_text)
        if isinstance(result, list):
            return _filter_entries(result)
    except json.JSONDecodeError:
        pass

    # Fix 1: Add missing commas between objects (}\s*{" -> },\n{")
    fixed = re.sub(r'\}\s*\{', '},\n{', json_text)
    try:
        result = json.loads(fixed)
        if isinstance(result, list):
            return _filter_entries(result)
    except json.JSONDecodeError:
        pass

    # Fix 2: Remove trailing commas before ]
    fixed = re.sub(r',\s*\]', ']', fixed)
    try:
        result = json.loads(fixed)
        if isinstance(result, list):
            return _filter_entries(result)
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
                return _filter_entries(result)
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


# ---------------------------------------------------------------------------
# Label schema: {"id", "speaker", "role", "instruct"}
#
# salvage_json_entries() above is the OLD speaker/text/instruct schema. It is
# still exported and still used by review_script.py, so it is left untouched.
# Everything below is the label schema used by the span classifier.
# ---------------------------------------------------------------------------

_LABEL_OBJECT_RE = re.compile(r'\{[^{}]*\}')
_LABEL_ID_RE = re.compile(r'"id"\s*:\s*"?(\d+)"?')
_LABEL_STRING_FIELD_RE = {
    "speaker": re.compile(r'"speaker"\s*:\s*"((?:[^"\\]|\\.)*)"'),
    "role": re.compile(r'"role"\s*:\s*"((?:[^"\\]|\\.)*)"'),
    "instruct": re.compile(r'"instruct"\s*:\s*"((?:[^"\\]|\\.)*)"'),
}

# Bareword fallback for models that emit complete-but-JSON-invalid output with
# UNQUOTED values: {"id": 1, "speaker": KIT, "role": dialogue, "instruct": "..."}.
# Observed live on 2 of 281 chunks (90 bareword values across them, all with
# finish_reason=stop -- not truncation). Without this the quoted pattern simply
# failed to match, the key was dropped in silence, and the whole chunk collapsed
# into one NARRATOR entry, costing 53 dialogue labels their voice.
#
# Only "speaker" and "role" get the fallback. They are the two fields that decide
# the cast, and in the logged responses they are exactly the two that appeared
# unquoted; "instruct" was always quoted, and it is free prose whose commas and
# braces make an unterminated bareword scan unsafe for no fidelity gain.
#
# The value stops at , } ] or a newline so it cannot swallow the following key,
# and must not begin with a quote so a well-formed value never takes this path.
_LABEL_BAREWORD_FIELD_RE = {
    "speaker": re.compile(r'"speaker"\s*:\s*([^"\s,}\]\n][^,}\]\n]*)'),
    "role": re.compile(r'"role"\s*:\s*([^"\s,}\]\n][^,}\]\n]*)'),
}

# JSON literals are not names. A bareword `null` means the model declined to
# answer, not that the character is called "Null".
_BAREWORD_NON_VALUES = {"null", "none", "nil", "undefined", "true", "false"}


def _unescape_json_string(raw):
    """Decode a JSON string body (the bit between the quotes)."""
    try:
        return json.loads('"' + raw + '"')
    except (json.JSONDecodeError, ValueError):
        return raw.replace('\\"', '"').replace('\\n', '\n').replace('\\\\', '\\')


# The entire label schema. Anything else the model sends is ignored on read;
# resolve_span_labels counts it so the discard is visible in the run log.
_LABEL_SCHEMA_KEYS = frozenset(("id", "speaker", "role", "instruct"))

# Keys a misspelling may be recovered onto. "text" is deliberately absent: under
# contract 1 it is never read, so no recovery path can resurrect model-supplied
# prose no matter what the model calls the field.
_RECOVERABLE_LABEL_KEYS = ("id", "speaker", "role", "instruct")

# Distance budget per target key: one edit per four characters, so the budget
# grows with the room a key has to be misspelled in. "instruct" gets 2, "role"
# and "speaker" get 1, "id" gets 0 -- "in" must never be recovered as "id".
_KEY_EDIT_BUDGET = {key: len(key) // 4 for key in _RECOVERABLE_LABEL_KEYS}

_NON_ALNUM_RE = re.compile(r"[^0-9a-z]+")


def _normalize_key(key):
    """Fold a key for comparison: case, spacing and punctuation are noise."""
    return _NON_ALNUM_RE.sub("", key.lower())


def _recover_label_keys(labels):
    """Fold misspelled schema keys back onto the key they can only have meant.

    This is NOT the banned fuzzy merging of speaker names. Names come from the
    book and two near-identical ones are routinely two different people
    (JON/JOHN); schema keys come from a four-word vocabulary this code publishes
    in its own prompt, so a near miss has exactly one possible intended meaning.
    The rule mirrors ``repair_speaker``: a bounded, exact edit distance (never a
    similarity ratio), and a uniqueness guard -- a key near two schema keys, or
    near none, is left alone for the unknown-key report in resolve_span_labels.

    Measured cost of not doing this: in one 281-chunk run, 31 labels across 13
    spellings ("instructor", "instruc", "in instruct", "instituct", ...) each
    silently lost their delivery direction.
    """
    recovered = {}
    for label in labels:
        if not isinstance(label, dict):
            continue
        for key in [k for k in label if isinstance(k, str) and k not in _LABEL_SCHEMA_KEYS]:
            normalized = _normalize_key(key)
            if not normalized:
                continue
            near = [
                target for target in _RECOVERABLE_LABEL_KEYS
                if Levenshtein.distance(
                    normalized, target, score_cutoff=_KEY_EDIT_BUDGET[target]
                ) <= _KEY_EDIT_BUDGET[target]
            ]
            if len(near) != 1 or near[0] in label:
                continue
            label[near[0]] = label.pop(key)
            recovered[(key, near[0])] = recovered.get((key, near[0]), 0) + 1

    # Loud, not silent -- same reason as the salvaged-bareword print above.
    for (key, target), count in sorted(recovered.items()):
        print(f'  Recovered misspelled key "{key}" as "{target}" on '
              f"{count} label(s)")
    return labels


def _label_id(label):
    """Return a label's span id as an int, or None if it has no usable id."""
    if not isinstance(label, dict):
        return None
    value = label.get("id")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return None


def _extract_label_object(raw):
    """Pull one label dict out of a (possibly truncated) JSON object fragment."""
    id_match = _LABEL_ID_RE.search(raw)
    if not id_match:
        return None
    label = {"id": int(id_match.group(1))}
    for key, pattern in _LABEL_STRING_FIELD_RE.items():
        match = pattern.search(raw)
        if match:
            label[key] = _unescape_json_string(match.group(1))
            continue
        # Quoted form wins; only an unquoted value reaches the fallback. The
        # bareword is NOT run through _unescape_json_string, which assumes a
        # JSON string body and would mangle a value containing a backslash.
        bareword = _LABEL_BAREWORD_FIELD_RE.get(key)
        if bareword is None:
            continue
        match = bareword.search(raw)
        if match:
            value = match.group(1).strip()
            if value and value.lower() not in _BAREWORD_NON_VALUES:
                label[key] = value
                # Loud, not silent: this is how a recurring formatting failure
                # stays visible instead of quietly costing labels again.
                print(f'  Salvaged unquoted "{key}" value {value!r} from a '
                      "malformed label object")
    return label


def salvage_label_entries(json_text):
    """Regex-salvage label objects from malformed / truncated classifier output.

    Field-order agnostic, and recovers a final object whose closing brace was
    cut off by the token limit. Returns a list of label dicts, or None.
    """
    if not json_text:
        return None

    # Reasoning blocks are pure noise here and their id-like fragments would be
    # salvaged as phantom labels. Strip before any regex touches the text.
    json_text = strip_thinking_tags(json_text)
    if not json_text:
        return None

    labels = []
    for raw in _LABEL_OBJECT_RE.findall(json_text):
        label = _extract_label_object(raw)
        if label is not None:
            labels.append(label)

    # A truncated tail such as '{"id": 12, "speaker": "ELENA"' has no closing
    # brace, so the scan above missed it. Recover it if it carries a speaker.
    tail_start = json_text.rfind('{')
    if tail_start > json_text.rfind('}'):
        tail = _extract_label_object(json_text[tail_start:])
        if tail is not None and tail.get("speaker"):
            labels.append(tail)

    return labels or None


def parse_label_array(json_text):
    """Strict array path: parse/repair a JSON array of labels. List or None."""
    if not json_text:
        return None

    parsed = repair_json_array(json_text)
    if parsed:
        usable = [label for label in parsed if _label_id(label) is not None]
        if usable:
            return usable
    return None


def parse_label_response(json_text):
    """Parse classifier output into label dicts. Returns a list or None."""
    labels = parse_label_array(json_text) or salvage_label_entries(json_text)
    return _recover_label_keys(labels) if labels else labels


def _find_balanced(text, opener, closer):
    """Return the first balanced ``opener``...``closer`` slice, string-aware."""
    start = text.find(opener)
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for index, char in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if char == '\\':
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return None


def _strip_code_fences(text):
    """Return the contents of the first fenced code block, or ``text`` itself."""
    if "```" not in text:
        return text
    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    return match.group(1).strip() if match else text


def labels_from_id_keyed_object(text):
    """Recover an id-keyed JSON OBJECT instead of the requested array.

    Observed live from qwen3:30b-a3b (thinking model), repeatedly::

        {"1": {"speaker": "NARRATOR", "role": "narration", "instruct": "..."},
         "2": {"speaker": "BILL", "role": "dialogue", "instruct": "..."}, ...}

    This is valid JSON with complete, unambiguous labels -- only the envelope
    is wrong -- so it is a first-class recovery, not a salvage: no prose is at
    risk and no information is missing. The KEY is authoritative; a conflicting
    inner "id" field is overwritten.
    """
    if not text:
        return None

    candidate = _strip_code_fences(strip_thinking_tags(text))
    blob = _find_balanced(candidate, '{', '}')
    if not blob:
        return None

    try:
        parsed = json.loads(blob)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(parsed, dict) or not parsed:
        return None

    labels = []
    for key, value in parsed.items():
        # ALL keys must be integer-like, else this is an ordinary object
        # (e.g. a single {"id": 1, "speaker": ...} label) and not a mapping.
        if not isinstance(key, str) or not key.strip().isdigit():
            return None
        if not isinstance(value, dict):
            return None
        label = dict(value)
        label["id"] = int(key.strip())
        labels.append(label)

    return labels or None


# Markdown fallback. Headers seen live: "**id 1**", "**Span 1**" (both with
# trailing markdown hard-break spaces); fields as "- **Speaker**: NARRATOR".
_MD_HEADER_RE = re.compile(
    r'^[ \t]*(?:#{1,6}[ \t]*)?(?:[-*+][ \t]*)?\*{0,2}[ \t]*(?:span|id)[ \t#]*(\d+)[ \t]*[:.)]?[ \t]*\*{0,2}[ \t]*$',
    re.IGNORECASE | re.MULTILINE,
)
_MD_FIELD_RE = re.compile(
    r'^[ \t]*(?:[-*+][ \t]*)?\*{0,2}[ \t]*(speaker|role|instruct)[ \t]*\*{0,2}[ \t]*[:=][ \t]*(.*?)[ \t]*$',
    re.IGNORECASE | re.MULTILINE,
)
# Quote/emphasis characters wrapping a markdown field value.
_MD_VALUE_TRIM = "\"'“”‘’*_`, \t"


def salvage_markdown_labels(text):
    """LAST-RESORT recovery of markdown label blocks (no JSON at all).

    Observed live from qwen3:30b-a3b, consistently structured::

        **id 1**
        - **Speaker**: NARRATOR
        - **Role**: narration
        - **Instruct**: "Neutral, even narration."

    Firing this means the model ignored the output contract, so the caller
    marks the chunk degraded even though the labels are usable.
    """
    if not text:
        return None

    body = strip_thinking_tags(text)
    if not body:
        return None

    headers = list(_MD_HEADER_RE.finditer(body))
    if not headers:
        return None

    labels = []
    for position, header in enumerate(headers):
        block_end = headers[position + 1].start() if position + 1 < len(headers) else len(body)
        block = body[header.end():block_end]

        label = {"id": int(header.group(1))}
        for field in _MD_FIELD_RE.finditer(block):
            value = field.group(2).strip(_MD_VALUE_TRIM)
            if value:
                label[field.group(1).lower()] = value

        # An id alone is not a label; require something to act on.
        if "speaker" in label or "role" in label:
            labels.append(label)

    return labels or None


# How the labels for a chunk were recovered, most faithful first.
LABEL_MODE_ARRAY = "array"
LABEL_MODE_OBJECT = "id-keyed object"
LABEL_MODE_SALVAGE = "regex salvage"
LABEL_MODE_MARKDOWN = "markdown salvage"


def extract_labels(text):
    """Recover labels from a raw LLM response. Returns ``(labels, mode)``.

    Tried in descending fidelity: the requested JSON array, an id-keyed JSON
    object, regex salvage of individual JSON objects (this is what survives a
    truncation), then markdown blocks. ``(None, None)`` when nothing is usable.

    Misspelled schema keys are recovered here, at the one point every parse mode
    passes through, so the retry predicate and the resolver always see the same
    labels -- they have diverged twice before, and a key recovered on only one
    side would be a third divergence.
    """
    labels, mode = _extract_labels(text)
    return (_recover_label_keys(labels) if labels else labels), mode


def _extract_labels(text):
    """extract_labels() without key recovery. See its docstring."""
    if not text:
        return None, None

    json_text = clean_json_string(text)
    if json_text:
        labels = parse_label_array(json_text)
        if labels:
            return labels, LABEL_MODE_ARRAY

    labels = labels_from_id_keyed_object(text)
    if labels:
        return labels, LABEL_MODE_OBJECT

    # Salvage the cleaned array text first, then ALWAYS retry against the raw
    # response if that yielded nothing. Falling back only when json_text was
    # None was too narrow: a wrong-but-truthy cleaned slice (prose brackets --
    # see clean_json_string) displaced the raw text and cost the chunk every
    # label, even though salvaging the raw text would have recovered them all.
    labels = salvage_label_entries(json_text) if json_text else None
    if not labels:
        labels = salvage_label_entries(text)
    if labels:
        return labels, LABEL_MODE_SALVAGE

    labels = salvage_markdown_labels(text)
    if labels:
        return labels, LABEL_MODE_MARKDOWN

    return None, None


def select_prompt(custom_prompt, default_prompt, config_key):
    """Choose between a saved custom prompt and the built-in default.

    A custom prompt is honoured only when it carries PROMPT_SCHEMA_MARKER,
    i.e. when it was written for the span-classifier schema. Anything older
    targets the retired "rewrite the book into speaker/text/instruct" schema
    and would produce label-less output, so it is rejected loudly and the
    built-in default is used instead.
    """
    if not custom_prompt or not custom_prompt.strip():
        return default_prompt

    if PROMPT_SCHEMA_MARKER in custom_prompt:
        return custom_prompt

    print(f"  {'!' * 60}")
    print(f"  WARNING: saved custom prompt '{config_key}' predates the span-classifier")
    print(f"  pipeline (missing '{PROMPT_SCHEMA_MARKER}' marker) -- using the built-in")
    print("  default instead. Re-customize starting from the new default if needed.")
    print(f"  {'!' * 60}")
    return default_prompt


def is_whitespace_span(span, source):
    """True for a span with no audible content (e.g. the "\\n\\n" between two
    quoted paragraphs).

    Such spans are excluded from the LLM loop entirely: the model never sees
    them, is never asked to label them, is never retried for skipping them, and
    they never degrade a chunk. Their label would be thrown away regardless --
    _absorb_whitespace_groups folds a whitespace-only group's TEXT into a
    neighbour and drops its speaker/instruct. They stay in the span list, so
    byte-identity and whitespace absorption are unaffected.
    """
    return not span.text(source).strip()


def visible_spans(spans, source):
    """The spans worth asking the LLM about."""
    return [span for span in spans if not is_whitespace_span(span, source)]


def build_span_payload(spans, source):
    """Render spans as the numbered listing the classifier receives.

    One JSON object per line: {"id", "kind", "text"}. The LLM sees the text so
    it can classify it; it is instructed never to send text back. Whitespace-only
    spans are omitted -- they cost tokens and invite phantom skips, and their
    ids stay the tokenizer's, so the visible ids simply have gaps.
    """
    return "\n".join(
        json.dumps(
            {"id": span.id, "kind": span.kind, "text": span.text(source)},
            ensure_ascii=False,
        )
        for span in spans
        if not is_whitespace_span(span, source)
    )


# Recovery modes ranked by fidelity, worst-last. When labels for one chunk are
# merged across attempts, the WORST contributing mode is reported, so an
# attempt-1 array followed by an attempt-2 markdown salvage still surfaces the
# markdown contract violation instead of hiding it behind the array.
_MODE_FIDELITY = {
    "array": 0,
    "id-keyed object": 1,
    "regex salvage": 2,
    "markdown salvage": 3,
}


def _worst_mode(current, new):
    """Return the lower-fidelity of two recovery modes."""
    if new is None:
        return current
    if current is None:
        return new
    return new if _MODE_FIDELITY.get(new, 0) > _MODE_FIDELITY.get(current, 0) else current


def _usable_field(name, value):
    """True when a label field carries information we can actually act on.

    ``role`` is special: only the two contract values mean anything, so an
    invented value ("thought" was observed live on a genuinely spoken line)
    counts as absent and is therefore fillable by a retry.
    """
    if not isinstance(value, str) or not value.strip():
        return False
    if name == "role":
        return value.strip().lower() in ("dialogue", "narration")
    return True


def _merge_label(old, new):
    """Fill blank fields from ``new``; never overwrite information already held.

    This is the cross-attempt merge, and it is deliberately NOT
    _label_completeness (which ranks whole labels and would let a retry replace
    a correct label with a differently-shaped one). Filling blanks makes
    "a retry can never make things worse" a structural property: a later
    attempt can only add ids, or fill fields that were missing, empty or --
    for role -- outside the contract.
    """
    if old is None:
        return dict(new)
    merged = dict(old)
    for field in ("speaker", "role", "instruct"):
        if not _usable_field(field, merged.get(field)) and _usable_field(field, new.get(field)):
            merged[field] = new[field]
    return merged


def _incomplete_span_ids(spans, merged, source=None):
    """Return ``(missing_ids, bad_role_ids, no_speaker_ids)`` -- what a retry asks for.

    Membership, never a count: live responses hallucinate ids past N (137
    labels for 41 spans in one logged response), so ``len(merged) >= len(spans)``
    would fake completeness. ``bad_role_ids`` mirrors resolve_span_labels'
    role_missing condition exactly, and ``no_speaker_ids`` mirrors its
    dialogue_without_speaker condition exactly, so the retry predicate and the
    degradation reason can never disagree.

    ``no_speaker_ids`` is the mirror image of ``bad_role_ids``: role is good and
    says "dialogue", but no usable name came with it, so a voice is lost. Before
    it existed such a label was neither missing, nor bad-role, nor unattested
    (the attestation pass skips a label with no speaker), so the attempt loop
    broke on attempt 1 and the span was narrated in silence -- observed on two
    live chunks with 26/27 such labels and ZERO retry lines.

    When ``source`` is given, whitespace-only spans are skipped: the model was
    never shown them, so it cannot be "missing" one.
    """
    if source is not None:
        spans = visible_spans(spans, source)

    missing = [span.id for span in spans if span.id not in merged]
    bad_role = []
    no_speaker = []
    for span in spans:
        label = merged.get(span.id)
        if label is None:
            continue
        raw_speaker = label.get("speaker")
        canonical = canonicalize(raw_speaker) if isinstance(raw_speaker, str) else ""
        if not _usable_field("role", label.get("role")):
            if canonical and canonical != NARRATOR:
                bad_role.append(span.id)
            continue
        # Exactly resolve_span_labels' `role == "dialogue" and not canonical`
        # branch, including its NARRATOR handling: NARRATOR canonicalizes to a
        # truthy value, so a NARRATOR label is not asked about there and must
        # not be asked about here either.
        role = label.get("role")
        role = role.strip().lower() if isinstance(role, str) else ""
        if role == "dialogue" and not canonical:
            no_speaker.append(span.id)
    return missing, bad_role, no_speaker


# Cap the id list in a nudge so a wholly-unlabelled chunk cannot balloon the prompt.
_NUDGE_ID_CAP = 50


def _format_id_list(ids):
    if len(ids) <= _NUDGE_ID_CAP:
        return ", ".join(str(i) for i in ids)
    head = ", ".join(str(i) for i in ids[:_NUDGE_ID_CAP])
    return f"{head} (and {len(ids) - _NUDGE_ID_CAP} more)"


def _retry_nudge(missing_ids, bad_role_ids, unattested=None, no_speaker_ids=None,
                 spelling_hints=None, contradicted=None):
    """Correction text appended to the retry prompt.

    Appended AFTER .format(), so it cannot interact with the {context}/{chunk}
    placeholders or the doubled braces in the template, and nothing is stored
    in config.json -- select_prompt / PROMPT_SCHEMA_MARKER are untouched, and a
    marker-bearing custom prompt keeps working.

    Naming the ids matters: logged misses are a contiguous SUFFIX of the id
    sequence (the model closes the array early, well under max_tokens), so a
    blind re-roll re-rolls the same "I am finished" judgement. Naming the gap
    corrects it instead.
    """
    parts = ["\n\nCORRECTION: your previous reply had problems."]
    if missing_ids:
        parts.append("These span ids are still missing a label: "
                     f"{_format_id_list(missing_ids)}.")
    if bad_role_ids:
        parts.append('These span ids had a "role" that was neither "dialogue" nor '
                     f'"narration": {_format_id_list(bad_role_ids)}.')
    if no_speaker_ids:
        parts.append('These span ids had role "dialogue" but no usable "speaker" '
                     f'name: {_format_id_list(no_speaker_ids)}. For each one, give '
                     "the speaking character's name exactly as the text spells it, "
                     'or use "NARRATOR" with role "narration" if the span is not '
                     "spoken dialogue.")
    if unattested:
        # Name the offending spelling AND say what to do instead. A bare "that
        # name is wrong" makes the model guess again; pointing it at the text
        # is what turns the rejection into a correction. The correct spelling
        # is in the text it was shown -- on measured runs the misread spelling
        # occurred zero times in the book and the right one hundreds of times.
        detail = "; ".join(f'id {span_id}: "{name}"' for span_id, name in unattested)
        parts.append(
            "These speaker names do not appear anywhere in the text you were "
            f"shown, so they cannot be right: {detail}. For each one, either "
            "copy the character's name EXACTLY as it is spelled in the span "
            "text above, or reuse a name from the character roster exactly, or "
            'label the span "NARRATOR" with role "narration" if the text never '
            "names the speaker. Do not invent a name and do not guess at a "
            "spelling.")
        # Candidate spellings, for the offenders where automatic repair DECLINED
        # (an ambiguous or unsupported repair). A free hint, not a mechanism:
        # per a28839d the prompt cannot enforce anything, so this only makes the
        # right answer easier to find -- the gate still re-checks whatever comes
        # back.
        hints = []
        for _, name in unattested:
            candidates = (spelling_hints or {}).get(name) or []
            if candidates:
                hints.append(f'"{name}" (did you mean '
                             + " or ".join(f'"{c}"' for c in candidates) + ")")
        if hints:
            parts.append("Established character spellings close to the rejected "
                         "names, if one of them is who you meant: "
                         + "; ".join(hints) + ".")
    if contradicted:
        # Same shape as the unattested hint above -- name the id, name what was
        # said, and point at the evidence in the text the model was shown --
        # because the same thing is true here: a bare "that is wrong" makes the
        # model guess again. The tag is NOT asserted to be the answer; the
        # model may re-state its label, and nothing here rewrites it.
        detail = "; ".join(
            f'id {span_id}: you said "{speaker}", but the narration right after '
            f'it attributes the line to "{tagged}"'
            for span_id, speaker, tagged in contradicted)
        parts.append(
            "These spans disagree with their own attribution tag: "
            f"{detail}. Re-read the narration immediately following each of "
            "these quotations and give the speaker the text attributes it to. "
            "If the tag belongs to a DIFFERENT quotation and your original "
            "answer was right, repeat it.")
    parts.append("Return the JSON array again. It MUST contain one object for every id "
                 'listed above, and every "role" must be exactly "dialogue" or "narration".')
    return " ".join(parts)


def _label_completeness(label):
    """Rank a label so a complete one beats a truncated fragment for the same id.

    A usable ``role`` is what actually decides the speaker, so it is the field
    worth ranking on; ``instruct`` breaks remaining ties.
    """
    if not isinstance(label, dict):
        return 0
    role = label.get("role")
    role = role.strip().lower() if isinstance(role, str) else ""
    score = 2 if role in ("dialogue", "narration") else 0
    raw_instruct = label.get("instruct")
    if isinstance(raw_instruct, str) and raw_instruct.strip():
        score += 1
    return score


def _speaker_is_established(canonical, roster_index):
    """True when ``canonical`` is already a roster name for this book.

    An established name needs no attestation: it was itself attested when it
    first entered the roster, so this is BOTH the correctness rule and the
    long-range attribution mechanism -- a character last named twenty pages
    ago is accepted at zero token cost, because the roster travels with the
    book instead of with the prompt.
    """
    if not roster_index:
        return False
    return roster_key(canonical) in roster_index


def _attestation_verdict(canonical, attest_window, roster_index=None):
    """attest_speaker() over the one window the gate uses. Pure passthrough,
    factored out so the retry predicate and the acceptance gate cannot drift
    apart (the same mistake _incomplete_span_ids exists to prevent for roles).

    ``roster_index`` enables speaker_canon's roster-name partial attestation:
    a multi-token label built around an established name is UNVERIFIABLE
    (accept-and-count) rather than refuted.
    """
    return attest_speaker(canonical, [attest_window] if attest_window else [],
                          roster_index=roster_index)


# Gate outcomes returned by _gate_speaker(). Named so the two call sites
# compare against the same constants rather than re-spelling them.
GATE_ESTABLISHED = "established"
GATE_ATTESTED = "attested"
GATE_UNVERIFIABLE = "unverifiable"
GATE_REPAIRED = "repaired"
GATE_REPAIR_REFUSED = "repair_refused"
GATE_REJECTED = "rejected"


def _gate_speaker(canonical, attest_window, roster_index, source_words,
                  following_text=None):
    """THE attestation decision, in one place: ``(name, outcome)``.

    Both the acceptance gate in resolve_span_labels and the retry predicate in
    _unattested_speaker_ids call this and nothing else, so they cannot apply
    different rules -- commit 2809180 exists because they once did, and a
    property test now pins their agreement. Adding repair here rather than at
    either call site is what keeps that property true for repaired labels too:
    a label the gate silently repairs must NOT be one the retry nags about.

    ``source_words`` is the whole book's folded word index; when it is empty
    repair is simply unavailable and the outcome is a plain rejection, so a
    caller that cannot supply it is never worse off than before.

    ``following_text`` is the narration immediately after this span's quotation
    (_closing_tag_text_by_id), or None when there is none. It VETOES a repair:
    repair_speaker's guards are all about SPELLING -- is this spelling in the
    book, is exactly one roster name one edit away, does the target appear in
    this window -- and in a two-character scene both candidates appear in the
    window, so the spelling evidence picks the wrong character as readily as the
    right one. The adjacent attribution tag is the only cheap evidence about who
    speaks THIS span, and it is checked against the REPAIRED name: when the tag
    names a different established character, the repair is refused and the label
    falls back to the plain rejection it would have had before repair existed
    (GATE_REPAIR_REFUSED -> NARRATOR, counted, loud, contract 7).

    That veto is deliberately NOT behind generation.check_attribution_tags. That
    flag guards a check that JUDGES THE MODEL -- it spends retries and can
    degrade a chunk over a label the model actually emitted. This one guards a
    rewrite THIS PIPELINE performs: it can only withhold a name the model never
    wrote, and its worst case is the behaviour the gate already promises. A
    guard on our own rewrite needs no opt-in.

    Read-only: ``roster_index`` is not mutated here.
    """
    if _speaker_is_established(canonical, roster_index):
        return canonical, GATE_ESTABLISHED
    verdict = _attestation_verdict(canonical, attest_window, roster_index)
    if verdict == UNATTESTED:
        # REJECTED WHEN THE REST OF THE GATE CANNOT SAVE IT. A "rescue the
        # label when its refuted tokens are words the book uses somewhere" rule
        # was tried here and reverted: checking tokens INDIVIDUALLY against the
        # whole book is weaker than the window test it bypasses, so it admitted
        # fabricated names built from real words -- TRANSFORMED PIG (the book
        # has "Transcendent Pig"; "transformed pig" appears zero times) and the
        # bare prose word DRAINED, each becoming a character with its own
        # voice. That is the recombination the phrase-adjacency rule in
        # attest_label exists to refuse, so a second path must not re-admit it.
        repaired = repair_speaker(
            canonical, [attest_window] if attest_window else [],
            roster_index or {}, source_words or set())
        if repaired:
            if contradicts_attribution(repaired, following_text, roster_index or {}):
                return canonical, GATE_REPAIR_REFUSED
            return repaired, GATE_REPAIRED
        return canonical, GATE_REJECTED
    if verdict == UNVERIFIABLE:
        return canonical, GATE_UNVERIFIABLE
    return canonical, GATE_ATTESTED


def _unattested_speaker_ids(spans, merged, source=None, roster=None,
                            attest_window=None, source_words=None):
    """Return ``[(span_id, canonical_name), ...]`` for dialogue labels whose
    speaker the source does not support -- what a retry should ask about.

    Mirrors the acceptance gate in resolve_span_labels exactly, because both go
    through _gate_speaker(): same established-roster shortcut, same verdict,
    same repair, same adjacent-tag veto on that repair. A rejection counts
    whether it is plain (GATE_REJECTED) or a refused repair
    (GATE_REPAIR_REFUSED) -- both narrate the span, so both are worth a retry.
    UNVERIFIABLE never
    appears here, because a label our check cannot evaluate is not something to
    nag the model about -- and neither does a REPAIRED one, because the gate is
    about to accept the repaired spelling.

    Read-only. Does not mutate ``roster``; a local copy tracks names accepted
    earlier in this same chunk so the shortcut behaves as it will at
    resolution time.
    """
    # Built from the FULL span list (before the visible filter) so adjacency is
    # the tokenizer's, exactly as _tag_contradictions sees it.
    tag_texts = _closing_tag_text_by_id(spans, source)
    if source is not None:
        spans = visible_spans(spans, source)

    roster_index = dict(roster) if roster else {}
    offenders = []
    for span in spans:
        label = merged.get(span.id)
        if label is None:
            continue
        role = label.get("role")
        role = role.strip().lower() if isinstance(role, str) else ""
        if role != "dialogue":
            continue
        raw_speaker = label.get("speaker")
        canonical = canonicalize(raw_speaker) if isinstance(raw_speaker, str) else ""
        if not canonical or canonical == NARRATOR:
            continue
        if is_placeholder_speaker(canonical):
            continue  # already rejected on its own terms
        name, outcome = _gate_speaker(canonical, attest_window, roster_index,
                                      source_words, tag_texts.get(span.id))
        if outcome in (GATE_REJECTED, GATE_REPAIR_REFUSED):
            offenders.append((span.id, canonical))
        else:
            remember_in_roster(roster_index, name)
    return offenders


# Closing quote marks a quotation span may end on. A dialogue span that does
# NOT end on one is mid-quotation, so the narration after it is not a closing
# attribution tag (see _tag_contradictions' bound).
_CLOSING_QUOTES = "\"'”’»」』"


def _closing_tag_text_by_id(spans, source):
    """``{span_id: the narration right after this quotation}`` for every span a
    closing attribution tag can be read from.

    THE bounds, in one place, so the two consumers -- the post-hoc contradiction
    report (_tag_contradictions) and the repair veto in _gate_speaker -- cannot
    drift apart, and so there is exactly one tag-adjacency rule in the pipeline:

      * the span must be QUOTED and end on a closing quote mark, so the tag
        closes it rather than sitting mid-quotation;
      * the very next span must be UNQUOTED and not whitespace-only (a
        whitespace-only neighbour is the paragraph break, and a tag on the far
        side of one introduces the NEXT quotation).

    Parsing the tag itself is speaker_canon's job (attribution_tag_name /
    contradicts_attribution); this returns raw source text only. Pure.
    """
    if source is None:
        return {}
    closing = tuple(_CLOSING_QUOTES)
    texts = {}
    for index, span in enumerate(spans[:-1]):
        if span.kind != QUOTED:
            continue
        if not span.text(source).rstrip().endswith(closing):
            continue
        following = spans[index + 1]
        if following.kind != UNQUOTED or is_whitespace_span(following, source):
            continue
        texts[span.id] = following.text(source)
    return texts


def _tag_contradictions(spans, speaker_by_id, source, roster_index):
    """``[(span_id, speaker, tagged_name), ...]`` for dialogue spans whose label
    is contradicted by the attribution tag in the very next span.

    THE GAP THIS CLOSES. Attestation asks whether the chosen name occurs
    somewhere nearby; nothing asked whether it is the name the adjacent tag
    gives. Measured on one clean 8,081-entry artifact: 60 of 1,117 checkable
    dialogue+tag pairs (5.4%) named a different established character than the
    tag did, all of them attested and none of them detected.

    DETECTS, NEVER REWRITES. A hit is routed exactly like an unattested
    speaker: into the retry nudge first (process_chunk), and into the
    degradation report if the retry does not resolve it (resolve_span_labels).
    Relabelling the span from the tag would be a second repair_speaker --
    nearby evidence silently overruling the model -- which is the failure this
    pipeline keeps trying to eliminate, not add.

    BOUNDS, chosen for PRECISION over recall (a false accusation costs a wasted
    retry and a false exit-3, which is what erodes trust in exit 3):

      * the labelled span must be a QUOTED span ending on a closing quote mark,
        so the tag closes it rather than sitting mid-quotation;
      * the tag must be in the IMMEDIATELY NEXT span, and that span must be
        UNQUOTED and not whitespace-only. A whitespace-only span between them
        is the paragraph break, and a tag on the far side of one introduces the
        NEXT quotation instead of closing this one;
      * speaker_canon.contradicts_attribution supplies the rest: no possessive
        tag, no uncapitalized/common-noun tag, no ambiguous name, and
        granularity/case/apostrophe variants are agreement, not conflict.

    Bound (4) is not perfect: an attributive-looking tag can be a verb with an
    object ("Kit said the last couple of words of the spell") and is then a
    false hit. Hand-checking a random 25 of the 64 hits on the artifact above
    found exactly one such case -- 24/25 genuine.

    ENGLISH-ONLY, degrading to silence: the tag pattern needs an English speech
    verb, so on a non-English book nothing matches and this returns [] for
    every span. See speaker_canon's Tier 3c section comment.

    Pure and read-only: ``roster_index`` is not mutated.
    """
    if source is None or not roster_index:
        return []

    contradictions = []
    for span_id, following_text in _closing_tag_text_by_id(spans, source).items():
        speaker = speaker_by_id.get(span_id)
        if not speaker or speaker == NARRATOR:
            continue
        tagged = contradicts_attribution(speaker, following_text, roster_index)
        if tagged:
            contradictions.append((span_id, speaker, tagged))
    return contradictions


def _contradicted_speaker_ids(spans, merged, source=None, roster=None):
    """The retry-offender view of _tag_contradictions: run it over the labels a
    chunk currently holds, before they are resolved.

    Mirrors _unattested_speaker_ids' shape and contract (read-only, dialogue
    labels only, NARRATOR skipped) so the retry predicate and the degradation
    count in resolve_span_labels apply the same rule -- the drift
    _incomplete_span_ids' docstring warns about.

    The roster used is the book's roster PLUS every canonical dialogue name in
    this chunk, so a character first named in this very chunk can still be the
    name a tag resolves to.
    """
    speaker_by_id = {}
    roster_index = dict(roster) if roster else {}
    for span in spans:
        label = merged.get(span.id)
        if not isinstance(label, dict):
            continue
        role = label.get("role")
        role = role.strip().lower() if isinstance(role, str) else ""
        if role != "dialogue":
            continue
        raw_speaker = label.get("speaker")
        canonical = canonicalize(raw_speaker) if isinstance(raw_speaker, str) else ""
        if not canonical or canonical == NARRATOR or is_placeholder_speaker(canonical):
            continue
        speaker_by_id[span.id] = canonical
        remember_in_roster(roster_index, canonical)

    return _tag_contradictions(spans, speaker_by_id, source, roster_index)


def _report_repairs(chunk_num, total_chunks, repairs):
    """Make every speaker repair visible: one console line per distinct repair
    plus a per-repair record in the forensic log.

    Contract 7 says degradation is loud, and a repair is the one gate outcome
    that is neither a rejection nor an untouched acceptance: the script ends up
    saying a name the model did not emit. A stat alone would leave an operator
    unable to answer "where did this name come from?", so the original -> repaired
    mapping goes into llm_responses.log next to the response it came from.
    Never raises: a failure to WRITE the log must not lose a chunk's work.
    """
    if not repairs:
        return

    counts = {}
    for original, repaired in repairs:
        counts[(original, repaired)] = counts.get((original, repaired), 0) + 1

    for (original, repaired), count in sorted(counts.items()):
        print(f"  REPAIRED speaker label \"{original}\" -> \"{repaired}\" "
              f"on {count} span(s): the book never spells it \"{original}\", and "
              f"exactly one established name is one edit away")

    try:
        log_dir = llm_log_dir()
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "llm_responses.log"), "a",
                  encoding="utf-8") as handle:
            for (original, repaired), count in sorted(counts.items()):
                handle.write(f"SPEAKER REPAIR | chunk {chunk_num}/{total_chunks} | "
                             f"{original!r} -> {repaired!r} | {count} span(s)\n")
    except OSError as exc:  # pragma: no cover - depends on the filesystem
        print(f"  WARNING: could not record speaker repairs in the log ({exc})")


def resolve_span_labels(spans, labels, source=None, roster=None,
                        attest_window=None, require_attested=False,
                        source_words=None, check_tags=False):
    """Resolve each span to (span, speaker, instruct) using the LLM's labels.

    A span is NARRATOR unless a label exists for its id AND that label says
    role == "dialogue" AND its speaker canonicalizes to a non-empty name.
    Labels for ids that do not exist are discarded. Returns
    (resolved, stats_dict).

    When ``source`` is given, whitespace-only spans are auto-resolved to
    NARRATOR with no instruct and counted in ``whitespace`` rather than
    ``fallback``: the model was never shown them, so a missing label is not a
    failure. ``labelled`` stays inclusive of them so callers may continue to
    rely on ``labelled + fallback == spans``.

    ``roster`` is an optional roster index (see speaker_canon.remember_in_roster)
    of speaker names already established EARLIER in this book. When given, an
    accepted speaker whose roster key matches an established name is normalized
    onto that established spelling -- so the observed "ABBE MARIGNAN" ->
    "ABBEMARIGNAN" drift cannot fork one character into two roster entries.
    Exact-key only (spellings differing solely in whitespace/hyphens/
    apostrophes); similar-but-distinct names (JON/JOHN) are never merged. The caller's dict is NOT mutated: a local copy is extended as
    the chunk resolves, so a spelling first seen mid-chunk participates in the
    same selection rule as the rest of the book.

    ATTESTATION GATE (``require_attested``, off by default). The speaker field
    is the one part of the schema the model still GENERATES rather than
    selects, and it misreads names: on a measured production run every one of
    MEMKEI, DARINE, CARMELO and ROSHAAUN occurred ZERO times in the source
    while the correct spelling occurred 176-721 times. With the gate on, a
    dialogue speaker is accepted iff

        (1) it is already a roster name for this book (see
            _speaker_is_established -- born attested, so free and unbounded in
            range), OR
        (2) attest_speaker() over ``attest_window`` does not return UNATTESTED.


    UNVERIFIABLE is ACCEPTED and counted separately: it means the check does
    not apply to this text (a title-only label, or a name present but not at a
    word boundary, as in unsegmented scripts), and rejecting on it would
    destroy books this module cannot tokenize. Only UNATTESTED -- positive
    evidence that the name appears nowhere near its own lines -- is refused.

    A refused speaker becomes NARRATOR, exactly like every other rejection
    here: prose is preserved verbatim and only the voice is lost. Because that
    trade is real (narrating a genuine line is itself a loss), the gate is
    designed to be paired with the retry nudge in process_chunk, which names
    the offending label and asks the model to copy the spelling from the text
    it was shown. Rejection is the fallback; re-asking is the fix.

    BOUNDED REPAIR (``source_words``). Before refusing, a refuted label is
    offered to speaker_canon.repair_speaker(), which folds it onto an already-
    established spelling when -- and only when -- the whole book never uses the
    label's spelling and exactly one roster name sits one edit away in the
    label's own window. Repairs are reported in ``speakers_repaired`` and
    itemized in ``repairs`` as (original, repaired) pairs; process_chunk prints
    and logs each one, because rewriting a name the model did not emit must not
    be silent (contract 7). ``source_words`` is speaker_canon.source_word_index()
    over the whole book; without it repair is unavailable and the gate behaves
    exactly as it did before.

    A repair is VETOED by the attribution tag next to the same line whenever
    that tag names a different established character (see _gate_speaker): the
    label is then refused like any other unattested one -- NARRATOR, counted in
    ``unattested_rejected``, itemized in ``refused_repairs`` and degrading the
    chunk. This veto does not depend on ``check_tags``: it withholds one of THIS
    pipeline's rewrites rather than judging the model's label, and its fallback
    is the behaviour the gate already promises.

    MEASURED on one 281-chunk production run replayed from its own logged
    responses: 89 repairs, 26 of them next to a machine-readable attribution
    tag, and all 26 tags AGREED with the repaired name -- so the veto refused 0
    and that run's output is unchanged. The remaining 63 sat next to a pronoun
    tag ("he said"), an action beat or nothing at all: no adjacent evidence
    exists, nothing here can adjudicate them, and they are deliberately left
    alone rather than guessed at. The veto is therefore cheap insurance against
    a failure class, not a fix for a measured error rate; what makes the
    unadjudicable majority visible is the repair block in the run summary.

    ATTRIBUTION-TAG CHECK (``check_tags``, off by default). After every speaker
    is decided, _tag_contradictions re-reads the result against the attribution
    tag in the span right after each quotation and counts the disagreements in
    ``tag_contradictions`` (itemized in ``contradictions``). Counted ONLY --
    the label is left exactly as the model gave it, because relabelling a span
    from adjacent evidence is repair_speaker's job description and this check
    exists to make wrong answers loud, not to invent new ones. process_chunk
    asks the model again first; what survives that lands here and degrades the
    chunk.
    """
    valid_ids = {span.id for span in spans}
    # Local copy: never mutate the caller's roster index (see docstring).
    roster_index = dict(roster) if roster else None
    by_id = {}
    discarded = 0

    for label in labels or []:
        label_id = _label_id(label)
        if label_id is None or label_id not in valid_ids:
            discarded += 1
            continue
        existing = by_id.get(label_id)
        if existing is not None and _label_completeness(existing) > _label_completeness(label):
            # salvage_label_entries() can recover a truncated tail object for an
            # id that already parsed cleanly. Never let the fragment overwrite
            # the complete label; equal completeness keeps last-write-wins.
            continue
        by_id[label_id] = label

    # Schema hygiene, counted once per label that is actually used (by_id is
    # already deduplicated). Only the strict-JSON path can carry keys outside
    # the schema -- _extract_label_object builds its dicts from a fixed list of
    # field names -- so a clean run stays silent here. What reaches this point
    # is what _recover_label_keys could NOT place: a key near two schema keys,
    # or near none ("inquest", "infty"). Those are still never read; the discard
    # is only made visible, so the residue of key recovery stays measurable.
    unknown_key_labels = 0
    unknown_keys = set()
    text_key_labels = 0
    for label in by_id.values():
        extra = {key for key in label if key not in _LABEL_SCHEMA_KEYS}
        if not extra:
            continue
        unknown_key_labels += 1
        unknown_keys |= extra
        if "text" in extra:
            # Contract 1: the model returns labels only. A "text" key is not a
            # typo, it is the model trying to supply book prose. It is ignored
            # exactly like any other unknown key (entry text is always
            # source[span.start:span.end]) but it is reported separately.
            text_key_labels += 1

    resolved = []
    labelled = 0
    role_missing = 0
    whitespace = 0
    placeholder_rejected = 0
    unattested_rejected = 0
    unverifiable_accepted = 0
    dialogue_without_speaker = 0
    repairs = []
    repairs_refused = []
    tag_texts = _closing_tag_text_by_id(spans, source)

    for span in spans:
        if source is not None and is_whitespace_span(span, source):
            # Never shown to the model; its label would be discarded by
            # _absorb_whitespace_groups anyway. Any label sent for this id is
            # ignored here, exactly as a phantom label would be.
            resolved.append((span, NARRATOR, None))
            whitespace += 1
            continue

        label = by_id.get(span.id)
        speaker = NARRATOR
        instruct = None

        if label is not None:
            labelled += 1
            role = label.get("role")
            role = role.strip().lower() if isinstance(role, str) else ""
            raw_speaker = label.get("speaker")
            canonical = canonicalize(raw_speaker) if isinstance(raw_speaker, str) else ""

            if role == "dialogue" and canonical and is_placeholder_speaker(canonical):
                # An invented enumerated placeholder ("SPEAKER 1") is not a
                # character; it would fragment the roster and claim its own
                # voice. NARRATOR is the safe direction -- prose is untouched.
                placeholder_rejected += 1
            elif role == "dialogue" and canonical:
                # Roster-aware acceptance point: resolve the spelling against
                # names already established in this book, then establish this
                # one for the rest of the chunk.
                #
                # The attestation gate sits here, as a sibling of the
                # placeholder rejection above and with the same safe direction
                # (NARRATOR, prose untouched). An established roster name
                # skips the check entirely -- see the docstring.
                accept = True
                # NARRATOR is the narrator SENTINEL, not a character name, so it
                # is never attested: the word "Narrator" does not appear in a
                # novel, and gating it rejected the narrator itself. The outcome
                # was unchanged (a refused speaker becomes NARRATOR, which it
                # already was), but every chunk where the model wrote
                # speaker=NARRATOR with role=dialogue -- which the prompt itself
                # invites for an unidentifiable speaker -- was falsely reported
                # DEGRADED, inflating unattested_rejected and forcing exit 3.
                #
                # This also restores agreement with _unattested_speaker_ids,
                # which always skipped NARRATOR: the two must apply identical
                # rules or the retry predicate and the degradation reason
                # disagree, exactly as _incomplete_span_ids' docstring warns.
                if require_attested and canonical != NARRATOR:
                    name, outcome = _gate_speaker(
                        canonical, attest_window, roster_index, source_words,
                        tag_texts.get(span.id))
                    if outcome == GATE_REJECTED:
                        accept = False
                        unattested_rejected += 1
                    elif outcome == GATE_REPAIR_REFUSED:
                        # A spelling repair was available but the tag next to
                        # this very line names someone else, so the repair was
                        # withheld and the label falls back to the plain
                        # rejection. Counted in unattested_rejected too (it IS
                        # one, and the chunk must degrade for it), and itemized
                        # separately because "the repair was wrong here" is a
                        # different thing for an operator to read than "the
                        # model invented a name".
                        accept = False
                        unattested_rejected += 1
                        repairs_refused.append(canonical)
                    elif outcome == GATE_UNVERIFIABLE:
                        unverifiable_accepted += 1
                    elif outcome == GATE_REPAIRED:
                        # Accepted, NOT counted as an offender -- but recorded,
                        # because the script will now say a name the model
                        # never wrote. process_chunk prints and logs these.
                        repairs.append((canonical, name))
                        canonical = name

                if accept:
                    if roster_index is not None:
                        canonical = remember_in_roster(roster_index, canonical)
                    speaker = canonical
            elif role == "dialogue" and not canonical:
                # role says dialogue but there is no usable name, so a voice is
                # lost. Counted (and reported) rather than silently narrated:
                # this is where a speaker cleared by the attestation retry ends
                # up if the retry supplies nothing, and it is also what a model
                # emitting role=dialogue with an empty speaker has always done.
                dialogue_without_speaker += 1
            elif role not in ("dialogue", "narration") and canonical and canonical != NARRATOR:
                # Contract says role decides. Honour it (NARRATOR is the safe
                # direction: prose is preserved, only the voice changes), but
                # count it so the warning tells the operator the model is
                # emitting a schema we cannot fully trust.
                role_missing += 1

            raw_instruct = label.get("instruct")
            if isinstance(raw_instruct, str) and raw_instruct.strip():
                instruct = raw_instruct.strip()

        resolved.append((span, speaker, instruct))

    # Tag agreement, on the FINAL speakers (post-gate, post-roster-resolution),
    # so it judges what the script will actually say. Read-only.
    contradictions = []
    if check_tags:
        tag_roster = dict(roster_index) if roster_index else {}
        speaker_by_id = {}
        for span, speaker, _ in resolved:
            if speaker and speaker != NARRATOR:
                speaker_by_id[span.id] = speaker
                remember_in_roster(tag_roster, speaker)
        contradictions = _tag_contradictions(spans, speaker_by_id, source, tag_roster)

    return resolved, {
        # Inclusive of auto-resolved whitespace spans, so callers keep the
        # `labelled + fallback == spans` accounting invariant.
        "labelled": labelled + whitespace,
        "fallback": len(spans) - labelled - whitespace,
        "whitespace": whitespace,
        "discarded": discarded,
        "role_missing": role_missing,
        "placeholder_rejected": placeholder_rejected,
        "unattested_rejected": unattested_rejected,
        "unverifiable_accepted": unverifiable_accepted,
        "speakers_repaired": len(repairs),
        "repairs": repairs,
        "repairs_refused": len(repairs_refused),
        "refused_repairs": repairs_refused,
        "dialogue_without_speaker": dialogue_without_speaker,
        "tag_contradictions": len(contradictions),
        "contradictions": contradictions,
        "unknown_key_labels": unknown_key_labels,
        "unknown_keys": sorted(unknown_keys),
        "text_key_labels": text_key_labels,
    }


def _merge_same_speaker(groups):
    """Collapse adjacent [speaker, text, instruct, kind] groups sharing a speaker.

    KIND-AWARE, and that is the point. Two adjacent QUOTED spans by one
    character are one continuous line and merging them is pure win (one TTS
    call instead of two). A QUOTED span merged with an adjacent UNQUOTED one is
    a different animal: the entry now holds a quotation AND the narration
    around it, and the renderer gives the whole entry ONE voice, so the
    narration is spoken by the character. So a non-NARRATOR group only absorbs
    a group of the SAME kind; NARRATOR merges freely, since every kind of text
    a narrator reads is read in the narrator voice anyway.

    A merged group keeps the kind of its first member, which is well defined
    precisely because only same-kind groups merge (NARRATOR's kind is never
    consulted again).
    """
    merged = []
    for speaker, text, instruct, kind in groups:
        if merged and merged[-1][0] == speaker and (
            speaker == NARRATOR or merged[-1][3] == kind
        ):
            merged[-1][1] += text
            if merged[-1][2] is None:
                merged[-1][2] = instruct
        else:
            merged.append([speaker, text, instruct, kind])
    return merged


def _narrate_attribution_spans(resolved):
    """Force an UNQUOTED span to NARRATOR when it sits immediately beside a
    QUOTED span carrying the SAME character label.

    WHY THIS AND NOT "UNQUOTED IMPLIES NARRATOR". The classifier sometimes
    extends a character label off the end of the quotation and onto the
    attribution tag or action beat next to it ("said Marcus", " Nita said.").
    That text is narration and must be narrated. But a blanket
    unquoted-implies-narrator rule is WRONG and was rejected before: books
    legitimately contain unquoted character speech -- telepathy, silent speech,
    an animal's voice, interior monologue -- written with no quote marks at
    all. Measured on one 8,083-entry artifact: of 65 character-voiced entries
    containing no quotation mark, roughly 29 were genuine unquoted speech. A
    blanket rule destroys those.

    ADJACENCY IS THE ONLY STRUCTURAL SIGNAL THAT SEPARATES THEM, and it only
    separates one of the two cases. An unquoted span pressed directly against a
    quotation the model gave the same speaker is that quotation's tag or beat;
    a STANDALONE unquoted span labelled with a character has no structural
    tell distinguishing "narration the model mislabelled" from "unquoted speech
    the author wrote". Nothing here guesses at the standalone case: it is left
    with its label, and the damage is bounded instead by _merge_same_speaker
    refusing to let it swallow an adjacent quotation.

    A whitespace-only span between the two blocks adjacency, because it is a
    paragraph break -- the same bound _closing_tag_text_by_id draws for the
    same reason. Whitespace spans already resolve to NARRATOR, so this falls
    out of the speaker comparison for free.

    MEASURED on the artifact above: 13,337 chars of narration were read in a
    character voice; this rule and the kind-aware merge together bring that to
    8,220 (-38%) and remove every entry that mixes quotation with narration
    under a character voice (73 -> 0), for +98 entries (+1.2%).

    Text is untouched (contract 2/4) -- only the speaker label changes.
    """
    narrated = []
    for index, (span, speaker, instruct) in enumerate(resolved):
        if speaker != NARRATOR and span.kind == UNQUOTED:
            neighbours = (
                resolved[index - 1] if index else None,
                resolved[index + 1] if index + 1 < len(resolved) else None,
            )
            if any(other and other[0].kind == QUOTED and other[1] == speaker
                   for other in neighbours):
                speaker = NARRATOR
                instruct = None
        narrated.append((span, speaker, instruct))
    return narrated


def _absorb_whitespace_groups(groups):
    """Fold whitespace-only groups into a neighbour, keeping bytes identical.

    The paragraph break between two speakers' lines is its own unquoted span,
    so it can resolve to a group whose text is just "\\n\\n". Emitted as an
    entry it is an unspeakable NARRATOR line that the editor UI can never
    finish rendering -- hundreds per book. It belongs on the PRECEDING entry
    (or the following one when it comes first), which changes no bytes.

    The absorbing group keeps its own kind. That matters: it is what stops the
    second _merge_same_speaker pass from using an absorbed paragraph break as a
    bridge between a character's quotation and the narration after it.
    """
    absorbed = []
    for group in groups:
        if not group[1].strip() and absorbed:
            absorbed[-1][1] += group[1]
        else:
            absorbed.append(group)

    # A leading whitespace-only group has no predecessor: push it forward.
    while len(absorbed) > 1 and not absorbed[0][1].strip():
        absorbed[1][1] = absorbed[0][1] + absorbed[1][1]
        del absorbed[0]

    return absorbed


def build_entries(resolved, source):
    """Merge consecutive same-speaker spans into verbatim script entries.

    Text is the exact concatenation of the group's source slices: quotes,
    attribution tags and whitespace all survive untouched. No entry is ever
    whitespace-only (see _absorb_whitespace_groups).

    Merging is quoted/unquoted aware so that no character-voiced entry ends up
    holding both a quotation and the narration around it -- see
    _merge_same_speaker and _narrate_attribution_spans for the evidence. Only
    speaker labels move; not one byte of text changes.
    """
    groups = _merge_same_speaker(
        [[speaker, span.text(source), instruct, span.kind]
         for span, speaker, instruct in _narrate_attribution_spans(resolved)]
    )
    # Absorbing a whitespace group can make two same-speaker groups adjacent,
    # so merge once more afterwards.
    groups = _merge_same_speaker(_absorb_whitespace_groups(groups))

    return [
        {
            "speaker": speaker,
            "text": text,
            "instruct": instruct or (
                DEFAULT_NARRATOR_INSTRUCT if speaker == NARRATOR
                else DEFAULT_CHARACTER_INSTRUCT
            ),
        }
        for speaker, text, instruct, _kind in groups
    ]


def _assert_chunk_verbatim(entries, chunk, chunk_num):
    """Byte-identity check. A mismatch is a bug in reassembly -- raise, never warn."""
    rebuilt = "".join(entry["text"] for entry in entries)
    if rebuilt == chunk:
        return

    offset = 0
    limit = min(len(rebuilt), len(chunk))
    while offset < limit and rebuilt[offset] == chunk[offset]:
        offset += 1

    raise AssertionError(
        f"VERBATIM INVARIANT VIOLATED on chunk {chunk_num}: reassembled text "
        f"({len(rebuilt)} chars) differs from source chunk ({len(chunk)} chars) "
        f"at offset {offset}.\n"
        f"  source   : {chunk[offset:offset + 80]!r}\n"
        f"  rebuilt  : {rebuilt[offset:offset + 80]!r}"
    )


# CP1252-as-UTF8 mojibake repair table; see fix_mojibake() for why every
# literal is written as an escape. Each key is 3 characters: "\u00e2\u20ac"
# (UTF-8 bytes E2 80 read as CP1252) plus whatever CP1252 maps the third
# byte to.
MOJIBAKE_REPLACEMENTS = {
    "\u00e2\u20ac\u2122": "\u2019",   # 0x99 -> right single quote
    "\u00e2\u20ac\u02dc": "\u2018",   # 0x98 -> left single quote
    "\u00e2\u20ac\u0153": "\u201c",   # 0x9c -> left double quote
    "\u00e2\u20ac\u009d": "\u201d",   # 0x9d is undefined in CP1252
    "\u00e2\u20ac?": "\u201d",         # ...and is sometimes rendered "?"
    "\u00e2\u20ac\u201d": "\u2014",   # 0x94 -> em dash
    "\u00e2\u20ac\u201c": "\u2013",   # 0x93 -> en dash
    "\u00e2\u20ac\u00a6": "\u2026",   # 0xa6 -> ellipsis
}


def fix_mojibake(text):
    """Fix common mojibake characters resulting from CP1252-as-UTF8.

    Every key and value in MOJIBAKE_REPLACEMENTS is written as an explicit
    \\u escape. That is not stylistic: the literals are themselves mojibake,
    so any editor, terminal or copy-paste that re-encodes this file silently
    rewrites them -- which had already happened twice here:

      * the right-single-quote entry's VALUE had decayed to three ASCII
        apostrophes, which Python read as the start of a triple-quoted
        string. It swallowed its own comment AND the whole next entry, so
        the left-single-quote mapping did not exist at all; and
      * the em-dash and en-dash KEYS had both decayed to the same ASCII
        double-quote byte, making them duplicate dict keys -- 7 literal
        entries collapsed to 6, and the en-dash mapping shadowed the em-dash
        one, so em-dash mojibake was never repaired.

    Escapes cannot decay this way, and a test asserts the dict holds exactly
    one entry per literal.

    Each key is what a UTF-8 byte sequence looks like decoded as CP1252:
    U+2014 EM DASH is E2 80 94, and CP1252 maps 0x94 to U+201D, hence
    "\\u00e2\\u20ac\\u201d".
    """
    for bad, good in MOJIBAKE_REPLACEMENTS.items():
        text = text.replace(bad, good)

    return text


def _split_keeping_separators(text, separator_pattern):
    """Split `text` on `separator_pattern`, keeping each separator attached to
    the piece that PRECEDES it, so that ``"".join(pieces) == text`` exactly.

    Lossless splitting is the whole point: the previous implementation used a
    plain ``re.split`` + ``strip()``, which discarded the separator entirely on
    the chunk-boundary path (the "\\n\\n" was only re-inserted when a paragraph
    was appended to a chunk already in progress). That made the concatenation of
    all chunks -- and therefore annotated_script.json -- differ from the source
    file by one paragraph break per chunk seam. No prose was lost (every dropped
    character was whitespace) but the paragraph-break signal at each seam was,
    which the TTS layer hears as a missing pause.
    """
    pieces = []
    pos = 0
    for match in re.finditer(separator_pattern, text):
        # A zero-width match cannot carry text forward; skip it.
        if match.end() == pos:
            continue
        pieces.append(text[pos:match.end()])
        pos = match.end()
    if pos < len(text):
        pieces.append(text[pos:])
    return pieces


def split_into_chunks(text, max_size=3000):
    """Split text into chunks at paragraph/sentence boundaries.

    LOSSLESS: ``"".join(split_into_chunks(t)) == t`` for any input, including
    leading/trailing whitespace, whitespace-only paragraphs and NBSPs. Every
    character of the source ends up in exactly one chunk, in order; separators
    ride along at the end of the paragraph they follow. Enforced by
    test_api.py::test_chunking_is_byte_lossless_over_source.
    """
    # Paragraphs, each carrying its own trailing blank-line separator.
    paragraphs = _split_keeping_separators(text, r'\n\s*\n')

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if current_chunk and len(current_chunk) + len(para) > max_size:
            chunks.append(current_chunk)
            current_chunk = ""

        if len(para) > max_size:
            # Oversized paragraph: fall back to sentence boundaries, again
            # keeping the inter-sentence whitespace attached.
            for sentence in _split_keeping_separators(para, r'(?<=[.!?])\s+'):
                if current_chunk and len(current_chunk) + len(sentence) > max_size:
                    chunks.append(current_chunk)
                    current_chunk = ""
                current_chunk += sentence
        else:
            current_chunk += para

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def build_context(chunk_num, total_chunks, previous_entries=None,
                  max_roster_names=None):
    """Positional note + speaker roster + a short tail of previous entries.

    Context is INPUT, not output: a truncated snippet of each recent entry is
    enough for tone/roster continuity and keeps the prompt cheap now that
    narration entries can be long.

    The roster block is capped at ``max_roster_names`` (default
    MAX_CONTEXT_ROSTER_NAMES) most-recently-spoken names, and says so when it
    truncates, so the model never infers the list is the book's full cast.
    """
    if max_roster_names is None:
        max_roster_names = MAX_CONTEXT_ROSTER_NAMES
    context_parts = []

    if chunk_num == 1:
        context_parts.append("(Beginning of text)")
    elif chunk_num == total_chunks:
        context_parts.append("(End of text)")
    else:
        context_parts.append(f"(Part {chunk_num} of {total_chunks})")

    if previous_entries and len(previous_entries) > 0:
        # Build character roster for name consistency across chunks. Walk
        # backwards and STOP at the cap, so this stays O(cap) rather than
        # rescanning the whole book every chunk; the names kept are therefore
        # the most recently spoken ones (see MAX_CONTEXT_ROSTER_NAMES).
        limit = max_roster_names if max_roster_names is not None else MAX_CONTEXT_ROSTER_NAMES
        recent = []
        seen = set()
        truncated_roster = False
        for entry in reversed(previous_entries):
            speaker = entry.get("speaker", "")
            if not speaker or speaker == NARRATOR or speaker in seen:
                continue
            if limit is not None and limit >= 0 and len(recent) >= limit:
                # There is at least one older name we are not showing. Stopping
                # here is why the header says "most recent N" and not
                # "N of M" -- the total is deliberately not computed.
                truncated_roster = True
                break
            seen.add(speaker)
            recent.append(speaker)

        characters_seen = sorted(recent)

        if characters_seen:
            # Say when the list is partial, so the model does not infer it is
            # the book's complete cast.
            header = (f"Characters in this book (most recent {len(characters_seen)})"
                      if truncated_roster else "Characters in this book")
            context_parts.append(f"{header}: {', '.join(characters_seen)}")

        # Include last few entries so the model can maintain style and tone continuity
        tail = previous_entries[-3:]
        context_parts.append("\nPrevious section ended with:")
        for entry in tail:
            text = entry.get("text", "")
            if len(text) > CONTEXT_SNIPPET_CHARS:
                text = text[:CONTEXT_SNIPPET_CHARS].rstrip() + "..."
            context_parts.append(json.dumps(
                {"speaker": entry.get("speaker", NARRATOR), "text": text},
                ensure_ascii=False,
            ))

    return "\n".join(context_parts)


def process_chunk(client, model_name, chunk, chunk_num, total_chunks, previous_entries=None, max_retries=2, system_prompt=None, user_prompt_template=None, max_tokens=4096, temperature=0.6, top_p=0.8, top_k=0, min_p=0, presence_penalty=0.0, banned_tokens=None, roster=None, max_context_roster_names=None, num_ctx=None, attest_window=None, require_attested=False, reasoning_effort=None, source_words=None, check_tags=False):
    """Classify one chunk's spans and rebuild its script entries verbatim.

    Returns ``(entries, stats)``. ``stats`` reports span counts and whether the
    chunk degraded; ``entries`` ALWAYS reproduces the chunk byte-for-byte, even
    when the LLM failed outright (the whole chunk then becomes NARRATOR).

    ``roster`` (optional, trailing keyword so existing callers/tests are
    unaffected) is a roster index of speaker names established by previous
    chunks; see resolve_span_labels.

    ``num_ctx``, when set, asks the server for that serving context window via
    ``extra_body={"options": {...}}`` (Ollama's documented option bag). It is
    BEST EFFORT: a server that does not understand the key ignores it, which
    is why the silent-truncation detection below is the part that is actually
    guaranteed to work.
    """
    # Use provided prompts or fall back to defaults
    sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    usr_template = user_prompt_template or DEFAULT_USER_PROMPT

    spans = tokenize(chunk)
    validate_spans(spans, chunk)

    if not spans:
        return [], {"spans": 0, "labelled": 0, "fallback": 0, "whitespace": 0,
                    "discarded": 0, "role_missing": 0, "placeholder_rejected": 0,
                    "degraded": False, "reason": None, "recovery": None,
                    "prompt_chars": 0, "prompt_tokens": None,
                    "prompt_truncation_events": 0}

    if not visible_spans(spans, chunk):
        # Nothing audible to classify (a chunk of pure whitespace). Resolve
        # locally rather than spending an LLM call on an empty span listing.
        resolved, stats = resolve_span_labels(spans, None, source=chunk)
        entries = build_entries(resolved, chunk)
        _assert_chunk_verbatim(entries, chunk, chunk_num)
        stats.update({"spans": len(spans), "degraded": False,
                      "reason": None, "recovery": None,
                      "prompt_chars": 0, "prompt_tokens": None,
                      "prompt_truncation_events": 0})
        return entries, stats

    context = build_context(chunk_num, total_chunks, previous_entries,
                            max_roster_names=max_context_roster_names)
    base_prompt = usr_template.format(context=context, chunk=build_span_payload(spans, chunk))

    # Labels accumulate ACROSS attempts, filling blanks only (see _merge_label),
    # so a retry can only ever improve the result. Live runs showed the model
    # closing the array early -- dropping a contiguous SUFFIX of ids with
    # finish_reason=stop, nowhere near max_tokens -- and separately inventing
    # role values ("thought"). Both are recoverable by re-asking for the gap.
    merged_labels = {}
    recovery = None
    truncated = False
    reason = None
    retry_nudge = ""
    prompt_chars = 0
    prompt_tokens = None
    prompt_truncation_events = 0

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": base_prompt + retry_nudge}
                ],
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                max_tokens=max_tokens,
                extra_body={
                    k: v for k, v in {
                        "top_k": top_k if top_k else None,
                        "min_p": min_p if min_p else None,
                        "banned_tokens": banned_tokens if banned_tokens else None,
                        # Best-effort context-window request. Omitted entirely
                        # when unconfigured, so a request is byte-identical to
                        # before for anyone who does not set generation.num_ctx.
                        "options": {"num_ctx": num_ctx} if num_ctx else None,
                        # Best-effort reasoning suppression, same omit-when-unset
                        # rule. A "thinking" model can spend its ENTIRE
                        # completion budget on reasoning and return empty
                        # content -- see the empty-content check below for the
                        # measured failure. reasoning_effort is a standard
                        # OpenAI field, so a backend that does not implement it
                        # ignores it rather than erroring.
                        #
                        # Sent via extra_body rather than as a named argument so
                        # the value reaches the wire untouched regardless of
                        # what the installed SDK's type hints allow.
                        "reasoning_effort": reasoning_effort or None,
                    }.items() if v is not None
                }
            )

            choice = response.choices[0]
            # `or ""`: a reasoning model returns content=None (not "") when it
            # never leaves the reasoning phase, which would crash .strip().
            text = (choice.message.content or "").strip()
            # Ollama's OpenAI-compat layer puts reasoning in message.reasoning.
            # Read for DIAGNOSTICS ONLY and never parsed for labels: the
            # docstring at strip_thinking_tags records a live run where a
            # <think> block salvaged 96 phantom span ids, so reasoning text is
            # exactly the id-shaped noise that must not reach extract_labels.
            reasoning_text = getattr(choice.message, "reasoning", None) or ""
            finish_reason = choice.finish_reason
            usage = getattr(response, 'usage', None)

            # Size of what we actually sent, for the silent-truncation
            # tripwire and for the forensic log.
            prompt_chars = len(sys_prompt) + len(base_prompt) + len(retry_nudge)
            prompt_tokens = getattr(usage, 'prompt_tokens', None) if usage else None

            # Log raw response for debugging
            log_dir = llm_log_dir()
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "llm_responses.log")
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(f"\n{'='*80}\n")
                lf.write(f"CHUNK {chunk_num}/{total_chunks} | attempt {attempt + 1} | finish_reason={finish_reason}\n")
                if usage:
                    lf.write(f"tokens: prompt={getattr(usage, 'prompt_tokens', '?')} completion={getattr(usage, 'completion_tokens', '?')}\n")
                if reasoning_text:
                    # Without this line the catastrophic case is a 166-byte log
                    # block with no indication that 4096 tokens were spent
                    # reasoning -- effectively undiagnosable.
                    lf.write(f"reasoning: {len(reasoning_text)} chars "
                             f"(not parsed; diagnostics only)\n")
                    lf.write(f"reasoning_head: {reasoning_text[:300]!r}\n")
                lf.write(f"{'─'*80}\n")
                lf.write(text)
                lf.write(f"\n{'='*80}\n")

            print(f"  finish_reason={finish_reason}", end="")
            if usage:
                print(f" | tokens: prompt={getattr(usage, 'prompt_tokens', '?')} completion={getattr(usage, 'completion_tokens', '?')}", end="")
            print()

            if looks_silently_truncated(prompt_chars, prompt_tokens):
                prompt_truncation_events += 1
                _print_silent_truncation_warning(
                    chunk_num, total_chunks, prompt_chars, prompt_tokens)

            truncated = finish_reason == "length"
            if truncated and not text:
                # The whole budget went to reasoning and NOTHING was emitted, so
                # every span on this chunk is about to be narrated. Distinct from
                # the ordinary truncation message below, which blames max_tokens
                # and sends the operator to raise a setting that (a) is clamped
                # to num_ctx minus the prompt anyway and (b) makes the request
                # slow enough to hit llm.timeout. Measured: this model spent
                # 4096 completion tokens and returned zero labels three times.
                print(f"  {'!' * 60}")
                print(f"  MODEL RETURNED NO CONTENT on chunk {chunk_num}/{total_chunks}: "
                      f"finish_reason=length with an empty message body"
                      + (f", and {len(reasoning_text)} chars of reasoning."
                         if reasoning_text else "."))
                print("  The model spent its entire completion budget 'thinking' and never")
                print("  emitted the label array. Raising generation.max_tokens does NOT fix")
                print("  this. Suppress reasoning instead:")
                print('    - set llm.reasoning_effort to "none" in config.json (verified to')
                print("      remove reasoning entirely on Ollama's /v1 endpoint)")
                print("    - or use a non-reasoning model for annotation")
                print("  NOTE: an Ollama-native \"think\": false is silently IGNORED on /v1.")
                print(f"  {'!' * 60}")
            elif truncated:
                print(f"  WARNING: Response was truncated (hit max_tokens={max_tokens}). "
                      "Unlabelled spans will fall back to NARRATOR.")

        except Exception as e:
            print(f"Error calling LLM API (attempt {attempt + 1}): {e}")
            if attempt < max_retries:
                continue
            reason = f"LLM call failed on all {max_retries + 1} attempts ({e})"
            break

        # Recover labels from whatever shape the model actually produced.
        labels, mode = extract_labels(text)

        if labels:
            recovery = _worst_mode(recovery, mode)
            if mode == LABEL_MODE_OBJECT:
                print(f"  Note: model returned an id-keyed JSON object instead of an array; "
                      f"recovered all {len(labels)} label(s) from it")
            elif mode == LABEL_MODE_SALVAGE:
                print(f"  Regex-salvaged {len(labels)} label(s) from a malformed/truncated response")
            elif mode == LABEL_MODE_MARKDOWN:
                print(f"  Recovered {len(labels)} label(s) from markdown blocks "
                      "(model ignored the JSON output contract)")

            before = len(merged_labels)
            for label in labels:
                label_id = _label_id(label)
                if label_id is not None:
                    merged_labels[label_id] = _merge_label(merged_labels.get(label_id), label)
            if attempt > 0:
                print(f"  Retry recovered {len(merged_labels) - before} new label(s)")
        else:
            print(f"Warning: Could not recover labels from chunk {chunk_num} response (attempt {attempt + 1})")
            print(f"Response preview: {text[:300]}...")
            if attempt < max_retries:
                print("Retrying...")
                continue
            if not merged_labels:
                reason = "no usable labels recovered from LLM response"
            break  # keep whatever earlier attempts recovered

        missing_ids, bad_role_ids, no_speaker_ids = _incomplete_span_ids(
            spans, merged_labels, source=chunk)
        unattested = (
            _unattested_speaker_ids(spans, merged_labels, source=chunk,
                                    roster=roster, attest_window=attest_window,
                                    source_words=source_words)
            if require_attested else []
        )
        contradicted = (
            _contradicted_speaker_ids(spans, merged_labels, source=chunk,
                                      roster=roster)
            if check_tags else []
        )

        if (not missing_ids and not bad_role_ids and not no_speaker_ids
                and not unattested and not contradicted):
            if attempt > 0:
                print(f"  Succeeded on retry {attempt + 1} -- all {len(spans)} spans labelled")
            break

        if truncated:
            # Re-rolling at the same max_tokens genuinely is pointless: accept
            # the partial labels and let the rest fall back to NARRATOR.
            break

        if attempt < max_retries:
            gaps = []
            if missing_ids:
                gaps.append(f"{len(missing_ids)} span(s) unlabelled")
            if bad_role_ids:
                gaps.append(f"{len(bad_role_ids)} label(s) with an unusable role")
            if no_speaker_ids:
                gaps.append(f"{len(no_speaker_ids)} dialogue label(s) with no speaker")
            if contradicted:
                gaps.append(f"{len(contradicted)} speaker label(s) contradicted by "
                            "their own attribution tag "
                            f"({', '.join(f'{s}/{t}' for _, s, t in contradicted[:3])}"
                            f"{', ...' if len(contradicted) > 3 else ''})")
            if unattested:
                gaps.append(f"{len(unattested)} speaker name(s) absent from the text "
                            f"({', '.join(name for _, name in unattested[:3])}"
                            f"{', ...' if len(unattested) > 3 else ''})")
            print(f"  {' and '.join(gaps)} -- retrying (attempt {attempt + 2}"
                  f"/{max_retries + 1}), naming the gaps")
            spelling_hints = {
                name: near_spellings(
                    name, [attest_window] if attest_window else [], roster or {})
                for _, name in unattested
            }
            retry_nudge = _retry_nudge(missing_ids, bad_role_ids, unattested,
                                       no_speaker_ids=no_speaker_ids,
                                       spelling_hints=spelling_hints,
                                       contradicted=contradicted)
            # Clear the refused speakers so the retry's answer can actually
            # land. _merge_label deliberately never overwrites a usable field
            # -- that is what makes "a retry can never make things worse" a
            # structural property -- and a misread name IS a usable string, so
            # without this the corrected spelling would be discarded and the
            # nudge would be decorative. Dropping it is consistent with that
            # property rather than an exception to it: our own gate has just
            # ruled that this value carries no usable information. If the retry
            # supplies nothing, the field stays blank and the span is narrated
            # and counted (see dialogue_without_speaker), so the outcome is
            # never quieter than a plain rejection.
            # Same reason for the tag contradictions, with one difference worth
            # stating: our check has NOT ruled the value unusable, only that it
            # disagrees with the adjacent tag. Clearing it is still what makes
            # the retry able to answer at all (_merge_label never overwrites a
            # usable field), and the nudge tells the model to repeat its answer
            # if it was right -- which re-fills the field. The residual cost of
            # a FALSE detection is therefore one span narrated (loud, counted
            # as dialogue_without_speaker) when the retry supplies nothing at
            # all; prose is untouched either way.
            for span_id, _, _ in contradicted:
                label = merged_labels.get(span_id)
                if isinstance(label, dict):
                    label.pop("speaker", None)
            for span_id, _ in unattested:
                label = merged_labels.get(span_id)
                if isinstance(label, dict):
                    label.pop("speaker", None)
            continue

        break

    # Reassemble regardless of what came back. Unlabelled spans -> NARRATOR,
    # so a failure costs labels, never prose. Phantom ids stay in the merged map
    # on purpose: resolve_span_labels discards and counts them (once each).
    resolved, stats = resolve_span_labels(
        spans, list(merged_labels.values()), source=chunk, roster=roster,
        attest_window=attest_window, require_attested=require_attested,
        source_words=source_words, check_tags=check_tags)
    _report_repairs(chunk_num, total_chunks, stats.get("repairs"))
    entries = build_entries(resolved, chunk)
    _assert_chunk_verbatim(entries, chunk, chunk_num)

    if truncated and reason is None:
        reason = "response truncated at max_tokens"
    if reason is None and recovery == LABEL_MODE_MARKDOWN:
        # Labels are usable, but the model ignored the output contract entirely.
        # Surface it: the next run may not be so structured.
        reason = "labels recovered via markdown salvage (model returned no JSON)"
    if reason is None and stats["fallback"] > 0:
        reason = "LLM did not label every span"
    if reason is None and stats["unattested_rejected"] > 0:
        # Loud by design (contract 7). A rejection does NOT raise `fallback`
        # -- the span was labelled, the speaker was refused -- so without this
        # the run would exit 0 with silently narrated dialogue.
        reason = (f"{stats['unattested_rejected']} speaker label(s) named a "
                  "character the source text does not support")
    if reason is None and stats.get("tag_contradictions", 0) > 0:
        # The retry did not resolve it, so the script will say a name the
        # neighbouring attribution tag disagrees with. Nothing was rewritten;
        # this is the operator's pointer to where to look (contract 7).
        detail = ", ".join(f"span {span_id}: {speaker} vs tag {tagged}"
                           for span_id, speaker, tagged in stats["contradictions"][:5])
        reason = (f"{stats['tag_contradictions']} speaker label(s) contradicted "
                  f"by the attribution tag next to them ({detail})")
    if reason is None and stats["dialogue_without_speaker"] > 0:
        reason = (f"{stats['dialogue_without_speaker']} label(s) claimed "
                  "role=dialogue with no usable speaker name")
    if reason is None and stats["role_missing"] > 0:
        # A model that labels every span but omits "role" produces a fully
        # narrated book. Prose is intact, every voice is gone -- exactly the
        # silent failure this stage exists to eliminate, so it degrades.
        reason = (f"{stats['role_missing']} label(s) named a speaker without "
                  "role=dialogue")

    stats["spans"] = len(spans)
    stats["prompt_chars"] = prompt_chars
    stats["prompt_tokens"] = prompt_tokens
    stats["prompt_truncation_events"] = prompt_truncation_events
    stats["recovery"] = recovery
    stats["degraded"] = reason is not None
    stats["reason"] = reason

    if stats["discarded"]:
        print(f"  Discarded {stats['discarded']} label(s) referring to nonexistent span ids")
    if stats.get("unknown_key_labels"):
        print(f"  Ignored unknown key(s) on {stats['unknown_key_labels']} label(s): "
              + ", ".join(f'"{key}"' for key in stats["unknown_keys"])
              + " -- not in the schema (id, speaker, role, instruct), so their "
              "values were discarded (a misspelled key costs that field)")
    if stats.get("text_key_labels"):
        print(f"  {stats['text_key_labels']} label(s) carried a \"text\" key -- IGNORED. "
              "The model must return labels only; entry text is always taken "
              "verbatim from the source")
    if stats["role_missing"]:
        print(f"  {stats['role_missing']} label(s) named a speaker without role=\"dialogue\" "
              "-- narrated instead (schema violation by the model)")
    if stats["placeholder_rejected"]:
        print(f"  Rejected {stats['placeholder_rejected']} invented placeholder speaker "
              "label(s) (e.g. \"SPEAKER 1\") -- narrated instead")
    if stats["unattested_rejected"]:
        print(f"  Rejected {stats['unattested_rejected']} speaker label(s) whose name does "
              "not appear in the text near their own lines -- narrated instead")
    for original in stats.get("refused_repairs") or []:
        print(f"  REFUSED to repair speaker label \"{original}\": a spelling repair was "
              "available, but the narration right after this line attributes it to "
              "another established character -- narrated instead")
    if stats.get("tag_contradictions"):
        for span_id, speaker, tagged in stats["contradictions"]:
            print(f"  CONTRADICTED speaker label on span {span_id}: labelled "
                  f"\"{speaker}\", but the narration right after it attributes "
                  f"the line to \"{tagged}\" -- label KEPT as the model gave it, "
                  "reported for review")
    if stats["unverifiable_accepted"]:
        print(f"  Accepted {stats['unverifiable_accepted']} speaker label(s) the source "
              "cannot confirm or refute (attestation does not apply to this text)")

    if stats["degraded"]:
        print(f"  {'!' * 60}")
        print(f"  DEGRADED chunk {chunk_num}/{total_chunks}: {reason}.")
        print(f"  {stats['labelled']}/{len(spans)} spans labelled, "
              f"{stats['fallback']} fell back to NARRATOR. "
              "All prose is preserved verbatim; only voice assignment is lost.")
        print(f"  {'!' * 60}")

    return entries, stats

def _write_script_output(all_entries):
    """Write annotated_script.json and clear stale chunks.json."""
    output_path = os.path.join("..", "annotated_script.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_entries, f, indent=2, ensure_ascii=False)

    chunks_path = os.path.join("..", "chunks.json")
    if os.path.exists(chunks_path):
        os.remove(chunks_path)
        print("Cleared old chunks.json")

    speakers = set(entry.get("speaker") or entry.get("type") or "UNKNOWN" for entry in all_entries)
    print(f"\nGenerated {len(all_entries)} script entries")
    print(f"Speakers found: {', '.join(sorted(speakers))}")
    print(f"Output saved to: {output_path}")


def run_single_speaker(book_content, speaker_name, instruct):
    """Bypass the LLM and emit one entry per text segment, all attributed
    to a single speaker. Used for first-person memoirs, non-fiction, etc.,
    where character detection is unnecessary."""
    # Canonicalize: the default "Narrator" fails every `!= "NARRATOR"` literal
    # comparison downstream (review_script.py) and misses voice-config lookups
    # keyed on the canonical form.
    canonical_speaker = canonicalize(speaker_name) or NARRATOR
    if canonical_speaker != speaker_name:
        print(f"Canonicalized speaker name: {speaker_name!r} -> {canonical_speaker!r}")

    segments = split_into_chunks(book_content, max_size=SINGLE_SPEAKER_MAX_CHARS)
    print(f"Split into {len(segments)} narration segments at paragraph/sentence boundaries")

    entries = [
        {"speaker": canonical_speaker, "text": segment, "instruct": instruct}
        for segment in segments
    ]

    if not entries:
        print("Error: No script entries generated (input text is empty?)")
        sys.exit(1)

    _write_script_output(entries)


def main():
    parser = argparse.ArgumentParser(description="Generate annotated audiobook script.")
    parser.add_argument("input_file_path", help="Path to the input text/markdown/EPUB text file.")
    parser.add_argument("--single-speaker", action="store_true",
                        help="Skip LLM and attribute the whole text to one speaker.")
    parser.add_argument("--speaker-name", default="Narrator",
                        help="Speaker name used in single-speaker mode (default: Narrator).")
    parser.add_argument("--instruct", default="Neutral narration.",
                        help="Voice direction used in single-speaker mode.")
    args = parser.parse_args()

    input_file_path = args.input_file_path
    print(f"Processing book from: {input_file_path}")

    if not os.path.exists(input_file_path):
        print(f"Error: Input file not found: {input_file_path}")
        sys.exit(1)

    with open(input_file_path, 'r', encoding='utf-8') as f:
        book_content = f.read()

    # Fix encoding artifacts
    book_content = fix_mojibake(book_content)

    print(f"Read {len(book_content)} characters")

    if args.single_speaker:
        print(f"Single-speaker mode: attributing all narration to '{args.speaker_name}'")
        run_single_speaker(book_content, args.speaker_name, args.instruct)
        return

    # Load LLM config
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load config.json: {e}")
    else:
        print("Warning: config.json not found. Using defaults.")

    llm_config = config.get("llm", {})
    base_url = llm_config.get("base_url", "http://localhost:11434/v1")
    api_key = llm_config.get("api_key", "local")
    model_name = llm_config.get("model_name", "richardyoung/qwen3-14b-abliterated:Q8_0")
    timeout = llm_config.get("timeout")
    # Optional reasoning suppression for "thinking" models. Unset = send
    # nothing, so the request is unchanged for existing users. "none" is the
    # value verified to work against Ollama's /v1 endpoint.
    reasoning_effort = llm_config.get("reasoning_effort")

    # Load custom prompts or use defaults. Custom prompts are only honoured
    # when they target the current span-label schema (see select_prompt).
    prompts_config = config.get("prompts", {})
    system_prompt = select_prompt(
        prompts_config.get("system_prompt"), DEFAULT_SYSTEM_PROMPT, "prompts.system_prompt")
    user_prompt_template = select_prompt(
        prompts_config.get("user_prompt"), DEFAULT_USER_PROMPT, "prompts.user_prompt")

    # Load generation settings
    generation_config = config.get("generation", {})
    chunk_size = generation_config.get("chunk_size", 3000)
    max_tokens = generation_config.get("max_tokens", 4096)
    temperature = generation_config.get("temperature", 0.6)
    top_p = generation_config.get("top_p", 0.8)
    top_k = generation_config.get("top_k", 0)
    min_p = generation_config.get("min_p", 0)
    presence_penalty = generation_config.get("presence_penalty", 0.0)
    banned_tokens = generation_config.get("banned_tokens", [])
    max_context_roster_names = generation_config.get(
        "max_context_roster_names", MAX_CONTEXT_ROSTER_NAMES)
    # Optional, best-effort serving-context-window request. Left as None when
    # unset so the outgoing request is unchanged for existing users.
    num_ctx = generation_config.get("num_ctx")
    # Attestation gate. DEFAULT ON: it refuses a label the book does not
    # support near its own lines, which is what stops the classifier inventing
    # a character out of words the prose supplies. The cost is real and worth
    # knowing: a real speaker named with a descriptor the book does not use
    # nearby is refused, and those lines are NARRATED -- 1 span in one run of a
    # novel, 21 in another, the variance being the model's rather than the
    # gate's. Rejections are named by chunk and span. Set false to accept every
    # label the model offers, fabricated ones included. Measure a book first
    # with tools/verify_attestation.py, which reports the would-be rejection
    # rate without running the model at all. This default must match
    # GenerationConfig in app.py -- the UI writes config.json but this
    # subprocess reads it directly, so a disagreement silently changes
    # behaviour depending on whether the key was ever saved.
    require_attested = bool(generation_config.get("require_attested_speakers", True))
    # Attribution-tag agreement check. DEFAULT ON: the book names the speaker
    # in the narration beside the line and nothing used to read it -- 5.4% of
    # checkable pairs carried a label their own tag contradicted, on a run that
    # reported success; with this on, 0.90%. It costs retries (84 -> 143 on one
    # book) and reports survivors by chunk and span instead of passing them
    # silently. Independent of the attestation gate -- a contradicted label is
    # usually a perfectly attested name, just the wrong one -- so it keeps its
    # own key. This default must match GenerationConfig in app.py: the UI writes
    # config.json but this subprocess reads it directly, so a disagreement here
    # silently changes behaviour depending on whether the key was ever saved.
    #
    # The English speech-verb lexicon it needs is NOT config-exposed. It is a
    # LANGUAGE property, not a per-book tunable, and a per-book verb list is
    # precisely the book-fitting this pipeline avoids; a wrong entry there
    # would produce false accusations, which is the one failure this check must
    # not have. A non-English book needs no setting: no verb matches, so the
    # check finds nothing and costs nothing.
    check_tags = bool(generation_config.get("check_attribution_tags", True))
    # How much preceding source text joins the current chunk as the
    # attestation window. Defaults to one chunk, so a name introduced in the
    # sentence before the chunk boundary still attests. The failure direction
    # is one-sided and safe: too small means more NARRATOR fallbacks (loud,
    # prose intact), too large approaches ungated behaviour.
    # `or chunk_size` (not a .get default) so an explicit null in config.json
    # -- which is what an unset Optional field round-trips as -- still falls
    # back rather than becoming a None slice bound.
    attestation_lookback_chars = (
        generation_config.get("attestation_lookback_chars") or chunk_size)

    print(f"Connecting to: {base_url}")
    print(f"Using model: {model_name}")
    print(f"Chunk size: {chunk_size} chars, Max tokens: {max_tokens}")
    if num_ctx:
        print(f"Requesting server context window (num_ctx): {num_ctx} "
              "(best effort -- ignored by servers that do not support it)")
    if banned_tokens:
        print(f"Banned tokens: {banned_tokens}")

    # Create OpenAI client with custom base URL
    # llm.timeout (seconds) overrides the OpenAI SDK's 600s default. Left out
    # of the call entirely when unset, so the constructed client is identical
    # to before for anyone who does not set it.
    #
    # This matters on slow local inference: a completion that needs longer than
    # the ceiling is killed after producing nothing usable, and the run then
    # retries it and is killed again. Measured on one local setup: 9.5 tok/s,
    # so generation.max_tokens=8192 implies ~861s and a guaranteed timeout at
    # the 600s default. Lowering max_tokens is the better first move -- this is
    # for machines where even that is not enough.
    client_kwargs = {"base_url": base_url, "api_key": api_key}
    if timeout:
        client_kwargs["timeout"] = timeout
        print(f"LLM request timeout: {timeout}s (llm.timeout)")
    client = OpenAI(**client_kwargs)

    # Split into chunks at natural boundaries
    chunks = split_into_chunks(book_content, max_size=chunk_size)
    total_chunks = len(chunks)

    print(f"Split into {total_chunks} chunks at paragraph/sentence boundaries")

    # Book-wide word index, built ONCE and only when the gate is on. It is what
    # lets speaker_canon.repair_speaker() ask "does the author ever spell it
    # this way?" -- a question a per-chunk window cannot answer, and the guard
    # that keeps repair off real names that simply are not in their own window.
    source_words = source_word_index(book_content) if require_attested else None
    if require_attested:
        print(f"Attestation gate ON; book word index: {len(source_words)} distinct words")

    all_entries = []
    # Roster index of speaker spellings established so far, threaded into every
    # chunk so a later spelling variant is snapped onto the established name
    # (speaker_canon.remember_in_roster): of two spellings that differ only in
    # their boundary marks, the more-punctuated one wins, order-independently.
    roster_index = {}
    degraded_chunks = []
    total_spans = 0
    total_fallback = 0
    total_placeholders = 0
    total_unattested = 0
    total_unverifiable = 0
    source_word_names = {}
    total_repaired = 0
    total_repairs_refused = 0
    total_contradicted = 0
    contradiction_examples = []
    repair_mappings = {}
    truncation_events = 0
    prompt_samples = []

    for i, chunk in enumerate(chunks, 1):
        print(f"Processing chunk {i}/{total_chunks} ({len(chunk)} chars)...")

        # Attestation window: the chunk the model was shown, plus a tail of the
        # preceding chunk so a name introduced just before the boundary still
        # counts. Built from the chunks themselves rather than by re-reading
        # the source, so it is exactly the text generation saw -- no offsets to
        # keep in sync and no mojibake mismatch.
        attest_window = None
        if require_attested:
            lookback = chunks[i - 2][-attestation_lookback_chars:] if i > 1 else ""
            attest_window = lookback + chunk

        previous = all_entries if len(all_entries) > 0 else None
        entries, stats = process_chunk(
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
            banned_tokens=banned_tokens,
            roster=roster_index,
            max_context_roster_names=max_context_roster_names,
            num_ctx=num_ctx,
            attest_window=attest_window,
            require_attested=require_attested,
            reasoning_effort=reasoning_effort,
            source_words=source_words,
            check_tags=check_tags,
        )
        all_entries.extend(entries)
        for entry in entries:
            speaker = entry.get("speaker")
            if speaker and speaker != NARRATOR:
                remember_in_roster(roster_index, speaker)
        total_spans += stats["spans"]
        total_fallback += stats["fallback"]
        total_placeholders += stats.get("placeholder_rejected", 0)
        total_unattested += stats.get("unattested_rejected", 0)
        total_unverifiable += stats.get("unverifiable_accepted", 0)
        total_repaired += stats.get("speakers_repaired", 0)
        total_repairs_refused += stats.get("repairs_refused", 0)
        total_contradicted += stats.get("tag_contradictions", 0)
        contradiction_examples.extend(
            (i, speaker, tagged) for _, speaker, tagged in stats.get("contradictions") or [])
        for original, repaired in stats.get("repairs") or []:
            repair_mappings[(original, repaired)] = (
                repair_mappings.get((original, repaired), 0) + 1)
        truncation_events += stats.get("prompt_truncation_events", 0)
        if stats.get("prompt_tokens"):
            prompt_samples.append((stats.get("prompt_chars", 0), stats["prompt_tokens"]))
        if stats["degraded"]:
            degraded_chunks.append((i, stats["reason"]))

        print(f"  Got {len(entries)} entries from {stats['spans']} spans "
              f"({stats['labelled']} labelled, {stats['fallback']} narrator fallback)")

    if not all_entries:
        print("Error: No script entries generated")
        sys.exit(1)

    # Always write the output first: the audiobook is complete either way.
    _write_script_output(all_entries)

    print(f"\n{'=' * 60}")
    print("Generation summary")
    print(f"  Chunks processed:        {total_chunks}")
    print(f"  Chunks degraded:         {len(degraded_chunks)}")
    print(f"  Spans total:             {total_spans}")
    print(f"  Spans fallback-labelled: {total_fallback} (read by NARRATOR)")
    print(f"  Placeholder labels rejected: {total_placeholders}")
    if require_attested:
        print(f"  Unattested speakers rejected: {total_unattested} "
              "(name absent from the text near its own lines)")
        print(f"  Unverifiable speakers accepted: {total_unverifiable} "
              "(attestation does not apply to this text)")
        for name, count in sorted(source_word_names.items()):
            print(f"    - \"{name}\" ({count} span(s))")
        # Loud by design (contract 7): these labels say a name the model never
        # emitted. Not a degradation -- no voice was lost -- but an operator
        # must be able to see, and audit, every rewrite.
        print(f"  Speaker labels repaired:  {total_repaired} "
              "(refuted spelling folded onto an established name)")
        for (original, repaired), count in sorted(repair_mappings.items()):
            print(f"    - \"{original}\" -> \"{repaired}\" ({count} span(s))")
        print(f"  Speaker repairs refused:  {total_repairs_refused} "
              "(the adjacent attribution tag named someone else; span narrated "
              "and counted above as an unattested rejection)")
        if total_repaired:
            # Repairs are NOT folded into the degradation signal and do NOT
            # change the exit code. Contract 7's exit 3 means "spans fell back
            # to NARRATOR"; a repair keeps a character voice, and on a normal
            # book there are dozens of them, so making them exit 3 would make
            # exit 3 permanent and destroy the signal instead of sharpening it.
            # What the wrong ones cost is a rejection -- which IS a fallback and
            # DOES exit 3 -- via the tag veto above. The rest are visible here.
            print(f"  {'!' * 58}")
            print(f"  {total_repaired} speaker label(s) in this book say a name the model "
                  "never emitted.")
            print("  Each survived the spelling guards AND the attribution tag next to its")
            print("  own line, but neither is proof of WHO speaks a line with no tag at all.")
            print("  Audit the mappings above before rendering audio.")
            print(f"  {'!' * 58}")
    else:
        print("  Attestation gate:        off "
              "(generation.require_attested_speakers)")
    if check_tags:
        print(f"  Attribution-tag contradictions: {total_contradicted} "
              "(label kept; the tag next to the line names someone else)")
        for chunk_num, speaker, tagged in contradiction_examples[:10]:
            print(f"    - chunk {chunk_num}: labelled \"{speaker}\", tag says \"{tagged}\"")
        if len(contradiction_examples) > 10:
            print(f"    - (and {len(contradiction_examples) - 10} more)")
    else:
        print("  Attribution-tag check:   off "
              "(generation.check_attribution_tags)")
    print(f"  Prompt-truncation warnings: {truncation_events}")
    flatline = detect_flatlined_prompt_tokens(prompt_samples)
    if flatline:
        value, count, total = flatline
        print(f"  {'!' * 58}")
        print(f"  SERVER CONTEXT WINDOW SUSPECT: prompt_tokens was exactly {value} on "
              f"{count} of {total} calls,")
        print("  despite the prompts varying substantially in size -- the hallmark of a")
        print("  server silently truncating prompts to a fixed window. The book was")
        print("  classified from partial prompts. Raise the server's context length")
        print("  (e.g. OLLAMA_CONTEXT_LENGTH, or generation.num_ctx in config.json)")
        print("  and re-run for materially better speaker attribution.")
        print(f"  {'!' * 58}")
    if degraded_chunks:
        print("  Degradation events:")
        for chunk_num, reason in degraded_chunks:
            print(f"    - chunk {chunk_num}: {reason}")
    print(f"{'=' * 60}")

    if degraded_chunks:
        print("\nWARNING: the script is COMPLETE (no prose was lost -- every span was "
              "reassembled verbatim), but some spans could not be classified and are "
              "attributed to NARRATOR. Review the script before rendering audio.")
        # Exit 3 = "output written, but degraded". Distinct from 0 (clean) and
        # from 1 (nothing produced). app.py's run_process() logs a nonzero code
        # as "failed with return code 3", which is exactly the intent: silent
        # degradation is what this stage exists to eliminate.
        sys.exit(EXIT_DEGRADED)


if __name__ == '__main__':
    main()
