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
"""

import os
import sys
import json
import re
import argparse
from openai import OpenAI
from default_prompts import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT
from span_tokenizer import tokenize, validate_spans
from speaker_canon import (
    UNATTESTED,
    UNVERIFIABLE,
    attest_speaker,
    canonicalize,
    remember_in_roster,
    resolve_against_roster,
    roster_key,
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


def _unescape_json_string(raw):
    """Decode a JSON string body (the bit between the quotes)."""
    try:
        return json.loads('"' + raw + '"')
    except (json.JSONDecodeError, ValueError):
        return raw.replace('\\"', '"').replace('\\n', '\n').replace('\\\\', '\\')


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
    return parse_label_array(json_text) or salvage_label_entries(json_text)


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
    """
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
    """Return ``(missing_ids, bad_role_ids)`` -- what a retry should ask for.

    Membership, never a count: live responses hallucinate ids past N (137
    labels for 41 spans in one logged response), so ``len(merged) >= len(spans)``
    would fake completeness. ``bad_role_ids`` mirrors resolve_span_labels'
    role_missing condition exactly, so the retry predicate and the degradation
    reason can never disagree.

    When ``source`` is given, whitespace-only spans are skipped: the model was
    never shown them, so it cannot be "missing" one.
    """
    if source is not None:
        spans = visible_spans(spans, source)

    missing = [span.id for span in spans if span.id not in merged]
    bad_role = []
    for span in spans:
        label = merged.get(span.id)
        if label is None or _usable_field("role", label.get("role")):
            continue
        raw_speaker = label.get("speaker")
        canonical = canonicalize(raw_speaker) if isinstance(raw_speaker, str) else ""
        if canonical and canonical != NARRATOR:
            bad_role.append(span.id)
    return missing, bad_role


# Cap the id list in a nudge so a wholly-unlabelled chunk cannot balloon the prompt.
_NUDGE_ID_CAP = 50


def _format_id_list(ids):
    if len(ids) <= _NUDGE_ID_CAP:
        return ", ".join(str(i) for i in ids)
    head = ", ".join(str(i) for i in ids[:_NUDGE_ID_CAP])
    return f"{head} (and {len(ids) - _NUDGE_ID_CAP} more)"


def _retry_nudge(missing_ids, bad_role_ids, unattested=None):
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


def _attestation_verdict(canonical, attest_window):
    """attest_speaker() over the one window the gate uses. Pure passthrough,
    factored out so the retry predicate and the acceptance gate cannot drift
    apart (the same mistake _incomplete_span_ids exists to prevent for roles).
    """
    return attest_speaker(canonical, [attest_window] if attest_window else [])


def _unattested_speaker_ids(spans, merged, source=None, roster=None,
                            attest_window=None):
    """Return ``[(span_id, canonical_name), ...]`` for dialogue labels whose
    speaker the source does not support -- what a retry should ask about.

    Mirrors the acceptance gate in resolve_span_labels exactly: same
    established-roster shortcut, same verdict function, and only UNATTESTED
    counts. UNVERIFIABLE never appears here, because a label our check cannot
    evaluate is not something to nag the model about.

    Read-only. Does not mutate ``roster``; a local copy tracks names accepted
    earlier in this same chunk so the shortcut behaves as it will at
    resolution time.
    """
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
        if _speaker_is_established(canonical, roster_index):
            continue
        if _attestation_verdict(canonical, attest_window) == UNATTESTED:
            offenders.append((span.id, canonical))
        else:
            remember_in_roster(roster_index, canonical)
    return offenders


def resolve_span_labels(spans, labels, source=None, roster=None,
                        attest_window=None, require_attested=False):
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

    resolved = []
    labelled = 0
    role_missing = 0
    whitespace = 0
    placeholder_rejected = 0
    unattested_rejected = 0
    unverifiable_accepted = 0
    dialogue_without_speaker = 0

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
                if (require_attested and canonical != NARRATOR
                        and not _speaker_is_established(canonical, roster_index)):
                    verdict = _attestation_verdict(canonical, attest_window)
                    if verdict == UNATTESTED:
                        accept = False
                        unattested_rejected += 1
                    elif verdict == UNVERIFIABLE:
                        unverifiable_accepted += 1

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
        "dialogue_without_speaker": dialogue_without_speaker,
    }


def _merge_same_speaker(groups):
    """Collapse adjacent [speaker, text, instruct] groups sharing a speaker."""
    merged = []
    for speaker, text, instruct in groups:
        if merged and merged[-1][0] == speaker:
            merged[-1][1] += text
            if merged[-1][2] is None:
                merged[-1][2] = instruct
        else:
            merged.append([speaker, text, instruct])
    return merged


def _absorb_whitespace_groups(groups):
    """Fold whitespace-only groups into a neighbour, keeping bytes identical.

    The paragraph break between two speakers' lines is its own unquoted span,
    so it can resolve to a group whose text is just "\\n\\n". Emitted as an
    entry it is an unspeakable NARRATOR line that the editor UI can never
    finish rendering -- hundreds per book. It belongs on the PRECEDING entry
    (or the following one when it comes first), which changes no bytes.
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
    """
    groups = _merge_same_speaker(
        [[speaker, span.text(source), instruct] for span, speaker, instruct in resolved]
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
        for speaker, text, instruct in groups
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


def process_chunk(client, model_name, chunk, chunk_num, total_chunks, previous_entries=None, max_retries=2, system_prompt=None, user_prompt_template=None, max_tokens=4096, temperature=0.6, top_p=0.8, top_k=0, min_p=0, presence_penalty=0.0, banned_tokens=None, roster=None, max_context_roster_names=None, num_ctx=None, attest_window=None, require_attested=False, reasoning_effort=None):
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

        missing_ids, bad_role_ids = _incomplete_span_ids(spans, merged_labels, source=chunk)
        unattested = (
            _unattested_speaker_ids(spans, merged_labels, source=chunk,
                                    roster=roster, attest_window=attest_window)
            if require_attested else []
        )

        if not missing_ids and not bad_role_ids and not unattested:
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
            if unattested:
                gaps.append(f"{len(unattested)} speaker name(s) absent from the text "
                            f"({', '.join(name for _, name in unattested[:3])}"
                            f"{', ...' if len(unattested) > 3 else ''})")
            print(f"  {' and '.join(gaps)} -- retrying (attempt {attempt + 2}"
                  f"/{max_retries + 1}), naming the gaps")
            retry_nudge = _retry_nudge(missing_ids, bad_role_ids, unattested)
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
        attest_window=attest_window, require_attested=require_attested)
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
    if stats["role_missing"]:
        print(f"  {stats['role_missing']} label(s) named a speaker without role=\"dialogue\" "
              "-- narrated instead (schema violation by the model)")
    if stats["placeholder_rejected"]:
        print(f"  Rejected {stats['placeholder_rejected']} invented placeholder speaker "
              "label(s) (e.g. \"SPEAKER 1\") -- narrated instead")
    if stats["unattested_rejected"]:
        print(f"  Rejected {stats['unattested_rejected']} speaker label(s) whose name does "
              "not appear in the text near their own lines -- narrated instead")
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
    # Attestation gate. DEFAULT OFF: it changes which labels survive and can
    # turn an exit-0 run into exit 3 (degraded), so it is opt-in rather than a
    # behaviour change existing users get silently. Measure a book first with
    # tools/verify_attestation.py, which reports the would-be rejection rate
    # without running the model at all.
    require_attested = bool(generation_config.get("require_attested_speakers", False))
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
    else:
        print("  Attestation gate:        off "
              "(generation.require_attested_speakers)")
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
