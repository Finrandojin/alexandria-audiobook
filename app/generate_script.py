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
from speaker_canon import canonicalize

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


def clean_json_string(text):
    """Clean and extract valid JSON array from LLM response."""
    text = strip_thinking_tags(text)

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

    def _filter_entries(lst):
        """Keep only dict entries; LLMs sometimes emit bare strings in the array."""
        filtered = [e for e in lst if isinstance(e, dict)]
        if len(filtered) < len(lst):
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

    # Salvage the cleaned array text when we have it, else the raw response:
    # clean_json_string() returns None for a hard truncation with no closing
    # bracket, which is exactly when salvage matters most.
    labels = salvage_label_entries(json_text or text)
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


def build_span_payload(spans, source):
    """Render spans as the numbered listing the classifier receives.

    One JSON object per line: {"id", "kind", "text"}. The LLM sees the text so
    it can classify it; it is instructed never to send text back.
    """
    return "\n".join(
        json.dumps(
            {"id": span.id, "kind": span.kind, "text": span.text(source)},
            ensure_ascii=False,
        )
        for span in spans
    )


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


def resolve_span_labels(spans, labels):
    """Resolve each span to (span, speaker, instruct) using the LLM's labels.

    A span is NARRATOR unless a label exists for its id AND that label says
    role == "dialogue" AND its speaker canonicalizes to a non-empty name.
    Labels for ids that do not exist are discarded. Returns
    (resolved, stats_dict).
    """
    valid_ids = {span.id for span in spans}
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

    for span in spans:
        label = by_id.get(span.id)
        speaker = NARRATOR
        instruct = None

        if label is not None:
            labelled += 1
            role = label.get("role")
            role = role.strip().lower() if isinstance(role, str) else ""
            raw_speaker = label.get("speaker")
            canonical = canonicalize(raw_speaker) if isinstance(raw_speaker, str) else ""

            if role == "dialogue" and canonical:
                speaker = canonical
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
        "labelled": labelled,
        "fallback": len(spans) - labelled,
        "discarded": discarded,
        "role_missing": role_missing,
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

def build_context(chunk_num, total_chunks, previous_entries=None):
    """Positional note + speaker roster + a short tail of previous entries.

    Context is INPUT, not output: a truncated snippet of each recent entry is
    enough for tone/roster continuity and keeps the prompt cheap now that
    narration entries can be long.
    """
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
            text = entry.get("text", "")
            if len(text) > CONTEXT_SNIPPET_CHARS:
                text = text[:CONTEXT_SNIPPET_CHARS].rstrip() + "..."
            context_parts.append(json.dumps(
                {"speaker": entry.get("speaker", NARRATOR), "text": text},
                ensure_ascii=False,
            ))

    return "\n".join(context_parts)


def process_chunk(client, model_name, chunk, chunk_num, total_chunks, previous_entries=None, max_retries=2, system_prompt=None, user_prompt_template=None, max_tokens=4096, temperature=0.6, top_p=0.8, top_k=0, min_p=0, presence_penalty=0.0, banned_tokens=None):
    """Classify one chunk's spans and rebuild its script entries verbatim.

    Returns ``(entries, stats)``. ``stats`` reports span counts and whether the
    chunk degraded; ``entries`` ALWAYS reproduces the chunk byte-for-byte, even
    when the LLM failed outright (the whole chunk then becomes NARRATOR).
    """
    # Use provided prompts or fall back to defaults
    sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    usr_template = user_prompt_template or DEFAULT_USER_PROMPT

    spans = tokenize(chunk)
    validate_spans(spans, chunk)

    if not spans:
        return [], {"spans": 0, "labelled": 0, "fallback": 0, "discarded": 0,
                    "role_missing": 0, "degraded": False, "reason": None,
                    "recovery": None}

    context = build_context(chunk_num, total_chunks, previous_entries)
    user_prompt = usr_template.format(context=context, chunk=build_span_payload(spans, chunk))

    labels = None
    recovery = None
    truncated = False
    reason = None

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
                        "top_k": top_k if top_k else None,
                        "min_p": min_p if min_p else None,
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

            truncated = finish_reason == "length"
            if truncated:
                print(f"  WARNING: Response was truncated (hit max_tokens={max_tokens}). "
                      "Unlabelled spans will fall back to NARRATOR.")

        except Exception as e:
            print(f"Error calling LLM API (attempt {attempt + 1}): {e}")
            if attempt < max_retries:
                continue
            reason = f"LLM call failed on all {max_retries + 1} attempts ({e})"
            break

        # Recover labels from whatever shape the model actually produced.
        labels, recovery = extract_labels(text)

        if labels:
            if attempt > 0:
                print(f"  Succeeded on retry {attempt + 1}")
            if recovery == LABEL_MODE_OBJECT:
                print(f"  Note: model returned an id-keyed JSON object instead of an array; "
                      f"recovered all {len(labels)} label(s) from it")
            elif recovery == LABEL_MODE_SALVAGE:
                print(f"  Regex-salvaged {len(labels)} label(s) from a malformed/truncated response")
            elif recovery == LABEL_MODE_MARKDOWN:
                print(f"  Recovered {len(labels)} label(s) from markdown blocks "
                      "(model ignored the JSON output contract)")
            break

        print(f"Warning: Could not recover labels from chunk {chunk_num} response (attempt {attempt + 1})")
        print(f"Response preview: {text[:300]}...")

        if attempt < max_retries:
            print("Retrying...")
        else:
            reason = "no usable labels recovered from LLM response"

    # Reassemble regardless of what came back. Unlabelled spans -> NARRATOR,
    # so a failure costs labels, never prose.
    resolved, stats = resolve_span_labels(spans, labels)
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
    if reason is None and stats["role_missing"] > 0:
        # A model that labels every span but omits "role" produces a fully
        # narrated book. Prose is intact, every voice is gone -- exactly the
        # silent failure this stage exists to eliminate, so it degrades.
        reason = (f"{stats['role_missing']} label(s) named a speaker without "
                  "role=dialogue")

    stats["spans"] = len(spans)
    stats["recovery"] = recovery
    stats["degraded"] = reason is not None
    stats["reason"] = reason

    if stats["discarded"]:
        print(f"  Discarded {stats['discarded']} label(s) referring to nonexistent span ids")
    if stats["role_missing"]:
        print(f"  {stats['role_missing']} label(s) named a speaker without role=\"dialogue\" "
              "-- narrated instead (schema violation by the model)")

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
    degraded_chunks = []
    total_spans = 0
    total_fallback = 0

    for i, chunk in enumerate(chunks, 1):
        print(f"Processing chunk {i}/{total_chunks} ({len(chunk)} chars)...")

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
            banned_tokens=banned_tokens
        )
        all_entries.extend(entries)
        total_spans += stats["spans"]
        total_fallback += stats["fallback"]
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
