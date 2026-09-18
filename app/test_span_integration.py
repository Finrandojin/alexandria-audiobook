#!/usr/bin/env python
"""Standalone integration tests for the span-classifier stage of generate_script.py.

Run directly -- no pytest, no live server, no real LLM:

    python app/test_span_integration.py

Exits 0 when every test passes, nonzero otherwise.

The contract under test: whatever the LLM does -- label everything, label half,
return garbage, or fall over entirely -- the entries built for a chunk always
concatenate back to that chunk BYTE FOR BYTE. Failure costs speaker labels,
never prose.
"""

import atexit
import inspect
import io
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout

# Windows consoles default to cp1252; the fixtures contain curly quotes.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Redirect generate_script's raw-response log to a tempdir for the whole
# process. Done at IMPORT time, not in setUpModule, so it also covers
# test_api.py's Section 15, which imports FakeClient/labels_for from here and
# drives process_chunk itself. Without this the suite appends fixture responses
# to the production logs/llm_responses.log -- 1200+ records of noise in the file
# used for live forensics.
_LOG_TMPDIR = tempfile.mkdtemp(prefix="alexandria_test_llm_log_")
os.environ["ALEXANDRIA_LLM_LOG_DIR"] = _LOG_TMPDIR
atexit.register(shutil.rmtree, _LOG_TMPDIR, True)

from default_prompts import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT  # noqa: E402
from span_tokenizer import Span, tokenize  # noqa: E402
import generate_script  # noqa: E402
from generate_script import (  # noqa: E402
    DEFAULT_NARRATOR_INSTRUCT,
    LABEL_MODE_ARRAY,
    LABEL_MODE_MARKDOWN,
    LABEL_MODE_OBJECT,
    LABEL_MODE_SALVAGE,
    NARRATOR,
    PROMPT_SCHEMA_MARKER,
    build_entries,
    build_span_payload,
    extract_labels,
    is_whitespace_span,
    llm_log_dir,
    labels_from_id_keyed_object,
    parse_label_response,
    process_chunk,
    resolve_span_labels,
    salvage_label_entries,
    salvage_markdown_labels,
    select_prompt,
    build_context,
    MAX_CONTEXT_ROSTER_NAMES,
    MOJIBAKE_REPLACEMENTS,
    clean_json_string,
    fix_mojibake,
    detect_flatlined_prompt_tokens,
    estimate_prompt_tokens,
    is_placeholder_speaker,
    looks_silently_truncated,
    strip_thinking_tags,
)
from speaker_canon import remember_in_roster, source_word_index


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

STRAIGHT_QUOTES = (
    '"I am leaving," he said.\n'
    '\n'
    'Marcus did not look up from the map. "Then go."\n'
)

CURLY_QUOTES = (
    "“You were never going to tell me,” she said. It wasn’t a question.\n"
    "\n"
    "“I was,” Marcus answered. “I was waiting for the right hour.”\n"
)

ATTRIBUTION_MID_SENTENCE = (
    '"We should leave," he said, pulling on his coat, "now, before they notice."'
)

# Two bare dialogue paragraphs: the blank line between them is its own
# unquoted span, and must never become a whitespace-only entry.
TWO_DIALOGUE_PARAGRAPHS = '"Hi."\n\n"Bye."'

FIXTURES = {
    "straight": STRAIGHT_QUOTES,
    "curly": CURLY_QUOTES,
    "mid_sentence": ATTRIBUTION_MID_SENTENCE,
    "two_dialogue_paragraphs": TWO_DIALOGUE_PARAGRAPHS,
}


# --------------------------------------------------------------------------
# REAL response shapes, excerpted verbatim from logs/llm_responses.log
# (live runs against Ollama). Trimmed to the first few spans; the trailing
# double spaces in the markdown fixtures are markdown hard breaks and are
# present in the originals -- keep them.
# --------------------------------------------------------------------------

# qwen3:30b-a3b-q4_K_M, CHUNK 1/4 attempt 2: valid JSON, wrong envelope.
REAL_ID_KEYED_OBJECT = (
    '{\n'
    '  "1": {"speaker": "NARRATOR", "role": "narration", "instruct": "Neutral narration"},\n'
    '  "2": {"speaker": "NARRATOR", "role": "dialogue", "instruct": "Speaking with confidence"},\n'
    '  "3": {"speaker": "NARRATOR", "role": "narration", "instruct": "Neutral narration"},\n'
    '  "4": {"speaker": "NARRATOR", "role": "dialogue", "instruct": "Speaking with conviction"},\n'
    '  "5": {"speaker": "NARRATOR", "role": "narration", "instruct": "Neutral narration"},\n'
    '  "6": {"speaker": "BILL", "role": "dialogue", "instruct": "Speaking with certainty"}\n'
    '}'
)

# qwen3:30b-a3b-q4_K_M, CHUNK 1/4 attempt 3: "**Span N**" markdown blocks.
REAL_MARKDOWN_SPAN = (
    "Here is the classification for each span:\n"
    "\n"
    "---\n"
    "\n"
    "**Span 1**  \n"
    "- **Speaker**: NARRATOR  \n"
    "- **Role**: narration  \n"
    "- **Instruct**: \"Neutral, even narration.\"  \n"
    "\n"
    "**Span 2**  \n"
    "- **Speaker**: NARRATOR  \n"
    "- **Role**: dialogue  \n"
    "- **Instruct**: \"Calm, conversational tone.\"  \n"
    "\n"
    "**Span 3**  \n"
    "- **Speaker**: BILL  \n"
    "- **Role**: dialogue  \n"
    "- **Instruct**: \"Speaking with certainty.\"  \n"
)

# qwen3:30b-a3b-q4_K_M, CHUNK 1/4 attempt 1: the "**id N**" header variant.
REAL_MARKDOWN_ID = (
    "Here is the classification for each span based on the rules provided:\n"
    "\n"
    "---\n"
    "\n"
    "**id 1**  \n"
    "- **Speaker**: NARRATOR  \n"
    "- **Role**: narration  \n"
    "- **Instruct**: \"Neutral, descriptive tone.\"  \n"
    "\n"
    "**id 2**  \n"
    "- **Speaker**: NARRATOR  \n"
    "- **Role**: dialogue  \n"
    "- **Instruct**: \"Confident, persuasive tone.\"  \n"
    "\n"
    "**id 3**  \n"
    "- **Speaker**: BILL  \n"
    "- **Role**: dialogue  \n"
    "- **Instruct**: \"Calm, matter-of-fact.\"  \n"
)

# A hard truncation (finish_reason=length) preceded by a reasoning block whose
# id-like fragments a bare regex salvage would mistake for labels. One live run
# discarded 96 phantom ids this way.
REAL_THINK_POLLUTED_TRUNCATION = (
    "<think>\n"
    "Okay, let me work through this. Span 500 looks like narration, so maybe\n"
    '{"id": 500, "speaker": "GHOST", "role": "dialogue", "instruct": "x"} and\n'
    '{"id": 501, "speaker": "PHANTOM", "role": "dialogue", "instruct": "y"}.\n'
    "Actually no, let me start over and emit the real array.\n"
    "</think>\n"
    '[{"id": 1, "speaker": "NARRATOR", "role": "narr'
)


# --------------------------------------------------------------------------
# Fake OpenAI-shaped client
# --------------------------------------------------------------------------

class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content, finish_reason):
        self.message = _FakeMessage(content)
        self.finish_reason = finish_reason


class FakeResponse:
    """Minimal stand-in for an OpenAI ChatCompletion response."""

    def __init__(self, content, finish_reason="stop", prompt_tokens=None):
        self.choices = [_FakeChoice(content, finish_reason)]
        # usage is None unless a test cares: the silent-truncation tripwire
        # reads usage.prompt_tokens and must stay quiet when it is absent.
        self.usage = (types.SimpleNamespace(prompt_tokens=prompt_tokens,
                                            completion_tokens=0)
                      if prompt_tokens is not None else None)


class _FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.last_kwargs = None
        self.prompts = []          # user prompt of every call, in order

    def create(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        self.prompts.append(kwargs["messages"][1]["content"])
        index = min(self.calls - 1, len(self.responses) - 1)
        item = self.responses[index]
        if isinstance(item, Exception):
            raise item
        return item


class FakeClient:
    """Injectable client: process_chunk only ever calls chat.completions.create."""

    def __init__(self, *responses):
        self.completions = _FakeCompletions(responses)
        self.chat = types.SimpleNamespace(completions=self.completions)

    @property
    def calls(self):
        return self.completions.calls

    @property
    def last_user_prompt(self):
        return self.completions.last_kwargs["messages"][1]["content"]

    @property
    def user_prompts(self):
        return self.completions.prompts


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def labels_for(source, speaker_of=None):
    """Build a well-formed label array covering every span of `source`.

    `speaker_of(span, text)` returns the character name for a quoted span;
    default is ELENA. Unquoted spans are always narration.
    """
    labels = []
    for span in tokenize(source):
        if span.kind == "quoted":
            name = speaker_of(span, span.text(source)) if speaker_of else "ELENA"
            labels.append({
                "id": span.id,
                "speaker": name,
                "role": "dialogue",
                "instruct": "Flat, controlled delivery.",
            })
        else:
            labels.append({
                "id": span.id,
                "speaker": "NARRATOR",
                "role": "narration",
                "instruct": DEFAULT_NARRATOR_INSTRUCT,
            })
    return labels


def run_chunk(client, chunk, **kwargs):
    """Call process_chunk with the real default prompts, capturing its output."""
    buffer = io.StringIO()
    options = dict(
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        user_prompt_template=DEFAULT_USER_PROMPT,
        max_retries=2,
    )
    options.update(kwargs)
    with redirect_stdout(buffer):
        entries, stats = process_chunk(client, "fake-model", chunk, 1, 1, **options)
    return entries, stats, buffer.getvalue()


def joined(entries):
    return "".join(entry["text"] for entry in entries)


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

class TestFullLabelling(unittest.TestCase):
    """The happy path: the LLM labels every span."""

    def test_dialogue_heavy_chunk_is_verbatim(self):
        chunk = STRAIGHT_QUOTES
        client = FakeClient(FakeResponse(json.dumps(labels_for(chunk))))
        entries, stats, _ = run_chunk(client, chunk)

        self.assertEqual(joined(entries), chunk, "entries must rebuild the chunk byte-for-byte")
        self.assertFalse(stats["degraded"])
        self.assertEqual(stats["fallback"], 0)
        self.assertEqual(stats["labelled"], stats["spans"])
        self.assertEqual(client.calls, 1)

    def test_quoted_spans_get_characters_and_tags_stay_narrator(self):
        chunk = STRAIGHT_QUOTES
        client = FakeClient(FakeResponse(json.dumps(labels_for(chunk))))
        entries, _, _ = run_chunk(client, chunk)

        speakers = [entry["speaker"] for entry in entries]
        self.assertIn("ELENA", speakers)
        self.assertIn(NARRATOR, speakers)

        # The attribution tag is narrated, verbatim, tag included.
        narrator_text = "".join(e["text"] for e in entries if e["speaker"] == NARRATOR)
        self.assertIn(" he said.", narrator_text)
        # ...and the quote keeps its quotation marks.
        elena_text = "".join(e["text"] for e in entries if e["speaker"] == "ELENA")
        self.assertIn('"I am leaving,"', elena_text)

    def test_speakers_are_canonicalized_uppercase(self):
        chunk = STRAIGHT_QUOTES
        raw_names = iter(["  elena ", "Dr. Vance"])
        labels = labels_for(chunk, speaker_of=lambda span, text: next(raw_names))
        client = FakeClient(FakeResponse(json.dumps(labels)))
        entries, _, _ = run_chunk(client, chunk)

        speakers = {entry["speaker"] for entry in entries}
        self.assertIn("ELENA", speakers)
        self.assertIn("VANCE", speakers, "honorific must be stripped by canonicalize()")
        for speaker in speakers:
            self.assertEqual(speaker, speaker.upper())

    def test_consecutive_same_speaker_spans_merge(self):
        # Every span narrated -> exactly one merged entry holding the whole chunk.
        chunk = STRAIGHT_QUOTES
        labels = [
            {"id": span.id, "speaker": "NARRATOR", "role": "narration", "instruct": "x"}
            for span in tokenize(chunk)
        ]
        client = FakeClient(FakeResponse(json.dumps(labels)))
        entries, _, _ = run_chunk(client, chunk)

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["text"], chunk)
        self.assertEqual(entries[0]["speaker"], NARRATOR)

    def test_curly_quotes_fixture(self):
        chunk = CURLY_QUOTES
        client = FakeClient(FakeResponse(json.dumps(labels_for(chunk))))
        entries, stats, _ = run_chunk(client, chunk)

        self.assertEqual(joined(entries), chunk)
        self.assertFalse(stats["degraded"])
        elena_text = "".join(e["text"] for e in entries if e["speaker"] == "ELENA")
        self.assertIn("“You were never going to tell me,”", elena_text)
        narrator_text = "".join(e["text"] for e in entries if e["speaker"] == NARRATOR)
        self.assertIn("It wasn’t a question.", narrator_text)

    def test_attribution_tag_mid_sentence(self):
        chunk = ATTRIBUTION_MID_SENTENCE
        client = FakeClient(FakeResponse(json.dumps(labels_for(chunk))))
        entries, stats, _ = run_chunk(client, chunk)

        self.assertEqual(joined(entries), chunk)
        self.assertFalse(stats["degraded"])
        narrator_text = "".join(e["text"] for e in entries if e["speaker"] == NARRATOR)
        self.assertIn(" he said, pulling on his coat, ", narrator_text)
        # Both halves of the split quotation are dialogue.
        self.assertEqual(len([e for e in entries if e["speaker"] == "ELENA"]), 2)


class TestTruncation(unittest.TestCase):
    """finish_reason='length' with only part of the labels present."""

    def test_half_labels_lose_no_prose(self):
        chunk = STRAIGHT_QUOTES
        full = labels_for(chunk)
        half = full[:max(1, len(full) // 2)]
        # Simulate a hard cut mid-array: no closing bracket.
        body = json.dumps(half)[:-1].rstrip()
        client = FakeClient(FakeResponse(body, finish_reason="length"))

        entries, stats, output = run_chunk(client, chunk)

        self.assertEqual(joined(entries), chunk, "truncation must not lose a single character")
        self.assertTrue(stats["degraded"])
        self.assertGreater(stats["fallback"], 0)
        self.assertEqual(stats["labelled"] + stats["fallback"], stats["spans"])
        self.assertIn("DEGRADED", output)
        self.assertIn("truncated", output.lower())

        # Every span the LLM did not label is narrated.
        labelled_ids = {label["id"] for label in half}
        for span in tokenize(chunk):
            if span.id not in labelled_ids:
                owner = [e for e in entries if span.text(chunk) in e["text"]]
                self.assertTrue(owner, f"span {span.id} vanished from the entries")

    def test_truncation_is_a_single_degradation_event(self):
        chunk = STRAIGHT_QUOTES
        full = labels_for(chunk)
        body = json.dumps(full[:1])[:-1]
        client = FakeClient(FakeResponse(body, finish_reason="length"))
        _, stats, _ = run_chunk(client, chunk)

        self.assertTrue(stats["degraded"])
        self.assertIsNotNone(stats["reason"])
        self.assertEqual(client.calls, 1, "a partial-but-usable response must not be retried")


class TestTotalFailure(unittest.TestCase):
    """Every LLM attempt raises."""

    def test_chunk_becomes_narrator_entries(self):
        chunk = CURLY_QUOTES
        client = FakeClient(RuntimeError("connection refused"))
        entries, stats, output = run_chunk(client, chunk)

        self.assertEqual(joined(entries), chunk)
        self.assertEqual(len(entries), 1, "all-NARRATOR spans merge into one entry")
        self.assertEqual(entries[0]["speaker"], NARRATOR)
        self.assertEqual(entries[0]["instruct"], DEFAULT_NARRATOR_INSTRUCT)
        self.assertTrue(stats["degraded"])
        self.assertEqual(stats["labelled"] - stats["whitespace"], 0,
                         "the model labelled nothing (whitespace spans auto-resolve)")
        self.assertEqual(stats["fallback"], stats["spans"] - stats["whitespace"])
        self.assertIn("DEGRADED", output)
        self.assertEqual(client.calls, 3, "should exhaust max_retries + 1 attempts")

    def test_recovers_on_retry(self):
        chunk = STRAIGHT_QUOTES
        client = FakeClient(
            RuntimeError("timeout"),
            FakeResponse(json.dumps(labels_for(chunk))),
        )
        entries, stats, _ = run_chunk(client, chunk)

        self.assertEqual(joined(entries), chunk)
        self.assertFalse(stats["degraded"])
        self.assertEqual(client.calls, 2)


class TestMalformedJson(unittest.TestCase):
    """Broken output: salvage what is there, narrate the rest."""

    def test_unparseable_garbage_falls_back(self):
        chunk = STRAIGHT_QUOTES
        client = FakeClient(FakeResponse("I'm sorry, I cannot help with that."))
        entries, stats, output = run_chunk(client, chunk)

        self.assertEqual(joined(entries), chunk)
        self.assertTrue(stats["degraded"])
        self.assertEqual(stats["fallback"], stats["spans"] - stats["whitespace"],
                         "every VISIBLE span fell back; whitespace auto-resolved")
        self.assertIn("DEGRADED", output)

    def test_salvageable_broken_json(self):
        chunk = STRAIGHT_QUOTES
        spans = tokenize(chunk)
        # Missing commas between objects, a stray trailing comma, no final bracket.
        broken = (
            "Here you go:\n```json\n["
            + "\n".join(
                '{"id": %d, "role": "%s", "speaker": "%s", "instruct": "Tense."}'
                % (
                    span.id,
                    "dialogue" if span.kind == "quoted" else "narration",
                    "ELENA" if span.kind == "quoted" else "NARRATOR",
                )
                for span in spans
            )
            + ","
        )
        client = FakeClient(FakeResponse(broken))
        entries, stats, _ = run_chunk(client, chunk)

        self.assertEqual(joined(entries), chunk)
        self.assertGreater(stats["labelled"], 0, "regex salvage should recover labels")
        self.assertIn("ELENA", {entry["speaker"] for entry in entries})

    def test_llm_echoing_text_is_ignored(self):
        # The one thing the LLM must never do. If it does it anyway, the text
        # field is dropped on the floor -- code owns the prose.
        chunk = STRAIGHT_QUOTES
        labels = labels_for(chunk)
        for label in labels:
            label["text"] = "COMPLETELY WRONG TEXT THE MODEL MADE UP"
        client = FakeClient(FakeResponse(json.dumps(labels)))
        entries, _, _ = run_chunk(client, chunk)

        self.assertEqual(joined(entries), chunk)
        self.assertNotIn("MADE UP", joined(entries))


class TestLabelResolution(unittest.TestCase):
    """Unit-level rules around label -> speaker resolution."""

    def test_unknown_ids_are_discarded(self):
        chunk = STRAIGHT_QUOTES
        spans = tokenize(chunk)
        labels = [{"id": 9999, "speaker": "GHOST", "role": "dialogue", "instruct": "x"}]
        resolved, stats = resolve_span_labels(spans, labels)

        self.assertEqual(stats["discarded"], 1)
        self.assertEqual(stats["labelled"], 0)
        self.assertTrue(all(speaker == NARRATOR for _, speaker, _ in resolved))

    def test_empty_speaker_becomes_narrator(self):
        chunk = STRAIGHT_QUOTES
        spans = tokenize(chunk)
        labels = [{"id": s.id, "speaker": "", "role": "dialogue", "instruct": "x"} for s in spans]
        resolved, stats = resolve_span_labels(spans, labels)

        self.assertEqual(stats["labelled"], len(spans))
        self.assertTrue(all(speaker == NARRATOR for _, speaker, _ in resolved))

    def test_role_narration_overrides_a_character_name(self):
        chunk = STRAIGHT_QUOTES
        spans = tokenize(chunk)
        labels = [{"id": s.id, "speaker": "ELENA", "role": "narration", "instruct": "x"} for s in spans]
        resolved, _ = resolve_span_labels(spans, labels)
        self.assertTrue(all(speaker == NARRATOR for _, speaker, _ in resolved))

    def test_complete_label_beats_a_truncated_duplicate(self):
        # salvage_label_entries() can recover a tail fragment for an id that
        # already parsed cleanly; the fragment must not overwrite it.
        spans = tokenize(STRAIGHT_QUOTES)
        first = spans[0]
        complete = {"id": first.id, "speaker": "ELENA", "role": "dialogue", "instruct": "Firm."}
        fragment = {"id": first.id, "speaker": "ELENA"}  # no role: truncated tail

        for order in ([complete, fragment], [fragment, complete]):
            with self.subTest(order=[("complete" if l is complete else "fragment") for l in order]):
                resolved, _ = resolve_span_labels(spans, order)
                speaker = resolved[0][1]
                instruct = resolved[0][2]
                self.assertEqual(speaker, "ELENA", "the complete label must win")
                self.assertEqual(instruct, "Firm.")

    def test_equal_completeness_keeps_last_write_wins(self):
        spans = tokenize(STRAIGHT_QUOTES)
        first = spans[0]
        labels = [
            {"id": first.id, "speaker": "ALPHA", "role": "dialogue", "instruct": "One."},
            {"id": first.id, "speaker": "BETA", "role": "dialogue", "instruct": "Two."},
        ]
        resolved, _ = resolve_span_labels(spans, labels)
        self.assertEqual(resolved[0][1], "BETA")

    def test_instruct_defaults(self):
        chunk = STRAIGHT_QUOTES
        spans = tokenize(chunk)
        labels = [{"id": s.id, "speaker": "ELENA", "role": "dialogue"} for s in spans]
        resolved, _ = resolve_span_labels(spans, labels)
        entries = build_entries(resolved, chunk)
        # The narration between the two quotations is handed back to NARRATOR
        # (see TestNarrationNeverGetsACharacterVoice), so the chunk is three
        # entries, each carrying the default instruct for its own speaker.
        self.assertEqual(joined(entries), chunk)
        for entry in entries:
            self.assertEqual(
                entry["instruct"],
                generate_script.DEFAULT_NARRATOR_INSTRUCT
                if entry["speaker"] == NARRATOR
                else generate_script.DEFAULT_CHARACTER_INSTRUCT,
            )

    def test_entry_schema_has_no_extra_fields(self):
        chunk = STRAIGHT_QUOTES
        client = FakeClient(FakeResponse(json.dumps(labels_for(chunk))))
        entries, _, _ = run_chunk(client, chunk)
        for entry in entries:
            self.assertEqual(set(entry.keys()), {"speaker", "text", "instruct"},
                             "role must not leak into the output schema")

    def test_verbatim_assertion_raises_on_corruption(self):
        with self.assertRaises(AssertionError):
            generate_script._assert_chunk_verbatim(
                [{"speaker": NARRATOR, "text": "not the chunk", "instruct": "x"}],
                STRAIGHT_QUOTES,
                1,
            )


class TestRoleMissingDegrades(unittest.TestCase):
    """A model that names speakers but omits "role" narrates the whole book."""

    @staticmethod
    def _labels_without_role(chunk):
        return [
            {"id": span.id,
             "speaker": "ELENA" if span.kind == "quoted" else "NARRATOR",
             "instruct": "Tense."}
            for span in tokenize(chunk)
        ]

    def test_speaker_without_role_marks_chunk_degraded(self):
        chunk = STRAIGHT_QUOTES
        client = FakeClient(FakeResponse(json.dumps(self._labels_without_role(chunk))))
        entries, stats, output = run_chunk(client, chunk)

        self.assertEqual(joined(entries), chunk, "prose must still be byte-identical")
        self.assertGreater(stats["role_missing"], 0)
        self.assertEqual(stats["fallback"], 0, "every span WAS labelled -- only role is missing")
        self.assertTrue(stats["degraded"], "role-less labels must not exit 0 silently")
        self.assertIn("role=dialogue", stats["reason"])
        self.assertIn("DEGRADED", output)
        self.assertEqual(client.calls, 3,
                         "role-less labels now trigger retries; the fake repeats them, so all 3 burn")

        # The book is fully narrated: that is the damage being surfaced.
        self.assertEqual({entry["speaker"] for entry in entries}, {NARRATOR})

    def test_degraded_reason_reaches_the_exit_3_summary(self):
        # main() adds a chunk to degraded_chunks iff stats["degraded"], and
        # exits EXIT_DEGRADED iff that list is non-empty. Assert the linkage.
        chunk = STRAIGHT_QUOTES
        client = FakeClient(FakeResponse(json.dumps(self._labels_without_role(chunk))))
        _, stats, _ = run_chunk(client, chunk)

        self.assertTrue(stats["degraded"])
        self.assertIsNotNone(stats["reason"])
        self.assertEqual(generate_script.EXIT_DEGRADED, 3)

    def test_explicit_narration_role_is_not_a_degradation(self):
        # role="narration" on a NARRATOR span is correct, not a schema violation.
        chunk = STRAIGHT_QUOTES
        labels = [
            {"id": span.id, "speaker": "NARRATOR", "role": "narration", "instruct": "Even."}
            for span in tokenize(chunk)
        ]
        _, stats, _ = run_chunk(FakeClient(FakeResponse(json.dumps(labels))), chunk)

        self.assertEqual(stats["role_missing"], 0)
        self.assertFalse(stats["degraded"])


class TestUnknownLabelKeysAreVisible(unittest.TestCase):
    """A key outside the schema is discarded -- but never silently.

    What reaches this path is the residue _recover_label_keys could not place:
    a key too far from every schema key, or near two of them. Nothing is
    repaired here; the discard is only reported, so that residue stays
    measurable rather than becoming invisible once recovery exists.
    """

    def test_unplaceable_key_is_counted_and_logged(self):
        chunk = STRAIGHT_QUOTES
        labels = labels_for(chunk)
        for label in labels:
            label["subtext"] = label.pop("instruct")
        _, stats, output = run_chunk(FakeClient(FakeResponse(json.dumps(labels))), chunk)

        self.assertEqual(stats["unknown_key_labels"], len(labels))
        self.assertEqual(stats["unknown_keys"], ["subtext"])
        self.assertIn("subtext", output)
        self.assertIn("Ignored unknown key", output)
        self.assertEqual(stats["text_key_labels"], 0)

    def test_known_schema_produces_no_warning(self):
        chunk = STRAIGHT_QUOTES
        labels = labels_for(chunk)
        _, stats, output = run_chunk(FakeClient(FakeResponse(json.dumps(labels))), chunk)

        self.assertEqual(stats["unknown_key_labels"], 0)
        self.assertEqual(stats["unknown_keys"], [])
        self.assertEqual(stats["text_key_labels"], 0)
        self.assertNotIn("Ignored unknown key", output)

    def test_text_key_is_not_honoured_and_is_reported_on_its_own(self):
        chunk = STRAIGHT_QUOTES
        labels = labels_for(chunk)
        for label in labels:
            label["text"] = "MODEL-INVENTED PROSE"
        entries, stats, output = run_chunk(FakeClient(FakeResponse(json.dumps(labels))), chunk)

        self.assertEqual(joined(entries), chunk, "prose must come from the source only")
        for entry in entries:
            self.assertNotIn("MODEL-INVENTED PROSE", entry["text"])
        self.assertEqual(stats["text_key_labels"], len(labels))
        self.assertIn('"text" key', output)


class TestMisspelledKeyRecovery(unittest.TestCase):
    """A misspelled schema key is folded back onto the key it can only mean.

    Properties, not a table of observed spellings: the rule must hold for any
    misspelling within the budget, and must never reach "text" or "id".
    """

    @staticmethod
    def _recovered(key):
        label = {"id": 1, key: "value"}
        with redirect_stdout(io.StringIO()):
            generate_script._recover_label_keys([label])
        return {k for k in label if k != "id"}

    def test_every_single_edit_of_a_long_key_is_recovered(self):
        # Exhaustive over one edit, so this cannot be fitted to any one book's
        # model output. Substitutions, deletions and insertions all count.
        target = "instruct"
        variants = {target[:i] + target[i + 1:] for i in range(len(target))}
        variants |= {target[:i] + "x" + target[i + 1:] for i in range(len(target))}
        variants |= {target[:i] + "x" + target[i:] for i in range(len(target) + 1)}
        for variant in sorted(variants - {target}):
            with self.subTest(variant=variant):
                self.assertEqual(self._recovered(variant), {target})

    def test_separators_and_case_are_not_misspellings(self):
        for variant in ("Instruct", "IN_STRUCT", "in instruct", " instruct "):
            with self.subTest(variant=variant):
                self.assertEqual(self._recovered(variant), {"instruct"})

    def test_nothing_ever_becomes_text(self):
        # Contract 1: the model never supplies prose. "text" is not a recovery
        # target, so no spelling of it -- however close -- can be honoured.
        for variant in ("text", "texts", "tex", "txt", "Text", "te xt", "textt"):
            with self.subTest(variant=variant):
                # Left exactly as sent: still unknown, still discarded, still
                # reported. Never turned INTO the schema's "text".
                self.assertEqual(self._recovered(variant), {variant})

    def test_short_keys_stay_strict(self):
        # "id" has a budget of zero: near-misses are common English words and
        # recovering them would invent span ids.
        for variant in ("in", "is", "if", "idx", "od"):
            with self.subTest(variant=variant):
                self.assertEqual(self._recovered(variant), {variant})

    def test_ambiguity_and_distance_leave_the_key_alone(self):
        for variant in ("inquest", "infty", "commentary"):
            with self.subTest(variant=variant):
                self.assertEqual(self._recovered(variant), {variant})

    def test_a_correct_key_is_never_overwritten(self):
        label = {"id": 1, "instruct": "real", "instructor": "guess"}
        with redirect_stdout(io.StringIO()):
            generate_script._recover_label_keys([label])
        self.assertEqual(label["instruct"], "real")

    def test_recovery_reaches_the_pipeline_and_is_logged(self):
        chunk = STRAIGHT_QUOTES
        labels = labels_for(chunk)
        for label in labels:
            label["instructor"] = label.pop("instruct")
        entries, stats, output = run_chunk(
            FakeClient(FakeResponse(json.dumps(labels))), chunk)

        self.assertEqual(joined(entries), chunk)
        self.assertEqual(stats["unknown_key_labels"], 0)
        self.assertIn("Recovered misspelled key", output)
        for entry in entries:
            self.assertNotEqual(entry["instruct"], "")


class TestNarrationNeverGetsACharacterVoice(unittest.TestCase):
    """A character-voiced entry must never also contain narration.

    The renderer gives one entry one voice, so a quotation merged with the
    attribution tag beside it means the tag is spoken by the character.
    """

    @staticmethod
    def _entries(chunk, speaker_of=lambda span: "ELENA"):
        # NOT labels_for(): that one forces every unquoted span to narration,
        # which is precisely the classifier mistake under test here.
        labels = [
            {"id": span.id, "speaker": speaker_of(span), "role": "dialogue",
             "instruct": "Flat."}
            for span in tokenize(chunk)
        ]
        entries, _, _ = run_chunk(FakeClient(FakeResponse(json.dumps(labels))), chunk)
        return entries

    def test_mislabelled_attribution_tag_is_handed_back_to_narrator(self):
        chunk = '"Go," he said.'
        entries = self._entries(chunk)

        self.assertEqual(joined(entries), chunk, "text must stay byte-identical")
        self.assertEqual([e["speaker"] for e in entries], ["ELENA", NARRATOR])
        self.assertEqual(entries[0]["text"], '"Go,"')
        self.assertEqual(entries[1]["text"], " he said.")

    def test_no_entry_mixes_quoted_and_unquoted_under_a_character(self):
        for name, chunk in FIXTURES.items():
            with self.subTest(fixture=name):
                entries = self._entries(chunk)
                self.assertEqual(joined(entries), chunk)
                for entry in entries:
                    if entry["speaker"] == NARRATOR:
                        continue
                    kinds = {span.kind for span in tokenize(entry["text"])
                             if entry["text"][span.start:span.end].strip()}
                    self.assertLessEqual(len(kinds), 1,
                                         f"{name}: mixed-kind character entry {entry!r}")

    def test_adjacent_quotations_by_one_speaker_still_merge(self):
        # Merging same-kind spans is the win worth keeping: one TTS call, not
        # two. Only the mixed-kind merge was ever the problem.
        chunk = '"Wait." "Please."'
        entries = self._entries(chunk)

        self.assertEqual(joined(entries), chunk)
        self.assertEqual([e["speaker"] for e in entries], ["ELENA"])

    def test_standalone_unquoted_speech_keeps_its_character(self):
        # Unquoted character speech is real (telepathy, silent speech, an
        # animal's voice). Nothing beside this span is a quotation by the same
        # speaker, so no structural rule can call it narration -- and a blanket
        # "unquoted implies NARRATOR" would silence every such line.
        chunk = "Come on, the dog said into her mind."
        entries = self._entries(chunk)

        self.assertEqual(joined(entries), chunk)
        self.assertEqual([e["speaker"] for e in entries], ["ELENA"])

    def test_merge_refuses_to_bridge_kinds_even_without_the_relabel(self):
        # build_entries' last line of defence, reached when adjacency did not
        # apply (a chunk boundary, a whitespace span between the two). Spans
        # are built by hand because the tokenizer never emits two consecutive
        # unquoted spans.
        source = '"Hi."\n\nElena left.'
        resolved = [
            (Span(1, 0, 5, "quoted"), "ELENA", None),
            (Span(2, 5, 7, "unquoted"), NARRATOR, None),      # the break
            (Span(3, 7, len(source), "unquoted"), "ELENA", None),
        ]
        entries = build_entries(resolved, source)

        self.assertEqual(joined(entries), source)
        self.assertEqual([e["text"] for e in entries], ['"Hi."\n\n', "Elena left."])

    def test_an_absorbed_paragraph_break_is_not_a_bridge(self):
        # The break between two quotations is absorbed into the preceding
        # entry; that must not let the narration after them merge back in.
        chunk = '"Hi."\n\n"Bye."\n\nElena put down the map.'
        entries = self._entries(chunk)

        self.assertEqual(joined(entries), chunk)
        self.assertEqual([e["speaker"] for e in entries], ["ELENA", NARRATOR])
        self.assertEqual(entries[0]["text"], '"Hi."\n\n"Bye."')


class TestNoWhitespaceOnlyEntries(unittest.TestCase):
    """A paragraph break between two speakers must not become its own entry."""

    def test_two_dialogue_paragraphs_produce_no_blank_entry(self):
        chunk = TWO_DIALOGUE_PARAGRAPHS
        speakers = iter(["ALPHA", "BETA"])
        labels = labels_for(chunk, speaker_of=lambda span, text: next(speakers))
        entries, _, _ = run_chunk(FakeClient(FakeResponse(json.dumps(labels))), chunk)

        self.assertEqual(joined(entries), chunk, "absorbing whitespace must change no bytes")
        for entry in entries:
            self.assertTrue(entry["text"].strip(), f"whitespace-only entry: {entry!r}")

        # The break lands on the PRECEDING entry.
        self.assertEqual([e["speaker"] for e in entries], ["ALPHA", "BETA"])
        self.assertEqual(entries[0]["text"], '"Hi."\n\n')
        self.assertEqual(entries[1]["text"], '"Bye."')

    def test_leading_whitespace_folds_into_the_following_entry(self):
        chunk = '\n\n"Hi."'
        entries, _, _ = run_chunk(
            FakeClient(FakeResponse(json.dumps(labels_for(chunk)))), chunk)

        self.assertEqual(joined(entries), chunk)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["speaker"], "ELENA")
        self.assertEqual(entries[0]["text"], chunk)

    def test_all_whitespace_chunk_is_preserved(self):
        # Degenerate but must not lose bytes or crash.
        chunk = "\n\n"
        entries, _, _ = run_chunk(FakeClient(FakeResponse("[]")), chunk)
        self.assertEqual(joined(entries), chunk)

    def test_no_whitespace_only_entries_across_every_fixture(self):
        for name, chunk in FIXTURES.items():
            for mode, response in (
                ("full", FakeResponse(json.dumps(labels_for(chunk)))),
                ("failure", RuntimeError("down")),
            ):
                with self.subTest(fixture=name, mode=mode):
                    entries, _, _ = run_chunk(FakeClient(response), chunk)
                    self.assertEqual(joined(entries), chunk)
                    for entry in entries:
                        self.assertTrue(entry["text"].strip(), f"{name}/{mode}: {entry!r}")


class TestSingleSpeakerCanonicalization(unittest.TestCase):
    """run_single_speaker must emit a canonical, comparison-safe speaker."""

    def _run(self, speaker_name, text="Line one.\n\nLine two."):
        captured = {}

        def fake_write(entries):
            captured["entries"] = entries

        original = generate_script._write_script_output
        generate_script._write_script_output = fake_write
        buffer = io.StringIO()
        try:
            with redirect_stdout(buffer):
                generate_script.run_single_speaker(text, speaker_name, "Neutral narration.")
        finally:
            generate_script._write_script_output = original
        return captured["entries"], buffer.getvalue()

    def test_default_narrator_is_uppercased(self):
        entries, output = self._run("Narrator")

        self.assertTrue(entries)
        for entry in entries:
            self.assertEqual(entry["speaker"], NARRATOR,
                             'must match the literal "NARRATOR" comparisons downstream')
        self.assertIn("Canonicalized", output)

    def test_custom_name_is_canonicalized(self):
        entries, _ = self._run("Dr. Vance")
        self.assertEqual({e["speaker"] for e in entries}, {"VANCE"})

    def test_already_canonical_name_is_untouched(self):
        entries, output = self._run("MARCUS")
        self.assertEqual({e["speaker"] for e in entries}, {"MARCUS"})
        self.assertNotIn("Canonicalized", output)

    def test_empty_name_falls_back_to_narrator(self):
        entries, _ = self._run("   ")
        self.assertEqual({e["speaker"] for e in entries}, {NARRATOR})

    def test_text_is_not_canonicalized(self):
        entries, _ = self._run("Narrator", text="Dr. Smith met Mr. Jones.")
        self.assertEqual(entries[0]["text"], "Dr. Smith met Mr. Jones.",
                         "canonicalization applies to LABELS only, never body text")


class TestRealResponseShapes(unittest.TestCase):
    """Shapes captured from live Ollama runs that the parser must recover."""

    # A 6-span chunk, so the 6-label real fixtures line up 1:1 with the spans.
    SIX_SPAN_CHUNK = '"One." said A. "Two." said B. "Three." he added.'

    def setUp(self):
        self.assertEqual(len(tokenize(self.SIX_SPAN_CHUNK)), 6,
                         "fixture chunk must tokenize to 6 spans")

    # --- shape (a): id-keyed JSON object ---------------------------------

    def test_id_keyed_object_parses_to_labels(self):
        labels = labels_from_id_keyed_object(REAL_ID_KEYED_OBJECT)

        self.assertEqual([label["id"] for label in labels], [1, 2, 3, 4, 5, 6])
        self.assertEqual(labels[5], {"id": 6, "speaker": "BILL", "role": "dialogue",
                                     "instruct": "Speaking with certainty"})

    def test_id_keyed_object_is_reported_as_the_object_mode(self):
        labels, mode = extract_labels(REAL_ID_KEYED_OBJECT)
        self.assertEqual(mode, LABEL_MODE_OBJECT)
        self.assertEqual(len(labels), 6)

    def test_object_key_wins_over_a_conflicting_inner_id(self):
        labels = labels_from_id_keyed_object(
            '{"7": {"id": 99, "speaker": "BILL", "role": "dialogue", "instruct": "x"}}')
        self.assertEqual(labels[0]["id"], 7)

    def test_id_keyed_object_recovery_is_not_degraded(self):
        chunk = self.SIX_SPAN_CHUNK
        entries, stats, output = run_chunk(
            FakeClient(FakeResponse(REAL_ID_KEYED_OBJECT)), chunk)

        self.assertEqual(joined(entries), chunk)
        self.assertEqual(stats["labelled"], 6, "all six spans labelled")
        self.assertEqual(stats["discarded"], 0)
        self.assertFalse(stats["degraded"],
                         "valid JSON with complete labels is a wrong envelope, not a degradation")
        self.assertEqual(stats["recovery"], LABEL_MODE_OBJECT)
        self.assertIn("id-keyed JSON object", output, "must print a one-line notice")
        self.assertIn("BILL", {entry["speaker"] for entry in entries})

    def test_ordinary_objects_are_not_mistaken_for_id_maps(self):
        for text in (
            '{"id": 1, "speaker": "BILL", "role": "dialogue", "instruct": "x"}',
            '{"labels": [{"id": 1}]}',
            '{"1": "NARRATOR", "2": "BILL"}',   # values not dicts
            '{}',
        ):
            with self.subTest(text=text[:40]):
                self.assertIsNone(labels_from_id_keyed_object(text))

    # --- shape (b): markdown label blocks --------------------------------

    def test_markdown_span_header_variant(self):
        labels = salvage_markdown_labels(REAL_MARKDOWN_SPAN)

        self.assertEqual([label["id"] for label in labels], [1, 2, 3])
        self.assertEqual(labels[0], {"id": 1, "speaker": "NARRATOR", "role": "narration",
                                     "instruct": "Neutral, even narration."})
        self.assertEqual(labels[2]["speaker"], "BILL")

    def test_markdown_id_header_variant(self):
        labels = salvage_markdown_labels(REAL_MARKDOWN_ID)

        self.assertEqual([label["id"] for label in labels], [1, 2, 3])
        self.assertEqual(labels[1]["role"], "dialogue")
        self.assertEqual(labels[1]["instruct"], "Confident, persuasive tone.",
                         "surrounding quotes and markdown hard-break spaces must be trimmed")

    def test_markdown_recovery_is_degraded_but_labels_apply(self):
        chunk = self.SIX_SPAN_CHUNK
        client = FakeClient(FakeResponse(REAL_MARKDOWN_SPAN))
        entries, stats, output = run_chunk(client, chunk)

        self.assertEqual(joined(entries), chunk, "prose byte-identical either way")
        self.assertEqual(stats["recovery"], LABEL_MODE_MARKDOWN)
        self.assertEqual(stats["labelled"], 3, "3 of 6 spans were labelled")
        self.assertEqual(stats["fallback"], 3)
        self.assertTrue(stats["degraded"], "ignoring the output contract must not exit 0")
        self.assertIn("markdown", stats["reason"].lower())
        self.assertIn("markdown blocks", output)
        self.assertIn("BILL", {entry["speaker"] for entry in entries})
        self.assertEqual(client.calls, 3,
                         "3 of 6 spans unlabelled -> the retry budget is spent trying to fill them")

    def test_markdown_is_last_resort_only(self):
        # A well-formed array that also happens to mention "Span 1" in prose
        # must still be read as an array.
        labels = json.dumps(labels_for(self.SIX_SPAN_CHUNK))
        text = "**Span 1** is tricky. Anyway:\n" + labels
        recovered, mode = extract_labels(text)
        self.assertEqual(mode, LABEL_MODE_ARRAY)
        self.assertEqual(len(recovered), 6)

    def test_prose_without_labels_recovers_nothing(self):
        self.assertEqual(extract_labels("I'm sorry, I cannot help with that."), (None, None))
        self.assertIsNone(salvage_markdown_labels("Here is a nice **bold** paragraph."))

    # --- item 3: think-tag pollution -------------------------------------

    def test_think_tags_are_stripped_before_salvage(self):
        labels = salvage_label_entries(REAL_THINK_POLLUTED_TRUNCATION)

        ids = [label["id"] for label in labels]
        self.assertEqual(ids, [1], "only the real truncated label survives")
        self.assertNotIn(500, ids, "phantom id from <think> reasoning was salvaged")
        self.assertNotIn(501, ids)

    def test_think_polluted_truncation_discards_no_phantom_ids(self):
        chunk = self.SIX_SPAN_CHUNK
        entries, stats, _ = run_chunk(
            FakeClient(FakeResponse(REAL_THINK_POLLUTED_TRUNCATION, finish_reason="length")),
            chunk)

        self.assertEqual(joined(entries), chunk)
        self.assertEqual(stats["discarded"], 0,
                         "phantom ids from reasoning text must never reach resolution")
        self.assertEqual(stats["labelled"], 1)
        self.assertEqual(stats["fallback"], 5)
        self.assertTrue(stats["degraded"])

    def test_unclosed_think_tag_is_stripped(self):
        self.assertEqual(strip_thinking_tags('ok<think>{"id": 9}').strip(), "ok")

    def test_extract_labels_modes_are_distinct(self):
        chunk = self.SIX_SPAN_CHUNK
        array_text = json.dumps(labels_for(chunk))
        cases = [
            (array_text, LABEL_MODE_ARRAY),
            # An array cut mid-object still repairs to an array when a complete
            # earlier object exists -- repair beats salvage, by design.
            (array_text[:-1].rstrip(), LABEL_MODE_ARRAY),
            (REAL_ID_KEYED_OBJECT, LABEL_MODE_OBJECT),
            # Real hard truncation from the log: nothing complete to repair.
            ('[{"id": 1, "speaker": "NARRATOR", "role": "narr', LABEL_MODE_SALVAGE),
            (REAL_MARKDOWN_ID, LABEL_MODE_MARKDOWN),
            ("nothing here", None),
        ]
        for text, expected in cases:
            with self.subTest(mode=expected):
                self.assertEqual(extract_labels(text)[1], expected)


class TestRetryOnIncompleteLabels(unittest.TestCase):
    """Live evidence: the model closes the array early, dropping a contiguous
    SUFFIX of span ids at finish_reason=stop (logged: missing [28] of 28, and
    [27, 28] of 28), and separately invents role values ("thought" on a real
    spoken line). Both are recoverable by re-asking for the named gaps."""

    CHUNK = '"One." said A. "Two." said B. "Three." he added.'   # 6 spans

    def setUp(self):
        self.spans = tokenize(self.CHUNK)
        self.assertEqual(len(self.spans), 6)

    def _labels(self, ids, speaker="ELENA", role=None, instruct="Firm."):
        out = []
        for span in self.spans:
            if span.id not in ids:
                continue
            label = {"id": span.id}
            is_quoted = span.kind == "quoted"
            label["speaker"] = speaker if is_quoted else "NARRATOR"
            resolved_role = role if role is not None else ("dialogue" if is_quoted else "narration")
            if resolved_role:
                label["role"] = resolved_role
            label["instruct"] = instruct
            out.append(label)
        return out

    def _json(self, ids, **kw):
        return json.dumps(self._labels(ids, **kw))

    # --- the headline case ------------------------------------------------

    def test_missing_suffix_is_recovered_on_retry(self):
        client = FakeClient(
            FakeResponse(self._json([1, 2, 3, 4, 5])),   # id 6 dropped, like the log
            FakeResponse(self._json([6])),               # retry supplies the gap
        )
        entries, stats, output = run_chunk(client, self.CHUNK)

        self.assertEqual(client.calls, 2, "one retry, no more")
        self.assertEqual(joined(entries), self.CHUNK)
        self.assertEqual(stats["labelled"], 6)
        self.assertEqual(stats["fallback"], 0)
        self.assertFalse(stats["degraded"], "a chunk fully labelled on retry is NOT degraded")
        self.assertIn("retrying", output.lower())
        self.assertIn("Retry recovered 1 new label(s)", output)

    def test_retry_prompt_names_the_missing_ids(self):
        client = FakeClient(
            FakeResponse(self._json([1, 2, 3, 4, 5])),
            FakeResponse(self._json([6])),
        )
        run_chunk(client, self.CHUNK)

        first, second = client.user_prompts[0], client.user_prompts[1]
        self.assertNotIn("CORRECTION", first, "attempt 1 must be the plain prompt")
        self.assertTrue(second.startswith(first),
                        "the nudge is appended, so the cached prefix is preserved")
        self.assertIn("CORRECTION", second)
        self.assertIn("still missing a label: 6", second)

    def test_role_missing_is_fixed_on_retry(self):
        # Logged live: {"id": 15, "speaker": "BILL", "role": "thought", ...}
        # on a genuinely spoken quoted span.
        bad = self._labels([1, 2, 3, 4, 5, 6])
        bad[0]["role"] = "thought"
        client = FakeClient(
            FakeResponse(json.dumps(bad)),
            FakeResponse(self._json([1])),
        )
        entries, stats, output = run_chunk(client, self.CHUNK)

        self.assertEqual(client.calls, 2)
        self.assertEqual(stats["role_missing"], 0, "the invalid role was filled in")
        self.assertFalse(stats["degraded"])
        self.assertEqual(entries[0]["speaker"], "ELENA")
        self.assertIn("unusable role", output)
        self.assertIn("neither \"dialogue\" nor \"narration\": 1", client.user_prompts[1])

    def test_dialogue_without_speaker_is_retried(self):
        # role is good and says "dialogue", but no name came with it. Before the
        # predicate had a branch for it this was neither missing, nor bad-role,
        # nor unattested, so the loop broke on attempt 1 and the span was
        # narrated in silence -- 26/27 such labels on a live chunk, ZERO retries.
        bad = self._labels([1, 2, 3, 4, 5, 6])
        bad[0]["speaker"] = ""
        client = FakeClient(
            FakeResponse(json.dumps(bad)),
            FakeResponse(self._json([1])),
        )
        entries, stats, output = run_chunk(client, self.CHUNK)

        self.assertEqual(client.calls, 2, "the gap must be retried, not swallowed")
        self.assertEqual(stats["dialogue_without_speaker"], 0, "the retry filled it")
        self.assertFalse(stats["degraded"])
        self.assertEqual(entries[0]["speaker"], "ELENA")
        self.assertEqual(joined(entries), self.CHUNK)
        self.assertIn("no speaker", output)
        self.assertIn('no usable "speaker"', client.user_prompts[1])
        self.assertIn("1", client.user_prompts[1])

    def test_narrator_dialogue_label_is_not_retried(self):
        # NARRATOR is a usable name, so resolve_span_labels does not count it as
        # dialogue_without_speaker -- the predicate must not nag about it either.
        labels = self._labels([1, 2, 3, 4, 5, 6], speaker="NARRATOR")
        for label in labels:
            label["role"] = "dialogue"
        client = FakeClient(FakeResponse(json.dumps(labels)))
        _, stats, _ = run_chunk(client, self.CHUNK)
        self.assertEqual(client.calls, 1, "nothing to correct")
        self.assertEqual(stats["dialogue_without_speaker"], 0)

    def test_predicate_and_resolver_agree_on_every_label_shape(self):
        # The recurring failure mode in this file: _incomplete_span_ids and
        # resolve_span_labels drift apart, so a label is either nagged about
        # without ever being counted, or counted without ever being retried.
        # Property, not fixture: sweep the label shapes and assert agreement.
        speakers = [None, "", "   ", "ELENA", "NARRATOR", "narrator", 7]
        roles = [None, "", "dialogue", "DIALOGUE ", "narration", "thought"]
        for speaker in speakers:
            for role in roles:
                label = {"id": 1}
                if speaker is not None:
                    label["speaker"] = speaker
                if role is not None:
                    label["role"] = role
                with self.subTest(speaker=speaker, role=role):
                    _, bad_role, no_speaker = generate_script._incomplete_span_ids(
                        self.spans, {1: label}, source=self.CHUNK)
                    _, stats = resolve_span_labels(
                        self.spans, [label], source=self.CHUNK, roster={})
                    self.assertEqual(
                        1 in bad_role, bool(stats["role_missing"]),
                        "bad_role must mirror role_missing exactly")
                    self.assertEqual(
                        1 in no_speaker, bool(stats["dialogue_without_speaker"]),
                        "no_speaker must mirror dialogue_without_speaker exactly")

    # --- monotonicity: a retry can never make things worse ----------------

    def test_worse_retry_cannot_overwrite_good_labels(self):
        good = self._labels([1, 2, 3, 4, 5], speaker="ELENA", instruct="Firm.")
        worse = [{"id": 1, "speaker": "WRONG", "instruct": "Sloppy."}]  # no role
        client = FakeClient(FakeResponse(json.dumps(good)), FakeResponse(json.dumps(worse)))
        entries, stats, _ = run_chunk(client, self.CHUNK)

        self.assertEqual(joined(entries), self.CHUNK)
        self.assertEqual(entries[0]["speaker"], "ELENA", "attempt 1's label must survive")
        self.assertEqual(entries[0]["instruct"], "Firm.")
        self.assertEqual(stats["fallback"], 1, "only span 6 was never labelled")
        self.assertTrue(stats["degraded"])
        self.assertEqual(client.calls, 3, "budget exhausted trying to fill span 6")

    def test_equal_field_from_a_later_attempt_does_not_replace_it(self):
        first = [{"id": 1, "speaker": "ELENA", "role": "dialogue", "instruct": "Firm."}]
        second = [{"id": 1, "speaker": "BETA", "role": "dialogue", "instruct": "Soft."}]
        client = FakeClient(FakeResponse(json.dumps(first)), FakeResponse(json.dumps(second)))
        entries, _, _ = run_chunk(client, self.CHUNK)

        self.assertEqual(entries[0]["speaker"], "ELENA")
        self.assertEqual(entries[0]["instruct"], "Firm.")

    def test_blank_fields_are_filled_but_present_ones_are_not(self):
        first = [{"id": 1, "speaker": "", "role": "dialogue", "instruct": "Firm."}]
        second = [{"id": 1, "speaker": "BETA", "role": "narration", "instruct": "Soft."}]
        client = FakeClient(FakeResponse(json.dumps(first)), FakeResponse(json.dumps(second)))
        entries, _, _ = run_chunk(client, self.CHUNK)

        # speaker was blank -> filled from attempt 2; role/instruct were present -> kept.
        self.assertEqual(entries[0]["speaker"], "BETA")
        self.assertEqual(entries[0]["instruct"], "Firm.")

    def test_good_first_attempt_survives_a_garbage_retry(self):
        client = FakeClient(
            FakeResponse(self._json([1, 2, 3, 4, 5])),
            FakeResponse("I'm sorry, I cannot help with that."),
        )
        entries, stats, _ = run_chunk(client, self.CHUNK)

        self.assertEqual(joined(entries), self.CHUNK)
        self.assertEqual(stats["labelled"], 5, "attempt 1's labels must not be dropped")
        self.assertEqual(stats["fallback"], 1)

    # --- completeness is membership, not a count --------------------------

    def test_hallucinated_ids_do_not_fake_completeness(self):
        # Logged live: 30 labels for 28 spans, and 137 labels for 41 spans.
        labels = self._labels([1, 2, 3, 4, 5]) + [
            {"id": 90, "speaker": "GHOST", "role": "dialogue", "instruct": "x"},
            {"id": 91, "speaker": "GHOST", "role": "dialogue", "instruct": "x"},
            {"id": 92, "speaker": "GHOST", "role": "dialogue", "instruct": "x"},
        ]
        client = FakeClient(FakeResponse(json.dumps(labels)))
        entries, stats, _ = run_chunk(client, self.CHUNK)

        self.assertEqual(joined(entries), self.CHUNK)
        self.assertEqual(client.calls, 3, "8 labels for 6 spans must not count as complete")
        self.assertEqual(stats["labelled"], 5)
        self.assertEqual(stats["fallback"], 1)
        self.assertEqual(stats["discarded"], 3, "distinct phantom ids, counted once each")

    # --- the truncation gate ----------------------------------------------

    def test_truncated_partial_is_accepted_without_retry(self):
        client = FakeClient(
            FakeResponse(self._json([1, 2, 3]), finish_reason="length"))
        entries, stats, _ = run_chunk(client, self.CHUNK)

        self.assertEqual(client.calls, 1, "re-rolling at the same max_tokens is pointless")
        self.assertEqual(joined(entries), self.CHUNK)
        self.assertEqual(stats["fallback"], 3)
        self.assertTrue(stats["degraded"])

    def test_truncated_with_nothing_usable_still_retries(self):
        # A truncated response we could not parse at all is a different failure
        # from a truncated response that gave us partial labels.
        client = FakeClient(FakeResponse("blah blah", finish_reason="length"))
        _, stats, _ = run_chunk(client, self.CHUNK)

        self.assertEqual(client.calls, 3)
        self.assertTrue(stats["degraded"])

    # --- budget --------------------------------------------------------------

    def test_never_exceeds_the_existing_budget(self):
        client = FakeClient(FakeResponse(self._json([1])))
        run_chunk(client, self.CHUNK, max_retries=2)
        self.assertEqual(client.calls, 3)

        client = FakeClient(FakeResponse(self._json([1])))
        run_chunk(client, self.CHUNK, max_retries=0)
        self.assertEqual(client.calls, 1, "max_retries=0 means exactly one attempt")

    def test_worst_recovery_mode_is_reported_across_attempts(self):
        # attempt 1 array (partial) + attempt 2 markdown -> markdown must win,
        # so the contract violation is not hidden behind the array.
        client = FakeClient(
            FakeResponse(self._json([1, 2, 3, 4, 5])),
            FakeResponse(REAL_MARKDOWN_SPAN),
        )
        _, stats, _ = run_chunk(client, self.CHUNK)
        self.assertEqual(stats["recovery"], LABEL_MODE_MARKDOWN)
        self.assertTrue(stats["degraded"])


class TestWhitespaceSpansOutOfTheLoop(unittest.TestCase):
    """A "\\n\\n" between two quoted paragraphs has no audible content and its
    label is discarded by _absorb_whitespace_groups regardless. It must never
    reach the model, never cost a retry, and never degrade a chunk."""

    CHUNK = TWO_DIALOGUE_PARAGRAPHS          # '"Hi."\n\n"Bye."'

    def setUp(self):
        self.spans = tokenize(self.CHUNK)
        self.assertEqual(len(self.spans), 3)
        self.assertEqual([s.id for s in self.spans], [1, 2, 3])
        self.assertTrue(is_whitespace_span(self.spans[1], self.CHUNK),
                        "span 2 is the paragraph break")

    def _visible_labels(self):
        return [
            {"id": 1, "speaker": "ALPHA", "role": "dialogue", "instruct": "Bright."},
            {"id": 3, "speaker": "BETA", "role": "dialogue", "instruct": "Flat."},
        ]

    # --- the payload -------------------------------------------------------

    def test_payload_omits_the_whitespace_span(self):
        payload = build_span_payload(self.spans, self.CHUNK)
        lines = payload.split("\n")

        self.assertEqual(len(lines), 2, "3 spans, 1 of them whitespace -> 2 payload lines")
        ids = [json.loads(line)["id"] for line in lines]
        self.assertEqual(ids, [1, 3], "ids stay the tokenizer's; the gap is the omission")
        self.assertNotIn('"id": 2', payload)

    def test_payload_ids_are_stable_not_renumbered(self):
        # The reassembly contract keys on tokenizer ids; renumbering would
        # silently misalign every label after a whitespace span.
        payload = build_span_payload(tokenize(STRAIGHT_QUOTES), STRAIGHT_QUOTES)
        ids = [json.loads(line)["id"] for line in payload.split("\n")]
        self.assertEqual(ids, sorted(ids))

    # --- resolution and degradation ---------------------------------------

    def test_labelling_only_visible_spans_is_not_degraded(self):
        client = FakeClient(FakeResponse(json.dumps(self._visible_labels())))
        entries, stats, output = run_chunk(client, self.CHUNK)

        self.assertEqual(client.calls, 1, "no retry for the unlabelled whitespace span")
        self.assertEqual(joined(entries), self.CHUNK, "byte-identity unaffected")
        self.assertFalse(stats["degraded"])
        self.assertEqual(stats["fallback"], 0, "whitespace is not a fallback")
        self.assertEqual(stats["whitespace"], 1)
        self.assertEqual(stats["labelled"] + stats["fallback"], stats["spans"],
                         "the accounting invariant test_api.py relies on")
        self.assertNotIn("DEGRADED", output)

    def test_whitespace_is_still_absorbed_into_the_preceding_entry(self):
        client = FakeClient(FakeResponse(json.dumps(self._visible_labels())))
        entries, _, _ = run_chunk(client, self.CHUNK)

        self.assertEqual([e["speaker"] for e in entries], ["ALPHA", "BETA"])
        self.assertEqual(entries[0]["text"], '"Hi."\n\n')
        self.assertEqual(entries[1]["text"], '"Bye."')
        for entry in entries:
            self.assertTrue(entry["text"].strip())

    def test_a_real_missing_span_still_degrades(self):
        # Guard against over-suppression: omitting id 3 must still be caught.
        client = FakeClient(FakeResponse(json.dumps(self._visible_labels()[:1])))
        _, stats, _ = run_chunk(client, self.CHUNK)

        self.assertEqual(client.calls, 3, "a genuinely missing visible span is retried")
        self.assertEqual(stats["fallback"], 1)
        self.assertTrue(stats["degraded"])

    def test_phantom_label_for_a_whitespace_id_is_ignored(self):
        labels = self._visible_labels() + [
            {"id": 2, "speaker": "GHOST", "role": "dialogue", "instruct": "x"}]
        client = FakeClient(FakeResponse(json.dumps(labels)))
        entries, stats, _ = run_chunk(client, self.CHUNK)

        self.assertEqual(joined(entries), self.CHUNK)
        self.assertNotIn("GHOST", {e["speaker"] for e in entries})
        self.assertFalse(stats["degraded"])

    def test_all_whitespace_chunk_skips_the_llm_entirely(self):
        client = FakeClient(FakeResponse("[]"))
        entries, stats, _ = run_chunk(client, "\n\n")

        self.assertEqual(client.calls, 0, "nothing audible to classify")
        self.assertEqual(joined(entries), "\n\n")
        self.assertFalse(stats["degraded"])
        self.assertEqual(stats["whitespace"], stats["spans"])

    def test_resolve_span_labels_without_source_is_unchanged(self):
        # Back-compat: direct callers that pass no source get the old behavior.
        resolved, stats = resolve_span_labels(self.spans, self._visible_labels())
        self.assertEqual(stats["whitespace"], 0)
        self.assertEqual(stats["fallback"], 1, "span 2 counts as fallback without source")


class TestLogIsolation(unittest.TestCase):
    """The suite must not append to the production forensic log."""

    def test_log_dir_honours_the_env_override(self):
        self.assertEqual(llm_log_dir(), os.environ["ALEXANDRIA_LLM_LOG_DIR"])
        self.assertEqual(llm_log_dir(), _LOG_TMPDIR)

    def test_process_chunk_writes_to_the_override_dir_only(self):
        production_log = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "logs", "llm_responses.log")
        before = os.path.getsize(production_log) if os.path.exists(production_log) else None

        run_chunk(FakeClient(FakeResponse("[]")), '"Hi." he said.')

        override_log = os.path.join(_LOG_TMPDIR, "llm_responses.log")
        self.assertTrue(os.path.exists(override_log), "log must land in the override dir")
        self.assertIn("CHUNK", io.open(override_log, encoding="utf-8").read())

        after = os.path.getsize(production_log) if os.path.exists(production_log) else None
        self.assertEqual(before, after, "production log must not grow during tests")

    def test_default_is_unchanged_without_the_env_var(self):
        saved = os.environ.pop("ALEXANDRIA_LLM_LOG_DIR")
        try:
            self.assertEqual(llm_log_dir(), generate_script.LLM_LOG_DIR)
            self.assertTrue(llm_log_dir().replace("\\", "/").endswith("/logs"))
        finally:
            os.environ["ALEXANDRIA_LLM_LOG_DIR"] = saved


class TestSalvageAndPayload(unittest.TestCase):
    def test_salvage_label_entries_is_field_order_agnostic(self):
        text = '[{"role": "dialogue", "instruct": "Calm.", "speaker": "MARCUS", "id": 4}]'
        labels = salvage_label_entries(text)
        self.assertEqual(labels, [{"id": 4, "speaker": "MARCUS", "role": "dialogue", "instruct": "Calm."}])

    def test_salvage_recovers_truncated_tail_object(self):
        text = '[{"id": 1, "speaker": "A", "role": "dialogue", "instruct": "x"}, {"id": 2, "speaker": "B"'
        labels = salvage_label_entries(text)
        self.assertEqual([label["id"] for label in labels], [1, 2])

    # --- unquoted (bareword) values: complete output, invalid JSON ---------
    #
    # Observed live: {"id": 1, "speaker": KIT, "role": dialogue, "instruct": "x"}
    # at finish_reason=stop. The quoted-only patterns dropped speaker and role in
    # silence, collapsing whole chunks into a single NARRATOR entry.

    def test_salvage_recovers_bareword_speaker_and_role(self):
        text = '[{"id": 1, "speaker": KIT, "role": dialogue, "instruct": "Angry."}]'
        labels = salvage_label_entries(text)
        self.assertEqual(labels, [{"id": 1, "speaker": "KIT", "role": "dialogue",
                                   "instruct": "Angry."}])

    def test_bareword_value_does_not_bleed_into_the_next_key(self):
        # The terminator rules are the whole safety story here: a greedy value
        # would swallow `, "role": ...` and produce a speaker named after the
        # rest of the object.
        cases = [
            '[{"id": 2, "speaker": ROSHAUN\'S FATHER, "role": dialogue, "instruct": "x"}]',
            '[{"id": 2, "role": dialogue, "speaker": ROSHAUN\'S FATHER, "instruct": "x"}]',
            '[{"id": 2, "instruct": "x", "role": dialogue, "speaker": ROSHAUN\'S FATHER}]',
        ]
        for text in cases:
            with self.subTest(text=text):
                label = salvage_label_entries(text)[0]
                self.assertEqual(label["speaker"], "ROSHAUN'S FATHER")
                self.assertEqual(label["role"], "dialogue")

    def test_bareword_fallback_never_preempts_a_quoted_value(self):
        text = '[{"id": 3, "speaker": "MARCUS", "role": "narration", "instruct": "Calm."}]'
        self.assertEqual(
            salvage_label_entries(text),
            [{"id": 3, "speaker": "MARCUS", "role": "narration", "instruct": "Calm."}])

    def test_bareword_json_literals_are_not_names(self):
        text = '[{"id": 4, "speaker": null, "role": dialogue, "instruct": "x"}]'
        label = salvage_label_entries(text)[0]
        self.assertNotIn("speaker", label, "`null` means no answer, not a character")

    def test_bareword_instruct_is_not_salvaged(self):
        # Deliberate: instruct was always quoted in the logged failures, and it
        # is free prose whose commas make an unterminated scan unsafe.
        text = '[{"id": 5, "speaker": KIT, "role": dialogue, "instruct": Angry, loud.}]'
        label = salvage_label_entries(text)[0]
        self.assertEqual(label["speaker"], "KIT")
        self.assertNotIn("instruct", label)

    def test_bareword_salvage_is_announced(self):
        chunk = '"One." said A.'
        text = '[{"id": 1, "speaker": KIT, "role": dialogue, "instruct": "Angry."}]'
        client = FakeClient(FakeResponse(text))
        entries, _, output = run_chunk(client, chunk)
        self.assertEqual(joined(entries), chunk)
        self.assertEqual(entries[0]["speaker"], "KIT",
                         "the cast must survive a model that forgot its quotes")
        self.assertIn("Salvaged unquoted", output)

    def test_parse_label_response_handles_clean_json(self):
        text = '[{"id": 1, "speaker": "A", "role": "dialogue", "instruct": "x"}]'
        self.assertEqual(len(parse_label_response(text)), 1)

    def test_span_payload_never_contains_offsets_and_is_one_line_per_span(self):
        spans = tokenize(STRAIGHT_QUOTES)
        visible = [s for s in spans if not is_whitespace_span(s, STRAIGHT_QUOTES)]
        self.assertLess(len(visible), len(spans), "fixture should have a whitespace span")

        payload = build_span_payload(spans, STRAIGHT_QUOTES)
        lines = payload.split("\n")
        self.assertEqual(len(lines), len(visible), "one line per VISIBLE span")
        for line in lines:
            record = json.loads(line)
            self.assertEqual(set(record.keys()), {"id", "kind", "text"})
            self.assertTrue(record["text"].strip(), "whitespace spans are never shown")

    def test_old_salvage_export_still_present_for_review_script(self):
        # review_script.py imports these three by name.
        for name in ("clean_json_string", "repair_json_array", "salvage_json_entries"):
            self.assertTrue(callable(getattr(generate_script, name)), name)


class TestPromptTemplate(unittest.TestCase):
    def test_user_template_formats_without_brace_errors(self):
        rendered = DEFAULT_USER_PROMPT.format(context="CTX", chunk="SPANS")
        self.assertIn("CTX", rendered)
        self.assertIn("SPANS", rendered)
        # Doubled braces in the template collapse to a literal JSON-ish hint.
        self.assertIn('{"id", "speaker", "role", "instruct"}', rendered)

    def test_system_prompt_forbids_emitting_text(self):
        self.assertIn('"role"', DEFAULT_SYSTEM_PROMPT)
        self.assertIn("NEVER output a \"text\" field", DEFAULT_SYSTEM_PROMPT)

    def test_both_halves_carry_the_schema_marker(self):
        self.assertIn(PROMPT_SCHEMA_MARKER, DEFAULT_SYSTEM_PROMPT)
        self.assertIn(PROMPT_SCHEMA_MARKER, DEFAULT_USER_PROMPT)


class TestPromptSchemaGuard(unittest.TestCase):
    """Saved config.json prompts written for the retired schema must not be used."""

    # A verbatim shape of the OLD generation prompt: no span-labels-v1 marker.
    STALE_PROMPT = (
        "You are a script writer converting books into audiobook scripts.\n"
        'Output [{"speaker": "NARRATOR", "text": "...", "instruct": "..."}]\n'
        "Drop attribution tags. Convert \"Dr.\" to \"Doctor\".\n"
    )

    def _select(self, custom, default, key):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            chosen = select_prompt(custom, default, key)
        return chosen, buffer.getvalue()

    def test_stale_custom_prompt_falls_back_to_default_with_warning(self):
        chosen, output = self._select(
            self.STALE_PROMPT, DEFAULT_SYSTEM_PROMPT, "prompts.system_prompt")

        self.assertEqual(chosen, DEFAULT_SYSTEM_PROMPT, "must fall back to the built-in default")
        self.assertNotIn("script writer", chosen)
        self.assertIn("WARNING", output)
        self.assertIn("prompts.system_prompt", output, "warning must name the config key")
        self.assertIn(PROMPT_SCHEMA_MARKER, output)

    def test_stale_user_template_names_its_own_config_key(self):
        chosen, output = self._select(
            "{context}\n\nSOURCE TEXT:\n{chunk}", DEFAULT_USER_PROMPT, "prompts.user_prompt")

        self.assertEqual(chosen, DEFAULT_USER_PROMPT)
        self.assertIn("prompts.user_prompt", output)
        self.assertNotIn("prompts.system_prompt", output)

    def test_marker_bearing_custom_prompt_is_used_as_is(self):
        custom = (
            "Custom classifier prompt (prompt schema: span-labels-v1).\n"
            "Return only labels.\n"
        )
        chosen, output = self._select(custom, DEFAULT_SYSTEM_PROMPT, "prompts.system_prompt")

        self.assertEqual(chosen, custom, "a marker-bearing custom prompt must be honoured verbatim")
        self.assertEqual(output, "", "no warning for a current-schema custom prompt")

    def test_absent_or_blank_custom_prompt_uses_default_silently(self):
        for value in (None, "", "   "):
            with self.subTest(value=repr(value)):
                chosen, output = self._select(value, DEFAULT_SYSTEM_PROMPT, "prompts.system_prompt")
                self.assertEqual(chosen, DEFAULT_SYSTEM_PROMPT)
                self.assertEqual(output, "", "an unset prompt is not a misconfiguration")

    def test_defaults_pass_their_own_guard(self):
        # The shipped defaults must never be rejected by the guard.
        for prompt, key in ((DEFAULT_SYSTEM_PROMPT, "prompts.system_prompt"),
                            (DEFAULT_USER_PROMPT, "prompts.user_prompt")):
            chosen, output = self._select(prompt, prompt, key)
            self.assertEqual(chosen, prompt)
            self.assertEqual(output, "")

    def test_stale_prompt_would_have_degraded_the_chunk(self):
        # Why the guard exists: old-schema output carries no ids, so every
        # span falls back to NARRATOR. Prose survives; all voices are lost.
        chunk = STRAIGHT_QUOTES
        old_style = json.dumps([
            {"speaker": "ELENA", "text": "I am leaving.", "instruct": "Firm."},
            {"speaker": "NARRATOR", "text": "Marcus did not look up.", "instruct": "Even."},
        ])
        client = FakeClient(FakeResponse(old_style))
        entries, stats, _ = run_chunk(client, chunk)

        self.assertEqual(joined(entries), chunk)
        self.assertEqual(stats["labelled"] - stats["whitespace"], 0,
                         "old-schema output carries no ids, so nothing is labelled")
        self.assertTrue(stats["degraded"])


class TestVerbatimAcrossAllFixtures(unittest.TestCase):
    """Property test: every fixture x every failure mode is byte-identical."""

    def _responses_for(self, chunk):
        full = json.dumps(labels_for(chunk))
        return {
            "full": FakeResponse(full),
            "truncated": FakeResponse(full[: len(full) // 2], finish_reason="length"),
            "empty_array": FakeResponse("[]"),
            "prose": FakeResponse("Sure! Here is the script."),
            "wrong_ids": FakeResponse('[{"id": 500, "speaker": "X", "role": "dialogue", "instruct": "y"}]'),
        }

    def test_all_fixtures_all_modes(self):
        for fixture_name, chunk in FIXTURES.items():
            for mode, response in self._responses_for(chunk).items():
                with self.subTest(fixture=fixture_name, mode=mode):
                    client = FakeClient(response)
                    entries, stats, _ = run_chunk(client, chunk)
                    self.assertEqual(joined(entries), chunk)
                    self.assertEqual(stats["labelled"] + stats["fallback"], stats["spans"])



# --------------------------------------------------------------------------
# F15: roster-aware spelling resolution + bounded roster context
# --------------------------------------------------------------------------

class TestRosterAwareSpellingAtGenerationSeam(unittest.TestCase):
    """process_chunk/resolve_span_labels accept labels THROUGH the roster.

    Exact whitespace-stripped key equality only; never fuzzy merging.
    """

    def _chunk_with_speaker(self, name):
        chunk = '"I am leaving," he said.'
        client = FakeClient(FakeResponse(json.dumps(
            labels_for(chunk, speaker_of=lambda span, text: name))))
        return chunk, client

    def test_drifted_spelling_snaps_onto_established_name(self):
        chunk, client = self._chunk_with_speaker("ABBEMARIGNAN")
        roster = {"ABBEMARIGNAN": "ABBE MARIGNAN"}
        entries, _, _ = run_chunk(client, chunk, roster=roster)
        self.assertEqual(joined(entries), chunk)
        self.assertIn("ABBE MARIGNAN", [e["speaker"] for e in entries])
        # Caller's roster index must not be mutated by the chunk.
        self.assertEqual(roster, {"ABBEMARIGNAN": "ABBE MARIGNAN"})

    def test_more_segmented_spelling_wins_even_if_seen_second(self):
        # Order-independence: the roster already holds the malformed form, and
        # the well-formed one arriving later still wins (and is promoted).
        chunk, client = self._chunk_with_speaker("ABBE MARIGNAN")
        roster = {"ABBEMARIGNAN": "ABBEMARIGNAN"}
        entries, _, _ = run_chunk(client, chunk, roster=roster)
        self.assertIn("ABBE MARIGNAN", [e["speaker"] for e in entries])
        # ... without mutating the caller's index.
        self.assertEqual(roster, {"ABBEMARIGNAN": "ABBEMARIGNAN"})

    def test_similar_names_are_never_merged(self):
        for established, incoming in (("JON", "JOHN"), ("ELLA", "BELLA")):
            with self.subTest(established=established):
                chunk, client = self._chunk_with_speaker(incoming)
                entries, _, _ = run_chunk(
                    client, chunk, roster={established: established})
                self.assertIn(incoming, [e["speaker"] for e in entries])

    def test_no_roster_is_backward_compatible(self):
        chunk, client = self._chunk_with_speaker("ABBEMARIGNAN")
        entries, _, _ = run_chunk(client, chunk)
        self.assertEqual(joined(entries), chunk)
        self.assertIn("ABBEMARIGNAN", [e["speaker"] for e in entries])


class TestBoundedRosterContext(unittest.TestCase):
    """build_context()'s roster block is capped, configurable and honest."""

    @staticmethod
    def _entries(count):
        return [{"speaker": f"CHAR{i:03d}", "text": "x", "instruct": "y"}
                for i in range(count)]

    @staticmethod
    def _roster_line(context):
        for line in context.splitlines():
            if line.startswith("Characters in this book"):
                return line
        return None

    def _names(self, context):
        line = self._roster_line(context)
        return [n.strip() for n in line.split(":", 1)[1].split(",")]

    def test_small_roster_is_unchanged_and_unannotated(self):
        context = build_context(2, 10, self._entries(3))
        line = self._roster_line(context)
        self.assertEqual(line, "Characters in this book: CHAR000, CHAR001, CHAR002")

    def test_roster_is_capped_at_the_default_limit(self):
        entries = self._entries(600)
        names = self._names(build_context(2, 10, entries))
        self.assertEqual(len(names), MAX_CONTEXT_ROSTER_NAMES)

    def test_cap_keeps_the_most_recent_names(self):
        entries = self._entries(600)
        names = self._names(build_context(2, 10, entries))
        # Most recently spoken = the highest-numbered CHARs.
        self.assertEqual(sorted(names), sorted(
            [f"CHAR{i:03d}" for i in range(600 - MAX_CONTEXT_ROSTER_NAMES, 600)]))

    def test_truncation_is_announced_only_when_it_happened(self):
        truncated = self._roster_line(build_context(2, 10, self._entries(600)))
        self.assertIn(f"most recent {MAX_CONTEXT_ROSTER_NAMES}", truncated)
        # No "of M": computing the total would mean scanning every entry.
        self.assertNotIn("of 600", truncated)
        untouched = self._roster_line(build_context(2, 10, self._entries(5)))
        self.assertNotIn("most recent", untouched)

    def test_cap_is_configurable(self):
        entries = self._entries(600)
        self.assertEqual(len(self._names(build_context(2, 10, entries,
                                                       max_roster_names=5))), 5)
        self.assertEqual(len(self._names(build_context(2, 10, entries,
                                                       max_roster_names=200))), 200)

    def test_deterministic_across_runs(self):
        entries = self._entries(600)
        first = build_context(2, 10, entries)
        for _ in range(3):
            self.assertEqual(build_context(2, 10, entries), first)

    def test_narrator_excluded_and_prompt_stays_bounded(self):
        entries = self._entries(600) + [{"speaker": NARRATOR, "text": "x", "instruct": "y"}]
        line = self._roster_line(build_context(2, 10, entries))
        self.assertNotIn(NARRATOR, line)
        self.assertLess(len(line), 1500)




class TestSilentPromptTruncationTripwire(unittest.TestCase):
    """A server that silently truncates the prompt must not go unnoticed.

    Production: prompt_tokens pinned at exactly 2050 for 640 of 1,294 records
    (Ollama's default num_ctx=2048), with a green summary throughout.
    """

    def test_ratio_check_fires_when_tokens_are_implausibly_low(self):
        # 40,000 chars is ~10,000 tokens of English; 2,050 reported means the
        # server kept about a fifth of it.
        self.assertTrue(looks_silently_truncated(40000, 2050))

    def test_ratio_check_silent_for_a_normal_ratio(self):
        self.assertFalse(looks_silently_truncated(8000, 2000))
        self.assertFalse(looks_silently_truncated(8000, 1300))

    def test_no_false_alarm_on_high_token_density_text(self):
        # CJK source text tokenizes to far MORE tokens per char, which can only
        # push prompt_tokens up relative to the chars/4 estimate.
        cjk_chars = 4000
        for prompt_tokens in (cjk_chars, cjk_chars * 2, cjk_chars // 2):
            with self.subTest(prompt_tokens=prompt_tokens):
                self.assertFalse(looks_silently_truncated(cjk_chars, prompt_tokens))

    def test_missing_or_zero_usage_is_never_a_warning(self):
        self.assertFalse(looks_silently_truncated(40000, None))
        self.assertFalse(looks_silently_truncated(40000, 0))
        self.assertFalse(looks_silently_truncated(0, 10))

    def test_estimate_is_chars_over_four(self):
        self.assertEqual(estimate_prompt_tokens("x" * 4000), 1000)
        self.assertEqual(estimate_prompt_tokens(""), 0)
        self.assertEqual(estimate_prompt_tokens(None), 0)

    def test_warning_fires_through_process_chunk_and_is_counted(self):
        chunk = STRAIGHT_QUOTES
        client = FakeClient(FakeResponse(json.dumps(labels_for(chunk)), prompt_tokens=1))
        entries, stats, output = run_chunk(client, chunk)
        self.assertEqual(joined(entries), chunk)          # prose still intact
        self.assertIn("PROMPT LIKELY TRUNCATED", output)
        self.assertIn("OLLAMA_CONTEXT_LENGTH", output)
        self.assertEqual(stats["prompt_truncation_events"], 1)
        self.assertGreater(stats["prompt_chars"], 0)
        # Diagnostic only -- it must not by itself degrade the chunk.
        self.assertFalse(stats["degraded"])

    def test_no_warning_for_a_healthy_call(self):
        chunk = STRAIGHT_QUOTES
        client = FakeClient(FakeResponse(json.dumps(labels_for(chunk)), prompt_tokens=100000))
        _, stats, output = run_chunk(client, chunk)
        self.assertNotIn("PROMPT LIKELY TRUNCATED", output)
        self.assertEqual(stats["prompt_truncation_events"], 0)

    def test_no_warning_when_the_server_reports_no_usage(self):
        chunk = STRAIGHT_QUOTES
        client = FakeClient(FakeResponse(json.dumps(labels_for(chunk))))
        _, stats, output = run_chunk(client, chunk)
        self.assertNotIn("PROMPT LIKELY TRUNCATED", output)
        self.assertEqual(stats["prompt_truncation_events"], 0)
        self.assertIsNone(stats["prompt_tokens"])


class TestFlatlineDetection(unittest.TestCase):
    """The run-level detector for the exact production signature."""

    def test_flatline_detected_when_tokens_pin_despite_varying_prompts(self):
        samples = [(5000 + 200 * i, 2050) for i in range(30)]
        result = detect_flatlined_prompt_tokens(samples)
        self.assertIsNotNone(result)
        value, count, total = result
        self.assertEqual((value, count, total), (2050, 30, 30))

    def test_no_flatline_when_tokens_track_prompt_size(self):
        samples = [(4000 + 100 * i, 1000 + 25 * i) for i in range(40)]
        self.assertIsNone(detect_flatlined_prompt_tokens(samples))

    def test_no_flatline_when_the_prompts_were_all_the_same_size(self):
        # Identical prompts SHOULD produce identical token counts.
        samples = [(8000, 2000) for _ in range(40)]
        self.assertIsNone(detect_flatlined_prompt_tokens(samples))

    def test_too_few_samples_is_never_a_verdict(self):
        self.assertIsNone(detect_flatlined_prompt_tokens([(5000 + 200 * i, 2050)
                                                          for i in range(10)]))
        self.assertIsNone(detect_flatlined_prompt_tokens([]))


class TestNumCtxRequest(unittest.TestCase):
    """Best-effort serving-context-window request."""

    def _extra_body(self, **kwargs):
        chunk = STRAIGHT_QUOTES
        client = FakeClient(FakeResponse(json.dumps(labels_for(chunk))))
        run_chunk(client, chunk, **kwargs)
        return client.completions.last_kwargs.get("extra_body", {})

    def test_options_omitted_entirely_when_unconfigured(self):
        # Users who never set generation.num_ctx must send an unchanged request.
        self.assertNotIn("options", self._extra_body())

    def test_options_present_when_configured(self):
        self.assertEqual(self._extra_body(num_ctx=8192), {"options": {"num_ctx": 8192}})


class TestPlaceholderLabelRejection(unittest.TestCase):
    """Invented enumerated placeholders are narrated, not voiced.

    A production run emitted 307 such entries. The pattern is deliberately
    language-neutral (one token + a bare number) and carries no word list.
    """

    def test_pattern_matches_enumerated_placeholders_only(self):
        for name in ("SPEAKER 1", "SPEAKER 2", "SPEAKER1", "VOICE 3", "CHARACTER 12"):
            with self.subTest(name=name):
                self.assertTrue(is_placeholder_speaker(name))
        for name in ("MARCUS", "ABBE MARIGNAN", "JEAN-LUC", "O'BRIEN", "NARRATOR",
                     "HENRY VIII", "MARY ANNE SMITH", "", None):
            with self.subTest(name=name):
                self.assertFalse(is_placeholder_speaker(name))

    def test_placeholder_label_is_narrated_and_counted(self):
        chunk = STRAIGHT_QUOTES
        client = FakeClient(FakeResponse(json.dumps(
            labels_for(chunk, speaker_of=lambda span, text: "SPEAKER 1"))))
        entries, stats, _ = run_chunk(client, chunk)
        self.assertEqual(joined(entries), chunk)
        self.assertEqual({e["speaker"] for e in entries}, {NARRATOR})
        # STRAIGHT_QUOTES has two quoted spans; both placeholders are rejected.
        self.assertEqual(stats["placeholder_rejected"], 2)

    def test_real_name_is_untouched(self):
        chunk = STRAIGHT_QUOTES
        client = FakeClient(FakeResponse(json.dumps(
            labels_for(chunk, speaker_of=lambda span, text: "MARCUS"))))
        entries, stats, _ = run_chunk(client, chunk)
        self.assertIn("MARCUS", [e["speaker"] for e in entries])
        self.assertEqual(stats["placeholder_rejected"], 0)

    def test_placeholder_never_enters_the_roster(self):
        chunk = STRAIGHT_QUOTES
        roster = {}
        client = FakeClient(FakeResponse(json.dumps(
            labels_for(chunk, speaker_of=lambda span, text: "SPEAKER 2"))))
        run_chunk(client, chunk, roster=roster)
        self.assertEqual(roster, {})




class TestBracketedPreambleRecovery(unittest.TestCase):
    """A ``[`` in prose must not cost the chunk its labels.

    clean_json_string() used to take the FIRST ``[`` anywhere in the response.
    A model prefacing its array with prose containing brackets handed back a
    truthy-but-wrong slice, which then displaced the raw text in
    extract_labels' salvage step -- total label loss for the chunk, with the
    real array sitting intact a few characters later.
    """

    REAL_ARRAY = '[{"id": 1, "speaker": "NARRATOR", "role": "narration", "instruct": "Neutral."}]'

    def test_prose_preamble_with_brackets(self):
        resp = "Sure! Here are the labels [see list below]:\n" + self.REAL_ARRAY
        self.assertEqual(clean_json_string(resp), self.REAL_ARRAY)
        labels, mode = extract_labels(resp)
        self.assertEqual(mode, LABEL_MODE_ARRAY)
        self.assertEqual([label["id"] for label in labels], [1])

    def test_markdown_preamble_with_bracketed_span_ids(self):
        resp = ("- span [1] is narration\n"
                "- span [2] is dialogue\n\n" + self.REAL_ARRAY)
        labels, _ = extract_labels(resp)
        self.assertEqual([label["id"] for label in labels], [1])

    def test_nested_array_inside_an_object_is_not_truncated(self):
        resp = ('[{"id": 1, "speaker": "A", "role": "dialogue", "instruct": "x", '
                '"tags": [1, 2]}, '
                '{"id": 2, "speaker": "NARRATOR", "role": "narration", "instruct": "y"}]')
        labels, mode = extract_labels(resp)
        self.assertEqual(mode, LABEL_MODE_ARRAY)
        self.assertEqual([label["id"] for label in labels], [1, 2])

    def test_truncated_array_after_a_prose_bracket_still_salvages(self):
        resp = ('Here goes [note] :\n[{"id": 1, "speaker": "A", "role": "dialogue", '
                '"instruct": "x"}, {"id": 2, "speaker": "B", "role": "dialogue"')
        labels, _ = extract_labels(resp)
        self.assertEqual([label["id"] for label in labels], [1, 2])

    def test_response_with_no_array_at_all_is_unchanged(self):
        self.assertIsNone(clean_json_string("Sure! Here is the script."))
        self.assertEqual(extract_labels("Sure! Here is the script."), (None, None))

    def test_bracketed_preamble_through_process_chunk_keeps_the_speaker(self):
        chunk = STRAIGHT_QUOTES
        resp = ("Certainly [as requested]:\n" +
                json.dumps(labels_for(chunk, speaker_of=lambda span, text: "MARCUS")))
        entries, stats, _ = run_chunk(chunk=chunk, client=FakeClient(FakeResponse(resp)))
        self.assertEqual(joined(entries), chunk)
        self.assertIn("MARCUS", [e["speaker"] for e in entries])
        self.assertFalse(stats["degraded"])


class TestMojibakeTable(unittest.TestCase):
    """The repair table's literals are themselves mojibake and had decayed.

    Two entries were lost: the right-single-quote VALUE became three ASCII
    apostrophes (a triple-quote that swallowed the next entry), and the em-
    and en-dash KEYS collapsed to the same byte, so 7 literals became 6 dict
    entries and em-dash mojibake was never repaired.
    """

    @staticmethod
    def _literal_count():
        """Count key literals in the dict's SOURCE, independent of the dict."""
        source = inspect.getsource(generate_script)
        body = source.split("MOJIBAKE_REPLACEMENTS = {", 1)[1].split("}", 1)[0]
        return len([line for line in body.splitlines() if ":" in line])

    def test_no_duplicate_keys_collapsed(self):
        self.assertEqual(len(MOJIBAKE_REPLACEMENTS), self._literal_count())

    def test_all_keys_and_values_are_distinct_where_expected(self):
        self.assertEqual(len(set(MOJIBAKE_REPLACEMENTS)), len(MOJIBAKE_REPLACEMENTS))

    def test_both_dashes_are_repaired(self):
        em_mojibake = "\u00e2\u20ac\u201d"
        en_mojibake = "\u00e2\u20ac\u201c"
        self.assertEqual(fix_mojibake(em_mojibake), "\u2014")
        self.assertEqual(fix_mojibake(en_mojibake), "\u2013")

    def test_quotes_and_ellipsis_are_repaired(self):
        cases = {
            "\u00e2\u20ac\u2122": "\u2019",
            "\u00e2\u20ac\u02dc": "\u2018",
            "\u00e2\u20ac\u0153": "\u201c",
            "\u00e2\u20ac\u009d": "\u201d",
            "\u00e2\u20ac\u00a6": "\u2026",
        }
        for bad, good in cases.items():
            with self.subTest(bad=bad):
                self.assertEqual(fix_mojibake(bad), good)

    def test_clean_text_is_untouched(self):
        clean = 'She said, "it\u2019s fine" \u2014 and left\u2026'
        self.assertEqual(fix_mojibake(clean), clean)


class TestLLMTimeoutConfig(unittest.TestCase):
    """main() must honour llm.timeout, and must not send it when unset.

    Source assertions, because the client is constructed inside main() around a
    network call. The property that matters is the CONDITIONAL: passing a
    timeout unconditionally would replace the SDK default for every existing
    install, which is the kind of silent behaviour change this repo avoids.
    """

    def setUp(self):
        self.source = inspect.getsource(generate_script)

    def test_main_reads_the_timeout_key(self):
        self.assertIn('llm_config.get("timeout")', self.source,
                      "main() must read llm.timeout from the llm config section")

    def test_timeout_is_only_passed_when_set(self):
        # The guard is what keeps the unset case byte-identical to before.
        self.assertIn('if timeout:', self.source)
        self.assertIn('client_kwargs["timeout"] = timeout', self.source)

    def test_client_is_built_from_kwargs_not_a_literal_call(self):
        # Guards against a refactor that reinstates a hardcoded OpenAI(...)
        # call and silently drops the timeout.
        self.assertIn("OpenAI(**client_kwargs)", self.source)


class TestAttestationGate(unittest.TestCase):
    """A speaker name the source does not support is refused, loudly.

    Properties only -- every fixture here is built in the test, and the
    assertions hold for any text. The gate is OFF by default, so the first
    test is the one that protects existing users.
    """

    def test_gate_is_off_by_default(self):
        # An unattested name is accepted when the gate is not requested, so
        # this change cannot alter any existing run.
        chunk = STRAIGHT_QUOTES
        client = FakeClient(FakeResponse(json.dumps(
            labels_for(chunk, speaker_of=lambda span, text: "ZZQXAL"))))
        entries, stats, _ = run_chunk(client, chunk)
        self.assertIn("ZZQXAL", [e["speaker"] for e in entries])
        self.assertEqual(stats["unattested_rejected"], 0)

    def test_unattested_speaker_is_narrated_and_counted(self):
        chunk = STRAIGHT_QUOTES  # names "Marcus", never "ZZQXAL"
        client = FakeClient(*[FakeResponse(json.dumps(
            labels_for(chunk, speaker_of=lambda span, text: "ZZQXAL")))] * 3)
        entries, stats, _ = run_chunk(
            client, chunk, require_attested=True, attest_window=chunk)
        # Prose is never the price of a rejected label.
        self.assertEqual(joined(entries), chunk)
        self.assertEqual({e["speaker"] for e in entries}, {NARRATOR})
        self.assertEqual(stats["unattested_rejected"], 2)
        # Contract 7: this must not pass silently.
        self.assertTrue(stats["degraded"])
        self.assertIn("source text does not support", stats["reason"])

    def test_attested_speaker_passes_the_gate(self):
        chunk = STRAIGHT_QUOTES  # "Marcus" appears in the narration
        client = FakeClient(FakeResponse(json.dumps(
            labels_for(chunk, speaker_of=lambda span, text: "MARCUS"))))
        entries, stats, _ = run_chunk(
            client, chunk, require_attested=True, attest_window=chunk)
        self.assertIn("MARCUS", [e["speaker"] for e in entries])
        self.assertEqual(stats["unattested_rejected"], 0)
        self.assertFalse(stats["degraded"])

    def test_established_roster_name_needs_no_local_attestation(self):
        # THE long-range property: a character named pages ago is accepted in a
        # chunk that never names them, at zero token cost. Without this clause
        # every unattributed line would be narrated.
        chunk = STRAIGHT_QUOTES
        roster = {}
        remember_in_roster(roster, "DAIRINE")  # established by an earlier chunk
        client = FakeClient(FakeResponse(json.dumps(
            labels_for(chunk, speaker_of=lambda span, text: "DAIRINE"))))
        entries, stats, _ = run_chunk(
            client, chunk, require_attested=True, attest_window=chunk,
            roster=roster)
        self.assertIn("DAIRINE", [e["speaker"] for e in entries])
        self.assertEqual(stats["unattested_rejected"], 0)

    def test_rejected_speaker_never_enters_the_roster(self):
        # A refused spelling must not become established -- otherwise the
        # roster clause above would launder it into every later chunk.
        chunk = STRAIGHT_QUOTES
        roster = {}
        client = FakeClient(*[FakeResponse(json.dumps(
            labels_for(chunk, speaker_of=lambda span, text: "ZZQXAL")))] * 3)
        run_chunk(client, chunk, require_attested=True, attest_window=chunk,
                  roster=roster)
        self.assertEqual(roster, {})

    def test_retry_nudge_names_the_offending_label(self):
        # Rejection is the fallback; re-asking is the fix. The retry prompt has
        # to say WHICH name failed and what to do instead, or the model just
        # re-rolls the same judgement.
        chunk = STRAIGHT_QUOTES
        bad = json.dumps(labels_for(chunk, speaker_of=lambda span, text: "ZZQXAL"))
        client = FakeClient(*[FakeResponse(bad)] * 3)
        run_chunk(client, chunk, require_attested=True, attest_window=chunk)
        self.assertGreater(len(client.user_prompts), 1, "a retry should have fired")
        retry_prompt = client.user_prompts[1]
        self.assertIn("ZZQXAL", retry_prompt)
        self.assertIn("do not appear anywhere in the text", retry_prompt)

    def test_nudge_recovers_the_correct_spelling(self):
        # The whole point: a corrected retry keeps the voice instead of losing
        # it to the narrator.
        chunk = STRAIGHT_QUOTES
        bad = json.dumps(labels_for(chunk, speaker_of=lambda span, text: "MARCVS"))
        good = json.dumps(labels_for(chunk, speaker_of=lambda span, text: "MARCUS"))
        client = FakeClient(FakeResponse(bad), FakeResponse(good))
        entries, stats, _ = run_chunk(
            client, chunk, require_attested=True, attest_window=chunk)
        self.assertIn("MARCUS", [e["speaker"] for e in entries])
        self.assertNotIn("MARCVS", [e["speaker"] for e in entries])
        self.assertEqual(stats["unattested_rejected"], 0)
        self.assertFalse(stats["degraded"])

    def test_unverifiable_label_is_accepted_not_rejected(self):
        # A title-only label yields no core tokens: nothing to confirm, so
        # nothing to refuse. Rejecting here would destroy scripts this module
        # cannot tokenize.
        chunk = STRAIGHT_QUOTES
        client = FakeClient(FakeResponse(json.dumps(
            labels_for(chunk, speaker_of=lambda span, text: "MISTER"))))
        entries, stats, _ = run_chunk(
            client, chunk, require_attested=True, attest_window=chunk)
        self.assertEqual(stats["unattested_rejected"], 0)
        self.assertEqual(stats["unverifiable_accepted"], 2)
        self.assertIn("MISTER", [e["speaker"] for e in entries])

    def test_narrator_with_dialogue_role_is_never_gated(self):
        # NARRATOR is the narrator sentinel, not a name. The word "Narrator"
        # appears in no novel, so attesting it rejected the narrator itself and
        # falsely degraded the chunk. The prompt explicitly tells the model to
        # use NARRATOR for an unidentifiable speaker, so this fires constantly.
        chunk = STRAIGHT_QUOTES
        labels = labels_for(chunk, speaker_of=lambda span, text: "NARRATOR")
        for label in labels:
            label["role"] = "dialogue"  # the shape observed live
        client = FakeClient(FakeResponse(json.dumps(labels)))
        entries, stats, _ = run_chunk(
            client, chunk, require_attested=True, attest_window=chunk)
        check = self.assertEqual
        check(stats["unattested_rejected"], 0)
        check(stats["degraded"], False)
        self.assertEqual(joined(entries), chunk)

    def test_gate_and_retry_predicate_agree_on_every_label(self):
        # The invariant _incomplete_span_ids' docstring states for roles, held
        # here for speakers: whatever the gate would REJECT is exactly what the
        # retry predicate must ASK about. If they diverge, a label is either
        # silently rejected with no retry (what NARRATOR did) or nagged about
        # forever without ever being refused.
        chunk = STRAIGHT_QUOTES
        spans = tokenize(chunk)
        for speaker in ("NARRATOR", "MARCUS", "ZZQXAL", "MISTER", "SPEAKER 1"):
            with self.subTest(speaker=speaker):
                labels = labels_for(chunk, speaker_of=lambda s, t: speaker)
                for label in labels:
                    label["role"] = "dialogue"
                merged = {generate_script._label_id(l): l for l in labels}

                asked = generate_script._unattested_speaker_ids(
                    spans, merged, source=chunk, roster=None, attest_window=chunk)
                _, stats = resolve_span_labels(
                    spans, labels, source=chunk, roster={},
                    attest_window=chunk, require_attested=True)

                self.assertEqual(
                    bool(asked), bool(stats["unattested_rejected"]),
                    f"retry predicate and gate disagree for {speaker!r}: "
                    f"asked={len(asked)} rejected={stats['unattested_rejected']}")

    def test_gate_and_retry_predicate_agree_on_repaired_labels_too(self):
        # The same invariant with repair in play, which is exactly where it is
        # easiest to break: a label the gate silently REPAIRS must not be one
        # the retry predicate keeps nagging about, and vice versa. Both go
        # through _gate_speaker, and this pins that they still do.
        chunk = 'Marcus and Marcas leaned over the map. "Then go."\n'
        spans = tokenize(chunk)
        source_words = source_word_index(chunk)
        roster = {}
        remember_in_roster(roster, "MARCUS")
        # MARCVS repairs (unique candidate), MARCS is ambiguous (two roster
        # names one edit away), ZZQXAL has no candidate at all.
        ambiguous_roster = dict(roster)
        remember_in_roster(ambiguous_roster, "MARCAS")
        cases = [
            ("MARCVS", roster, False),
            ("MARCS", ambiguous_roster, True),
            ("ZZQXAL", roster, True),
            ("MARCUS", roster, False),
        ]
        for speaker, book_roster, expect_rejected in cases:
            with self.subTest(speaker=speaker):
                labels = labels_for(chunk, speaker_of=lambda s, t: speaker)
                for label in labels:
                    label["role"] = "dialogue"
                merged = {generate_script._label_id(l): l for l in labels}

                asked = generate_script._unattested_speaker_ids(
                    spans, merged, source=chunk, roster=book_roster,
                    attest_window=chunk, source_words=source_words)
                _, stats = resolve_span_labels(
                    spans, labels, source=chunk, roster=book_roster,
                    attest_window=chunk, require_attested=True,
                    source_words=source_words)

                self.assertEqual(
                    bool(asked), bool(stats["unattested_rejected"]),
                    f"retry predicate and gate disagree for {speaker!r}: "
                    f"asked={len(asked)} rejected={stats['unattested_rejected']}")
                self.assertEqual(bool(stats["unattested_rejected"]), expect_rejected,
                                 f"unexpected gate outcome for {speaker!r}")

    def test_label_built_on_a_roster_name_is_accepted_as_unverifiable(self):
        # Change 1: "<established name>'S FATHER" is not positive evidence of a
        # WRONG label, so it must not be refused. The voice is kept and the
        # acceptance is counted.
        chunk = STRAIGHT_QUOTES  # names "Marcus"
        roster = {}
        remember_in_roster(roster, "MARCUS")
        client = FakeClient(FakeResponse(json.dumps(
            labels_for(chunk, speaker_of=lambda span, text: "MARCUS'S FATHER"))))
        entries, stats, _ = run_chunk(
            client, chunk, require_attested=True, attest_window=chunk,
            roster=roster)
        self.assertEqual(stats["unattested_rejected"], 0)
        # Counted once: accepting it establishes it, and the second span then
        # takes the established-roster shortcut.
        self.assertEqual(stats["unverifiable_accepted"], 1)
        self.assertIn("MARCUS'S FATHER", [e["speaker"] for e in entries])

    def test_repair_keeps_the_voice_without_spending_a_retry(self):
        # Change 2, and its whole economic argument: a misread spelling the
        # book never uses, one edit from an established name that is right
        # there in the window, is repaired in place -- ONE LLM call, no retry,
        # and the line keeps a character voice instead of being narrated.
        chunk = STRAIGHT_QUOTES  # names "Marcus", never "MARCVS"
        roster = {}
        remember_in_roster(roster, "MARCUS")
        client = FakeClient(FakeResponse(json.dumps(
            labels_for(chunk, speaker_of=lambda span, text: "MARCVS"))))
        entries, stats, output = run_chunk(
            client, chunk, require_attested=True, attest_window=chunk,
            roster=roster, source_words=source_word_index(chunk))

        self.assertEqual(len(client.user_prompts), 1,
                         "repair must not cost a retry")
        self.assertIn("MARCUS", [e["speaker"] for e in entries])
        self.assertNotIn("MARCVS", [e["speaker"] for e in entries])
        self.assertEqual(stats["unattested_rejected"], 0)
        self.assertEqual(stats["speakers_repaired"], 2)
        self.assertEqual(stats["repairs"], [("MARCVS", "MARCUS")] * 2)
        self.assertFalse(stats["degraded"])
        # Contract 7: a rewritten name is never silent.
        self.assertIn("REPAIRED", output)
        self.assertIn("MARCVS", output)
        # And prose is untouched, as under every other gate outcome.
        self.assertEqual(joined(entries), chunk)

    def test_repair_is_unavailable_without_a_book_word_index(self):
        # The default is the old behaviour: no book index, no repair.
        chunk = STRAIGHT_QUOTES
        roster = {}
        remember_in_roster(roster, "MARCUS")
        client = FakeClient(*[FakeResponse(json.dumps(
            labels_for(chunk, speaker_of=lambda span, text: "MARCVS")))] * 3)
        _, stats, _ = run_chunk(client, chunk, require_attested=True,
                                attest_window=chunk, roster=roster)
        self.assertEqual(stats["speakers_repaired"], 0)
        self.assertEqual(stats["unattested_rejected"], 2)

    def test_ambiguous_repair_is_refused_and_retried(self):
        # Two established names one edit away: the evidence does not pick a
        # winner, so the label is refused and the retry fires as before.
        chunk = 'Marcus and Marcas leaned over the map. "Then go."\n'
        roster = {}
        remember_in_roster(roster, "MARCUS")
        remember_in_roster(roster, "MARCAS")
        client = FakeClient(*[FakeResponse(json.dumps(
            labels_for(chunk, speaker_of=lambda span, text: "MARCS")))] * 3)
        entries, stats, _ = run_chunk(
            client, chunk, require_attested=True, attest_window=chunk,
            roster=roster, source_words=source_word_index(chunk))
        self.assertEqual(stats["speakers_repaired"], 0)
        self.assertGreater(stats["unattested_rejected"], 0)
        self.assertGreater(len(client.user_prompts), 1, "a retry should have fired")
        self.assertEqual(joined(entries), chunk)

    def test_retry_nudge_offers_candidate_spellings_when_repair_declines(self):
        # A free prompt hint, not a mechanism: the gate still re-checks
        # whatever comes back.
        chunk = 'Marcus and Marcas leaned over the map. "Then go."\n'
        roster = {}
        remember_in_roster(roster, "MARCUS")
        remember_in_roster(roster, "MARCAS")
        client = FakeClient(*[FakeResponse(json.dumps(
            labels_for(chunk, speaker_of=lambda span, text: "MARCS")))] * 3)
        run_chunk(client, chunk, require_attested=True, attest_window=chunk,
                  roster=roster, source_words=source_word_index(chunk))
        retry_prompt = client.user_prompts[1]
        self.assertIn("did you mean", retry_prompt)
        self.assertIn("MARCUS", retry_prompt)

    def test_no_window_means_no_confirmation(self):
        # Guards against a caller enabling the gate without supplying a window
        # and silently narrating the whole book: with no window, an
        # unestablished name cannot attest, and the run degrades loudly rather
        # than quietly.
        chunk = STRAIGHT_QUOTES
        client = FakeClient(*[FakeResponse(json.dumps(
            labels_for(chunk, speaker_of=lambda span, text: "MARCUS")))] * 3)
        entries, stats, _ = run_chunk(client, chunk, require_attested=True)
        self.assertEqual(joined(entries), chunk)
        self.assertTrue(stats["degraded"])


# Two lines, each closed by its own attribution tag. Synthesized: no book's
# names, no book's phrasing -- only the shape English narration always uses.
TAGGED_DIALOGUE = (
    '"Yeah," Annika said.\n'
    '\n'
    '"No," Borel said.\n'
)


def _tagged_roster():
    roster = {}
    remember_in_roster(roster, "ANNIKA")
    remember_in_roster(roster, "BOREL")
    return roster


def _label_first_quote(name):
    """Label the first quoted span `name` and the second one BOREL."""
    seen = []

    def speaker_of(span, text):
        seen.append(span.id)
        return name if len(seen) == 1 else "BOREL"

    return speaker_of


class TestAttributionTagCheck(unittest.TestCase):
    """A speaker label that contradicts the attribution tag beside it is
    retried, then reported -- and never rewritten.

    Properties only; the fixtures are built here. The check is OFF by default,
    so the first test is the one that protects existing users.
    """

    def test_check_is_off_by_default(self):
        chunk = TAGGED_DIALOGUE
        client = FakeClient(FakeResponse(json.dumps(
            labels_for(chunk, speaker_of=_label_first_quote("BOREL")))))
        entries, stats, _ = run_chunk(client, chunk, roster=_tagged_roster())
        self.assertEqual(stats.get("tag_contradictions", 0), 0)
        self.assertEqual(len(client.user_prompts), 1, "no retry when off")
        self.assertEqual(joined(entries), chunk)

    def test_contradiction_becomes_a_retry_offender(self):
        chunk = TAGGED_DIALOGUE  # line one is tagged "Annika said"
        bad = json.dumps(labels_for(chunk, speaker_of=_label_first_quote("BOREL")))
        client = FakeClient(*[FakeResponse(bad)] * 3)
        run_chunk(client, chunk, check_tags=True, roster=_tagged_roster())
        self.assertGreater(len(client.user_prompts), 1, "a retry should have fired")
        retry_prompt = client.user_prompts[1]
        self.assertIn("attribution tag", retry_prompt)
        self.assertIn("ANNIKA", retry_prompt)

    def test_a_corrected_retry_clears_it(self):
        chunk = TAGGED_DIALOGUE
        bad = json.dumps(labels_for(chunk, speaker_of=_label_first_quote("BOREL")))
        good = json.dumps(labels_for(chunk, speaker_of=_label_first_quote("ANNIKA")))
        client = FakeClient(FakeResponse(bad), FakeResponse(good))
        entries, stats, _ = run_chunk(chunk=chunk, client=client, check_tags=True,
                                      roster=_tagged_roster())
        self.assertEqual(stats["tag_contradictions"], 0)
        self.assertFalse(stats["degraded"])
        self.assertIn("ANNIKA", [e["speaker"] for e in entries])

    def test_unresolved_contradiction_degrades_but_never_rewrites(self):
        # DETECT, DO NOT AUTO-REWRITE: the model's label survives verbatim and
        # the operator is told where to look (contract 7).
        chunk = TAGGED_DIALOGUE
        bad = json.dumps(labels_for(chunk, speaker_of=_label_first_quote("BOREL")))
        client = FakeClient(*[FakeResponse(bad)] * 3)
        entries, stats, output = run_chunk(client, chunk, check_tags=True,
                                           roster=_tagged_roster())
        self.assertEqual(joined(entries), chunk, "prose is never the price")
        self.assertEqual(stats["tag_contradictions"], 1)
        self.assertTrue(stats["degraded"])
        self.assertIn("contradicted by the attribution tag", stats["reason"])
        self.assertIn("BOREL", [e["speaker"] for e in entries])
        self.assertNotIn("ANNIKA", [e["speaker"] for e in entries],
                         "the tag's name must NOT be imposed on the label")
        self.assertIn("CONTRADICTED", output)

    def test_agreeing_labels_are_never_flagged(self):
        chunk = TAGGED_DIALOGUE
        client = FakeClient(FakeResponse(json.dumps(
            labels_for(chunk, speaker_of=_label_first_quote("ANNIKA")))))
        entries, stats, _ = run_chunk(client, chunk, check_tags=True,
                                      roster=_tagged_roster())
        self.assertEqual(stats["tag_contradictions"], 0)
        self.assertFalse(stats["degraded"])

    def test_a_non_english_book_produces_no_detections(self):
        # The lexicon is English-only and DEGRADES TO DOING NOTHING: with a
        # deliberately wrong label on every line, a French chunk still yields
        # zero detections, zero retries and no degradation.
        chunk = (
            '"Oui," dit Annika.\n'
            '\n'
            '"Non," dit Borel.\n'
        )
        client = FakeClient(FakeResponse(json.dumps(
            labels_for(chunk, speaker_of=_label_first_quote("BOREL")))))
        entries, stats, _ = run_chunk(client, chunk, check_tags=True,
                                      roster=_tagged_roster())
        self.assertEqual(stats["tag_contradictions"], 0)
        self.assertEqual(len(client.user_prompts), 1, "no retry on a French book")
        self.assertEqual(joined(entries), chunk)

    def test_possessive_tag_is_not_flagged_end_to_end(self):
        # The false-positive class that a naive name-grab gets wrong: the label
        # naming a person via another character is CORRECT.
        chunk = '"Sit down," said Annika\'s father.\n'
        roster = _tagged_roster()
        remember_in_roster(roster, "ANNIKA'S FATHER")
        client = FakeClient(FakeResponse(json.dumps(
            labels_for(chunk, speaker_of=lambda span, text: "ANNIKA'S FATHER"))))
        entries, stats, _ = run_chunk(client, chunk, check_tags=True, roster=roster)
        self.assertEqual(stats["tag_contradictions"], 0)
        self.assertEqual(joined(entries), chunk)


if __name__ == "__main__":
    unittest.main(verbosity=2)
