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

import io
import json
import os
import sys
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

from default_prompts import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT  # noqa: E402
from span_tokenizer import tokenize  # noqa: E402
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
    labels_from_id_keyed_object,
    parse_label_response,
    process_chunk,
    resolve_span_labels,
    salvage_label_entries,
    salvage_markdown_labels,
    select_prompt,
    strip_thinking_tags,
)


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

    def __init__(self, content, finish_reason="stop"):
        self.choices = [_FakeChoice(content, finish_reason)]
        self.usage = None


class _FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.last_kwargs = None

    def create(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
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
        self.assertEqual(stats["labelled"], 0)
        self.assertEqual(stats["fallback"], stats["spans"])
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
        self.assertEqual(stats["fallback"], stats["spans"])
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
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["instruct"], generate_script.DEFAULT_CHARACTER_INSTRUCT)

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
        entries, stats, output = run_chunk(
            FakeClient(FakeResponse(REAL_MARKDOWN_SPAN)), chunk)

        self.assertEqual(joined(entries), chunk, "prose byte-identical either way")
        self.assertEqual(stats["recovery"], LABEL_MODE_MARKDOWN)
        self.assertEqual(stats["labelled"], 3, "3 of 6 spans were labelled")
        self.assertEqual(stats["fallback"], 3)
        self.assertTrue(stats["degraded"], "ignoring the output contract must not exit 0")
        self.assertIn("markdown", stats["reason"].lower())
        self.assertIn("markdown blocks", output)
        self.assertIn("BILL", {entry["speaker"] for entry in entries})

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


class TestSalvageAndPayload(unittest.TestCase):
    def test_salvage_label_entries_is_field_order_agnostic(self):
        text = '[{"role": "dialogue", "instruct": "Calm.", "speaker": "MARCUS", "id": 4}]'
        labels = salvage_label_entries(text)
        self.assertEqual(labels, [{"id": 4, "speaker": "MARCUS", "role": "dialogue", "instruct": "Calm."}])

    def test_salvage_recovers_truncated_tail_object(self):
        text = '[{"id": 1, "speaker": "A", "role": "dialogue", "instruct": "x"}, {"id": 2, "speaker": "B"'
        labels = salvage_label_entries(text)
        self.assertEqual([label["id"] for label in labels], [1, 2])

    def test_parse_label_response_handles_clean_json(self):
        text = '[{"id": 1, "speaker": "A", "role": "dialogue", "instruct": "x"}]'
        self.assertEqual(len(parse_label_response(text)), 1)

    def test_span_payload_never_contains_offsets_and_is_one_line_per_span(self):
        payload = build_span_payload(tokenize(STRAIGHT_QUOTES), STRAIGHT_QUOTES)
        lines = payload.split("\n")
        self.assertEqual(len(lines), len(tokenize(STRAIGHT_QUOTES)))
        for line in lines:
            record = json.loads(line)
            self.assertEqual(set(record.keys()), {"id", "kind", "text"})

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
        self.assertEqual(stats["labelled"], 0)
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
