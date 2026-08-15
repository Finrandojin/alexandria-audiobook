#!/usr/bin/env python
"""Standalone unit tests for app/span_tokenizer.py.

Run directly -- no pytest, no live server required:

    python app/test_span_tokenizer.py

Exits 0 when every test passes, nonzero otherwise.
"""

import os
import sys
import time
import unittest

# Windows consoles default to cp1252; the fixtures contain curly quotes.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from span_tokenizer import (  # noqa: E402
    MAX_AMBIGUOUS_SINGLE_SPAN,
    QUOTED,
    UNQUOTED,
    Span,
    reassemble,
    tokenize,
    validate_spans,
)


def segments(source):
    """Return [(kind, text), ...] -- the readable view of a tokenization."""
    return [(s.kind, source[s.start:s.end]) for s in tokenize(source)]


# --------------------------------------------------------------------------
# The corpus. Every fixture is checked by the property test below.
# --------------------------------------------------------------------------

PROSE_MIXED = (
    'Marcus set down the lantern. "We can\'t stay here," he said, and the\n'
    'words hung between them like smoke. "Not after what the Joneses\' boy\n'
    'told me."\n'
    '\n'
    "Elena did not answer at first. She was thinking of the '90s, of the\n"
    "summer her father had said, 'Everything ends, and that is the whole of\n"
    "it.'\n"
    '\n'
    '"Don\'t," she said finally. "Don\'t say \'ends\' to me tonight."\n'
)

PROSE_CURLY = (
    "“You were never going to tell me,” she said. It wasn’t a "
    "question.\n"
    "\n"
    "“I was,” Marcus answered. “I was waiting for the right "
    "hour, and then there wasn’t one.” He turned the glass in his "
    "hand. “She told me ‘go away’ and I went.”\n"
    "\n"
    "— Then go again, Elena thought, and said nothing at all.\n"
)

# The audit probe: a false opener plus a distant possessive used to swallow
# 1139 characters of narration into one "quoted" span.
RUNAWAY_PROBE = (
    "He said 'no more. "
    + ("Filler narration text here. " * 40)
    + "The dogs' tails wagged."
)

# Same shape, but the only candidate closer sits in a later paragraph.
CLOSER_IN_LATER_PARAGRAPH = (
    "He said 'no more and walked out into the rain.\n"
    "\n"
    "The dogs' tails wagged when he reached the gate.\n"
)

# --- non-English conventions ------------------------------------------------
# French typography puts a space INSIDE the guillemets. Real books use U+00A0
# (no-break) or U+202F (narrow no-break) as often as U+0020; all three are part
# of the quoted span and must survive byte-exactly.
NBSP = " "
NNBSP = " "

FRENCH_NBSP = "«" + NBSP + "Je pars." + NBSP + "»"
FRENCH_NNBSP = "«" + NNBSP + "Je pars." + NNBSP + "»"

FRENCH_INNER = "«" + NBSP + "Bonjour," + NBSP + "»"

MULTILINGUAL_DOC = (
    FRENCH_INNER + " dit-il. "
    "「こんにちは」と彼は言った。"
    "„Guten Tag,“ sagte sie. "
    '"Hello," he said.\n'
)

CORPUS = [
    # 1-4: the item-3 splitting rule, straight and curly, double and single
    '"I am leaving," he said.',
    "“I am leaving,” he said.",
    "'Hello,' she said.",
    "‘Hello,’ she said.",
    # 5-6: nested quotes -- outer quotation is ONE span
    '"She told me \'go away\' yesterday," he said.',
    "“She told me ‘go away’ yesterday,” he said.",
    # 7-9: apostrophes that are not delimiters
    "Don't touch Jones' hat; 'tis his.",
    "It doesn’t matter, 'twas Jones’ idea.",
    "Back in the '90s, things were different, 'round here at least.",
    # 10-12: unterminated quotations
    'He said, "I am leaving',
    "He said, 'I am leaving",
    "He said, ‘I am leaving",
    # 13-14: dialogue the tokenizer must NOT try to classify
    "— I am leaving, he said.",
    "I am leaving, he said, and then he left.",
    # 15-17: structural edges
    '"Yes."',
    '"A""B"',
    'Those are the dogs\' and those are the cats\'',
    # 18-20: whitespace, newlines, mixed glyphs in one document
    "   ",
    '"I am leaving,\nand I am not coming back," he said.\n',
    "“First,” he said. \"Second,\" she replied. ‘Third,’ "
    "they agreed.",
    # 21-23: realistic prose
    PROSE_MIXED,
    PROSE_CURLY,
    'The letter read: "To whom it may concern -- I regret nothing." Below it,\n'
    "unsigned, someone had written 'liar' in pencil. Marcus didn't laugh.\n",
    # 24-25: the runaway-opener probe and its paragraph-crossing sibling
    RUNAWAY_PROBE,
    CLOSER_IN_LATER_PARAGRAPH,
    # 26-31: guillemets, plain and with French inner spacing
    "« Je pars. »",
    "« Je pars, » dit-il.",
    FRENCH_NBSP,
    FRENCH_NNBSP,
    "Il a dit « Je pars » et il est parti.",
    "Il a dit « Je pars et il n'est jamais revenu.",
    # 32-35: single guillemets
    "‹ Oui ›, dit-elle.",
    "« Elle a dit ‹ va-t'en › hier, » dit-il.",
    "‹ Oui",
    # 35-39: CJK corner brackets
    "「こんにちは」と彼は言った。",
    "「彼は『さようなら』と言った」と彼女は言った。",
    "『白い箱』を開けた。",
    "「終わりだ",
    # 40-43: German / Polish low-high
    "„Ich gehe,“ sagte er.",
    "„Sie sagte ‚geh weg‘ gestern,“ sagte er.",
    "‚Ja‘, sagte sie.",
    "„Ich gehe",
    # 44-45: full-width and a mixed-convention document
    "＂やあ＂と彼は言った。",
    MULTILINGUAL_DOC,
]


class TestProperties(unittest.TestCase):
    """The invariants every caller depends on, for every fixture."""

    def test_join_equals_source(self):
        for index, source in enumerate(CORPUS, start=1):
            spans = tokenize(source)
            joined = "".join(source[s.start:s.end] for s in spans)
            self.assertEqual(
                joined, source, "fixture %d did not reassemble verbatim" % index
            )
            self.assertEqual(reassemble(spans, source), source)

    def test_contiguous_covering_and_numbered(self):
        for index, source in enumerate(CORPUS, start=1):
            spans = tokenize(source)
            self.assertTrue(spans, "fixture %d produced no spans" % index)
            self.assertEqual(spans[0].start, 0, "fixture %d start" % index)
            self.assertEqual(spans[-1].end, len(source), "fixture %d end" % index)
            for position, span in enumerate(spans):
                self.assertEqual(span.id, position + 1, "fixture %d ids" % index)
                self.assertGreater(span.end, span.start, "fixture %d empty" % index)
                self.assertIn(span.kind, (QUOTED, UNQUOTED))
            for first, second in zip(spans, spans[1:]):
                self.assertEqual(
                    first.end, second.start, "fixture %d contiguity" % index
                )

    def test_validate_spans_accepts_every_fixture(self):
        for index, source in enumerate(CORPUS, start=1):
            self.assertTrue(
                validate_spans(tokenize(source), source),
                "fixture %d failed validation" % index,
            )

    def test_validate_spans_rejects_a_broken_tiling(self):
        source = '"Go," he said.'
        good = tokenize(source)
        broken = [Span(id=1, start=0, end=5, kind=QUOTED)]
        with self.assertRaises(ValueError):
            validate_spans(broken, source)
        # renumbered out of order
        scrambled = [
            Span(id=s.id + 1, start=s.start, end=s.end, kind=s.kind) for s in good
        ]
        with self.assertRaises(ValueError):
            validate_spans(scrambled, source)

    def test_empty_source(self):
        self.assertEqual(tokenize(""), [])
        self.assertEqual(reassemble([], ""), "")
        self.assertTrue(validate_spans([], ""))

    def test_book_sized_input_stays_linear(self):
        # Regression guard: the opener rule asks "does a closer exist later?".
        # When that was a rescan of the tail, this input took ~60s. The bound
        # is deliberately loose so it flags a return to quadratic, not a slow
        # machine.
        adversarial = "He said 'x " * 20000
        start = time.time()
        spans = tokenize(adversarial)
        elapsed = time.time() - start
        self.assertTrue(validate_spans(spans, adversarial))
        self.assertLess(elapsed, 10.0, "tokenize() went quadratic: %.1fs" % elapsed)

        prose = PROSE_MIXED * 2000
        start = time.time()
        spans = tokenize(prose)
        elapsed = time.time() - start
        self.assertTrue(validate_spans(spans, prose))
        self.assertLess(elapsed, 10.0, "tokenize() went quadratic: %.1fs" % elapsed)


class TestSplittingRule(unittest.TestCase):
    """Item 3: quote marks belong to the quoted span; the tag stays unquoted."""

    def test_straight_double(self):
        self.assertEqual(
            segments('"I am leaving," he said.'),
            [(QUOTED, '"I am leaving,"'), (UNQUOTED, " he said.")],
        )

    def test_curly_double(self):
        self.assertEqual(
            segments("“I am leaving,” he said."),
            [(QUOTED, "“I am leaving,”"), (UNQUOTED, " he said.")],
        )

    def test_leading_whitespace_belongs_to_the_unquoted_span(self):
        self.assertEqual(
            segments('He said, "I am leaving." Then he left.'),
            [
                (UNQUOTED, "He said, "),
                (QUOTED, '"I am leaving."'),
                (UNQUOTED, " Then he left."),
            ],
        )

    def test_span_text_helper(self):
        source = '"Go," he said.'
        spans = tokenize(source)
        self.assertEqual(spans[0].text(source), '"Go,"')
        self.assertEqual(spans[0].as_dict()["kind"], QUOTED)


class TestSingleQuotes(unittest.TestCase):
    """Item 4: single-quoted dialogue vs. apostrophes."""

    def test_single_quoted_dialogue(self):
        self.assertEqual(
            segments("'Hello,' she said."),
            [(QUOTED, "'Hello,'"), (UNQUOTED, " she said.")],
        )

    def test_curly_single_quoted_dialogue(self):
        self.assertEqual(
            segments("‘Hello,’ she said."),
            [(QUOTED, "‘Hello,’"), (UNQUOTED, " she said.")],
        )

    def test_apostrophe_inside_word_is_not_a_delimiter(self):
        source = "Don't stop; it doesn’t matter."
        self.assertEqual(segments(source), [(UNQUOTED, source)])

    def test_leading_elision_does_not_open_a_span(self):
        source = "'tis a fine day, and 'twas a finer night."
        self.assertEqual(segments(source), [(UNQUOTED, source)])

    def test_trailing_possessive_does_not_open_a_span(self):
        source = "Those are Jones' boots and the Joneses’ cart."
        self.assertEqual(segments(source), [(UNQUOTED, source)])

    def test_possessive_at_very_end_of_input(self):
        source = "Those boots are Jones'"
        self.assertEqual(segments(source), [(UNQUOTED, source)])

    def test_decade_elision_is_not_an_opener(self):
        source = "Back in the '90s nobody asked."
        self.assertEqual(segments(source), [(UNQUOTED, source)])

    def test_apostrophe_survives_inside_single_quoted_dialogue(self):
        self.assertEqual(
            segments("'It doesn't matter,' she said."),
            [(QUOTED, "'It doesn't matter,'"), (UNQUOTED, " she said.")],
        )

    def test_curly_apostrophe_survives_inside_curly_dialogue(self):
        self.assertEqual(
            segments("‘It doesn’t matter,’ she said."),
            [
                (QUOTED, "‘It doesn’t matter,’"),
                (UNQUOTED, " she said."),
            ],
        )

    def test_two_single_quoted_phrases(self):
        self.assertEqual(
            segments("She said 'yes' and then 'no' and left."),
            [
                (UNQUOTED, "She said "),
                (QUOTED, "'yes'"),
                (UNQUOTED, " and then "),
                (QUOTED, "'no'"),
                (UNQUOTED, " and left."),
            ],
        )


class TestRunawayOpenerBounds(unittest.TestCase):
    """Rule 5's two bounds: an ambiguous opener cannot swallow narration."""

    def test_audit_probe_filler_stays_unquoted(self):
        spans = tokenize(RUNAWAY_PROBE)
        self.assertEqual(
            segments(RUNAWAY_PROBE), [(UNQUOTED, RUNAWAY_PROBE)]
        )
        # Nothing anywhere may be quoted, and certainly nothing page-sized.
        for span in spans:
            self.assertEqual(span.kind, UNQUOTED)
        self.assertNotIn("Filler narration", "".join(
            s.text(RUNAWAY_PROBE) for s in spans if s.kind == QUOTED
        ))

    def test_closer_only_in_a_later_paragraph_does_not_open(self):
        source = CLOSER_IN_LATER_PARAGRAPH
        self.assertEqual(segments(source), [(UNQUOTED, source)])

    def test_paragraph_bound_uses_blank_line_not_single_newline(self):
        # A single newline inside a paragraph must NOT end the search.
        source = "She said 'Everything ends,\nand that is all.' He nodded."
        self.assertEqual(
            segments(source),
            [
                (UNQUOTED, "She said "),
                (QUOTED, "'Everything ends,\nand that is all.'"),
                (UNQUOTED, " He nodded."),
            ],
        )

    def test_blank_line_with_horizontal_whitespace_counts_as_a_break(self):
        source = "He said 'no more and left.\n   \nThe dogs' tails wagged.\n"
        self.assertEqual(segments(source), [(UNQUOTED, source)])

    def test_legitimate_dialogue_within_one_paragraph_still_quotes(self):
        source = (
            "'I have thought about it for a long time, and I do not think "
            "there is any way around it,' she said at last."
        )
        spans = tokenize(source)
        self.assertEqual(spans[0].kind, QUOTED)
        self.assertTrue(spans[0].text(source).endswith("around it,'"))
        self.assertEqual(spans[1].kind, UNQUOTED)

    def test_span_just_under_the_length_cap_still_quotes(self):
        body = "word " * 90  # ~450 chars, comfortably under the 500 cap
        source = "She said '" + body + "end.' He nodded."
        spans = tokenize(source)
        self.assertLess(len(body), MAX_AMBIGUOUS_SINGLE_SPAN)
        self.assertEqual(spans[1].kind, QUOTED)

    def test_span_over_the_length_cap_does_not_open(self):
        body = "word " * 200  # ~1000 chars, over the cap, single paragraph
        source = "She said '" + body + "end.' He nodded."
        self.assertEqual(segments(source), [(UNQUOTED, source)])

    def test_curly_single_is_subject_to_both_bounds_too(self):
        # ‘ used to be exempt. Measured on a real ebook, that let a wrong-way
        # smart apostrophe or an elided name open page-long "quotations".
        long_body = "word " * 200
        source = "She said ‘" + long_body + "end.’ He nodded."
        self.assertEqual(segments(source), [(UNQUOTED, source)])

        across = "She said ‘no more and left.\n\nThe rain kept on.’ He nodded."
        self.assertEqual(segments(across), [(UNQUOTED, across)])

    def test_curly_single_after_a_letter_never_opens(self):
        # Wrong-direction smart apostrophe: rule 1 rejects it exactly as it
        # rejects O'Brien / Jones'. No lexicon involved.
        source = "The cousins‘ tails wagged and the valuta‘s value fell."
        self.assertEqual(segments(source), [(UNQUOTED, source)])

    def test_curly_single_bounded_but_not_subject_to_rules_2_to_4(self):
        # Rules 2-4 read the FOLLOWING character to tell apostrophe from
        # quote -- a question ‘ does not raise. Elisions and digits still open.
        self.assertEqual(
            segments("‘Tis done,’ he said."),
            [(QUOTED, "‘Tis done,’"), (UNQUOTED, " he said.")],
        )
        self.assertEqual(
            segments("‘1984 was the year,’ she said."),
            [(QUOTED, "‘1984 was the year,’"), (UNQUOTED, " she said.")],
        )


class TestBritishSingleQuotedDialogue(unittest.TestCase):
    """Regression guard: books using ‘ … ’ as the PRIMARY dialogue mark."""

    def test_ordinary_single_quoted_dialogue_tokenizes(self):
        source = (
            "‘Hello,’ he said. ‘I did not expect you.’\n\n"
            "She shrugged. ‘Nor did I,’ she answered, ‘but here we are.’"
        )
        quoted = [t for kind, t in segments(source) if kind == QUOTED]
        self.assertEqual(
            quoted,
            [
                "‘Hello,’",
                "‘I did not expect you.’",
                "‘Nor did I,’",
                "‘but here we are.’",
            ],
        )
        self.assertEqual("".join(t for _, t in segments(source)), source)

    def test_apostrophes_inside_single_quoted_dialogue_do_not_close_it(self):
        source = "‘It doesn’t matter,’ she said."
        self.assertEqual(
            segments(source),
            [(QUOTED, "‘It doesn’t matter,’"), (UNQUOTED, " she said.")],
        )

    def test_double_quotes_are_exempt_from_both_bounds(self):
        long_body = "word " * 200
        source = 'She said "' + long_body + 'end." He nodded.'
        spans = tokenize(source)
        self.assertEqual(spans[1].kind, QUOTED)
        self.assertGreater(spans[1].end - spans[1].start, MAX_AMBIGUOUS_SINGLE_SPAN)


class TestNestedQuotes(unittest.TestCase):
    """Item 4: the outer quotation is ONE span; inner quotes stay inside."""

    def test_single_nested_in_double(self):
        self.assertEqual(
            segments('"She told me \'go away\' yesterday," he said.'),
            [
                (QUOTED, '"She told me \'go away\' yesterday,"'),
                (UNQUOTED, " he said."),
            ],
        )

    def test_curly_single_nested_in_curly_double(self):
        source = (
            "“She told me ‘go away’ yesterday,” he said."
        )
        self.assertEqual(
            segments(source),
            [
                (QUOTED, "“She told me ‘go away’ yesterday,”"),
                (UNQUOTED, " he said."),
            ],
        )

    def test_double_nested_in_single(self):
        self.assertEqual(
            segments("'She said \"go away\" to me,' he noted."),
            [
                (QUOTED, "'She said \"go away\" to me,'"),
                (UNQUOTED, " he noted."),
            ],
        )


class TestUnterminatedQuotes(unittest.TestCase):
    """Item 4: no crash, no lost text."""

    def test_unterminated_double_runs_to_end(self):
        self.assertEqual(
            segments('He said, "I am leaving'),
            [(UNQUOTED, "He said, "), (QUOTED, '"I am leaving')],
        )

    def test_unterminated_curly_double_runs_to_end(self):
        self.assertEqual(
            segments("He said, “I am leaving"),
            [(UNQUOTED, "He said, "), (QUOTED, "“I am leaving")],
        )

    def test_unterminated_curly_single_stays_unquoted(self):
        # Rule 5 now covers ‘ as well: with no closer at all, narration is the
        # safer reading. Text is intact either way.
        source = "He said, ‘I am leaving"
        self.assertEqual(segments(source), [(UNQUOTED, source)])

    def test_unterminated_ambiguous_single_stays_unquoted(self):
        # Rule 5: with no valid closer, apostrophe is the safer reading and
        # narrator fallback is the safe failure mode.
        source = "He said, 'I am leaving"
        self.assertEqual(segments(source), [(UNQUOTED, source)])

    def test_quote_opening_at_index_zero_has_no_empty_leading_span(self):
        spans = tokenize('"Yes."')
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].kind, QUOTED)
        self.assertEqual(spans[0].start, 0)


class TestUnquotedDialogue(unittest.TestCase):
    """Item 4: em-dash and unquoted dialogue stay UNQUOTED for the LLM."""

    def test_em_dash_dialogue(self):
        source = "— I am leaving, he said.\n— Then go."
        self.assertEqual(segments(source), [(UNQUOTED, source)])

    def test_unquoted_dialogue(self):
        source = "I am leaving, he said, and she did not stop him."
        self.assertEqual(segments(source), [(UNQUOTED, source)])

    def test_attribution_tag_is_never_dropped(self):
        source = '"I am leaving," said John, who never returned.'
        spans = tokenize(source)
        self.assertEqual(spans[-1].kind, UNQUOTED)
        self.assertIn("said John", spans[-1].text(source))


class TestStructuralEdges(unittest.TestCase):
    def test_adjacent_quotes_produce_no_empty_span(self):
        self.assertEqual(
            segments('"A""B"'), [(QUOTED, '"A"'), (QUOTED, '"B"')]
        )

    def test_whitespace_only_input(self):
        self.assertEqual(segments("   \n\t "), [(UNQUOTED, "   \n\t ")])

    def test_quote_spanning_a_newline(self):
        source = '"I am leaving,\nand not coming back," he said.\n'
        self.assertEqual(
            segments(source),
            [
                (QUOTED, '"I am leaving,\nand not coming back,"'),
                (UNQUOTED, " he said.\n"),
            ],
        )

    def test_mixed_straight_and_curly_in_one_document(self):
        source = (
            "“First,” he said. \"Second,\" she replied. "
            "‘Third,’ they agreed."
        )
        kinds = [s.kind for s in tokenize(source)]
        self.assertEqual(
            kinds,
            [QUOTED, UNQUOTED, QUOTED, UNQUOTED, QUOTED, UNQUOTED],
        )
        quoted = [s.text(source) for s in tokenize(source) if s.kind == QUOTED]
        self.assertEqual(
            quoted,
            ["“First,”", '"Second,"', "‘Third,’"],
        )

    def test_realistic_prose_quoted_spans(self):
        quoted = [
            s.text(PROSE_MIXED)
            for s in tokenize(PROSE_MIXED)
            if s.kind == QUOTED
        ]
        self.assertEqual(
            quoted,
            [
                '"We can\'t stay here,"',
                '"Not after what the Joneses\' boy\ntold me."',
                "'Everything ends, and that is the whole of\nit.'",
                '"Don\'t,"',
                '"Don\'t say \'ends\' to me tonight."',
            ],
        )

    def test_realistic_curly_prose_quoted_spans(self):
        quoted = [
            s.text(PROSE_CURLY)
            for s in tokenize(PROSE_CURLY)
            if s.kind == QUOTED
        ]
        self.assertEqual(
            quoted,
            [
                "“You were never going to tell me,”",
                "“I was,”",
                "“I was waiting for the right hour, and then there "
                "wasn’t one.”",
                "“She told me ‘go away’ and I went.”",
            ],
        )
        # The em-dash line must remain narration.
        tail = tokenize(PROSE_CURLY)[-1].text(PROSE_CURLY)
        self.assertIn("— Then go again", tail)


class TestGuillemets(unittest.TestCase):
    """French / Russian / Spanish « » and the single ‹ ›."""

    def test_simple_guillemets(self):
        self.assertEqual(segments("« Je pars. »"), [(QUOTED, "« Je pars. »")])

    def test_attribution_tag_after_guillemets(self):
        self.assertEqual(
            segments("« Je pars, » dit-il."),
            [(QUOTED, "« Je pars, »"), (UNQUOTED, " dit-il.")],
        )

    def test_leading_narration_before_guillemets(self):
        self.assertEqual(
            segments("Il a dit « Je pars » et il est parti."),
            [
                (UNQUOTED, "Il a dit "),
                (QUOTED, "« Je pars »"),
                (UNQUOTED, " et il est parti."),
            ],
        )

    def test_french_inner_nbsp_is_preserved_byte_exactly(self):
        for source in (FRENCH_NBSP, FRENCH_NNBSP):
            spans = tokenize(source)
            self.assertEqual(len(spans), 1)
            self.assertEqual(spans[0].kind, QUOTED)
            # byte-exact, including the no-break space glyphs themselves
            self.assertEqual(spans[0].text(source), source)
            self.assertEqual(
                spans[0].text(source).encode("utf-8"), source.encode("utf-8")
            )

    def test_nbsp_variants_are_distinct_and_not_normalized(self):
        self.assertNotEqual(FRENCH_NBSP, FRENCH_NNBSP)
        self.assertIn(" ", tokenize(FRENCH_NBSP)[0].text(FRENCH_NBSP))
        self.assertIn(" ", tokenize(FRENCH_NNBSP)[0].text(FRENCH_NNBSP))

    def test_unterminated_guillemet_runs_to_end(self):
        source = "Il a dit « Je pars et il n'est jamais revenu."
        self.assertEqual(
            segments(source),
            [(UNQUOTED, "Il a dit "), (QUOTED, "« Je pars et il n'est jamais revenu.")],
        )

    def test_single_guillemets(self):
        self.assertEqual(
            segments("‹ Oui ›, dit-elle."),
            [(QUOTED, "‹ Oui ›"), (UNQUOTED, ", dit-elle.")],
        )

    def test_single_guillemets_nested_in_double(self):
        source = "« Elle a dit ‹ va-t'en › hier, » dit-il."
        self.assertEqual(
            segments(source),
            [
                (QUOTED, "« Elle a dit ‹ va-t'en › hier, »"),
                (UNQUOTED, " dit-il."),
            ],
        )

    def test_unterminated_single_guillemet_runs_to_end(self):
        self.assertEqual(segments("‹ Oui"), [(QUOTED, "‹ Oui")])

    def test_russian_guillemets_with_nested_low_high(self):
        source = "Он сказал: «Она сказала „уходи“ вчера», и ушёл."
        self.assertEqual(
            segments(source),
            [
                (UNQUOTED, "Он сказал: "),
                (QUOTED, "«Она сказала „уходи“ вчера»"),
                (UNQUOTED, ", и ушёл."),
            ],
        )


class TestCornerBrackets(unittest.TestCase):
    """Japanese / Chinese 「」 and 『』."""

    def test_simple_corner_brackets(self):
        self.assertEqual(
            segments("「こんにちは」と彼は言った。"),
            [(QUOTED, "「こんにちは」"), (UNQUOTED, "と彼は言った。")],
        )

    def test_white_corner_brackets(self):
        self.assertEqual(
            segments("彼は『さようなら』と言った。"),
            [
                (UNQUOTED, "彼は"),
                (QUOTED, "『さようなら』"),
                (UNQUOTED, "と言った。"),
            ],
        )

    def test_white_nested_in_plain_corner_brackets(self):
        source = "「彼は『さようなら』と言った」と彼女は言った。"
        self.assertEqual(
            segments(source),
            [
                (QUOTED, "「彼は『さようなら』と言った」"),
                (UNQUOTED, "と彼女は言った。"),
            ],
        )

    def test_unterminated_corner_bracket_runs_to_end(self):
        self.assertEqual(segments("「終わりだ"), [(QUOTED, "「終わりだ")])

    def test_full_width_double_quote(self):
        self.assertEqual(
            segments("＂やあ＂と彼は言った。"),
            [(QUOTED, "＂やあ＂"), (UNQUOTED, "と彼は言った。")],
        )


class TestLowHighQuotes(unittest.TestCase):
    """German / Polish „ … “ and ‚ … ‘.

    ``“`` is an OPENER in English and a CLOSER here. State decides: in
    NARRATION it opens; inside a ``„`` span only ``„``'s closers are consulted.
    """

    def test_simple_low_high(self):
        self.assertEqual(
            segments("„Ich gehe,“ sagte er."),
            [(QUOTED, "„Ich gehe,“"), (UNQUOTED, " sagte er.")],
        )

    def test_low_high_single(self):
        self.assertEqual(
            segments("‚Ja‘, sagte sie."),
            [(QUOTED, "‚Ja‘"), (UNQUOTED, ", sagte sie.")],
        )

    def test_single_nested_in_double_low_high(self):
        source = "„Sie sagte ‚geh weg‘ gestern,“ sagte er."
        self.assertEqual(
            segments(source),
            [
                (QUOTED, "„Sie sagte ‚geh weg‘ gestern,“"),
                (UNQUOTED, " sagte er."),
            ],
        )

    def test_low_high_also_closes_on_right_curly(self):
        # Some typesetters close „ with ” rather than “.
        self.assertEqual(
            segments("„Ich gehe,” sagte er."),
            [(QUOTED, "„Ich gehe,”"), (UNQUOTED, " sagte er.")],
        )

    def test_unterminated_low_high_runs_to_end(self):
        self.assertEqual(segments("„Ich gehe"), [(QUOTED, "„Ich gehe")])


class TestEnglishUnchangedByNewConventions(unittest.TestCase):
    """Regression guard: adding „ and ‚ must not disturb “ ” ‘ ’ or " '."""

    def test_curly_double_still_opens_in_narration(self):
        self.assertEqual(
            segments("“I am leaving,” he said."),
            [(QUOTED, "“I am leaving,”"), (UNQUOTED, " he said.")],
        )

    def test_straight_double_still_closes_on_curly_and_vice_versa(self):
        self.assertEqual(segments('"Mixed,” he said.')[0][1], '"Mixed,”')
        self.assertEqual(segments('“Mixed," he said.')[0][1], '“Mixed,"')

    def test_curly_single_still_opens_in_narration(self):
        self.assertEqual(
            segments("‘Hello,’ she said."),
            [(QUOTED, "‘Hello,’"), (UNQUOTED, " she said.")],
        )

    def test_ambiguous_single_machinery_untouched(self):
        self.assertEqual(
            segments("Don't touch Jones' hat; 'tis his."),
            [(UNQUOTED, "Don't touch Jones' hat; 'tis his.")],
        )
        self.assertEqual(
            segments("'Hello,' she said."),
            [(QUOTED, "'Hello,'"), (UNQUOTED, " she said.")],
        )
        self.assertEqual(segments(RUNAWAY_PROBE), [(UNQUOTED, RUNAWAY_PROBE)])

    def test_english_prose_fixtures_are_byte_identical_to_before(self):
        for source in (PROSE_MIXED, PROSE_CURLY):
            self.assertEqual(reassemble(tokenize(source), source), source)


class TestMixedConventionDocument(unittest.TestCase):
    def test_four_conventions_in_one_document(self):
        source = MULTILINGUAL_DOC
        quoted = [s.text(source) for s in tokenize(source) if s.kind == QUOTED]
        self.assertEqual(
            quoted,
            [
                FRENCH_INNER,
                "「こんにちは」",
                "„Guten Tag,“",
                '"Hello,"',
            ],
        )
        self.assertTrue(validate_spans(tokenize(source), source))

    def test_opener_of_one_convention_is_not_closed_by_another(self):
        # A stray » inside an English quotation must not close it, and a stray
        # " inside a guillemet quotation must not close that.
        self.assertEqual(
            segments('"a » b" c'),
            [(QUOTED, '"a » b"'), (UNQUOTED, " c")],
        )
        self.assertEqual(
            segments('« a " b » c'),
            [(QUOTED, '« a " b »'), (UNQUOTED, " c")],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
