"""A repair that removes every marker and breaks the prose is not a repair.

WHAT THIS PREVENTS. The first version of the repairer substituted a dash for
every unresolvable run. Runs that were "em dash + closing quote" became two
dashes, so 69 dialogue lines opened a quote and never closed it. Quote
imbalance went to 25.1% where undamaged books run 0.5-3.4%.

The consequence was not cosmetic. index18's chunk 10 fed the model a passage
whose speech never closed; it generated until it hit the 16,384-token ceiling,
failed coverage, and retried at 6.5 minutes an attempt, nine times. The repair
had reported 100% success.

So the repairer now checks its own output, and the check distinguishes two
things:

    REGRESSIONS - shapes measured to break generation. These refuse the write.
    SHARE       - a heuristic. A quote spanning paragraphs is normal prose and
                  counts as unbalanced, so this cannot reach zero and only
                  warns.

These tests pin both, and pin that the historical failure is refused.
"""
import unittest

from repair_source_encoding import (LAST_RESORT, MAX_UNBALANCED_QUOTE_SHARE,
                                    quote_balance, repair,
                                    structural_regressions)

OPEN = "“"
CLOSE = "”"
FFFD = "�"


class QuoteBalanceTest(unittest.TestCase):

    def test_balanced_prose_reports_no_imbalance(self):
        text = f"{OPEN}Hello there.{CLOSE}\n{OPEN}And again.{CLOSE}\n"
        unbalanced, quoted, share = quote_balance(text)
        self.assertEqual(0, unbalanced)
        self.assertEqual(2, quoted)
        self.assertEqual(0.0, share)

    def test_lines_without_quotes_are_not_counted(self):
        text = "Plain narration with no speech at all.\nAnother line.\n"
        _unbalanced, quoted, _share = quote_balance(text)
        self.assertEqual(0, quoted,
                         "only lines containing quotes belong in the ratio")


class StructuralRegressionTest(unittest.TestCase):

    def test_the_exact_shape_that_broke_chunk_10(self):
        """A line that opens speech and ends in a substituted dash."""
        text = f"{OPEN}This means a life debt, my Florice{LAST_RESORT}\n"
        found = structural_regressions(text)
        self.assertEqual(1, found["open_quote_ended_with_dash"])

    def test_a_line_closing_speech_it_never_opened(self):
        text = f"{LAST_RESORT}{LAST_RESORT}More men to die.{CLOSE}\n"
        found = structural_regressions(text)
        self.assertEqual(1, found["close_quote_started_with_dash"])

    def test_correctly_repaired_dialogue_has_no_regressions(self):
        text = (f"{OPEN}This means a life debt, my Florice{LAST_RESORT}{CLOSE}\n"
                f"{OPEN}{LAST_RESORT}More men to die.{CLOSE}\n")
        found = structural_regressions(text)
        self.assertEqual(0, found["open_quote_ended_with_dash"])
        self.assertEqual(0, found["close_quote_started_with_dash"])


class RepairPreservesStructureTest(unittest.TestCase):

    def test_a_damaged_dialogue_line_comes_back_closed(self):
        """End to end: the run that was 'dash + closing quote'.

        This is the case the first version got wrong, and it is the reason the
        repairer closes speech before substituting anything.
        """
        damaged = (f"{FFFD}This means a life debt, my Florice{FFFD}{FFFD}\n\n"
                   f"{FFFD}Wait, what?!{FFFD}{FFFD}\n")
        repaired, applied, _examples = repair(damaged)
        self.assertNotIn(FFFD, repaired)
        found = structural_regressions(repaired)
        self.assertEqual(0, found["open_quote_ended_with_dash"],
                         f"repair left a harmful shape: {repaired!r}")
        self.assertEqual(0, found["close_quote_started_with_dash"],
                         f"repair left a harmful shape: {repaired!r}")
        for line in repaired.split("\n"):
            if line.strip():
                self.assertEqual(line.count(OPEN), line.count(CLOSE),
                                 f"speech left unclosed in {line!r}")

    def test_substitution_is_recorded_not_silent(self):
        """A character replaced by a stand-in must be countable.

        These are positions where the original is unrecoverable. Reporting
        them is what separates 'repaired' from 'papered over'.
        """
        damaged = f"a{FFFD}b {FFFD}quoted{FFFD}\n"
        _repaired, applied, _examples = repair(damaged)
        self.assertTrue(applied,
                        "every substitution must appear in the applied counts")

    def test_the_limit_stays_meaningful(self):
        """The limit must sit above books that demonstrably generate."""
        self.assertGreater(MAX_UNBALANCED_QUOTE_SHARE, 0.0)
        # Calibrated on the whole corpus: owarimonogatari3 is 11.7% and
        # generates 110/110 chunks, so a limit at or below that would flag a
        # working book. mushoku18 reaches 20.3%.
        self.assertGreater(MAX_UNBALANCED_QUOTE_SHARE, 0.117)
        self.assertLessEqual(MAX_UNBALANCED_QUOTE_SHARE, 0.40)




class RepetitionTrapTest(unittest.TestCase):
    """A long repeating sequence makes the model generate to its token ceiling.

    index18's chunk 10 carried 25 repetitions of a damaged pair. Every attempt
    ran to 16,384 tokens and failed coverage; three runs died on it. The file
    was damaged twice - an earlier lossy conversion left literal "?" between
    letters ("O?o?o?o?h?h?h", a roar), and the bad decode left U+FFFD - so
    both kinds are collapsed.

    The line the tests must NOT touch is the author's own elongation:
    "three feet deeeeeeeeeeeeeeep" is style, not damage.
    """

    def test_a_damaged_repeated_pair_is_collapsed(self):
        damaged = "“" + (FFFD + "?") * 25 + "!!”\n"
        repaired, applied, _examples = repair(damaged)
        self.assertNotIn(FFFD, repaired)
        self.assertLess(len(repaired), 30,
                        f"25 repetitions should collapse, got {repaired!r}")

    def test_a_pre_existing_repetition_is_collapsed(self):
        text = "“O?o?o?o?o?o?o?o?h?h?h?h?h?h?h?h?h!!”\n"
        repaired, _applied, _examples = repair(text)
        self.assertEqual(0, structural_regressions(repaired)["repetition_traps"],
                         f"still a trap: {repaired!r}")

    def test_authorial_elongation_is_left_alone(self):
        """A repeated LETTER is style; only punctuation runs are damage."""
        text = "That river is only three feet deeeeeeeeeeeeeeeeeeep!!\n"
        repaired, _applied, _examples = repair(text)
        self.assertIn("deeeeeeeeeeeeeeeeeeep", repaired,
                      "the author's elongated word must survive")

    def test_a_trap_is_a_reported_regression(self):
        trapped = "“" + ("—?" * 12) + "!!”\n"
        self.assertGreater(
            structural_regressions(trapped)["repetition_traps"], 0,
            "a long repeating punctuation run must be reported as harmful")




class InlineQuotePairingTest(unittest.TestCase):
    """A quoted phrase mid-sentence is not speech at a line boundary.

    "dodging one or two ?cannon shots,? he couldn't rest easy" is a quoted
    phrase inside narration. None of the positional rules match it - there is
    no sentence punctuation before the marker and no line break after it - so
    the opener fell through to a dash, leaving

        dodging one or two -cannon shots," he couldn't

    which combines a quote delimiter with narration. index18's three-pass arm
    failed pass 1 on chunk 98 of 164 for exactly that, and 45 lines had the
    shape.

    The rule pairs an unmatched closing quote with an opener earlier in the
    same line, which is the only evidence available and is local enough to be
    safe.
    """

    def test_a_mid_sentence_quoted_phrase_gets_its_opener(self):
        damaged = f"dodging one or two {FFFD}cannon shots,{FFFD} he could not rest\n"
        repaired, _applied, _examples = repair(damaged)
        self.assertIn(f"{OPEN}cannon shots,{CLOSE}", repaired,
                      f"expected a paired quote, got {repaired!r}")

    def test_narration_without_any_closing_quote_is_left_to_other_rules(self):
        """No unmatched closer means no evidence for an opener here."""
        damaged = f"he ran forward{FFFD} and the ground shook\n"
        repaired, _applied, _examples = repair(damaged)
        self.assertNotIn(OPEN, repaired,
                         "an opener must not be invented without a closer to "
                         f"pair it with: {repaired!r}")

    def test_balanced_speech_is_untouched(self):
        text = f"{OPEN}Already balanced,{CLOSE} he said.\n"
        repaired, _applied, _examples = repair(text)
        self.assertEqual(text, repaired)


if __name__ == "__main__":
    unittest.main()
