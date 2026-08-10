"""Every way of generating a script must answer the same questions the same way.

WHY THIS FILE EXISTS. Three defects in one day, all the same shape: a decision
implemented more than once, and the copies disagreeing.

  1. "is an adjacent duplicate block a defect?" lived in FOUR places.
     chunk_quality and pass_quality had it right - only flag when the source
     contains the block once. script_repair and the whole-book audit refused
     any block the source did not contain exactly once. grimgar03 opens with
     its title eight times, so it failed at chunk 1, was fixed, generated all
     49 chunks, and was then rejected AGAIN at the final gate by the other
     wrong copy.

  2. "how much decode damage may a source carry?" had THREE answers.
     generate_script graded it at 0.5%, three_pass_generate refused any count,
     and the per-entry audit blocked at any count. A repaired index18 would
     generate single-pass and be refused three-pass, for identical input.

  3. The per-chunk gate refused what the source gate had just accepted, so
     relaxing the front door moved the refusal to chunk 31 rather than opening
     the book.

Unit tests did not catch any of these, because each copy was correct against
its own tests. What was missing was a test that the copies AGREE. That is what
this file is: it asserts the shared definitions are the only definitions, and
that the same input gets the same verdict from every path.

WHEN THIS FAILS, DO NOT EDIT THE TEST. A new local threshold or a second
duplicate rule is the defect it is built to catch. Point the new caller at the
shared function instead.
"""
import inspect
import re
import unittest

import chunk_quality
import generate_script
import pass_quality
import script_preflight
import script_repair
import three_pass_generate


class ReplacementPolicyAgreementTest(unittest.TestCase):
    """One definition of how much decode damage is acceptable."""

    def test_the_policy_lives_in_exactly_one_place(self):
        defining = [module.__name__ for module in
                    (script_preflight, generate_script, three_pass_generate,
                     chunk_quality, pass_quality, script_repair)
                    if "MAX_REPLACEMENT_SHARE = " in inspect.getsource(module)]
        self.assertEqual(["script_preflight"], defining,
                         "the replacement-character limit must be defined once; "
                         f"found definitions in {defining}")

    def test_both_generators_use_the_shared_check(self):
        for module in (generate_script, three_pass_generate):
            with self.subTest(module=module.__name__):
                source = inspect.getsource(module)
                self.assertIn("replacement_load_is_acceptable", source,
                              f"{module.__name__} must ask the shared policy, "
                              "not re-derive a threshold")

    def test_generators_agree_on_the_same_input(self):
        """The concrete case that broke: repaired index18's damage level."""
        repaired = int(0.0026 * 476376)     # index18 after repair, 0.26%
        corrupt = int(0.0140 * 476376)      # index18 raw, 1.40%
        total = 476376
        self.assertTrue(
            script_preflight.replacement_load_is_acceptable(repaired, total),
            "a repaired book must be accepted by every path")
        self.assertFalse(
            script_preflight.replacement_load_is_acceptable(corrupt, total),
            "a mis-decoded book must still be refused by every path")

    def test_a_clean_source_is_always_acceptable(self):
        self.assertTrue(
            script_preflight.replacement_load_is_acceptable(0, 1000))


class DuplicateBlockAgreementTest(unittest.TestCase):
    """Every caller must treat a source-repeated block as faithful."""

    TITLE = "Grimgar of Fantasy and Ash: Volume 3"

    def _texts_and_source(self):
        texts = [self.TITLE.casefold()] * 4
        source = "\n".join([self.TITLE] * 8)
        return texts, source

    def test_the_detector_reports_the_source_count(self):
        texts, source = self._texts_and_source()
        findings = script_preflight.find_adjacent_duplicate_blocks(texts, source)
        self.assertTrue(findings)
        self.assertGreaterEqual(
            findings[0]["details"]["source_occurrences"], 2,
            "the detector must report how often the SOURCE has the block; "
            "every caller's decision depends on it")

    def test_no_caller_blocks_a_source_repeated_block(self):
        texts, source = self._texts_and_source()
        findings = script_preflight.find_adjacent_duplicate_blocks(texts, source)
        self.assertEqual("manual_review", findings[0]["severity"])

        entries = [{"text": self.TITLE, "speaker": "NARRATOR"}] * 4
        repair = script_repair.build_deterministic_repair(entries, source)
        self.assertEqual([], repair["unresolved"],
                         "the repair path must not call a faithful repeat "
                         "unresolvable")

    def test_every_caller_keys_off_source_occurrences(self):
        """Guards the two callers that were already correct.

        chunk_quality and pass_quality flag only when the source contains the
        block once. If either ever drops that condition it starts refusing
        faithful repeats, which is defect 1 all over again.
        """
        for module in (chunk_quality, pass_quality):
            with self.subTest(module=module.__name__):
                source = inspect.getsource(module)
                index = source.find("find_adjacent_duplicate_blocks(")
                self.assertGreater(index, 0)
                window = source[index:index + 400]
                self.assertIn("source_occurrences", window,
                              f"{module.__name__} must consider how often the "
                              "source contains the block before flagging it")


class GateOrderingTest(unittest.TestCase):
    """A later gate must not refuse what an earlier one accepted."""

    def test_entry_audit_does_not_re_refuse_accepted_damage(self):
        """index18's exact failure: front door open, chunk 31 shut.

        The per-entry audit used to mark any replacement character blocking,
        so a source admitted by the graded gate could never produce an
        acceptable entry.
        """
        source = inspect.getsource(script_preflight.audit_script)
        marker = 'unicode_report["replacement_character_count"]'
        self.assertIn(marker, source)
        window = source[source.find(marker):source.find(marker) + 300]
        self.assertNotIn('_finding("blocking"', window,
                         "replacement characters the source gate accepted must "
                         "not be blocking at entry level")

    def test_unsafe_controls_stay_blocking_everywhere(self):
        """The thing that must NOT be relaxed by any of this."""
        source = inspect.getsource(script_preflight.audit_script)
        self.assertIn('unicode_report["unsafe_controls"]', source)
        marker = 'if unicode_report["unsafe_controls"]:'
        window = source[source.find(marker):source.find(marker) + 200]
        self.assertIn('"blocking"', window)


if __name__ == "__main__":
    unittest.main()
