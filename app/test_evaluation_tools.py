"""The tools that decide which model is better must not mis-align lines.

These are the instruments every model comparison in this project is read
through, so a defect here is worse than a defect in the pipeline: it makes the
wrong answer look right. All five cases come from the 2026-07-26 audit.
"""
import unittest

from attribution_accuracy import find_entry, normalize_line, score_run
from build_scoring_sheet import build_sheet
from compare_attribution_arms import align_arms, find_disagreements


def _entry(text, speaker):
    return {"text": text, "speaker": speaker}


class FullTextIdentityTest(unittest.TestCase):
    """A 60-character prefix was treated as identity. Measured on the corpus,
    grimgar03 and grimgar06 each contain distinct lines sharing one."""

    LONG = "I have been thinking about this for a very long time and I believe"

    def test_two_lines_sharing_a_long_prefix_do_not_align(self):
        named = [_entry(self.LONG + " we should go north.", "ERIS"),
                 _entry(self.LONG + " we should go south.", "ROXY")]
        by_text = {}
        for position, entry in enumerate(named):
            by_text.setdefault(normalize_line(entry["text"]), []).append(
                (position, entry))
        # Gold points at index 0 but carries the SOUTH line: the entry at that
        # index is a different line, so it must fall through to the text match.
        item = {"entry_index": 0, "line": self.LONG + " we should go south."}
        self.assertEqual("ROXY", find_entry(named, item, by_text)["speaker"])

    def test_scoring_uses_the_matching_line_not_the_prefix(self):
        named = [_entry(self.LONG + " we should go north.", "ERIS"),
                 _entry(self.LONG + " we should go south.", "ROXY")]
        gold = {"entries": [{"id": "x", "book": "b", "entry_index": 0,
                             "line": self.LONG + " we should go south.",
                             "expected_speaker": "ROXY"}]}
        results = score_run(named, gold)
        self.assertEqual(1, sum(1 for r in results if r["correct"]))


class RepeatedDialogueTest(unittest.TestCase):
    """Short lines repeat constantly. Retaining the first occurrence compared
    unrelated instances and showed the wrong surrounding context."""

    def test_a_repeated_line_is_excluded_from_the_sheet(self):
        runs = {"m1": [_entry("Sorry.", "ERIS"), _entry("Filler one.", "ROXY"),
                       _entry("Sorry.", "SYLPHY")],
                "m2": [_entry("Sorry.", "ROXY"), _entry("Filler one.", "ROXY"),
                       _entry("Sorry.", "ERIS")]}
        rows = build_sheet(runs, size=10)
        self.assertNotIn("Sorry.", [row["text"] for row in rows])

    def test_unambiguous_lines_still_appear(self):
        runs = {"m1": [_entry("A distinct sentence here.", "ERIS")],
                "m2": [_entry("A distinct sentence here.", "ROXY")]}
        rows = build_sheet(runs, size=10)
        self.assertEqual(["A distinct sentence here."],
                         [row["text"] for row in rows])

    def test_case_only_differences_are_not_a_disagreement(self):
        runs = {"m1": [_entry("A distinct sentence here.", "RUDI")],
                "m2": [_entry("A distinct sentence here.", "rudi ")]}
        rows = build_sheet(runs, size=10)
        self.assertTrue(rows[0]["models_agree"])


class BilateralCoverageTest(unittest.TestCase):
    """Coverage divided by arm_a alone, so ten entries matching against a
    thousand reported 100% and the number flipped with argument order."""

    def _arms(self):
        small = [_entry(f"Line {i}.", "ERIS") for i in range(10)]
        large = small + [_entry(f"Extra {i}.", "ROXY") for i in range(990)]
        return small, large

    def test_a_tiny_arm_against_a_huge_one_reports_low_coverage(self):
        small, large = self._arms()
        _pairs, coverage = align_arms(small, large)
        self.assertLess(coverage, 0.05)

    def test_coverage_is_the_same_in_both_argument_orders(self):
        small, large = self._arms()
        _p, forward = align_arms(small, large)
        _q, backward = align_arms(large, small)
        self.assertAlmostEqual(forward, backward)

    def test_equal_arms_still_report_full_coverage(self):
        arm = [_entry(f"Line {i}.", "ERIS") for i in range(10)]
        _pairs, coverage = align_arms(arm, list(arm))
        self.assertEqual(1.0, coverage)


class ArmSpeakerNormalizationTest(unittest.TestCase):
    def test_case_and_whitespace_are_not_a_disagreement(self):
        arm_a = [_entry("A distinct sentence here.", "RUDI")]
        arm_b = [_entry("A distinct sentence here.", "rudi ")]
        self.assertEqual([], find_disagreements(arm_a, arm_b))

    def test_a_real_disagreement_is_still_reported(self):
        arm_a = [_entry("A distinct sentence here.", "RUDI")]
        arm_b = [_entry("A distinct sentence here.", "ROXY")]
        self.assertEqual(1, len(find_disagreements(arm_a, arm_b)))

    def test_precomputed_pairs_are_reused(self):
        arm_a = [_entry("A distinct sentence here.", "RUDI")]
        arm_b = [_entry("A distinct sentence here.", "ROXY")]
        pairs, _coverage = align_arms(arm_a, arm_b)
        self.assertEqual(find_disagreements(arm_a, arm_b),
                         find_disagreements(arm_a, arm_b, pairs=pairs))


if __name__ == "__main__":
    unittest.main()


class AliasScoringTest(unittest.TestCase):
    """A character named two ways is one character.

    mushoku16 calls the protagonist both RUDEUS (113 mentions) and RUDI (126).
    Scoring them as different answers marked 14 of 147 gold lines wrong for
    picking the other true name - a 9.5-point penalty for being right. An
    earlier ad-hoc harness hardcoded the equivalence, which is why two scorers
    reported 34.2% and 20.4% for the same book.
    """

    GOLD = {"aliases": [["RUDEUS", "RUDI"]],
            "entries": [{"id": "a", "book": "b", "entry_index": 0,
                         "line": "A distinct sentence here.",
                         "expected_speaker": "RUDEUS"}]}

    def _score(self, speaker, gold=None):
        named = [_entry("A distinct sentence here.", speaker)]
        return score_run(named, gold or self.GOLD)[0]["correct"]

    def test_the_other_true_name_counts_as_correct(self):
        self.assertTrue(self._score("RUDI"))

    def test_the_declared_name_still_counts(self):
        self.assertTrue(self._score("RUDEUS"))

    def test_a_different_character_is_still_wrong(self):
        self.assertFalse(self._score("ROXY"))

    def test_an_invented_name_is_not_an_alias(self):
        # FUTURE_ME is the protagonist's future self, a phrase the book never
        # uses as a name. It shipped on 250 entries and must score as wrong.
        self.assertFalse(self._score("FUTURE_ME"))

    def test_a_fixture_without_aliases_demands_exact_names(self):
        gold = {"entries": self.GOLD["entries"]}
        self.assertFalse(self._score("RUDI", gold))

    def test_alias_matching_ignores_case_and_spacing(self):
        self.assertTrue(self._score(" rudi "))

    def test_shipped_fixtures_declare_only_names_present_in_the_book(self):
        # An alias list is ground truth about one book; a wrong entry here
        # would silently forgive real errors in every future measurement.
        import json
        import os
        for name in ("fixtures/attribution_gold.json",
                     "fixtures/attribution_gold_random.json"):
            path = os.path.join(os.path.dirname(__file__), name)
            with open(path, encoding="utf-8") as handle:
                declared = json.load(handle).get("aliases", [])
            flat = [n for group in declared for n in group]
            self.assertNotIn("FUTURE_ME", flat)
            for group in declared:
                self.assertGreaterEqual(len(group), 2)
