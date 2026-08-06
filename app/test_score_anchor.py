"""A ceiling below the arms it bounds must not pass silently.

WHAT HAPPENED. The voice comparison scores each generated arm against a
`human_vs_human` anchor - the same narrator reading different held-out
material - because an ECAPA similarity means nothing without knowing what one
person scores against herself.

On 2026-08-06 three language arms were scored and read side by side:

    English    ceiling 0.809   arms 0.690 / 0.757    median clip 7.33s
    Japanese   ceiling 0.796   arms 0.755 / 0.779    median clip 4.71s
    Chinese    ceiling 0.691   arms 0.720 / 0.765    median clip 3.17s

In Chinese the narrator matched herself WORSE than a synthetic voice matched
her, so the anchor was not measuring speaker identity there and its arm
comparison could not be read. The scorer printed "human_vs_human is the
CEILING" and said nothing, and all three sets were compared as though sound.

These tests pin the detection, not the explanation. Short clips are the
leading hypothesis for why the Chinese anchor collapsed - it has the shortest
audio by a wide margin - but that is untested and the code says so.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.ljspeech_score import find_invalid_anchors


def summary(ceiling, **arms):
    out = {"human_vs_human": {"ecapa": ceiling}}
    out.update({name: {"ecapa": value} for name, value in arms.items()})
    return out


class AnchorValidationTest(unittest.TestCase):

    ARMS = ["lora", "clone"]

    def test_the_chinese_case_is_flagged(self):
        """The real numbers that went unremarked."""
        found = find_invalid_anchors(
            summary(0.691, lora=0.720, clone=0.765), self.ARMS)
        self.assertEqual([item["arm"] for item in found], ["lora", "clone"])
        self.assertEqual(found[0]["ceiling_ecapa"], 0.691)

    def test_the_english_and_japanese_cases_are_not(self):
        """A check that fires on a sound anchor is worse than none - it would
        train the reader to ignore it."""
        self.assertEqual(find_invalid_anchors(
            summary(0.809, lora=0.690, clone=0.757), self.ARMS), [])
        self.assertEqual(find_invalid_anchors(
            summary(0.796, lora=0.755, clone=0.779), self.ARMS), [])

    def test_only_the_offending_arm_is_named(self):
        """One arm over the ceiling does not condemn the other."""
        found = find_invalid_anchors(
            summary(0.700, lora=0.650, clone=0.800), self.ARMS)
        self.assertEqual([item["arm"] for item in found], ["clone"])

    def test_equal_to_the_ceiling_is_not_a_violation(self):
        """Strictly greater. An arm matching the ceiling is suspicious but not
        contradictory, and a boundary that flags ties would fire on rounding."""
        self.assertEqual(find_invalid_anchors(
            summary(0.750, lora=0.750), ["lora"]), [])

    def test_a_missing_anchor_is_not_reported_as_invalid(self):
        """No ceiling is a different problem - absent evidence, not
        contradictory evidence - and must not be dressed up as this one."""
        self.assertEqual(find_invalid_anchors({"lora": {"ecapa": 0.9}},
                                              ["lora"]), [])
        self.assertEqual(find_invalid_anchors(
            {"human_vs_human": {"ecapa": None}, "lora": {"ecapa": 0.9}},
            ["lora"]), [])

    def test_an_arm_without_a_score_is_skipped(self):
        found = find_invalid_anchors(
            {"human_vs_human": {"ecapa": 0.7}, "lora": {"ecapa": None},
             "clone": {"ecapa": 0.8}}, self.ARMS)
        self.assertEqual([item["arm"] for item in found], ["clone"])

    def test_real_artifacts_agree_with_the_check(self):
        """Run against the artifacts on disk, so the thresholds stay tied to
        measured data rather than to numbers retyped into a test."""
        import json
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        expected = {"ljspeech": False, "kokoro": False, "aishell3": True}
        checked = 0
        for tag, should_flag in expected.items():
            path = os.path.join(root, "ab_test_runtime", "experiments",
                                f"{tag}_score.json")
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as fh:
                doc = json.load(fh)
            found = find_invalid_anchors(doc["summary"], doc["arms"])
            self.assertEqual(bool(found), should_flag,
                             f"{tag}: anchor verdict changed")
            checked += 1
        self.assertGreater(checked, 0, "no score artifacts found to check")


if __name__ == "__main__":
    unittest.main()
