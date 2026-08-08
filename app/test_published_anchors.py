"""No score artifact on disk may publish comparisons under a broken ceiling.

WHAT `test_score_anchor.py` ALREADY COVERS, AND WHAT IT DOES NOT.
`find_invalid_anchors` is tested there against constructed inputs: given a
ceiling below its arms, does the detector notice. That pins the detector.

It does not pin the artifacts. Goal 6.1 is not "the detector works", it is
"0 comparisons published from an eval set with an invalid anchor" - a claim
about the files this repo actually ships. A correct detector whose output
nobody reads produces exactly the 2026-08-06 failure it was written for: the
Chinese anchor sat below both its arms, the scorer printed a line about the
ceiling, and all three sets were read side by side anyway.

So this walks every `*_score.json` in the evidence tree and asserts the arms
are bounded. It is deliberately a test over real artifacts rather than
fixtures: fixtures cannot go stale, and staleness is the failure mode here -
someone re-scores one set, the anchor regresses, and nothing objects because
the detector still passes its own unit tests.

An artifact that predates the `anchor_invalid` field is not silently accepted.
Missing evidence and clean evidence are different states, and treating the
first as the second is how an unbounded comparison gets published.
"""
import glob
import json
import os
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENTS = os.path.join(REPO, "ab_test_runtime", "experiments")

# The key under `summary` holding the ceiling. Every arm beside it is bounded
# by it, which is what makes a bare max() over the other keys correct here.
ANCHOR_KEY = "human_vs_human"


def score_artifacts():
    return sorted(glob.glob(os.path.join(EXPERIMENTS, "*_score.json")))


class PublishedAnchorsTest(unittest.TestCase):

    def test_some_score_artifacts_exist(self):
        """Guards the rest of this file from passing vacuously.

        Every assertion below iterates over the artifact list, so an empty
        list would make them all pass while checking nothing - the shape of
        green test suite that proves the opposite of what it claims.
        """
        self.assertTrue(score_artifacts(),
                        "no *_score.json artifacts found; the anchor "
                        "assertions below would pass without checking "
                        "anything")

    def test_every_artifact_records_anchor_validity(self):
        for path in score_artifacts():
            with self.subTest(artifact=os.path.basename(path)):
                with open(path, encoding="utf-8") as handle:
                    data = json.load(handle)
                self.assertIn(
                    "anchor_invalid", data,
                    f"{os.path.basename(path)} predates the anchor check; "
                    "re-score it rather than assuming it was sound")

    def test_no_artifact_publishes_an_invalid_anchor(self):
        for path in score_artifacts():
            with self.subTest(artifact=os.path.basename(path)):
                with open(path, encoding="utf-8") as handle:
                    data = json.load(handle)
                self.assertEqual(
                    data.get("anchor_invalid"), [],
                    f"{os.path.basename(path)} publishes comparisons under an "
                    f"invalid anchor: {data.get('anchor_invalid')}")

    def test_anchor_actually_exceeds_every_arm(self):
        """Re-derive the verdict instead of trusting the recorded flag.

        `anchor_invalid` is written by the same run that computed the scores.
        If that run's logic regresses, the flag regresses with it and an
        artifact stays green while being wrong. Reading the summary directly
        is an independent check on the same file.
        """
        for path in score_artifacts():
            with self.subTest(artifact=os.path.basename(path)):
                with open(path, encoding="utf-8") as handle:
                    summary = json.load(handle).get("summary") or {}
                anchor = (summary.get(ANCHOR_KEY) or {}).get("ecapa")
                # Not a skip: a score artifact with no ceiling is precisely
                # the unbounded comparison this goal forbids, and skipping
                # would report it as green. Goal 6.4 keeps this suite at zero
                # skips for the same reason.
                self.assertIsNotNone(
                    anchor,
                    f"{os.path.basename(path)} has no {ANCHOR_KEY} arm, so "
                    "its comparisons are unbounded")
                for arm, values in summary.items():
                    if arm == ANCHOR_KEY or not isinstance(values, dict):
                        continue
                    score = values.get("ecapa")
                    if score is None:
                        continue
                    self.assertLess(
                        score, anchor,
                        f"{os.path.basename(path)}: arm {arm} scores "
                        f"{score:.4f} against a ceiling of {anchor:.4f}")


if __name__ == "__main__":
    unittest.main()
