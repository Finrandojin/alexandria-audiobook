"""The reference clip must represent the dataset it anchors.

WHAT THIS PROTECTS. `train_lora.py` extracts the speaker embedding from ONE
reference clip and applies it to every training sample. The dataset builder
chose `ref_index = 0` - whatever came first - and never checked it.

Measured across the 75 shipped adapters:

    reference mismatched (<0.3):  7 adapters, 6 of them poor  (86%)
    reference matching:          67 adapters, 9 of them poor  (13%)

6.4x more likely to fail, correlation +0.76 with adapter quality. The worst
case was anchored to a clip scoring -0.026 against its own dataset - actively
not that speaker - while a 0.882 clip sat unused in the same data.

The tests use a stubbed similarity function. What is under test is the
SELECTION LOGIC and its refusal behaviour, not the speaker model: whether the
medoid is identified, and whether an unavailable model degrades to "no opinion"
rather than to a confident wrong answer.
"""
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import voice_reference
from voice_reference import select_reference_sample


def make_clips(n):
    """n touchable files; contents irrelevant, similarity is stubbed."""
    tmp = tempfile.TemporaryDirectory()
    paths = []
    for i in range(n):
        p = os.path.join(tmp.name, f"sample_{i:03d}.wav")
        with open(p, "wb") as fh:
            fh.write(b"\0")
        paths.append(p)
    return tmp, paths


def similarity_from_groups(groups):
    """Stub: clips in the same group are similar, across groups they are not.

    Mirrors the real failure - a dataset whose clips are mostly one speaker
    with a misdiarized minority.
    """
    def fn(pairs, timeout=600):
        out = []
        for a, b in pairs:
            ga = next(g for g, members in groups.items() if a in members)
            gb = next(g for g, members in groups.items() if b in members)
            out.append(0.85 if ga == gb else 0.05)
        return out
    return fn


class MedoidSelectionTest(unittest.TestCase):

    def test_it_picks_the_majority_speaker_not_the_first_clip(self):
        """THE BUG. Clip 0 belongs to the minority speaker, so anchoring to it
        would train the whole adapter on the wrong voice."""
        tmp, paths = make_clips(6)
        self.addCleanup(tmp.cleanup)
        groups = {"intruder": {paths[0]}, "narrator": set(paths[1:])}
        with patch.object(voice_reference, "_speaker_similarities",
                          similarity_from_groups(groups)):
            pick, score = select_reference_sample(paths)
        self.assertIsNotNone(pick)
        self.assertNotEqual(pick, 0, "picked the intruder clip")
        self.assertGreater(score, 0.5)

    def test_a_clean_dataset_yields_a_high_score(self):
        tmp, paths = make_clips(5)
        self.addCleanup(tmp.cleanup)
        with patch.object(voice_reference, "_speaker_similarities",
                          similarity_from_groups({"all": set(paths)})):
            pick, score = select_reference_sample(paths)
        self.assertIsNotNone(pick)
        self.assertAlmostEqual(score, 0.85, places=2)

    def test_no_model_means_no_opinion_not_a_guess(self):
        """Returning 0 on failure would be indistinguishable from a real
        decision and would hide the very defect this exists to catch."""
        tmp, paths = make_clips(5)
        self.addCleanup(tmp.cleanup)
        with patch.object(voice_reference, "_speaker_similarities",
                          lambda pairs, timeout=600: None):
            self.assertEqual(select_reference_sample(paths), (None, None))

    def test_a_truncated_result_is_refused(self):
        """Fewer scores than pairs means something went wrong mid-batch;
        scoring on a partial result would silently weight some clips more."""
        tmp, paths = make_clips(5)
        self.addCleanup(tmp.cleanup)
        with patch.object(voice_reference, "_speaker_similarities",
                          lambda pairs, timeout=600: [0.9]):
            self.assertEqual(select_reference_sample(paths), (None, None))

    def test_too_few_clips_is_refused(self):
        tmp, paths = make_clips(2)
        self.addCleanup(tmp.cleanup)
        self.assertEqual(select_reference_sample(paths), (None, None))

    def test_missing_files_are_skipped_not_compared(self):
        tmp, paths = make_clips(4)
        self.addCleanup(tmp.cleanup)
        os.unlink(paths[1])
        seen = {}

        def fn(pairs, timeout=600):
            seen["paths"] = {p for pair in pairs for p in pair}
            return [0.8] * len(pairs)
        with patch.object(voice_reference, "_speaker_similarities", fn):
            pick, _ = select_reference_sample(paths)
        self.assertIsNotNone(pick)
        self.assertNotIn(paths[1], seen["paths"])

    def test_the_returned_index_addresses_the_original_list(self):
        """The caller maps this index back onto its own sample list, so an
        index into the filtered subset would select the wrong clip."""
        tmp, paths = make_clips(5)
        self.addCleanup(tmp.cleanup)
        os.unlink(paths[0])
        groups = {"odd": {paths[1]}, "rest": set(paths[2:])}
        with patch.object(voice_reference, "_speaker_similarities",
                          similarity_from_groups(groups)):
            pick, _ = select_reference_sample(paths)
        self.assertIn(pick, (2, 3, 4),
                      "index does not address the original list")

    def test_the_comparison_count_is_bounded(self):
        """This runs inside a save request and the pair count is quadratic."""
        tmp, paths = make_clips(40)
        self.addCleanup(tmp.cleanup)
        seen = {}

        def fn(pairs, timeout=600):
            seen["n"] = len(pairs)
            return [0.8] * len(pairs)
        with patch.object(voice_reference, "_speaker_similarities", fn):
            select_reference_sample(paths)
        cap = voice_reference.MAX_CLIPS
        self.assertLessEqual(seen["n"], cap * (cap - 1) // 2)


if __name__ == "__main__":
    unittest.main()


class WeakMedoidTest(unittest.TestCase):
    """The best of a bad lot is not a recommendation.

    Retraining ten adapters with an explicit medoid recovered nine. The tenth,
    `breathy_baritone_30s_m_fantasy`, went 0.705 -> 0.597 - and its medoid
    scored 0.49, the lowest in the batch. On a dataset where even the most
    representative clip is mediocre, replacing a reference that happened to be
    good makes things worse.
    """

    def test_a_weak_medoid_is_reported_but_not_recommended(self):
        tmp, paths = make_clips(5)
        self.addCleanup(tmp.cleanup)
        with patch.object(voice_reference, "_speaker_similarities",
                          lambda pairs, timeout=600: [0.30] * len(pairs)):
            pick, score = select_reference_sample(paths)
        self.assertIsNone(pick, "recommended the best of a bad lot")
        self.assertAlmostEqual(score, 0.30, places=2,
                               msg="the score must still be reported so the "
                                   "caller can log why it declined")

    def test_a_strong_medoid_is_still_recommended(self):
        tmp, paths = make_clips(5)
        self.addCleanup(tmp.cleanup)
        with patch.object(voice_reference, "_speaker_similarities",
                          lambda pairs, timeout=600: [0.85] * len(pairs)):
            pick, score = select_reference_sample(paths)
        self.assertIsNotNone(pick)
        self.assertGreaterEqual(score, voice_reference.MIN_USABLE_SIMILARITY)

    def test_the_threshold_sits_between_the_measured_cases(self):
        """0.49 regressed an adapter; 0.71-0.85 recovered nine of them."""
        self.assertGreater(voice_reference.MIN_USABLE_SIMILARITY, 0.49)
        self.assertLess(voice_reference.MIN_USABLE_SIMILARITY, 0.71)
