"""Tests for the Chinese attribution harness's label mapping.

The first run of this harness reported 0 correct and 150 unparsed on every arm
of both datasets, and was logged OK. Nothing crashed; the artifact looked like
a completed experiment. Two defects combined to produce that:

  namespace mismatch   `speaker_ids` holds a ROLE ID ('2') while the model
                       answers with a TAG ('[C0]' -> '0'). The two were
                       compared directly, so no answer could ever be correct
                       however good the model was.
  silent tallying      a bare `except Exception` counted transport failures as
                       "unparsed", so a run against a dead server was
                       indistinguishable from a run where the model answered
                       badly.

The real instance shape, from jy_test.json:

    roleid2idx  = {'2': '[C0]', '0': '[C1]', '1': '[C2]'}
    speaker_ids = ['2']            # role id, NOT the tag
    target_idx  = 5

Note the mapping is deliberately NOT identity - role '0' maps to '[C1]'. A test
using an identity mapping would pass against the broken code, which is why the
fixtures here keep the real shuffled shape.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.chinese_attribution import build_prompt, normalise, truth_of

INST = {
    "text": ["甲说话。", "乙回答。", "丙插嘴。", "又一句。", "再一句。", "目标句。"],
    "target_idx": 5,
    "roleid2idx": {"2": "[C0]", "0": "[C1]", "1": "[C2]"},
    "speaker_ids": ["2"],
}


class TestTruthMapping(unittest.TestCase):

    def test_truth_resolves_through_the_mapping(self):
        # role '2' -> '[C0]' -> '0'. The broken version returned '2'.
        self.assertEqual(truth_of(INST), "0")

    def test_mapping_is_not_identity(self):
        # Guards the fixture itself: if this became an identity map the other
        # tests would pass against the original bug.
        inst = dict(INST, speaker_ids=["0"])
        self.assertEqual(truth_of(inst), "1")

    def test_truth_matches_a_correct_model_answer(self):
        # End to end: the gold and a correct prediction must compare equal.
        self.assertEqual(truth_of(INST), normalise("[C0]"))

    def test_truth_does_not_match_a_wrong_answer(self):
        self.assertNotEqual(truth_of(INST), normalise("[C2]"))

    def test_missing_speaker_is_none(self):
        self.assertIsNone(truth_of(dict(INST, speaker_ids=[])))
        self.assertIsNone(truth_of(dict(INST, speaker_ids=None)))

    def test_speaker_absent_from_mapping_is_none(self):
        # Dropping the row is right; inventing a tag would fabricate a label.
        self.assertIsNone(truth_of(dict(INST, speaker_ids=["99"])))


class TestPrompt(unittest.TestCase):

    def test_candidates_are_the_tags_not_the_keys(self):
        p = build_prompt(INST)
        self.assertIn("[C0]", p)
        self.assertIn("[C1]", p)
        self.assertIn("[C2]", p)

    def test_candidate_count_matches_the_roles(self):
        p = build_prompt(INST).splitlines()[0]
        self.assertEqual(p.count("[C"), 3)

    def test_target_line_is_marked_once(self):
        p = build_prompt(INST)
        self.assertEqual(p.count("WHO SPEAKS THIS"), 1)
        self.assertIn("目标句。  <<< WHO SPEAKS THIS", p)

    def test_no_roles_still_builds(self):
        p = build_prompt(dict(INST, roleid2idx={}))
        self.assertIn("[C0]", p)


class TestNormalise(unittest.TestCase):

    def test_extracts_tag_index(self):
        self.assertEqual(normalise("[C3]"), "3")
        self.assertEqual(normalise("c12"), "12")
        self.assertEqual(normalise("The answer is [C1]."), "1")

    def test_unparseable_is_none(self):
        # A Chinese name rather than a tag is the failure mode actually seen.
        self.assertIsNone(normalise("郭靖"))
        self.assertIsNone(normalise(""))
        self.assertIsNone(normalise(None))


if __name__ == "__main__":
    unittest.main()
