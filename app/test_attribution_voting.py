import unittest
from unittest.mock import patch

from three_pass_generate import majority_vote


class MajorityVoteTest(unittest.TestCase):
    """Greedy decoding commits to one path and cannot recover. Measured on
    mushoku16, in 30 of 49 disagreements greedy chose a speaker that not one of
    three independent samples picked, and in one scene it scattered four lines
    addressed to Rudi across AISHA and NANAHOSHI while the vote said ROXY every
    time."""

    def test_unanimous_vote_wins(self):
        winner, confidence = majority_vote(["ROXY", "ROXY", "ROXY"])
        self.assertEqual(winner, "ROXY")
        self.assertEqual(confidence, 1.0)

    def test_majority_wins_over_minority(self):
        winner, confidence = majority_vote(["ROXY", "AISHA", "ROXY"])
        self.assertEqual(winner, "ROXY")
        self.assertAlmostEqual(confidence, 2 / 3)

    def test_three_way_split_keeps_the_first(self):
        # No majority exists. Falling back to the first sample keeps the result
        # deterministic rather than dependent on dict ordering.
        winner, confidence = majority_vote(["ROXY", "AISHA", "ERIS"])
        self.assertEqual(winner, "ROXY")
        self.assertAlmostEqual(confidence, 1 / 3)

    def test_single_sample_is_full_confidence(self):
        winner, confidence = majority_vote(["ROXY"])
        self.assertEqual(winner, "ROXY")
        self.assertEqual(confidence, 1.0)

    def test_empty_votes_are_handled(self):
        winner, confidence = majority_vote([])
        self.assertIsNone(winner)
        self.assertEqual(confidence, 0.0)

    def test_none_votes_are_ignored_when_a_real_answer_exists(self):
        winner, confidence = majority_vote([None, "ROXY", "ROXY"])
        self.assertEqual(winner, "ROXY")


class VoteSeedTest(unittest.TestCase):
    """LM Studio honours seed (verified: same seed twice is identical, a
    different seed differs, no seed varies). Fixed per-vote seeds keep voting
    reproducible, so there is no accuracy-versus-determinism tradeoff."""

    def test_seeds_are_distinct_and_stable(self):
        from three_pass_generate import vote_seeds
        self.assertEqual(vote_seeds(3), vote_seeds(3))
        self.assertEqual(len(set(vote_seeds(3))), 3)
        self.assertEqual(len(set(vote_seeds(5))), 5)

    def test_seed_reaches_the_request(self):
        from generate_script import LLMGenParams, build_extra_body
        body = build_extra_body(LLMGenParams(seed=7))
        self.assertEqual(body["seed"], 7)

    def test_no_seed_is_omitted(self):
        from generate_script import LLMGenParams, build_extra_body
        self.assertNotIn("seed", build_extra_body(LLMGenParams()))


if __name__ == "__main__":
    unittest.main()


class VotedAttributionTest(unittest.TestCase):
    """votes=1 must stay byte-identical to the greedy path so the feature is
    inert unless switched on."""

    BATCH = [{"type": "SPOKEN", "text": "Are you okay, Rudi?"},
             {"type": "SPOKEN", "text": "Sorry."}]

    def test_single_vote_delegates_to_greedy(self):
        from three_pass_generate import attribute_batch_voted
        with patch("three_pass_generate.attribute_batch") as greedy:
            greedy.return_value = [{"speaker": "ROXY"}, {"speaker": "RUDI"}]
            entries, conf = attribute_batch_voted(
                None, "m", self.BATCH, object(), roster=[], votes=1)
        self.assertEqual(greedy.call_count, 1)
        self.assertEqual([e["speaker"] for e in entries], ["ROXY", "RUDI"])
        self.assertEqual(conf, [1.0, 1.0])

    def test_majority_across_samples(self):
        from dataclasses import dataclass
        from three_pass_generate import attribute_batch_voted

        @dataclass
        class P:
            seed: int = None
            temperature: float = 0.0
            attribute_temperature: float = 0.0

        ballots = [
            [{"speaker": "ROXY"}, {"speaker": "RUDI"}],
            [{"speaker": "ROXY"}, {"speaker": "ROXY"}],
            [{"speaker": "AISHA"}, {"speaker": "RUDI"}],
        ]
        with patch("three_pass_generate.attribute_batch", side_effect=ballots) as m:
            entries, conf = attribute_batch_voted(
                None, "m", self.BATCH, P(), roster=[], votes=3)
        self.assertEqual(m.call_count, 3)
        self.assertEqual([e["speaker"] for e in entries], ["ROXY", "RUDI"])
        self.assertAlmostEqual(conf[0], 2 / 3)
        self.assertAlmostEqual(conf[1], 2 / 3)

    def test_each_sample_gets_its_own_seed(self):
        from dataclasses import dataclass
        from three_pass_generate import attribute_batch_voted, vote_seeds

        @dataclass
        class P:
            seed: int = None
            temperature: float = 0.0
            attribute_temperature: float = 0.0

        seen = []

        def record(client, model, batch, params, roster, **kw):
            seen.append(params.seed)
            return [{"speaker": "ROXY"}, {"speaker": "RUDI"}]

        with patch("three_pass_generate.attribute_batch", side_effect=record):
            attribute_batch_voted(None, "m", self.BATCH, P(), roster=[], votes=3)
        self.assertEqual(seen, vote_seeds(3))

    def test_failed_samples_are_skipped(self):
        from dataclasses import dataclass
        from three_pass_generate import attribute_batch_voted

        @dataclass
        class P:
            seed: int = None
            temperature: float = 0.0
            attribute_temperature: float = 0.0

        ballots = [None, [{"speaker": "ROXY"}, {"speaker": "RUDI"}], []]
        with patch("three_pass_generate.attribute_batch", side_effect=ballots):
            entries, conf = attribute_batch_voted(
                None, "m", self.BATCH, P(), roster=[], votes=3)
        self.assertEqual([e["speaker"] for e in entries], ["ROXY", "RUDI"])
