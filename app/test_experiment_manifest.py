"""An experiment artifact must let a later reader recompute every number.

Aggregate tables cannot support an architecture decision: they cannot
distinguish a real result from a prompt, roster, alias, indexing or scoring
difference. Raised by external review of the 2026-07-26 results, which reported
49.0% conditional selection with no per-line record behind it.
"""
import json
import os
import tempfile
import unittest

from experiments.manifest import ExperimentRecord

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "fixtures", "attribution_gold_random.json")


def _record():
    return ExperimentRecord(
        name="unit", repo=REPO, model_name="test-model",
        base_url="http://localhost:1234/v1", gold_path=GOLD,
        decoding={"temperature": 0.0, "max_tokens": 24})


class ManifestTest(unittest.TestCase):
    def test_it_pins_the_gold_fixture_by_hash(self):
        meta = _record().meta
        self.assertEqual(64, len(meta["gold_sha256"]))
        self.assertEqual(147, meta["gold_lines"])

    def test_it_records_the_code_state_including_dirtiness(self):
        # A commit alone does not identify the code if the tree was dirty.
        git = _record().meta["git"]
        self.assertIn("commit", git)
        self.assertIn("dirty", git)

    def test_summary_is_recomputable_from_the_rows(self):
        record = _record()
        record.add("open", "a", "L1", "ROXY", "ROXY", True, candidates=["ROXY"])
        record.add("open", "b", "L2", "ERIS", "ROXY", False, candidates=["ERIS"])
        record.add("open", "c", "L3", "NINA", "ROXY", False, candidates=["ROXY"])
        summary = record.summary()["open"]
        self.assertEqual(3, summary["n"])
        self.assertAlmostEqual(1 / 3, summary["accuracy"])
        # Conditional accuracy counts only lines whose answer was available.
        self.assertEqual(2, summary["available"])
        self.assertAlmostEqual(0.5, summary["conditional"])

    def test_prompts_are_hashed_not_stored(self):
        record = _record()
        record.add("open", "a", "L", "ROXY", "ROXY", True, prompt="x" * 5000)
        row = record.rows[0]
        self.assertEqual(64, len(row["prompt_sha256"]))
        self.assertEqual(5000, row["prompt_chars"])
        self.assertNotIn("x" * 100, json.dumps(row))

    def test_raw_responses_are_kept_verbatim(self):
        # The parse outcome is often the story; a summary would hide it.
        record = _record()
        record.add("open", "a", "L", "ROXY", None, False, raw="I think ROXY?")
        self.assertEqual("I think ROXY?", record.rows[0]["raw_response"])

    def test_the_written_artifact_round_trips(self):
        record = _record()
        record.add("open", "a", "L", "ROXY", "ROXY", True, candidates=["ROXY"],
                   provenance="scene")
        with tempfile.TemporaryDirectory() as tmp:
            path = record.write(os.path.join(tmp, "run.json"))
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        self.assertEqual(1, len(payload["rows"]))
        self.assertEqual("scene", payload["rows"][0]["candidate_provenance"])
        self.assertIn("elapsed_s", payload["meta"])
        self.assertIn("lmstudio", payload["meta"])

    def test_bookkeeping_never_breaks_a_run(self):
        # A server that cannot be reached must not abort the experiment.
        from experiments.manifest import lmstudio_state
        state = lmstudio_state("http://127.0.0.1:9", "nope")
        self.assertIsInstance(state, dict)


if __name__ == "__main__":
    unittest.main()
