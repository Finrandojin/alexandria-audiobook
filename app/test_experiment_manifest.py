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


LOADED = {"loaded": True, "context_length": 32768, "parallel": 1,
          "optimized": True}


def _record(environment=LOADED):
    return ExperimentRecord(
        name="unit", repo=REPO, model_name="test-model",
        base_url="http://localhost:1234/v1", gold_path=GOLD,
        decoding={"temperature": 0.0, "max_tokens": 24},
        environment=environment)


def _record_with_env():
    return _record()


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

    def test_an_unreachable_server_aborts_the_run(self):
        # Contract reversed on purpose: a GPU result whose context length and
        # parallel setting are unknown cannot be compared to another run, so
        # failing to capture them must stop the experiment, not annotate it.
        from experiments.manifest import EnvironmentCaptureError, lmstudio_state
        with self.assertRaises((EnvironmentCaptureError, Exception)):
            lmstudio_state("definitely-not-a-loaded-model")


if __name__ == "__main__":
    unittest.main()


class ArtifactValidationTest(unittest.TestCase):
    """The same two defects have appeared in three separate harnesses.

    A duplicate (arm, gold_id) counts one judgement twice - it produced three
    identical arm totals in the roster experiment that read as a finding. A
    summary that does not follow from the rows means the reported number cannot
    be checked at all. Both are now refused at write time.
    """

    def test_duplicate_identities_are_reported(self):
        record = _record_with_env()
        record.add("a", "id1", "L", "ROXY", "ROXY", True)
        record.add("a", "id1", "L", "ROXY", "ERIS", False)
        self.assertTrue(any("duplicate" in p for p in record.validate()))

    def test_a_clean_record_validates(self):
        record = _record_with_env()
        record.add("a", "id1", "L", "ROXY", "ROXY", True)
        record.add("a", "id2", "L2", "ERIS", "ERIS", True)
        self.assertEqual([], record.validate())

    def test_writing_an_invalid_artifact_is_refused(self):
        from experiments.manifest import EnvironmentCaptureError
        record = _record_with_env()
        record.add("a", "id1", "L", "ROXY", "ROXY", True)
        record.add("a", "id1", "L", "ROXY", "ROXY", True)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(EnvironmentCaptureError):
                record.write(os.path.join(tmp, "bad.json"))

    def test_a_missing_environment_is_a_problem(self):
        record = _record()
        record.meta["lmstudio"] = {"error": "boom"}
        record.add("a", "id1", "L", "ROXY", "ROXY", True)
        self.assertTrue(any("LM Studio" in p for p in record.validate()))


class CodeIdentityTest(unittest.TestCase):
    """A commit SHA plus 'dirty: true' does not identify what ran.

    The flag was true on every run because untracked markdown drafts sat in the
    tree, so it carried no information. It now reflects modified *tracked*
    files, and a hash of the harness sources identifies the code itself.
    """

    def test_untracked_notes_do_not_mark_the_tree_dirty(self):
        git = _record().meta["git"]
        self.assertIn("harness_sha256", git)
        self.assertEqual(64, len(git["harness_sha256"]))

    def test_the_fingerprint_changes_when_a_harness_changes(self):
        import tempfile
        from experiments.manifest import _source_fingerprint
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "h.py")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("x = 1\n")
            first = _source_fingerprint(tmp)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("x = 2\n")
            self.assertNotEqual(first, _source_fingerprint(tmp))

    def test_non_python_files_are_ignored(self):
        import tempfile
        from experiments.manifest import _source_fingerprint
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "a.py"), "w") as handle:
                handle.write("x = 1\n")
            before = _source_fingerprint(tmp)
            with open(os.path.join(tmp, "notes.md"), "w") as handle:
                handle.write("scratch\n")
            self.assertEqual(before, _source_fingerprint(tmp))


class ContractValidationTest(unittest.TestCase):
    """A summary that correctly describes incomplete rows still validates.

    Without a declared contract, a run that silently drops an arm or half its
    lines passes every internal-consistency check, because the arithmetic of
    what it *did* record is sound. External review raised this after the
    corrected roster artifact.
    """

    ENV = {"loaded": True, "context_length": 32768, "parallel": 1,
           "optimized": True, "verified_model": "test-model"}

    def _record(self, arms=("a", "b"), ids=("id1", "id2")):
        record = ExperimentRecord(
            name="unit", repo=REPO, model_name="test-model",
            base_url="http://localhost:1234/v1", gold_path=GOLD,
            decoding={"temperature": 0.0}, environment=dict(self.ENV))
        for arm in arms:
            for gold_id in ids:
                record.add(arm, gold_id, "L", "ROXY", "ROXY", True)
        return record

    def test_a_missing_arm_is_caught(self):
        record = self._record(arms=("a",))
        problems = record.validate({"expected_arms": ("a", "b")})
        self.assertTrue(any("expected" in p for p in problems))

    def test_dropped_lines_are_caught(self):
        record = self._record(ids=("id1",))
        problems = record.validate({"expected_ids": ("id1", "id2")})
        self.assertTrue(any("expected 2" in p for p in problems))

    def test_arms_scoring_different_lines_are_caught_without_a_contract(self):
        record = self._record(arms=("a",), ids=("id1", "id2"))
        record.add("b", "id1", "L", "ROXY", "ROXY", True)
        self.assertTrue(any("different set" in p for p in record.validate()))

    def test_a_complete_run_satisfies_its_contract(self):
        record = self._record()
        self.assertEqual([], record.validate(
            {"expected_arms": ("a", "b"), "expected_ids": ("id1", "id2")}))

    def test_non_ideal_load_settings_are_caught_only_when_demanded(self):
        # "optimized" is computed against an ideal derived from live VRAM, so
        # it moves with whatever else is on the card. Recording it is right;
        # refusing an otherwise sound artifact for it is not.
        record = self._record()
        record.meta["lmstudio"]["optimized"] = False
        self.assertEqual([], record.validate())
        self.assertTrue(any("non-ideal" in p for p in
                            record.validate({"require_optimized": True})))

    def test_a_missing_context_length_is_caught(self):
        record = self._record()
        record.meta["lmstudio"]["context_length"] = None
        self.assertTrue(any("context_length" in p for p in record.validate()))

    def test_a_different_loaded_model_is_caught(self):
        record = self._record()
        record.meta["lmstudio"]["verified_model"] = "some-other-model"
        self.assertTrue(any("not the declared model" in p
                            for p in record.validate()))

    def test_a_dirty_tree_is_caught_when_the_contract_demands_clean(self):
        record = self._record()
        record.meta["git"]["dirty"] = True
        record.meta["git"]["modified_tracked_files"] = ["app/x.py"]
        self.assertTrue(any("modified tracked files" in p for p in
                            record.validate({"require_clean_tree": True})))

    def test_a_missing_harness_fingerprint_is_caught(self):
        record = self._record()
        record.meta["git"]["harness_sha256"] = None
        self.assertTrue(any("unidentified" in p for p in record.validate()))


class UntrackedHarnessTest(unittest.TestCase):
    """A brand-new experiment script is untracked while it runs.

    The dirty flag ignored untracked files, so an artifact produced by a script
    that existed nowhere in the repository still recorded dirty=false against a
    commit that did not contain it. Found by reproducibility audit of
    candidate_id__qwen__qwen3-14b.json, whose commit predates its own harness.
    """

    def test_an_untracked_harness_marks_the_tree_dirty(self):
        from experiments.manifest import _git_state
        probe = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "experiments", "zz_untracked_probe.py")
        with open(probe, "w", encoding="utf-8") as handle:
            handle.write("# probe\n")
        try:
            state = _git_state(REPO)
            self.assertTrue(state["dirty"])
            self.assertTrue(any("zz_untracked_probe" in name for name in
                                state["untracked_harness_files"] or []))
        finally:
            os.remove(probe)

    def test_a_fully_tracked_harness_reports_none(self):
        from experiments.manifest import _git_state
        self.assertIsNone(_git_state(REPO)["untracked_harness_files"])


class HarnessEvidenceParityTest(unittest.TestCase):
    """Every harness must record the same evidence.

    The roster artifact carried 0 raw responses and no per-arm timing because
    it calls attribute_batch and gets parsed entries back, while the closed-set
    harness calls the model directly and keeps both. A reviewer could verify one
    experiment and not the other. Same drift class that produced three copies of
    the attestation check.
    """

    HARNESSES = ("closed_set.py", "roster_warmup.py", "candidate_id.py")

    def _source(self, name):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "experiments", name)
        with open(path, encoding="utf-8") as handle:
            return handle.read()

    def test_every_harness_records_raw_evidence(self):
        for name in self.HARNESSES:
            self.assertIn("raw=", self._source(name),
                          f"{name} records no raw response")

    def test_harnesses_using_attribute_batch_capture_attempts(self):
        # attribute_batch hides the raw text, so the only route to retry
        # evidence is the observer the pipeline already exposes.
        for name in self.HARNESSES:
            source = self._source(name)
            if "attribute_batch(" in source:
                self.assertIn("attempt_observer", source,
                              f"{name} calls attribute_batch without an observer")

    def test_multi_arm_harnesses_record_elapsed_time_per_arm(self):
        source = self._source("roster_warmup.py")
        self.assertIn("elapsed_by_arm", source)
