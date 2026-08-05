"""Tests for the experiment infrastructure whose failures are SILENT.

Most bugs in this directory announce themselves - a harness crashes, a script
raises. These four do not. They produce plausible numbers that are wrong, or
they quietly stop protecting the ledger, and nobody notices until a result is
already believed.

Each test here corresponds to a defect that actually occurred:

  gold-follows-book   Eleven harnesses let EXPERIMENT_BOOK be set while GOLD
                      hardcoded grimgar03's fixture, so switching books scored
                      one book's lines against another's gold: 3 matched lines
                      of 162, 0.0% on every arm. It cost two runs.
  scoring             `same()` was copy-pasted into eighteen harnesses and all
                      copies shared a punctuation bug that turned 162 correct
                      rows across the ledger into errors.
  manifest guards     ExperimentRecord.validate refused two bad artifacts in a
                      single session - 238 duplicate identities, and an arms
                      contract mismatch. Nothing tested that it still fires.
  analysis robustness Two analyses broke on real data: a Counter KeyError when
                      a category was empty, and collect_results dying on an
                      artifact that used "rows" for a count.

Imports are lazy where a module needs openai, because update_test_inventory
imports every test module in an environment that has neither openai nor pytest.
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

APP = os.path.dirname(os.path.abspath(__file__))
if APP not in sys.path:
    sys.path.insert(0, APP)

EXPERIMENTS = os.path.join(APP, "experiments")


class GoldFollowsBookTest(unittest.TestCase):
    """A harness must never resolve gold independently of the book."""

    def test_no_harness_hardcodes_a_book_specific_gold_default(self):
        offenders = []
        for path in sorted(glob.glob(os.path.join(EXPERIMENTS, "*.py"))):
            source = open(path, encoding="utf-8").read()
            if "EXPERIMENT_GOLD" not in source:
                continue
            # The default must mention BOOK. A literal book name in the default
            # is the exact defect: BOOK moves, GOLD does not, and the run scores
            # one book against another's answers.
            match = re.search(
                r'EXPERIMENT_GOLD"\s*,\s*\n?\s*(f?)"([^"]+)"', source)
            if not match:
                continue
            is_fstring, default = match.groups()
            names = ("grimgar03", "index18", "mushoku16", "owarimonogatari3",
                     "grimgar06", "mushoku18")
            if any(n in default for n in names) and not (
                    is_fstring and "{BOOK}" in default):
                offenders.append(f"{os.path.basename(path)}: {default}")
        self.assertEqual(
            offenders, [],
            "these harnesses hardcode a book into the gold default, so setting "
            "EXPERIMENT_BOOK alone scores the wrong fixture:\n  "
            + "\n  ".join(offenders))


class ScoringTest(unittest.TestCase):
    """The single definition of 'right speaker' used by every harness."""

    def setUp(self):
        from experiments import scoring
        self.scoring = scoring

    def test_punctuation_does_not_make_a_correct_answer_wrong(self):
        """The bug that turned 162 correct rows across the ledger into errors."""
        self.assertTrue(self.scoring.same_speaker("MR. PRIEST", "MR PRIEST"))
        self.assertTrue(self.scoring.same_speaker("MS. SHORT HAIR",
                                                  "MS SHORT HAIR"))

    def test_distinct_characters_are_not_merged(self):
        """Normalisation must not be so aggressive it fuses the cast."""
        self.assertFalse(self.scoring.same_speaker("HARUHIRO", "RANTA"))
        self.assertFalse(self.scoring.same_speaker("MR. TALL", "MR. SHORT"))
        self.assertFalse(self.scoring.same_speaker("YUME", "YUMEKO"))

    def test_empty_prediction_is_never_correct(self):
        """An unanswered row must score wrong, not vacuously right - dropping
        these is what made one arm read eleven points high."""
        self.assertFalse(self.scoring.same_speaker("HARUHIRO", ""))
        self.assertFalse(self.scoring.same_speaker("HARUHIRO", None))

    def test_aliases_apply_in_both_directions(self):
        groups = self.scoring.alias_groups(
            {"aliases": [["KUZAKU", "KUZAK"], ["MERRY", "MERIYA"]]})
        self.assertTrue(self.scoring.same_speaker("KUZAKU", "KUZAK", groups))
        self.assertTrue(self.scoring.same_speaker("KUZAK", "KUZAKU", groups))
        self.assertFalse(self.scoring.same_speaker("KUZAKU", "MERRY", groups))

    def test_single_member_alias_groups_are_ignored(self):
        """A one-name group can only create false matches."""
        groups = self.scoring.alias_groups({"aliases": [["HARUHIRO"]]})
        self.assertEqual(groups, [])

    def test_romanisation_is_off_by_default(self):
        """Kept out of the headline deliberately: exact spelling is what makes
        a name usable downstream for voice assignment."""
        self.assertFalse(self.scoring.same_speaker("RUDEUS", "RUDIUS"))

    def test_alias_groups_accepts_a_fixture_without_aliases(self):
        self.assertEqual(self.scoring.alias_groups({}), [])
        self.assertEqual(self.scoring.alias_groups(None), [])


class ManifestGuardTest(unittest.TestCase):
    """The guards that refused two bad artifacts in one session."""

    def _record(self, name="guard_test"):
        from experiments.manifest import ExperimentRecord
        gold = os.path.join(APP, "fixtures", "attribution_gold_grimgar03.json")
        return ExperimentRecord(
            name, os.path.dirname(APP), "test-model", "http://localhost/v1",
            gold, {"temperature": 0.0},
            environment={"loaded": True, "context_length": 32768,
                         "parallel": 1, "optimized": None},
            notes="unit test")

    def test_duplicate_identities_are_reported(self):
        """238 duplicates is how the batch_contiguity bug surfaced."""
        record = self._record()
        for _ in range(2):
            record.add("armA", "id-1", "line", "HARUHIRO", "HARUHIRO", True)
        problems = record.validate()
        self.assertTrue(any("duplicate" in p for p in problems), problems)

    def test_arms_contract_mismatch_is_reported(self):
        """An arm silently added or dropped still summarises correctly."""
        record = self._record()
        record.add("armA", "id-1", "line", "HARUHIRO", "HARUHIRO", True)
        problems = record.validate(contract={"expected_arms": ("armA", "armB")})
        self.assertTrue(any("arms" in p for p in problems), problems)

    def test_missing_environment_is_reported(self):
        from experiments.manifest import ExperimentRecord
        gold = os.path.join(APP, "fixtures", "attribution_gold_grimgar03.json")
        record = ExperimentRecord(
            "guard_test", os.path.dirname(APP), "m", "http://localhost/v1",
            gold, {}, environment={}, notes="unit test")
        record.add("armA", "id-1", "line", "HARUHIRO", "HARUHIRO", True)
        problems = record.validate()
        self.assertTrue(any("context_length" in p or "LM Studio" in p
                            for p in problems), problems)

    def test_write_refuses_an_invalid_artifact(self):
        """No bad number reaches the ledger: the failure must be a refusal to
        write, not a warning beside a file that then gets read."""
        from experiments.manifest import EnvironmentCaptureError
        record = self._record()
        for _ in range(2):
            record.add("armA", "id-1", "line", "HARUHIRO", "HARUHIRO", True)
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "artifact.json")
            with self.assertRaises(EnvironmentCaptureError):
                record.write(target)
            self.assertFalse(os.path.exists(target),
                             "a rejected artifact must not be left on disk")

    def test_a_clean_record_writes(self):
        """The guard must not be so strict that valid runs cannot record."""
        record = self._record()
        record.add("armA", "id-1", "line", "HARUHIRO", "HARUHIRO", True)
        record.add("armB", "id-1", "line", "HARUHIRO", "RANTA", False)
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "artifact.json")
            record.write(target, contract={"expected_arms": ("armA", "armB")})
            written = json.load(open(target))
        self.assertEqual(written["meta"]["validation"], "ok")
        self.assertEqual(len(written["rows"]), 2)


class CollectResultsRobustnessTest(unittest.TestCase):
    """The index generator must survive an artifact it does not understand."""

    def test_rows_as_a_count_does_not_crash_the_index(self):
        """segmentation_classifier.json used 'rows' for a count, and iterating
        an int killed the whole index rather than skipping one file."""
        source = open(os.path.join(os.path.dirname(APP), "collect_results.py"),
                      encoding="utf-8").read()
        self.assertIn("isinstance(rr, list)", source,
                      "collect_results must check the shape of 'rows' before "
                      "iterating it")

    def test_csv_uses_repository_lf_line_endings(self):
        """CRLF made every newly indexed row fail git diff --check."""
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy2(os.path.join(os.path.dirname(APP), "collect_results.py"),
                         os.path.join(tmp, "collect_results.py"))
            audit = os.path.join(tmp, "ab_test_runtime", "audit")
            os.makedirs(audit)
            for name in ("artifact_structural_audit.json",
                         "legacy_attribution_audit.json"):
                with open(os.path.join(audit, name), "w", encoding="utf-8") as handle:
                    json.dump({"artifacts": []}, handle)
            experiments = os.path.join(tmp, "ab_test_runtime", "experiments")
            os.makedirs(experiments)
            with open(os.path.join(experiments, "probe.json"), "w",
                      encoding="utf-8") as handle:
                json.dump({"status": "complete"}, handle)
            subprocess.run(
                [sys.executable, "collect_results.py"], cwd=tmp,
                capture_output=True, check=True)
            content = open(os.path.join(tmp, "results_index.csv"), "rb").read()
        self.assertIn(b"\n", content)
        self.assertNotIn(b"\r\n", content)

    def test_results_index_check_fails_when_audit_status_changes(self):
        """CI must reject an index whose evidence label no longer matches its audit."""
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy2(os.path.join(os.path.dirname(APP), "collect_results.py"),
                         os.path.join(tmp, "collect_results.py"))
            experiments = os.path.join(tmp, "ab_test_runtime", "experiments")
            audit = os.path.join(tmp, "ab_test_runtime", "audit")
            os.makedirs(experiments)
            os.makedirs(audit)
            artifact = "probe.json"
            with open(os.path.join(experiments, artifact), "w", encoding="utf-8") as handle:
                json.dump({"status": "complete"}, handle)
            with open(os.path.join(audit, "artifact_structural_audit.json"),
                      "w", encoding="utf-8") as handle:
                json.dump({"artifacts": [{"artifact": artifact,
                                           "classification": "exploratory"}]}, handle)
            with open(os.path.join(audit, "legacy_attribution_audit.json"),
                      "w", encoding="utf-8") as handle:
                json.dump({"artifacts": []}, handle)
            subprocess.run([sys.executable, "collect_results.py"], cwd=tmp,
                           capture_output=True, check=True)
            structural = os.path.join(audit, "artifact_structural_audit.json")
            with open(structural, "w", encoding="utf-8") as handle:
                json.dump({"artifacts": [{"artifact": artifact,
                                           "classification": "supported_structure"}]}, handle)
            checked = subprocess.run([sys.executable, "collect_results.py", "--check"],
                                     cwd=tmp, capture_output=True, text=True)
        self.assertNotEqual(0, checked.returncode)
        self.assertIn("results index is stale", checked.stderr)

    def test_results_index_is_independent_of_local_timezone(self):
        """The checked-in index must render identically on local and CI hosts."""
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy2(os.path.join(os.path.dirname(APP), "collect_results.py"),
                         os.path.join(tmp, "collect_results.py"))
            experiments = os.path.join(tmp, "ab_test_runtime", "experiments")
            audit = os.path.join(tmp, "ab_test_runtime", "audit")
            os.makedirs(experiments)
            os.makedirs(audit)
            artifact = "probe.json"
            with open(os.path.join(experiments, artifact), "w", encoding="utf-8") as handle:
                json.dump({"meta": {"experiment": "probe", "finished": 1,
                                     "git": {}, "validation": "ok"},
                           "rows": [{"arm": "base", "correct": True}]}, handle)
            for name in ("artifact_structural_audit.json",
                         "legacy_attribution_audit.json"):
                with open(os.path.join(audit, name), "w", encoding="utf-8") as handle:
                    json.dump({"artifacts": []}, handle)
            env = dict(os.environ, TZ="America/Chicago")
            subprocess.run([sys.executable, "collect_results.py"], cwd=tmp,
                           env=env, capture_output=True, check=True)
            env["TZ"] = "UTC"
            checked = subprocess.run([sys.executable, "collect_results.py", "--check"],
                                     cwd=tmp, env=env, capture_output=True, text=True)
        self.assertEqual(0, checked.returncode, checked.stderr)

    def test_tts_provenance_rows_are_not_misindexed_as_attribution(self):
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy2(os.path.join(os.path.dirname(APP), "collect_results.py"),
                         os.path.join(tmp, "collect_results.py"))
            experiments = os.path.join(tmp, "ab_test_runtime", "experiments")
            audit = os.path.join(tmp, "ab_test_runtime", "audit")
            os.makedirs(experiments)
            os.makedirs(audit)
            with open(os.path.join(experiments, "tts.json"), "w", encoding="utf-8") as handle:
                json.dump({"provenance": {"git": {}},
                           "rows": [{"arm": "raw", "correct": True}]}, handle)
            for name in ("artifact_structural_audit.json",
                         "legacy_attribution_audit.json"):
                with open(os.path.join(audit, name), "w", encoding="utf-8") as handle:
                    json.dump({"artifacts": []}, handle)
            subprocess.run([sys.executable, "collect_results.py"], cwd=tmp,
                           capture_output=True, check=True)
            content = open(os.path.join(tmp, "results_index.csv"),
                           encoding="utf-8").read()
        self.assertIn("NOT INDEXED: TTS provenance artifact", content)
        self.assertNotIn(",raw,", content)


class AnalysisScriptTest(unittest.TestCase):
    """The offline analyses must at least be importable and syntactically sound.

    They are re-runnable and fail loudly, so this is deliberately a light gate -
    it catches a broken edit, not a wrong conclusion.
    """

    def test_every_experiment_module_parses(self):
        import ast
        broken = []
        for path in sorted(glob.glob(os.path.join(EXPERIMENTS, "*.py"))):
            try:
                ast.parse(open(path, encoding="utf-8").read())
            except SyntaxError as exc:
                broken.append(f"{os.path.basename(path)}: {exc}")
        self.assertEqual(broken, [], "\n".join(broken))

    def test_offline_analyses_declare_their_caveats(self):
        """Every analysis that reports an ORACLE or fitted number must say so
        in its docstring. Reporting a fitted number as achievable is the
        closed-oracle mistake this ledger already had to retract."""
        import ast
        missing = []
        for name in ("cluster_vs_name.py", "realizable_router.py",
                     "adapter_vs_cascade_overlap.py"):
            path = os.path.join(EXPERIMENTS, name)
            if not os.path.exists(path):
                continue
            doc = ast.get_docstring(ast.parse(open(path, encoding="utf-8").read())) or ""
            if "oracle" not in doc.lower() and "upper bound" not in doc.lower():
                missing.append(name)
        self.assertEqual(missing, [],
                         "these report a fitted/oracle quantity without saying "
                         "so in the docstring: " + ", ".join(missing))


if __name__ == "__main__":
    unittest.main()
