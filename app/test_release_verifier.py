import contextlib
import io
import json
import os
import re
import unittest
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

import ci_env
import diagnose_pr_changes
import mark_pr_ready
import verify_release


class CiEnvParityTests(unittest.TestCase):
    """The local verifier is only useful if it predicts CI, which means the set
    of libraries it hides must match the set CI actually lacks."""

    def _workflow(self):
        path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "tests.yml"
        return path.read_text(encoding="utf-8")

    def _requirements(self):
        return (Path(__file__).resolve().parent / "requirements.txt").read_text(encoding="utf-8")

    def test_blocked_modules_match_what_ci_omits(self):
        # CI pip-installs requirements.txt minus an explicit exclusion list.
        match = re.search(r"grep -vE '\^\(([^)]+)\)==' requirements\.txt", self._workflow())
        self.assertIsNotNone(
            match, "could not find CI's pip exclusion in tests.yml; update this test with it")
        excluded = set(match.group(1).split("|"))

        # Plus anything the workflow never installs because it is not declared.
        declared = {
            re.split(r"[=<>~\[]", line, 1)[0].strip().lower()
            for line in self._requirements().splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        undeclared = {m for m in ci_env.BLOCKED_MODULES if m not in declared}

        self.assertEqual(
            set(ci_env.BLOCKED_MODULES), excluded | undeclared,
            "ci_env.BLOCKED_MODULES has drifted from what CI installs")

    def test_torch_is_absent_from_requirements(self):
        # The premise of blocking torch: prod gets it from torch.js, not pip.
        # If torch is ever pinned here, CI would have it and blocking is wrong.
        declared = {
            re.split(r"[=<>~\[]", line, 1)[0].strip().lower()
            for line in self._requirements().splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        self.assertNotIn("torch", declared)

    def test_ci_runs_the_shared_release_verifier(self):
        workflow = self._workflow()
        self.assertIn(
            "python verify_release.py --json-report release-report.json", workflow)
        self.assertNotIn("python -m unittest discover", workflow)
        self.assertIn("actions/upload-artifact@v6", workflow)
        self.assertIn("Diagnose whether failure exists on the base commit", workflow)
        self.assertIn("fetch-depth: 0", workflow)

    def test_changed_file_hints_name_the_missing_generator(self):
        self.assertEqual([], diagnose_pr_changes.get_snapshot_hints({
            "app/test_example.py", "app/unit_test_inventory.json",
            "app/routers/example.py", "app/api_contract/openapi.json",
        }))
        hints = diagnose_pr_changes.get_snapshot_hints({
            "app/test_example.py", "app/routers/example.py",
        })
        self.assertEqual(2, len(hints))
        self.assertIn("update_test_inventory.py", hints[0])
        self.assertIn("update_api_contract_snapshots.py", hints[1])

    def test_mark_ready_requires_current_mergeable_head_and_successful_checks(self):
        head = "abc123"
        ready = {
            "isDraft": True, "headRefOid": head, "mergeable": "MERGEABLE",
            "mergeStateStatus": "CLEAN",
            "statusCheckRollup": [{
                "name": "test", "status": "completed", "conclusion": "success",
            }],
        }
        self.assertEqual([], mark_pr_ready.get_readiness_errors(ready, head))
        ready["headRefOid"] = "stale"
        ready["statusCheckRollup"][0]["conclusion"] = "FAILURE"
        errors = mark_pr_ready.get_readiness_errors(ready, head)
        self.assertTrue(any("does not match" in error for error in errors))
        self.assertTrue(any("not completed successfully" in error for error in errors))

    def test_mark_ready_parses_https_and_ssh_origins(self):
        expected = "on22s/alexandria-audiobook2"
        self.assertEqual(expected, mark_pr_ready.get_origin_repo(
            "https://github.com/on22s/alexandria-audiobook2.git"))
        self.assertEqual(expected, mark_pr_ready.get_origin_repo(
            "git@github.com:on22s/alexandria-audiobook2.git"))

    def test_block_ml_imports_hides_an_installed_module(self):
        # Prove the finder really blocks, using a module that IS installed here
        # (json), so the test is meaningful whether or not torch is present.
        code = (
            "import ci_env, sys\n"
            "ci_env.block_ml_imports(['json'])\n"
            "try:\n"
            "    import json\n"
            "except ImportError as e:\n"
            "    print('BLOCKED')\n"
            "else:\n"
            "    print('LEAKED')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True)
        self.assertIn("BLOCKED", result.stdout)


class ReleaseVerifierTests(unittest.TestCase):
    def test_python_compilation_includes_nonignored_untracked_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
            (repo / ".gitignore").write_text("ignored.py\nenv/\n", encoding="utf-8")
            (repo / "tracked.py").write_text("TRACKED = True\n", encoding="utf-8")
            subprocess.run(["git", "add", ".gitignore", "tracked.py"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-qm", "baseline"], cwd=repo, check=True)
            (repo / "staged.py").write_text("STAGED = True\n", encoding="utf-8")
            subprocess.run(["git", "add", "staged.py"], cwd=repo, check=True)
            (repo / "untracked.py").write_text("UNTRACKED = True\n", encoding="utf-8")
            (repo / "ignored.py").write_text("not valid python !\n", encoding="utf-8")
            (repo / "env").mkdir()
            (repo / "env" / "ignored_env.py").write_text("not valid python !\n", encoding="utf-8")

            paths = verify_release.get_python_paths(repo)

            self.assertEqual(
                [repo / "staged.py", repo / "tracked.py", repo / "untracked.py"], paths
            )
            verify_release.compile_python_files(repo)

    @unittest.skipUnless(os.name == "posix", "process-group behavior is POSIX-specific")
    def test_failed_command_terminates_grandchild_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            pid_path = Path(tmp) / "grandchild.pid"
            code = (
                "import subprocess,sys; "
                "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'], "
                "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
                f"open({str(pid_path)!r},'w').write(str(child.pid)); sys.exit(3)"
            )

            with self.assertRaisesRegex(RuntimeError, "exit status 3"):
                verify_release.run_command("failing tree", [sys.executable, "-c", code], tmp)

            grandchild_pid = int(pid_path.read_text(encoding="utf-8"))
            with self.assertRaises(ProcessLookupError):
                os.kill(grandchild_pid, 0)

    def test_keyboard_interrupt_stops_child_group_before_propagating(self):
        class InterruptingOutput:
            def __iter__(self):
                raise KeyboardInterrupt

            def close(self):
                pass

        process = unittest.mock.Mock(stdout=InterruptingOutput())
        with patch.object(verify_release.subprocess, "Popen", return_value=process), \
             patch.object(verify_release, "stop_process_group") as stop:
            with self.assertRaises(KeyboardInterrupt):
                verify_release.run_command("interrupted", ["command"], ".")

        stop.assert_called_once_with(process, interrupt=True)

    def test_api_summary_uses_suite_inventory_for_quick_and_full_results(self):
        quick = self._api_summary("quick", "skipped")
        self.assertEqual(
            quick["counts"], verify_release.validate_api_summary(quick, False)
        )
        full = self._api_summary("full", "passed")
        self.assertEqual(
            full["counts"], verify_release.validate_api_summary(full, True)
        )

    def test_api_summary_rejects_wrong_skip_identity_with_same_counts(self):
        summary = self._api_summary("quick", "passed")
        summary["tests"][0]["status"] = "skipped"
        summary["counts"].update(passed=1, skipped=1)
        with self.assertRaisesRegex(ValueError, "Unexpected quick API skips"):
            verify_release.validate_api_summary(summary, False)

    def test_api_summary_rejects_inconsistent_counts_and_duplicate_names(self):
        summary = self._api_summary("quick", "skipped")
        summary["counts"]["passed"] = 2
        with self.assertRaisesRegex(ValueError, "do not match test records"):
            verify_release.validate_api_summary(summary, False)

        summary = self._api_summary("quick", "skipped")
        summary["tests"][1]["name"] = summary["tests"][0]["name"]
        with self.assertRaisesRegex(ValueError, "non-empty and unique"):
            verify_release.validate_api_summary(summary, False)

    @staticmethod
    def _api_summary(mode, full_only_status):
        tests = [
            {"name": "always", "requires_full": False, "status": "passed"},
            {"name": "gpu", "requires_full": True, "status": full_only_status},
        ]
        if full_only_status == "skipped":
            tests[1]["reason"] = "requires --full"
        passed = sum(test["status"] == "passed" for test in tests)
        skipped = sum(test["status"] == "skipped" for test in tests)
        return {
            "schema_version": 1, "mode": mode,
            "counts": {"passed": passed, "failed": 0, "skipped": skipped, "total": 2},
            "tests": tests,
        }

    def test_unittest_summary_rejects_skips_and_missing_success(self):
        verify_release.validate_unittest_output("Ran 87 tests in 1.0s\n\nOK\n")
        with self.assertRaisesRegex(ValueError, "2 skipped"):
            verify_release.validate_unittest_output(
                "Ran 87 tests in 1.0s\n\nOK (skipped=2)\n"
            )
        with self.assertRaisesRegex(ValueError, "successful summary"):
            verify_release.validate_unittest_output("process stopped")

    def test_json_report_records_successful_gates_and_api_results(self):
        api_result = {
            "counts": {"passed": 1, "failed": 0, "skipped": 1, "total": 2},
            "skips": [{"name": "gpu", "reason": "requires --full"}],
        }
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(verify_release, "compile_python_files", return_value=None), \
             patch.object(verify_release, "run_command"), \
             patch.object(verify_release, "run_api_suite", return_value=api_result):
            report_path = Path(tmp) / "report.json"
            self.assertEqual(0, verify_release.main(["--json-report", str(report_path)]))

            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual("passed", report["status"])
        self.assertEqual("quick", report["mode"])
        self.assertEqual(
            ["compile_python", "test_inventory", "unit_tests", "api_contract", "api_tests"],
            [gate["name"] for gate in report["gates"]],
        )
        self.assertEqual(api_result, report["gates"][-1]["result"])
        self.assertTrue(all(gate["status"] == "passed" for gate in report["gates"]))

    def test_json_report_is_written_on_failure_with_secrets_redacted(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(
                 verify_release, "compile_python_files",
                 side_effect=ValueError("token=do-not-store compilation failed"),
             ):
            report_path = Path(tmp) / "report.json"
            # Suppress the simulated "RELEASE VERIFICATION FAILED" that main()
            # prints; otherwise it leaks into the suite output and reads like a
            # real failure on every run.
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(1, verify_release.main(["--json-report", str(report_path)]))

            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual("failed", report["status"])
        self.assertEqual("compile_python", report["failure"]["gate"])
        self.assertEqual("token=[REDACTED] compilation failed", report["failure"]["message"])
        self.assertNotIn("do-not-store", json.dumps(report))

    def test_json_report_marks_interrupted_gate_and_returns_130(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(verify_release, "compile_python_files", side_effect=KeyboardInterrupt):
            report_path = Path(tmp) / "report.json"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    130, verify_release.main(["--json-report", str(report_path)])
                )
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual("failed", report["status"])
        self.assertEqual("KeyboardInterrupt", report["failure"]["type"])
        self.assertEqual("failed", report["gates"][0]["status"])


if __name__ == "__main__":
    unittest.main()
