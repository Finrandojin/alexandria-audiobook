import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from experiments.manifest import _source_fingerprint
from experiments.provenance import (
    get_harness_sha256_at_commit, get_reproducible_harness_source)


class ReproducibleHarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name)
        self.harness = self.repo / "app" / "experiments"
        self.harness.mkdir(parents=True)
        subprocess.run(["git", "init", "-q", "-b", "main"], cwd=self.repo,
                       check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"],
                       cwd=self.repo, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=self.repo,
                       check=True)

    def tearDown(self):
        self.temp.cleanup()

    def commit(self, text):
        (self.harness / "probe.py").write_text(text, encoding="utf-8")
        subprocess.run(["git", "add", "app/experiments/probe.py"],
                       cwd=self.repo, check=True)
        subprocess.run(["git", "commit", "-q", "-m", text], cwd=self.repo,
                       check=True)
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=self.repo, text=True).strip()

    def block(self, commit, fingerprint, dirty):
        return {"git": {"commit": commit, "harness_sha256": fingerprint,
                        "dirty": dirty}}

    def test_clean_run_must_match_its_recorded_commit(self):
        commit = self.commit("one\n")
        fingerprint = get_harness_sha256_at_commit(str(self.repo), commit)
        self.assertEqual(commit, get_reproducible_harness_source(
            self.block(commit, fingerprint, False), str(self.repo)))
        self.commit("two\n")
        current = _source_fingerprint(str(self.harness))
        self.assertIsNone(get_reproducible_harness_source(
            self.block(commit, current, False), str(self.repo)))

    def test_dirty_run_can_match_exact_descendant_commit(self):
        recorded = self.commit("one\n")
        captured = self.commit("two\n")
        fingerprint = get_harness_sha256_at_commit(str(self.repo), captured)
        self.assertEqual(captured, get_reproducible_harness_source(
            self.block(recorded, fingerprint, True), str(self.repo)))

    def test_dirty_uncommitted_harness_can_match_worktree(self):
        recorded = self.commit("one\n")
        (self.harness / "probe.py").write_text("dirty\n", encoding="utf-8")
        fingerprint = _source_fingerprint(str(self.harness))
        self.assertEqual("WORKTREE", get_reproducible_harness_source(
            self.block(recorded, fingerprint, True), str(self.repo)))

    def test_unrelated_branch_does_not_rescue_recorded_run(self):
        recorded = self.commit("one\n")
        subprocess.run(["git", "checkout", "-q", "-b", "sibling"],
                       cwd=self.repo, check=True)
        sibling = self.commit("sibling\n")
        fingerprint = get_harness_sha256_at_commit(str(self.repo), sibling)
        subprocess.run(["git", "checkout", "-q", "main"], cwd=self.repo,
                       check=True)
        self.assertIsNone(get_reproducible_harness_source(
            self.block(recorded, fingerprint, True), str(self.repo)))

    def test_malformed_identity_is_rejected(self):
        commit = self.commit("one\n")
        self.assertIsNone(get_reproducible_harness_source(
            self.block(commit, "not-a-sha", True), str(self.repo)))


if __name__ == "__main__":
    unittest.main()
