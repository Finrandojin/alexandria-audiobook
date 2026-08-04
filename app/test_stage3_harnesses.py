import os
import subprocess
import sys
import tempfile
import unittest


APP = os.path.dirname(os.path.abspath(__file__))


class TestStage3HarnessesFailLoudly(unittest.TestCase):
    def run_harness(self, name, *args):
        return subprocess.run(
            [sys.executable, os.path.join(APP, "experiments", name), *args],
            cwd=APP, text=True, capture_output=True, timeout=20)

    def test_clone_phase_is_required(self):
        result = self.run_harness("clone_vs_lora.py")
        self.assertNotEqual(0, result.returncode)
        self.assertIn("--phase", result.stderr)

    def test_saturation_phase_is_required(self):
        result = self.run_harness("voice_data_saturation.py")
        self.assertNotEqual(0, result.returncode)
        self.assertIn("--phase", result.stderr)

    def test_clone_score_refuses_missing_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            result = self.run_harness(
                "clone_vs_lora.py", "--phase", "score", "--json",
                os.path.join(td, "missing.json"))
        self.assertNotEqual(0, result.returncode)
        self.assertIn("generation manifest does not exist", result.stderr)

    def test_saturation_score_refuses_missing_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            result = self.run_harness(
                "voice_data_saturation.py", "--phase", "score", "--json",
                os.path.join(td, "missing.json"))
        self.assertNotEqual(0, result.returncode)
        self.assertIn("generation manifest does not exist", result.stderr)


if __name__ == "__main__":
    unittest.main()
