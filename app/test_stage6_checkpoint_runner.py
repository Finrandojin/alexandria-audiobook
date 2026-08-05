from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_stage6_listening as runner


class Stage6RunnerTests(unittest.TestCase):
    def test_gpu_wrapper_delegates_to_shared_approved_runner(self):
        with patch.object(runner, "run_gpu_job") as run:
            runner.run_gpu("stage", 30, "experiment.py", ["--flag"])
        run.assert_called_once_with(
            runner.REPO, runner.APP, runner.PYTHON, "stage", 30,
            "experiment.py", ["--flag"], "stage6_listening.log")


if __name__ == "__main__":
    unittest.main()
