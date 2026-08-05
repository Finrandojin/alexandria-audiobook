import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_stage6_listening as runner


class Stage6RunnerTests(unittest.TestCase):
    def test_gpu_wrapper_sets_approved_lock_and_repository_queue_log(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(runner, "REPO", temp), \
                    patch.object(runner, "APP", temp), \
                    patch.object(runner, "GPU_JOB", "/gpu_job.sh"), \
                    patch.object(runner, "PYTHON", "/python"), \
                    patch("run_stage6_listening.subprocess.run") as run:
                runner.run_gpu("stage", 30, "experiment.py", ["--flag"])
        command = run.call_args.args[0]
        env = run.call_args.kwargs["env"]
        self.assertEqual(
            ["/gpu_job.sh", "stage", "timeout", "30", "/python", "-u",
             "experiment.py", "--flag"], command)
        self.assertEqual(os.path.expanduser("~/.alexandria_gpu.lock"),
                         env["GPU_LOCK"])
        self.assertEqual(os.path.join(temp, "ab_test_runtime", "logs",
                                      "gpu_jobq.log"), env["GPU_QLOG"])
        self.assertTrue(run.call_args.kwargs["check"])


if __name__ == "__main__":
    unittest.main()
