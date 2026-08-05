import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import local_gpu_job


class LocalGpuJobTests(unittest.TestCase):
    def test_runner_uses_one_approved_lock_queue_and_checked_timeout(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch("local_gpu_job.subprocess.run") as run:
            app = os.path.join(temp, "app")
            os.makedirs(app)
            local_gpu_job.run_gpu_job(
                temp, app, "/python", "job", 30, "script.py", ["--x"],
                "job.log")
        self.assertEqual(
            [os.path.join(temp, "gpu_job.sh"), "job", "timeout", "30",
             "/python", "-u", "script.py", "--x"], run.call_args.args[0])
        self.assertEqual(app, run.call_args.kwargs["cwd"])
        self.assertTrue(run.call_args.kwargs["check"])
        env = run.call_args.kwargs["env"]
        self.assertEqual(os.path.expanduser("~/.alexandria_gpu.lock"),
                         env["GPU_LOCK"])
        self.assertEqual(os.path.join(temp, "ab_test_runtime", "logs",
                                      "gpu_jobq.log"), env["GPU_QLOG"])


if __name__ == "__main__":
    unittest.main()
