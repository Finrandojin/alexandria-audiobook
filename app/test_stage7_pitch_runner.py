import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_stage7_pitch as runner


class Stage7PitchRunnerTests(unittest.TestCase):
    def test_pilot_gate_requires_three_real_measurements(self):
        good = {"rows": [{"pitch_status": "measured"}] * 3
                         + [{"pitch_status": "tracker_failure"}]}
        runner.require_pilot_gate(good)
        with self.assertRaisesRegex(RuntimeError, "only 2/4"):
            runner.require_pilot_gate({
                "rows": [{"pitch_status": "measured"}] * 2
                        + [{"pitch_status": "tracker_failure"}] * 2})

    def test_full_job_is_all_adapters_seeds_and_passages(self):
        with patch.object(runner.os.path, "exists", return_value=False), \
                patch.object(runner, "run_gpu_job") as run_gpu, \
                patch.object(runner, "validate_artifact") as validate:
            runner.ensure_full(75)
        arguments = run_gpu.call_args.args[6]
        self.assertEqual(["--out-dir", runner.FULL_DIR, "--out", runner.FULL],
                         arguments)
        validate.assert_called_once_with(runner.FULL, expected_count=1350)


if __name__ == "__main__":
    unittest.main()
