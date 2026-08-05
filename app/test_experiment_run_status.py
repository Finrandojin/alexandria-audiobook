import os
import tempfile
import unittest

from experiments.run_status import (finish_experiment_status,
    load_experiment_status, record_experiment_stage, start_experiment_status)


class ExperimentRunStatusTests(unittest.TestCase):
    def test_status_records_checkpoint_identity_and_resume_decision(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "status.json")
            checkpoint = os.path.join(root, "checkpoint.json")
            with open(checkpoint, "w", encoding="utf-8") as handle:
                handle.write("{}")
            start_experiment_status(path, "test", __file__)
            record_experiment_stage(path, "render", checkpoint, True)
            finish_experiment_status(path, "human_pending", "Collect ratings.")
            status = load_experiment_status(path)
        self.assertEqual("human_pending", status["status"])
        self.assertEqual("validated_existing", status["stages"][0]["resume"])
        self.assertEqual(64, len(status["stages"][0]["checkpoint_sha256"]))
        self.assertEqual("Collect ratings.", status["next_action"])

    def test_existing_or_malformed_status_fails_loudly(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "status.json")
            start_experiment_status(path, "test", __file__)
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                start_experiment_status(path, "test", __file__)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("{}")
            with self.assertRaisesRegex(RuntimeError, "invalid"):
                load_experiment_status(path)


if __name__ == "__main__":
    unittest.main()
