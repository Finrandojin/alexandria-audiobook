import json
import os
import sys
import tempfile
import unittest
import wave
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_stage4_checkpoint as runner

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments.nonprose_replication import summarize
from experiments.nonprose_category_expansion import (
    summarize as category_summarize)


class Stage4CheckpointRunnerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=runner.REPO)
        self.wavs = []

    def tearDown(self):
        self.tmp.cleanup()

    def _wav(self, name):
        path = os.path.join(self.tmp.name, name)
        with wave.open(path, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b"\0\0" * 160)
        self.wavs.append(path)
        return os.path.relpath(path, runner.REPO)

    def _artifact(self):
        rows = []
        for label in ("nonprose", "prose"):
            rows.append({
                "adapter": "adapter", "seed": 7, "pair": 0,
                "class": label, "uid": label + "-uid",
                "source_sha256": label + "-sha", "wav": self._wav(label),
                "transcript": "heard", "words": 2, "heard_words": 2,
                "errors": 0, "substitutions": 0, "deletions": 0,
                "insertions": 0, "failed": False, "threshold": 1,
                "non_speech": False, "possible_truncation": False,
            })
        return {
            "status": "complete",
            "provenance": {
                "script": "nonprose_replication.py", "written": "now",
                "host": "test", "git": {"commit": "x",
                "harness_sha256": "0" * 64},
                "args": {"source": "source.json", "config": "config.json",
                         "adapters": ["adapter"], "seeds": [7], "limit": 1,
                         "out_dir": "audio", "out": "artifact.json"},
            },
            "selection": {"pairs": [{
                "nonprose_uid": "nonprose-uid",
                "prose_uid": "prose-uid",
                "nonprose_sha256": "nonprose-sha",
                "prose_sha256": "prose-sha",
                "nonprose_features": {}, "prose_features": {},
                "absolute_feature_gap": {},
            }]},
            "rows": rows,
            "summary": summarize(rows),
        }

    def _write(self, doc):
        path = os.path.join(self.tmp.name, "artifact.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(doc, handle)
        return path

    @patch.object(runner, "_provenance_harness_matches", return_value=True)
    def test_strict_validator_decodes_complete_exact_matrix(self, _match):
        path = self._write(self._artifact())
        result = runner.validate_stage4_artifact(path, 2)
        self.assertEqual(2, len(result["rows"]))

    @patch.object(runner, "_provenance_harness_matches", return_value=True)
    def test_validator_rejects_duplicate_matrix_cell(self, _match):
        doc = self._artifact()
        doc["rows"][1].update(doc["rows"][0])
        with self.assertRaisesRegex(runner.ArtifactValidationError,
                                    "input identity|matrix keys"):
            runner.validate_stage4_artifact(self._write(doc), 2)

    @patch.object(runner, "_provenance_harness_matches", return_value=True)
    def test_validator_recomputes_summary(self, _match):
        doc = self._artifact()
        doc["summary"][0]["failed"] = 99
        with self.assertRaisesRegex(runner.ArtifactValidationError,
                                    "summary does not exactly recompute"):
            runner.validate_stage4_artifact(self._write(doc), 2)

    @patch.object(runner, "_provenance_harness_matches", return_value=True)
    def test_validator_reads_entire_wav_and_rejects_truncation(self, _match):
        doc = self._artifact()
        wav = os.path.join(runner.REPO, doc["rows"][0]["wav"])
        with open(wav, "rb+") as handle:
            handle.truncate(50)
        with self.assertRaisesRegex(runner.ArtifactValidationError,
                                    "not fully decodable"):
            runner.validate_stage4_artifact(self._write(doc), 2)

    @patch.object(runner, "_provenance_harness_matches", return_value=True)
    def test_validator_requires_error_breakdown_identity(self, _match):
        doc = self._artifact()
        doc["rows"][0]["errors"] = 1
        with self.assertRaisesRegex(runner.ArtifactValidationError,
                                    "error breakdown is wrong"):
            runner.validate_stage4_artifact(self._write(doc), 2)

    @patch.object(runner, "_provenance_harness_matches", return_value=True)
    def test_same_validator_covers_category_expansion(self, _match):
        doc = self._artifact()
        doc["provenance"]["script"] = "nonprose_category_expansion.py"
        args = doc["provenance"]["args"]
        args["limit_per_category"] = args.pop("limit")
        doc["selection"]["categories"] = ["urls"]
        pair = doc["selection"]["pairs"][0]
        pair["probe_uid"] = pair.pop("nonprose_uid")
        pair["probe_sha256"] = pair.pop("nonprose_sha256")
        pair["probe_features"] = pair.pop("nonprose_features")
        pair["category"] = "urls"
        for row in doc["rows"]:
            row["category"] = "urls"
            if row["class"] == "nonprose":
                row["class"] = "probe"
        doc["summary"] = category_summarize(doc["rows"])
        matrix = {("adapter", 7, 0, "probe"),
                  ("adapter", 7, 0, "prose")}
        result = runner.validate_stage4_artifact(
            self._write(doc), 2, matrix,
            expected_script="nonprose_category_expansion.py",
            class_prefixes={"probe": "probe", "prose": "prose"},
            extra_arg_fields=("limit_per_category",),
            expected_categories=("urls",),
            summary_function=category_summarize)
        self.assertEqual(2, len(result["rows"]))

    @patch.object(runner, "_provenance_harness_matches", return_value=True)
    def test_category_validator_rejects_wrong_row_category(self, _match):
        doc = self._artifact()
        doc["provenance"]["script"] = "nonprose_category_expansion.py"
        args = doc["provenance"]["args"]
        args["limit_per_category"] = args.pop("limit")
        doc["selection"]["categories"] = ["urls"]
        pair = doc["selection"]["pairs"][0]
        pair["probe_uid"] = pair.pop("nonprose_uid")
        pair["probe_sha256"] = pair.pop("nonprose_sha256")
        pair["probe_features"] = pair.pop("nonprose_features")
        pair["category"] = "urls"
        for row in doc["rows"]:
            row["category"] = "wrong"
            if row["class"] == "nonprose":
                row["class"] = "probe"
        doc["summary"] = category_summarize(doc["rows"])
        with self.assertRaisesRegex(runner.ArtifactValidationError,
                                    "category does not match"):
            runner.validate_stage4_artifact(
                self._write(doc), 2,
                expected_script="nonprose_category_expansion.py",
                class_prefixes={"probe": "probe", "prose": "prose"},
                extra_arg_fields=("limit_per_category",),
                expected_categories=("urls",),
                summary_function=category_summarize)


if __name__ == "__main__":
    unittest.main()
