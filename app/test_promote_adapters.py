import os
import sys
import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import promote_adapters


class AdapterSourceTests(unittest.TestCase):
    def test_gate_path_wins_over_same_named_legacy_source(self):
        with tempfile.TemporaryDirectory() as root:
            legacy_root = Path(root, "legacy")
            decontam_root = Path(root, "decontaminate")
            Path(legacy_root, "voice", "adapter").mkdir(parents=True)
            gated = Path(decontam_root, "batch1", "voice", "adapter")
            gated.mkdir(parents=True)
            gates = Path(root, "gates")
            gates.mkdir()
            Path(gates, "gate_promote__voice.json").write_text(json.dumps({
                "adapter": str(gated), "median_ecapa": 0.7
            }), encoding="utf-8")
            with patch.object(promote_adapters, "GATES", str(gates)), \
                 patch.object(promote_adapters, "SOURCE", str(legacy_root)), \
                 patch.object(promote_adapters, "DECONTAMINATE_SOURCE",
                              str(decontam_root)):
                self.assertEqual(str(gated),
                                 promote_adapters.get_adapter_source("voice"))

    def test_resolves_one_decontamination_batch(self):
        with tempfile.TemporaryDirectory() as root:
            adapter = Path(root, "batch4", "voice", "adapter")
            adapter.mkdir(parents=True)
            with patch.object(promote_adapters, "SOURCE", os.path.join(root, "legacy")), \
                 patch.object(promote_adapters, "DECONTAMINATE_SOURCE", root):
                self.assertEqual(str(adapter),
                                 promote_adapters.get_adapter_source("voice"))

    def test_refuses_ambiguous_decontamination_sources(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root, "batch1", "voice", "adapter").mkdir(parents=True)
            Path(root, "batch2", "voice", "adapter").mkdir(parents=True)
            with patch.object(promote_adapters, "SOURCE", os.path.join(root, "legacy")), \
                 patch.object(promote_adapters, "DECONTAMINATE_SOURCE", root):
                self.assertIsNone(promote_adapters.get_adapter_source("voice"))

    def test_installed_gate_score_overrides_stale_baseline(self):
        with tempfile.TemporaryDirectory() as root:
            gates = Path(root, "gates")
            models = Path(root, "models")
            gates.mkdir()
            models.mkdir()
            Path(gates, "library_voice_fidelity_n10.json").write_text(
                json.dumps({"results": [{"adapter": "voice", "ecapa": 0.4}]}),
                encoding="utf-8")
            Path(models, "manifest.json").write_text(json.dumps([
                {"id": "voice", "gate_ecapa": 0.7}
            ]), encoding="utf-8")
            with patch.object(promote_adapters, "GATES", str(gates)), \
                 patch.object(promote_adapters, "MODELS", str(models)):
                self.assertEqual(0.7, promote_adapters.shipped_scores()["voice"])

    def test_manifest_maps_training_num_samples_to_sample_count(self):
        with tempfile.TemporaryDirectory() as root:
            models = Path(root, "models")
            source = Path(root, "source")
            models.mkdir()
            source.mkdir()
            Path(models, "manifest.json").write_text(
                json.dumps([{"id": "voice", "sample_count": 200}]),
                encoding="utf-8")
            Path(source, "training_meta.json").write_text(
                json.dumps({"num_samples": 180}), encoding="utf-8")
            with patch.object(promote_adapters, "MODELS", str(models)), \
                 patch.object(promote_adapters, "get_adapter_source",
                              return_value=str(source)):
                promote_adapters.update_manifest({"voice": 0.7}, "stamp")
            manifest = json.loads(Path(models, "manifest.json").read_text())
            self.assertEqual(180, manifest[0]["sample_count"])


if __name__ == "__main__":
    unittest.main()
