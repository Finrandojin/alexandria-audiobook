import math
import os
import sys
import tempfile
import unittest
import wave
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments import pitch_profile_matrix as pitch


class PitchProfileMatrixTests(unittest.TestCase):
    def test_locked_fixture_matches_six_distinct_source_passages(self):
        passages, rule, chunks = pitch.load_passages(pitch.DEFAULT_SOURCE)
        self.assertEqual(6, len(passages))
        self.assertEqual(6, len({row["category"] for row in passages}))
        self.assertTrue(rule)
        self.assertEqual(os.path.join(pitch.REPO, "chunks.json"), chunks)
        for row in passages:
            self.assertEqual(64, len(row["source_sha256"]))

    def test_default_manifest_has_75_complete_adapters(self):
        manifest = os.path.join(pitch.REPO, "lora_models", "manifest.json")
        adapters = pitch.load_adapters(manifest)
        self.assertEqual(75, len(adapters))
        self.assertEqual(75, len({row["adapter"] for row in adapters}))
        self.assertTrue(all(row["declared_mean_f0"] > 0 for row in adapters))

    def test_measure_pitch_tracks_a_decodable_sine_wave(self):
        with tempfile.TemporaryDirectory() as temp:
            path = os.path.join(temp, "tone.wav")
            rate = 16000
            samples = (0.2 * np.sin(
                2 * math.pi * 120 * np.arange(rate * 2) / rate))
            import soundfile as sf
            sf.write(path, samples, rate)
            result = pitch.measure_pitch(path)
        self.assertEqual("measured", result["pitch_status"])
        self.assertAlmostEqual(120, result["median_pitch_hz"], delta=2)
        self.assertGreater(result["voiced_coverage"], 0.8)
        self.assertGreaterEqual(result["voiced_frames"], pitch.MIN_VOICED_FRAMES)

    def test_measure_pitch_surfaces_silence_as_tracker_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            path = os.path.join(temp, "silence.wav")
            with wave.open(path, "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(b"\0\0" * 32000)
            result = pitch.measure_pitch(path)
        self.assertEqual("tracker_failure", result["pitch_status"])
        self.assertIn("voiced frames", result["pitch_error"])

    def test_octave_flags_are_explicit_leads_not_ground_truth(self):
        rows = [
            {"adapter": "a", "median_pitch_hz": 120.0,
             "declared_mean_f0": 120.0, "pitch_status": "measured",
             "frame_octave_fraction": 0.0},
            {"adapter": "a", "median_pitch_hz": 240.0,
             "declared_mean_f0": 120.0, "pitch_status": "measured",
             "frame_octave_fraction": 0.0},
        ]
        flagged = pitch.add_octave_flags(rows)
        self.assertFalse(flagged[0]["likely_octave_error"])
        self.assertTrue(flagged[1]["likely_octave_error"])
        self.assertIn("declared_mean", flagged[1]["likely_octave_reasons"])

    def test_summary_reports_dispersion_and_threshold_disagreement(self):
        adapters = [
            {"adapter": "low", "declared_mean_f0": 180.0},
            {"adapter": "high", "declared_mean_f0": 150.0},
        ]
        rows = []
        for adapter, values in (("low", (100.0, 110.0)),
                                ("high", (200.0, 210.0))):
            for value in values:
                rows.append({
                    "adapter": adapter, "pitch_status": "measured",
                    "median_pitch_hz": value, "pitch_iqr_hz": 5.0,
                    "voiced_coverage": 0.9, "likely_octave_error": False,
                })
        adapter_summary, summary = pitch.summarize(rows, adapters)
        self.assertEqual(2, len(adapter_summary))
        self.assertEqual(1, summary["voice_pairs"])
        self.assertEqual(1, summary["pairs_beyond_typical_dispersion"])
        self.assertEqual(2, summary[
            "declared_vs_measured_165hz_side_disagreements"])

    def test_row_validation_rejects_duplicates_and_recomputes_measurement(self):
        with tempfile.TemporaryDirectory(dir=pitch.REPO) as temp:
            path = os.path.join(temp, "row.wav")
            with wave.open(path, "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(b"\1\0" * 320)
            relative = os.path.relpath(path, pitch.REPO)
            measurement = {"pitch_status": "tracker_failure",
                           "pitch_error": "only 0 voiced frames",
                           "pitch_frames": 1, "voiced_frames": 0,
                           "voiced_coverage": 0.0}
            row = {"adapter": "a", "seed": 1, "passage": 0,
                   "category": "narration", "uid": "u",
                   "source_sha256": "s", "wav": relative,
                   "declared_mean_f0": 120.0, **measurement}
            expected = [(('a', 1, 0),
                         ('u', 's', 'narration', relative, 120.0))]
            with patch.object(pitch, "measure_pitch", return_value=measurement):
                self.assertEqual({('a', 1, 0)},
                                 pitch.validate_rows([row], expected))
                with self.assertRaisesRegex(pitch.PitchProfileError,
                                            "duplicate or foreign"):
                    pitch.validate_rows([row, row], expected)


if __name__ == "__main__":
    unittest.main()
