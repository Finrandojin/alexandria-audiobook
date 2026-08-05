import os
import sys
import tempfile
import unittest
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.nonprose_replication import (
    archive_checkpoint, feature_gap, get_resumable_rows, save_checkpoint,
    summarize, surface_features, validate_resumed_rows)


class TestNonproseReplication(unittest.TestCase):
    def test_surface_features_are_explicit_and_bounded(self):
        features = surface_features("ISBN 12 / Title!")
        self.assertEqual(16, features["chars"])
        self.assertEqual(2, features["words"])
        for key in ("digit_fraction", "uppercase_word_fraction",
                    "punctuation_fraction"):
            self.assertGreaterEqual(features[key], 0)
            self.assertLessEqual(features[key], 1)

    def test_feature_gap_does_not_claim_exact_matching(self):
        gap = feature_gap("ISBN 12.", "A sentence.")
        self.assertGreater(gap["digit_fraction"], 0)
        self.assertGreater(gap["uppercase_word_fraction"], 0)

    def test_summary_keeps_adapter_seed_and_class_separate(self):
        base = {"adapter": "a", "seed": 1, "words": 10,
                "errors": 2, "failed": True, "substitutions": 1,
                "deletions": 0, "insertions": 1}
        rows = [{**base, "class": "nonprose"},
                {**base, "class": "prose", "errors": 0,
                 "failed": False, "substitutions": 0, "insertions": 0}]
        summary = summarize(rows)
        self.assertEqual(2, len(summary))
        by_class = {r["class"]: r for r in summary}
        self.assertEqual(1, by_class["nonprose"]["insertions"])
        self.assertEqual(0, by_class["prose"]["insertions"])

    def test_checkpoint_roundtrip_requires_exact_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "new", "checkpoint.json")
            rows = [{"adapter": "a"}]
            fingerprint = {"schema": 1, "config": "abc"}
            save_checkpoint(path, fingerprint, rows)
            restored, mismatch = get_resumable_rows(path, fingerprint)
            self.assertEqual(rows, restored)
            self.assertIsNone(mismatch)
            restored, mismatch = get_resumable_rows(
                path, {"schema": 1, "config": "changed"})
            self.assertEqual([], restored)
            self.assertIn("does not match", mismatch)

    def test_mismatched_checkpoint_is_archived_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "checkpoint.json")
            save_checkpoint(path, {"old": True}, [])
            archived = archive_checkpoint(path, "identity changed")
            self.assertFalse(os.path.exists(path))
            self.assertTrue(os.path.isfile(archived))
            self.assertIn(".stale-", archived)
            save_checkpoint(path, {"new": True}, [])
            second = archive_checkpoint(path, "identity changed again")
            self.assertNotEqual(archived, second)
            self.assertTrue(os.path.isfile(archived))
            self.assertTrue(os.path.isfile(second))

    def test_resumed_row_requires_matching_identity_and_decodable_wav(self):
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with tempfile.TemporaryDirectory(dir=repo) as tmp:
            wav = os.path.join(tmp, "row.wav")
            with wave.open(wav, "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(b"\0\0" * 160)
            relative = os.path.relpath(wav, repo)
            row = {"adapter": "a", "seed": 1, "pair": 0,
                   "class": "prose", "uid": "u", "source_sha256": "s",
                   "wav": relative, "words": 1, "errors": 0,
                   "failed": False, "substitutions": 0, "deletions": 0,
                   "insertions": 0, "transcript": "ok"}
            expected = [(('a', 1, 0, 'prose'), ('u', 's', relative))]
            self.assertEqual({('a', 1, 0, 'prose')},
                             validate_resumed_rows([row], expected))
            row["errors"] = 1
            with self.assertRaisesRegex(ValueError,
                                        "inconsistent error counts"):
                validate_resumed_rows([row], expected)
            row["errors"] = 0
            with open(wav, "wb") as handle:
                handle.write(b"not audio")
            with self.assertRaisesRegex(ValueError, "unusable WAV"):
                validate_resumed_rows([row], expected)

    def test_resumed_row_rejects_wav_outside_repository(self):
        row = {"adapter": "a", "seed": 1, "pair": 0,
               "class": "prose", "uid": "u", "source_sha256": "s",
               "wav": "../outside.wav", "words": 1, "errors": 0,
               "failed": False, "substitutions": 0, "deletions": 0,
               "insertions": 0, "transcript": "ok"}
        expected = [(('a', 1, 0, 'prose'),
                     ('u', 's', '../outside.wav'))]
        with self.assertRaisesRegex(ValueError, "escapes the repository"):
            validate_resumed_rows([row], expected)

    def test_resumed_rows_reject_duplicate_keys(self):
        row = {"adapter": "a", "seed": 1, "pair": 0, "class": "prose"}
        with self.assertRaisesRegex(ValueError, "duplicate row key"):
            validate_resumed_rows(
                [row, row],
                [(('a', 1, 0, 'prose'), ('u', 's', 'missing.wav'))])


if __name__ == "__main__":
    unittest.main()
