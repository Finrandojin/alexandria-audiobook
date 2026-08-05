import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio_validation import GeneratedAudioError, validate_generated_audio
from project import ProjectManager


def _write_wav(path, frames=2400):
    sf.write(path, np.zeros(frames, dtype=np.float32), 24000, format="WAV")


class SharedAudioValidationTests(unittest.TestCase):
    def test_missing_empty_and_zero_frame_outputs_fail_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "missing.wav")
            with self.assertRaisesRegex(GeneratedAudioError, "wrote no file"):
                validate_generated_audio(missing, "test")

            empty = os.path.join(tmp, "empty.wav")
            open(empty, "wb").close()
            with self.assertRaisesRegex(GeneratedAudioError, "empty file"):
                validate_generated_audio(empty, "test")

            zero = os.path.join(tmp, "zero.wav")
            _write_wav(zero, frames=0)
            with self.assertRaisesRegex(GeneratedAudioError, "no audio frames"):
                validate_generated_audio(zero, "test")

    def test_decodable_partial_riff_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "partial.wav")
            _write_wav(path, frames=24000)
            with open(path, "rb") as handle:
                partial = handle.read(os.path.getsize(path) // 4)
            with open(path, "wb") as handle:
                handle.write(partial)
            self.assertGreater(sf.info(path).frames, 0)
            with self.assertRaisesRegex(GeneratedAudioError, "truncated audio"):
                validate_generated_audio(path, "test")

    def test_valid_audio_passes_after_full_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "valid.wav")
            _write_wav(path)
            self.assertEqual(path, validate_generated_audio(path, "test"))


class _BatchEngine:
    def __init__(self, write_output):
        self.write_output = write_output
        self.saw_stale = None

    def generate_batch(self, chunks, voice_config, output_dir, batch_seed):
        idx = chunks[0]["index"]
        path = os.path.join(output_dir, f"temp_batch_{idx}.wav")
        self.saw_stale = os.path.exists(path)
        if self.write_output:
            _write_wav(path)
        return {"completed": [idx], "failed": []}


class _SingleEngine:
    def generate_voice(self, text, instruct, speaker, voice_config, output_path):
        with open(output_path, "wb") as handle:
            handle.write(b"not audio")
        return True


class ProductionCompletionBoundaryTests(unittest.TestCase):
    def _manager(self, root):
        manager = ProjectManager(root)
        with open(manager.chunks_path, "w", encoding="utf-8") as handle:
            json.dump([{"uid": "stable", "speaker": "NARRATOR",
                        "text": "Hello.", "instruct": "",
                        "status": "pending"}], handle)
        with open(manager.voice_config_path, "w", encoding="utf-8") as handle:
            json.dump({"NARRATOR": {"type": "custom", "voice": "Ryan"}},
                      handle)
        return manager

    def test_batch_deletes_stale_output_before_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = self._manager(tmp)
            stale = os.path.join(tmp, "temp_batch_0.wav")
            _write_wav(stale)
            manager.engine = _BatchEngine(write_output=True)
            with patch.object(manager, "_export_chunk_audio",
                              return_value="voicelines/final.wav"):
                result = manager.generate_chunks_batch([0], batch_size=1)
            self.assertFalse(manager.engine.saw_stale)
            self.assertEqual([0], result["completed"])
            chunk = manager.load_chunks()[0]
            self.assertEqual("done", chunk["status"])
            self.assertIsNone(chunk["error"])

    def test_false_batch_completion_without_fresh_file_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = self._manager(tmp)
            stale = os.path.join(tmp, "temp_batch_0.wav")
            _write_wav(stale)
            manager.engine = _BatchEngine(write_output=False)
            result = manager.generate_chunks_batch([0], batch_size=1)
            self.assertFalse(manager.engine.saw_stale)
            self.assertEqual([], result["completed"])
            self.assertIn("Temp audio file not found", result["failed"][0][1])
            chunk = manager.load_chunks()[0]
            self.assertEqual("error", chunk["status"])
            self.assertIn("Temp audio file not found", chunk["error"])

    def test_single_chunk_rejects_undecodable_success_and_records_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = self._manager(tmp)
            manager.engine = _SingleEngine()
            success, error = manager.generate_chunk_audio(0)
            self.assertFalse(success)
            self.assertIn("undecodable audio", error)
            chunk = manager.load_chunks()[0]
            self.assertEqual("error", chunk["status"])
            self.assertEqual(error, chunk["error"])


if __name__ == "__main__":
    unittest.main()
