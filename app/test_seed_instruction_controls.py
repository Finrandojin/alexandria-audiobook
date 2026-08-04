import os
import sys
import tempfile
import unittest
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.seed_instruction_controls import file_sha256, wav_info


class TestSeedInstructionControls(unittest.TestCase):
    def test_wav_info_hashes_bytes_and_reports_duration(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "probe.wav")
            with wave.open(path, "wb") as fh:
                fh.setnchannels(1)
                fh.setsampwidth(2)
                fh.setframerate(8000)
                fh.writeframes(b"\x00\x00" * 4000)
            info = wav_info(path)
            self.assertEqual(file_sha256(path), info["sha256"])
            self.assertEqual(4000, info["frames"])
            self.assertEqual(8000, info["rate"])
            self.assertEqual(0.5, info["duration_s"])


if __name__ == "__main__":
    unittest.main()
