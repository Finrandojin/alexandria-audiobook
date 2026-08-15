import os
import tempfile
import unittest

import numpy as np
import soundfile as sf

from experiments.asr_backends import run_energy_vad


class EnergyVadTests(unittest.TestCase):
    def test_splits_long_silence_but_keeps_short_pause(self):
        rate = 16000
        tone = np.sin(np.arange(rate // 2) * 2 * np.pi * 220 / rate) * 0.2
        audio = np.concatenate((tone, np.zeros(rate // 10), tone,
                                np.zeros(rate // 2), tone))
        with tempfile.TemporaryDirectory() as work:
            path = os.path.join(work, "probe.wav")
            sf.write(path, audio, rate)
            _text, segments = run_energy_vad(path)
        self.assertEqual(2, len(segments))
        self.assertAlmostEqual(0, segments[0][0], places=2)
        self.assertAlmostEqual(1.6, segments[1][0], places=2)


if __name__ == "__main__":
    unittest.main()
