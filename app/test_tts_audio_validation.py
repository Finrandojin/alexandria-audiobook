"""tts.py must not hand back audio it has not looked at.

THE ASYMMETRY THIS CLOSES. `project.py` validated audio when assembling chunks
into an export. `tts.py`, the layer that CREATES the audio, called sf.write and
returned - eleven `os.path.exists`/`getsize` checks and no decoding. A guard in
the assembly layer while the generating layer is unguarded is worse than no
guard: the pipeline looks protected.

Every generation path - lora, clone, custom, design, and the three batch
variants - funnels through `TTSEngine._save_wav`, so the check belongs there
rather than in seven callers that would drift apart. These tests assert that
funnel property as well as the checking, because if a future path writes
directly the coverage silently disappears.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio_validation import GeneratedAudioError, validate_generated_audio

TTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tts.py")


def _wav(path, seconds=0.5, rate=24000):
    import numpy as np
    import soundfile as sf
    sf.write(path, np.zeros(int(rate * seconds), dtype="float32"), rate)
    return path


class TestSaveWavIsTheOnlyFunnel(unittest.TestCase):
    """Structural: if a generation path stops using _save_wav, the validation
    added there stops covering it, and nothing else would say so."""

    def _paths(self):
        import ast
        with open(TTS_PATH, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        return {n.name: ast.unparse(n) for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef)}

    def test_every_generation_path_writes_through_save_wav(self):
        expected = ["generate_lora_voice", "generate_voice_design",
                    "_local_generate_custom", "_local_generate_clone",
                    "_local_batch_custom", "_local_batch_clone",
                    "_local_batch_lora"]
        fns = self._paths()
        for name in expected:
            with self.subTest(path=name):
                self.assertIn(name, fns, f"{name} missing from tts.py")
                src = fns[name]
                self.assertIn("_save_wav", src,
                              f"{name} no longer writes through _save_wav, so "
                              f"it is no longer validated")

    def test_save_wav_validates(self):
        src = self._paths()["_save_wav"]
        self.assertIn("validate_generated_audio", src)
        self.assertIn("remove_stale_audio", src,
                      "_save_wav must clear a stale file before writing")


class TestValidationRejectsBadAudio(unittest.TestCase):
    """Behavioural, against the validator _save_wav now calls."""

    def test_a_truncated_wav_is_rejected(self):
        """The case a decode check alone misses.

        Truncating a real render still decodes - libsndfile returns the frames
        that happen to be present. Only the declared RIFF size catches it.
        """
        with tempfile.TemporaryDirectory() as tmp:
            good = _wav(os.path.join(tmp, "good.wav"), seconds=2.0)
            with open(good, "rb") as fh:
                whole = fh.read()
            bad = os.path.join(tmp, "bad.wav")
            with open(bad, "wb") as fh:
                fh.write(whole[:len(whole) // 4])

            # Prove the premise: it really does decode.
            import soundfile as sf
            self.assertGreater(sf.info(bad).frames, 0,
                               "premise failed - this test would pass for the "
                               "wrong reason if the file were unreadable")

            with self.assertRaises(GeneratedAudioError) as cm:
                validate_generated_audio(bad, "test")
            self.assertIn("truncated", str(cm.exception))

    def test_a_non_audio_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "a.wav")
            with open(p, "wb") as fh:
                fh.write(b"not audio at all")
            with self.assertRaises(GeneratedAudioError):
                validate_generated_audio(p, "test")

    def test_an_empty_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "a.wav")
            open(p, "wb").close()
            with self.assertRaises(GeneratedAudioError) as cm:
                validate_generated_audio(p, "test")
            self.assertIn("empty", str(cm.exception))

    def test_a_missing_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(GeneratedAudioError):
                validate_generated_audio(os.path.join(tmp, "nope.wav"), "test")

    def test_real_audio_passes(self):
        """A validator that fails closed on everything is worse than none."""
        with tempfile.TemporaryDirectory() as tmp:
            p = _wav(os.path.join(tmp, "a.wav"))
            self.assertEqual(validate_generated_audio(p, "test"), p)


if __name__ == "__main__":
    unittest.main()
