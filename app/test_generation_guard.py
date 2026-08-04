"""Tests for the render guard.

The defect: tts.py's generate_* methods return False on failure instead of
raising, and every harness ignored that boolean and checked os.path.exists
instead. With a WAV left at that path by an earlier run, a FAILED generation
was scored as a success on STALE AUDIO - invisible to the harness, and wrong in
a direction nobody checks.

Each test below corresponds to one way that could happen.
"""
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.generation import GenerationFailed, render


class FakeEngine:
    """Stands in for TTSEngine. `behaviour` decides what generation does."""

    def __init__(self, behaviour="ok", payload=b"RIFFfake"):
        self.behaviour = behaviour
        self.payload = payload
        self.calls = 0

    def _do(self, path):
        self.calls += 1
        if self.behaviour == "returns_false":
            return False
        if self.behaviour == "no_file":
            return True
        if self.behaviour == "empty_file":
            open(path, "wb").close()
            return True
        if self.behaviour == "truncated_wav":
            # Header only, no sample data - this one fails to decode.
            import io, soundfile as sf, numpy as np
            buf = io.BytesIO()
            sf.write(buf, np.zeros(2400, dtype="float32"), 24000, format="WAV")
            with open(path, "wb") as fh:
                fh.write(buf.getvalue()[:28])
            return True
        if self.behaviour == "partially_truncated_wav":
            # THE HARD CASE, and the one the first version of this guard
            # missed. A real render cut off mid-write still DECODES -
            # libsndfile returns the frames that happen to be present rather
            # than raising - so `sf.info` accepts it and the short, wrong file
            # is scored. Only the declared RIFF size catches it.
            import io, soundfile as sf, numpy as np
            buf = io.BytesIO()
            sf.write(buf, np.zeros(24000, dtype="float32"), 24000,
                     format="WAV")
            whole = buf.getvalue()
            with open(path, "wb") as fh:
                fh.write(whole[:len(whole) // 4])
            return True
        if self.behaviour == "not_audio":
            with open(path, "wb") as fh:
                fh.write(b"this is plainly not a wav file at all")
            return True
        if self.behaviour == "header_no_samples":
            import soundfile as sf, numpy as np
            sf.write(path, np.zeros(0, dtype="float32"), 24000, format="WAV")
            return True
        if self.behaviour == "returns_none":
            import soundfile as sf, numpy as np
            sf.write(path, np.zeros(2400, dtype="float32"), 24000, format="WAV")
            return None
        # The success path must write DECODABLE audio, or every test here
        # passes for the wrong reason once decodability is checked.
        import soundfile as sf, numpy as np
        sf.write(path, np.zeros(2400, dtype="float32"), 24000, format="WAV")
        return True

    def generate_lora_voice(self, text, instruct, voice_data, path):
        return self._do(path)

    def generate_clone_voice(self, text, speaker, voice_config, path):
        return self._do(path)

    def generate_custom_voice(self, text, instruct, speaker, voice_config, path):
        return self._do(path)


LORA = {"type": "lora", "adapter_id": "a", "adapter_path": "lora_models/a"}


class TestStaleAudio(unittest.TestCase):
    """The failure that motivated the module."""

    def test_stale_file_does_not_mask_a_false_return(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "seg.wav")
            with open(path, "wb") as fh:
                fh.write(b"STALE AUDIO FROM AN EARLIER RUN")
            engine = FakeEngine("returns_false")
            with self.assertRaises(GenerationFailed):
                render(engine, "text", "", "NARRATOR", {}, LORA, path)

    def test_stale_file_is_deleted_before_generation(self):
        # Even when generation later succeeds, the old bytes must be gone -
        # otherwise a partially-written new file could be scored against
        # leftovers from the old one.
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "seg.wav")
            with open(path, "wb") as fh:
                fh.write(b"STALE")
            engine = FakeEngine("ok")
            render(engine, "text", "", "NARRATOR", {}, LORA, path)
            # The success path writes real WAV now that decodability is
            # checked, so assert the stale bytes are GONE and the file is
            # audio - which is the thing this test always meant.
            with open(path, "rb") as fh:
                head = fh.read(4)
            self.assertEqual(head, b"RIFF")
            import soundfile as sf
            self.assertGreater(sf.info(path).frames, 0)

    def test_stale_file_removed_even_when_generation_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "seg.wav")
            with open(path, "wb") as fh:
                fh.write(b"STALE")
            engine = FakeEngine("no_file")
            with self.assertRaises(GenerationFailed):
                render(engine, "text", "", "NARRATOR", {}, LORA, path)
            self.assertFalse(os.path.exists(path))


class TestFailureModes(unittest.TestCase):

    def test_false_return_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(GenerationFailed) as cm:
                render(FakeEngine("returns_false"), "t", "", "S", {}, LORA,
                       os.path.join(tmp, "a.wav"))
            self.assertIn("returned False", str(cm.exception))

    def test_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(GenerationFailed) as cm:
                render(FakeEngine("no_file"), "t", "", "S", {}, LORA,
                       os.path.join(tmp, "a.wav"))
            self.assertIn("wrote no file", str(cm.exception))

    def test_empty_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(GenerationFailed) as cm:
                render(FakeEngine("empty_file"), "t", "", "S", {}, LORA,
                       os.path.join(tmp, "a.wav"))
            self.assertIn("empty file", str(cm.exception))

    def test_none_return_is_success(self):
        # Some paths return None on success; treating that as failure would
        # discard good audio.
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.wav")
            self.assertEqual(
                render(FakeEngine("returns_none"), "t", "", "S", {}, LORA, path),
                path)


class TestDispatch(unittest.TestCase):
    """Routing must match production, or a defect found here is not a defect
    a listener would hear."""

    def _category_used(self, voice_data):
        seen = {}

        class Recorder(FakeEngine):
            def generate_lora_voice(self, *a):
                seen["cat"] = "lora"
                return super().generate_lora_voice(*a)

            def generate_clone_voice(self, *a):
                seen["cat"] = "clone"
                return super().generate_clone_voice(*a)

            def generate_custom_voice(self, *a):
                seen["cat"] = "custom"
                return super().generate_custom_voice(*a)

        with tempfile.TemporaryDirectory() as tmp:
            render(Recorder(), "t", "", "S", {}, voice_data,
                   os.path.join(tmp, "a.wav"))
        return seen["cat"]

    def test_lora_routes_to_lora(self):
        self.assertEqual(self._category_used(LORA), "lora")

    def test_clone_routes_to_clone(self):
        self.assertEqual(self._category_used({"type": "clone"}), "clone")

    def test_custom_is_the_fallback(self):
        self.assertEqual(self._category_used({"type": "custom"}), "custom")
        self.assertEqual(self._category_used({}), "custom")


class TestUndecodableAudio(unittest.TestCase):
    """Non-empty is not the same as usable.

    Existence and size were the only checks, so a file that is bytes but not
    audio scored as a successful render. Same defect as stale audio, one layer
    down: these bytes are fresh, and still not sound.
    """

    def test_a_truncated_wav_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(GenerationFailed) as cm:
                render(FakeEngine("truncated_wav"), "t", "", "S", {}, LORA,
                       os.path.join(tmp, "a.wav"))
            self.assertIn("undecodable", str(cm.exception))

    def test_a_partially_truncated_wav_is_rejected(self):
        """Decoding is not enough. Truncating a real 195,884-byte render to
        5,000 bytes still decodes to 2,478 frames instead of raising, so a
        decode check alone passes it. The RIFF header declares the size the
        file should be; a shortfall is truncation whatever it decodes to."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.wav")
            with self.assertRaises(GenerationFailed) as cm:
                render(FakeEngine("partially_truncated_wav"), "t", "", "S", {},
                       LORA, path)
            self.assertIn("truncated", str(cm.exception))
            self.assertIn("missing", str(cm.exception))

    def test_a_decodable_but_short_file_is_still_caught(self):
        """Guard against the guard: confirm the truncated file really does
        decode, so this test cannot pass merely because the file is unreadable
        - which is what made the first version look like it worked."""
        import soundfile as sf
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.wav")
            FakeEngine("partially_truncated_wav")._do(path)
            self.assertGreater(sf.info(path).frames, 0,
                               "file should decode; the point is that decoding "
                               "is insufficient")

    def test_trailing_metadata_is_not_treated_as_truncation(self):
        """Larger than declared is legal - trailing chunks are common. Only a
        shortfall is an error, or the guard rejects valid audio."""
        from experiments.generation import _check_riff_completeness
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.wav")
            FakeEngine("ok")._do(path)
            with open(path, "ab") as fh:
                fh.write(b"LIST" + b"\x00" * 64)
            _check_riff_completeness(path, "test", "S")   # must not raise

    def test_a_file_that_is_not_audio_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(GenerationFailed):
                render(FakeEngine("not_audio"), "t", "", "S", {}, LORA,
                       os.path.join(tmp, "a.wav"))

    def test_a_valid_header_with_no_samples_is_rejected(self):
        """A header can be well-formed and describe nothing. Zero frames is
        silence no listener would accept and no WER metric would flag."""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(GenerationFailed) as cm:
                render(FakeEngine("header_no_samples"), "t", "", "S", {}, LORA,
                       os.path.join(tmp, "a.wav"))
            self.assertIn("no audio", str(cm.exception))

    def test_real_audio_still_passes(self):
        """The guard must not reject good renders - a validator that fails
        closed on everything is worse than none."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.wav")
            self.assertEqual(
                render(FakeEngine("ok"), "t", "", "S", {}, LORA, path), path)
            import soundfile as sf
            self.assertEqual(sf.info(path).frames, 2400)


if __name__ == "__main__":
    unittest.main()
