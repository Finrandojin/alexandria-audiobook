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
        if self.behaviour == "returns_none":
            with open(path, "wb") as fh:
                fh.write(self.payload)
            return None
        with open(path, "wb") as fh:
            fh.write(self.payload)
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
            engine = FakeEngine("ok", payload=b"FRESH")
            render(engine, "text", "", "NARRATOR", {}, LORA, path)
            self.assertEqual(open(path, "rb").read(), b"FRESH")

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


if __name__ == "__main__":
    unittest.main()
