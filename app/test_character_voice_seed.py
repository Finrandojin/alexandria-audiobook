"""A character's voice must be one draw for the whole book.

`generate_lora_voice` ignored the seed field entirely until 2026-08-04, so
every line of every LoRA voice was an independent draw of that voice. The user
identified it by ear as "multiple narrators" before any metric did - the same
instability had been measured twice and misfiled twice.

Fixing the plumbing made the field WORK. It did not make anything SET one:
70 of 71 characters in the shipped config still carried "-1", so a character's
voice was still redrawn per line, now deliberately rather than accidentally.
These tests cover the assignment side.

Measured, for scale: seeded generation is byte-identical across fresh
processes; unseeded, one adapter's pitch moves across a 32.4 Hz median band,
which is wider than the gap between many distinct voices in the pool.
"""
import ast
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import character_voice_seed

APP = os.path.dirname(os.path.abspath(__file__))


class TestSeedDerivation(unittest.TestCase):

    def test_the_same_character_always_gets_the_same_seed(self):
        """The whole point: stable across runs, books and regenerations."""
        self.assertEqual(character_voice_seed("NARRATOR"),
                         character_voice_seed("NARRATOR"))

    def test_different_characters_get_different_seeds(self):
        """A single global seed would be reproducible AND wrong - every
        character would share one draw of the voice."""
        names = ["NARRATOR", "EMILIA", "SUBARU", "REINHARD", "FELT", "ROM"]
        self.assertEqual(len({character_voice_seed(n) for n in names}),
                         len(names))

    def test_spelling_case_does_not_change_the_voice(self):
        """voice_config keys arrive in inconsistent case from different
        writers; 'Subaru' and 'SUBARU' are one character and must not get two
        different voices."""
        self.assertEqual(character_voice_seed("Subaru"),
                         character_voice_seed("SUBARU"))
        self.assertEqual(character_voice_seed(" Subaru "),
                         character_voice_seed("Subaru"))

    def test_the_seed_is_never_negative(self):
        """-1 is the sentinel for 'draw randomly' and seeds are tested with
        `int(...) >= 0`. A negative derived seed would silently mean unseeded,
        which is the exact bug this exists to prevent."""
        for n in ["A", "z", "NARRATOR", "名前", "x" * 200, "!!!"]:
            with self.subTest(name=n):
                self.assertGreaterEqual(character_voice_seed(n), 0)

    def test_an_empty_name_does_not_crash(self):
        for n in ("", None, "   "):
            self.assertEqual(character_voice_seed(n), 0)

    def test_the_seed_fits_what_torch_accepts(self):
        for n in ["NARRATOR", "EMILIA", "x" * 50]:
            self.assertLess(character_voice_seed(n), 2 ** 31)


class TestAssignmentSitesUseIt(unittest.TestCase):
    """Two places write a voice entry, and both hard-coded -1 independently.

    Asserting the call sites rather than only the helper, because a correct
    helper nothing calls is what the previous state effectively was.
    """

    def _source(self, rel):
        with open(os.path.join(APP, rel), encoding="utf-8") as fh:
            return fh.read()

    def test_voice_suggestion_apply_sets_a_stable_seed(self):
        src = self._source(os.path.join("routers", "voices.py"))
        self.assertIn("character_voice_seed", src)
        self.assertNotIn('"seed": "-1"', src,
                         "a hard-coded -1 means the voice is redrawn per line")

    def test_persona_generation_sets_a_stable_seed(self):
        src = self._source("generate_personas.py")
        self.assertIn("character_voice_seed", src)
        self.assertNotIn('"seed": -1\n', src,
                         "a hard-coded -1 means the voice is redrawn per line")

    def test_neither_site_seeds_from_a_literal(self):
        """A constant seed shared by every character is as wrong as none."""
        for rel in (os.path.join("routers", "voices.py"),
                    "generate_personas.py"):
            with self.subTest(file=rel):
                tree = ast.parse(self._source(rel))
                for node in ast.walk(tree):
                    if not (isinstance(node, ast.Call)
                            and getattr(node.func, "id", "")
                            == "character_voice_seed"):
                        continue
                    self.assertTrue(node.args, "seed derived from nothing")
                    self.assertNotIsInstance(
                        node.args[0], ast.Constant,
                        f"{rel} derives every character's seed from a literal")


if __name__ == "__main__":
    unittest.main()
