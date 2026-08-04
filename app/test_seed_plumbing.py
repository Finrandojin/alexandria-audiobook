"""Every generation path must actually read the seed it is given.

THE BUG THIS LOCKS DOWN. `generate_lora_voice` contained 121 lines and zero
occurrences of the word `seed`, while its three sibling paths all read
`voice_data["seed"]` and called `torch.manual_seed`. So the seed field was
accepted, stored in voice_config, displayed in the UI - and silently ignored.
Every line of every `lora` voice was an independent draw, including NARRATOR,
which speaks 1,581 of this book's 2,606 lines.

It survived because nothing tested for it. Six experiment comparisons were run
on the contaminated output and had to be re-run, and the *measured* consequence
was large: seeding cut the non-prose failure count from 21/25 to 11/25.

The user found it by LISTENING and saying it sounded like several narrators.
The same instability had already been measured twice and misfiled twice - once
as YIN octave error, once as model behaviour.

WHY THIS TESTS PLUMBING AND NOT AUDIO. An end-to-end waveform check needs the
model on a GPU, so it can only ever run on one machine and would be skipped
everywhere else - which is the same as not existing. The defect was never
subtle acoustics; it was a line of code that was absent. Absence is exactly
what a source-level test catches, on any machine, in milliseconds.

The end-to-end proof is separate and already recorded: seed=7 produced
byte-identical output across three runs (99,840 samples, identical hash) and
seed=-1 did not.
"""
import ast
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tts.py")

# Every method that actually drives the model. If a new one is added it belongs
# here, because the failure mode is silence: an unseeded path looks completely
# normal until someone listens to a whole chapter.
#
# NOTE the layering, which the first draft of this file got wrong. Public
# `generate_clone_voice` and `generate_custom_voice` are three-line dispatchers
# that pick local or external and delegate; they hold no seed logic and must
# not be asserted to. The generation lives one level down, in the `_local_*`
# implementations. Asserting against the wrapper produced four confident
# failures against correct code.
SEEDED_METHODS = [
    "generate_lora_voice",
    "generate_voice_design",
    "_local_generate_custom",
    "_local_generate_clone",
    "_local_batch_custom",
    "_local_batch_clone",
    "_local_batch_lora",
]

# The public entry points, which must delegate rather than generate. If one of
# these ever grows its own generation call it needs to move into the list above.
DISPATCHERS = ["generate_clone_voice", "generate_custom_voice"]


def method_node(name):
    with open(TTS_PATH, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


class TestSeedIsRead(unittest.TestCase):

    def test_every_generation_path_exists(self):
        for name in SEEDED_METHODS:
            self.assertIsNotNone(method_node(name), f"{name} not found in tts.py")

    def test_every_generation_path_reads_the_seed(self):
        """The exact absence that shipped: no mention of `seed` at all."""
        for name in SEEDED_METHODS:
            with self.subTest(method=name):
                src = ast.unparse(method_node(name))
                self.assertIn("seed", src,
                              f"{name} never mentions `seed`; this is the "
                              f"defect that made every line a new voice")

    def test_every_generation_path_calls_manual_seed(self):
        """Reading the value is not enough - it has to reach torch.

        Verified before the fix that `torch.manual_seed` alone IS sufficient
        for reproducibility here, so this is the operative call and not a
        stand-in for one.
        """
        for name in SEEDED_METHODS:
            with self.subTest(method=name):
                node = method_node(name)
                calls = [n for n in ast.walk(node)
                         if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Attribute)
                         and n.func.attr == "manual_seed"]
                self.assertTrue(calls, f"{name} does not call manual_seed")

    def test_seed_is_not_a_hardcoded_constant(self):
        """A path that seeds from a literal is reproducible AND wrong: every
        character would share one voice draw, which sounds like one narrator
        reading everybody - the opposite of, and just as broken as, the bug
        this file is about."""
        for name in SEEDED_METHODS:
            with self.subTest(method=name):
                node = method_node(name)
                for call in [n for n in ast.walk(node)
                             if isinstance(n, ast.Call)
                             and isinstance(n.func, ast.Attribute)
                             and n.func.attr == "manual_seed"]:
                    self.assertFalse(
                        call.args and isinstance(call.args[0], ast.Constant),
                        f"{name} seeds from a literal")

    def test_dispatchers_delegate_and_do_not_generate(self):
        """The public clone/custom entry points route to local or external and
        nothing else. This is the layering the first draft of this test got
        wrong; encoding it keeps the next reader from making the same call."""
        for name in DISPATCHERS:
            with self.subTest(method=name):
                node = method_node(name)
                self.assertIsNotNone(node, f"{name} not found")
                src = ast.unparse(node)
                self.assertNotIn("manual_seed", src,
                                 f"{name} now generates; move it into "
                                 f"SEEDED_METHODS")
                self.assertTrue(
                    "_local_generate" in src,
                    f"{name} no longer delegates to a _local_generate_* path")

    def test_negative_seed_does_not_seed(self):
        """-1 means "draw randomly" and is the shipping default on 70 of 71
        characters, so an unconditional manual_seed would freeze production
        into one voice draw per character."""
        for name in SEEDED_METHODS:
            with self.subTest(method=name):
                node = method_node(name)
                guarded = any(
                    isinstance(n, ast.If)
                    and any(isinstance(c, ast.Call)
                            and isinstance(c.func, ast.Attribute)
                            and c.func.attr == "manual_seed"
                            for c in ast.walk(n))
                    for n in ast.walk(node))
                self.assertTrue(guarded,
                                f"{name} calls manual_seed unconditionally")


if __name__ == "__main__":
    unittest.main()
