"""Tests for the pre-TTS speech normaliser.

This exists because of a measured defect, not a theory: a 349-character table
of contents separated by U+2022 bullets produced 24.2 seconds of audio at
normal level that whisper transcribed as "* * * * * * * *" - Qwen3-TTS
vocalising instead of reading. It would have shipped, because nothing in this
project inspects generated audio.

The risk in fixing it is over-reach. This function runs on EVERY line of every
book, so a rule that mangles ordinary prose is far worse than the bug it fixes.
Most of these tests therefore check that normal text is left alone.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tts import normalize_for_speech


class TestStructuralMarks(unittest.TestCase):
    """Bullets separate items; they must become breaks, not words."""

    def test_the_measured_defect(self):
        text = ("Contents. • Cover. • Insert. • Title Page. • Copyright. "
                "• Prologue The Waste Heat of the Beginning.")
        out = normalize_for_speech(text)
        self.assertNotIn("•", out)
        self.assertIn("Contents.", out)
        self.assertIn("Cover.", out)
        # The items must stay separated, not run into one another.
        self.assertIn("Cover. Insert.", out)

    def test_bullet_becomes_a_break_not_a_word(self):
        out = normalize_for_speech("Apples • Oranges")
        self.assertNotIn("bullet", out.lower())
        self.assertIn("Apples.", out)

    def test_all_structural_marks_removed(self):
        for mark in "•·▪◦‣∙■□◆●▲─━―*_~":
            out = normalize_for_speech(f"before {mark} after")
            self.assertNotIn(mark, out, f"{mark!r} survived")

    def test_runs_of_marks_collapse(self):
        # "■■■" is one scene break, not three sentence ends.
        self.assertEqual(normalize_for_speech("one ■■■ two"), "one. two.")


class TestSpokenSymbols(unittest.TestCase):
    """Some symbols are words the writer expects read aloud."""

    def test_copyright_sign_is_spoken(self):
        self.assertIn("copyright", normalize_for_speech("© 2016 Tappei").lower())

    def test_ampersand_is_spoken(self):
        self.assertIn("and", normalize_for_speech("Tom & Jerry"))

    def test_already_spelled_word_is_not_doubled(self):
        # "Copyright © 2016" must not read "copyright copyright 2016".
        out = normalize_for_speech("Copyright © 2016 Tappei")
        self.assertEqual(out.lower().count("copyright"), 1)

    def test_daggers_are_dropped_not_spoken(self):
        # A footnote dagger is a reference mark; reading it is nonsense.
        out = normalize_for_speech("a claim† here")
        self.assertNotIn("†", out)
        self.assertNotIn("dagger", out.lower())


class TestLeavesProseAlone(unittest.TestCase):
    """The function runs on every line; over-reach is worse than the bug."""

    def test_plain_sentence_is_unchanged(self):
        self.assertEqual(normalize_for_speech("Hello world."), "Hello world.")

    def test_typographic_quotes_are_preserved(self):
        # 59,004 of the library's non-ASCII characters are U+2019 and the
        # model reads them correctly. Touching them would be pure risk.
        text = "She said “hello” and it’s fine."
        self.assertEqual(normalize_for_speech(text), text)

    def test_dialogue_with_dashes_survives(self):
        text = "Wait—what do you mean?"
        self.assertIn("Wait", normalize_for_speech(text))

    def test_ellipsis_is_preserved(self):
        self.assertIn("…", normalize_for_speech("I guess… maybe."))

    def test_empty_and_none_are_safe(self):
        self.assertEqual(normalize_for_speech(""), "")
        self.assertEqual(normalize_for_speech(None), None)

    def test_symbol_only_text_does_not_become_a_bare_period(self):
        # A chunk that is nothing but marks has nothing to say.
        self.assertEqual(normalize_for_speech("• • •"), "")

    def test_terminal_punctuation_is_not_duplicated(self):
        self.assertEqual(normalize_for_speech("Done."), "Done.")
        self.assertFalse(normalize_for_speech("Done.").endswith(".."))


class TestIdempotence(unittest.TestCase):
    """It sits at several entry points; double application must be harmless."""

    def test_applying_twice_changes_nothing(self):
        for text in ["Contents. • Cover. • Insert.", "Copyright © 2016",
                     "Tom & Jerry", "Plain prose here.", "one ■■■ two"]:
            once = normalize_for_speech(text)
            self.assertEqual(normalize_for_speech(once), once, repr(text))


if __name__ == "__main__":
    unittest.main()
