import unittest

from verbalization import (ELONGATION_HINT, SUNG_HINT, classify,
                           extract_delivery_cues, is_pictographic_kana)


class ClassifyTest(unittest.TestCase):

    def test_scene_breaks(self):
        # Ordered by measured library frequency: U+2500 alone outweighs every
        # other symbol combined (62,921 occurrences across 4,195 novels).
        for char in "─◇◆■□○━█△":
            self.assertEqual(classify(char), "scene_break", char)

    def test_verbalized_symbols(self):
        for char in "∞←→↑°©×":
            self.assertEqual(classify(char), "verbalize", char)

    def test_elongation(self):
        for char in "~～":
            self.assertEqual(classify(char), "elongation", char)

    def test_music_is_not_verbalized(self):
        # A music note usually brackets sung dialogue; turning it into the word
        # "music note" is worse than leaving it.
        for char in "♪♫":
            self.assertEqual(classify(char), "music", char)

    def test_unmapped_symbol_is_flagged_not_dropped(self):
        self.assertEqual(classify("⌘"), "review")
        self.assertEqual(classify("🍖"), "review")

    def test_ordinary_text_is_speakable(self):
        for char in "aZ9 .,!?'\"-":
            self.assertEqual(classify(char), "speakable", repr(char))


class PictographicKanaTest(unittest.TestCase):

    def test_lone_kana_is_pictographic(self):
        # "her mouth へ" - a shape, not language.
        self.assertTrue(is_pictographic_kana("へ", " ."))

    def test_kana_among_kana_is_language(self):
        self.assertFalse(is_pictographic_kana("へ", "んな"))

    def test_latin_is_never_pictographic(self):
        self.assertFalse(is_pictographic_kana("a", " ."))


class DeliveryCueTest(unittest.TestCase):

    def test_bracketing_music_notes_become_a_sung_hint(self):
        text, hints = extract_delivery_cues("♪ La la la ♪")
        self.assertEqual(text, "La la la")
        self.assertIn(SUNG_HINT, hints)

    def test_elongation_moves_into_a_hint(self):
        text, hints = extract_delivery_cues("Yaaay~")
        self.assertEqual(text, "Yaaay")
        self.assertIn(ELONGATION_HINT, hints)

    def test_lone_music_note_is_left_alone(self):
        # Mid-sentence, not bracketing: no safe interpretation, so don't guess.
        text, hints = extract_delivery_cues("a ♪ sound")
        self.assertEqual(text, "a ♪ sound")
        self.assertEqual(hints, [])

    def test_clean_text_is_untouched(self):
        text, hints = extract_delivery_cues("Nothing to do here.")
        self.assertEqual(text, "Nothing to do here.")
        self.assertEqual(hints, [])


if __name__ == "__main__":
    unittest.main()
