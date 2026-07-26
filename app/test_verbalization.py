import unittest

from verbalization import (ELONGATION_HINT, SCENE_BREAK_MARKER, SET_APART_HINT,
                           SUNG_HINT,
                           classify, extract_delivery_cues,
                           is_pictographic_kana, split_bracketed_spans,
                           strip_emoji_dividers)


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


class BracketedSpanTest(unittest.TestCase):
    """Angle brackets mark a different delivery mode, not text to read out.
    In mushoku16 they wrap precognition visions - "<I saw the figure of a
    person.>" - and elsewhere in the library they wrap system announcements.
    Either way the brackets are never spoken, and the span inside wants its own
    delivery, so it becomes its own part."""

    def test_bracketed_span_is_split_out(self):
        parts = split_bracketed_spans(
            "He readied himself. <I saw the figure of a person.> I shuddered.")
        self.assertEqual(len(parts), 3)
        self.assertEqual(parts[0], ("He readied himself.", False))
        self.assertEqual(parts[1], ("I saw the figure of a person.", True))
        self.assertEqual(parts[2], ("I shuddered.", False))

    def test_brackets_are_never_kept(self):
        parts = split_bracketed_spans("<My vision was covered in light.>")
        self.assertEqual(parts, [("My vision was covered in light.", True)])

    def test_text_without_brackets_is_one_plain_part(self):
        parts = split_bracketed_spans("Nothing bracketed here.")
        self.assertEqual(parts, [("Nothing bracketed here.", False)])

    def test_lone_angle_bracket_is_left_alone(self):
        # A stray < with no closing partner is not a span; do not guess.
        parts = split_bracketed_spans("The value is < 5 in that case.")
        self.assertEqual(parts, [("The value is < 5 in that case.", False)])

    def test_an_emoticon_is_not_treated_as_a_span(self):
        # "><" is a face, not a bracket pair.
        parts = split_bracketed_spans("Sorry >< I forgot.")
        self.assertEqual(len(parts), 1)
        self.assertFalse(parts[0][1])


if __name__ == "__main__":
    unittest.main()


class EmojiDividerTest(unittest.TestCase):
    """A run of repeated emoji is a section divider, not something to voice.

    Each emoji is Unicode category So, so without this every one would be
    reported for review individually - and if the reviewer waved them through,
    the narrator would read a decorative separator aloud.
    """

    def test_a_run_becomes_a_scene_break_marker(self):
        text, runs = strip_emoji_dividers("swamp. \U0001F409 \U0001F409 \U0001F409 Her memories")
        self.assertEqual(1, runs)
        self.assertIn(SCENE_BREAK_MARKER, text)
        self.assertNotIn("\U0001F409", text)

    def test_a_lone_emoji_is_left_alone(self):
        text, runs = strip_emoji_dividers("She held the \U0001F409 charm.")
        self.assertEqual(0, runs)
        self.assertIn("\U0001F409", text)

    def test_two_adjacent_emoji_count_as_a_run(self):
        _text, runs = strip_emoji_dividers("end ✨✨ start")
        self.assertEqual(1, runs)

    def test_the_marker_classifies_as_a_scene_break(self):
        # The whole point of folding to this glyph is to reuse the pause path.
        self.assertEqual("scene_break", classify(SCENE_BREAK_MARKER))

    def test_prose_without_emoji_is_returned_unchanged(self):
        original = "She held the charm tightly."
        text, runs = strip_emoji_dividers(original)
        self.assertEqual((original, 0), (text, runs))


class EmojiDividerSplitTest(unittest.TestCase):
    def test_a_divider_run_splits_the_entry_and_carries_a_pause(self):
        from project import split_on_unspeakable
        entry = {"speaker": "NARRATOR", "instruct": "Calm.",
                 "text": "swamp. \U0001F409 \U0001F409 \U0001F409 Her memories were a blur."}
        parts, review = split_on_unspeakable(entry, 700)
        self.assertEqual([], review)
        self.assertEqual(2, len(parts))
        self.assertEqual(700, parts[0]["pause_after"])
        self.assertNotIn("pause_after", parts[1])
        self.assertNotIn("\U0001F409", "".join(p["text"] for p in parts))

    def test_a_lone_emoji_still_reaches_review(self):
        from project import split_on_unspeakable
        entry = {"speaker": "NARRATOR", "instruct": "Calm.",
                 "text": "She held the \U0001F409 charm."}
        parts, review = split_on_unspeakable(entry, 700)
        self.assertEqual(["\U0001F409"], review)
        self.assertEqual(1, len(parts))
