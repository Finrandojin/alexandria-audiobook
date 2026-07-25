import unittest

from project import get_speakable_entries, group_into_chunks
from tts import DEFAULT_PAUSE_MS


class SpeakableEntryTests(unittest.TestCase):
    def test_nonverbal_dialogue_becomes_pause_without_mutating_input(self):
        entries = [
            {"speaker": "A", "text": "Wait here.", "instruct": "quiet"},
            {"speaker": "A", "text": "――――", "instruct": "silent"},
            {"speaker": "B", "text": "I understand.", "instruct": "calm"},
        ]
        prepared = get_speakable_entries(entries)
        self.assertEqual(["Wait here.", "I understand."],
                         [entry["text"] for entry in prepared])
        self.assertEqual(DEFAULT_PAUSE_MS, prepared[0]["pause_after"])
        self.assertNotIn("pause_after", entries[0])

    def test_block_glyphs_and_leading_marks_are_not_sent_to_tts(self):
        entries = [
            {"speaker": "A", "text": "…", "instruct": "silent"},
            {"speaker": "A", "text": "■■●■", "instruct": "noise"},
        ]
        self.assertEqual([], group_into_chunks(entries))

    def test_spoken_words_with_punctuation_remain_speakable(self):
        entries = [{"speaker": "A", "text": "No—wait!", "instruct": "urgent"}]
        self.assertEqual("No—wait!", group_into_chunks(entries)[0]["text"])


class UnspeakablePassthroughTest(unittest.TestCase):
    """Scene-break glyphs embedded in prose reached TTS as text. is_nonverbal_text
    only fires on entries with no alphanumerics at all, so a break inside a
    paragraph passed every gate: 57 of mushoku16's 76 shipped into the script
    while the run reported one failure."""

    def test_embedded_scene_break_becomes_a_pause(self):
        from project import get_speakable_entries
        entries = [{"speaker": "NARRATOR", "instruct": "",
                    "text": "I wonder if she hates mice\n\n\u25a0\n\nIt seems that a cat"}]
        out = get_speakable_entries(entries)
        self.assertEqual(len(out), 2)
        self.assertNotIn("\u25a0", out[0]["text"])
        self.assertNotIn("\u25a0", out[1]["text"])
        self.assertGreaterEqual(out[0].get("pause_after", 0), 1000)

    def test_leading_scene_break_makes_no_orphan_pause(self):
        from project import get_speakable_entries
        entries = [{"speaker": "NARRATOR", "instruct": "",
                    "text": "\u25a0\n\nThe day began."}]
        out = get_speakable_entries(entries)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["text"], "The day began.")

    def test_symbol_is_verbalized(self):
        from project import get_speakable_entries
        entries = [{"speaker": "NARRATOR", "instruct": "",
                    "text": "the value is \u221e."}]
        out = get_speakable_entries(entries)
        self.assertEqual(out[0]["text"], "the value is infinity.")

    def test_elongation_moves_into_instruct(self):
        from project import get_speakable_entries
        from verbalization import ELONGATION_HINT
        entries = [{"speaker": "ERIS", "instruct": "Cheerful.", "text": "Yaaay~"}]
        out = get_speakable_entries(entries)
        self.assertEqual(out[0]["text"], "Yaaay")
        self.assertIn("Cheerful.", out[0]["instruct"])
        self.assertIn(ELONGATION_HINT, out[0]["instruct"])

    def test_clean_entry_is_unchanged(self):
        from project import get_speakable_entries
        entries = [{"speaker": "ERIS", "instruct": "Flat.", "text": "Nothing odd."}]
        out = get_speakable_entries(entries)
        self.assertEqual(out[0]["text"], "Nothing odd.")
        self.assertEqual(out[0]["instruct"], "Flat.")

    def test_caller_data_is_not_mutated(self):
        from project import get_speakable_entries
        entries = [{"speaker": "NARRATOR", "instruct": "", "text": "a\n\n\u25a0\n\nb"}]
        get_speakable_entries(entries)
        self.assertIn("\u25a0", entries[0]["text"])


if __name__ == "__main__":
    unittest.main()


class ReviewFlagTest(unittest.TestCase):
    """Unmapped glyphs and pictographic kana are reported, never guessed at."""

    def test_pictographic_kana_is_flagged_and_left_alone(self):
        from project import split_on_unspeakable
        entry = {"speaker": "NARRATOR", "instruct": "",
                 "text": "Eris pouts, her mouth へ."}
        parts, review = split_on_unspeakable(entry, 1000)
        self.assertIn("へ", parts[0]["text"])
        self.assertIn("へ", review)

    def test_repeated_unmapped_glyph_is_counted_each_time(self):
        # text.index() would have found only the first occurrence.
        from project import split_on_unspeakable
        entry = {"speaker": "NARRATOR", "instruct": "", "text": "a ⌘ b ⌘ c"}
        _parts, review = split_on_unspeakable(entry, 1000)
        self.assertEqual(review.count("⌘"), 2)

    def test_kana_in_real_japanese_is_not_flagged(self):
        from project import split_on_unspeakable
        entry = {"speaker": "NARRATOR", "instruct": "", "text": "こんにちは"}
        _parts, review = split_on_unspeakable(entry, 1000)
        self.assertEqual(review, [])
