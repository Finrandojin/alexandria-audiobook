"""An unmapped glyph must reach a human, not vanish.

split_on_unspeakable deliberately leaves a character it cannot classify in the
text and reports it, because guessing at an unknown symbol is worse than
flagging it. The only production caller bound that report to _review and threw
it away, so "reported" meant reported to nobody: the glyph went to TTS with no
warning anywhere. Found by audit.
"""
import logging
import unittest

from project import get_speakable_entries, group_into_chunks


class ReviewCharactersReachTheCallerTest(unittest.TestCase):
    ENTRY = {"speaker": "NARRATOR", "instruct": "Calm.",
             "text": "Eris pouts ⌘"}

    def test_an_unmapped_symbol_is_collected(self):
        review = []
        get_speakable_entries([self.ENTRY], review_sink=review)
        self.assertEqual(["⌘"], [item["character"] for item in review])

    def test_the_report_says_which_entry_it_came_from(self):
        review = []
        get_speakable_entries([{"speaker": "N", "instruct": "", "text": "ok."},
                               self.ENTRY], review_sink=review)
        self.assertEqual(1, review[0]["entry_index"])

    def test_ordinary_text_reports_nothing(self):
        review = []
        get_speakable_entries([{"speaker": "N", "instruct": "", "text": "Hello."}],
                              review_sink=review)
        self.assertEqual([], review)

    def test_a_mapped_glyph_is_not_flagged(self):
        # ■ is a known scene break, not an unknown symbol.
        review = []
        get_speakable_entries([{"speaker": "N", "instruct": "",
                                "text": "One. ■ Two."}], review_sink=review)
        self.assertEqual([], review)

    def test_grouping_passes_the_report_through(self):
        review = []
        group_into_chunks([self.ENTRY], review_sink=review)
        self.assertEqual(["⌘"], [item["character"] for item in review])

    def test_the_symbol_still_reaches_the_text(self):
        # Reporting must not become silent removal: the design prefers a
        # visible glyph a human can act on over a guess.
        chunks = group_into_chunks([self.ENTRY])
        self.assertIn("⌘", chunks[0]["text"])

    def test_callers_that_pass_no_sink_still_work(self):
        self.assertEqual(1, len(group_into_chunks([self.ENTRY])))


class ReviewIsLoggedTest(unittest.TestCase):
    def test_building_chunks_warns_about_unmapped_glyphs(self):
        import project
        with self.assertLogs(project.logger, level=logging.WARNING) as captured:
            project.log_review_characters(
                [{"entry_index": 3, "character": "⌘"},
                 {"entry_index": 9, "character": "⌘"}])
        message = "\n".join(captured.output)
        self.assertIn("⌘", message)
        self.assertIn("2", message)

    def test_nothing_is_logged_when_there_is_nothing_to_review(self):
        import project
        with self.assertNoLogs(project.logger, level=logging.WARNING):
            project.log_review_characters([])


if __name__ == "__main__":
    unittest.main()
