import unittest

from build_scoring_sheet import build_sheet, neighbour_context


class BuildSheetTest(unittest.TestCase):
    """One shared sample scored once per model, rather than pairwise arms:
    two identical runs disagree on 37.4% of speakers, which swamps any real
    between-model difference."""

    def _runs(self):
        return {
            "modelA": [{"speaker": "ERIS", "text": "Hello there."},
                       {"speaker": "NARRATOR", "text": "The wind blew."},
                       {"speaker": "ROXY", "text": "Good morning."}],
            "modelB": [{"speaker": "ERIS", "text": "Hello  there."},
                       {"speaker": "NARRATOR", "text": "The wind blew."},
                       {"speaker": "SYLPHY", "text": "Good morning."}],
        }

    def test_only_shared_lines_are_sampled(self):
        runs = self._runs()
        runs["modelB"].append({"speaker": "ERIS", "text": "Only in B."})
        rows = build_sheet(runs, size=10)
        self.assertNotIn("Only in B.", [r["text"] for r in rows])

    def test_whitespace_variation_still_counts_as_shared(self):
        rows = build_sheet(self._runs(), size=10)
        self.assertIn("Hello there.", [r["text"] for r in rows])

    def test_narrator_only_lines_are_excluded(self):
        rows = build_sheet(self._runs(), size=10)
        self.assertNotIn("The wind blew.", [r["text"] for r in rows])

    def test_disagreement_is_marked(self):
        rows = build_sheet(self._runs(), size=10)
        row = next(r for r in rows if r["text"] == "Good morning.")
        self.assertFalse(row["models_agree"])
        self.assertEqual(row["answers"], {"modelA": "ROXY", "modelB": "SYLPHY"})

    def test_agreement_is_marked(self):
        rows = build_sheet(self._runs(), size=10)
        row = next(r for r in rows if r["text"] == "Hello there.")
        self.assertTrue(row["models_agree"])

    def test_correct_speaker_starts_blank(self):
        rows = build_sheet(self._runs(), size=10)
        self.assertTrue(all(r["correct_speaker"] == "" for r in rows))

    def test_sampling_is_reproducible(self):
        runs = {"m": [{"speaker": "X", "text": f"line {i}"} for i in range(200)]}
        first = build_sheet(runs, size=20, seed=3)
        second = build_sheet(runs, size=20, seed=3)
        self.assertEqual([r["text"] for r in first], [r["text"] for r in second])

    def test_no_runs_yields_no_rows(self):
        self.assertEqual(build_sheet({}, size=10), [])


class NeighbourContextTest(unittest.TestCase):
    """A line alone is often unanswerable - "Huh, what is it?" names nobody.
    The surrounding narration is what identifies the speaker.

    Searching the source text for the line was tried and abandoned: 35 of 50
    lines could not be located, and a short common line matched the wrong
    occurrence entirely."""

    ENTRIES = [
        {"speaker": "NARRATOR", "text": "Roxy turned away from the window."},
        {"speaker": "NARRATOR", "text": "She had been waiting for hours."},
        {"speaker": "ROXY", "text": "Huh, what is it?"},
        {"speaker": "NARRATOR", "text": "Rudeus looked up, startled by her tone."},
        {"speaker": "RUDEUS", "text": "Nothing important."},
    ]

    def test_context_surrounds_the_line(self):
        before, after = neighbour_context(self.ENTRIES, 2, window=2)
        self.assertIn("She had been waiting for hours.", before)
        self.assertIn("Rudeus looked up, startled by her tone.", after)

    def test_window_is_respected(self):
        before, after = neighbour_context(self.ENTRIES, 2, window=1)
        self.assertEqual(len(before), 1)
        self.assertEqual(len(after), 1)

    def test_start_of_book_has_no_leading_context(self):
        before, after = neighbour_context(self.ENTRIES, 0, window=3)
        self.assertEqual(before, [])
        self.assertTrue(after)

    def test_end_of_book_has_no_trailing_context(self):
        before, after = neighbour_context(self.ENTRIES, 4, window=3)
        self.assertTrue(before)
        self.assertEqual(after, [])

    def test_neighbour_speakers_are_not_exposed(self):
        # They are model output and may be wrong; showing them would bias the
        # judgement being asked for.
        before, after = neighbour_context(self.ENTRIES, 2, window=2)
        self.assertTrue(all(isinstance(item, str) for item in before + after))
        self.assertNotIn("RUDEUS", " ".join(after))

    def test_rows_carry_context(self):
        runs = {"m": self.ENTRIES}
        rows = build_sheet(runs, size=10, window=2)
        row = next(r for r in rows if r["text"] == "Huh, what is it?")
        self.assertIn("She had been waiting for hours.", row["context_before"])
        self.assertIn("Rudeus looked up, startled by her tone.", row["context_after"])


if __name__ == "__main__":
    unittest.main()
