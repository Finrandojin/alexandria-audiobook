import unittest

from build_scoring_sheet import build_sheet


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


if __name__ == "__main__":
    unittest.main()
