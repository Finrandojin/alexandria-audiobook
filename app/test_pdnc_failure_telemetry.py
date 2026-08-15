import tempfile
import unittest
from pathlib import Path

from experiments.pdnc_failure_telemetry import (
    classify_error, context_mentions, roster_match)
from experiments.pdnc_fixture import load_novel


class PdncFailureTelemetryTests(unittest.TestCase):
    def test_roster_and_alias_context_matching(self):
        groups = [{"ELIZABETH", "LIZZY", "MISS ELIZABETH"}]
        roster = ["MR. BENNET", "ELIZABETH"]
        self.assertTrue(roster_match("LIZZY", roster, groups))
        self.assertTrue(context_mentions(
            "ELIZABETH", "Lizzy replied quietly.", groups))
        self.assertFalse(context_mentions(
            "ELIZABETH", "Eliza replied quietly.", groups))

    def test_failure_class_precedence(self):
        self.assertEqual(classify_error(False, None, False, True),
                         "gold_missing_from_roster")
        self.assertEqual(classify_error(True, None, False, True),
                         "missing_prediction_or_batch_failure")
        self.assertEqual(classify_error(True, "OTHER", False, True),
                         "invalid_or_out_of_roster_prediction")
        self.assertEqual(classify_error(True, "OTHER", True, True),
                         "missed_explicit_context_evidence")
        self.assertEqual(classify_error(True, "OTHER", True, False),
                         "valid_candidate_selection_error")

    def test_load_novel_accepts_native_pdnc_layout(self):
        with tempfile.TemporaryDirectory() as folder:
            novel = Path(folder) / "Book"
            novel.mkdir()
            (novel / "quotation_info.csv").write_text(
                "quoteText,speaker\nHello,Alice\n", encoding="utf-8")
            (novel / "character_info.csv").write_text(
                "Main Name,Aliases\nAlice,\n", encoding="utf-8")
            (novel / "novel_text.txt").write_text("Hello", encoding="utf-8")
            quotes, characters, text = load_novel(folder, "Book")
        self.assertEqual(quotes[0]["speaker"], "Alice")
        self.assertEqual(characters[0]["Main Name"], "Alice")
        self.assertEqual(text, "Hello")


if __name__ == "__main__":
    unittest.main()
