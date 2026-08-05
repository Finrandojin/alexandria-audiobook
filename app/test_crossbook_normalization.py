import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.crossbook_normalization import (
    load_locked_samples, normalize_date, normalize_identifier, summarize)


class CrossbookNormalizationTests(unittest.TestCase):
    def test_locked_fixture_has_two_categories_across_three_books(self):
        samples, rule = load_locked_samples()
        self.assertTrue(rule)
        self.assertEqual(6, len(samples))
        self.assertEqual(3, len({row["book"] for row in samples}))
        self.assertEqual({"identifier", "date"},
                         {row["category"] for row in samples})
        self.assertTrue(all(len(row["source_sha256"]) == 64 for row in samples))

    def test_identifiers_are_read_digitwise(self):
        self.assertEqual("e book number eight six three",
                         normalize_identifier("eBook-No. 863"))
        with self.assertRaisesRegex(ValueError, "unsupported identifier"):
            normalize_identifier("ISBN 863")

    def test_dates_have_explicit_ordinal_and_year(self):
        self.assertEqual("release date June twenty seventh, two thousand eight",
                         normalize_date("Release Date: Jun 27, 2008"))
        with self.assertRaisesRegex(ValueError, "unsupported date"):
            normalize_date("27/06/2008")

    def test_summary_keeps_books_categories_and_arms_separate(self):
        rows = []
        for book in ("a", "b", "c"):
            for category in ("identifier", "date"):
                for arm in ("raw", "normalized"):
                    rows.append({"book": book, "category": category, "arm": arm,
                                 "words": 2, "errors": arm == "raw",
                                 "failed": False})
        result = summarize(rows)
        self.assertEqual(12, len(result))
        self.assertTrue(all(row["n"] == 1 for row in result))


if __name__ == "__main__":
    unittest.main()
