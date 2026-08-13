import unittest

from experiments.source_span_coverage import get_experiment_chunks
from generate_script import split_into_chunks
from source_span_coverage import (format_tagged_source, get_source_spans,
                                  get_span_coverage_findings)


class SourceSpanCoverageTests(unittest.TestCase):
    def test_experiment_uses_production_preprocessing(self):
        raw = "The story begins. " + "echo " * 20 + "The story continues."
        self.assertNotEqual(split_into_chunks(raw, 6000),
                            get_experiment_chunks(raw, 6000))
        self.assertLess(get_experiment_chunks(raw, 6000)[0].count("echo"), 20)

    def test_spans_keep_order_and_stable_ids(self):
        spans = get_source_spans("First sentence. Second?\n\nThird paragraph!")
        self.assertEqual(["S001", "S002", "S003"], [span["id"] for span in spans])
        self.assertEqual("[S001] First sentence.\n[S002] Second?\n[S003] Third paragraph!",
                         format_tagged_source(spans))

    def test_complete_declarations_pass(self):
        spans = get_source_spans("First. Second.")
        entries = [
            {"speaker": "NARRATOR", "text": "First.", "instruct": "",
             "source_span_ids": ["S001"]},
            {"speaker": "NARRATOR", "text": "Second.", "instruct": "",
             "source_span_ids": ["S002"]},
        ]
        self.assertEqual([], get_span_coverage_findings(spans, entries))

    def test_missing_unknown_and_malformed_declarations_fail_loudly(self):
        spans = get_source_spans("First. Second. Third.")
        entries = [
            {"source_span_ids": ["S001", "S999"]},
            {"source_span_ids": "S002"},
        ]
        findings = get_span_coverage_findings(spans, entries)
        self.assertEqual(
            {"invalid_source_span_ids", "unknown_source_span_ids", "uncovered_source_spans"},
            {finding["code"] for finding in findings})
        missing = next(item for item in findings if item["code"] == "uncovered_source_spans")
        self.assertEqual(["S002", "S003"], missing["span_ids"])


if __name__ == "__main__":
    unittest.main()
