import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import audit_legacy_attribution as audit


class LegacyAttributionAuditTests(unittest.TestCase):
    def test_current_inventory_is_exhaustive_and_semantically_bounded(self):
        result = audit.build_audit()
        rows = result["artifacts"]
        self.assertEqual(112, len(rows))
        self.assertEqual(112, len({row["artifact"] for row in rows}))
        self.assertEqual(112, sum(result["summary"].values()))
        self.assertTrue(all(row["semantic_limit"] for row in rows))
        self.assertTrue(any(row["classification"] == "historical_only"
                            for row in rows))

    def test_current_gold_rescore_exposes_changed_and_unmapped_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = {
                "entries": [{"id": "one", "expected_speaker": "ALICE"}],
                "aliases": [["ALICE", "AL"]],
            }
            path = Path(tmp, "gold.json")
            path.write_text(json.dumps(fixture), encoding="utf-8")
            rows = [
                {"id": "one", "expected": "BOB", "predicted": "AL",
                 "correct": False},
                {"id": "missing", "expected": "BOB", "predicted": "BOB",
                 "correct": True},
            ]
            meta = {"gold_path": "gold.json", "gold_sha256": "old",
                    "gold_lines": 2}
            with mock.patch.object(audit, "REPO", tmp):
                result = audit._current_gold(meta, rows)
            self.assertTrue(result["hash_changed"])
            self.assertTrue(result["line_count_changed"])
            self.assertEqual(1, result["expected_changed_rows"])
            self.assertEqual(1, result["correctness_changed_rows"])
            self.assertEqual(1, result["missing_rows"])

    def test_markdown_lists_every_artifact_once(self):
        result = audit.build_audit()
        rendered = audit.render_markdown(result)
        for row in result["artifacts"]:
            self.assertEqual(1, rendered.count(f"`{row['artifact']}`"))


if __name__ == "__main__":
    unittest.main()
