import json
from pathlib import Path
import tempfile
import unittest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import audit_experiment_artifacts as audit


class StructuralAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, document):
        path = self.root / name
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def test_classifies_all_identity_contracts_without_scientific_claims(self):
        self.write("supported.json", {
            "status": "complete", "rows": [],
            "provenance": {"seed": 7, "args": {"seed": 7},
                           "git": {"commit": "abc", "dirty": True,
                                   "harness_sha256": "0" * 64}}})
        self.write("provisional.json", {
            "rows": [{"arm": "x"}],
            "meta": {"git": {"commit": "def", "dirty": False}}})
        self.write("unidentified.json", {"rows": []})
        self.write("list.json", [])
        result = audit.build_audit(str(self.root))

        by_name = {row["artifact"]: row for row in result["artifacts"]}
        self.assertEqual("supported_structure",
                         by_name["supported.json"]["classification"])
        self.assertEqual("provisional",
                         by_name["provisional.json"]["classification"])
        self.assertEqual("exploratory",
                         by_name["unidentified.json"]["classification"])
        self.assertEqual("exploratory",
                         by_name["list.json"]["classification"])
        self.assertIn("do not validate scientific conclusions", result["scope"])

    def test_unreadable_json_is_visible_not_skipped(self):
        (self.root / "broken.json").write_text("{", encoding="utf-8")
        result = audit.build_audit(str(self.root))
        self.assertEqual(1, len(result["artifacts"]))
        self.assertIn("unreadable JSON", result["artifacts"][0]["reason"])


if __name__ == "__main__":
    unittest.main()
