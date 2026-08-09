import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from routers import scripts_library
from script_preflight import audit_script, audit_unicode_text
from script_repair import build_deterministic_repair
from speaker_repair import apply_speaker_selections, build_speaker_review
from content_repair import apply_content_selections, build_content_review


def _entry(text, speaker="Narrator", instruct="Read naturally"):
    return {"text": text, "speaker": speaker, "instruct": instruct}


class ScriptPreflightTests(unittest.TestCase):
    def test_content_review_is_selective_and_normalization_is_lossless(self):
        entries = [_entry("Copyright © Publisher", instruct="  Calm,   clear. "),
                   _entry("Chapter One", instruct="Dramatic and nuanced.")]
        review = build_content_review(entries)
        self.assertEqual([1], [item["entry_number"] for item in review["front_matter"]])
        self.assertEqual("Calm, clear.", review["direction_normalizations"][0]["suggested"])
        self.assertEqual(1, len(review["direction_normalizations"]))

    def test_content_apply_requires_expected_values_and_does_not_mutate(self):
        entries = [_entry("Copyright"), _entry("Story", instruct="Old direction")]
        repair = apply_content_selections(entries,
            [{"entry_number": 1, "expected_text": "Copyright"}],
            [{"entry_number": 2, "expected_instruct": "Old direction",
              "new_instruct": " New   direction "}])
        self.assertEqual("Copyright", entries[0]["text"])
        self.assertEqual(["Story"], [entry["text"] for entry in repair["entries"]])
        self.assertEqual("New direction", repair["entries"][0]["instruct"])
        with self.assertRaises(ValueError):
            apply_content_selections(entries,
                [{"entry_number": 1, "expected_text": "Changed"}], [])
    def test_speaker_review_only_suggests_strong_third_person_case(self):
        entries = [_entry("Subaru looked toward the door.", "SUBARU"),
                   _entry("She said she would return.", "SUBARU")]
        candidates = build_speaker_review(entries)
        self.assertEqual("NARRATOR", candidates[0]["suggested_speaker"])
        self.assertIsNone(candidates[1]["suggested_speaker"])
        self.assertEqual(2, candidates[0]["next"]["entry_number"])

    def test_speaker_apply_requires_expected_value_and_does_not_mutate(self):
        entries = [_entry("Subaru looked around.", "SUBARU")]
        repaired = apply_speaker_selections(entries, [{"entry_number": 1,
            "expected_speaker": "SUBARU", "new_speaker": "NARRATOR"}])
        self.assertEqual("SUBARU", entries[0]["speaker"])
        self.assertEqual("NARRATOR", repaired["entries"][0]["speaker"])
        with self.assertRaises(ValueError):
            apply_speaker_selections(entries, [{"entry_number": 1,
                "expected_speaker": "EMILIA", "new_speaker": "NARRATOR"}])
    def test_deterministic_repair_is_source_backed_and_does_not_mutate_input(self):
        entries = [_entry("Please take саге of it."), _entry("First repeated line."),
                   _entry("Second repeated line."), _entry("First repeated line."),
                   _entry("Second repeated line.")]
        original = json.loads(json.dumps(entries))
        source = "Please take саге of it. First repeated line. Second repeated line."

        repair = build_deterministic_repair(entries, source)

        self.assertEqual(original, entries)
        self.assertEqual("Please take care of it.", repair["entries"][0]["text"])
        self.assertEqual(3, len(repair["entries"]))
        self.assertEqual([], repair["unresolved"])

    def test_deterministic_repair_refuses_unproven_unicode_and_keeps_source_duplicates(self):
        """Nothing is deleted here, and only the unicode token blocks.

        The source repeats this block, so emitting it twice is faithful
        transcription rather than a defect. It used to be reported as
        `unresolved`, which rejects the chunk: grimgar03 opens with its title
        eight times and could not be generated at all, failing chunk 1 of 49
        after 17 attempts in which the model was correct every time.

        The conservatism that matters is unchanged - the entries are still
        returned untouched, because deleting text the source supports would
        corrupt what the model was asked to reproduce. Only the BLOCKING
        changed, from rejecting the chunk to recording the decision.
        """
        entries = [_entry("Unknown жук token."), _entry("Repeated long first line."),
                   _entry("Repeated long second line."), _entry("Repeated long first line."),
                   _entry("Repeated long second line.")]
        source = "Repeated long first line. Repeated long second line. " * 2

        repair = build_deterministic_repair(entries, source)

        self.assertEqual(entries, repair["entries"])
        self.assertEqual(1, len(repair["unresolved"]))
        self.assertEqual("unsupported_cyrillic_character",
                         repair["unresolved"][0]["reason"])
        self.assertIn("adjacent_duplicate_block_kept",
                      [c.get("type") for c in repair["changes"]])

    def test_empty_entry_becomes_explicit_pause_on_previous_spoken_entry(self):
        entries = [_entry("Spoken line."), _entry("", instruct="A beat of silence."),
                   _entry("Following line.")]

        repair = build_deterministic_repair(entries, "Spoken line. Following line.")

        self.assertEqual(2, len(repair["entries"]))
        self.assertEqual(1000, repair["entries"][0]["pause_after"])
        self.assertEqual("empty_entry_to_pause", repair["changes"][0]["type"])
        self.assertNotIn("pause_after", entries[0])

    def test_empty_entry_repair_refuses_to_overwrite_existing_pause(self):
        previous = _entry("Spoken line.")
        previous["pause_after"] = 750
        entries = [previous, _entry("")]

        repair = build_deterministic_repair(entries, "Spoken line.")

        self.assertEqual(entries, repair["entries"])
        self.assertEqual("previous_pause_already_set", repair["unresolved"][0]["reason"])

    def test_reports_empty_text_and_cyrillic_homoglyphs_as_blocking(self):
        entries = [_entry(""), _entry("The саге was quiet.")]

        report = audit_script(entries)

        self.assertEqual(2, report["counts"]["blocking"])
        self.assertFalse(report["can_apply_repairs"])
        self.assertEqual(
            {"empty_text", "introduced_unicode_script"},
            {finding["code"] for finding in report["findings"]},
        )

    def test_nonprose_risk_is_manual_and_validation_state_is_explicit(self):
        report = audit_script([
            _entry("Reference AB-1234-Z"),
            _entry("Visit https://example.com/help"),
            _entry("One • Two • Three"),
            _entry("She arrived on 12 August 2026."),
        ])
        findings = [item for item in report["findings"]
                    if item["code"] == "nonprose_speech_risk"]
        self.assertEqual([1, 2, 3], [item["entry_numbers"][0] for item in findings])
        self.assertEqual(0, report["counts"]["blocking"])
        for finding in findings:
            validation = finding["details"]["validation"]
            self.assertTrue(validation["recommended"])
            self.assertEqual("not_run", validation["status"])
        identifier = findings[0]["details"]
        self.assertEqual("Reference AB-1234-Z.", identifier["normalized_preview"])
        self.assertEqual([], identifier["transformations"])

    def test_legitimate_japanese_is_allowed_when_source_backed(self):
        report = audit_script([_entry("彼はありがとうと言った。")], "彼はありがとうと言った。")
        self.assertEqual(0, report["counts"]["blocking"])

    def test_introduced_hiragana_and_mixed_word_are_reported(self):
        report = audit_script([_entry("Subあru saw a cat.")], "Subaru saw a cat.")
        codes = {finding["code"] for finding in report["findings"]}
        self.assertIn("introduced_unicode_script", codes)
        self.assertIn("mixed_script_word", codes)

    def test_unicode_audit_reports_nfc_and_unsafe_characters(self):
        report = audit_unicode_text("Cafe\u0301\ufffd\x00")
        self.assertFalse(report["is_nfc"])
        self.assertEqual(1, report["replacement_character_count"])
        self.assertEqual(["U+0000"], report["unsafe_controls"])

    def test_reports_adjacent_multi_entry_duplicate_with_source_evidence(self):
        block = [_entry("A sufficiently long first line."), _entry("A sufficiently long second line.")]
        source = "A sufficiently long first line. A sufficiently long second line."

        report = audit_script(block + block, source)

        duplicate = next(f for f in report["findings"] if f["code"] == "adjacent_duplicate_block")
        self.assertEqual([1, 2, 3, 4], duplicate["entry_numbers"])
        self.assertEqual(1, duplicate["details"]["source_occurrences"])

    def test_does_not_flag_non_adjacent_or_short_repetition_as_duplicate_block(self):
        entries = [_entry("Yes."), _entry("Yes."), _entry("Long repeated phrase here."),
                   _entry("Intervening phrase here."), _entry("Long repeated phrase here.")]

        report = audit_script(entries)

        self.assertNotIn("adjacent_duplicate_block", {f["code"] for f in report["findings"]})

    def test_flags_paraphrased_adjacent_pair_as_non_blocking_manual_review(self):
        first = "The old sailor walked slowly down the narrow creaking wooden pier before dawn."
        second = "The old sailor walked slowly down the narrow creaking wooden dock before dawn."

        report = audit_script([_entry(first), _entry(second)], source_text=first)

        near_dup = next(f for f in report["findings"] if f["code"] == "adjacent_near_duplicate")
        self.assertEqual([1, 2], near_dup["entry_numbers"])
        self.assertEqual("manual_review", near_dup["severity"])
        self.assertGreaterEqual(near_dup["details"]["similarity"], 0.90)
        self.assertTrue(report["can_apply_repairs"])

    def test_flags_identical_adjacent_pair_too_short_for_block_detector(self):
        line = "A repeated sentence of decent length here."

        report = audit_script([_entry(line), _entry(line)], source_text=line)

        near_dup = next(f for f in report["findings"] if f["code"] == "adjacent_near_duplicate")
        self.assertEqual([1, 2], near_dup["entry_numbers"])
        self.assertEqual(1.0, near_dup["details"]["similarity"])

    def test_does_not_flag_near_duplicate_when_both_sides_are_source_backed(self):
        line = "The captain repeated the same warning twice."
        source = f"{line} {line}"

        report = audit_script([_entry(line), _entry(line)], source_text=source)

        self.assertNotIn("adjacent_near_duplicate", {f["code"] for f in report["findings"]})

    def test_does_not_flag_short_echo_as_near_duplicate(self):
        report = audit_script([_entry("No."), _entry("No.")])

        self.assertNotIn("adjacent_near_duplicate", {f["code"] for f in report["findings"]})

    def test_near_duplicate_skips_entries_already_covered_by_exact_block(self):
        block = [_entry("A sufficiently long first line."), _entry("A sufficiently long second line.")]
        source = "A sufficiently long first line. A sufficiently long second line."

        report = audit_script(block + block, source)

        codes = {f["code"] for f in report["findings"]}
        self.assertIn("adjacent_duplicate_block", codes)
        self.assertNotIn("adjacent_near_duplicate", codes)

    def test_does_not_flag_distinct_adjacent_entries_as_near_duplicate(self):
        entries = [_entry("The weather in the mountains was cold and clear this morning."),
                   _entry("My favorite recipe requires flour sugar eggs and melted butter today.")]

        report = audit_script(entries)

        self.assertNotIn("adjacent_near_duplicate", {f["code"] for f in report["findings"]})

    def test_audit_script_surfaces_near_duplicate_alongside_other_manual_review_findings(self):
        first = "The old sailor walked slowly down the narrow creaking wooden pier before dawn."
        second = "The old sailor walked slowly down the narrow creaking wooden dock before dawn."
        entries = [_entry(first), _entry(second), _entry("Some other line entirely.", instruct="")]

        report = audit_script(entries, source_text=first)

        codes = {f["code"] for f in report["findings"]}
        self.assertIn("adjacent_near_duplicate", codes)
        self.assertIn("missing_instruction", codes)
        near_dup = next(f for f in report["findings"] if f["code"] == "adjacent_near_duplicate")
        self.assertEqual([1, 2], near_dup["entry_numbers"])
        self.assertGreaterEqual(report["counts"]["manual_review"], 2)

    def test_reuses_generic_speaker_policy_and_only_reports_narration_for_character(self):
        entries = [_entry("Subaru looked toward the door.", "Guard 2"),
                   _entry("Subaru looked toward the door.", "Narrator")]

        report = audit_script(entries, is_generic_speaker_fn=lambda name: name == "Guard 2")

        generic = [f for f in report["findings"] if f["code"] == "generic_speaker"]
        narration = [f for f in report["findings"] if f["code"] == "possible_misattributed_narration"]
        self.assertEqual([1], generic[0]["entry_numbers"])
        self.assertEqual([1], narration[0]["entry_numbers"])

    def test_endpoint_reads_saved_script_and_source_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            scripts = Path(tmp, "scripts")
            uploads = Path(tmp, "uploads")
            scripts.mkdir()
            uploads.mkdir()
            script_path = scripts / "book.json"
            source_path = uploads / "book.txt"
            original = json.dumps([_entry("The саге was quiet.")])
            script_path.write_text(original, encoding="utf-8")
            source_path.write_text("The sage was quiet.", encoding="utf-8")

            with patch.object(scripts_library, "SCRIPTS_DIR", str(scripts)), \
                 patch.object(scripts_library, "UPLOADS_DIR", str(uploads)):
                report = asyncio.run(scripts_library.preflight_saved_script(
                    "book", scripts_library.ScriptPreflightRequest(source_filename="book.txt")))

            self.assertEqual(1, report["counts"]["blocking"])
            self.assertEqual(original, script_path.read_text(encoding="utf-8"))

    def test_endpoint_rejects_non_text_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            scripts = Path(tmp, "scripts")
            scripts.mkdir()
            (scripts / "book.json").write_text("[]", encoding="utf-8")
            with patch.object(scripts_library, "SCRIPTS_DIR", str(scripts)):
                with self.assertRaises(HTTPException) as raised:
                    asyncio.run(scripts_library.preflight_saved_script(
                        "book", scripts_library.ScriptPreflightRequest(source_filename="book.epub")))
            self.assertEqual(400, raised.exception.status_code)

    def test_apply_requires_current_preview_and_creates_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            scripts = Path(tmp, "scripts")
            uploads = Path(tmp, "uploads")
            scripts.mkdir()
            uploads.mkdir()
            script_path = scripts / "book.json"
            script_path.write_text(json.dumps([_entry("Take саге.")]), encoding="utf-8")
            (uploads / "book.txt").write_text("Take саге.", encoding="utf-8")
            request = scripts_library.ScriptRepairRequest(source_filename="book.txt")

            with patch.object(scripts_library, "SCRIPTS_DIR", str(scripts)), \
                 patch.object(scripts_library, "UPLOADS_DIR", str(uploads)):
                preview = asyncio.run(scripts_library.preview_deterministic_repair("book", request))
                stale = scripts_library.ScriptRepairRequest(
                    source_filename="book.txt", expected_sha256="0" * 64)
                with self.assertRaises(HTTPException) as raised:
                    asyncio.run(scripts_library.apply_deterministic_repair("book", stale))
                apply_request = scripts_library.ScriptRepairRequest(
                    source_filename="book.txt", expected_sha256=preview["sha256"])
                result = asyncio.run(scripts_library.apply_deterministic_repair("book", apply_request))

            self.assertEqual(409, raised.exception.status_code)
            self.assertEqual("repaired", result["status"])
            self.assertEqual("Take care.", json.loads(script_path.read_text())[0]["text"])
            self.assertTrue((scripts / result["backup"]).exists())


if __name__ == "__main__":
    unittest.main()
