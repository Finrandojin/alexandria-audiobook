from pass_quality import MIN_ORDERED_TRIGRAM_RECALL
import unittest
from pass_quality import (split_outer_quote_regions, validate_attribution,
                          validate_segment_quality)
from pass_quality import validate_attribution
from pass_quality import validate_instruct, index_head_check
import default_prompts


def _seg(text, type_="NARRATOR"):
    return {"type": type_, "text": text}


class SegmentQualityTests(unittest.TestCase):
    def test_chapter_title_in_source_credit_is_not_dialogue(self):
        source = ('Light Novel Adaptation found in Volume 9, Interlude '
                  '“To Each Their Vows”.')
        self.assertEqual(
            [{"type": "NARRATOR",
              "text": "Light Novel Adaptation found in Volume 9, Interlude "
                      "To Each Their Vows."}],
            split_outer_quote_regions(source))

    def test_complete_segment_passes(self):
        source = " ".join(f"word{i}" for i in range(50))
        report = validate_segment_quality(source, [_seg(source)])
        self.assertTrue(report["passed"], report["findings"])

    def test_missing_type_field_fails(self):
        report = validate_segment_quality("A line.", [{"text": "A line."}])
        codes = {f["code"] for f in report["findings"]}
        self.assertIn("missing_fields", codes)

    def test_invalid_type_value_fails(self):
        report = validate_segment_quality("A line.", [{"type": "MARCUS", "text": "A line."}])
        codes = {f["code"] for f in report["findings"]}
        self.assertIn("invalid_type", codes)

    def test_severe_truncation_fails_recall(self):
        source = " ".join(f"word{i}" for i in range(100))
        report = validate_segment_quality(source, [_seg("word0 word1")])
        codes = {f["code"] for f in report["findings"]}
        self.assertIn("low_source_token_recall", codes)

    def test_rejects_narration_combined_with_quoted_dialogue(self):
        source = '"Yes," Ilya said. "Stay behind me."'
        entries = [_seg("Yes,", "SPOKEN"),
                   _seg('Ilya said. "Stay behind me."', "NARRATOR")]
        codes = {f["code"] for f in validate_segment_quality(source, entries)["findings"]}
        self.assertIn("mixed_quote_region", codes)

    def test_rejects_attribution_tag_marked_spoken(self):
        source = '"No," Mara repeated, more firmly.'
        entries = [_seg("No,", "SPOKEN"),
                   _seg("Mara repeated, more firmly.", "SPOKEN")]
        codes = {f["code"] for f in validate_segment_quality(source, entries)["findings"]}
        self.assertIn("quote_region_misclassified", codes)

    def test_accepts_quote_aware_dialogue_and_attribution(self):
        source = '"Yes," Ilya said. "Stay behind me."'
        entries = [_seg("Yes,", "SPOKEN"), _seg("Ilya said.", "NARRATOR"),
                   _seg("Stay behind me.", "SPOKEN")]
        report = validate_segment_quality(source, entries)
        self.assertTrue(report["passed"], report["findings"])


class AttributionTests(unittest.TestCase):
    def test_attribution_passes_when_all_spoken_named(self):
        frozen = [{"type": "NARRATOR", "text": "The room was cold."},
                  {"type": "SPOKEN", "text": "Tell me."}]
        resp = [{"n": 0, "head": "The room was", "speaker": "NARRATOR"},
                {"n": 1, "head": "Tell me.", "speaker": "ELENA"}]
        report = validate_attribution(frozen, resp)
        self.assertTrue(report["passed"], report["findings"])

    def test_attribution_passes_when_response_reordered_by_index(self):
        # Binding is by index, so a reordered response still validates + binds.
        frozen = [{"type": "NARRATOR", "text": "The room was cold."},
                  {"type": "SPOKEN", "text": "Tell me."}]
        resp = [{"n": 1, "head": "Tell me.", "speaker": "ELENA"},
                {"n": 0, "head": "The room was", "speaker": "NARRATOR"}]
        report = validate_attribution(frozen, resp)
        self.assertTrue(report["passed"], report["findings"])

    def test_attribution_fails_when_spoken_left_as_narrator(self):
        frozen = [{"type": "SPOKEN", "text": "Tell me."}]
        resp = [{"n": 0, "head": "Tell me.", "speaker": "NARRATOR"}]
        report = validate_attribution(frozen, resp)
        codes = {f["code"] for f in report["findings"]}
        self.assertIn("spoken_not_named", codes)

    def test_attribution_ignores_legacy_head_when_index_is_valid(self):
        frozen = [{"type": "SPOKEN", "text": "Tell me."}]
        resp = [{"n": 0, "head": "Something else entirely", "speaker": "ELENA"}]
        report = validate_attribution(frozen, resp)
        self.assertTrue(report["passed"])

    def test_attribution_fails_on_duplicate_or_missing_index(self):
        frozen = [{"type": "SPOKEN", "text": "Tell me."},
                  {"type": "SPOKEN", "text": "Now go."}]
        resp = [{"n": 0, "head": "Tell me.", "speaker": "ELENA"},
                {"n": 0, "head": "Tell me.", "speaker": "ELENA"}]  # dup 0, index 1 missing
        report = validate_attribution(frozen, resp)
        codes = {f["code"] for f in report["findings"]}
        self.assertIn("alignment_violated", codes)


class IndexHeadCheckTests(unittest.TestCase):
    def test_reproduces_gemma_drift_survives(self):
        # The exact class of failure that crashed the run: the model would drift
        # on a long line's full text. Under the new contract it only echoes the
        # head + speaker, so a correct head/speaker validates regardless of what
        # it would have done to the body.
        frozen = [{"type": "NARRATOR",
                   "text": "The strength in those clinging fingers was weak, and not "
                           "even Beatrice knew what she was trying to do with this touch."}]
        resp = [{"n": 0, "head": "The strength in those", "speaker": "NARRATOR"}]
        ok, reason, ordered = index_head_check(frozen, resp)
        self.assertTrue(ok, reason)
        self.assertEqual(0, ordered[0]["n"])

    def test_punctuation_only_line_needs_no_head(self):
        frozen = [{"type": "SPOKEN", "text": "――――."}]
        ok, _, _ = index_head_check(frozen, [{"n": 0, "head": "", "speaker": "RYUZU"}])
        self.assertTrue(ok)

    def test_head_is_ignored_when_immutable_indexes_are_complete(self):
        frozen = [{"type": "SPOKEN", "text": "The old door opened."},
                  {"type": "SPOKEN", "text": "The old guard waited."}]
        short = [{"n": 0, "head": "The", "speaker": "A"},
                 {"n": 1, "head": "The old guard", "speaker": "B"}]
        ambiguous = [{"n": 0, "head": "The old", "speaker": "A"},
                     {"n": 1, "head": "The old guard", "speaker": "B"}]
        self.assertTrue(index_head_check(frozen, short)[0])
        self.assertTrue(index_head_check(frozen, ambiguous)[0])

    def test_rejects_duplicate_or_missing_immutable_indexes(self):
        frozen = [{"type": "SPOKEN", "text": "A"},
                  {"type": "SPOKEN", "text": "B"}]
        response = [{"n": 0, "speaker": "A"}, {"n": 0, "speaker": "B"}]
        self.assertFalse(index_head_check(frozen, response)[0])

    def test_accepts_integral_float_index(self):
        frozen = [{"type": "SPOKEN", "text": "Tell me now."}]
        ok, reason, _ = index_head_check(
            frozen, [{"n": 0.0, "head": "Tell me now", "speaker": "ELENA"}])
        self.assertTrue(ok, reason)

    def test_non_string_model_fields_fail_validation_without_crashing(self):
        frozen = [{"type": "SPOKEN", "text": "Tell me now."}]
        attribution = validate_attribution(
            frozen, [{"n": 0, "head": "Tell me now", "speaker": {"name": "ELENA"}}])
        instruct = validate_instruct(
            [{"speaker": "ELENA", "text": "Tell me now."}],
            [{"n": 0, "head": "Tell me now", "instruct": ["firm"]}])
        self.assertFalse(attribution["passed"])
        self.assertFalse(instruct["passed"])


class InstructValidatorTests(unittest.TestCase):
    def test_passes_when_all_have_instruct(self):
        prior = [{"speaker": "ELENA", "text": "Tell me."}]
        resp = [{"n": 0, "head": "Tell me.", "instruct": "Firm, quiet."}]
        report = validate_instruct(prior, resp)
        self.assertTrue(report["passed"], report["findings"])

    def test_fails_when_instruct_missing_or_empty(self):
        prior = [{"speaker": "ELENA", "text": "Tell me."}]
        resp = [{"n": 0, "head": "Tell me.", "instruct": "  "}]
        report = validate_instruct(prior, resp)
        codes = {f["code"] for f in report["findings"]}
        self.assertIn("missing_instruct", codes)

    def test_ignores_legacy_head_when_index_is_valid(self):
        prior = [{"speaker": "ELENA", "text": "Tell me."}]
        resp = [{"n": 0, "head": "Completely different words", "instruct": "Firm."}]
        report = validate_instruct(prior, resp)
        self.assertTrue(report["passed"])


class PromptLoaderTests(unittest.TestCase):
    def test_three_pass_prompts_load_and_have_placeholders(self):
        seg_sys, seg_usr = default_prompts.load_segment_prompts()
        self.assertIn("{chunk}", seg_usr)
        self.assertTrue(seg_sys.strip())
        att_sys, att_usr = default_prompts.load_attribute_prompts()
        self.assertIn("{batch}", att_usr)
        self.assertIn("{roster}", att_usr)
        ins_sys, ins_usr = default_prompts.load_instruct_prompts()
        self.assertIn("{batch}", ins_usr)


class AttributionRejectsTypeAsSpeakerTest(unittest.TestCase):
    """The attribution prompt shows each entry as {"n", "type", "text"}, and the
    model sometimes echoes the type back as the speaker. The gate required a
    SPOKEN span's speaker to be non-empty and not NARRATOR, which "SPOKEN"
    satisfies - so it passed. Measured live: 326 entries in one book, and the
    leak appeared in 8 of 9 books overnight."""

    def test_literal_type_name_is_rejected(self):
        frozen = [{"type": "SPOKEN", "text": "Oh-ho."}]
        report = validate_attribution(frozen, [{"n": 0, "head": "Oh-ho.",
                                                "speaker": "SPOKEN"}])
        self.assertFalse(report["passed"])
        self.assertTrue(any("type" in f.get("code", "")
                            or "type" in f.get("message", "").lower()
                            for f in report["findings"]))

    def test_rejection_is_case_insensitive(self):
        frozen = [{"type": "SPOKEN", "text": "Oh-ho."}]
        report = validate_attribution(frozen, [{"n": 0, "head": "Oh-ho.",
                                                "speaker": "spoken"}])
        self.assertFalse(report["passed"])

    def test_a_real_name_still_passes(self):
        frozen = [{"type": "SPOKEN", "text": "Oh-ho."}]
        report = validate_attribution(frozen, [{"n": 0, "head": "Oh-ho.",
                                                "speaker": "SHIRON"}])
        self.assertTrue(report["passed"], report["findings"])

    def test_unknown_placeholder_is_still_allowed(self):
        # UNKNOWN is the legitimate placeholder stabilize_speaker_identities
        # assigns; it must not be swept up with the type-name rejection.
        frozen = [{"type": "SPOKEN", "text": "Oh-ho."}]
        report = validate_attribution(frozen, [{"n": 0, "head": "Oh-ho.",
                                                "speaker": "UNKNOWN"}])
        self.assertTrue(report["passed"], report["findings"])


if __name__ == "__main__":
    unittest.main()


class GateParityAndMessagesTests(unittest.TestCase):
    def test_segment_gate_rejects_introduced_cyrillic(self):
        src = "the quick brown fox jumps over the lazy dog today"
        rep = validate_segment_quality(src, [{"type": "NARRATOR", "text": src.replace("fox", "fох")}])
        self.assertIn("unsupported_cyrillic", {f["code"] for f in rep["findings"]})

    def test_segment_findings_carry_message(self):
        rep = validate_segment_quality("a b c", [{"type": "MARCUS", "text": "a b c"}])
        for f in rep["findings"]:
            self.assertIn("message", f, f"finding {f['code']} lacks a message")

    def test_attribution_and_instruct_findings_carry_message(self):
        att = validate_attribution([{"type": "SPOKEN", "text": "Hi."}],
                                   [{"n": 0, "head": "Hi.", "speaker": "NARRATOR"}])
        ins = validate_instruct([{"speaker": "X", "text": "Hi."}],
                                [{"n": 0, "head": "Hi.", "instruct": ""}])
        for f in att["findings"] + ins["findings"]:
            self.assertIn("message", f)

    def test_segment_thresholds_are_shared_with_chunk_quality(self):
        import chunk_quality
        self.assertIs(MIN_ORDERED_TRIGRAM_RECALL, chunk_quality.MIN_ORDERED_TRIGRAM_RECALL)

    def test_introduced_unicode_finding_uses_dict_shape(self):
        # Must match chunk_quality's shape: characters are dicts with
        # character/codepoint/name, not bare "U+XXXX" strings, so consumers
        # that read finding["characters"][0]["character"] don't KeyError.
        src = "the quick brown fox jumps over the lazy dog today"
        rep = validate_segment_quality(src, [{"type": "NARRATOR", "text": src + " café"}])
        uni = [f for f in rep["findings"] if f["code"] == "unsupported_unicode_character"]
        self.assertTrue(uni, "expected an unsupported_unicode_character finding")
        char = uni[0]["characters"][0]
        self.assertEqual({"character", "codepoint", "name"}, set(char.keys()))
        self.assertEqual("é", char["character"])
