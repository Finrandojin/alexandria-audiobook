import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from chunk_quality import validate_chunk_quality
import generate_script
from source_normalization import (normalize_extreme_phrase_repetitions,
                                  normalize_homoglyph_words,
                                  normalize_known_source_corruptions)


def _entry(text, speaker="NARRATOR", instruct="Read naturally."):
    return {"speaker": speaker, "text": text, "instruct": instruct}


def generate_script_test_client(entry_lists):
    from types import SimpleNamespace
    import json
    responses = iter(entry_lists)

    def create(**_kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=json.dumps(next(responses))),
            finish_reason="stop")], usage=None)

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


class ChunkQualityTests(unittest.TestCase):
    def test_adaptive_split_recombines_and_carries_context(self):
        source = ("First section " + "word " * 200 + ".\n\n" +
                  "Second section " + "word " * 200 + ".")
        first, second = generate_script.split_failed_chunk(source)
        first_entries = [_entry(first)]
        second_entries = [_entry(second)]
        with patch.object(generate_script, "process_chunk",
                          side_effect=[[], first_entries, second_entries]) as process:
            entries, split = generate_script.process_chunk_adaptively(
                object(), "model", source, 1, 1, generate_script.LLMGenParams(),
                previous_entries=[_entry("Earlier")])
        self.assertTrue(split)
        self.assertEqual(first_entries + second_entries, entries)
        self.assertEqual([_entry("Earlier")] + first_entries,
                         process.call_args_list[2].kwargs["previous_entries"])

    def test_adaptive_split_refuses_short_or_boundaryless_chunk(self):
        self.assertEqual([], generate_script.split_failed_chunk("short text"))
        self.assertEqual([], generate_script.split_failed_chunk("x" * 2000))

    def test_adaptive_split_handles_grimgar_sized_failure(self):
        source = "\n\n".join((label + " " + "word " * 70).strip()
                              for label in ("First", "Second", "Third", "Fourth"))

        first, second = generate_script.split_failed_chunk(source)

        self.assertGreaterEqual(len(first), 400)
        self.assertGreaterEqual(len(second), 400)
        self.assertEqual(source, first + "\n\n" + second)

    def test_retry_action_splits_only_repeated_severe_truncation(self):
        severe = {"metrics": {"output_source_ratio": 0.2}, "findings": [
            {"code": "low_source_token_recall"},
            {"code": "low_ordered_trigram_recall"},
        ]}
        mild = {"metrics": {"output_source_ratio": 0.8}, "findings": [
            {"code": "low_source_token_recall"},
            {"code": "low_ordered_trigram_recall"},
        ]}

        self.assertEqual("retry", generate_script.get_chunk_retry_action(
            severe, 1, allow_early_split=True))
        self.assertEqual("split", generate_script.get_chunk_retry_action(
            severe, 2, allow_early_split=True))
        self.assertEqual("retry", generate_script.get_chunk_retry_action(
            severe, 2, allow_early_split=False))
        self.assertEqual("retry", generate_script.get_chunk_retry_action(
            mild, 2, allow_early_split=True))

    def test_adaptive_split_still_attempts_second_half_after_first_half_fails(self):
        # Regression: a real overnight batch run lost 71/78 remaining chunks of
        # one book because the first split half failed and the second half was
        # never even attempted, even though live reproduction testing showed
        # this content has a real (measured ~40%) per-attempt success rate --
        # one unlucky sample on half 1 must not forfeit half 2's independent
        # chance. The whole chunk is still reported as failed (checkpoint/
        # resume requires every accepted chunk to be gapless), but part 2 must
        # have been genuinely tried.
        source = ("First section " + "word " * 200 + ".\n\n" +
                  "Second section " + "word " * 200 + ".")
        with patch.object(generate_script, "process_chunk",
                          side_effect=[[], [], [_entry("second, tried anyway")]]) as process:
            entries, split = generate_script.process_chunk_adaptively(
                object(), "model", source, 1, 1, generate_script.LLMGenParams(),
                previous_entries=[_entry("Earlier")])
        self.assertEqual([], entries)
        self.assertTrue(split)
        # original + part 1 + part 2 -- part 2 must have been called.
        self.assertEqual(3, process.call_count)
        self.assertEqual([_entry("Earlier")],
                         process.call_args_list[2].kwargs["previous_entries"])

    def test_adaptive_split_fails_cleanly_when_both_halves_fail(self):
        source = ("First section " + "word " * 200 + ".\n\n" +
                  "Second section " + "word " * 200 + ".")
        with patch.object(generate_script, "process_chunk",
                          side_effect=[[], [], []]) as process:
            entries, split = generate_script.process_chunk_adaptively(
                object(), "model", source, 1, 1, generate_script.LLMGenParams())
        self.assertEqual([], entries)
        self.assertTrue(split)
        self.assertEqual(3, process.call_count)

    def test_adaptive_split_recurses_when_a_half_itself_needs_splitting(self):
        # Regression for a live 2026-07-19 failure: a chunk that still fails
        # after being split in half must have ITS failing half split again,
        # not given up on -- the same "a smaller target has an independent
        # chance" logic that justifies splitting at all, applied recursively.
        section = lambda label: f"{label} intro. " + "Word word word. " * 120
        source = section("First") + "\n\n" + section("Second")
        part1, part2 = generate_script.split_failed_chunk(source)
        part1a, part1b = generate_script.split_failed_chunk(part1)
        part1a_entries = [_entry(part1a)]
        part1b_entries = [_entry(part1b)]
        part2_entries = [_entry(part2)]
        with patch.object(generate_script, "process_chunk", side_effect=[
                [],               # whole chunk fails
                [],               # part 1 (whole) fails
                part1a_entries,   # part 1's own first sub-split half succeeds
                part1b_entries,   # part 1's own second sub-split half succeeds
                part2_entries,    # part 2 (whole) succeeds, never needed a split
        ]) as process:
            entries, split = generate_script.process_chunk_adaptively(
                object(), "model", source, 1, 1, generate_script.LLMGenParams())
        self.assertTrue(split)
        self.assertEqual(part1a_entries + part1b_entries + part2_entries, entries)
        self.assertEqual(5, process.call_count)

    def test_book_request_preflight_uses_real_chunks_and_parallel_slots(self):
        report = generate_script.build_book_request_preflight(
            ["short text", "x" * 6000], "system", "{context}\n{chunk}",
            10000, 16384, 2)
        self.assertEqual(2, report["chunk_count"])
        self.assertEqual(8192, report["per_slot_context"])
        self.assertGreater(report["worst_predicted_tokens"], report["average_predicted_tokens"])
        self.assertEqual(report["worst_predicted_tokens"] * 3,
                         report["required_total_context"]["3"])

    def test_book_request_preflight_reports_context_miss(self):
        report = generate_script.build_book_request_preflight(
            ["x" * 6000], "s", "{context}{chunk}", 10000, 8192, 2)
        self.assertFalse(report["predicted_fits"])

    def test_final_gate_requires_whole_book_quality_and_zero_blockers(self):
        self.assertTrue(generate_script.passes_final_generation_gate(
            {"passed": True}, {"counts": {"blocking": 0}}))
        self.assertFalse(generate_script.passes_final_generation_gate(
            {"passed": False}, {"counts": {"blocking": 0}}))
        self.assertFalse(generate_script.passes_final_generation_gate(
            {"passed": True}, {"counts": {"blocking": 1}}))
        self.assertFalse(generate_script.passes_final_generation_gate(
            {"passed": True}, {"counts": {"blocking": 0}}, [{"reason": "unsafe"}]))

    def test_final_repair_removes_unique_duplicate_across_chunk_boundary(self):
        first = _entry("A sufficiently long first source line.")
        second = _entry("A sufficiently long second source line.")
        source = f"{first['text']} {second['text']}"

        repaired = generate_script.build_final_generation_repair(
            [first, second, copy.deepcopy(first), copy.deepcopy(second)], source)

        self.assertEqual([first, second], repaired["entries"])
        self.assertEqual("adjacent_duplicate_block", repaired["changes"][0]["type"])
        self.assertEqual([], repaired["unresolved"])

    def test_quality_manifest_summarizes_chunks_without_copying_entries(self):
        accepted = [{"chunk_number": 1, "source_sha256": "source",
                     "entries": [_entry("Text")], "quality": {"passed": True}}]
        manifest = generate_script.build_generation_quality_manifest(
            "verified", {"source_sha256": "book"}, accepted, [])
        self.assertEqual(1, manifest["accepted_chunk_count"])
        self.assertEqual(1, manifest["chunks"][0]["entry_count"])
        self.assertNotIn("entries", manifest["chunks"][0])
        self.assertEqual([], manifest["chunks"][0]["attempts"])
        # Near-miss telemetry defaults to False/0 when nothing was salvaged.
        self.assertFalse(manifest["chunks"][0]["near_miss_accepted"])
        self.assertEqual(0, manifest["near_miss_accepted_count"])

    def test_quality_manifest_persists_near_miss_accepted_chunks(self):
        accepted = [
            {"chunk_number": 1, "source_sha256": "a", "entries": [_entry("x")],
             "quality": {"passed": True}},
            {"chunk_number": 2, "source_sha256": "b", "entries": [_entry("y")],
             "quality": {"passed": False}, "near_miss_accepted": True},
        ]
        manifest = generate_script.build_generation_quality_manifest(
            "complete", {"source_sha256": "book"}, accepted, [])
        self.assertEqual(1, manifest["near_miss_accepted_count"])
        self.assertFalse(manifest["chunks"][0]["near_miss_accepted"])
        self.assertTrue(manifest["chunks"][1]["near_miss_accepted"])

    def test_adaptive_split_labels_full_and_split_attempt_telemetry(self):
        source = ("First section " + "word " * 200 + ".\n\n" +
                  "Second section " + "word " * 200 + ".")
        attempts = []

        def process(*_args, **kwargs):
            attempt = {"attempt": 1}
            kwargs["attempt_observer"](attempt)
            attempt["outcome"] = "quality_rejected"
            return []

        with patch.object(generate_script, "process_chunk", side_effect=process):
            generate_script.process_chunk_adaptively(
                object(), "model", source, 1, 1, generate_script.LLMGenParams(),
                attempt_observer=attempts.append)

        self.assertEqual(["full", "split", "split"],
                         [attempt["phase"] for attempt in attempts])
        self.assertEqual([1, 2], [attempt["split_part"] for attempt in attempts[1:]])
        self.assertTrue(all(attempt["outcome"] == "quality_rejected"
                            for attempt in attempts))

    def test_known_source_corruption_normalizes_with_location_evidence(self):
        original = "First line.\nTake саге now."

        normalized, changes = normalize_known_source_corruptions(original)

        self.assertEqual("First line.\nTake care now.", normalized)
        self.assertEqual("First line.\nTake саге now.", original)
        self.assertEqual((2, 6, "саге", "care"),
                         (changes[0]["line"], changes[0]["column"],
                          changes[0]["before"], changes[0]["after"]))

    def test_known_source_nap_corruption_is_normalized(self):
        normalized, changes = normalize_known_source_corruptions(
            "She intended to пар until their arrival.")

        self.assertEqual("She intended to nap until their arrival.", normalized)
        self.assertEqual("пар", changes[0]["before"])
        self.assertEqual("nap", changes[0]["after"])

    def test_inline_illustration_caption_is_removed_with_evidence(self):
        source = ("Narration before. Illustration from Volume 10, coloring by "
                  "Norvak (source) Subaru continued speaking.")
        normalized, changes = normalize_known_source_corruptions(source)
        self.assertEqual("Narration before. Subaru continued speaking.", normalized)
        self.assertEqual("illustration_caption", changes[0]["rule"])
        self.assertIn("Norvak", changes[0]["before"])

    def test_extreme_phrase_repetition_is_collapsed_with_evidence(self):
        source = "She cried, “" + " ".join(["to my someone"] * 43) + "!”"

        normalized, changes = normalize_extreme_phrase_repetitions(source)

        self.assertEqual(3, normalized.count("to my someone"))
        self.assertEqual(43, changes[0]["repetitions"])
        self.assertEqual(3, changes[0]["phrase_words"])
        self.assertEqual("extreme_phrase_repetition", changes[0]["rule"])
        self.assertTrue(normalized.endswith("someone…!”"))

    def test_normal_repetition_and_authorial_elongation_are_untouched(self):
        source = "No " * 13 + "deeeeeeeeeeeeeeeeeeep!"

        normalized, changes = normalize_extreme_phrase_repetitions(source)

        self.assertEqual(source, normalized)
        self.assertEqual([], changes)

    def test_homoglyph_word_normalizes_with_location_evidence(self):
        padding = "The narrator continued speaking calmly for a while. " * 10
        original = padding + "\nSubаru answered."  # Cyrillic а

        normalized, changes = normalize_homoglyph_words(original)

        self.assertEqual(padding + "\nSubaru answered.", normalized)
        self.assertEqual(1, len(changes))
        self.assertEqual(("Subаru", "Subaru", "homoglyph", 2, 1),
                         (changes[0]["before"], changes[0]["after"],
                          changes[0]["rule"], changes[0]["line"],
                          changes[0]["column"]))

    def test_homoglyph_all_cyrillic_lookalike_word_is_normalized(self):
        padding = "The narrator continued speaking calmly for a while. " * 40
        normalized, changes = normalize_homoglyph_words(
            padding + "He would саге for them.")  # all-Cyrillic lookalikes

        self.assertTrue(normalized.endswith("He would care for them."))
        self.assertEqual([("саге", "care")],
                         [(change["before"], change["after"]) for change in changes])

    def test_homoglyph_leaves_genuine_cyrillic_text_untouched(self):
        original = "Он сказал привет и ушёл домой рано утром."

        normalized, changes = normalize_homoglyph_words(original)

        self.assertEqual(original, normalized)
        self.assertEqual([], changes)

    def test_homoglyph_word_with_unmappable_character_is_untouched(self):
        padding = "The narrator continued speaking calmly for a while. " * 10
        original = padding + "Subжru answered."  # ж has no Latin homoglyph

        normalized, changes = normalize_homoglyph_words(original)

        self.assertEqual(original, normalized)
        self.assertEqual([], changes)

    def test_strip_known_front_matter_removes_manifesto_and_toc(self):
        # Regression for a live 2026-07-19 failure: every "wn" upload in the
        # corpus opens with this fan-compiler's translator's-note essay and
        # chapter listing, which isn't dialogue or narration -- the
        # annotation model can't handle it and chunk 1 failed near-zero
        # recall no matter how many times it was retried or split. Confirmed
        # across 5 real "wn" files that the first "Original ... Chapter -
        # Complete." / "Original Translation by ..." marker pair is always
        # immediately followed by the real chapter 1 prose.
        source = (
            "﻿Manifesto.\n\n"
            "Hello, this is a translator. Some notes about this project.\n\n"
            "P.S. Much thanks to Joy Kirbs from Discord for aiding me in redesigning the covers.\n\n"
            "Table of Contents for Volume 1.\n\n"
            "Chapter 1 - The Beginning.\n\n"
            "Original Web Novel Chapter ― Complete.\n\n"
            "Original Translation by Someone ― Complete.\n\n"
            "The dragon carriage rattles along quietly as it continues down the highway."
        )

        stripped, removed = generate_script.strip_known_front_matter(source)

        self.assertEqual(
            "The dragon carriage rattles along quietly as it continues down the highway.",
            stripped)
        self.assertGreater(removed["removed_chars"], 0)
        self.assertTrue(source.startswith("﻿Manifesto."))

    def test_strip_known_front_matter_leaves_normal_books_untouched(self):
        source = "The dragon carriage rattles along quietly as it continues down the highway."

        stripped, removed = generate_script.strip_known_front_matter(source)

        self.assertEqual(source, stripped)
        self.assertIsNone(removed)

    def test_strip_known_front_matter_leaves_unrecognized_manifesto_shape_untouched(self):
        source = "Manifesto.\n\nSome other kind of front matter with no known anchor.\n\nStory text."

        stripped, removed = generate_script.strip_known_front_matter(source)

        self.assertEqual(source, stripped)
        self.assertIsNone(removed)

    def test_generation_repairs_empty_silence_before_quality_acceptance(self):
        source = "First spoken line. Following spoken line."
        response_entries = [
            _entry("First spoken line."),
            _entry("", instruct="A beat of silence."),
            _entry("Following spoken line."),
        ]
        client = generate_script_test_client([response_entries])
        params = generate_script.LLMGenParams("system", "{chunk}")

        result = generate_script.process_chunk(client, "model", source, 1, 1, params)

        self.assertEqual(2, len(result))
        self.assertEqual(1000, result[0]["pause_after"])

    def test_generation_checkpoint_roundtrip_requires_validated_matching_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = str(Path(tmp, "book.json"))
            params = generate_script.LLMGenParams("system", "{chunk}")
            chunks = ["First source chunk.", "Second source chunk."]
            fingerprint = generate_script.get_generation_fingerprint(
                "\n".join(chunks), chunks, "model", "http://local", params, 6000)
            accepted = [{
                "chunk_number": 1,
                "source_sha256": fingerprint["chunk_sha256"][0],
                "entries": [_entry(chunks[0])],
                "quality": {"passed": True},
            }]

            generate_script.save_generation_checkpoint(output, fingerprint, accepted)

            self.assertEqual(accepted, generate_script.load_generation_checkpoint(output, fingerprint))
            changed = dict(fingerprint, source_sha256="changed")
            self.assertEqual([], generate_script.load_generation_checkpoint(output, changed))
            generate_script.clear_generation_checkpoint(output)
            self.assertFalse(Path(generate_script.get_generation_checkpoint_path(output)).exists())

    def test_generation_checkpoint_rejects_unvalidated_chunk(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = str(Path(tmp, "book.json"))
            params = generate_script.LLMGenParams("system", "{chunk}")
            fingerprint = generate_script.get_generation_fingerprint(
                "source", ["source"], "model", "http://local", params, 6000)
            rejected = [{"chunk_number": 1,
                         "source_sha256": fingerprint["chunk_sha256"][0],
                         "entries": [_entry("source")], "quality": {"passed": False}}]
            generate_script.save_generation_checkpoint(output, fingerprint, rejected)

            self.assertEqual([], generate_script.load_generation_checkpoint(output, fingerprint))

    def test_complete_resegmented_response_passes_without_mutation(self):
        source = "One complete sentence appears here. Another sentence follows it."
        entries = [_entry("One complete sentence appears here."),
                   _entry("Another sentence follows it.")]
        original = copy.deepcopy(entries)

        report = validate_chunk_quality(source, entries)

        self.assertTrue(report["passed"])
        self.assertEqual(1.0, report["metrics"]["source_token_recall"])
        self.assertEqual(original, entries)

    def test_volume_10_style_early_stop_fails_all_coverage_signals(self):
        source = " ".join(f"sourceword{index}" for index in range(1000))
        entries = [_entry(" ".join(f"sourceword{index}" for index in range(100)))]

        report = validate_chunk_quality(source, entries)

        self.assertFalse(report["passed"])
        self.assertEqual(
            {"low_source_token_recall", "low_ordered_trigram_recall", "output_source_ratio"},
            {finding["code"] for finding in report["findings"]},
        )

    def test_boundary_allows_lowest_calibrated_intact_coverage(self):
        source_words = [f"word{index}" for index in range(100)]
        source = " ".join(source_words)
        entries = [_entry(" ".join(source_words[:93]))]

        report = validate_chunk_quality(source, entries)

        self.assertTrue(report["passed"])

    def test_malformed_and_empty_entries_fail_structural_checks(self):
        report = validate_chunk_quality("A spoken line.", [
            {"speaker": "NARRATOR", "text": ""}, "bad entry",
        ])

        codes = {finding["code"] for finding in report["findings"]}
        self.assertIn("missing_fields", codes)
        self.assertIn("empty_text", codes)
        self.assertIn("invalid_entry", codes)

    def test_source_cyrillic_is_reported_but_only_new_cyrillic_blocks(self):
        copied = validate_chunk_quality("Take саге now.", [_entry("Take саге now.")])
        introduced = validate_chunk_quality("Take care now.", [_entry("Take саге now.")])

        self.assertTrue(copied["passed"])
        self.assertEqual(["а", "г", "е", "с"], copied["source_cyrillic"])
        self.assertIn("unsupported_cyrillic", {item["code"] for item in introduced["findings"]})

    def test_source_unsupported_adjacent_block_is_explicitly_reported(self):
        block = [_entry("A sufficiently long first line."),
                 _entry("A sufficiently long second line.")]
        source = "A sufficiently long first line. A sufficiently long second line."

        report = validate_chunk_quality(source, block + block)

        self.assertIn("source_unsupported_duplicate",
                      {item["code"] for item in report["findings"]})

    def test_complete_japanese_text_uses_character_units(self):
        source = "彼はありがとうと言った"
        report = validate_chunk_quality(source, [_entry(source)])
        self.assertTrue(report["passed"])
        self.assertGreater(report["metrics"]["source_tokens"], 3)

    def test_missing_japanese_span_fails_recall(self):
        source = "彼はありがとうと言った"
        report = validate_chunk_quality(source, [_entry("彼は言った")])
        self.assertFalse(report["passed"])
        self.assertIn("low_source_token_recall", {item["code"] for item in report["findings"]})

    def test_introduced_hiragana_reports_codepoint(self):
        report = validate_chunk_quality("Subaru saw a cat", [_entry("Subあru saw a cat")])
        finding = next(item for item in report["findings"]
                       if item["code"] == "unsupported_unicode_character")
        self.assertEqual("U+3042", finding["characters"][0]["codepoint"])

    def test_canonically_equivalent_accents_compare_equal(self):
        report = validate_chunk_quality("Caf\u00e9", [_entry("Cafe\u0301")])
        self.assertTrue(report["passed"])

    def test_dropped_middle_paragraph_reports_missing_span(self):
        first = [f"alpha{index}" for index in range(50)]
        dropped = [f"bravo{index}" for index in range(40)]
        last = [f"charlie{index}" for index in range(50)]
        source = " ".join(first + dropped + last)
        entries = [_entry(" ".join(first)), _entry(" ".join(last))]

        report = validate_chunk_quality(source, entries)

        self.assertFalse(report["passed"])
        span = report["missing_source_spans"][0]
        self.assertEqual(50, span["start_token"])
        self.assertEqual(40, span["token_count"])
        self.assertEqual(" ".join(dropped[:12]), span["preview"])

    def test_passing_report_has_no_missing_spans(self):
        source = "One complete sentence appears here. Another sentence follows it."
        report = validate_chunk_quality(source, [_entry(source)])
        self.assertTrue(report["passed"])
        self.assertEqual([], report["missing_source_spans"])

    def test_missing_japanese_span_reports_character_preview(self):
        report = validate_chunk_quality("\u5f7c\u306f\u3042\u308a\u304c\u3068\u3046\u3068\u8a00\u3063\u305f", [_entry("\u5f7c\u306f\u8a00\u3063\u305f")])
        self.assertFalse(report["passed"])
        span = report["missing_source_spans"][0]
        self.assertEqual(6, span["token_count"])
        self.assertEqual("\u3042 \u308a \u304c \u3068 \u3046 \u3068", span["preview"])


def _trigram_near_miss_text(word_count=100, swap_stride=25):
    """Reproduce every source word (unigram recall 1.0, ratio 1.0) but swap a
    few adjacent pairs so ordered-trigram recall lands in the near-miss band
    [ACCEPT_TRIGRAM_NEAR_MISS_FLOOR, MIN_ORDERED_TRIGRAM_RECALL). Deterministic."""
    words = [f"word{index}" for index in range(word_count)]
    index = 0
    while index + 1 < word_count:
        words[index], words[index + 1] = words[index + 1], words[index]
        index += swap_stride
    return " ".join(f"word{index}" for index in range(word_count)), " ".join(words)


def _always_returns(payload):
    from types import SimpleNamespace
    import json
    content = json.dumps(payload)

    def create(**_kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=content), finish_reason="stop")], usage=None)

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


class TrigramNearMissAcceptanceTests(unittest.TestCase):
    def test_predicate_accepts_only_trigram_only_failure_at_or_above_floor(self):
        from chunk_quality import (is_trigram_only_near_miss,
                                    ACCEPT_TRIGRAM_NEAR_MISS_FLOOR)

        def report(passed, codes, trigram):
            return {"passed": passed,
                    "findings": [{"code": code} for code in codes],
                    "metrics": {"ordered_trigram_recall": trigram}}

        self.assertFalse(is_trigram_only_near_miss(report(True, [], 0.99)))
        self.assertTrue(is_trigram_only_near_miss(
            report(False, ["low_ordered_trigram_recall"], ACCEPT_TRIGRAM_NEAR_MISS_FLOOR)))
        self.assertTrue(is_trigram_only_near_miss(
            report(False, ["low_ordered_trigram_recall"], 0.88)))
        self.assertFalse(is_trigram_only_near_miss(
            report(False, ["low_ordered_trigram_recall"],
                   ACCEPT_TRIGRAM_NEAR_MISS_FLOOR - 0.01)))
        # Any second defect (truncation, ratio, etc.) disqualifies it.
        self.assertFalse(is_trigram_only_near_miss(
            report(False, ["low_ordered_trigram_recall", "low_source_token_recall"], 0.88)))

    def test_real_validator_produces_a_trigram_only_near_miss(self):
        from chunk_quality import is_trigram_only_near_miss
        source, near_miss = _trigram_near_miss_text()
        report = validate_chunk_quality(source, [_entry(near_miss)])

        self.assertTrue(is_trigram_only_near_miss(report))
        self.assertEqual({"low_ordered_trigram_recall"},
                         {finding["code"] for finding in report["findings"]})
        self.assertGreaterEqual(report["metrics"]["source_token_recall"], 0.90)

    def test_near_miss_accepted_only_after_full_and_split_exhaustion(self):
        # An unsplittable chunk (too short for split_failed_chunk) whose every
        # attempt is a trigram-only near-miss: full retries exhaust, no split is
        # possible, so the captured near-miss is accepted rather than failing.
        source, near_miss = _trigram_near_miss_text()
        self.assertEqual([], generate_script.split_failed_chunk(source))  # can't split
        client = _always_returns([_entry(near_miss)])
        params = generate_script.LLMGenParams("system", "{chunk}", 200, 0.1, 1)

        entries, adaptively_split = generate_script.process_chunk_adaptively(
            client, "model", source, 1, 1, params)

        self.assertTrue(entries)
        self.assertFalse(adaptively_split)
        self.assertFalse(validate_chunk_quality(source, entries)["passed"])
        from chunk_quality import is_trigram_only_near_miss
        self.assertTrue(is_trigram_only_near_miss(validate_chunk_quality(source, entries)))

    def test_near_miss_not_accepted_when_a_later_attempt_passes_cleanly(self):
        # Regression guard for zero blast radius: if a clean pass is still
        # reachable, the near-miss fallback must NOT pre-empt it.
        source, near_miss = _trigram_near_miss_text()
        clean = [_entry(source)]
        client = generate_script_test_client([[_entry(near_miss)], clean])
        params = generate_script.LLMGenParams("system", "{chunk}", 200, 0.1, 1)

        result = generate_script.process_chunk(
            client, "model", source, 1, 1, params, max_retries=1)

        self.assertEqual(clean, result)

    def test_split_recombination_accepted_as_near_miss(self):
        # The V23/V25 case: the full chunk never reaches near-miss quality (it
        # truncates), but each smaller split half does, and the recombined whole
        # lands as a trigram-only near-miss. Must be accepted, not forfeited.
        from types import SimpleNamespace
        import json
        import re
        # 300 words across sentence boundaries so split_failed_chunk can split.
        words = [f"word{index}" for index in range(300)]
        source = ". ".join(" ".join(words[i:i + 10]) for i in range(0, 300, 10)) + "."

        def create(**kwargs):
            chunk_words = re.findall(r"word\d+", kwargs["messages"][-1]["content"])
            if len(chunk_words) > 200:                 # the full chunk: truncate
                payload = [_entry(" ".join(chunk_words[:3]))]
            else:                                      # a split half: near-miss echo
                swapped = list(chunk_words)
                i = 0
                while i + 1 < len(swapped):
                    swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
                    i += 30
                payload = [_entry(" ".join(swapped))]
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(payload)),
                finish_reason="stop")], usage=None)

        client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)))
        params = generate_script.LLMGenParams("system", "{chunk}", 400, 0.1, 1)

        entries, adaptively_split = generate_script.process_chunk_adaptively(
            client, "model", source, 1, 1, params)

        self.assertTrue(entries)
        self.assertTrue(adaptively_split)
        from chunk_quality import is_trigram_only_near_miss
        self.assertTrue(is_trigram_only_near_miss(validate_chunk_quality(source, entries)))

    def test_hard_failure_is_not_rescued_as_near_miss(self):
        # Severe truncation (fails source-token recall too) is never a near-miss,
        # so an unsplittable all-truncation chunk still fails outright.
        source = " ".join(f"word{index}" for index in range(100))
        truncated = [_entry("word0 word1 word2")]
        client = _always_returns(truncated)
        params = generate_script.LLMGenParams("system", "{chunk}", 200, 0.1, 1)

        entries, _ = generate_script.process_chunk_adaptively(
            client, "model", source, 1, 1, params)

        self.assertEqual([], entries)


if __name__ == "__main__":
    unittest.main()
