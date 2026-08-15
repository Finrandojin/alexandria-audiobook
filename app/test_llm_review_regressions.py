from pathlib import Path
import json
import os
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import tempfile

import generate_script
import core
import review_script
from lmstudio_settings import (get_effective_max_tokens, get_next_retry_max_tokens,
                               TokenBudgetError)


class LlmReviewTests(unittest.TestCase):
    def test_response_log_is_isolated_by_run_id(self):
        with patch.dict(os.environ, {"ALEXANDRIA_RUN_ID": "run_test"}):
            path = generate_script.get_response_log_path("llm_responses.log")
        self.assertTrue(path.endswith("logs/responses/run_test/llm_responses.log"))

    @staticmethod
    def _client_with_responses(contents):
        responses = iter(contents)
        def create(**_kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=next(responses)), finish_reason="stop")], usage=None)
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    def test_chunk_quality_failure_retries_then_accepts_complete_response(self):
        source = " ".join(f"word{index}" for index in range(20))
        incomplete = [{"speaker": "NARRATOR", "text": "word0 word1", "instruct": "neutral"}]
        complete = [{"speaker": "NARRATOR", "text": source, "instruct": "neutral"}]
        client = self._client_with_responses([json.dumps(incomplete), json.dumps(complete)])
        params = generate_script.LLMGenParams("system", "{chunk}", 100, 0.1, 1)

        result = generate_script.process_chunk(
            client, "model", source, 1, 1, params, max_retries=1)

        self.assertEqual(complete, result)

    def test_process_chunk_default_budget_is_five_attempts(self):
        # Regression: default max_retries must stay 4 (5 total attempts), not
        # silently regress to the old 2 (3 attempts). Live reproduction of two
        # real overnight batch failures measured a genuine ~40% single-attempt
        # success rate for the specific failure this budget exists to absorb
        # (the model stopping a few lines into a chunk); 3 attempts recovers
        # ~78% of the time, 5 attempts ~92%.
        source = " ".join(f"word{index}" for index in range(20))
        incomplete = json.dumps([{"speaker": "NARRATOR", "text": "word0", "instruct": "n"}])
        complete = json.dumps([{"speaker": "NARRATOR", "text": source, "instruct": "n"}])
        # Fails 4 times, succeeds on the 5th -- only reachable if the default
        # budget is actually 5 attempts, not 3.
        client = self._client_with_responses(
            [incomplete, incomplete, incomplete, incomplete, complete])
        params = generate_script.LLMGenParams("system", "{chunk}", 100, 0.1, 1)

        result = generate_script.process_chunk(client, "model", source, 1, 1, params)

        self.assertEqual(json.loads(complete), result)

    def test_process_chunk_default_budget_stops_after_exactly_five_attempts(self):
        source = " ".join(f"word{index}" for index in range(20))
        incomplete = json.dumps([
            {"speaker": "NARRATOR", "text": "word0", "instruct": "n"}])
        calls = 0

        def create(**_kwargs):
            nonlocal calls
            calls += 1
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=incomplete), finish_reason="stop")],
                usage=None)

        client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)))
        params = generate_script.LLMGenParams("system", "{chunk}", 100, 0.1, 1)

        result = generate_script.process_chunk(client, "model", source, 1, 1, params)

        self.assertEqual([], result)
        self.assertEqual(5, calls)

    def test_attempt_observer_receives_token_and_finish_metrics(self):
        source = "one two three four five"
        response = json.dumps([{"speaker": "NARRATOR", "text": source,
                                "instruct": "neutral"}])
        usage = SimpleNamespace(prompt_tokens=12, completion_tokens=8)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **_kwargs: SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=response),
                                         finish_reason="stop")], usage=usage))))
        attempts = []

        generate_script.process_chunk(
            client, "model", source, 1, 1,
            generate_script.LLMGenParams("system", "{chunk}", 100, 0.1, 1),
            attempt_observer=attempts.append)

        self.assertEqual(1, len(attempts))
        self.assertEqual("stop", attempts[0]["finish_reason"])
        self.assertEqual(12, attempts[0]["prompt_tokens"])
        self.assertEqual(8, attempts[0]["completion_tokens"])
        self.assertEqual("accepted", attempts[0]["outcome"])

    def test_attempt_observer_records_quality_rejection_codes(self):
        source = " ".join(f"word{index}" for index in range(20))
        incomplete = json.dumps([
            {"speaker": "NARRATOR", "text": "word0", "instruct": "neutral"}])
        attempts = []

        generate_script.process_chunk(
            self._client_with_responses([incomplete]), "model", source, 1, 1,
            generate_script.LLMGenParams("system", "{chunk}", 100, 0.1, 1),
            max_retries=0, attempt_observer=attempts.append)

        self.assertEqual("quality_rejected", attempts[0]["outcome"])
        self.assertIn("low_source_token_recall", attempts[0]["failure_codes"])

    def test_deterministic_api_errors_retain_the_retry_budget(self):
        calls = []
        attempts = []

        def unavailable(**_kwargs):
            calls.append(True)
            raise ConnectionError("offline")

        client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=unavailable)))
        result = generate_script.call_llm_for_entries(
            client, "model", "system", "user",
            generate_script.LLMGenParams(max_tokens=100, temperature=0.0),
            "llm_responses.log", "TEST", max_retries=2,
            attempt_observer=attempts.append)

        self.assertEqual([], result)
        self.assertEqual(3, len(calls))
        self.assertEqual(3, len(attempts))
        self.assertTrue(all(
            attempt["outcome"] == "api_error" for attempt in attempts))

    def test_chunk_quality_exhaustion_returns_failure_even_with_stop_reason(self):
        source = " ".join(f"word{index}" for index in range(20))
        incomplete = json.dumps([{"speaker": "NARRATOR", "text": "word0", "instruct": "neutral"}])
        client = self._client_with_responses([incomplete, incomplete])
        params = generate_script.LLMGenParams("system", "{chunk}", 100, 0.1, 1)

        result = generate_script.process_chunk(
            client, "model", source, 1, 1, params, max_retries=1)

        self.assertEqual([], result)

    def test_early_split_decider_stops_full_chunk_after_second_severe_failure(self):
        source = " ".join(f"word{index}" for index in range(20))
        incomplete = json.dumps([
            {"speaker": "NARRATOR", "text": "word0", "instruct": "neutral"}])
        calls = 0

        def create(**_kwargs):
            nonlocal calls
            calls += 1
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=incomplete), finish_reason="stop")],
                usage=None)

        client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)))
        result = generate_script.process_chunk(
            client, "model", source, 1, 1,
            generate_script.LLMGenParams("system", "{chunk}", 100, 0.1, 1),
            allow_early_split=True)

        self.assertEqual([], result)
        self.assertEqual(2, calls)

    def test_token_budget_uses_fallback_without_verified_context(self):
        self.assertEqual(4096, get_effective_max_tokens(4096, None, [], 16000))

    def test_token_budget_scales_with_verified_context(self):
        self.assertEqual(16000, get_effective_max_tokens(4096, 98304, [], 16000))

    def test_token_budget_reserves_prompt_space(self):
        messages = [{"role": "user", "content": "x" * 18000}]
        self.assertEqual(1680, get_effective_max_tokens(4096, 8192, messages, 16000))

    def test_token_budget_enforces_task_ceiling(self):
        self.assertEqual(6000, get_effective_max_tokens(2000, 98304, [], 6000))

    def test_token_budget_rejects_prompt_larger_than_context(self):
        with self.assertRaises(TokenBudgetError):
            get_effective_max_tokens(100, 1000, [{"role": "user", "content": "x" * 3000}], 500)

    def test_token_budget_rejects_invalid_context(self):
        with self.assertRaises(ValueError):
            get_effective_max_tokens(100, "not-a-number", [], 500)

    def test_adjacent_json_arrays_are_combined_in_order(self):
        first = [{"speaker": "NARRATOR", "text": "one", "instruct": "neutral"}]
        second = [{"speaker": "TWO", "text": "two", "instruct": "quiet"}]

        cleaned = generate_script.clean_json_string(
            json.dumps(first) + "\n" + json.dumps(second))

        self.assertEqual(first + second, generate_script.repair_json_array(cleaned))

    def test_adjacent_json_array_overlap_is_rejected(self):
        first = [{"speaker": "NARRATOR", "text": "before the repeated words here",
                  "instruct": "neutral"}]
        second = [{"speaker": "OTHER", "text": "repeated words here after",
                   "instruct": "quiet"}]

        with self.assertRaisesRegex(
                generate_script.AdjacentArrayOverlapError, "repeated words here"):
            generate_script.clean_json_string(
                json.dumps(first) + "\n" + json.dumps(second))

    def test_adjacent_json_array_overlap_retries_with_specific_feedback(self):
        prompts = []
        first = [{"speaker": "NARRATOR", "text": "before repeated words here",
                  "instruct": "neutral"}]
        overlap = [{"speaker": "OTHER", "text": "repeated words here after",
                    "instruct": "quiet"}]
        complete = [{"speaker": "NARRATOR", "text": "complete", "instruct": "neutral"}]
        responses = iter([json.dumps(first) + "\n" + json.dumps(overlap),
                          json.dumps(complete)])

        def create(**kwargs):
            prompts.append(kwargs["messages"][1]["content"])
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=next(responses)), finish_reason="stop")],
                usage=None)

        client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)))
        attempts = []
        result = generate_script.call_llm_for_entries(
            client, "model", "system", "text", generate_script.LLMGenParams(),
            "test_responses.log", "TEST", max_retries=1,
            attempt_observer=attempts.append)

        self.assertEqual(complete, result)
        self.assertNotIn("adjacent_array_overlap", prompts[0])
        self.assertIn("adjacent_array_overlap", prompts[1])
        self.assertEqual(["adjacent_array_overlap"], attempts[0]["failure_codes"])
        self.assertEqual("accepted", attempts[1]["outcome"])

    def test_two_word_adjacent_array_boundary_is_not_treated_as_overlap(self):
        first = [{"speaker": "NARRATOR", "text": "before repeated words",
                  "instruct": "neutral"}]
        second = [{"speaker": "OTHER", "text": "repeated words after",
                   "instruct": "quiet"}]

        cleaned = generate_script.clean_json_string(
            json.dumps(first) + "\n" + json.dumps(second))

        self.assertEqual(first + second, generate_script.repair_json_array(cleaned))

    def test_text_between_json_arrays_is_not_silently_discarded(self):
        first = [{"speaker": "NARRATOR", "text": "one", "instruct": "neutral"}]
        second = [{"speaker": "TWO", "text": "two", "instruct": "quiet"}]

        cleaned = generate_script.clean_json_string(
            json.dumps(first) + " malformed " + json.dumps(second))

        self.assertIsNone(generate_script.repair_json_array(cleaned))

    def test_exhausted_ambiguous_arrays_use_quality_gated_raw_salvage(self):
        source = "one two three four five six"
        first = [{"speaker": "NARRATOR", "text": "one two three",
                  "instruct": "neutral"}]
        second = [{"speaker": "NARRATOR", "text": "four five six",
                   "instruct": "neutral"}]
        response = json.dumps(first) + "\nmodel commentary\n" + json.dumps(second)
        attempts = []

        result = generate_script.process_chunk(
            self._client_with_responses([response]), "model", source, 1, 1,
            generate_script.LLMGenParams("system", "{chunk}", 100, 0.1, 1),
            max_retries=0, attempt_observer=attempts.append)

        self.assertEqual(source, " ".join(entry["text"] for entry in result))
        self.assertEqual("accepted", attempts[0]["outcome"])
        self.assertEqual(["missing_json_array"], attempts[0]["recovery_codes"])
        self.assertNotIn("failure_codes", attempts[0])

    def test_exhausted_incomplete_raw_salvage_still_fails_quality_gate(self):
        source = "one two three four five six seven eight nine ten"
        incomplete = [{"speaker": "NARRATOR", "text": "one two",
                       "instruct": "neutral"}]
        response = json.dumps(incomplete) + "\nmodel commentary\n[{broken}]"
        attempts = []

        result = generate_script.process_chunk(
            self._client_with_responses([response]), "model", source, 1, 1,
            generate_script.LLMGenParams("system", "{chunk}", 100, 0.1, 1),
            max_retries=0, attempt_observer=attempts.append)

        self.assertEqual([], result)
        self.assertEqual("quality_rejected", attempts[0]["outcome"])
        self.assertIn("low_source_token_recall", attempts[0]["failure_codes"])

    def test_bracketed_trailing_prose_does_not_discard_valid_array(self):
        entries = [{"speaker": "NARRATOR", "text": "one", "instruct": "neutral"}]

        cleaned = generate_script.clean_json_string(
            json.dumps(entries) + "\nNote: preserve [speaker] labels.")

        self.assertEqual(entries, generate_script.repair_json_array(cleaned))

    def test_malformed_trailing_entry_array_is_rejected(self):
        entries = [{"speaker": "NARRATOR", "text": "one", "instruct": "neutral"}]

        cleaned = generate_script.clean_json_string(
            json.dumps(entries) + '\n[{"speaker":}]')

        self.assertIsNone(cleaned)

    def test_retry_budget_only_increases_for_incomplete_output(self):
        self.assertEqual(6144, get_next_retry_max_tokens(
            4096, "token_truncated", 16384))
        self.assertEqual(9216, get_next_retry_max_tokens(
            6144, "incomplete_output", 16384))
        self.assertEqual(4096, get_next_retry_max_tokens(
            4096, "quality_failure", 16384))
        self.assertEqual(7000, get_next_retry_max_tokens(
            6144, "token_truncated", 7000))

    def test_length_retries_progressively_increase_api_budget(self):
        calls = []
        prompts = []
        valid = json.dumps([
            {"speaker": "NARRATOR", "text": "done", "instruct": "neutral"}
        ])
        contents = iter([
            (valid, "length"),
            (valid, "length"),
            (valid, "stop"),
        ])

        def create(**kwargs):
            calls.append(kwargs["max_tokens"])
            prompts.append(kwargs["messages"])
            content, reason = next(contents)
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=content), finish_reason=reason)], usage=None)

        client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)))
        params = generate_script.LLMGenParams(max_tokens=4096, hard_max_tokens=16384)

        result = generate_script.call_llm_for_entries(
            client, "model", "system", "text", params,
            "test_responses.log", "TEST", max_retries=2)

        self.assertEqual("done", result[0]["text"])
        self.assertEqual([4096, 6144, 9216], calls)
        self.assertTrue(all(messages[1]["content"] == "text" for messages in prompts))

    def test_incomplete_stop_does_not_increase_unspent_token_budget(self):
        calls = []
        incomplete = json.dumps([{"speaker": "NARRATOR", "text": "one",
                                  "instruct": "neutral"}])
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=400)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: (calls.append(kwargs["max_tokens"]) or SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=incomplete),
                                         finish_reason="stop")], usage=usage)))))
        quality = lambda _entries: {"passed": False,
            "metrics": {"output_source_ratio": 0.1},
            "findings": [{"code": "low_source_token_recall", "value": 0.1}]}

        result = generate_script.call_llm_for_entries(
            client, "model", "system", "text", generate_script.LLMGenParams(max_tokens=4096),
            "test_responses.log", "TEST", max_retries=1, validate_entries=quality)

        self.assertEqual([], result)
        self.assertEqual([4096, 4096], calls)

    def test_retry_feedback_for_truncation_cluster_is_plain_english(self):
        quality = {"metrics": {"source_token_recall": 0.53},
                   "findings": [{"code": "low_source_token_recall", "value": 0.53},
                               {"code": "low_ordered_trigram_recall", "value": 0.51},
                               {"code": "output_source_ratio", "value": 0.53}]}
        message = generate_script._build_retry_feedback_message(quality)
        self.assertIn("about 53%", message)
        self.assertIn("stopping early", message)
        self.assertNotIn("low_source_token_recall", message)
        self.assertNotIn("{", message)

    def test_retry_feedback_for_truncation_cluster_without_recall_metric(self):
        quality = {"metrics": {}, "findings": [{"code": "output_source_ratio"}]}
        message = generate_script._build_retry_feedback_message(quality)
        self.assertIn("too little", message)
        self.assertIn("stopping early", message)

    def test_retry_feedback_for_other_codes_uses_finding_messages(self):
        quality = {"metrics": {}, "findings": [
            {"code": "missing_fields", "message": "Entry is missing required fields."},
            {"code": "empty_text", "message": "Entry contains no speakable text."}]}
        message = generate_script._build_retry_feedback_message(quality)
        self.assertEqual(
            "Entry is missing required fields. Entry contains no speakable text.",
            message)

    def test_retry_feedback_falls_back_to_json_when_no_messages_present(self):
        # An unrelated code with no message, to exercise the defensive final
        # fallback (truncation-cluster codes always short-circuit earlier).
        quality = {"metrics": {}, "findings": [{"code": "some_future_code"}]}
        message = generate_script._build_retry_feedback_message(quality)
        self.assertEqual(json.dumps(quality["findings"], ensure_ascii=False), message)

    def test_incomplete_stop_retry_prompt_uses_plain_english_not_raw_codes(self):
        prompts = []
        incomplete = json.dumps([{"speaker": "NARRATOR", "text": "one",
                                  "instruct": "neutral"}])
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=400)

        def create(**kwargs):
            prompts.append(kwargs["messages"][-1]["content"])
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=incomplete), finish_reason="stop")],
                usage=usage)

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        quality = lambda _entries: {"passed": False,
            "metrics": {"source_token_recall": 0.3, "output_source_ratio": 0.1},
            "findings": [{"code": "low_source_token_recall", "value": 0.3}]}

        generate_script.call_llm_for_entries(
            client, "model", "system", "text", generate_script.LLMGenParams(max_tokens=4096),
            "test_responses.log", "TEST", max_retries=1, validate_entries=quality)

        self.assertNotIn("low_source_token_recall", prompts[1])
        self.assertIn("stopping early", prompts[1])
        self.assertIn("about 30%", prompts[1])

    def test_is_near_miss_recall_true_above_threshold(self):
        self.assertTrue(generate_script._is_near_miss_recall(
            {"source_token_recall": 0.86}))
        self.assertTrue(generate_script._is_near_miss_recall(
            {"source_token_recall": 0.75}))

    def test_is_near_miss_recall_false_below_threshold_or_missing(self):
        self.assertFalse(generate_script._is_near_miss_recall(
            {"source_token_recall": 0.5}))
        self.assertFalse(generate_script._is_near_miss_recall(
            {"source_token_recall": 0.05}))
        self.assertFalse(generate_script._is_near_miss_recall({}))
        self.assertFalse(generate_script._is_near_miss_recall(None))

    def test_retry_feedback_near_miss_uses_different_wording(self):
        quality = {"metrics": {"source_token_recall": 0.86},
                   "findings": [{"code": "low_source_token_recall", "value": 0.86}]}
        message = generate_script._build_retry_feedback_message(quality)
        self.assertIn("close but only covered about 86%", message)
        self.assertNotIn("stopping early", message)
        self.assertNotIn("ENTIRE", message)

    def test_retry_feedback_catastrophic_keeps_original_wording(self):
        quality = {"metrics": {"source_token_recall": 0.11},
                   "findings": [{"code": "low_source_token_recall", "value": 0.11}]}
        message = generate_script._build_retry_feedback_message(quality)
        self.assertIn("stopping early", message)
        self.assertIn("ENTIRE", message)
        self.assertNotIn("close but", message)
        self.assertNotIn("missing content includes", message)

    def test_retry_feedback_catastrophic_quotes_missing_spans(self):
        quality = {"metrics": {"source_token_recall": 0.11},
                   "findings": [{"code": "low_source_token_recall", "value": 0.11}],
                   "missing_source_spans": [
                       {"start_token": 50, "token_count": 40,
                        "preview": "the queen turned to face the shattered gate"}]}
        message = generate_script._build_retry_feedback_message(quality)
        self.assertIn("stopping early", message)
        self.assertIn('"the queen turned to face the shattered gate ..."', message)
        self.assertIn("about 40 words", message)

    def test_retry_feedback_near_miss_quotes_missing_spans(self):
        quality = {"metrics": {"source_token_recall": 0.86},
                   "findings": [{"code": "low_source_token_recall", "value": 0.86}],
                   "missing_source_spans": [
                       {"start_token": 10, "token_count": 8, "preview": "a short lost aside"},
                       {"start_token": 90, "token_count": 6, "preview": "another dropped phrase"}]}
        message = generate_script._build_retry_feedback_message(quality)
        self.assertIn("close but", message)
        self.assertIn('"a short lost aside ..."', message)
        self.assertIn('"another dropped phrase ..."', message)

    def test_retry_feedback_unchanged_without_spans_key(self):
        quality = {"metrics": {"source_token_recall": 0.86},
                   "findings": [{"code": "low_source_token_recall", "value": 0.86}]}
        message = generate_script._build_retry_feedback_message(quality)
        self.assertIn("close but only covered about 86%", message)
        self.assertNotIn("missing content includes", message)

    def test_process_chunk_grants_bonus_retry_after_near_miss_final_attempt(self):
        source = " ".join(f"word{index}" for index in range(20))
        # 5 catastrophic-then-near-miss responses to exhaust the default
        # budget (max_retries=4, 5 attempts), then a complete 6th (bonus).
        tiny = json.dumps([{"speaker": "NARRATOR", "text": "word0", "instruct": "n"}])
        near_miss = json.dumps([{"speaker": "NARRATOR", "text": " ".join(
            f"word{index}" for index in range(16)), "instruct": "n"}])
        complete = json.dumps([{"speaker": "NARRATOR", "text": source, "instruct": "n"}])
        client = self._client_with_responses(
            [tiny, tiny, tiny, tiny, near_miss, complete])
        params = generate_script.LLMGenParams("system", "{chunk}", 100, 0.1, 1)

        result = generate_script.process_chunk(client, "model", source, 1, 1, params)

        self.assertEqual(json.loads(complete), result)

    def test_process_chunk_bonus_retry_is_exactly_one_attempt(self):
        source = " ".join(f"word{index}" for index in range(20))
        tiny = json.dumps([{"speaker": "NARRATOR", "text": "word0", "instruct": "n"}])
        near_miss = json.dumps([{"speaker": "NARRATOR", "text": " ".join(
            f"word{index}" for index in range(16)), "instruct": "n"}])
        # Final regular attempt (5th) is a near miss; the bonus (6th) call
        # is ALSO a near miss - must not grant a second bonus.
        client = self._client_with_responses(
            [tiny, tiny, tiny, tiny, near_miss, near_miss])
        params = generate_script.LLMGenParams("system", "{chunk}", 100, 0.1, 1)

        result = generate_script.process_chunk(client, "model", source, 1, 1, params)

        self.assertEqual([], result)

    def test_process_chunk_no_bonus_retry_for_catastrophic_final_attempt(self):
        # Regression guard: matches
        # test_process_chunk_default_budget_stops_after_exactly_five_attempts's
        # fixture shape exactly - every attempt is catastrophic (~5% recall,
        # not a near miss), so no bonus call should ever fire.
        source = " ".join(f"word{index}" for index in range(20))
        incomplete = json.dumps([
            {"speaker": "NARRATOR", "text": "word0", "instruct": "n"}])
        calls = 0

        def create(**_kwargs):
            nonlocal calls
            calls += 1
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=incomplete), finish_reason="stop")],
                usage=None)

        client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)))
        params = generate_script.LLMGenParams("system", "{chunk}", 100, 0.1, 1)

        result = generate_script.process_chunk(client, "model", source, 1, 1, params)

        self.assertEqual([], result)
        self.assertEqual(5, calls)

    def test_near_limit_incomplete_stop_increases_budget(self):
        quality = {"metrics": {"output_source_ratio": 0.2},
                   "findings": [{"code": "output_source_ratio"}]}
        self.assertEqual("increase_tokens", generate_script.get_quality_retry_policy(
            "stop", 950, 1000, quality))

    def test_context_clamped_truncation_does_not_retry(self):
        calls = []
        valid = json.dumps([
            {"speaker": "NARRATOR", "text": "done", "instruct": "neutral"}
        ])

        def create(**kwargs):
            calls.append(kwargs["max_tokens"])
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=valid), finish_reason="length")], usage=None)

        client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)))
        params = generate_script.LLMGenParams(
            max_tokens=4096, context_length=2000, hard_max_tokens=16384)

        result = generate_script.call_llm_for_entries(
            client, "model", "system", "text", params,
            "test_responses.log", "TEST", max_retries=2)

        self.assertEqual([], result)
        self.assertEqual(1, len(calls))
        self.assertLess(calls[0], params.max_tokens)

    def test_hard_capped_truncation_does_not_retry(self):
        calls = []
        valid = json.dumps([
            {"speaker": "NARRATOR", "text": "done", "instruct": "neutral"}
        ])

        def create(**kwargs):
            calls.append(kwargs["max_tokens"])
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=valid), finish_reason="length")], usage=None)

        client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)))
        params = generate_script.LLMGenParams(max_tokens=4096, hard_max_tokens=4096)

        result = generate_script.call_llm_for_entries(
            client, "model", "system", "text", params,
            "test_responses.log", "TEST", max_retries=2)

        self.assertEqual([], result)
        self.assertEqual([4096], calls)

    def test_llm_salvage_waits_until_retries_are_exhausted(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="[]"), finish_reason="stop"
            )],
            usage=None,
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_kwargs: response)
            )
        )
        params = generate_script.LLMGenParams("system", "{text}", 100, 0.1, 1, 0, 0, 0, "")
        complete = [{"type": "narration", "text": "complete"}]
        with patch.object(generate_script, "clean_json_string", return_value="[]"), \
             patch.object(generate_script, "repair_json_array", side_effect=[[], complete]), \
             patch.object(generate_script, "salvage_json_entries", return_value=[{"text": "partial"}]) as salvage:
            result = generate_script.call_llm_for_entries(
                client, "model", "system", "text", params,
                "test_responses.log", "TEST", max_retries=1
            )
        self.assertEqual(result, complete)
        salvage.assert_not_called()

    def test_review_help_does_not_advertise_unimplemented_source_mode(self):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("review_script.py")),
             "--help"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertNotIn("--source", result.stdout)


class ReviewDiffAlignmentTests(unittest.TestCase):
    @staticmethod
    def entry(text, speaker="NARRATOR", instruct="neutral"):
        return {"text": text, "speaker": speaker, "instruct": instruct}

    def test_middle_insertion_does_not_shift_later_comparisons(self):
        original = [self.entry("first"), self.entry("second"), self.entry("third")]
        corrected = [
            self.entry("first"), self.entry("inserted", "SUBARU", "urgent"),
            self.entry("second"), self.entry("third"),
        ]

        stats = review_script.diff_entries(original, corrected)

        self.assertEqual(stats["entries_added"], 1)
        self.assertEqual(stats["entries_removed"], 0)
        self.assertEqual(stats["text_changed"], 0)
        self.assertEqual(stats["speaker_changed"], 0)
        self.assertEqual(stats["instruct_changed"], 0)

    def test_middle_removal_does_not_shift_later_comparisons(self):
        original = [self.entry("first"), self.entry("removed"), self.entry("third")]
        corrected = [self.entry("first"), self.entry("third")]

        stats = review_script.diff_entries(original, corrected)

        self.assertEqual(stats["entries_added"], 0)
        self.assertEqual(stats["entries_removed"], 1)
        self.assertEqual(stats["text_changed"], 0)
        self.assertEqual(stats["speaker_changed"], 0)
        self.assertEqual(stats["instruct_changed"], 0)

    def test_rewrite_and_metadata_changes_remain_visible(self):
        original = [self.entry("old wording"), self.entry("same text")]
        corrected = [
            self.entry("new wording"),
            self.entry("same text", speaker="SUBARU", instruct="quietly"),
        ]
        highlights = {"text": [], "speaker": []}

        stats = review_script.diff_entries(original, corrected, highlights)

        self.assertEqual(stats["text_changed"], 1)
        self.assertEqual(stats["speaker_changed"], 1)
        self.assertEqual(stats["instruct_changed"], 1)
        self.assertEqual(stats["entries_changed"], 2)
        self.assertEqual(highlights["text"][0]["before"], "old wording")
        self.assertEqual(highlights["text"][0]["after"], "new wording")

    def test_insertion_does_not_pair_adjacent_july_report_lines(self):
        spirit_explanation = "That’s the tricky thing about spirit mages."
        old_man_question = "By the way, old man, what is it you’re planning on doing?"
        original = [self.entry(old_man_question), self.entry("following line")]
        corrected = [
            self.entry(spirit_explanation), self.entry(old_man_question),
            self.entry("following line"),
        ]
        highlights = {"text": [], "speaker": []}

        stats = review_script.diff_entries(original, corrected, highlights)

        self.assertEqual(stats["entries_added"], 1)
        self.assertEqual(stats["text_changed"], 0)
        self.assertEqual(highlights["text"], [])

    def test_replacement_with_insertion_counts_each_kind_once(self):
        original = [self.entry("first"), self.entry("old"), self.entry("last")]
        corrected = [
            self.entry("first"), self.entry("new"), self.entry("extra"), self.entry("last"),
        ]

        stats = review_script.diff_entries(original, corrected)

        self.assertEqual(stats["text_changed"], 1)
        self.assertEqual(stats["entries_added"], 1)
        self.assertEqual(stats["entries_removed"], 0)

    def test_addition_and_removal_are_counted_when_length_is_unchanged(self):
        original = [self.entry("first"), self.entry("removed"), self.entry("anchor")]
        corrected = [self.entry("inserted"), self.entry("first"), self.entry("anchor")]

        stats = review_script.diff_entries(original, corrected)

        self.assertEqual(stats["entries_added"], 1)
        self.assertEqual(stats["entries_removed"], 1)
        self.assertEqual(stats["text_changed"], 0)

    def test_highlight_pool_cap_is_preserved(self):
        original = [self.entry(f"old {index}") for index in range(510)]
        corrected = [self.entry(f"new {index}") for index in range(510)]
        highlights = {"text": [], "speaker": []}

        stats = review_script.diff_entries(original, corrected, highlights)

        self.assertEqual(stats["text_changed"], 510)
        self.assertEqual(len(highlights["text"]), review_script._MAX_HIGHLIGHT_POOL)

    def test_speaker_change_includes_entry_number_and_neighbor_context(self):
        original = [
            self.entry("before"), self.entry("He said as the cart passed."), self.entry("after"),
        ]
        corrected = [
            self.entry("before"), self.entry("He said as the cart passed.", speaker="KENJI"),
            self.entry("after"),
        ]
        highlights = {"text": [], "speaker": []}

        review_script.diff_entries(original, corrected, highlights, entry_offset=100)

        change = highlights["speaker"][0]
        self.assertEqual(change["entry_number"], 102)
        self.assertEqual(change["context_before"], "before")
        self.assertEqual(change["context_after"], "after")
        self.assertIn("Narrator-to-character", change["manual_review_reason"])

    def test_character_to_character_change_is_not_automatically_flagged(self):
        original = [self.entry("Hello", speaker="MAN")]
        corrected = [self.entry("Hello", speaker="KENJI")]
        highlights = {"text": [], "speaker": []}

        review_script.diff_entries(original, corrected, highlights)

        self.assertNotIn("manual_review_reason", highlights["speaker"][0])

    def test_speaker_markdown_does_not_assert_correction(self):
        highlights = {"text_rewrites": [], "speaker_changes": [{
            "text": "He said as the cart passed.", "before": "NARRATOR", "after": "KENJI",
            "entry_number": 102, "context_before": "Before.", "context_after": "After.",
            "manual_review_reason": "Narrator-to-character changes alter the reading voice.",
        }]}

        markdown = "\n".join(core._markdown_diff_highlights_lines(highlights))

        self.assertIn("Speaker changes to verify", markdown)
        self.assertIn("Entry 102", markdown)
        self.assertIn("Previous", markdown)
        self.assertIn("Manual check recommended", markdown)
        self.assertNotIn("corrected to", markdown)


class ReviewFailureReportingTests(unittest.TestCase):
    def test_failed_section_uses_human_entry_range_and_stable_ratio(self):
        section = review_script.get_failed_section(
            batch=3, zero_based_start=50, length=25,
            category="text_length_mismatch", word_ratio=0.912345,
        )

        self.assertEqual(section, {
            "batch": 3,
            "entry_start": 51,
            "entry_end": 75,
            "category": "text_length_mismatch",
            "word_ratio": 0.9123,
        })

    def test_failed_sections_parser_and_markdown_explain_safe_retry(self):
        lines = [
            'FAILED_SECTIONS_JSON: {"sections":[{"batch":3,"entry_start":51,'
            '"entry_end":75,"category":"text_length_mismatch","word_ratio":0.91}],'
            '"original_entries_preserved":true,"checkpoint_retained":true,'
            '"retry_from_batch":3}'
        ]

        failures = core._extract_failed_sections(lines)
        markdown = core._markdown_failed_sections_lines(failures)

        self.assertEqual(failures["sections"][0]["entry_start"], 51)
        self.assertTrue(any("entries 51–75" in line for line in markdown))
        self.assertTrue(any("original entries" in line for line in markdown))
        self.assertTrue(any("single-book review" in line for line in markdown))

    def test_malformed_failed_sections_are_not_reported(self):
        self.assertEqual(
            core._extract_failed_sections(['FAILED_SECTIONS_JSON: {"sections":"bad"}']),
            {"sections": []},
        )


class ReviewSummaryTests(unittest.TestCase):
    def test_incomplete_summary_is_deterministic_and_does_not_call_llm(self):
        stats = {"total_changes": 12, "batches_failed": 1, "batches_skipped_vram": 0}
        with patch.object(core, "_llm_summarize_report") as summarize:
            lines = core._insert_llm_summary(["Report"], 1, stats, incomplete=True)

        summarize.assert_not_called()
        self.assertIn("This review was incomplete", lines[4])
        self.assertIn("1 section(s) failed", lines[4])
        self.assertIn("12 change(s)", lines[4])

    def test_unsupported_quality_claim_uses_deterministic_fallback(self):
        stats = {"total_changes": 3}
        with patch.object(
                core, "_llm_summarize_report",
                return_value="Everything looks great and all issues were fixed."):
            lines = core._insert_llm_summary(["Report"], 1, stats)

        self.assertIn("without recorded failed or skipped sections", lines[4])
        self.assertNotIn("Everything looks great", lines[4])

    def test_evidence_bound_llm_summary_is_kept_for_complete_run(self):
        summary = "The pass reported three text changes; inspect the examples below."
        with patch.object(core, "_llm_summarize_report", return_value=summary):
            lines = core._insert_llm_summary(["Report"], 1, {"total_changes": 3})

        self.assertEqual(lines[4], summary)


class ReviewChangeDensityTests(unittest.TestCase):
    def test_high_change_density_warns_with_separate_structural_counts(self):
        stats = {
            "entries_before": 2615, "entries_changed": 613,
            "entries_added": 8, "entries_removed": 1,
        }

        lines = core._markdown_change_density_lines(stats)

        self.assertEqual(len(lines), 1)
        self.assertIn("613 of 2615", lines[0])
        self.assertIn("23.4%", lines[0])
        self.assertIn("+8/-1", lines[0])
        self.assertIn("before generating audio", lines[0])

    def test_low_density_and_small_scripts_do_not_warn(self):
        self.assertEqual(core._markdown_change_density_lines({
            "entries_before": 1000, "entries_changed": 199,
        }), [])
        self.assertEqual(core._markdown_change_density_lines({
            "entries_before": 99, "entries_changed": 99,
        }), [])


class DedupeSpeakersResolverTests(unittest.TestCase):
    """Covers the Area 5 fix: dedupe_speakers' registry pre-seed must resolve
    a punctuation/spacing variant to the label actually used in this script,
    the same way generation's _identity_key normalization does."""

    def test_registry_alias_resolves_via_shared_identity_normalization(self):
        entries = [
            {"speaker": "MR. SMITH", "text": "Hello there, stranger."},
            {"speaker": "JOHN SMITH", "text": "Another line of dialogue."},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = os.path.join(tmp, "aliases.json")
            with open(registry_path, "w", encoding="utf-8") as f:
                json.dump({"MR SMITH": "JOHN SMITH"}, f)

            # Force the LLM step to fail so only the registry's forced aliases
            # apply -- isolates the resolver fix from any model behavior.
            client = MagicMock()
            client.chat.completions.create.side_effect = RuntimeError("LLM unavailable")

            mapping, renamed, changes = review_script.dedupe_speakers(
                client, "model", entries, registry_path=registry_path)

        self.assertEqual({"MR. SMITH": "JOHN SMITH"}, mapping)
        self.assertEqual(1, renamed)
        self.assertEqual([(0, "speaker", "JOHN SMITH")], changes)


class PassSpecificSalvageTests(unittest.TestCase):
    def test_salvages_pass2_and_pass3_object_shapes(self):
        pass2 = ('[{"n":0,"head":"Tell me now","speaker":"ELENA"},'
                 '{"n":1,"head":"The room was","speaker":"NARRATOR"}, BROKEN]')
        pass3 = ('[{"n":0,"head":"Tell me now","instruct":"Firm."},'
                 '{"n":1,"head":"The room was","instruct":"Neutral."}, BROKEN]')
        self.assertEqual(["ELENA", "NARRATOR"],
                         [e["speaker"] for e in generate_script.salvage_json_entries(pass2)])
        self.assertEqual(["Firm.", "Neutral."],
                         [e["instruct"] for e in generate_script.salvage_json_entries(pass3)])

    def test_single_pass_chunk_size_validates_config_and_cli(self):
        self.assertEqual(3000, generate_script.get_valid_chunk_size(3000))
        self.assertEqual(500, generate_script.get_valid_chunk_size(3000, 500))
        for config_value in (0, -1, "3000", True):
            with self.assertRaises(ValueError):
                generate_script.get_valid_chunk_size(config_value)

    def test_review_rejects_text_only_salvage_even_when_word_count_matches(self):
        original = [{"speaker": "NARRATOR", "text": "one two three four five",
                     "instruct": "Neutral."}]
        client = LlmReviewTests._client_with_responses([
            '[{"text":"one two three four five"}, BROKEN]'])
        params = generate_script.LLMGenParams(max_tokens=100, temperature=0.1)
        self.assertIsNone(review_script.review_batch(
            client, "m", original, 1, 1, params, max_retries=0))
