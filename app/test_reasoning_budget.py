import unittest

from generate_script import LLMGenParams
from three_pass_generate import (ReasoningAllowance, resolve_completion_ceiling)


class ResolveCompletionCeilingTest(unittest.TestCase):

    def test_non_reasoning_model_keeps_todays_ceiling(self):
        params = LLMGenParams(segment_output_ratio=3.0)
        ceiling = resolve_completion_ceiling(
            source_words=400, params=params, reasoning_allowance=0)
        self.assertEqual(ceiling, 1200)

    def test_floor_still_applies(self):
        params = LLMGenParams(segment_output_ratio=3.0)
        ceiling = resolve_completion_ceiling(
            source_words=10, params=params, reasoning_allowance=0)
        self.assertEqual(ceiling, 512)

    def test_reasoning_allowance_is_added_on_top(self):
        params = LLMGenParams(segment_output_ratio=3.0)
        ceiling = resolve_completion_ceiling(
            source_words=400, params=params, reasoning_allowance=2048)
        self.assertEqual(ceiling, 1200 + 2048)

    def test_allowance_does_not_shrink_the_visible_budget(self):
        params = LLMGenParams(segment_output_ratio=3.0)
        without = resolve_completion_ceiling(
            source_words=400, params=params, reasoning_allowance=0)
        with_allowance = resolve_completion_ceiling(
            source_words=400, params=params, reasoning_allowance=5000)
        self.assertGreater(with_allowance, without)


class ReasoningAllowanceTest(unittest.TestCase):

    def test_cold_start_is_zero(self):
        allowance = ReasoningAllowance()
        self.assertEqual(allowance.current(), 0)

    def test_first_observation_lifts_to_the_floor(self):
        allowance = ReasoningAllowance()
        allowance.observe(1500)
        self.assertGreaterEqual(allowance.current(), 1024)

    def test_non_reasoning_model_stays_at_zero(self):
        allowance = ReasoningAllowance()
        for _ in range(10):
            allowance.observe(0)
        self.assertEqual(allowance.current(), 0)

    def test_none_observations_stay_at_zero(self):
        allowance = ReasoningAllowance()
        for _ in range(10):
            allowance.observe(None)
        self.assertEqual(allowance.current(), 0)

    def test_allowance_tracks_p95_of_observations(self):
        allowance = ReasoningAllowance()
        for value in list(range(1000, 3000, 100)) + [9000]:
            allowance.observe(value)
        self.assertGreaterEqual(allowance.current(), 2800)
        self.assertLessEqual(allowance.current(), 9000)


class ReasoningOverflowTest(unittest.TestCase):

    def test_empty_content_with_reasoning_is_overflow(self):
        from generate_script import classify_length_finish
        self.assertEqual(
            classify_length_finish(content="", reasoning_tokens=9987,
                                   already_escalated=False),
            "escalate_once")

    def test_second_overflow_fails_fast(self):
        from generate_script import classify_length_finish
        self.assertEqual(
            classify_length_finish(content="", reasoning_tokens=9987,
                                   already_escalated=True),
            "reasoning_overflow")

    def test_truncated_visible_output_is_not_overflow(self):
        from generate_script import classify_length_finish
        self.assertEqual(
            classify_length_finish(content='[{"n": 0, "speaker": "ARARAGI"}',
                                   reasoning_tokens=0, already_escalated=False),
            "truncated_output")

    def test_non_reasoning_model_never_reports_overflow(self):
        from generate_script import classify_length_finish
        self.assertEqual(
            classify_length_finish(content="", reasoning_tokens=None,
                                   already_escalated=True),
            "truncated_output")


class TruncatedObservationTest(unittest.TestCase):
    """A truncated response reports only the reasoning it emitted before the
    ceiling cut it off. Taking that as the true cost makes the allowance creep
    up one step per chunk, truncating every chunk on the way - observed live,
    2700 -> 4005 over ten chunks while the model wanted ~7100."""

    def test_truncated_observation_is_inflated(self):
        allowance = ReasoningAllowance()
        allowance.observe(2600, truncated=True)
        self.assertGreaterEqual(allowance.current(), 5200)

    def test_complete_observation_is_taken_at_face_value(self):
        allowance = ReasoningAllowance()
        allowance.observe(2600, truncated=False)
        self.assertEqual(allowance.current(), 2600)

    def test_truncation_does_not_affect_a_non_reasoning_model(self):
        allowance = ReasoningAllowance()
        allowance.observe(0, truncated=True)
        allowance.observe(None, truncated=True)
        self.assertEqual(allowance.current(), 0)


class SegmentAllowanceWiringTest(unittest.TestCase):
    """resolve_completion_ceiling is used ONLY by pass 1, but the allowance was
    fed only by the pass-2/3 observers, so it stayed zero for the pass that
    consumes it. Observed live: thinking-on segmentation kept reporting
    "cannot grow beyond 2700" (5.0 x ~540 words = visible output only)."""

    def test_segment_attempts_feed_the_allowance(self):
        import inspect

        import three_pass_generate as tp
        source = inspect.getsource(tp.run_three_pass)
        segment_loop = source[source.index("seg_base = elapsed_s.get"):]
        segment_loop = segment_loop[:segment_loop.index("elapsed_s[\"attribute\"]")]
        self.assertIn("reasoning_allowance.observe", segment_loop,
                      "pass 1 must feed the allowance it consumes")

    def test_ceiling_grows_once_reasoning_is_observed(self):
        from generate_script import LLMGenParams
        from three_pass_generate import ReasoningAllowance, resolve_completion_ceiling
        params = LLMGenParams(segment_output_ratio=5.0)
        allowance = ReasoningAllowance()
        cold = resolve_completion_ceiling(540, params, allowance.current())
        allowance.observe(7101)
        warm = resolve_completion_ceiling(540, params, allowance.current())
        self.assertEqual(cold, 2700)
        self.assertGreater(warm, 9000)


class ReasoningEffortPlumbingTest(unittest.TestCase):

    def test_reasoning_effort_reaches_extra_body(self):
        from generate_script import build_extra_body
        body = build_extra_body(LLMGenParams(top_k=40, reasoning_effort="none"))
        self.assertEqual(body["reasoning_effort"], "none")
        self.assertEqual(body["top_k"], 40)

    def test_unset_reasoning_effort_is_omitted(self):
        from generate_script import build_extra_body
        body = build_extra_body(LLMGenParams(top_k=40))
        self.assertNotIn("reasoning_effort", body)


if __name__ == "__main__":
    unittest.main()


class NullContentTest(unittest.TestCase):
    """A reasoning model that spends its whole budget thinking may return JSON
    null for content. `.strip()` on that raised AttributeError, which the caller
    booked as a generic api_error - so the one-escalation-then-overflow policy
    never ran on the exact responses it was written for. Found by audit."""

    def test_null_content_classifies_as_escalate_then_overflow(self):
        from generate_script import classify_length_finish
        self.assertEqual(
            "escalate_once",
            classify_length_finish(content=None, reasoning_tokens=9987,
                                   already_escalated=False))
        self.assertEqual(
            "reasoning_overflow",
            classify_length_finish(content=None, reasoning_tokens=9987,
                                   already_escalated=True))

    def test_null_content_without_reasoning_is_ordinary_truncation(self):
        from generate_script import classify_length_finish
        self.assertEqual(
            "truncated_output",
            classify_length_finish(content=None, reasoning_tokens=0,
                                   already_escalated=False))

    def test_a_null_content_response_does_not_raise(self):
        # The regression itself: reading .content on a null-content response.
        from types import SimpleNamespace
        choice = SimpleNamespace(message=SimpleNamespace(content=None),
                                 finish_reason="length")
        self.assertEqual("", (choice.message.content or "").strip())
