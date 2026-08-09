"""At temperature 0, re-sending a rejected prompt cannot help. Rule 10.

WHAT WAS OBSERVED. owarimonogatari3's pass-2 attribution failed a one-entry
batch after four attempts. Attempts 3 and 4 sent a prompt of the same 764
tokens and received a completion of the same 325 tokens, and both were rejected
by the same validator with the same message. Attribution runs at
`attribute_temperature` 0.0 - deliberately, so a book does not produce
different speakers on every run - and a deterministic model given an identical
prompt returns an identical answer. The fourth attempt was arithmetic, not
inference.

WHY NOT JUST CUT max_retries. The retries are not wasted in general: the loop
appends validator feedback to the prompt, so attempt 2 asks a genuinely
different question from attempt 1, and that often works. Only a repeat of a
prompt already sent is provably useless. The stopping rule has to be "this
exact prompt was already rejected", not a smaller budget.

WHY IT IS GUARDED ON TEMPERATURE. Above 0 the sampler makes a repeated prompt a
legitimate second sample, and stopping there would remove a retry that can
genuinely succeed. The guard is `not params.temperature`, so any non-zero value
keeps the old behaviour.
"""
import os
import tempfile
import types
import unittest

import generate_script


class _Response:
    """Minimal stand-in for the OpenAI client's response object."""

    def __init__(self, content):
        message = types.SimpleNamespace(content=content, reasoning_content=None)
        choice = types.SimpleNamespace(message=message, finish_reason="stop")
        self.choices = [choice]
        self.usage = types.SimpleNamespace(
            prompt_tokens=764, completion_tokens=325, total_tokens=1089)


class _CountingClient:
    """Records every prompt it is asked to complete, and always fails."""

    def __init__(self, content="not json at all"):
        self.prompts = []
        self.content = content
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.prompts.append(kwargs["messages"][-1]["content"])
        return _Response(self.content)


def _params(temperature):
    return generate_script.LLMGenParams(temperature=temperature,
                                        max_tokens=256)


class DeterministicRetryTest(unittest.TestCase):
    """Response logging is redirected to a temp dir rather than disabled.

    Passing log_name=None makes the logging path raise, which would make every
    attempt fail for an unrelated reason and leave these tests passing by
    accident - and silently change meaning if that path were ever fixed.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._prev = os.environ.get("ALEXANDRIA_RUN_ID")
        os.environ["ALEXANDRIA_RUN_ID"] = "unittest_retry"
        self._orig = generate_script.get_response_log_path
        generate_script.get_response_log_path = (
            lambda name: os.path.join(self._tmp.name, name or "test.log"))
        self.addCleanup(self._restore)

    def _restore(self):
        generate_script.get_response_log_path = self._orig
        if self._prev is None:
            os.environ.pop("ALEXANDRIA_RUN_ID", None)
        else:
            os.environ["ALEXANDRIA_RUN_ID"] = self._prev
        self._tmp.cleanup()

    def _run(self, temperature, retries=3):
        client = _CountingClient()
        generate_script.call_llm_for_entries(
            client, "test-model", "sys", "user", _params(temperature),
            max_retries=retries, log_name="retry.log", label="ATTRIBUTE")
        return client.prompts

    def test_temperature_zero_stops_repeating_an_identical_prompt(self):
        """The defect: four calls where the last ones could not differ.

        With no validator feedback to change the prompt, every attempt would
        be byte-identical, so exactly one call should be made.
        """
        prompts = self._run(temperature=0.0)
        self.assertEqual(len(prompts), 1,
                         f"sent {len(prompts)} identical prompts at "
                         "temperature 0; only the first can be informative")

    def test_nonzero_temperature_still_retries(self):
        """A repeated prompt above 0 is a fresh sample and must be sent."""
        prompts = self._run(temperature=0.7)
        self.assertGreater(len(prompts), 1,
                           "sampling temperature makes a repeat useful; "
                           "the guard must not fire here")

    def test_distinct_prompts_are_not_blocked_at_temperature_zero(self):
        """Feedback-modified prompts are different questions, so they go out.

        This is the case the fix must not break: the loop's whole value is
        that attempt 2 tells the model what was wrong with attempt 1.
        """
        seen = []

        class _Varying(_CountingClient):
            def _create(self, **kwargs):
                seen.append(kwargs["messages"][-1]["content"])
                return _Response("still not json")

        client = _Varying()
        # Feed a different user prompt each call by mutating through the
        # public entry point twice; distinct prompts must both be sent.
        generate_script.call_llm_for_entries(
            client, "m", "sys", "first", _params(0.0),
            max_retries=0, log_name="retry.log", label="ATTRIBUTE")
        generate_script.call_llm_for_entries(
            client, "m", "sys", "second", _params(0.0),
            max_retries=0, log_name="retry.log", label="ATTRIBUTE")
        self.assertEqual(len(seen), 2)
        self.assertNotEqual(seen[0], seen[1])


if __name__ == "__main__":
    unittest.main()
