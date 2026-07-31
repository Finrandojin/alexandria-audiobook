"""The distill_eval shim must satisfy the real LLM path, with no GPU.

`distill_eval` runs a peft adapter through `attribute_batch` behind a small
object that imitates the part of the OpenAI client `call_llm_for_entries`
actually uses. If that imitation drifts - a renamed attribute, a changed
finish_reason contract - every scored row silently becomes a failed batch, and
the run would report the adapter as catastrophic rather than the harness as
broken. That failure is invisible without a GPU and a loaded 14B unless it is
tested here.

These drive the REAL attribute_batch and assert on what it produced: correct
arity, speakers bound to the right indices, and the frozen text byte-exact.

Imports are deliberately lazy. `update_test_inventory` imports every test
module with plain importlib in an environment that has neither pytest nor
openai, and a top-level import of either fails the release verifier rather
than the test.
"""
import importlib.util
import json
import os
import sys
import types
import unittest

APP = os.path.dirname(os.path.abspath(__file__))
if APP not in sys.path:
    sys.path.insert(0, APP)

BATCH = [{"type": "SPOKEN", "text": "Where are we going?"},
         {"type": "SPOKEN", "text": "Somewhere quieter."}]


def _dependencies():
    """Import the production path, or None when the environment lacks it."""
    try:
        from generate_script import LLMGenParams
        from three_pass_generate import attribute_batch
    except Exception:
        return None
    return LLMGenParams, attribute_batch


def _load_distill_eval():
    spec = importlib.util.spec_from_file_location(
        "distill_eval_under_test",
        os.path.join(APP, "experiments", "distill_eval.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Tok:
    pad_token_id = 0
    eos_token_id = 0

    def __init__(self, reply):
        self.reply = reply
        self.messages = None

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False):
        self.messages = messages
        return json.dumps(messages)

    def __call__(self, text, return_tensors=None):
        class Ids(list):
            shape = (1, 3)

        class Enc(dict):
            def to(self, device):
                return self

        return Enc(input_ids=Ids([[1, 2, 3]]))

    def decode(self, generated, skip_special_tokens=True):
        return self.reply


class _Model:
    device = "cpu"

    def __init__(self):
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return [[1, 2, 3, 4, 5, 6]]


class DistillEvalShimTest(unittest.TestCase):
    """The shim is the seam between a peft model and the shipped LLM path."""

    @classmethod
    def setUpClass(cls):
        if "torch" not in sys.modules:
            # The shim imports torch only for no_grad; nothing here needs the
            # real one, and CI has no GPU stack.
            torch = types.ModuleType("torch")

            class _NoGrad:
                def __enter__(self):
                    return None

                def __exit__(self, *exc):
                    return False

            torch.no_grad = lambda: _NoGrad()
            sys.modules["torch"] = torch
        cls.deps = _dependencies()
        cls.module = _load_distill_eval() if cls.deps else None

    def setUp(self):
        if not self.deps:
            self.skipTest("openai/production path unavailable in this environment")

    def _params(self):
        LLMGenParams, _ = self.deps
        return LLMGenParams(max_tokens=800, context_length=32768,
                            temperature=0.0, attribute_temperature=0.0,
                            top_p=0.8, reasoning_effort="none")

    def test_shim_drives_attribute_batch_and_freezes_text(self):
        _, attribute_batch = self.deps
        reply = json.dumps([{"n": 0, "head": "Where", "speaker": "HARUHIRO"},
                            {"n": 1, "head": "Somewhere", "speaker": "RANTA"}])
        client = self.module.LocalClient(_Model(), _Tok(reply))
        out = attribute_batch(client, "stub", BATCH, self._params(),
                              ["HARUHIRO", "RANTA"], neighbor_contexts=[{}, {}],
                              source_text=" ".join(e["text"] for e in BATCH))

        self.assertEqual([o["speaker"] for o in out], ["HARUHIRO", "RANTA"])
        # The text freeze is the whole reason attribution returns only n/head/
        # speaker; a shim that mangled the response could still round-trip text.
        self.assertEqual([o["text"] for o in out], [e["text"] for e in BATCH])

    def test_temperature_zero_is_greedy_not_sampled(self):
        """Every other harness here ran deterministic; a sampled arm is not
        comparable to any of them."""
        model = _Model()
        client = self.module.LocalClient(model, _Tok("[]"))
        client.create(messages=[{"role": "user", "content": "x"}],
                      temperature=0.0, max_tokens=16)
        self.assertIs(model.kwargs["do_sample"], False)
        self.assertNotIn("temperature", model.kwargs)

        client.create(messages=[{"role": "user", "content": "x"}],
                      temperature=0.7, max_tokens=16)
        self.assertIs(model.kwargs["do_sample"], True)
        self.assertEqual(model.kwargs["temperature"], 0.7)

    def test_truncated_generation_reports_finish_reason_length(self):
        """call_llm_for_entries' retry policy branches on
        finish_reason=='length'. A shim that always said 'stop' would turn a
        truncated response into an unexplained parse failure."""
        client = self.module.LocalClient(_Model(), _Tok("["))
        # _Model returns 6 tokens for a 3-token prompt, so 3 were generated.
        self.assertEqual(
            client.create(messages=[], max_tokens=3).choices[0].finish_reason,
            "length")
        self.assertEqual(
            client.create(messages=[], max_tokens=99).choices[0].finish_reason,
            "stop")

    def test_response_exposes_only_what_the_caller_reads(self):
        response = self.module.LocalClient(_Model(), _Tok("[]")).create(
            messages=[], max_tokens=8)
        self.assertIsInstance(response.choices[0].message.content, str)
        # getattr(response, 'usage', None) is how the caller reads it; None is
        # a value the caller already handles, and inventing token counts would
        # put fabricated numbers into an artifact.
        self.assertIsNone(response.usage)


if __name__ == "__main__":
    unittest.main()
