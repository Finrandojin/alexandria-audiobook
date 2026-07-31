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
"""
import importlib.util
import json
import os
import sys
import types

import pytest

APP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP)

from generate_script import LLMGenParams
from three_pass_generate import attribute_batch

BATCH = [{"type": "SPOKEN", "text": "Where are we going?"},
         {"type": "SPOKEN", "text": "Somewhere quieter."}]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "distill_eval", os.path.join(APP, "experiments", "distill_eval.py"))
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


@pytest.fixture(autouse=True)
def _stub_torch(monkeypatch):
    """The shim imports torch only for no_grad; nothing here needs the real one."""
    if "torch" in sys.modules:
        return
    torch = types.ModuleType("torch")

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *exc):
            return False

    torch.no_grad = lambda: _NoGrad()
    monkeypatch.setitem(sys.modules, "torch", torch)


def _params():
    return LLMGenParams(max_tokens=800, context_length=32768, temperature=0.0,
                        attribute_temperature=0.0, top_p=0.8,
                        reasoning_effort="none")


def test_shim_drives_attribute_batch_and_freezes_text():
    module = _load_module()
    reply = json.dumps([{"n": 0, "head": "Where", "speaker": "HARUHIRO"},
                        {"n": 1, "head": "Somewhere", "speaker": "RANTA"}])
    client = module.LocalClient(_Model(), _Tok(reply))
    out = attribute_batch(client, "stub", BATCH, _params(),
                          ["HARUHIRO", "RANTA"], neighbor_contexts=[{}, {}],
                          source_text=" ".join(e["text"] for e in BATCH))

    assert [o["speaker"] for o in out] == ["HARUHIRO", "RANTA"]
    # The text freeze is the whole reason attribution returns only n/head/
    # speaker; a shim that mangled the response could still round-trip text.
    assert [o["text"] for o in out] == [e["text"] for e in BATCH]


def test_temperature_zero_is_greedy_not_sampled():
    """Every other harness here ran deterministic; a sampled arm is not
    comparable to any of them."""
    module = _load_module()
    model = _Model()
    client = module.LocalClient(model, _Tok("[]"))
    client.create(messages=[{"role": "user", "content": "x"}], temperature=0.0,
                  max_tokens=16)
    assert model.kwargs["do_sample"] is False
    assert "temperature" not in model.kwargs

    client.create(messages=[{"role": "user", "content": "x"}], temperature=0.7,
                  max_tokens=16)
    assert model.kwargs["do_sample"] is True
    assert model.kwargs["temperature"] == 0.7


def test_truncated_generation_reports_finish_reason_length():
    """call_llm_for_entries' retry policy branches on finish_reason=='length'.
    A shim that always said 'stop' would turn a truncated response into an
    unexplained parse failure."""
    module = _load_module()
    client = module.LocalClient(_Model(), _Tok("["))
    # _Model returns 6 tokens for a 3-token prompt, so 3 were generated.
    assert client.create(messages=[], max_tokens=3).choices[0].finish_reason == "length"
    assert client.create(messages=[], max_tokens=99).choices[0].finish_reason == "stop"


def test_response_exposes_only_what_the_caller_reads():
    module = _load_module()
    response = module.LocalClient(_Model(), _Tok("[]")).create(
        messages=[], max_tokens=8)
    assert isinstance(response.choices[0].message.content, str)
    # getattr(response, 'usage', None) is how the caller reads it; None is a
    # value the caller already handles, and inventing token counts would put
    # fabricated numbers into an artifact.
    assert response.usage is None
