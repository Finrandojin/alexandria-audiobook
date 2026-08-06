"""The training learning rate must be one value, not four.

WHAT THIS PROTECTS. The rate was defined independently in four places -
train_lora.py's CLI, the API request model, the UI input, and
batch_train_lora.py - and they disagreed: the first three said 5e-6 and only
batch_train_lora said 1e-6. Every adapter in the library that works was
trained through the one that said 1e-6.

THE EVIDENCE, measured 2026-08-06 over five adapters and two conditions:

    5e-6   2 of 2 ran away. 163.8 seconds of audio for a 7.3 second line, on
           every render, hitting the token ceiling and never emitting an end
           of speech.
    1e-6   3 of 3 stopped correctly, median duration 1.01x, 0.87x and 0.94x
           the human reading, in English, Japanese and Chinese.

WHY A TEST AND NOT JUST A FIX. Training loss does not show the failure - the
runaway adapters reached 2.9 and 3.4, which look ordinary - so a drifted
default is invisible until a full training run and the generation run after it
have both been spent. Four independent copies of a number will drift again;
this asserts they cannot.

The two Python entry points now import the constant, so they cannot disagree.
The UI input and batch_train_lora.py cannot import it - one is HTML, and the
other deliberately imports no third-party module at top level - so they are
checked by reading them.
"""
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_lora import DEFAULT_LEARNING_RATE

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.dirname(os.path.abspath(__file__))


class LearningRateIsSingleSourced(unittest.TestCase):

    def test_the_constant_is_the_rate_that_produced_working_adapters(self):
        """1e-6 is not a preference. Every adapter that stops was trained at
        it, and both trained at 5e-6 ran away."""
        self.assertEqual(DEFAULT_LEARNING_RATE, 1e-6)

    def test_the_cli_uses_the_constant(self):
        """Parsed from a real argv rather than read off the source, so a
        literal reintroduced anywhere in the chain is caught."""
        import train_lora
        argv = sys.argv
        sys.argv = ["train_lora.py", "--data_dir", "d", "--output_dir", "o"]
        try:
            args = train_lora.parse_args()
        finally:
            sys.argv = argv
        self.assertEqual(args.lr, DEFAULT_LEARNING_RATE)

    def test_the_api_request_model_uses_the_constant(self):
        from routers.lora import LoraTrainingRequest
        req = LoraTrainingRequest(name="n", dataset_id="d")
        self.assertEqual(req.lr, DEFAULT_LEARNING_RATE)

    def test_the_api_still_permits_an_explicit_override(self):
        """Consolidating the default must not remove the choice - a
        researcher deliberately sweeping the rate is a supported use."""
        from routers.lora import LoraTrainingRequest
        req = LoraTrainingRequest(name="n", dataset_id="d", lr=5e-6)
        self.assertEqual(req.lr, 5e-6)

    def test_the_ui_input_matches(self):
        """The Training tab's prefilled value is what most adapters are
        actually trained at, so it is the one that matters most."""
        with open(os.path.join(APP, "static", "index.html"),
                  encoding="utf-8") as fh:
            html = fh.read()
        found = re.search(r'id="lora-lr"[^>]*value="([^"]+)"', html)
        self.assertIsNotNone(found, "the Training tab's lr input moved")
        self.assertEqual(float(found.group(1)), DEFAULT_LEARNING_RATE)

    def test_batch_train_lora_matches(self):
        path = os.path.join(REPO, "batch_train_lora.py")
        if not os.path.exists(path):
            self.fail("batch_train_lora.py is missing; it drives the Voice Lab "
                      "training stage and its default is load-bearing")
        with open(path, encoding="utf-8") as fh:
            source = fh.read()
        found = re.search(r'"--lr",\s*type=float,\s*default=([0-9.e-]+)', source)
        self.assertIsNotNone(found, "batch_train_lora's --lr default moved")
        self.assertEqual(float(found.group(1)), DEFAULT_LEARNING_RATE)

    def test_no_bare_five_e_minus_six_default_remains(self):
        """The specific wrong value, in the specific place it did damage. A
        deliberate --lr 5e-6 on a command line is fine; a DEFAULT is not."""
        offenders = []
        for name in ("train_lora.py", "routers/lora.py"):
            with open(os.path.join(APP, name), encoding="utf-8") as fh:
                for i, line in enumerate(fh, 1):
                    if re.search(r'default\s*=\s*5e-0?6', line):
                        offenders.append(f"{name}:{i}")
        self.assertEqual(offenders, [], f"5e-6 reintroduced as a default: {offenders}")


if __name__ == "__main__":
    unittest.main()
