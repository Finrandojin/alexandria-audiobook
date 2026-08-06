"""A configured adapter that is not actually serving must be noticed.

The measured stake: the distilled adapter is +9.8 points cross-book and +14.6
through the served path. A server started without it, or with its scale at
zero, answers every request happily at base quality - nothing fails, the book
generates, it is simply much worse and no artifact says why.

That is the same shape as the seed bug: a configured field silently ignored,
which cost six contaminated comparisons before anyone noticed by ear. These
tests cover the ways it could go unnoticed again.
"""
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from attribution_adapter import (AdapterError, adapter_config, check_adapter,
                                 describe)

CFG = {"llm_mode": "local",
       "llm_local": {"base_url": "http://x/v1",
                     "attribution_adapter": {"path": "/m/adapter_mixed.gguf",
                                             "scale": 1.0}}}


def served(entries):
    return patch("attribution_adapter.served_adapters", return_value=entries)


class ConfigTest(unittest.TestCase):

    def test_it_reads_from_the_active_mode_block(self):
        """The adapter travels with the endpoint it belongs to. A second
        location could disagree with base_url about which server is meant."""
        spec = adapter_config(CFG)
        self.assertEqual(spec["path"], "/m/adapter_mixed.gguf")
        self.assertEqual(spec["scale"], 1.0)

    def test_remote_mode_reads_the_remote_block(self):
        cfg = {"llm_mode": "remote",
               "llm_local": {"attribution_adapter": {"path": "/local.gguf"}},
               "llm_remote": {"attribution_adapter": {"path": "/remote.gguf"}}}
        self.assertEqual(adapter_config(cfg)["path"], "/remote.gguf")

    def test_no_adapter_configured_is_valid(self):
        self.assertIsNone(adapter_config({"llm_mode": "local",
                                          "llm_local": {}}))
        ok, msg = check_adapter({"llm_mode": "local", "llm_local": {}}, "http://x")
        self.assertTrue(ok)


class ServingTest(unittest.TestCase):

    def test_the_configured_adapter_serving_is_ok(self):
        with served([{"path": "/srv/adapter_mixed.gguf", "scale": 1.0}]):
            ok, msg = check_adapter(CFG, "http://x/v1")
        self.assertTrue(ok)
        self.assertIn("adapter_mixed.gguf", msg)

    def test_a_missing_adapter_is_reported(self):
        with served([{"path": "/srv/something_else.gguf", "scale": 1.0}]):
            ok, msg = check_adapter(CFG, "http://x/v1")
        self.assertFalse(ok)
        self.assertIn("NOT loaded", msg)
        self.assertIn("something_else.gguf", msg,
                      "the message should say what IS loaded")

    def test_no_adapters_at_all_is_reported(self):
        with served([]):
            ok, msg = check_adapter(CFG, "http://x/v1")
        self.assertFalse(ok)

    def test_scale_zero_is_reported(self):
        """THE QUIET ONE. The adapter is loaded, the name matches, and it
        contributes nothing - generation runs at base quality while every
        surface check passes."""
        with served([{"path": "/srv/adapter_mixed.gguf", "scale": 0.0}]):
            ok, msg = check_adapter(CFG, "http://x/v1")
        self.assertFalse(ok)
        self.assertIn("scale", msg)
        self.assertIn("BASE quality", msg)

    def test_it_matches_on_basename_not_full_path(self):
        """The server's copy lives somewhere else than ours; requiring an
        identical absolute path would fail on every real deployment."""
        with served([{"path": "/opt/models/adapter_mixed.gguf", "scale": 1.0}]):
            ok, _ = check_adapter(CFG, "http://x/v1")
        self.assertTrue(ok)

    def test_an_endpoint_that_cannot_be_asked_is_UNKNOWN_not_absent(self):
        """LM Studio and Ollama do not implement /lora-adapters. Treating
        silence as 'missing' would cry wolf on every non-llama.cpp setup, and a
        warning that always fires is one nobody reads."""
        with served(None):
            ok, msg = check_adapter(CFG, "http://x/v1")
        self.assertFalse(ok)
        self.assertIn("cannot verify", msg)


class RequireTest(unittest.TestCase):

    def test_require_raises(self):
        with served([]):
            with self.assertRaises(AdapterError):
                check_adapter(CFG, "http://x/v1", require=True)

    def test_the_default_warns_rather_than_dying(self):
        """A book half-generated overnight should not die because a server was
        restarted without its adapter - but the run must carry the fact."""
        with served([]):
            ok, msg = check_adapter(CFG, "http://x/v1")
        self.assertFalse(ok)
        self.assertTrue(msg)

    def test_require_can_be_set_in_config(self):
        cfg = {"llm_mode": "local",
               "llm_local": {"attribution_adapter":
                             {"path": "/a.gguf", "require": True}}}
        with served([]):
            with self.assertRaises(AdapterError):
                check_adapter(cfg, "http://x/v1")


class DescribeTest(unittest.TestCase):

    def test_it_records_what_was_expected(self):
        self.assertIn("adapter_mixed.gguf", describe(CFG))
        self.assertEqual(describe({}), "attribution_adapter=none")


class WiringTest(unittest.TestCase):

    def test_generate_script_performs_the_check(self):
        """A correct module nothing calls is the state this replaces."""
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "generate_script.py")
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("check_adapter", src)
        self.assertIn("attribution_adapter", src)


if __name__ == "__main__":
    unittest.main()
