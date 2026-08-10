"""A loaded config must be writable back out unchanged.

WHAT BROKE. `load_app_config` validates some sections into pydantic models -
`generation.three_pass_model_profiles` becomes
Dict[str, ThreePassModelProfile]. Ten modules load the config and write it back
(llm_bench's concurrency cache, generate_personas, find_nicknames,
review_script, project, three_pass_generate, core, ...), and every one of them
raised

    TypeError: Object of type ThreePassModelProfile is not JSON serializable

as soon as a profile was configured. It surfaced running --dedupe-speakers on
index18: the run died inside a cache write that its own comment calls
"best-effort".

three_pass_generate had already worked around this locally with
`as_profile_mapping`, whose docstring records the same crash class. A
per-module workaround leaves the other nine broken, so the conversion belongs
where the data is produced.

TWO THINGS ARE PINNED HERE. That the loaded config is JSON-native, and that
None-valued profile fields stay dropped - an unset key must fall through to the
caller's default rather than override it with None, which is the semantic
`as_profile_mapping` established and which callers rely on.
"""
import json
import os
import tempfile
import unittest

from config_settings import load_app_config
from three_pass_generate import as_profile_mapping

PROFILED_CONFIG = {
    "generation": {
        "chunk_size": 6000,
        "three_pass_model_profiles": {
            "some-model": {
                "segment_output_ratio": 3.5,
                "attribute_temperature": None,
            }
        },
    },
    "llm_local": {"base_url": "http://127.0.0.1:8090/v1",
                  "model_name": "test-model"},
}


class ConfigIsJsonNativeTest(unittest.TestCase):

    def setUp(self):
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(PROFILED_CONFIG, handle)
        handle.close()
        self.path = handle.name
        self.addCleanup(os.unlink, self.path)

    def test_a_loaded_config_can_be_written_back(self):
        """The exact failure: load, then serialise."""
        config = load_app_config(self.path)
        try:
            json.dumps(config)
        except TypeError as exc:                            # pragma: no cover
            self.fail(f"loaded config is not JSON-serializable: {exc}")

    def test_profiles_come_back_as_plain_dicts(self):
        config = load_app_config(self.path)
        profiles = config["generation"]["three_pass_model_profiles"]
        self.assertIsInstance(profiles["some-model"], dict)

    def test_unset_profile_fields_are_dropped_not_none(self):
        """An unset key must fall through to the caller's default.

        `attribute_temperature` is None in the fixture. If it survived as None
        the caller's `profile.get("attribute_temperature", default)` would
        return None and override the default with nothing.
        """
        config = load_app_config(self.path)
        profile = config["generation"]["three_pass_model_profiles"]["some-model"]
        self.assertNotIn("attribute_temperature", profile)
        self.assertEqual(3.5, profile["segment_output_ratio"])

    def test_the_local_workaround_still_accepts_the_new_shape(self):
        """three_pass_generate's helper must keep working on plain dicts."""
        config = load_app_config(self.path)
        profile = config["generation"]["three_pass_model_profiles"]["some-model"]
        mapping = as_profile_mapping(profile)
        self.assertIsInstance(mapping, dict)
        self.assertEqual(3.5, mapping["segment_output_ratio"])

    def test_a_config_without_profiles_is_unaffected(self):
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8")
        json.dump({"llm_local": {"model_name": "m"}}, handle)
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        config = load_app_config(handle.name)
        json.dumps(config)
        self.assertEqual("m", config["llm_local"]["model_name"])


if __name__ == "__main__":
    unittest.main()
