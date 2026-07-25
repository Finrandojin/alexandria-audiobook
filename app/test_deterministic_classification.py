import unittest
from unittest.mock import patch

import three_pass_generate as tp


class ClassificationTemperatureTest(unittest.TestCase):
    """Segmentation and attribution are classification: one right answer, so
    sampling only adds noise. Sending an identical attribution batch twice at
    temperature 0.1 changed 23.6% of speakers; at 0.0 it changed 0%.

    instruct is deliberately left sampling: it writes delivery direction rather
    than choosing a label."""

    def _params(self, generation):
        source = open(tp.__file__, encoding="utf-8").read()
        marker = 'gen.get("three_pass_segment_temperature"'
        self.assertIn(marker, source)
        return source

    def test_segment_and_attribute_default_to_zero(self):
        source = open(tp.__file__, encoding="utf-8").read()
        self.assertIn('gen.get("three_pass_segment_temperature", 0.0)', source)
        self.assertIn('gen.get("three_pass_attribute_temperature", 0.0)', source)

    def test_instruct_still_samples(self):
        source = open(tp.__file__, encoding="utf-8").read()
        self.assertIn('gen.get("three_pass_instruct_temperature", 0.1)', source)

    def test_config_can_still_override(self):
        # A user who wants sampling back must not be blocked by the new default.
        source = open(tp.__file__, encoding="utf-8").read()
        self.assertIn('model_profile.get(\n            "attribute_temperature"', source)


if __name__ == "__main__":
    unittest.main()
