import unittest

from narrator_prompt import (add_narrator_prior, get_valid_narrator_name,
                             is_narrator_attested)


class NarratorPromptTests(unittest.TestCase):
    def test_valid_name_is_canonical_and_blank_disables_hint(self):
        self.assertEqual("ALEXIS IVANOVITCH",
                         get_valid_narrator_name(" Alexis   Ivanovitch "))
        self.assertIsNone(get_valid_narrator_name("  "))

    def test_placeholder_labels_are_not_character_names(self):
        for value in ("NARRATOR", "unknown"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                get_valid_narrator_name(value)

    def test_attestation_requires_three_source_mentions(self):
        source = "Alexis entered. Alexis spoke. Then Alexis left."
        self.assertTrue(is_narrator_attested("Alexis", source))
        self.assertFalse(is_narrator_attested("Ivanovitch", source))

    def test_prior_names_narrator_without_mutating_base_prompt(self):
        base = "Assign every line."
        prompt = add_narrator_prior(base, "alexis")

        self.assertEqual("Assign every line.", base)
        self.assertIn("first person by ALEXIS", prompt)
        self.assertIn("ALEXIS's own voice", prompt)


if __name__ == "__main__":
    unittest.main()
