"""Tests for the voice-config split repair.

This script rewrites a file the user hand-tunes through the UI, so every way it
could quietly pick the wrong voice is worth a test. The bug it repairs was
itself silent: eight characters in the live book cast in two voices, invisible
because both spellings resolved exactly through `voice_config.get(speaker)`.

  case-insensitive canon    'SUBARU' was in the alias map and resolved; 'Subaru'
                            was not. Case-sensitive matching is the whole
                            reason the split existed.
  same-voice not flagged    Two spellings sharing a voice are harmless.
                            Reporting them would bury the ones that matter.
  type outranks lines       Ranking by line count first gave PUCK an
                            auto-created custom voice over a character LoRA on
                            a 1-vs-0 count.
  disputed surfaced         Where the two rules disagree the answer is
                            arguable and must be visible, not silent.
  losers keep their keys    The script still refers to characters by their
                            original spelling. Deleting a key would send those
                            lines to a fallback - a wrong voice traded for none.
  no mutation of input      Rule 17: apply_merges returns a new dict rather
                            than editing the config it was handed.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from repair_voice_config import (apply_merges, canonical, find_splits,
                                 voice_signature)

LORA = {"type": "lora", "voice": "Ryan", "seed": "-1"}
CUSTOM = {"type": "custom", "voice": "Aiden", "seed": "-1"}
DESIGN = {"type": "design", "voice": "Ryan", "seed": "-1"}
CLONE = {"type": "clone", "voice": "Ryan", "seed": "-1"}


class TestCanonical(unittest.TestCase):

    def test_alias_map_is_case_insensitive(self):
        # The live bug: the map held 'SUBARU' and the script said 'Subaru'.
        aliases = {"SUBARU": "NATSUKI SUBARU"}
        self.assertEqual(canonical("Subaru", aliases), "NATSUKI SUBARU")
        self.assertEqual(canonical("SUBARU", aliases), "NATSUKI SUBARU")

    def test_unmapped_name_folds_to_upper(self):
        self.assertEqual(canonical("Anastasia", {}), "ANASTASIA")
        self.assertEqual(canonical("ANASTASIA", {}), "ANASTASIA")

    def test_empty_and_missing_are_safe(self):
        self.assertEqual(canonical("", {}), "")
        self.assertEqual(canonical(None, None), "")


class TestFindSplits(unittest.TestCase):

    def test_same_voice_under_two_spellings_is_not_a_split(self):
        # Subaru is duplicated but both point at one voice - not a defect.
        config = {"Subaru": dict(LORA), "NATSUKI SUBARU": dict(LORA)}
        aliases = {"SUBARU": "NATSUKI SUBARU"}
        self.assertEqual(find_splits(config, aliases, {}), [])

    def test_different_voices_under_two_spellings_is_a_split(self):
        config = {"Anastasia": dict(LORA), "ANASTASIA": dict(CUSTOM)}
        splits = find_splits(config, {}, {"Anastasia": 68, "ANASTASIA": 2})
        self.assertEqual(len(splits), 1)
        self.assertEqual(splits[0]["canonical"], "ANASTASIA")
        self.assertEqual(splits[0]["winner"], "Anastasia")

    def test_voice_type_outranks_line_count(self):
        # PUCK: the LoRA has zero lines and must still win.
        config = {"Puck": dict(LORA), "PUCK": dict(CUSTOM)}
        splits = find_splits(config, {}, {"PUCK": 1, "Puck": 0})
        self.assertEqual(splits[0]["winner"], "Puck")

    def test_clone_outranks_custom(self):
        config = {"New Voice": dict(CLONE), "NEW VOICE": dict(CUSTOM)}
        splits = find_splits(config, {}, {"NEW VOICE": 1, "New Voice": 0})
        self.assertEqual(splits[0]["winner"], "New Voice")

    def test_disagreement_between_rules_is_flagged(self):
        config = {"Man 2": dict(DESIGN), "MAN 2": dict(CUSTOM)}
        splits = find_splits(config, {}, {"MAN 2": 8, "Man 2": 4})
        self.assertEqual(splits[0]["winner"], "Man 2")
        self.assertTrue(splits[0]["disputed"])
        self.assertIn("more lines", splits[0]["reason"])

    def test_agreement_is_not_flagged_disputed(self):
        config = {"Anastasia": dict(LORA), "ANASTASIA": dict(CUSTOM)}
        splits = find_splits(config, {}, {"Anastasia": 68, "ANASTASIA": 2})
        self.assertFalse(splits[0]["disputed"])

    def test_line_count_breaks_ties_within_a_type(self):
        config = {"Man A": dict(CUSTOM), "MAN A": dict(CUSTOM)}
        # Same type but different voices, so still a split.
        config["MAN A"] = {"type": "custom", "voice": "Zed", "seed": "-1"}
        splits = find_splits(config, {}, {"MAN A": 9, "Man A": 1})
        self.assertEqual(splits[0]["winner"], "MAN A")

    def test_non_dict_entries_are_ignored(self):
        # Older configs stored a bare voice name; it must not crash the scan.
        config = {"Legacy": "Ryan", "Anastasia": dict(LORA),
                  "ANASTASIA": dict(CUSTOM)}
        splits = find_splits(config, {}, {})
        self.assertEqual([s["canonical"] for s in splits], ["ANASTASIA"])


class TestApplyMerges(unittest.TestCase):

    def test_losers_adopt_the_winner_voice(self):
        config = {"Anastasia": dict(LORA), "ANASTASIA": dict(CUSTOM)}
        splits = find_splits(config, {}, {"Anastasia": 68, "ANASTASIA": 2})
        merged = apply_merges(config, splits)
        self.assertEqual(voice_signature(merged["ANASTASIA"]),
                         voice_signature(LORA))

    def test_loser_keys_are_kept_not_deleted(self):
        # The script still says 'ANASTASIA'; dropping the key would send those
        # lines to a fallback voice instead of the right one.
        config = {"Anastasia": dict(LORA), "ANASTASIA": dict(CUSTOM)}
        splits = find_splits(config, {}, {"Anastasia": 68, "ANASTASIA": 2})
        self.assertIn("ANASTASIA", apply_merges(config, splits))

    def test_input_config_is_not_mutated(self):
        config = {"Anastasia": dict(LORA), "ANASTASIA": dict(CUSTOM)}
        splits = find_splits(config, {}, {"Anastasia": 68, "ANASTASIA": 2})
        apply_merges(config, splits)
        self.assertEqual(config["ANASTASIA"]["type"], "custom")

    def test_merged_entries_are_independent_copies(self):
        # A shared dict would make a later edit to one character silently
        # change another.
        config = {"Anastasia": dict(LORA), "ANASTASIA": dict(CUSTOM)}
        splits = find_splits(config, {}, {"Anastasia": 68, "ANASTASIA": 2})
        merged = apply_merges(config, splits)
        merged["ANASTASIA"]["voice"] = "CHANGED"
        self.assertEqual(merged["Anastasia"]["voice"], "Ryan")

    def test_untouched_characters_survive(self):
        config = {"Anastasia": dict(LORA), "ANASTASIA": dict(CUSTOM),
                  "NARRATOR": dict(DESIGN)}
        splits = find_splits(config, {}, {"Anastasia": 68, "ANASTASIA": 2})
        merged = apply_merges(config, splits)
        self.assertEqual(voice_signature(merged["NARRATOR"]),
                         voice_signature(DESIGN))


if __name__ == "__main__":
    unittest.main()
