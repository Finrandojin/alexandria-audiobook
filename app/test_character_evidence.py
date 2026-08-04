"""Tests for reading character gender out of narration.

Three cheaper approaches were tried on the live book and rejected with numbers,
and the tests below encode why, so nobody reintroduces one:

    own dialogue          Subaru read FEMALE - his lines are full of "she" and
                          "her" because he talks about Emilia and Felt.
    pronouns near name    "Subaru looked at her" - 71% masculine, unusable.
    single-name sentences "She waited. Subaru saw her." - still only 74%.

What works is grammatical binding: a reflexive must agree with its clause
subject, and the possessor of a body part in a transitive clause is
overwhelmingly the subject. On the live book that reads Subaru 90/12 MALE,
Emilia 0/9 FEMALE, ROM 9/1 and Reinhard 14/2 - all correct.

Abstention is a feature. FELT (10/15) and SATELLA (5/13) come back "unknown"
rather than being forced, because a wrong answer costs a main character their
voice while silence costs nothing - "unknown" already means "do not filter, do
not penalise" everywhere downstream.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from character_evidence import (aliases_for, gender_from_narration,
                                narration_text)


class TestGrammaticalBinding(unittest.TestCase):

    def test_body_part_possessive_binds_to_the_named_subject(self):
        text = ("Subaru scratched his head. Subaru rubbed his eyes. "
                "Subaru clenched his fists.")
        gender, _, _ = gender_from_narration(text, "Subaru")
        self.assertEqual(gender, "male")

    def test_reflexive_binds_to_the_named_subject(self):
        text = ("Emilia steadied herself. Emilia told herself to breathe. "
                "Emilia found herself alone.")
        gender, _, _ = gender_from_narration(text, "Emilia")
        self.assertEqual(gender, "female")

    def test_object_pronoun_is_not_counted(self):
        # The failure mode of every proximity approach: the pronoun belongs to
        # someone else entirely.
        text = ("Subaru looked at her. Subaru called out to her. "
                "Subaru reached for her.")
        gender, confidence, ev = gender_from_narration(text, "Subaru")
        self.assertEqual(gender, "unknown")
        self.assertEqual(ev["total"], 0)

    def test_another_characters_body_part_does_not_leak(self):
        text = "Emilia watched. Subaru scratched his head. " * 3
        gender, _, _ = gender_from_narration(text, "Emilia")
        self.assertEqual(gender, "unknown")


class TestAbstention(unittest.TestCase):

    def test_thin_evidence_is_unknown(self):
        gender, conf, _ = gender_from_narration("Rom raised his hand.", "Rom")
        self.assertEqual(gender, "unknown")
        self.assertEqual(conf, "unknown")

    def test_mixed_evidence_is_unknown_not_forced(self):
        # An androgynous or non-human character SHOULD land here. Forcing a
        # majority verdict would invent a fact.
        text = ("Puck raised his hand. Puck lowered her head. "
                "Puck shook his head. Puck closed her eyes.")
        gender, _, ev = gender_from_narration(text, "Puck")
        self.assertEqual(gender, "unknown")
        self.assertEqual(ev["total"], 4)

    def test_empty_inputs_are_safe(self):
        self.assertEqual(gender_from_narration("", "X")[0], "unknown")
        self.assertEqual(gender_from_narration("text", "")[0], "unknown")

    def test_confidence_scales_with_evidence(self):
        thin = "A raised his hand. A rubbed his eyes. A shook his head."
        strong = " ".join(["A scratched his head."] * 12)
        self.assertEqual(gender_from_narration(thin, "A")[1], "medium")
        self.assertEqual(gender_from_narration(strong, "A")[1], "high")


class TestAliases(unittest.TestCase):

    def test_evidence_is_pooled_across_spellings(self):
        # 'NATSUKI SUBARU' reads 0/0 alone because the prose says "Subaru".
        text = " ".join(["Subaru scratched his head."] * 5)
        aliases = {"SUBARU": "NATSUKI SUBARU", "Subaru": "NATSUKI SUBARU"}
        gender, _, ev = gender_from_narration(text, "NATSUKI SUBARU",
                                              aliases=aliases)
        self.assertEqual(gender, "male")
        self.assertGreaterEqual(ev["masculine"], 5)

    def test_alias_lookup_is_case_insensitive_and_bidirectional(self):
        aliases = {"SUBARU": "NATSUKI SUBARU"}
        self.assertIn("SUBARU", {a.upper() for a in
                                 aliases_for("NATSUKI SUBARU", aliases)})
        self.assertIn("NATSUKI SUBARU",
                      {a.upper() for a in aliases_for("Subaru", aliases)})

    def test_no_alias_map_is_safe(self):
        self.assertEqual(aliases_for("Subaru", None), set())


class TestNarrationText(unittest.TestCase):

    def test_dialogue_is_excluded(self):
        entries = [{"speaker": "NARRATOR", "text": "He walked."},
                   {"speaker": "Subaru", "text": "She is over there."}]
        joined = narration_text(entries)
        self.assertIn("He walked.", joined)
        self.assertNotIn("She is over there.", joined)

    def test_handles_missing_fields(self):
        entries = [{"speaker": "NARRATOR"}, {"text": "orphan"}, "junk"]
        self.assertEqual(narration_text(entries), "")


if __name__ == "__main__":
    unittest.main()
