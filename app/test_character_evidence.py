"""Tests for reading character gender out of narration.

Three cheaper approaches were tried on the live book and rejected with numbers,
and the tests below encode why, so nobody reintroduces one:

    own dialogue          Subaru read FEMALE - his lines are full of "she" and
                          "her" because he talks about Emilia and Felt.
    pronouns near name    "Subaru looked at her" - 71% masculine, unusable.
    single-name sentences "She waited. Subaru saw her." - still only 74%.

What works better is two constructions that USUALLY bind to the clause subject
- a reflexive, and the possessor of a body part - plus an INTERVENING-NAME rule
that discards a match when another known character is named in between. The
constructions alone were not enough: an external review reproduced "Subaru
watched Emilia raise her hand" scoring feminine for Subaru, so the original
claim that they "cannot float to another referent" was wrong.

With the rule and a roster, on the live book: Subaru 83/11 MALE, Reinhard 14/1
MALE, ROM 9/0 MALE, Emilia 0/9 FEMALE, SATELLA 3/12 FEMALE - all correct, and
cleaner than without it.

Abstention is a feature. FELT (9/14) comes back "unknown" rather than being
forced, because a wrong answer costs a main character their voice while silence
costs nothing - "unknown" already means "do not filter, do not penalise"
everywhere downstream.
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


class TestInterveningName(unittest.TestCase):
    """The rule that separates this from plain proximity.

    An external review reproduced the defect it fixes: "Subaru watched Emilia
    raise her hand" repeated three times returned female for Subaru, because
    the regex only required the name to precede `her <body part>` within a
    window. A nearer named character is the likelier subject, so the match is
    discarded.
    """

    REPRO = "Subaru watched Emilia raise her hand. " * 3

    def test_another_named_character_blocks_the_match(self):
        gender, _, ev = gender_from_narration(self.REPRO, "Subaru",
                                              roster=["Emilia"])
        self.assertEqual(gender, "unknown")
        self.assertEqual(ev["feminine"], 0)

    def test_the_rule_holds_without_a_roster(self):
        # This test previously asserted the OPPOSITE - that with no roster the
        # function fell back to proximity and answered "female" for Subaru.
        # That degradation was documented rather than fixed, and an external
        # review was right that a rule which only works when the caller already
        # supplies the answer is barely a rule. Capitalised non-sentence-initial
        # tokens are now treated as intervening characters, so the review's own
        # reproduction abstains with no roster at all.
        gender, _, ev = gender_from_narration(self.REPRO, "Subaru")
        self.assertEqual(gender, "unknown")
        self.assertEqual(ev["feminine"], 0)

    def test_the_true_referent_is_still_attributed(self):
        # Abstaining for Subaru must not cost Emilia her evidence; a rule that
        # blocked both would be safe and useless.
        self.assertEqual(gender_from_narration(self.REPRO, "Emilia")[0],
                         "female")

    def test_a_sentence_initial_capital_is_not_a_name(self):
        # Every sentence starts capitalised. Treating those as intervening
        # characters would block essentially all evidence.
        text = "Subaru shook his head. Rain fell. " * 3
        self.assertEqual(gender_from_narration(text, "Subaru")[0], "male")

    def test_a_stoplisted_capital_does_not_block(self):
        # A capitalised weekday sitting inside the window is not a character,
        # and blocking on it would throw away good evidence. This is what the
        # stoplist is for.
        text = "Subaru on Monday shook his head. " * 3
        self.assertEqual(gender_from_narration(text, "Subaru")[0], "male")

    def test_only_a_name_BETWEEN_target_and_construction_blocks(self):
        # The rule is directional and scoped, and both halves matter.
        #
        # A name AFTER the construction, or outside the window, is not evidence
        # about who "his" refers to - blocking on it would abstain on nearly
        # every sentence in a populated scene.
        far = ("Subaru shook his head. " + "The road went on for miles. " * 3
               + "Emilia was elsewhere. ") * 3
        self.assertEqual(gender_from_narration(far, "Subaru")[0], "male")

        # A name BEFORE the target is not intervening either. In "Emilia saw
        # Subaru shake his head" the pronoun really does bind to Subaru, and
        # answering "male" is correct rather than lucky.
        before = "Emilia saw Subaru shake his head. " * 3
        self.assertEqual(gender_from_narration(before, "Subaru")[0], "male")

        # Only the sandwiched case is ambiguous, and that is the review's
        # reproduction asserted above.

    def test_direct_binding_survives_the_rule(self):
        text = "Subaru scratched his head. " * 3
        self.assertEqual(
            gender_from_narration(text, "Subaru", roster=["Emilia"])[0], "male")

    def test_unrelated_roster_names_do_not_block(self):
        text = "Subaru scratched his head. " * 3
        self.assertEqual(
            gender_from_narration(text, "Subaru",
                                  roster=["Reinhard", "Felt"])[0], "male")

    def test_reflexive_is_blocked_too(self):
        text = "Subaru watched Emilia steady herself. " * 3
        gender, _, ev = gender_from_narration(text, "Subaru", roster=["Emilia"])
        self.assertEqual(ev["feminine"], 0)
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
