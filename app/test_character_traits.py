"""Tests for character trait inference used in voice casting.

The defect these guard against was measured on the live book, not imagined:
a character's own dialogue was treated as evidence about that character, but
the gendered words someone SPEAKS describe whoever they are talking about.

    Subaru, 412 lines: 46 feminine tokens vs 27 masculine
                       ("she" x20, "her" x11, "girl" x10)
                       -> classified FEMALE

Same for ROM and Reinhard; Emilia came out male. The signal is inverted for
precisely the characters who speak most, and it fed a soft penalty in
get_voice_allocation that pushed correct-gender voices down the ranking.

Determining this properly needs coreference - which pronoun refers to whom -
so the fix is to stop guessing from the wrong text and return "unknown", which
every consumer already handles as "do not filter, do not penalise".
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from routers.voices import _infer_character_traits, _infer_character_gender

# A male character talking about female characters, which is the real shape of
# the bug rather than a contrived string.
SUBARU_LINES = [
    "Emilia, are you all right? She looked like she was about to collapse.",
    "That girl saved me. I owe her everything.",
    "Felt! Give her back the insignia, she needs it!",
    "She's the one who healed me. Her magic is incredible.",
]


class TestDialogueIsNotEvidenceAboutItsSpeaker(unittest.TestCase):

    def test_male_speaker_talking_about_women_is_not_classified_female(self):
        traits = _infer_character_traits("Subaru", "", SUBARU_LINES)
        self.assertNotEqual(traits["gender"], "female")

    def test_dialogue_alone_yields_unknown(self):
        # "unknown" is the correct answer here: no hard filter, no soft
        # penalty. A confident wrong guess is worse than no guess.
        traits = _infer_character_traits("Subaru", "", SUBARU_LINES)
        self.assertEqual(traits["gender"], "unknown")
        self.assertEqual(traits["gender_confidence"], "unknown")

    def test_the_raw_counter_still_reads_female_on_that_text(self):
        # The underlying function is not wrong about the TEXT - it is being
        # asked the wrong question. Documenting that keeps the fix honest:
        # the bug was the source, not the counter.
        self.assertEqual(_infer_character_gender(" ".join(SUBARU_LINES)),
                         "female")


class TestValidEvidenceStillWorks(unittest.TestCase):

    def test_character_label_is_high_confidence(self):
        traits = _infer_character_traits("LITTLE GIRL", "", SUBARU_LINES)
        self.assertEqual(traits["gender"], "female")
        self.assertEqual(traits["gender_confidence"], "high")

    def test_persona_description_is_medium_confidence(self):
        traits = _infer_character_traits(
            "Subaru", "A teenage boy with black hair; he is stubborn.", [])
        self.assertEqual(traits["gender"], "male")
        self.assertEqual(traits["gender_confidence"], "medium")

    def test_label_outranks_persona(self):
        traits = _infer_character_traits(
            "OLD MAN", "she is elderly and softly spoken", [])
        self.assertEqual(traits["gender"], "male")
        self.assertEqual(traits["gender_confidence"], "high")

    def test_persona_survives_misleading_dialogue(self):
        # The whole point: a real description must not be overridden or
        # diluted by what the character happens to talk about.
        traits = _infer_character_traits(
            "Subaru", "A teenage boy, black hair, stubborn.", SUBARU_LINES)
        self.assertEqual(traits["gender"], "male")


class TestNoEvidence(unittest.TestCase):

    def test_nothing_at_all_is_unknown(self):
        traits = _infer_character_traits("X", "", [])
        self.assertEqual(traits["gender"], "unknown")

    def test_empty_inputs_do_not_raise(self):
        self.assertEqual(_infer_character_traits("", "", [])["gender"], "unknown")


if __name__ == "__main__":
    unittest.main()
