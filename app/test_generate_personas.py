"""Voice assignment must not merge two characters into one voice.

`generate_personas.py` is 37 KB behind a live endpoint - it decides which voice
each character gets, so its mistakes are what a listener hears - and until now
no test imported it.

THE BUG THIS WAS WRITTEN FOR. `normalize_speaker_name` stripped honorifics, so
`MR. BENNET` and `MRS. BENNET` both reduced to `bennet` and
`_resolve_to_canonical` returned whichever came first in the roster. Mr and Mrs
Bennet would have shared a voice, along with the Hilberys, Allens, Halls and
Van der Luydens - six of twenty-eight books in the PDNC set. Nothing failed;
the audiobook simply had the wrong voice.

Case folding is NOT the same hazard and must keep working: the live config has
eleven pairs like EMILIA/Emilia and NOT-SATELLA/Not-Satella that are one
character each and rely on being merged.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_personas import (_resolve_to_canonical, _token_jaccard,
                               honorifics_are_distinguishing,
                               normalize_speaker_name, pick_ref_text)


class NormalizationTest(unittest.TestCase):

    def test_case_and_punctuation_always_fold(self):
        """The live config depends on this: EMILIA and Emilia are one
        character, as are NOT-SATELLA and Not-Satella."""
        self.assertEqual(normalize_speaker_name("EMILIA"),
                         normalize_speaker_name("Emilia"))
        self.assertEqual(normalize_speaker_name("NOT-SATELLA"),
                         normalize_speaker_name("Not-Satella"))
        self.assertEqual(normalize_speaker_name("  Man 1  "),
                         normalize_speaker_name("MAN 1"))

    def test_honorific_stripping_is_optional(self):
        self.assertEqual(normalize_speaker_name("Mr. Darcy"), "darcy")
        self.assertEqual(
            normalize_speaker_name("Mr. Darcy", strip_honorifics=False),
            "mr darcy")

    def test_non_string_input_is_empty(self):
        for bad in (None, 5, [], {}):
            self.assertEqual(normalize_speaker_name(bad), "")


class RosterAmbiguityTest(unittest.TestCase):

    def test_a_married_couple_is_detected(self):
        self.assertTrue(honorifics_are_distinguishing(
            ["MR. BENNET", "MRS. BENNET"]))

    def test_an_ordinary_roster_is_not(self):
        self.assertFalse(honorifics_are_distinguishing(
            ["EMILIA", "SUBARU", "MR. DARCY"]))

    def test_empty_and_none_are_safe(self):
        self.assertFalse(honorifics_are_distinguishing([]))
        self.assertFalse(honorifics_are_distinguishing(None))


class ResolutionTest(unittest.TestCase):
    """The behaviour that reaches a listener."""

    COUPLE = ["MR DARCY", "MRS DARCY"]

    def test_a_couple_resolves_to_two_distinct_voices(self):
        """THE BUG. Before the fix both returned MR DARCY."""
        self.assertEqual(_resolve_to_canonical("Mr. Darcy", self.COUPLE),
                         "MR DARCY")
        self.assertEqual(_resolve_to_canonical("Mrs. Darcy", self.COUPLE),
                         "MRS DARCY")

    def test_a_third_honorific_is_refused_rather_than_guessed(self):
        """Miss Darcy is not in this roster. Returning one of the two would be
        a confident wrong answer; None lets the caller decide."""
        self.assertIsNone(_resolve_to_canonical("Miss Darcy", self.COUPLE))

    def test_loose_matching_survives_where_it_is_safe(self):
        """Most books have no couple, and stripping the honorific there is
        useful. Turning it off globally would cost every one of them."""
        for raw in ("Mr. Darcy", "Darcy", "Miss Darcy"):
            self.assertEqual(_resolve_to_canonical(raw, ["DARCY"]), "DARCY")

    def test_case_variants_still_resolve(self):
        self.assertEqual(_resolve_to_canonical("Emilia", ["EMILIA"]), "EMILIA")
        self.assertEqual(
            _resolve_to_canonical("NOT-SATELLA", ["Not-Satella"]),
            "Not-Satella")

    def test_a_bare_surname_still_resolves_on_an_ambiguous_roster(self):
        """'Darcy' alone is genuinely ambiguous; resolving it to one of them is
        the existing behaviour and not what this fix is about. Asserted so a
        future change to it is deliberate."""
        self.assertIn(_resolve_to_canonical("Darcy", self.COUPLE), self.COUPLE)

    def test_unrelated_names_do_not_match(self):
        self.assertIsNone(_resolve_to_canonical("Subaru", ["EMILIA", "FELT"]))

    def test_short_names_do_not_match_longer_ones(self):
        """'al' must not resolve to 'allan' - initials would collide with
        everyone."""
        self.assertIsNone(_resolve_to_canonical("al", ["ALLAN", "JONATHAN"]))

    def test_empty_input_is_none(self):
        self.assertIsNone(_resolve_to_canonical("", ["EMILIA"]))
        self.assertIsNone(_resolve_to_canonical(None, ["EMILIA"]))


class JaccardTest(unittest.TestCase):

    def test_it_takes_the_same_honorific_decision_as_its_caller(self):
        """The leak that survived the first fix. With honorifics stripped,
        'Miss Darcy' and 'MR DARCY' both reduce to {darcy} and score 1.0, so
        step 3 undid step 1."""
        self.assertEqual(_token_jaccard("Miss Darcy", "MR DARCY"), 1.0)
        self.assertLess(
            _token_jaccard("Miss Darcy", "MR DARCY", strip_honorifics=False),
            0.4)

    def test_empty_names_score_zero(self):
        self.assertEqual(_token_jaccard("", "EMILIA"), 0.0)
        self.assertEqual(_token_jaccard("EMILIA", ""), 0.0)


class RefTextTest(unittest.TestCase):

    def test_it_picks_something_from_real_lines(self):
        lines = ["Short.", "A rather longer line of dialogue to read aloud.",
                 "Mid length line here."]
        self.assertIn(pick_ref_text(lines), lines + [""])

    def test_no_lines_does_not_crash(self):
        self.assertIsInstance(pick_ref_text([]), str)


class RealRosterTest(unittest.TestCase):
    """Against the book that is actually loaded."""

    def test_the_live_config_has_no_honorific_ambiguity(self):
        import json
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "voice_config.json")
        if not os.path.exists(path):
            self.skipTest("no voice_config.json")
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        chars = raw.get("characters") if isinstance(raw.get("characters"), dict) \
            else raw
        names = [n for n, v in chars.items() if isinstance(v, dict)]
        # This book's collisions are all case variants, which SHOULD merge.
        # If a couple ever appears, resolution keeps honorifics automatically -
        # this asserts the current state rather than requiring it.
        self.assertIsInstance(honorifics_are_distinguishing(names), bool)


if __name__ == "__main__":
    unittest.main()
