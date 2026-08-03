"""Tests for the three harnesses built from the third-party repo review.

The failure mode these guard against is the one Rule 19 names: a harness that
runs, prints a plausible number, and is wrong. None of these would crash. Each
test corresponds to a specific way one of the three could quietly mislead.

  validation normalisation   A gate that fires on correct audio gets switched
                             off within a day. If "25" vs "twenty five" counts
                             as two errors, every segment with a number fails
                             and the whole check is discarded as noise.
  threshold shape            The threshold is the entire policy. If it does not
                             scale with length, either long segments all fail on
                             ASR noise or short gibberish passes.
  truncation asymmetry       Losing the tail is what a listener notices most.
                             A pure error count treats a dropped ending as a few
                             deletions, so it is flagged separately - and must
                             not fire on ordinary short segments.
  BookNLP denominator        Two ways to flatter the baseline: drop the quotes
                             it declined to attribute, or align a repeated line
                             to the wrong occurrence. Either inflates it, and
                             both look like clean data afterwards.
  blend spec parsing         Weights that do not normalise mean the blend that
                             gets generated is not the blend that was asked
                             for, and the audio would be judged under the wrong
                             label.

Written as unittest, not pytest, because update_test_inventory imports every
test module in an environment with neither pytest nor openai.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.tts_output_validation import (compute_threshold, say_number,
                                               validate, word_errors, words)
from experiments.voice_blending import (blend_capacity, parse_blend_spec)
from experiments.booknlp_baseline import (align_to_gold, character_names,
                                          parse_booknlp)


class TestValidationNormalisation(unittest.TestCase):
    """A gate that fires on correct audio is a gate that gets turned off."""

    def test_digits_match_their_spelling(self):
        # ASR writes numbers as words. Without normalisation this is 2 errors
        # on a perfectly good segment.
        errors, _, _ = word_errors("I have 25 apples", "I have twenty five apples")
        self.assertEqual(errors, 0)

    def test_punctuation_and_case_ignored(self):
        errors, _, _ = word_errors('"Stop!" she cried.', "stop she cried")
        self.assertEqual(errors, 0)

    def test_real_substitution_still_counted(self):
        # Normalisation must not be so aggressive that it hides a wrong word.
        errors, _, _ = word_errors("the cat sat", "the dog sat")
        self.assertEqual(errors, 1)

    def test_dropped_words_counted(self):
        errors, n, _ = word_errors("one two three four", "one four")
        self.assertEqual(errors, 2)
        self.assertEqual(n, 4)

    def test_hallucinated_words_counted(self):
        # Qwen3-TTS repeating or inventing speech is the failure this exists for.
        errors, _, _ = word_errors("hello there", "hello there there there")
        self.assertEqual(errors, 2)

    def test_words_drops_empty_tokens(self):
        self.assertEqual(words("--- '' ---"), [])

    def test_number_verbalisation_across_ranges(self):
        # Every one of these is a segment that would otherwise be charged a
        # false error for containing a perfectly ordinary number.
        self.assertEqual(say_number("7"), ["seven"])
        self.assertEqual(say_number("13"), ["thirteen"])
        self.assertEqual(say_number("20"), ["twenty"])
        self.assertEqual(say_number("25"), ["twenty", "five"])
        self.assertEqual(say_number("100"), ["one", "hundred"])
        self.assertEqual(say_number("342"), ["three", "hundred", "forty", "two"])

    def test_long_numbers_read_digitwise(self):
        # A year or an id has no single correct reading; digit-by-digit at
        # least stays stable rather than inventing one.
        self.assertEqual(say_number("1984"),
                         ["one", "nine", "eight", "four"])

    def test_round_hundred_has_no_trailing_zero(self):
        self.assertNotIn("zero", say_number("300"))


class TestThreshold(unittest.TestCase):
    """The threshold is the policy; a flat one makes the gate useless."""

    def test_scales_with_length(self):
        self.assertLess(compute_threshold(10), compute_threshold(200))

    def test_moderate_allows_one_per_ten_words(self):
        self.assertEqual(compute_threshold(10, "moderate"), 1)
        self.assertEqual(compute_threshold(11, "moderate"), 2)

    def test_intolerant_allows_nothing_at_any_length(self):
        self.assertEqual(compute_threshold(1, "intolerant"), 0)
        self.assertEqual(compute_threshold(5000, "intolerant"), 0)

    def test_never_negative(self):
        # high strictness subtracts 1; a short segment must not get a negative
        # budget, which would fail even a perfect transcript.
        self.assertGreaterEqual(compute_threshold(1, "high"), 0)
        self.assertFalse(validate("hello world", "hello world", "high")["failed"])

    def test_unknown_strictness_raises(self):
        with self.assertRaises(ValueError):
            compute_threshold(10, "extremely")


class TestValidateVerdict(unittest.TestCase):

    def test_clean_segment_passes(self):
        r = validate("The quick brown fox jumps over the lazy dog",
                     "the quick brown fox jumps over the lazy dog")
        self.assertFalse(r["failed"])
        self.assertEqual(r["errors"], 0)

    def test_gibberish_fails(self):
        r = validate("The quick brown fox jumps over the lazy dog",
                     "completely different words entirely unrelated here now")
        self.assertTrue(r["failed"])

    def test_truncation_flagged_on_lost_tail(self):
        source = " ".join(["word"] * 20)
        r = validate(source, "word word word")
        self.assertTrue(r["possible_truncation"])

    def test_truncation_not_flagged_on_short_clean_segment(self):
        # The asymmetric check must not fire on every short line, or it is noise.
        r = validate("yes", "yes")
        self.assertFalse(r["possible_truncation"])

    def test_detail_reports_what_was_heard(self):
        # A gate that only emits a number cannot be debugged or trusted.
        r = validate("the cat sat", "the dog sat")
        self.assertTrue(any(d["expected"] == "cat" and d["heard"] == "dog"
                            for d in r["detail"]))


class TestBookNLPParsing(unittest.TestCase):

    ENTITIES = [
        {"COREF": "1", "prop": "PROP", "text": "Elizabeth"},
        {"COREF": "1", "prop": "PRON", "text": "she"},
        {"COREF": "1", "prop": "PRON", "text": "she"},
        {"COREF": "1", "prop": "PRON", "text": "she"},
        {"COREF": "2", "prop": "PRON", "text": "he"},
    ]

    def test_proper_name_beats_more_frequent_pronoun(self):
        # "she" outnumbers "Elizabeth" 3:1, but a pronoun is not a speaker label.
        names = character_names(self.ENTITIES)
        self.assertEqual(names["1"], "Elizabeth")

    def test_character_with_no_proper_mention_still_labelled(self):
        # Dropping it would remove BookNLP's hard cases from the denominator.
        self.assertIn("2", character_names(self.ENTITIES))

    def test_declined_quotes_kept_as_wrong_not_dropped(self):
        quotes = [{"quote": "Hello there", "char_id": "1"},
                  {"quote": "Who said this", "char_id": "-1"}]
        rows = parse_booknlp(quotes, self.ENTITIES)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1][1], "")

    def test_repeated_lines_are_not_aligned(self):
        # "Yes." appears twice; matching either occurrence would be a coin flip
        # dressed up as data.
        gold = {"entries": [
            {"id": "a", "line": "Yes.", "expected_speaker": "ANNA"},
            {"id": "b", "line": "Yes.", "expected_speaker": "BEN"},
            {"id": "c", "line": "Distinct line here.", "expected_speaker": "ANNA"}]}
        rows = parse_booknlp(
            [{"quote": "Yes.", "char_id": "1"},
             {"quote": "Distinct line here.", "char_id": "1"}], self.ENTITIES)
        matched, unmatched = align_to_gold(rows, gold)
        self.assertEqual([m["id"] for m in matched], ["c"])
        self.assertEqual(unmatched, 2)

    def test_special_speakers_excluded_from_scoring(self):
        gold = {"entries": [
            {"id": "a", "line": "Narration here.", "expected_speaker": "NOT_DIALOGUE"},
            {"id": "b", "line": "Real speech.", "expected_speaker": "ANNA"}]}
        rows = parse_booknlp([{"quote": "Narration here.", "char_id": "1"},
                              {"quote": "Real speech.", "char_id": "1"}],
                             self.ENTITIES)
        matched, _ = align_to_gold(rows, gold)
        self.assertEqual([m["id"] for m in matched], ["b"])


class TestBlendSpec(unittest.TestCase):

    def test_weights_normalise_to_one(self):
        blend = parse_blend_spec("alpha:60,beta:40")
        self.assertAlmostEqual(sum(w for _, w in blend), 1.0)
        self.assertAlmostEqual(dict(blend)["alpha"], 0.6)

    def test_unnormalised_weights_still_normalise(self):
        # ':3,:2' must mean the same blend as ':60,:40', or the audio generated
        # is not the blend that was labelled.
        self.assertAlmostEqual(dict(parse_blend_spec("a:3,b:2"))["a"], 0.6)

    def test_missing_weights_split_evenly(self):
        blend = dict(parse_blend_spec("a,b"))
        self.assertAlmostEqual(blend["a"], 0.5)
        self.assertAlmostEqual(blend["b"], 0.5)

    def test_repeated_voice_rejected(self):
        # Blending a voice with itself is the identity, silently mislabelled.
        with self.assertRaises(ValueError):
            parse_blend_spec("a:50,a:50")

    def test_malformed_specs_rejected(self):
        for bad in ("", "   ", ":50", "a:0,b:0", "a:-10,b:110"):
            with self.assertRaises(ValueError):
                parse_blend_spec(bad)

    def test_capacity_counts_even_split_once(self):
        # {A,B} at 50/50 is one voice, not two - counting it twice would
        # overstate the prize.
        self.assertEqual(blend_capacity(2, weight_steps=(50,)), 3)

    def test_capacity_counts_asymmetric_both_ways(self):
        self.assertEqual(blend_capacity(2, weight_steps=(65,)), 4)

    def test_capacity_degenerate_inputs(self):
        self.assertEqual(blend_capacity(0), 0)
        self.assertEqual(blend_capacity(1), 1)
        self.assertEqual(blend_capacity(5, max_components=1), 5)


if __name__ == "__main__":
    unittest.main()
