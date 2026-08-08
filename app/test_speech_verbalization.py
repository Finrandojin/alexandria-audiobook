"""Nothing without a spoken form may reach the TTS engine. Goal 5.1.

WHAT WAS ACTUALLY WRONG. `normalize_for_speech` is the single call every voice
path in `tts.py` makes before synthesis, and it handled `■` only because that
character happens to sit in `SPEECH_BREAKS`. Measured 2026-08-08, these all
passed through it unchanged and reached the engine:

    ♪  U+266A    ∞  U+221E    ★  U+2605    →  U+2192    ♥  U+2665
    �  U+FFFD - not a character at all, the residue of a decoding failure

The exposure was latent rather than active: an audit of all 48 saved books
found 56 unspeakable characters in the raw text and 0 surviving normalization,
because the source gate refuses the one book carrying 6,662 U+FFFD before it
is ever saved. So the protection was the gate, not the normaliser, and
anything reaching `scripts/` by another route was unprotected.

WHY THESE TESTS AND NOT A ROUND-TRIP CHECK. Asserting that output differs from
input would pass on any mutation at all, including one that deleted the
sentence. Each test below names the behaviour it wants: this symbol becomes
these words, that one disappears, and the characters around both survive
intact.
"""
import unicodedata
import unittest

from speech_text import (REPLACEMENT_CHARACTER, VERBALIZED_SYMBOLS,
                         get_speech_normalization, normalize_for_speech)


class VerbalizationTest(unittest.TestCase):

    def test_named_symbols_are_spoken_as_words(self):
        spoken = normalize_for_speech("The cost is 5 × 3 and time is ∞ here")
        self.assertIn("times", spoken)
        self.assertIn("infinity", spoken)
        self.assertNotIn("×", spoken)
        self.assertNotIn("∞", spoken)

    def test_unknown_symbols_are_dropped_not_read(self):
        for symbol in ("♪", "★", "♥", "☂", "⚑"):
            with self.subTest(symbol=symbol):
                spoken = normalize_for_speech(f"She hummed {symbol} softly")
                self.assertNotIn(symbol, spoken)

    def test_replacement_character_never_survives(self):
        """U+FFFD is a decoding failure, and index18 carries 6,662 of them."""
        text = f"A{REPLACEMENT_CHARACTER}B said {REPLACEMENT_CHARACTER} hello"
        spoken = normalize_for_speech(text)
        self.assertNotIn(REPLACEMENT_CHARACTER, spoken)

    def test_surrounding_words_survive_intact(self):
        """The words either side of a dropped symbol must not be damaged."""
        spoken = normalize_for_speech("Haruhiro ♪ shouted at Ranta")
        self.assertIn("Haruhiro", spoken)
        self.assertIn("Ranta", spoken)
        self.assertIn("shouted", spoken)

    def test_cjk_and_accented_letters_are_left_alone(self):
        """Speakable text is not collateral: these are letters, not symbols."""
        for text in ("彼女は静かに言った", "Zoë café naïve", "こんにちは"):
            with self.subTest(text=text):
                spoken = normalize_for_speech(text)
                for ch in text:
                    self.assertIn(ch, spoken)

    def test_box_drawing_still_becomes_a_sentence_break(self):
        """Pre-existing SPEECH_BREAKS behaviour must not regress to a drop."""
        spoken = normalize_for_speech("First part ■ second part")
        self.assertNotIn("■", spoken)
        self.assertIn("First part", spoken)
        self.assertIn("second part", spoken)

    def test_currency_is_not_treated_as_unspeakable(self):
        """Sc is excluded on purpose - these have spoken forms."""
        for symbol in ("$", "£", "€", "¥"):
            with self.subTest(symbol=symbol):
                self.assertIn(symbol, normalize_for_speech(f"It cost {symbol}5"))

    def test_drops_are_recorded_rather_than_silent(self):
        result = get_speech_normalization("A ♪ B")
        kinds = [t["type"] for t in result["transformations"]]
        self.assertIn("dropped_unspeakable", kinds)
        dropped = next(t for t in result["transformations"]
                       if t["type"] == "dropped_unspeakable")
        self.assertEqual(dropped["symbols"], ["♪"])
        self.assertEqual(dropped["count"], 1)

    def test_no_unspeakable_category_survives_a_mixed_line(self):
        """The catch-all, exercised over one character per covered category.

        The named tables can only cover symbols someone thought of; this
        asserts the general rule holds for characters nobody listed.
        """
        exotic = "".join(("⌘", "⍟", "☭", "⛁", ""))
        spoken = normalize_for_speech(f"He said {exotic} and left")
        for ch in exotic:
            with self.subTest(codepoint=f"U+{ord(ch):04X}"):
                self.assertNotIn(ch, spoken)
        self.assertIn("left", spoken)

    def test_every_verbalized_symbol_actually_maps_to_words(self):
        """A table entry that produced nothing would be a silent drop."""
        for symbol, word in VERBALIZED_SYMBOLS.items():
            with self.subTest(symbol=symbol):
                self.assertTrue(word.strip(),
                                f"{symbol} maps to an empty spoken form")
                self.assertIn(word, normalize_for_speech(f"x {symbol} y"))

    def test_verbalized_symbols_are_not_also_swept_by_the_catch_all(self):
        """Order dependency: naming a symbol must beat the category rule."""
        for symbol in VERBALIZED_SYMBOLS:
            with self.subTest(symbol=symbol):
                category = unicodedata.category(symbol)
                if category not in {"So", "Sm", "Sk"}:
                    continue
                self.assertIn(VERBALIZED_SYMBOLS[symbol],
                              normalize_for_speech(f"a {symbol} b"))


if __name__ == "__main__":
    unittest.main()
