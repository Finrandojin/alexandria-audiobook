"""The pronunciation lexicon must change audio, never the record of it.

46% of the active book's lines carry a Latinized Japanese name, so a lexicon
bug reaches nearly half the audiobook. The dangerous failures are not crashes:
they are a respelling that fires where it should not, or one that fires and
leaves no trace, so the listener hears one thing and the script says another.
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pronunciation
from pronunciation import apply_pronunciation, load_lexicon


class LexiconTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        pronunciation._cache.update({"path": None, "mtime": None,
                                     "entries": {}, "pattern": None})

    def tearDown(self):
        self.tmp.cleanup()
        pronunciation._cache.update({"path": None, "mtime": None,
                                     "entries": {}, "pattern": None})

    def write(self, names):
        p = os.path.join(self.tmp.name, "pron.json")
        with open(p, "w", encoding="utf-8") as fh:
            json.dump({"names": names}, fh)
        return p

    # ── the substitution itself ──────────────────────────────────────────

    def test_a_name_is_respelled(self):
        p = self.write({"Subaru": "Soo-bah-roo"})
        out, applied = apply_pronunciation("or so Subaru thought", p)
        self.assertEqual(out, "or so Soo-bah-roo thought")
        self.assertEqual(applied, [{"name": "Subaru", "spoken": "Soo-bah-roo"}])

    def test_every_substitution_is_reported(self):
        """A silent respelling is untraceable: the audio and the script would
        disagree with nothing recording why."""
        p = self.write({"Rom": "Rohm", "Felt": "Felt"})
        _, applied = apply_pronunciation("Rom and Felt and Rom", p)
        self.assertEqual([a["name"] for a in applied], ["Rom", "Felt", "Rom"])

    def test_longest_key_wins(self):
        """With both 'Natsuki' and 'Natsuki Subaru' present, matching the
        shorter key inside the longer name leaves a half-substituted phrase."""
        p = self.write({"Natsuki": "Nat-ski",
                        "Natsuki Subaru": "Nat-ski Soo-bah-roo"})
        out, _ = apply_pronunciation("Natsuki Subaru arrived", p)
        self.assertEqual(out, "Nat-ski Soo-bah-roo arrived")

    # ── where it must NOT fire ───────────────────────────────────────────

    def test_it_does_not_fire_inside_a_longer_word(self):
        """'Rom' must not rewrite 'Romance' or 'from'."""
        p = self.write({"Rom": "Rohm"})
        out, applied = apply_pronunciation("a Romance from Rome", p)
        self.assertEqual(out, "a Romance from Rome")
        self.assertEqual(applied, [])

    def test_it_does_not_fire_across_an_apostrophe(self):
        """`Subaru's` is the same name and would be broken by a naive swap
        that ignored the trailing possessive."""
        p = self.write({"Subaru": "Soo-bah-roo"})
        out, applied = apply_pronunciation("Subaru's hand", p)
        self.assertEqual(out, "Subaru's hand")
        self.assertEqual(applied, [])

    def test_an_empty_respelling_is_ignored(self):
        """The shipped file lists every name with an empty value, so the
        mechanism is present and inert until an entry is MEASURED to help.
        An empty value must not delete the name from the text."""
        p = self.write({"Subaru": "", "Felt": "   "})
        out, applied = apply_pronunciation("Subaru met Felt", p)
        self.assertEqual(out, "Subaru met Felt")
        self.assertEqual(applied, [])

    # ── it must never stop a book generating ─────────────────────────────

    def test_a_missing_lexicon_is_not_an_error(self):
        out, applied = apply_pronunciation(
            "text", os.path.join(self.tmp.name, "absent.json"))
        self.assertEqual(out, "text")
        self.assertEqual(applied, [])

    def test_a_malformed_lexicon_degrades_to_no_substitution(self):
        p = os.path.join(self.tmp.name, "bad.json")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("{ this is not json")
        out, applied = apply_pronunciation("Subaru", p)
        self.assertEqual(out, "Subaru")
        self.assertEqual(applied, [])

    def test_non_string_entries_are_skipped(self):
        p = os.path.join(self.tmp.name, "odd.json")
        with open(p, "w", encoding="utf-8") as fh:
            json.dump({"names": {"Subaru": 5, "Felt": None, "Rom": "Rohm"}}, fh)
        out, _ = apply_pronunciation("Subaru Felt Rom", p)
        self.assertEqual(out, "Subaru Felt Rohm")

    def test_an_edit_takes_effect(self):
        """Cached on mtime, so correcting a name does not need a restart."""
        p = self.write({"Subaru": "First"})
        self.assertEqual(apply_pronunciation("Subaru", p)[0], "First")
        import time
        time.sleep(0.01)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump({"names": {"Subaru": "Second"}}, fh)
        os.utime(p, (os.path.getatime(p), os.path.getmtime(p) + 10))
        self.assertEqual(apply_pronunciation("Subaru", p)[0], "Second")


class ShippedLexiconTest(unittest.TestCase):
    """The file that ships with the repo."""

    def test_it_ships_inert(self):
        """Entries must be measured, not guessed. If this ever fails it means
        somebody added a respelling - which is fine, but it should be because
        proper_noun_pronunciation.py showed it helped."""
        pronunciation._cache.update({"path": None, "mtime": None,
                                     "entries": {}, "pattern": None})
        entries = load_lexicon()
        for name, spoken in entries.items():
            self.assertTrue(spoken.strip(),
                            f"{name} has a respelling; was it measured?")

    def test_the_shipped_file_parses(self):
        path = pronunciation.DEFAULT_PATH
        if not os.path.exists(path):
            self.skipTest("no pronunciation.json in this checkout")
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        self.assertIn("names", doc)
        self.assertIsInstance(doc["names"], dict)


class RosterDerivationTest(unittest.TestCase):
    """The name list comes from the roster, and aliases inform it WITHOUT
    leaking one character's pronunciation onto another's spelling."""

    def test_a_nickname_does_not_inherit_the_canonical_pronunciation(self):
        """THE TRAP. character_aliases maps 'BETTY' -> 'BEATRICE' because they
        are one character. They are NOT one sound. Substituting across the
        alias would put a word in the audio that is not in the book, which is
        worse than mispronouncing it."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        p = os.path.join(tmp.name, "pron.json")
        with open(p, "w", encoding="utf-8") as fh:
            json.dump({"names": {"Beatrice": "Bee-ah-tree-chay"}}, fh)
        out, applied = apply_pronunciation("Betty smiled at Beatrice", p)
        self.assertEqual(out, "Betty smiled at Bee-ah-tree-chay")
        self.assertEqual([a["name"] for a in applied], ["Beatrice"])

    def _fixture(self, chunks, aliases=None, voice_config=None):
        """Drive character_forms from files this test owns.

        These two tests used to call character_forms() bare, against whatever
        book happened to be checked out. That made them skip in CI - where no
        script is committed - and the release verifier counts a skip as a
        failure. It also meant the Felt assertion only ran when the loaded
        book contained Felt, so the collision logic went unexercised exactly
        when someone changed books.
        """
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        paths = {}
        for key, data in (("script_path", chunks),
                          ("aliases_path", aliases or {}),
                          ("voice_config_path", voice_config or {})):
            p = os.path.join(tmp.name, key + ".json")
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
            paths[key] = p
        from pronunciation import character_forms
        return character_forms(**paths)

    def test_forms_are_derived_from_the_roster_and_the_text(self):
        """A roster name that never appears in the text gets no entry: the
        lexicon describes what will be spoken, not who was cast."""
        forms = self._fixture(
            [{"speaker": "BEATRICE", "text": "Beatrice waited by the door."},
             {"speaker": "SUBARU", "text": "Subaru said nothing."}],
            voice_config={"characters": {"BEATRICE": {}, "SUBARU": {},
                                         "NEVER_SPOKEN": {}}})
        self.assertIn("Beatrice", forms)
        self.assertNotIn("NEVER_SPOKEN", forms)
        self.assertNotIn("Never_Spoken", forms)
        for name, info in forms.items():
            self.assertGreater(info["occurrences"], 0,
                               f"{name} listed but never occurs")

    def test_a_name_colliding_with_a_common_word_is_flagged(self):
        """`Felt` is a character and `felt` is a verb. Case-sensitivity
        protects the verb; the flag warns whoever fills the file in."""
        forms = self._fixture(
            [{"speaker": "FELT", "text": "Felt felt the cold, and Felt ran."}],
            voice_config={"characters": {"FELT": {}}})
        self.assertIn("Felt", forms)
        self.assertTrue(forms["Felt"]["collides_with_common_word"])
        self.assertEqual(forms["Felt"]["occurrences"], 2)
        self.assertEqual(forms["Felt"]["lowercase_occurrences"], 1)

    def test_a_name_with_no_lowercase_twin_is_not_flagged(self):
        """The counterpart: flagging every name would make the flag useless."""
        forms = self._fixture(
            [{"speaker": "SUBARU", "text": "Subaru felt the cold."}],
            voice_config={"characters": {"SUBARU": {}}})
        self.assertFalse(forms["Subaru"]["collides_with_common_word"])

    def test_a_respelled_name_does_not_touch_its_lowercase_word(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        p = os.path.join(tmp.name, "pron.json")
        with open(p, "w", encoding="utf-8") as fh:
            json.dump({"names": {"Felt": "Fehlt"}}, fh)
        out, _ = apply_pronunciation("Felt felt the cold", p)
        self.assertEqual(out, "Fehlt felt the cold")


class NormalizationIntegrationTest(unittest.TestCase):
    """The lexicon has to reach the text the model actually receives."""

    def test_normalization_reports_the_substitution(self):
        from speech_text import get_speech_normalization
        import pronunciation as pr
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        p = os.path.join(tmp.name, "pron.json")
        with open(p, "w", encoding="utf-8") as fh:
            json.dump({"names": {"Subaru": "Soo-bah-roo"}}, fh)
        original, pr.DEFAULT_PATH = pr.DEFAULT_PATH, p
        pr._cache.update({"path": None, "mtime": None, "entries": {},
                          "pattern": None})
        try:
            out = get_speech_normalization("Subaru walked on.")
            self.assertIn("Soo-bah-roo", out["text"])
            kinds = [t["type"] for t in out["transformations"]]
            self.assertIn("pronunciation_lexicon", kinds)
        finally:
            pr.DEFAULT_PATH = original
            pr._cache.update({"path": None, "mtime": None, "entries": {},
                              "pattern": None})


if __name__ == "__main__":
    unittest.main()
