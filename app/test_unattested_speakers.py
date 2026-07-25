import unittest

from pass_quality import validate_attribution


class RejectUnattestedSpeakerTest(unittest.TestCase):
    """The roster gate filters what goes IN; nothing filtered what came OUT.
    The model invented FUTURE_ME - the protagonist's future self, which the
    book never names - and shipped it on 250 entries, 13% of one book. WEARING,
    MAILMAN, ARUMANFI and SWORD_GOD_GARU_FARION came through the same hole:
    279 entries, 14.6% of the book, attributed to names not in the text."""

    # Long enough to clear MIN_SOURCE_FOR_ATTESTATION: the gate deliberately
    # stands down on fragments, where a real name may appear only once.
    SOURCE = (("Roxy smiled at Rudi. Roxy and Eris spoke, and Eris laughed. "
               "Rudi looked at Roxy again. ") * 90
              + "I remembered what my future me had written. " * 4)

    def _report(self, speaker):
        frozen = [{"type": "SPOKEN", "text": "Hello there."}]
        return validate_attribution(
            frozen, [{"n": 0, "head": "Hello there.", "speaker": speaker}],
            source_text=self.SOURCE)

    def test_invented_speaker_is_rejected(self):
        report = self._report("FUTURE_ME")
        self.assertFalse(report["passed"])
        self.assertTrue(any(f.get("code") == "speaker_not_in_source"
                            for f in report["findings"]))

    def test_real_character_passes(self):
        self.assertTrue(self._report("ROXY")["passed"], self._report("ROXY"))

    def test_common_word_is_rejected(self):
        # "wearing"-class inventions: a word from the prose, not a name.
        self.assertFalse(self._report("REMEMBERED")["passed"])

    def test_unknown_placeholder_still_allowed(self):
        self.assertTrue(self._report("UNKNOWN")["passed"])

    def test_without_source_nothing_is_rejected(self):
        # Callers that pass no source keep today's behaviour exactly.
        frozen = [{"type": "SPOKEN", "text": "Hello there."}]
        report = validate_attribution(
            frozen, [{"n": 0, "head": "Hello there.", "speaker": "FUTURE_ME"}])
        self.assertTrue(report["passed"])


if __name__ == "__main__":
    unittest.main()
