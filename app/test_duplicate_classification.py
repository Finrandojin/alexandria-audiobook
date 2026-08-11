"""Deciding whether the model duplicated text, or invented it.

The repair can only delete a duplicate copy when it is sure the source really
carries that content. Get the question wrong in the safe direction and a
finished book is thrown away; get it wrong in the unsafe direction and real
prose is deleted. Both directions are pinned here.

WHAT WENT WRONG. Alice in Wonderland generated all 25 chunks, hit the final
whole-book gate, and was refused - all 919 entries discarded. Two of its
entries were emitted twice, the repair knew how to drop the second copy, and
the classifier called the block "not in the source" for two reasons that were
both artefacts of matching rather than defects in the book:

  1. Entries are not contiguous spans of the source. "Explain yourself!" and
     Alice's reply have "said Alice" between them in the book, so the joined
     block occurs zero times while each line occurs once.

  2. `\\w` includes UNDERSCORE, and Project Gutenberg marks italics with it.
     The source says "explain _myself_"; the model correctly drops the
     markup and writes "myself"; the two never matched. 23 of the 28
     public-domain novels use the convention.
"""
import unittest

import script_preflight
from script_repair import build_deterministic_repair


class UnderscoreMarkupTest(unittest.TestCase):
    """Gutenberg italics must not make a word unmatchable."""

    def test_emphasis_markers_do_not_change_a_word(self):
        source = script_preflight._normalize_words("I can't explain _myself_")
        model = script_preflight._normalize_words("I can't explain myself")
        self.assertEqual(model, source,
                         "the model correctly drops _italic_ markup; the "
                         "matcher must not treat that as different text")

    def test_a_real_underscore_word_still_tokenizes(self):
        self.assertEqual("snake case",
                         script_preflight._normalize_words("snake_case"))


class SourceOccurrenceTest(unittest.TestCase):

    N = staticmethod(script_preflight._normalize_words)

    def test_text_present_once_is_counted_once(self):
        source = self.N("the caterpillar asked her to explain herself clearly")
        self.assertEqual(1, script_preflight.source_occurrences_for_text(
            source, self.N("asked her to explain herself clearly")))

    def test_orthographic_drift_still_counts(self):
        """Carroll writes ca'n't; the model writes can't."""
        source = self.N("I ca'n't explain myself, I'm afraid, sir, she said")
        found = script_preflight.source_occurrences_for_text(
            source, self.N("I can't explain myself, I'm afraid, sir"))
        self.assertGreaterEqual(found, 1,
                                "one apostrophe must not make a faithful line "
                                "look invented")

    def test_text_absent_from_the_source_is_zero(self):
        """The genuine invention case must survive - this is the safety net."""
        source = self.N("nothing here resembles the sentence being sought")
        self.assertEqual(0, script_preflight.source_occurrences_for_text(
            source, self.N("the model made this whole sentence up entirely")))


class DuplicateRepairTest(unittest.TestCase):

    SOURCE = ('"Explain yourself!" said the Caterpillar sternly.\n\n'
              '"I can\'t explain _myself_, I\'m afraid, sir," said Alice.\n')

    def _entries(self):
        pair = [{"speaker": "CATERPILLAR",
                 "text": '"Explain yourself!" said the Caterpillar sternly.'},
                {"speaker": "ALICE",
                 "text": "I can't explain myself, I'm afraid, sir"}]
        return [dict(e) for e in pair + pair]

    def test_a_duplicated_pair_is_removed_not_refused(self):
        result = build_deterministic_repair(self._entries(), self.SOURCE)
        self.assertEqual([], result["unresolved"],
                         "a block whose lines are both in the source is a "
                         "duplication the repair can fix, not an invention")
        self.assertEqual(2, len(result["entries"]),
                         "the second copy should be gone")

    def test_invented_repetition_is_still_refused(self):
        """The direction that must NOT be relaxed.

        If the content is nowhere in the source, neither copy is authoritative
        and deleting one is a guess. This stays unresolved and fails the gate.
        """
        pair = [{"speaker": "NARRATOR",
                 "text": "A sentence the book never contained at all here."},
                {"speaker": "NARRATOR",
                 "text": "Another fabricated sentence with no source at all."}]
        entries = [dict(e) for e in pair + pair]
        result = build_deterministic_repair(entries, self.SOURCE)
        self.assertTrue(result["unresolved"],
                        "invented repetition must remain unresolved")
        self.assertEqual(4, len(result["entries"]),
                         "nothing may be deleted when the source cannot "
                         "adjudicate which copy is real")

    def test_a_block_the_source_repeats_is_kept(self):
        """Faithful transcription of a real repeat - grimgar03's title."""
        title = "Grimgar of Fantasy and Ash"
        entries = [{"speaker": "NARRATOR", "text": title} for _ in range(4)]
        source = "\n".join([title] * 8)
        result = build_deterministic_repair(entries, source)
        self.assertEqual([], result["unresolved"])
        self.assertEqual(4, len(result["entries"]),
                         "the source repeats this block; deleting a copy "
                         "would corrupt faithful text")


if __name__ == "__main__":
    unittest.main()
