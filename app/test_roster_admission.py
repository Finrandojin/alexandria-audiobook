"""Every path that adds a name to the roster must use the same gate.

The roster is fed back to every later batch as the list of established
characters, so one bad name there propagates: measured on mushoku16, WEARING
was invented at entry 11 and reached entry 1,106. build_roster guards
admission with MIN_ROSTER_ATTESTATIONS, deliberately stricter than the
per-entry output gate, because a roster mistake is repeated and an entry
mistake is not.

The success path appended accepted speakers straight to the roster with no
check at all, so a name attested twice - enough to clear the entry gate at
MIN_NAME_ATTESTATIONS - was advertised to every subsequent batch as
established. Found by audit, after an earlier change merged two copies of the
attestation check and missed this third caller.
"""
import unittest

from three_pass_generate import (MIN_ROSTER_ATTESTATIONS, attested_new_speakers,
                                 build_roster)

# Long enough to clear MIN_SOURCE_FOR_ATTESTATION, so the gate actually judges.
FILLER = "The road went on and the day was long. " * 200
# ROXY is named far more than the roster threshold; TWICE exactly twice, which
# clears the entry gate (2) but not the roster gate (3).
SOURCE = (("Roxy smiled at the door. " * 40) + FILLER
          + "Twice arrived. Later, Twice left again.")


class AttestedNewSpeakersTest(unittest.TestCase):
    def test_a_well_attested_speaker_is_admitted(self):
        self.assertEqual(
            ["ROXY"],
            attested_new_speakers([{"speaker": "ROXY"}], set(), SOURCE))

    def test_a_speaker_below_the_roster_threshold_is_refused(self):
        # This is the audit's regression case: exactly two attestations.
        self.assertEqual(
            [], attested_new_speakers([{"speaker": "TWICE"}], set(), SOURCE))

    def test_the_roster_threshold_is_stricter_than_the_entry_gate(self):
        # Both gates must agree on well-attested names; they differ only in how
        # much evidence they demand, and that difference is the point.
        from pass_quality import is_attested_name
        self.assertTrue(is_attested_name("TWICE", SOURCE))
        self.assertFalse(
            is_attested_name("TWICE", SOURCE, MIN_ROSTER_ATTESTATIONS))

    def test_narrator_and_unknown_are_never_admitted(self):
        entries = [{"speaker": "NARRATOR"}, {"speaker": "UNKNOWN"}]
        self.assertEqual([], attested_new_speakers(entries, set(), SOURCE))

    def test_names_already_seen_are_not_repeated(self):
        self.assertEqual(
            [], attested_new_speakers([{"speaker": "ROXY"}], {"ROXY"}, SOURCE))

    def test_a_name_repeated_within_one_batch_is_returned_once(self):
        entries = [{"speaker": "ROXY"}, {"speaker": "ROXY"}]
        self.assertEqual(["ROXY"], attested_new_speakers(entries, set(), SOURCE))

    def test_without_source_text_every_new_speaker_is_admitted(self):
        # Callers with no source keep today's permissive behaviour.
        entries = [{"speaker": "TWICE"}]
        self.assertEqual(["TWICE"], attested_new_speakers(entries, set(), None))

    def test_it_agrees_with_build_roster_on_the_same_input(self):
        # The two admission paths must not drift again.
        entries = [{"speaker": "ROXY"}, {"speaker": "TWICE"}]
        self.assertEqual(build_roster(entries, SOURCE),
                         attested_new_speakers(entries, set(), SOURCE))

    def test_the_caller_is_not_mutated(self):
        seen = {"ROXY"}
        attested_new_speakers([{"speaker": "ROXY"}], seen, SOURCE)
        self.assertEqual({"ROXY"}, seen)


if __name__ == "__main__":
    unittest.main()
