"""The source gate grades replacement characters and still refuses controls.

WHY IT CHANGED. The gate refused a book at ANY U+FFFD. index18 - one of the
four gold-annotated books every attribution goal measures on - carries 6,662 of
them, 1.40% of the file, from a wrong-codec decode. It was excluded everywhere.

Deterministic repair (repair_source_encoding.py) brings that to 0.26%, and the
remainder is genuinely ambiguous - accented letters, ellipses and dashes that
neither a rule nor an LLM pass resolved reliably. Under the old gate a repaired
book was exactly as unusable as a corrupt one, which made repair pointless.

WHAT DID NOT CHANGE, AND MUST NOT. Unsafe control characters are still an
absolute refusal at any count. They are never legitimate prose and can break
downstream parsing, so they are a different question from a lost accent.

THE THRESHOLD IS EVIDENCE-BASED, NOT ROUND. 0.5% sits between the two measured
figures for the same book, so the raw file is still rejected and the repaired
one runs. These tests pin both sides of that, using index18's real numbers -
a threshold that admitted the corrupt file would defeat the point.
"""
import unittest

import generate_script


TOTAL_CHARS = 476376          # index18
CORRUPT_COUNT = 6662          # 1.398%
REPAIRED_COUNT = 1235         # 0.259%


def share(count):
    return count / TOTAL_CHARS


class SourceGateThresholdTest(unittest.TestCase):

    def test_threshold_sits_between_corrupt_and_repaired(self):
        """The whole design in one assertion, on real measurements."""
        limit = generate_script.MAX_REPLACEMENT_SHARE
        self.assertGreater(share(CORRUPT_COUNT), limit,
                           "the corrupt file must still be refused")
        self.assertLess(share(REPAIRED_COUNT), limit,
                        "the repaired file must be allowed to generate")

    def test_a_clean_file_is_far_below_the_limit(self):
        self.assertLess(share(0), generate_script.MAX_REPLACEMENT_SHARE)

    def test_the_limit_is_small_enough_to_stay_meaningful(self):
        """A limit set high enough to admit anything is not a gate.

        1% would accept index18 uncorrected on some chapters; 5% would accept
        essentially any mis-decoded file. This pins the order of magnitude so
        a later 'just raise it a bit' has to argue with a number.
        """
        self.assertLessEqual(generate_script.MAX_REPLACEMENT_SHARE, 0.01)
        self.assertGreater(generate_script.MAX_REPLACEMENT_SHARE, 0.0)

    def test_unsafe_controls_remain_an_absolute_refusal(self):
        """Read from the source: the control check must not gain a threshold.

        The two checks are adjacent in the file and it would be easy for a
        later edit to grade both. Only the replacement-character one is
        graded, and this fails if the control branch ever compares a share.
        """
        import inspect
        source = inspect.getsource(generate_script.main)
        marker = 'if source_unicode["unsafe_controls"]:'
        self.assertIn(marker, source,
                      "unsafe controls must stay an unconditional refusal")
        after = source.split(marker, 1)[1][:400]
        self.assertIn("sys.exit(1)", after)
        self.assertNotIn("MAX_REPLACEMENT_SHARE", after.split("sys.exit(1)")[0],
                         "the control refusal must not be made conditional on "
                         "a share threshold")


if __name__ == "__main__":
    unittest.main()
