"""A source that repeats itself must not be rejected for being reproduced.

WHAT THIS COST. grimgar03 could not be generated at all. Its source opens with
"Grimgar of Fantasy and Ash: Volume 3" eight times - the file carries it 49
times - and the model transcribed those title lines faithfully on every one of
17 attempts. `find_adjacent_duplicate_blocks` flagged entries 1-8 as an
adjacent duplicate block, and the repair refused to act unless the block
appeared in the source EXACTLY once. At 49 occurrences it returned
"unresolved", the chunk was rejected, and after the retry budget the book
failed at chunk 1 of 49.

The model was right every time. The gate was wrong.

THE OLD TEST CONFLATED OPPOSITE CASES. `source_occurrences != 1` treats "the
model invented this repetition" and "the source genuinely repeats this" as the
same failure, when they call for opposite handling:

    0    invented - nothing can decide which copy is correct. Unresolved.
    1    duplicated - the source has it once, so delete the second copy.
    >=2  faithful - the SOURCE repeats it, so keep both. Deleting a copy
         would corrupt the text the model was asked to reproduce exactly.

These tests pin all three, because the middle case is the one the repair
already handled correctly and must keep handling.
"""
import unittest

from script_repair import build_deterministic_repair


def entries(*texts):
    return [{"text": text, "speaker": "NARRATOR"} for text in texts]


class FaithfulDuplicateBlockTest(unittest.TestCase):

    def test_source_repeating_a_block_is_not_a_defect(self):
        """grimgar03's actual shape: a title the source repeats many times."""
        title = "Grimgar of Fantasy and Ash: Volume 3"
        source = "\n".join([title] * 8) + "\nTable of Contents\n"
        result = build_deterministic_repair(entries(title, title, title, title),
                                    source_text=source)
        self.assertEqual(result["unresolved"], [],
                         "a block the source itself repeats must not be "
                         "reported as an unresolved repair")
        self.assertEqual(len(result["entries"]), 4,
                         "faithful repeats must be kept, not deleted")

    def test_a_block_duplicated_by_the_model_is_still_removed(self):
        """The case the repair already handled - it must keep working."""
        first = "The morning air was sharp and cold."
        second = "Haruhiro rubbed his eyes and sat up."
        source = f"{first}\n{second}\nSomething else entirely happened next.\n"
        result = build_deterministic_repair(entries(first, second, first, second),
                                    source_text=source)
        self.assertEqual(result["unresolved"], [])
        self.assertEqual(len(result["entries"]), 2,
                         "a block appearing once in source but emitted twice "
                         "should have its second copy removed")

    def test_a_block_absent_from_the_source_stays_unresolved(self):
        """Invented repetition: no rule can pick the correct copy."""
        made_up = "This sentence never appears in the source at all."
        other = "Neither does this one, which is also invented."
        source = "The actual source text is about something completely else.\n"
        result = build_deterministic_repair(entries(made_up, other, made_up, other),
                                    source_text=source)
        self.assertTrue(result["unresolved"],
                        "a repetition absent from the source cannot be "
                        "repaired safely and must stay unresolved")
        self.assertEqual(result["unresolved"][0]["reason"],
                         "duplicate_not_in_source")

    def test_the_kept_case_is_recorded_as_a_change(self):
        """Keeping must be visible, not silent.

        A repair that decides to do nothing is still a decision about the
        text, and the manifest is where that decision has to be readable -
        otherwise a future reader cannot tell a block that was checked and
        kept from one that was never examined.
        """
        title = "Volume 3"
        long_title = "Grimgar of Fantasy and Ash: Volume 3"
        source = "\n".join([long_title] * 6)
        result = build_deterministic_repair(entries(long_title, long_title,
                                            long_title, long_title),
                                    source_text=source)
        kinds = [c.get("type") for c in result.get("changes", [])]
        self.assertIn("adjacent_duplicate_block_kept", kinds)
        self.assertFalse(title == long_title)  # guards the fixture itself


if __name__ == "__main__":
    unittest.main()
