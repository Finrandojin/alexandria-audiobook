"""The gold set is the only reliable instrument here; its builder must not
introduce the defects the measurements already suffered from.

Two independent readers agreeing at 94-97% is what made the existing set
trustworthy, so the tool has to support that protocol rather than just emit
rows for one judge.
"""
import json
import os
import tempfile
import unittest

from gold_set_builder import (agreement, build, context, eligible_indexes,
                              merge, read_filled, validate)


def _segmented():
    return [
        {"type": "NARRATOR", "text": "The room was cold."},
        {"type": "SPOKEN", "text": "Tell me the truth."},
        {"type": "NARRATOR", "text": "Roxy shook her head."},
        {"type": "SPOKEN", "text": "Sorry."},
        {"type": "NARRATOR", "text": "Eris looked away."},
        {"type": "SPOKEN", "text": "Sorry."},
        {"type": "SPOKEN", "text": "I will not say it again."},
        {"type": "SPOKEN", "text": "ok"},
    ]


class EligibilityTest(unittest.TestCase):
    def test_repeated_lines_are_excluded(self):
        # "Sorry." occurs twice, so a judgement on it cannot be attached to one
        # position. This is the defect that reached two separate harnesses.
        self.assertEqual([1, 6], eligible_indexes(_segmented()))

    def test_narration_is_not_offered_for_judgement(self):
        for index in eligible_indexes(_segmented()):
            self.assertEqual("SPOKEN", _segmented()[index]["type"])

    def test_very_short_lines_are_skipped(self):
        self.assertNotIn(7, eligible_indexes(_segmented()))


class AdaptiveWindowTest(unittest.TestCase):
    """Owner policy: expand the window until identity is supported.

    A fixed window forced 13 answerable lines to be marked AMBIGUOUS because no
    name appeared nearby. Excluding them or accepting the abstention would both
    change the task being measured, so the window grows instead.
    """

    def _book(self):
        # Roxy is named once, far above the line being judged.
        rows = [{"type": "NARRATOR", "text": "Roxy stepped into the room."}]
        rows += [{"type": "NARRATOR", "text": f"Filler line {n}."}
                 for n in range(12)]
        rows += [{"type": "SPOKEN", "text": "I will not say it again."}]
        return rows

    def test_the_window_grows_until_a_cast_name_appears(self):
        from gold_set_builder import context
        book = self._book()
        target = len(book) - 1
        narrow, _ = context(book, target, before=2, after=1)
        self.assertNotIn("Roxy", narrow)
        wide, _ = context(book, target, before=2, after=1,
                          names={"ROXY"}, min_names=1)
        self.assertIn("Roxy", wide)

    def test_expansion_is_capped(self):
        from gold_set_builder import context
        book = self._book()
        before, after = context(book, len(book) - 1, before=2, after=1,
                                names={"NOBODY"}, min_names=1,
                                max_before=4, max_after=2)
        self.assertLessEqual(len(before.split("\n\n")), 5)

    def test_names_match_case_insensitively(self):
        # BRI-CHAN vs the book's "Bri-chan": str.title() has broken this
        # project three times, so the window search must not depend on case.
        from gold_set_builder import context
        book = [{"type": "NARRATOR", "text": "Bri-chan waved."},
                {"type": "NARRATOR", "text": "Filler."},
                {"type": "SPOKEN", "text": "Move out."}]
        wide, _ = context(book, 2, before=1, after=0,
                          names={"BRI-CHAN"}, min_names=1)
        self.assertIn("Bri-chan", wide)

    def test_no_names_given_means_a_fixed_window(self):
        from gold_set_builder import context
        book = self._book()
        self.assertEqual(context(book, len(book) - 1, before=2, after=1),
                         context(book, len(book) - 1, before=2, after=1,
                                 names=None))


class ContextTest(unittest.TestCase):
    def test_context_comes_from_the_segmented_entries_in_order(self):
        before, after = context(_segmented(), 3, before=2, after=2)
        self.assertIn("Roxy shook her head.", before)
        self.assertIn("Eris looked away.", after)
        self.assertNotIn("Sorry.", before)

    def test_context_is_clamped_at_the_book_edges(self):
        before, after = context(_segmented(), 1, before=9, after=9)
        self.assertIn("The room was cold.", before)
        self.assertTrue(after)


class BuildTest(unittest.TestCase):
    def test_batches_are_capped_so_a_judge_cannot_silently_truncate(self):
        batches, _pool = build(_segmented(), "b", count=2, batch_size=1)
        self.assertEqual([1, 1], [len(x["rows"]) for x in batches])
        self.assertEqual("1 of 2", batches[0]["batch"])

    def test_rows_carry_blank_answer_fields_and_an_index(self):
        batches, _ = build(_segmented(), "b", count=1, batch_size=5)
        row = batches[0]["rows"][0]
        self.assertEqual("", row["ANSWER"])
        self.assertIn("entry_index", row)
        self.assertTrue(row["id"].startswith("b-"))

    def test_sampling_is_deterministic_for_a_seed(self):
        a, _ = build(_segmented(), "b", count=2, batch_size=5, seed=3)
        b, _ = build(_segmented(), "b", count=2, batch_size=5, seed=3)
        self.assertEqual(a, b)


class ValidationTest(unittest.TestCase):
    def _batches(self):
        return build(_segmented(), "b", count=2, batch_size=5)[0]

    def test_unanswered_rows_are_reported(self):
        problems = validate({}, self._batches())
        self.assertTrue(any("unanswered" in p for p in problems))

    def test_an_answer_absent_from_the_passage_is_flagged(self):
        batches = self._batches()
        gold_id = batches[0]["rows"][0]["id"]
        problems = validate({gold_id: {"answer": "FUTURE_ME", "reasoning": ""}},
                            batches)
        self.assertTrue(any("invented" in p for p in problems))

    def test_narrator_and_ambiguous_are_never_flagged_as_invented(self):
        batches = self._batches()
        answers = {row["id"]: {"answer": value, "reasoning": ""}
                   for row, value in zip(
                       [r for b in batches for r in b["rows"]],
                       ["NARRATOR", "AMBIGUOUS"])}
        self.assertEqual([], [p for p in validate(answers, batches)
                              if "invented" in p])

    def test_answers_for_unknown_ids_are_reported(self):
        problems = validate({"b-99999": {"answer": "ROXY", "reasoning": ""}},
                            self._batches())
        self.assertTrue(any("unknown ids" in p for p in problems))


class MergeTest(unittest.TestCase):
    def test_fixture_matches_the_shape_the_scorer_expects(self):
        batches = build(_segmented(), "b", count=2, batch_size=5)[0]
        gold_id = batches[0]["rows"][0]["id"]
        fixture = merge({gold_id: {"answer": "ROXY", "reasoning": "tag"}},
                        batches, "b", "run", "judge", [["RUDEUS", "RUDI"]])
        entry = fixture["entries"][0]
        self.assertEqual({"id", "book", "entry_index", "line",
                          "expected_speaker", "judged_by", "reasoning"},
                         set(entry))
        self.assertEqual([["RUDEUS", "RUDI"]], fixture["aliases"])

    def test_blank_answers_are_dropped_not_merged_as_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "f.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"rows": [{"id": "b-1", "ANSWER": "  ", "reasoning": ""},
                                    {"id": "b-2", "ANSWER": "ROXY", "reasoning": ""}]},
                          handle)
            self.assertEqual(["b-2"], sorted(read_filled([path])))


class AgreementTest(unittest.TestCase):
    """The protocol that made the existing set trustworthy: two readers, and a
    human only sees where they differ."""

    def test_only_disagreements_are_returned(self):
        first = {"a": {"answer": "ROXY"}, "b": {"answer": "ERIS"}}
        second = {"a": {"answer": "ROXY"}, "b": {"answer": "NINA"}}
        agreed, differ = agreement(first, second)
        self.assertEqual(1, agreed)
        self.assertEqual([("b", "ERIS", "NINA")], differ)

    def test_alias_forms_count_as_agreement(self):
        # RUDI and RUDEUS are one character; scoring them as a disagreement
        # cost 14 of 147 lines when the scorer lacked alias support.
        first = {"a": {"answer": "RUDI"}}
        second = {"a": {"answer": "RUDEUS"}}
        agreed, differ = agreement(first, second, [["RUDEUS", "RUDI"]])
        self.assertEqual((1, []), (agreed, differ))

    def test_lines_only_one_judge_saw_are_ignored(self):
        agreed, differ = agreement({"a": {"answer": "ROXY"}},
                                   {"b": {"answer": "ERIS"}})
        self.assertEqual((0, []), (agreed, differ))


if __name__ == "__main__":
    unittest.main()


class SupportClassificationTest(unittest.TestCase):
    """A name absent from the window is not the same as an invented one.

    The first validator conflated them, so a judge following the rule marked 13
    answerable lines AMBIGUOUS - lines whose speaker is recoverable from 'he',
    'she' or first-person continuity but whose name is not printed nearby.
    Retaining those as ambiguous gold would reward abstention on recoverable
    lines and silently change the task being measured.
    """

    ROW = {"passage_before": "She turned away.", "line": "I said no.",
           "passage_after": "The door closed."}
    BOOK = "Roxy walked in. She turned away. I said no. The door closed."

    def test_a_name_printed_in_the_window_is_window_supported(self):
        from gold_set_builder import support_for
        row = dict(self.ROW, passage_after="Roxy shook her head.")
        self.assertEqual("window", support_for("ROXY", row, self.BOOK))

    def test_a_name_only_in_the_wider_book_is_inferred_not_invented(self):
        from gold_set_builder import support_for
        self.assertEqual("book", support_for("ROXY", self.ROW, self.BOOK))

    def test_a_name_absent_from_the_book_is_invented(self):
        from gold_set_builder import support_for
        self.assertEqual("absent", support_for("FUTURE_ME", self.ROW, self.BOOK))

    def test_narrator_and_ambiguous_are_not_names(self):
        from gold_set_builder import support_for
        for value in ("NARRATOR", "AMBIGUOUS", "UNKNOWN"):
            self.assertEqual("not_a_name", support_for(value, self.ROW, self.BOOK))

    def test_validation_accepts_a_wider_context_name_when_given_the_book(self):
        batches = [{"rows": [dict(self.ROW, id="b-1", entry_index=1)]}]
        answers = {"b-1": {"answer": "ROXY", "reasoning": ""}}
        self.assertEqual([], validate(answers, batches, self.BOOK))

    def test_validation_still_refuses_an_invented_name(self):
        batches = [{"rows": [dict(self.ROW, id="b-1", entry_index=1)]}]
        answers = {"b-1": {"answer": "FUTURE_ME", "reasoning": ""}}
        self.assertTrue(any("invented" in p for p in
                            validate(answers, batches, self.BOOK)))

    def test_summary_counts_where_the_evidence_lives(self):
        from gold_set_builder import support_summary
        batches = [{"rows": [dict(self.ROW, id="b-1", entry_index=1),
                             dict(self.ROW, id="b-2", entry_index=2)]}]
        answers = {"b-1": {"answer": "ROXY", "reasoning": ""},
                   "b-2": {"answer": "AMBIGUOUS", "reasoning": ""}}
        counts, book_only = support_summary(answers, batches, self.BOOK)
        self.assertEqual(1, counts["book"])
        self.assertEqual(["b-1"], book_only)


class InstructionsTest(unittest.TestCase):
    def test_judges_are_told_inference_is_allowed(self):
        from gold_set_builder import INSTRUCTIONS
        self.assertIn("legitimate inference", INSTRUCTIONS)
        self.assertIn("appears nowhere in the book", INSTRUCTIONS)
