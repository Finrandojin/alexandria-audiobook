"""Standalone unit tests for speaker_canon.py.

Run directly (no pytest required):
    python app/test_speaker_canon.py

Exits 0 on all-pass, nonzero if any assertion fails.
"""

import sys
import copy

from speaker_canon import canonicalize, suggest_aliases

_failures = []


def check(condition, message):
    if not condition:
        _failures.append(message)


def check_equal(actual, expected, label):
    check(actual == expected, f"{label}: expected {expected!r}, got {actual!r}")


def test_basic_case_and_whitespace_normalization():
    check_equal(canonicalize("mark"), "MARK", "lowercase 'mark'")
    check_equal(canonicalize(" MARK "), "MARK", "padded ' MARK '")
    check_equal(canonicalize("  Mark   Twain  "), "MARK TWAIN", "internal whitespace run collapse")


def test_parenthetical_removal():
    check_equal(canonicalize("MARK (shouting)"), "MARK", "trailing parenthetical")
    check_equal(canonicalize("MARK (angrily) enters"), "MARK ENTERS", "mid-string parenthetical")


def test_honorific_stripping():
    check_equal(canonicalize("Mr. Mark"), "MARK", "'Mr. Mark' honorific")
    check_equal(canonicalize("mr mark"), "MARK", "lowercase honorific, no period")
    check_equal(canonicalize("Dr. Smith"), "SMITH", "'Dr. Smith' honorific")
    check_equal(canonicalize("Professor Xavier"), "XAVIER", "'Professor' honorific")
    check_equal(canonicalize("Captain Hook"), "HOOK", "'Captain' honorific")
    check_equal(canonicalize("Mrs. Robinson"), "ROBINSON", "'Mrs.' honorific")


def test_honorific_alone_never_empty():
    check_equal(canonicalize("Dr."), "DR", "'Dr.' alone must not canonicalize to empty")
    check_equal(canonicalize("Mr."), "MR", "'Mr.' alone must not canonicalize to empty")
    check_equal(canonicalize("Sir"), "SIR", "'Sir' alone must not canonicalize to empty")


def test_all_variants_converge():
    variants = ["MARK (shouting)", "Mr. Mark", "mark", " MARK "]
    canon = {canonicalize(v) for v in variants}
    check(canon == {"MARK"}, f"all MARK variants should canonicalize identically, got {canon}")


def test_accent_normalization():
    check_equal(canonicalize("José"), "JOSE", "'José' accent stripped")
    check_equal(canonicalize("François"), "FRANCOIS", "'François' accent stripped")


def test_narrator_canonicalization():
    check_equal(canonicalize("narrator"), "NARRATOR", "'narrator' lowercase")
    check_equal(canonicalize("Narrator"), "NARRATOR", "'Narrator' titlecase")
    check_equal(canonicalize("NARRATOR"), "NARRATOR", "'NARRATOR' already canonical")
    check_equal(canonicalize("  narrator  "), "NARRATOR", "padded narrator")


def test_apostrophes_preserved():
    check_equal(canonicalize("O'Brien"), "O'BRIEN", "apostrophe kept in O'Brien")
    check_equal(canonicalize("o'brien"), "O'BRIEN", "lowercase o'brien")


def test_hyphens_preserved():
    check_equal(canonicalize("Jean-Luc"), "JEAN-LUC", "hyphen kept in Jean-Luc")


def test_stray_punctuation_stripped():
    check_equal(canonicalize("Mark!"), "MARK", "trailing exclamation stripped")
    check_equal(canonicalize("Mark:"), "MARK", "trailing colon stripped")
    check_equal(canonicalize('"Mark"'), "MARK", "surrounding quotes stripped")


def test_empty_and_whitespace_input():
    check_equal(canonicalize(""), "", "empty string input")
    check_equal(canonicalize("   "), "", "whitespace-only input")
    check_equal(canonicalize(None), "", "None input")


def test_wrapping_quotes_stripped():
    # F11: LLM-emitted labels wrapped in quotes must unwrap to the same
    # canonical form as the unquoted name, or the roster fragments.
    check_equal(canonicalize("'Mother of Monsters'"), "MOTHER OF MONSTERS",
                "straight single quotes wrapping a multi-word name")
    check_equal(canonicalize('"BLACK SCOUT"'), "BLACK SCOUT",
                "straight double quotes wrapping a name")
    check_equal(canonicalize("‘Mother of Monsters’"), "MOTHER OF MONSTERS",
                "curly single quotes (‘...’) wrapping a name")
    check_equal(canonicalize("“Black Scout”"), "BLACK SCOUT",
                "curly double quotes (“...”) wrapping a name")
    check_equal(canonicalize("'\"BLACK SCOUT\"'"), "BLACK SCOUT",
                "nested single-then-double wrapping quotes")
    check_equal(canonicalize("''X''"), "X",
                "double-wrapped straight single quotes unwind fully")


def test_unmatched_apostrophes_survive_wrapping_fix():
    # These must be UNCHANGED by the F11 wrapping-quote fix: only a matched
    # pair at both ends is a "wrapper"; a lone leading/trailing apostrophe
    # is possessive or elision, not a wrapper.
    check_equal(canonicalize("O'Brien"), "O'BRIEN", "internal apostrophe untouched")
    check_equal(canonicalize("Jones'"), "JONES'", "unmatched trailing possessive apostrophe kept")
    check_equal(canonicalize("'tis"), "'TIS", "unmatched leading elision apostrophe kept")


def test_wrapping_quotes_and_roster_fragmentation():
    # The concrete production bug: a quoted and unquoted form of the same
    # name must canonicalize identically so the roster doesn't fragment.
    check_equal(
        canonicalize("'MOTHER OF MONSTERS'"),
        canonicalize("MOTHER OF MONSTERS"),
        "quoted and unquoted forms must converge to the same canonical name",
    )


def test_idempotency():
    samples = [
        "MARK (shouting)", "Mr. Mark", "Dr.", "José", "narrator", "O'Brien",
        "'Mother of Monsters'", '"BLACK SCOUT"',
        "‘Mother of Monsters’", "“Black Scout”",
        "'\"BLACK SCOUT\"'", "''X''", "Jones'", "'tis",
    ]
    for s in samples:
        once = canonicalize(s)
        twice = canonicalize(once)
        check_equal(twice, once, f"idempotency for {s!r}")


def test_suggest_aliases_expected_pairs():
    roster = ["JON", "JOHN", "ELLA", "BELLA", "MARCUS", "ELENA"]
    roster_copy = copy.deepcopy(roster)

    suggestions = suggest_aliases(roster)

    def has_pair(a, b):
        for s in suggestions:
            names = {s["name"], s["alias_of"]}
            if names == {a, b}:
                return True
        return False

    check(has_pair("JON", "JOHN"), "expected a JON/JOHN suggestion")
    check(has_pair("ELLA", "BELLA"), "expected an ELLA/BELLA suggestion")
    check(not has_pair("MARCUS", "ELENA"), "MARCUS/ELENA must NOT be suggested (unrelated names)")

    # Roster argument must be untouched (no mutation, no merging).
    check_equal(roster, roster_copy, "suggest_aliases must not mutate its roster argument")
    check_equal(len(roster), 6, "suggest_aliases must not remove/merge roster entries")

    for s in suggestions:
        check("name" in s and "alias_of" in s and "score" in s, f"suggestion missing keys: {s}")
        check(0.0 <= s["score"] <= 1.0, f"score out of [0,1] range: {s}")


def test_suggest_aliases_excludes_narrator():
    roster = ["NARRATOR", "narrator", "MARK", "MARC"]
    suggestions = suggest_aliases(roster)
    for s in suggestions:
        check(s["name"] != "NARRATOR" and s["alias_of"] != "NARRATOR",
              f"NARRATOR must never appear in suggestions: {s}")


def test_suggest_aliases_empty_and_single():
    check_equal(suggest_aliases([]), [], "empty roster -> no suggestions")
    check_equal(suggest_aliases(["SOLO"]), [], "single-name roster -> no suggestions")


def test_suggest_aliases_direction_rule():
    # Shorter name is suggested as the alias of the longer name.
    suggestions = suggest_aliases(["JON", "JOHN"])
    check(len(suggestions) == 1, "expected exactly one JON/JOHN suggestion")
    if suggestions:
        s = suggestions[0]
        check_equal(s["name"], "JON", "shorter name should be 'name' (the alias)")
        check_equal(s["alias_of"], "JOHN", "longer name should be 'alias_of' (the target)")


def main():
    tests = [
        test_basic_case_and_whitespace_normalization,
        test_parenthetical_removal,
        test_honorific_stripping,
        test_honorific_alone_never_empty,
        test_all_variants_converge,
        test_accent_normalization,
        test_narrator_canonicalization,
        test_apostrophes_preserved,
        test_hyphens_preserved,
        test_stray_punctuation_stripped,
        test_empty_and_whitespace_input,
        test_wrapping_quotes_stripped,
        test_unmatched_apostrophes_survive_wrapping_fix,
        test_wrapping_quotes_and_roster_fragmentation,
        test_idempotency,
        test_suggest_aliases_expected_pairs,
        test_suggest_aliases_excludes_narrator,
        test_suggest_aliases_empty_and_single,
        test_suggest_aliases_direction_rule,
    ]

    for t in tests:
        t()

    if _failures:
        print(f"FAILED: {len(_failures)} assertion(s) failed")
        for f in _failures:
            print(f"  - {f}")
        return 1

    print(f"OK: all tests passed ({len(tests)} test functions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
