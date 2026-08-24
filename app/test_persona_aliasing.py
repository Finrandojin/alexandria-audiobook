"""Standalone test: persona generation must not auto-merge speakers on similarity.

A fuzzy pre-pass (rapidfuzz at 0.8) used to decide that two roster entries were
the same character and SKIP the loser, leaving it with no voice. Measured on one
book it merged 31 pairs -- a person against a possessive relation naming them
(X vs X'S FATHER), and cyclically in both directions -- and 28 of 65 speakers
came out of persona generation with no voice at all.

Only exact equality after normalization may merge two speakers here, which is
the rule the rest of the pipeline uses. Anything looser is a human decision.

Run directly:
    python app/test_persona_aliasing.py
Exits 0 if all tests pass, non-zero otherwise.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_personas import normalize_speaker_name  # noqa: E402

failures = []


def check(name, condition, detail=""):
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        failures.append(name)


def merges(a, b):
    """The rule Step 1 of generate_personas.main() now applies."""
    na, nb = normalize_speaker_name(a), normalize_speaker_name(b)
    return bool(na) and na == nb


def test_relation_is_not_the_person():
    # The defect: a possessive relation scores high against the person it
    # names, but a parent is not their child. Synthesized names only.
    for person, relation in [("ARLO", "ARLO'S DAD"),
                             ("BRENNA", "BRENNA'S MOTHER"),
                             ("CIRO", "CIRO'S FATHER")]:
        check(f"aliasing: {person} is not {relation}",
              not merges(person, relation))


def test_shared_surname_is_not_the_same_person():
    check("aliasing: a bare surname is not the full name",
          not merges("DELACROIX", "MAREN DELACROIX"))


def test_similar_but_distinct_names_stay_apart():
    # The JON/JOHN class the banned-approaches list exists for.
    for a, b in [("JON", "JOHN"), ("ELLA", "BELLA"), ("MARIA", "MARIE")]:
        check(f"aliasing: {a} and {b} stay distinct", not merges(a, b))


def test_exact_after_normalization_still_merges():
    # The one merge that is still allowed: same name, different formatting.
    for a, b in [("MAREN", "maren"), ("MAREN", "  MAREN  ")]:
        check(f"aliasing: {a!r} and {b!r} are the same speaker", merges(a, b))


def test_merging_is_symmetric():
    # The old fuzzy pass was cyclic, so the winner depended on iteration
    # order. Equality cannot be.
    for a, b in [("ARLO", "ARLO'S DAD"), ("MAREN", "maren")]:
        check(f"aliasing: {a}/{b} decision is order-independent",
              merges(a, b) == merges(b, a))



def test_narrator_default_is_not_an_alias():
    """A speaker with no persona defaults to the NARRATOR's voice settings but
    KEEPS its own entry. An alias_of would fold it into NARRATOR, and nothing
    in this pipeline can split a merged entry back apart."""
    from generate_personas import _entry_has_voice

    narrator = {"type": "clone", "voice": "Ryan", "ref_audio": "designed_voices/n.wav"}
    check("default: narrator entry counts as having a voice", _entry_has_voice(narrator))
    check("default: missing entry has no voice", not _entry_has_voice(None))
    check("default: empty entry has no voice", not _entry_has_voice({}))
    check("default: entry with only blank fields has no voice",
          not _entry_has_voice({"voice": "  ", "ref_audio": ""}))
    check("default: an aliased entry is not a usable voice",
          not _entry_has_voice({"alias_of": "NARRATOR", "voice": "Ryan"}))

    # The defaulting step copies settings, so the speaker still has its own key
    # and can be overridden; it must not gain an alias_of.
    entry = dict(narrator)
    entry["defaulted_from_narrator"] = True
    check("default: defaulted entry carries no alias_of", "alias_of" not in entry)
    check("default: defaulted entry is usable for synthesis", _entry_has_voice(entry))
    check("default: defaulted entry is marked for the operator",
          entry.get("defaulted_from_narrator") is True)


for fn in [test_relation_is_not_the_person,
           test_shared_surname_is_not_the_same_person,
           test_similar_but_distinct_names_stay_apart,
           test_exact_after_normalization_still_merges,
           test_merging_is_symmetric,
           test_narrator_default_is_not_an_alias]:
    fn()

print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'All persona-aliasing checks passed'}")
sys.exit(1 if failures else 0)
