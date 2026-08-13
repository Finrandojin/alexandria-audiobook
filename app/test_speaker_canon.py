"""Standalone unit tests for speaker_canon.py.

Run directly (no pytest required):
    python app/test_speaker_canon.py

Exits 0 on all-pass, nonzero if any assertion fails.
"""

import sys
import copy

from speaker_canon import (
    GENDERED_TITLES,
    canonicalize,
    remember_in_roster,
    resolve_against_roster,
    roster_key,
    suggest_aliases,
    attest_label,
    attest_speaker,
    repair_speaker,
    source_word_index,
    ATTESTED,
    UNATTESTED,
    UNVERIFIABLE,
    _core_tokens,
    _is_distance_one,
)

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


def test_rank_titles_are_dropped():
    # Rank says nothing about identity that the surname doesn't; the same
    # person is "Dr. Millman", "Doctor Millman" and "Millman" in one book.
    check_equal(canonicalize("Dr. Smith"), "SMITH", "'Dr. Smith' rank title")
    check_equal(canonicalize("Dr. Millman"), "MILLMAN", "'Dr. Millman' rank title")
    check_equal(canonicalize("Professor Xavier"), "XAVIER", "'Professor' rank title")
    check_equal(canonicalize("Captain Hook"), "HOOK", "'Captain' rank title")
    check_equal(canonicalize("Lt. Col. Blimp"), "BLIMP", "stacked rank titles")


def test_gendered_titles_are_preserved_and_normalized():
    # The husband/wife bug: dropping these merged two characters into one
    # voice, irreversibly, at annotation time.
    check_equal(canonicalize("Mr. Smith"), "MISTER SMITH", "'Mr.' preserved as MISTER")
    check_equal(canonicalize("mr mark"), "MISTER MARK", "lowercase title, no period")
    check_equal(canonicalize("Mrs. Robinson"), "MISSUS ROBINSON", "'Mrs.' preserved as MISSUS")
    check_equal(canonicalize("Mister Smith"), "MISTER SMITH", "spelled-out 'Mister' unifies with 'Mr.'")
    check_equal(canonicalize("Missus Robinson"), "MISSUS ROBINSON", "spelled-out 'Missus'")
    check_equal(canonicalize("Ms. Doe"), "MS DOE", "'Ms.' preserved")
    check_equal(canonicalize("Miss Doe"), "MISS DOE", "'Miss' preserved")
    check_equal(canonicalize("Mx. Doe"), "MX DOE", "'Mx.' preserved")
    check_equal(canonicalize("Sir John"), "SIR JOHN", "'Sir' preserved")
    check_equal(canonicalize("Lady John"), "LADY JOHN", "'Lady' preserved")
    check_equal(canonicalize("Lord Byron"), "LORD BYRON", "'Lord' preserved")
    check_equal(canonicalize("Dame Edna"), "DAME EDNA", "'Dame' preserved")


def test_gendered_titles_are_language_preserving():
    # French abbreviations unify with their spelled-out forms (one character),
    # but are never folded onto the English titles (a different word in the
    # author's text, and MONSIEUR/MADAME must stay apart).
    check_equal(canonicalize("Mme Bovary"), "MADAME BOVARY", "'Mme' -> MADAME")
    check_equal(canonicalize("Madame Bovary"), "MADAME BOVARY", "'Madame' -> MADAME")
    check_equal(canonicalize("M. Marambot"), "MONSIEUR MARAMBOT", "'M.' -> MONSIEUR")
    check_equal(canonicalize("Monsieur Marambot"), "MONSIEUR MARAMBOT", "'Monsieur' -> MONSIEUR")
    check_equal(canonicalize("Mlle Cocotte"), "MADEMOISELLE COCOTTE", "'Mlle' -> MADEMOISELLE")
    check_equal(canonicalize("Mademoiselle Cocotte"), "MADEMOISELLE COCOTTE",
                "'Mademoiselle' -> MADEMOISELLE")
    check(canonicalize("Mme Dufour") != canonicalize("Monsieur Dufour"),
          "MADAME DUFOUR and MONSIEUR DUFOUR must stay distinct")
    check(canonicalize("Mrs. Bovary") != canonicalize("Mme Bovary"),
          "English MISSUS is not folded onto French MADAME")


def test_gendered_titles_keep_husband_and_wife_apart():
    # The defect this two-class split exists to fix.
    check(canonicalize("Mr. Smith") != canonicalize("Mrs. Smith"),
          "Mr. Smith and Mrs. Smith must NOT canonicalize to the same name")
    check(canonicalize("Mr. Smith") != canonicalize("Smith"),
          "Mr. Smith and a bare Smith are distinct roster entries")


def test_fr_is_preserved_not_dropped():
    # FR is a name-constituent risk of the same class as the already-removed
    # ST. Preserving it contains the risk: the worst case is a stray prefix,
    # not a silently different character.
    check_equal(canonicalize("Fr. Simon"), "FR SIMON", "'Fr.' preserved")
    check_equal(canonicalize("Fraser"), "FRASER", "'Fraser' is not 'Fr' + 'aser'")
    # FATHER was never a title and must not become one.
    check_equal(canonicalize("Father Milon"), "FATHER MILON", "'Father' is part of the name")


def test_title_alone_never_empty():
    check_equal(canonicalize("Dr."), "DR", "'Dr.' alone must not canonicalize to empty")
    check_equal(canonicalize("Mr."), "MR", "'Mr.' alone must not canonicalize to empty")
    check_equal(canonicalize("Mrs."), "MRS", "'Mrs.' alone must not canonicalize to empty")
    check_equal(canonicalize("Sir"), "SIR", "'Sir' alone must not canonicalize to empty")
    check_equal(canonicalize("M."), "M", "a bare 'M.' keeps itself rather than becoming MONSIEUR")


def test_all_variants_converge():
    variants = ["MARK (shouting)", "mark", " MARK "]
    canon = {canonicalize(v) for v in variants}
    check(canon == {"MARK"}, f"all MARK variants should canonicalize identically, got {canon}")
    titled = {canonicalize(v) for v in ["Mr. Mark", "mr mark", "MISTER MARK", "Mister Mark"]}
    check(titled == {"MISTER MARK"},
          f"all MISTER MARK variants should canonicalize identically, got {titled}")


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
        # Stacked honorifics: a single-pass strip made these NON-idempotent,
        # and the pipeline canonicalizes twice (entry, then roster), so the
        # script and the voices roster disagreed and the character was split
        # across two voices.
        "Mr. St. Clair", "Sir Lt. Col. Blimp", "Mrs. Dr. Watson",
        "Dr. Dr.", "Mr Mr Mr Smith", "St. John", "ST JOHN RIVERS",
        "'Mother of Monsters'", '"BLACK SCOUT"',
        "‘Mother of Monsters’", "“Black Scout”",
        "'\"BLACK SCOUT\"'", "''X''", "Jones'", "'tis",
        # Preserved gender-marking titles re-enter canonicalize() as their
        # normalized spellings, so those spellings must themselves be
        # recognized titles that map to themselves.
        "Mr. Smith", "MISTER SMITH", "Mrs. Smith", "MISSUS SMITH",
        "Mme Bovary", "MADAME BOVARY", "M. Marambot", "MONSIEUR MARAMBOT",
        "Mlle Cocotte", "MADEMOISELLE COCOTTE", "Ms. Doe", "Miss Doe",
        "Mx. Doe", "Sir John", "Lady John", "Fr. Simon", "Mrs.", "M.",
        "Mr. Mrs.", "Mrs. Dr. Watson", "Sir Lt. Col. Blimp",
    ]
    for s in samples:
        once = canonicalize(s)
        twice = canonicalize(once)
        check_equal(twice, once, f"idempotency for {s!r}")



def test_stacked_titles_resolve_to_a_fixpoint():
    # The whole run of leading titles is consumed in one pass, and the FIRST
    # gender-marking title in the run becomes the preserved prefix.
    check_equal(canonicalize("Sir Lt. Col. Blimp"), "SIR BLIMP", "three stacked titles")
    check_equal(canonicalize("Mrs. Dr. Watson"), "MISSUS WATSON", "two stacked titles")
    check_equal(canonicalize("Mr Mr Mr Smith"), "MISTER SMITH", "repeated title")
    check_equal(canonicalize("Dr. Mrs. Watson"), "MISSUS WATSON",
                "a rank title before the gendered one does not shadow it")
    # The never-empty guard survives the loop.
    check_equal(canonicalize("Dr."), "DR", "bare title keeps itself")
    check_equal(canonicalize("Dr. Dr."), "DR", "all-title name keeps the last one")
    check_equal(canonicalize("Mr. Mrs."), "MRS", "all-title name is never emptied")


def test_double_canonicalization_matches_single():
    # The live failure mode: generate_script.py canonicalizes for the script
    # entry and remember_in_roster canonicalizes again for the roster. Any
    # name where those two disagree is a character split across two voices.
    for name in ["Mr. St. Clair", "Sir Lt. Col. Blimp", "Mrs. Dr. Watson",
                 "St. John Rivers", "Dr. Watson", "Captain Hook"]:
        once = canonicalize(name)
        check_equal(canonicalize(once), once, f"double canonicalization of {name!r}")


def test_saint_names_are_not_honorifics():
    # "ST" is a name constituent, not a title. While it was in _HONORIFICS,
    # "ST JOHN RIVERS" became "JOHN RIVERS" -- a different character.
    check_equal(canonicalize("St. John Rivers"), "ST JOHN RIVERS", "St. John Rivers")
    check_equal(canonicalize("ST JOHN RIVERS"), "ST JOHN RIVERS", "already-canonical form")
    check_equal(canonicalize("Mr. St. Clair"), "MISTER ST CLAIR", "title resolved, saint kept")
    check_equal(canonicalize("St. Laurent"), "ST LAURENT", "St. Laurent")
    # The two spellings of the same character still unify via the roster key.
    check_equal(roster_key("ST. JOHN"), roster_key("ST JOHN"), "punctuated saint name keys alike")


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



# ---------------------------------------------------------------------------
# Tier 1b: resolve_against_roster()
# ---------------------------------------------------------------------------


def _index(*names):
    index = {}
    for name in names:
        remember_in_roster(index, name)
    return index


def test_roster_key_strips_whitespace_only():
    check_equal(roster_key("ABBE MARIGNAN"), "ABBEMARIGNAN", "spaced key")
    check_equal(roster_key("abbemarignan"), "ABBEMARIGNAN", "unspaced key")
    check(roster_key("JON") != roster_key("JOHN"), "JON/JOHN keys must differ")
    check(roster_key("ELLA") != roster_key("BELLA"), "ELLA/BELLA keys must differ")


def test_most_boundary_marks_wins_in_both_arrival_orders():
    # The selection rule is ORDER-INDEPENDENT: the more-punctuated spelling
    # wins whichever one the roster met first. This is what makes generation
    # (chunk order) and review (entry order) agree, and what stops a single
    # malformed first sighting becoming canonical for the whole book.
    spaced_first = _index("ABBE MARIGNAN", "ABBEMARIGNAN")
    check_equal(resolve_against_roster("ABBEMARIGNAN", spaced_first), "ABBE MARIGNAN",
                "spaced form wins when seen first")
    unspaced_first = _index("ABBEMARIGNAN", "ABBE MARIGNAN")
    check_equal(resolve_against_roster("ABBEMARIGNAN", unspaced_first), "ABBE MARIGNAN",
                "spaced form ALSO wins when the malformed one was seen first")
    check_equal(sorted(spaced_first.values()), sorted(unspaced_first.values()),
                "both arrival orders converge on the same roster")



def test_punctuation_variants_unify():
    check_equal(roster_key("O'BRIEN"), roster_key("OBRIEN"), "apostrophe key")
    check_equal(roster_key("JEAN-LUC"), roster_key("JEAN LUC"), "hyphen vs space key")
    check_equal(roster_key("ABBE MARIGNAN'S NIECE"), roster_key("ABBEMARIGNAN'S NIECE"),
                "real production pair keys alike")
    check(roster_key("O'BRIEN") != roster_key("OBRIAN"), "OBRIAN is a different name")


def test_apostrophe_variant_wins_in_both_arrival_orders():
    forward = _index("O'BRIEN", "OBRIEN")
    check_equal(resolve_against_roster("OBRIEN", forward), "O'BRIEN",
                "apostrophe form wins when seen first")
    backward = _index("OBRIEN", "O'BRIEN")
    check_equal(resolve_against_roster("OBRIEN", backward), "O'BRIEN",
                "apostrophe form ALSO wins when the bare form was seen first")
    check_equal(sorted(forward.values()), sorted(backward.values()),
                "both arrival orders converge")


def test_hyphen_vs_space_is_a_deterministic_tie():
    # Same letters, same length -- neither is more likely correct, so the
    # incumbent keeps the slot. Deterministic, and documented as a tie.
    hyphen_first = _index("JEAN-LUC", "JEAN LUC")
    check_equal(list(hyphen_first.values()), ["JEAN-LUC"], "hyphen incumbent kept")
    space_first = _index("JEAN LUC", "JEAN-LUC")
    check_equal(list(space_first.values()), ["JEAN LUC"], "space incumbent kept")
    check_equal(len(hyphen_first), 1, "the pair is still unified either way")
    check_equal(len(space_first), 1, "the pair is still unified either way")


def test_boundary_marks_beat_a_bare_run_of_letters():
    # Longer canonical form == more boundary marks, since the letters match.
    index = _index("ABBEMARIGNANSNIECE")
    check_equal(remember_in_roster(index, "ABBE MARIGNAN'S NIECE"),
                "ABBE MARIGNAN'S NIECE", "most-punctuated spelling is promoted")


def test_remember_returns_the_winner_and_promotes_in_place():
    index = _index("ABBEMARIGNAN")
    check_equal(remember_in_roster(index, "ABBE MARIGNAN"), "ABBE MARIGNAN",
                "a better spelling is promoted and returned immediately")
    check_equal(list(index.values()), ["ABBE MARIGNAN"], "index holds only the winner")
    check_equal(remember_in_roster(index, "ABBEMARIGNAN"), "ABBE MARIGNAN",
                "a worse spelling is normalized onto the winner")


def test_ties_keep_the_incumbent():
    index = _index("JEAN LUC")
    check_equal(remember_in_roster(index, "JEAN-LUC PICARD"), "JEAN-LUC PICARD",
                "different key -- unrelated name, not a tie")
    tie = _index("MARY ANNE")
    check_equal(remember_in_roster(tie, "MARY ANNE"), "MARY ANNE",
                "identical spelling is a no-op")
    check_equal(len(tie), 1, "no spurious entries on a tie")


def test_resolve_snaps_onto_established_spelling():
    roster = _index("ABBE MARIGNAN", "MARCUS")
    check_equal(resolve_against_roster("ABBEMARIGNAN", roster), "ABBE MARIGNAN",
                "drifted spelling snaps onto the established spaced form")
    check_equal(resolve_against_roster("abbe marignan", roster), "ABBE MARIGNAN",
                "already-correct spelling is unchanged")


def test_resolve_never_merges_similar_names():
    roster = _index("JON", "ELLA")
    check_equal(resolve_against_roster("JOHN", roster), "JOHN", "JOHN stays distinct from JON")
    check_equal(resolve_against_roster("BELLA", roster), "BELLA", "BELLA stays distinct from ELLA")
    check_equal(resolve_against_roster("JON", _index("JOHN")), "JON",
                "JON stays distinct from JOHN")
    check_equal(resolve_against_roster("ELLA", _index("BELLA")), "ELLA",
                "ELLA stays distinct from BELLA")
    both = _index("JON", "JOHN", "ELLA", "BELLA")
    check_equal(len(both), 4, "four similar names remain four roster entries")


def test_resolve_passthrough_and_narrator():
    roster = _index("ABBE MARIGNAN")
    check_equal(resolve_against_roster("Mr. Mark", roster), "MISTER MARK",
                "unmatched name is just canonicalized")
    check_equal(resolve_against_roster("NARRATOR", roster), "NARRATOR", "NARRATOR untouched")
    check_equal(resolve_against_roster("narrator", _index("NARRATOR")), "NARRATOR",
                "narrator still canonicalizes to NARRATOR with a roster present")
    check_equal(resolve_against_roster("", roster), "", "empty input")
    check_equal(resolve_against_roster("MARK", {}), "MARK", "empty roster")
    check_equal(resolve_against_roster("MARK", None), "MARK", "None roster")


def test_resolve_is_idempotent():
    roster = _index("ABBE MARIGNAN", "JON")
    once = resolve_against_roster("ABBEMARIGNAN", roster)
    check_equal(resolve_against_roster(once, roster), once, "idempotent resolution")
    check_equal(resolve_against_roster(resolve_against_roster("JOHN", roster), roster), "JOHN",
                "idempotent for a non-matching name")


def test_resolve_does_not_mutate_the_index():
    roster = _index("ABBE MARIGNAN")
    snapshot = copy.deepcopy(roster)
    resolve_against_roster("SOMEONE ELSE", roster)
    check_equal(roster, snapshot, "resolve_against_roster is read-only")


def test_empty_name_is_not_recorded():
    index = {}
    check_equal(remember_in_roster(index, ""), "", "empty name returns empty")
    check_equal(remember_in_roster(index, "   "), "", "whitespace-only name returns empty")
    check_equal(index, {}, "nothing recorded for an empty name")



# ---------------------------------------------------------------------------
# Real-roster safety property
#
# The 578 distinct canonical speaker labels produced by a real 976-chunk
# production run (9,374 script entries). Embedded verbatim so the safety
# property below is pinned permanently and needs no external file.
#
# The property: the roster key may unify spellings that differ only in their
# boundary marks, and on REAL data that must not sweep up two genuinely
# different characters. Measured: the whitespace-only key produced 2 collision
# families; the current, wider key (whitespace + hyphens + apostrophes)
# produces exactly the same 2 -- both the known ABBE MARIGNAN pair. Widening
# the key any further must be re-measured against this fixture first.
# ---------------------------------------------------------------------------

REAL_PRODUCTION_ROSTER = [
    "NARRATOR", "ZOLA", "MAUPASSANT", "FLAUBERT", "MERCHANT", "HOSTLER", "DRIVER",
    "TRAVELLER", "COUNT HUBERT", "LOISEAU", "BOULE DE SUIF", "MADAME LOISEAU", "NUN",
    "CORNUDET", "GERMAN OFFICER", "INNKEEPER", "COUNTESS", "BEADLE", "CARRE-LAMADON",
    "MONSIEUR FOLLENVIE", "MADAME CARRE-LAMADON", "MONSIEUR SAVAGE",
    "MONSIEUR MORISSOT", "MONSIEUR SAUVAGE", "MORISSOT", "SAUVAGE", "CAPTAIN",
    "SOLDIER", "GENERAL ROLAND", "CAPTAIN'S WIFE", "PIEDELOT", "LONG-LEGS",
    "YOUNGER WOMAN", "OLDER WOMAN", "BERTHINE", "CHARACTER", "BERTHINE'S FATHER",
    "MONSIEUR LAVIGNE", "JEAN KERDEREN", "LUC LE GANIDEC", "FATHER MILON", "COLONEL",
    "MASSAREL", "PEASANT", "PHYSICIAN", "COMMANDANT", "LIEUTENANT PICART", "NOTABLES",
    "DOCTOR", "M DE VARNETOT", "EX-LIEUTENANT PICART", "SENTINEL", "OLDER MAN",
    "GENERAL DE G", "M MARTINI", "PARISSE'S WIFE", "PARISSE", "JEAN DE CARMELIN",
    "GRIBOIS", "LIEUTENANT OTTO", "MAJOR", "GRAF VON FARSBERG", "FIFI",
    "LITTLE BARON WILHELM", "LIEUTENANT FRITZ", "PAMELA", "EVA", "RACHEL", "GROUP",
    "ENGLISHMAN", "M DUBUIS", "ENGLISHMEN", "COLONEL LAPORTE", "MOTHER SAVAGE",
    "MY FRIEND SERVAL", "COUNT DE GARENS", "MARCHAS", "PRIEST", "HERMANCE",
    "SISTER SAINT-BENEDICT", "PIQUEDENT", "WORKING-WOMAN", "ANGELE", "BARON D'ETRAILLE",
    "CHILD", "BROTHER-IN-LAW", "HENRIETTE", "PAUL", "WAITER", "CARAVAN", "CHENET",
    "MARIE-LOUISE", "MADAME CARAVAN", "DOCTOR CHENET", "LANDLORD", "DOMINO PLAYER",
    "CARAVAN'S WIFE", "BRAUX", "BRAUX'S WIFE", "CARAVAN'S MOTHER", "COMTE D'ETRAILLE",
    "RENE LAMANOIR", "LEON CHENAL", "MEDERIC", "RENARDET", "LA ROQUE", "PRINCIPETTE",
    "MAYOR", "MAGISTRATE", "MOTHER LA ROQUE", "WOODCUTTER", "RENADET", "SERVANT",
    "MAILLOCHON", "CHICOT", "LABOUISE", "MALOUREAU", "MOIRON", "WHEELWRIGHT",
    "WHEELWRIGHT'S WIFE", "JEAN", "GEORGES LOUIS", "JUDGE", "BERTHA", "BONNET", "MARIN",
    "PETITPAS", "MASSOULIGNY", "MME BRUMENT", "DEFENDANT BRUMENT", "CORNU", "JACQUES",
    "RAVET", "NURSES", "MOTHER", "WIFE", "BERTHE", "GILBERTE", "HENRI FONTAL", "FINOT",
    "PEASANT MAN", "PEASANT WOMAN", "OSIME FAVET", "BONIFACE", "CAVALIER", "MASTER",
    "CESAIRE DENTU", "ROSE", "MASTER VALLIN", "SPEAKER", "THEODULE SABOT",
    "VILLAGE RESIDENTS", "ABBE", "SABOT", "PADOIE", "VARAJOU", "BOY", "VARAROU",
    "OTHER", "M LOISEAU", "MME FORESTIER", "LOISEL", "MME LOISEAU", "JEWELER",
    "FORESTIER", "SOMEONE", "JOVIS", "M MALLET", "PAUL BESSAND", "HENRI SIMON",
    "PIERRE CARNIER", "JEAN D'ARVILLE", "FRANCOIS D'ARVILLE", "MARQUIS D'ARVILLE",
    "LADY", "ULRICH KUNSI", "FATHER HAUSER", "OLD HARI", "LOUISE", "JEAN HAUSER",
    "GASPARD HARI", "MOTHER HAUSER", "LOUISE HAUSER", "MONSIEUR PARENT", "JULIE",
    "PARENT", "LIMOUSIN", "GEORGE", "BARMAID", "ZOE", "QUEEN HORTENSE", "M CIMME",
    "CELESTE", "COLOMBEL", "LITTLE MAID", "MAID", "MME CIMME", "DYING WOMAN",
    "MME COLOMBEL", "CIMME", "TIMBUCTOO", "CHANTAL", "PEARL", "MME CHANTAL",
    "CHANTAL LADIES", "CHANTAL'S WIFE", "ABBE MARIGNAN", "ABBE MARIGNAN'S NIECE",
    "MELANIE", "COUNT JEAN DES BARRETS", "COMTESSE", "ABBE MAUDUIT",
    "COMTESSE'S HUSBAND", "ROSSET", "SAVAL", "ROMANTIN", "MATHILDE",
    "HECTOR DE GRIBELIN", "MME SIMON", "MME DE GRIBELIN", "HECTOR", "COUNT",
    "COMTESSE DE MASCARET", "COMTE DE MASCARET", "ROGER DE SALNIS", "BERNARD GRANDIN",
    "FRANCOIS TESSIER", "M FLOREAL FLAMEL", "MME FLOREAL FLAMEL", "MY UNCLE SOSTHENES",
    "ABBEMARIGNAN", "BOISRENE", "LAURENT", "THE CRIPPLE", "HENRI BONCLAIR",
    "MONSIEUR LERAS", "MADAME MARAMBALLE", "ALEXANDRE", "MARAMBALLE", "WORKMAN",
    "DOMESTIC", "PAUL PAVILLY", "PORTER", "ITALIAN WOMAN", "FRANCESCA RONDOLO",
    "MOTHER RONDOLO", "CARLOTTA", "MONSIEUR LANTIN", "MONSIEUR LANTIN'S WIFE",
    "ANTONIA SAVERNINI", "WIDOW SAVERNINI", "SAINT MICHAEL", "SATAN",
    "JACQUES DE RANDAL", "IRENE", "D'APREVAL", "MONSIEUR DE CADOUR", "MME DE CADOUR",
    "TELLIER", "TEURNEVAU", "PINIPESSE", "GENTLEMAN", "MADAME TELLIER", "RAPHAELE",
    "MME TELLIER", "ROSA", "RIVET", "CARPENTER", "GUARD", "MME TOURNEVAU",
    "MONSIEUR PHILIPPE", "FERNANDE", "MONSIEUR TOURENEVAU", "MONSIEUR VASSE",
    "MONSIEUR DUPUIS", "DENIS", "M MARAMBOT", "MARAMBOT", "OFFICER", "LAWYER",
    "GEORGES DUPORTIN", "PIERRE LETOILE", "GONTRAN", "ROGER DES ANNETTES",
    "MARQUIS DE LA TOUR-SAMUEL", "MOTHER CLOCHETTE", "COLLETTE", "CAILLAIRD",
    "CAILLAIRD'S WIFE", "BONDEL", "BONDEL'S WIFE", "TANCRET", "ABBEMARIGNAN'S NIECE",
    "PERE JOSEPH", "POSTMAN", "MAITRE RAMEAU", "SERGEANT", "LECACHEUR", "SERVANT GIRL",
    "LECACHEUR'S WIFE", "BRIGADIER SENATEUR", "MME LECACHEUR", "LENIENT", "SEVERIN",
    "LECHEUR", "LEUILLET", "MME LEUILLET", "MY UNCLE JULES", "HE", "SHE", "BRIGADIER",
    "RANDEL", "BUTCHER", "PROSECUTOR", "RENARD", "MADAME RENARD",
    "PRESIDENT OF THE COURT", "OLD BREDEL'S SON", "BEAURAIN", "MADAME BEAURAIN",
    "THE MAYOR", "BENOIST", "MARTINE", "MOTHER BEAURAIN", "BENOIST'S MOTHER", "VALLIN",
    "COMTE DE LORMERIN", "LISE", "RENEE", "LORMERIN", "BARNES", "PATIN", "DESIREE",
    "AUCTIONEER", "WOMAN", "PATIN'S WIDOW", "PARROT", "MAITRE ANTHIME", "PUBLIC CRIER",
    "CORPORAL OF GENDARMES", "MAITRE HAUCHECORNNE", "MAITRE HAUCHECORNE",
    "FARMER OF CRIQUETOT", "HORSE DEALER", "HAUCHECORNE", "PEOPLE",
    "TOINE BURNT-BRANDY", "CUSTOMER", "TOINE BURNT-BRANDY'S WIFE", "CELESTIN MAULOISEL",
    "PROSPER HORSLAVILLE", "HORSLAVILLE", "PASSER-BY", "RAOUL AUBERTIN", "MME HUSSON",
    "FRANCOISE", "BARBESOL", "ABBE MALON", "DAUPHINE", "LOUIS PHILIPPE", "GENERAL",
    "ISIDORE ROSIER", "COMMANDANT DESBARRES", "GRENADIERS", "M D'HUBIERES'S WIFE",
    "M D'HUBIERES", "MADAME D'HUBIERES", "FATHER TUVACHE", "AGED LADY", "JEAN VALLIN",
    "VALLINS' MOTHER", "VALLINS' FATHER", "CHARLOT", "PEASANT MOTHER",
    "VICOMTE GONTRAN-JOSEPH DE SIGNOLES", "ONE OF THE LADIES", "THE HUSBAND",
    "THE LADY", "MARQUIS", "MONGILET", "MADAME ROUBERE", "MADAME HENRIETTE LETORE",
    "HUSBAND", "NEWSPAPER", "PRIME MINISTER", "PATISSOT", "SALESPERSON", "BOIVIN",
    "PATISSOT'S COUSIN", "LITTLE MAN", "GARDENER", "PATISSOT'S COMPANION", "JOURNALIST",
    "NOVELIST", "TOUGH", "AN OLD GENTLEMAN", "A YOUNG MAN", "OCTAVIE",
    "STRAPPING FELLOW", "PERDRIX", "RADE", "SOMBRETERR", "SOMBRETERRE", "MAN",
    "PAUL MURET", "MME MURET D'ARTUS", "MY FRIEND", "OLD WOMAN", "M DE MEROUL",
    "MME DE MEROUL", "JOSEPH MOURADOUR", "MONSIEUR DE MEROUL", "SPEAKER 1", "SPEAKER 2",
    "OLD AMABLE", "CESAIRE", "CELESTE LEVESQUE", "CESAIRE HOULBREQUE",
    "OLD AMABLE HOULBREQUE", "ABBE RAFFIN", "AMABLE HOULBREQUE", "COUNTRYMAN",
    "ANOTHER COUNTRYMAN", "MALIVOIRE", "VICTOR LECOQ", "DADDY MALIVOIRE", "NEIGHBOR",
    "OLD MAN", "BARON RENE DU TREILLES", "MAITRE LEBRUMENT", "PEASANT DRIVER", "FARMER",
    "LA RAPET", "SICK-NURSE", "HONORE", "HONORE'S MAN", "RAPET", "MOTHER BONTEMPS",
    "OLD BARON DES RAVOTS", "JOSEPH", "EVERYONE", "RENE DE BOURNEVAL",
    "WALTER SCHNAFFS", "BIG SOLDIER", "ANOTHER OFFICER", "BIG OFFICER", "YOUNG OFFICER",
    "BLOUGNE-SUR-MER CORRESPONDENT", "SEAMAN", "JAVEL SENIOR", "JAVEL JUNIOR",
    "UNKNOWN", "LABARBE", "SAINT ANTHONY", "MAYOR CHICOT", "PRUSSIAN SOLDIER",
    "PIG-PORTRAYING-SOLDIER", "MME LEMOINE", "MME LEFEVRE", "BAKER", "CHAUK PIT WORKER",
    "QUARRYMAN", "BRIDEGROOM", "BRIDE", "COMPANION", "MATTHEW", "MELIE", "OREILLE",
    "MME OREILLE", "OREILLE'S WIFE", "CLERK", "RABOT", "MAITRE CANEVILLE",
    "MAITRE POIRET", "RABOT'S WIFE", "MAITRE BELOHOMME", "CANIVEAU", "BELHOMME",
    "SIDOINE", "TAILLE", "TOUCHARD", "ANNA", "OLD TOUCHARD", "SAUVETANIN", "COMPANY",
    "LEBRUMENT", "FATHER-IN-LAW", "CONDUCTOR", "TWO LADIES", "COOK", "INSPECTOR",
    "HENRY BARRAL", "M D'ARNELLES", "SIMON RADEVIN", "MADAME RADEVIN", "GRANDFATHER",
    "GONTRAN-JOSEPH DE SIGNOLES", "SUICIDE LETTER WRITER", "YOUNG WOMAN", "GUEST",
    "ANOTHER GUEST", "WRITER", "SIMON", "SOMEONE ELSE", "PHILIP REMY", "LA BLANCHOTTE",
    "LEMONNIER", "LEMONNIER'S WIFE", "M LEMONNIER", "MONSIEUR DUFOUR", "MME DUFOUR",
    "MME DUFOUR'S HUSBAND", "MME DUFOUR'S DAUGHTER", "ONE OF THE BOATING MEN",
    "YOUNG MAN", "SAME BOATING MAN", "SAME SPEAKER", "BOATING MAN", "HENRI", "HIMSELF",
    "MADAME DUFOUR", "HENRIETTE'S HUSBAND", "MARGOT", "SIMONE", "ROSALE",
    "OLD VARAMBOT AND HIS WIFE", "SANDRES", "MADAME SANDRES", "MME SANDRES", "M SAVAL",
    "LITTLE SERVANT GIRL", "MARGUERITE", "SUZANNE", "FATHER SIMON", "ZIDORE",
    "SISTER EULALIE", "THE JUDGE", "THE ENGLISHWOMAN", "MADEMOISELLE COCOTTE", "I",
    "FRANCOIS", "SELF", "COURBATAILLE", "UNCLE JOSEPH", "LARGE WOMAN", "CHILDREN",
    "MOTHER MAGLOIRE", "JULES CHICOT", "NOTARY", "BOITELLE", "SON", "FATHER", "ANTOINE",
    "NEGRESS", "A YOUNG WOMAN", "OLD MAIDEN AUNT", "ONE OF THE GENTLEMEN",
    "POIREL DE LA VOULTE", "MOTHER OF MONSTERS", "HOUSEHOLD", "M MILIAL", "MATHURIN",
    "JEREMIE", "OWNER", "PHYSICIAN FRIEND", "FLORENTIN", "EMMA", "FORTUNE-TELLER",
    "AUTHOR", "MADAME HERMET", "PATIENT", "MME HERMET", "GEORGE HERMET", "FOOTMAN",
    "SECRETARY", "MARINEL", "PROJECT GUTENBERG", "THE FOUNDATION",
]

def _collision_families(names):
    families = {}
    for name in names:
        families.setdefault(roster_key(name), []).append(name)
    return sorted(sorted(v) for v in families.values() if len(v) > 1)


def test_real_roster_fixture_is_intact():
    # The fixture is real corpus data used as *input*; the only property that
    # matters here is that it hasn't been accidentally truncated or
    # duplicated by an edit, not its exact size (which is one corpus's, and
    # this project is upstream-bound).
    check(len(REAL_PRODUCTION_ROSTER) > 0, "fixture is non-empty")
    check_equal(len(set(REAL_PRODUCTION_ROSTER)), len(REAL_PRODUCTION_ROSTER),
                "fixture labels are distinct")


def _whitespace_only_key(name):
    """The pre-widening roster key: canonical form with whitespace collapsed,
    but hyphens/apostrophes left in place (unlike the current roster_key())."""
    return "".join(canonicalize(name).split(" "))


def test_real_roster_key_introduces_no_new_collisions():
    # roster_key() was widened from stripping whitespace only to stripping
    # all boundary marks (hyphens, apostrophes too). That widening must not
    # merge any two real characters who weren't already merged under the
    # narrower, whitespace-only key -- i.e. every collision family under the
    # current key must also collide under the whitespace-only key. Corpus-
    # independent: makes no assumption about which or how many families
    # exist, only that widening didn't introduce NEW ones.
    full_families = _collision_families(REAL_PRODUCTION_ROSTER)
    check(len(full_families) > 0, "real roster produces at least one collision family")

    whitespace_families = {}
    for name in REAL_PRODUCTION_ROSTER:
        whitespace_families.setdefault(_whitespace_only_key(name), []).append(name)
    whitespace_family_sets = [frozenset(v) for v in whitespace_families.values() if len(v) > 1]

    for family in full_families:
        family_set = frozenset(family)
        check(any(family_set <= ws_family for ws_family in whitespace_family_sets),
              f"family {family} collides under the full key but is new under boundary-mark widening")

    # Readable example from the corpus: a boundary-mark drift pair collides
    # under both keyings.
    check_equal(roster_key("ABBE MARIGNAN"), roster_key("ABBEMARIGNAN"),
                "ABBE MARIGNAN / ABBEMARIGNAN collide under the full key")
    check_equal(_whitespace_only_key("ABBE MARIGNAN"), _whitespace_only_key("ABBEMARIGNAN"),
                "and also under the whitespace-only key")


def _bare_name(name):
    """The canonical name with its gender-marking prefix (if any) removed."""
    tokens = canonicalize(name).split(" ")
    if len(tokens) > 1 and tokens[0] in GENDERED_TITLES:
        return " ".join(tokens[1:])
    return " ".join(tokens)


def test_real_roster_merges_never_mix_two_surnames():
    # Safety property for the title tables: every merge family must be one
    # person under two spellings, i.e. the labels in it must agree on the
    # bare name once the title is set aside. A title table that swallowed a
    # name constituent (the "ST JOHN RIVERS" -> "JOHN RIVERS" failure mode)
    # would show up as a family mixing two distinct surnames.
    families = _collision_families(REAL_PRODUCTION_ROSTER)
    check(len(families) > 0, "real roster produces at least one collision family")
    for family in families:
        bare = {roster_key(_bare_name(name)) for name in family}
        check(len(bare) == 1,
              f"collision family {family} mixes distinct bare names {sorted(bare)}")

    # Readable example: a boundary-mark drift pair and a gendered-title pair
    # both collide, for a real character from the corpus.
    check_equal(roster_key("ABBE MARIGNAN"), roster_key("ABBEMARIGNAN"),
                "boundary-mark drift unifies onto the same roster key")
    check_equal(canonicalize("Mme Tellier"), canonicalize("Madame Tellier"),
                "abbreviated and spelled-out gendered titles unify onto the same key")


def test_real_roster_keeps_cross_gender_pairs_distinct():
    # The point of preserving the titles: MONSIEUR X and MADAME X are two
    # people and must never share a roster entry. Derived programmatically
    # (not a hardcoded surname list) from whatever cross-gender pairs the
    # fixture happens to contain.
    pairs = {}
    for name in REAL_PRODUCTION_ROSTER:
        tokens = canonicalize(name).split(" ")
        if len(tokens) > 1 and tokens[0] in GENDERED_TITLES:
            pairs.setdefault(" ".join(tokens[1:]), set()).add(tokens[0])
    cross_gender = sorted(k for k, v in pairs.items() if len(v) > 1)
    check(len(cross_gender) > 0,
          "the real roster contains at least one cross-gender surname pair")
    index = {}
    for name in REAL_PRODUCTION_ROSTER:
        remember_in_roster(index, name)
    for surname in cross_gender:
        for title in pairs[surname]:
            check(resolve_against_roster(f"{title} {surname}", index) == f"{title} {surname}",
                  f"{title} {surname} keeps its own roster entry")

    # Readable example: Madame Dufour and Monsieur Dufour never merge.
    check(canonicalize("Mme Dufour") != canonicalize("Monsieur Dufour"),
          "Mme Dufour and Monsieur Dufour must stay distinct")


def test_real_roster_is_idempotent_end_to_end():
    for name in REAL_PRODUCTION_ROSTER:
        once = canonicalize(name)
        check_equal(canonicalize(once), once, f"idempotency for {name!r}")


def test_real_roster_resolves_to_fewer_names_by_exactly_the_merged_collisions():
    index = {}
    for name in REAL_PRODUCTION_ROSTER:
        remember_in_roster(index, name)
    families = _collision_families(REAL_PRODUCTION_ROSTER)
    merged_away = sum(len(family) - 1 for family in families)
    check_equal(len(index), len(REAL_PRODUCTION_ROSTER) - merged_away,
                "the index shrinks by exactly one entry per extra spelling "
                "in each collision family")

    # Readable examples of the resolution in action.
    check_equal(resolve_against_roster("ABBEMARIGNAN", index), "ABBE MARIGNAN",
                "the drifted spelling resolves onto the good one")
    check_equal(resolve_against_roster("MME TELLIER", index), "MADAME TELLIER",
                "the abbreviated French title resolves onto the spelled-out one")
    check_equal(resolve_against_roster("M DE MEROUL", index), "MONSIEUR DE MEROUL",
                "and so does the abbreviated MONSIEUR")


# ---------------------------------------------------------------------------
# Tier 3: attest_label() / _core_tokens()
# ---------------------------------------------------------------------------

def test_core_tokens_filters_stopwords_titles_possessives_accents():
    check_equal(_core_tokens("MISTER SMITH"), ["SMITH"], "title filtered out")
    check_equal(_core_tokens("THE DE LA CRUZ"), ["CRUZ"],
                "stopwords THE/DE/LA filtered, surname kept")
    check_equal(_core_tokens("DAIRINE'S"), ["DAIRINE"], "possessive stripped")
    check_equal(_core_tokens("DAIRINE"), ["DAIRINE"], "plain token unaffected")
    check_equal(_core_tokens("MISTER"), [], "pure title label has zero core tokens")
    check_equal(_core_tokens(""), [], "empty label has zero core tokens")


def test_attest_label_fully_attested_not_flagged():
    windows = ["Alice walked into the room.", "Nobody else was around."]
    result = attest_label("ALICE", windows)
    check(result["attested"] is True, "ALICE attested when 'Alice' appears in a window")
    check_equal(result["missing_tokens"], [], "no missing tokens when attested")


def test_attest_label_missing_one_token_is_flagged():
    windows = ["Alice walked into the room, alone."]
    result = attest_label("ALICE SMITH", windows)
    check(result["attested"] is False, "ALICE SMITH not attested when SMITH never appears")
    check_equal(result["missing_tokens"], ["SMITH"], "SMITH reported missing")


def test_attest_label_tokens_elsewhere_but_not_in_own_window_is_flagged():
    # Synthetic "transposed name" case: TRANSFORMED and PIG both appear
    # SOMEWHERE in the synthetic source, but never together, and never in
    # this label's own windows -- attestation must still fail, because
    # attest_label only looks at the windows it's given, not the whole book.
    label = "TRANSFORMED PIG"
    own_windows = [
        "A stranger entered the tavern and ordered a drink quietly.",
        "The stranger paid and left without another word.",
    ]
    # These sentences exist elsewhere in the (hypothetical) source, but are
    # never passed in as this label's windows, simulating "elsewhere in the
    # book, not near this label's lines".
    _elsewhere_in_book = "The wizard had transformed into a pig hours earlier."
    result = attest_label(label, own_windows)
    check(result["attested"] is False,
          "TRANSFORMED PIG not attested from windows that never mention either token")
    check_equal(sorted(result["missing_tokens"]), ["PIG", "TRANSFORMED"],
                "both core tokens reported missing")


def test_attest_label_is_pure_and_deterministic():
    windows = ["Bob said hello.", "Bob left quickly."]
    windows_copy = list(windows)
    result1 = attest_label("BOB", windows)
    result2 = attest_label("BOB", windows)
    check_equal(result1, result2, "same inputs produce the same output twice")
    check_equal(windows, windows_copy, "windows list is left untouched by the call")

    # Trivial/unknown case: a label with zero core tokens is conservatively
    # flagged unattested rather than vacuously attested.
    trivial = attest_label("MISTER", ["Mister anything goes here."])
    check(trivial["attested"] is False, "zero-core-token label is conservatively unattested")
    check_equal(trivial["missing_tokens"], [], "no tokens to report missing in the trivial case")


def test_attest_label_matches_curly_apostrophe_across_the_glyph_gap():
    # Label uses a straight apostrophe (typical LLM output); the source uses
    # a curly one (typical of the author's own prose). The glyph difference
    # carries no identifying information and must not cause a false flag.
    windows = ["O'Malley walked in.", "‘No,’ said O’Malley."]
    result = attest_label("O'MALLEY", windows)
    check(result["attested"] is True, "curly source apostrophe matches straight label apostrophe")
    check_equal(result["missing_tokens"], [], "no missing tokens once apostrophe glyph is normalized")


def test_attest_label_distinct_apostrophe_names_stay_distinct():
    # Normalizing the apostrophe GLYPH must not blur two different names
    # that both happen to contain an apostrophe.
    result = attest_label("O'MALLEY", ["O'Brien walked in."])
    check(result["attested"] is False, "O'MALLEY is not attested by an O'BRIEN window")
    check_equal(result["missing_tokens"], ["O'MALLEY"], "distinct apostrophe-bearing name reported missing")


def test_attest_label_matches_hyphenated_token_verbatim():
    windows = ["The sun-walker entered the hall.", "Sun-walker spoke softly."]
    result = attest_label("SUN-WALKER", windows)
    check(result["attested"] is True, "hyphenated label token matches hyphenated source occurrence")
    check_equal(result["missing_tokens"], [], "no missing tokens for verbatim hyphenated match")


def test_attest_label_matches_hyphenated_token_against_space_variant():
    # Same name, but the source spells it with a space instead of a hyphen.
    windows = ["The sun walker entered the hall."]
    result = attest_label("SUN-WALKER", windows)
    check(result["attested"] is True,
          "hyphenated label token matches a space-separated source variant")
    check_equal(result["missing_tokens"], [], "no missing tokens for the space-variant match")


# ---------------------------------------------------------------------------
# Unicode coverage of the character classes, and attest_speaker()'s three-way
# verdict.
#
# These assert PROPERTIES over programmatically chosen sample scripts, not the
# contents of any book: "a name in a non-Latin script survives
# canonicalization", "two distinct names never share a roster key", "a label
# our check cannot confirm is never reported as refuted". The sample strings
# are language specimens (a Cyrillic name, a Hebrew name, ...), chosen to
# cover cased/uncased and segmented/unsegmented writing systems; no corpus is
# pinned and no test depends on any particular novel.
# ---------------------------------------------------------------------------

# One specimen per writing-system property that the character classes have to
# survive. Cased-Latin is covered by the whole suite above; these are the ones
# an ASCII class silently destroyed.
_NON_LATIN_SPECIMENS = [
    ("Cyrillic", "Ирина"),
    ("Greek", "Μένων"),
    ("Hebrew", "שרה"),
    ("Arabic", "محمد"),
    ("Han", "林"),
    ("Hiragana", "さくら"),
    ("Devanagari", "सीता"),
]


def test_canonicalize_preserves_non_latin_names():
    # An ASCII _ALLOWED_CHARS_RE stripped every one of these to "", and
    # resolve_span_labels only accepts a speaker when canonicalize() returns
    # something truthy -- so this property is the difference between a
    # non-Latin book having character voices and having none at all.
    for script, name in _NON_LATIN_SPECIMENS:
        canonical = canonicalize(name)
        check(canonical != "", f"{script} name survives canonicalization (not emptied)")
        check(len(canonical) == len(name),
              f"{script} name keeps all of its characters")


def test_canonicalize_is_idempotent_for_non_latin_names():
    # Frozen contract 6 applies to every script, not just Latin.
    for script, name in _NON_LATIN_SPECIMENS:
        once = canonicalize(name)
        check_equal(canonicalize(once), once, f"{script} canonicalization is idempotent")


def test_non_latin_names_do_not_collide_on_one_roster_key():
    # THE regression this pins. While _KEY_STRIP_RE was [^A-Z0-9] it removed
    # every non-Latin letter, so once canonicalize() preserved them all these
    # names would have keyed to "" -- one roster entry, one voice, for an
    # entire cast, irreversibly (aliasing can merge but never split).
    keys = {}
    for script, name in _NON_LATIN_SPECIMENS:
        key = roster_key(name)
        check(key != "", f"{script} name has a non-empty roster key")
        check(key not in keys,
              f"{script} name does not share a roster key with {keys.get(key)}")
        keys[key] = script

    index = {}
    for _, name in _NON_LATIN_SPECIMENS:
        remember_in_roster(index, name)
    check_equal(len(index), len(_NON_LATIN_SPECIMENS),
                "every distinct non-Latin name gets its own roster entry")


def test_non_latin_boundary_mark_unification_still_works():
    # The Tier 1b contract is unchanged for non-Latin scripts: spellings that
    # differ ONLY in boundary marks unify, and the more-punctuated form wins.
    index = {}
    remember_in_roster(index, "АННАМАРИЯ")
    winner = remember_in_roster(index, "АННА МАРИЯ")
    check_equal(winner, "АННА МАРИЯ",
                "more-punctuated Cyrillic spelling wins the roster slot")
    check_equal(len(index), 1, "the two Cyrillic spellings unified into one entry")


def test_non_latin_distinct_names_are_never_unified():
    # The no-fuzzy-merge contract, in a non-Latin script: one letter apart is
    # still two different people.
    index = {}
    remember_in_roster(index, "ИРИНА")
    remember_in_roster(index, "ИРИНЫ")
    check_equal(len(index), 2, "Cyrillic names differing by a letter stay distinct")


def test_core_tokens_extracted_from_non_latin_labels():
    # _WORD_RE as [A-Za-z0-9'‘’] extracted ZERO tokens from these, and a
    # zero-core-token label is reported unattested -- so the Voices UI badge
    # was wrong for every speaker in every non-Latin book.
    for script, name in _NON_LATIN_SPECIMENS:
        tokens = _core_tokens(canonicalize(name))
        check(len(tokens) > 0, f"{script} label yields at least one core token")


def test_attest_speaker_attested_when_name_is_present():
    for script, name in _NON_LATIN_SPECIMENS:
        window = f"{name} — {name}."
        verdict = attest_speaker(canonicalize(name), [window])
        check(verdict in (ATTESTED, UNVERIFIABLE),
              f"{script} name present in its window is not reported refuted "
              f"(got {verdict!r})")


def test_attest_speaker_unattested_only_on_positive_evidence():
    # A name that appears nowhere in its windows, in any form, is the ONLY
    # case that may be reported UNATTESTED -- this is the verdict a gate is
    # allowed to reject on.
    verdict = attest_speaker("ZZQXAL", ["Alice spoke to Bob about the weather."])
    check_equal(verdict, UNATTESTED, "a name absent in every form is UNATTESTED")

    verdict = attest_speaker("ALICE", ["Alice spoke to Bob about the weather."])
    check_equal(verdict, ATTESTED, "a whole-word present name is ATTESTED")


def test_attest_speaker_zero_core_tokens_is_unverifiable_never_unattested():
    # attest_label reports this as attested=False (a conservative boolean).
    # A gate must NOT treat it as refuted, or a label made only of titles is
    # destroyed on no evidence.
    check_equal(attest_speaker("MISTER", ["Mister anything goes here."]),
                UNVERIFIABLE, "title-only label is UNVERIFIABLE, not UNATTESTED")
    check_equal(attest_label("MISTER", ["Mister anything."])["attested"], False,
                "attest_label's existing boolean contract is unchanged")


def test_attest_speaker_substring_only_match_is_unverifiable():
    # Unsegmented scripts (no spaces between words) put a correct name inside
    # a longer run, where whole-word matching can never confirm it. Same shape
    # as a too-short Latin token inside a longer word. Both are "cannot tell",
    # and must not be rejectable.
    check_equal(attest_speaker("林", ["林考言道，很好。"]), UNVERIFIABLE,
                "name inside an unsegmented run is UNVERIFIABLE, not UNATTESTED")
    check_equal(attest_speaker("AL", ["Alice waited."]), UNVERIFIABLE,
                "short token inside a longer word is UNVERIFIABLE, not UNATTESTED")


def test_attest_speaker_is_pure_and_idempotent():
    label = "ALICE SMITH"
    windows = ["Alice walked in.", "Smith followed."]
    windows_before = copy.deepcopy(windows)

    first = attest_speaker(label, windows)
    second = attest_speaker(label, windows)
    check_equal(first, second, "attest_speaker is idempotent on equal inputs")
    check_equal(windows, windows_before, "attest_speaker does not mutate its windows")
    check_equal(label, "ALICE SMITH", "attest_speaker does not mutate its label")


def test_attest_speaker_handles_empty_and_missing_windows():
    # A label with core tokens and no windows at all is refuted-by-absence,
    # which is correct: nothing confirms it. Callers that cannot build windows
    # must decide not to gate, rather than passing [] and rejecting the world.
    check_equal(attest_speaker("ALICE", []), UNATTESTED,
                "no windows means no confirmation")
    check_equal(attest_speaker("", ["anything"]), UNVERIFIABLE,
                "an empty label is UNVERIFIABLE")


def test_attest_speaker_verdicts_are_the_exported_constants():
    # Callers compare against the constants; pin their values so a rename
    # cannot silently change the wire/report vocabulary.
    check_equal(ATTESTED, "attested", "ATTESTED constant value")
    check_equal(UNATTESTED, "unattested", "UNATTESTED constant value")
    check_equal(UNVERIFIABLE, "unverifiable", "UNVERIFIABLE constant value")


# ---------------------------------------------------------------------------
# Roster-name partial attestation (attest_speaker's roster_index argument)
# ---------------------------------------------------------------------------

def _roster(*names):
    """Build a roster index the way generate_script does."""
    index = {}
    for name in names:
        remember_in_roster(index, name)
    return index


def test_partial_attestation_accepts_a_label_built_on_a_roster_name():
    # "<name>'S FATHER" describes a real person the text names obliquely. The
    # book has established the name, so the label is not positively WRONG --
    # it is unconfirmable, which is what UNVERIFIABLE means.
    windows = ["Brannoc ran ahead down the shingle, shouting."]
    label = canonicalize("Brannoc's father")
    check_equal(attest_speaker(label, windows), UNATTESTED,
                "without a roster the label is still refuted")
    check_equal(attest_speaker(label, windows, roster_index=_roster("BRANNOC")),
                UNVERIFIABLE,
                "an attested ROSTER token softens the verdict to unverifiable")


def test_partial_attestation_requires_the_attested_token_to_be_a_roster_name():
    # The rejected weaker rule ("any attested token") would be satisfied by any
    # common noun copied out of the prose. This is what keeps it narrow.
    windows = ["A heron lifted off the water."]
    check_equal(
        attest_speaker("TRANSFORMED HERON", windows, roster_index=_roster("BRANNOC")),
        UNATTESTED,
        "a common noun from the prose is not a roster name")


def test_partial_attestation_needs_two_or_more_core_tokens():
    windows = ["The lamp guttered and went out."]
    check_equal(attest_speaker("BRANNOC", windows, roster_index=_roster("BRANNOC OF ESK")),
                UNATTESTED,
                "a single-token label is not partially attested")


def test_partial_attestation_still_refutes_when_every_token_is_missing():
    windows = ["The lamp guttered and went out."]
    check_equal(
        attest_speaker("BRANNOC ESKELLAN", windows, roster_index=_roster("BRANNOC")),
        UNATTESTED,
        "no attested token at all is still positive evidence")


def test_attest_label_boolean_contract_is_unchanged_by_the_roster_rule():
    # attest_label takes no roster and must keep its boolean meaning: the UI
    # badge and the label_flags endpoint depend on it.
    windows = ["Brannoc ran ahead down the shingle, shouting."]
    result = attest_label(canonicalize("Brannoc's father"), windows)
    check(result["attested"] is False, "attest_label stays False for a missing token")
    check_equal(result["missing_tokens"], ["FATHER"], "attest_label still names the gap")


# ---------------------------------------------------------------------------
# repair_speaker(): the five adversarial cases FIRST
#
# These are the acceptance criteria for the repair design, not the clean
# repairs it was built for. Every name and every sentence here is invented.
# ---------------------------------------------------------------------------

def _book(*sentences):
    return source_word_index(" ".join(sentences))


def test_repair_refuses_a_real_character_named_elsewhere_in_the_book():
    # ADVERSARIAL 1 (ANDRE -> ANDREA). Veldor is a real character whose own
    # scene attributes his lines with pronouns only, and Velder happens to be
    # standing there. The whole-book guard is what stops the merge: the author
    # spells "Veldor" somewhere, so it is not a refuted spelling.
    windows = ['Velder set down the tray. "Then say it plainly," he answered.']
    book = _book("Veldor had come down from the pass a week earlier.",
                 "Velder set down the tray.")
    roster = _roster("VELDER")
    check_equal(attest_speaker("VELDOR", windows, roster_index=roster), UNATTESTED,
                "the gate still refutes the label")
    check_equal(repair_speaker("VELDOR", windows, roster, book), None,
                "a spelling the book uses is never repaired")


def test_repair_of_a_name_the_book_never_contains_is_an_accepted_limitation():
    # The honest other half of ADVERSARIAL 1, pinned so nobody claims the
    # design is airtight: if the character's name appears NOWHERE in the book,
    # repair does fire and is wrong. Documented in repair_speaker's docstring.
    windows = ['Velder set down the tray. "Then say it plainly," he answered.']
    book = _book("Velder set down the tray.")
    check_equal(repair_speaker("VELDOR", windows, _roster("VELDER"), book),
                "VELDER",
                "known limitation: a name absent from the whole book is repaired")


def test_repair_refuses_a_target_the_source_never_attested():
    # ADVERSARIAL 2 (MARIA -> MARIE with a polluted roster). "MIRVE" reached
    # the roster from a substring-only (UNVERIFIABLE) acceptance -- the book
    # contains "Mirvenholm", never the bare word. Amendment (b): a repair must
    # not target a roster name that was itself never source-attested, or the
    # pollution rewrites real names onto itself.
    windows = ["The road out of Mirvenholm was flooded to the axles."]
    book = _book("The road out of Mirvenholm was flooded to the axles.",
                 "Mirva waited at the ford.")
    roster = _roster("MIRVE")
    check_equal(repair_speaker("MIRVA", windows, roster, book), None,
                "a roster name the book never spells as a word is not a target")


def test_repair_refuses_across_a_diacritic_fold():
    # ADVERSARIAL 3 (ANDRÉ -> ANDREA). Accents fold on BOTH sides, so the
    # accented spelling is found in the book's own word index and the two real
    # people are left apart.
    windows = ['Theona turned the key. "It was never locked," she said.']
    book = _book("Thíona had carried the lamp up herself.",
                 "Theona turned the key.")
    check_equal(repair_speaker("THIONA", windows, _roster("THEONA"), book), None,
                "an accented real name is protected by the folded word index")


def test_repair_refuses_a_transliteration_variant_the_book_uses():
    # ADVERSARIAL 4 (YUSUF -> YUSUP). Both spellings occur in the book, so
    # neither is refuted.
    windows = ['Yusap looked up. "Not today," he said.']
    book = _book("Yusaf of the northern house signed the charter.",
                 "Yusap looked up.")
    check_equal(repair_speaker("YUSAF", windows, _roster("YUSAP"), book), None,
                "a variant the book itself uses is never repaired")


def test_repair_refuses_a_plural_singular_pair_the_book_uses():
    # ADVERSARIAL 5 (TWIN -> TWINS). Any book that uses the singular word
    # anywhere is protected; a book that never does is the documented residual.
    # (A pair like this is doubly protected in practice: the singular is also a
    # SUBSTRING of the plural, so the gate returns UNVERIFIABLE and never
    # reaches repair at all. This asserts the repair guard directly.)
    windows = ["The twins had not spoken since the crossing."]
    book = _book("The twins had not spoken since the crossing.",
                 "Each twin carried half the water.")
    check_equal(repair_speaker("TWIN", windows, _roster("TWINS"), book), None,
                "a singular the book uses as a word is never repaired")


# ---------------------------------------------------------------------------
# repair_speaker(): the guards
# ---------------------------------------------------------------------------

WINDOW_BRANNOC = ['Brannoc set down the lamp. "It is done," he said.']
BOOK_BRANNOC = _book('Brannoc set down the lamp. "It is done," he said.',
                     "Vella waited by the door.")


def test_repair_folds_a_unique_distance_one_misspelling():
    check_equal(repair_speaker("BRANOC", WINDOW_BRANNOC, _roster("BRANNOC"),
                               BOOK_BRANNOC),
                "BRANNOC",
                "one refuted token, one established candidate one edit away")


def test_repair_keeps_attested_tokens_verbatim():
    windows = ["Brannoc of Esk set down the lamp."]
    book = _book("Brannoc of Esk set down the lamp.")
    check_equal(repair_speaker("BRANOC OF ESK", windows,
                               _roster("BRANNOC", "ESK"), book),
                "BRANNOC OF ESK",
                "only the refuted token is substituted; the rest is untouched")


def test_repair_refuses_when_two_candidates_are_one_edit_away():
    windows = ["Brannoc and Brannic argued by the fire."]
    book = _book("Brannoc and Brannic argued by the fire.")
    check_equal(repair_speaker("BRANNC", windows, _roster("BRANNOC", "BRANNIC"),
                               book),
                None,
                "ambiguity refuses -- the evidence does not pick a winner")


def test_repair_refuses_when_the_ambiguity_is_only_visible_in_the_roster():
    # The ambiguity guard is roster-WIDE, not window-local: a second candidate
    # that is not in this window still blocks the repair. This is one of the
    # two guards that replaced the proposed minimum-token-length floor.
    windows = ["Brannoc argued by the fire."]
    book = _book("Brannoc argued by the fire.", "Brannic kept the ledger.")
    check_equal(repair_speaker("BRANNC", windows, _roster("BRANNOC", "BRANNIC"),
                               book),
                None,
                "a distance-1 roster name outside the window still blocks")


def test_repair_refuses_a_candidate_absent_from_the_window():
    windows = ["The lamp guttered and went out."]
    check_equal(repair_speaker("BRANOC", windows, _roster("BRANNOC"),
                               BOOK_BRANNOC),
                None,
                "a candidate must be attested in the label's own window")


def test_repair_refuses_a_window_word_that_is_not_a_roster_name():
    check_equal(repair_speaker("BRANOC", WINDOW_BRANNOC, _roster("VELLA"),
                               BOOK_BRANNOC),
                None,
                "candidates come from the roster, not from the prose at large")


def test_repair_refuses_a_label_that_is_already_established():
    roster = _roster("BRANOC", "BRANNOC")
    check_equal(repair_speaker("BRANOC", WINDOW_BRANNOC, roster, BOOK_BRANNOC),
                None,
                "an established name is never repaired")


def test_repair_refuses_at_distance_two():
    check_equal(repair_speaker("BRENNIC", WINDOW_BRANNOC, _roster("BRANNOC"),
                               BOOK_BRANNOC),
                None,
                "two edits is not a misspelling we can claim to have identified")


def test_repair_refuses_without_a_roster_or_a_book_index():
    check_equal(repair_speaker("BRANOC", WINDOW_BRANNOC, {}, BOOK_BRANNOC), None,
                "an empty roster offers no candidates")
    check_equal(repair_speaker("BRANOC", WINDOW_BRANNOC, _roster("BRANNOC"), set()),
                None,
                "no book index means no evidence, so no repair")
    check_equal(repair_speaker("BRANOC", [], _roster("BRANNOC"), BOOK_BRANNOC),
                None, "no window means no repair")
    check_equal(repair_speaker("NARRATOR", WINDOW_BRANNOC, _roster("BRANNOC"),
                               BOOK_BRANNOC),
                None, "the narrator sentinel is never repaired")


def test_repair_is_pure_idempotent_and_mutates_nothing():
    roster = _roster("BRANNOC")
    roster_before = copy.deepcopy(roster)
    windows = copy.deepcopy(WINDOW_BRANNOC)
    windows_before = copy.deepcopy(windows)
    book = set(BOOK_BRANNOC)
    book_before = set(book)

    first = repair_speaker("BRANOC", windows, roster, book)
    second = repair_speaker("BRANOC", windows, roster, book)
    check_equal(first, second, "repair_speaker is deterministic")
    check_equal(roster, roster_before, "repair_speaker does not mutate the roster")
    check_equal(windows, windows_before, "repair_speaker does not mutate its windows")
    check_equal(book, book_before, "repair_speaker does not mutate the word index")
    # Idempotent in the sense that matters: a repaired name attests, so there
    # is nothing left to repair and the second pass declines.
    check_equal(repair_speaker(first, windows, roster, book), None,
                "repairing a repaired name is a no-op")


def test_repair_is_unreachable_for_an_unsegmented_script_label():
    # A name present as a substring but not at a word boundary -- what a
    # Chinese/Japanese/Thai book looks like to this tokenizer -- is
    # UNVERIFIABLE, so the gate accepts it and repair is never consulted.
    windows = ["リナは黙っていた。"]
    check_equal(attest_speaker("リナ", windows), UNVERIFIABLE,
                "an unsegmented-script label stays unverifiable")


def test_distance_one_predicate_is_exact_and_bounded():
    check(_is_distance_one("BRANOC", "BRANNOC"), "one insertion")
    check(_is_distance_one("BRANNOC", "BRANOC"), "one deletion")
    check(_is_distance_one("BRANNOC", "BRANNIC"), "one substitution")
    check(not _is_distance_one("BRANNOC", "BRANNOC"), "zero edits is not distance 1")
    check(not _is_distance_one("BRANNOC", "BRENNIC"), "two substitutions")
    check(not _is_distance_one("BRANNOC", "BRAN"), "a length gap of 3")
    check(_is_distance_one("ИРИНА", "ИРИН"),
          "the predicate is code-point based, not ASCII")


def main():
    tests = [
        test_basic_case_and_whitespace_normalization,
        test_parenthetical_removal,
        test_rank_titles_are_dropped,
        test_gendered_titles_are_preserved_and_normalized,
        test_gendered_titles_are_language_preserving,
        test_gendered_titles_keep_husband_and_wife_apart,
        test_fr_is_preserved_not_dropped,
        test_title_alone_never_empty,
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
        test_stacked_titles_resolve_to_a_fixpoint,
        test_double_canonicalization_matches_single,
        test_saint_names_are_not_honorifics,
        test_roster_key_strips_whitespace_only,
        test_most_boundary_marks_wins_in_both_arrival_orders,
        test_punctuation_variants_unify,
        test_apostrophe_variant_wins_in_both_arrival_orders,
        test_hyphen_vs_space_is_a_deterministic_tie,
        test_boundary_marks_beat_a_bare_run_of_letters,
        test_remember_returns_the_winner_and_promotes_in_place,
        test_ties_keep_the_incumbent,
        test_resolve_snaps_onto_established_spelling,
        test_resolve_never_merges_similar_names,
        test_resolve_passthrough_and_narrator,
        test_resolve_is_idempotent,
        test_resolve_does_not_mutate_the_index,
        test_empty_name_is_not_recorded,
        test_real_roster_fixture_is_intact,
        test_real_roster_key_introduces_no_new_collisions,
        test_real_roster_merges_never_mix_two_surnames,
        test_real_roster_keeps_cross_gender_pairs_distinct,
        test_real_roster_is_idempotent_end_to_end,
        test_real_roster_resolves_to_fewer_names_by_exactly_the_merged_collisions,
        test_suggest_aliases_expected_pairs,
        test_suggest_aliases_excludes_narrator,
        test_suggest_aliases_empty_and_single,
        test_suggest_aliases_direction_rule,
        test_core_tokens_filters_stopwords_titles_possessives_accents,
        test_attest_label_fully_attested_not_flagged,
        test_attest_label_missing_one_token_is_flagged,
        test_attest_label_tokens_elsewhere_but_not_in_own_window_is_flagged,
        test_attest_label_is_pure_and_deterministic,
        test_attest_label_matches_curly_apostrophe_across_the_glyph_gap,
        test_attest_label_distinct_apostrophe_names_stay_distinct,
        test_attest_label_matches_hyphenated_token_verbatim,
        test_attest_label_matches_hyphenated_token_against_space_variant,
        test_canonicalize_preserves_non_latin_names,
        test_canonicalize_is_idempotent_for_non_latin_names,
        test_non_latin_names_do_not_collide_on_one_roster_key,
        test_non_latin_boundary_mark_unification_still_works,
        test_non_latin_distinct_names_are_never_unified,
        test_core_tokens_extracted_from_non_latin_labels,
        test_attest_speaker_attested_when_name_is_present,
        test_attest_speaker_unattested_only_on_positive_evidence,
        test_attest_speaker_zero_core_tokens_is_unverifiable_never_unattested,
        test_attest_speaker_substring_only_match_is_unverifiable,
        test_attest_speaker_is_pure_and_idempotent,
        test_attest_speaker_handles_empty_and_missing_windows,
        test_attest_speaker_verdicts_are_the_exported_constants,
        test_partial_attestation_accepts_a_label_built_on_a_roster_name,
        test_partial_attestation_requires_the_attested_token_to_be_a_roster_name,
        test_partial_attestation_needs_two_or_more_core_tokens,
        test_partial_attestation_still_refutes_when_every_token_is_missing,
        test_attest_label_boolean_contract_is_unchanged_by_the_roster_rule,
        test_repair_refuses_a_real_character_named_elsewhere_in_the_book,
        test_repair_of_a_name_the_book_never_contains_is_an_accepted_limitation,
        test_repair_refuses_a_target_the_source_never_attested,
        test_repair_refuses_across_a_diacritic_fold,
        test_repair_refuses_a_transliteration_variant_the_book_uses,
        test_repair_refuses_a_plural_singular_pair_the_book_uses,
        test_repair_folds_a_unique_distance_one_misspelling,
        test_repair_keeps_attested_tokens_verbatim,
        test_repair_refuses_when_two_candidates_are_one_edit_away,
        test_repair_refuses_when_the_ambiguity_is_only_visible_in_the_roster,
        test_repair_refuses_a_candidate_absent_from_the_window,
        test_repair_refuses_a_window_word_that_is_not_a_roster_name,
        test_repair_refuses_a_label_that_is_already_established,
        test_repair_refuses_at_distance_two,
        test_repair_refuses_without_a_roster_or_a_book_index,
        test_repair_is_pure_idempotent_and_mutates_nothing,
        test_repair_is_unreachable_for_an_unsegmented_script_label,
        test_distance_one_predicate_is_exact_and_bounded,
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
