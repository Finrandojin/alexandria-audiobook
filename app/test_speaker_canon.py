"""Standalone unit tests for speaker_canon.py.

Run directly (no pytest required):
    python app/test_speaker_canon.py

Exits 0 on all-pass, nonzero if any assertion fails.
"""

import sys
import copy

from speaker_canon import (
    canonicalize,
    remember_in_roster,
    resolve_against_roster,
    roster_key,
    suggest_aliases,
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
    check_equal(resolve_against_roster("Mr. Mark", roster), "MARK",
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

EXPECTED_COLLISION_FAMILIES = [
    ["ABBE MARIGNAN", "ABBEMARIGNAN"],
    ["ABBE MARIGNAN'S NIECE", "ABBEMARIGNAN'S NIECE"],
]


def _collision_families(names):
    families = {}
    for name in names:
        families.setdefault(roster_key(name), []).append(name)
    return sorted(sorted(v) for v in families.values() if len(v) > 1)


def test_real_roster_fixture_is_intact():
    check_equal(len(REAL_PRODUCTION_ROSTER), 578, "fixture holds all 578 real labels")
    check_equal(len(set(REAL_PRODUCTION_ROSTER)), 578, "fixture labels are distinct")


def test_real_roster_key_introduces_no_new_collisions():
    # Exactly the two known ABBE MARIGNAN families -- no others. A future
    # widening of roster_key() that merges two real characters fails here.
    check_equal(_collision_families(REAL_PRODUCTION_ROSTER),
                EXPECTED_COLLISION_FAMILIES,
                "real-roster collision families")


def test_real_roster_resolves_to_two_fewer_names():
    index = {}
    for name in REAL_PRODUCTION_ROSTER:
        remember_in_roster(index, name)
    check_equal(len(index), 576, "578 labels consolidate to 576 characters")
    check_equal(resolve_against_roster("ABBEMARIGNAN", index), "ABBE MARIGNAN",
                "the drifted spelling resolves onto the good one")
    check_equal(resolve_against_roster("ABBEMARIGNAN'S NIECE", index),
                "ABBE MARIGNAN'S NIECE", "and so does the second family")



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
        test_real_roster_resolves_to_two_fewer_names,
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
