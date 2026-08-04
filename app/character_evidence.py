"""Read a character's gender out of the narration, using grammatical binding.

THE PROBLEM WITH THE OBVIOUS APPROACHES, each measured on the live book before
being discarded:

    a character's own dialogue     Subaru's 412 lines hold 46 feminine tokens
                                   against 27 masculine, because he talks about
                                   Emilia, Felt and Satella constantly. He
                                   classified FEMALE, as did ROM and Reinhard.
                                   Dialogue is evidence about its SUBJECT, not
                                   its speaker.

    pronouns near the name         "Subaru looked at her" counts a feminine
                                   token for Subaru. 71% masculine - not usable.

    single-name sentences only     removes cross-talk within a sentence but not
                                   across one: "She was waiting. Subaru saw
                                   her." Still only 74% masculine.

WHAT WORKS is not proximity but GRAMMAR. Two constructions bind a pronoun to
the clause subject and cannot float to another referent:

    reflexives              "Subaru pulled himself up" - a reflexive must agree
                            with the subject of its own clause.
    body-part possessives   "Subaru scratched his head" - the possessor of a
                            body part in a transitive clause is overwhelmingly
                            the subject, not a bystander.

On the same book that gives Subaru 79 masculine against 12 feminine - MALE, and
correct - while Emilia reads 0/8 feminine, ROM 9/1 and Reinhard 13/2, all
correct.

DELIBERATELY CONSERVATIVE. Narration only, never dialogue. A clear majority and
a minimum count are both required, and anything short returns "unknown" - which
every consumer already treats as "do not filter, do not penalise". Being silent
is cheap; being confidently wrong costs a main character their voice.

WHAT THIS IS NOT. It is not coreference and does not pretend to be. It finds
the cases where English grammar makes the referent unambiguous and abstains
elsewhere. It also says nothing about characters who are androgynous, non-human
or deliberately ambiguous - those come back "unknown" or "mixed", which is the
correct answer for them rather than a failure.
"""
import re

BODY_PARTS = ("head|face|eyes|eye|hand|hands|shoulder|shoulders|arm|arms|"
              "chest|mouth|lips|hair|body|voice|feet|foot|back|finger|fingers|"
              "neck|throat|cheek|cheeks|brow|fist|fists|knees|legs|leg")

# Window sizes differ because the constructions do. A reflexive can sit further
# from the name ("Subaru, still shaking, pulled himself up"); a body-part
# possessive is usually adjacent, and a wide window there starts catching other
# people's hands.
REFLEXIVE_WINDOW = 60
POSSESSIVE_WINDOW = 40

MIN_EVIDENCE = 3
MAJORITY = 0.8


def _count(narration, name, window_reflexive=REFLEXIVE_WINDOW,
           window_possessive=POSSESSIVE_WINDOW):
    n = re.escape(name)
    masc = len(re.findall(rf"\b{n}\b[^.!?]{{0,{window_reflexive}}}?\bhimself\b",
                          narration, re.I))
    masc += len(re.findall(
        rf"\b{n}\b[^.!?]{{0,{window_possessive}}}?\bhis ({BODY_PARTS})\b",
        narration, re.I))
    fem = len(re.findall(rf"\b{n}\b[^.!?]{{0,{window_reflexive}}}?\bherself\b",
                         narration, re.I))
    fem += len(re.findall(
        rf"\b{n}\b[^.!?]{{0,{window_possessive}}}?\bher ({BODY_PARTS})\b",
        narration, re.I))
    return masc, fem


def gender_from_narration(narration, name, min_evidence=MIN_EVIDENCE,
                          majority=MAJORITY, aliases=None):
    """-> (gender, confidence, evidence dict).

    gender is "male", "female" or "unknown". Confidence is "high" when the
    evidence is plentiful and one-sided, "medium" when it is thinner, and
    "unknown" when the answer is unknown - so a caller can tell a firm reading
    from a marginal one without re-deriving it.
    """
    if not narration or not name:
        return "unknown", "unknown", {"masculine": 0, "feminine": 0}
    # Narration uses whatever the author uses. 'NATSUKI SUBARU' reads 0/0
    # because the prose says "Subaru"; without folding aliases the character
    # with the richest evidence in the book looks like the one with none.
    names = {name} | set(aliases_for(name, aliases))
    masc = fem = 0
    for candidate in names:
        m, f = _count(narration, candidate)
        masc += m
        fem += f
    total = masc + fem
    evidence = {"masculine": masc, "feminine": fem, "total": total}
    if total < min_evidence:
        return "unknown", "unknown", evidence
    share = max(masc, fem) / total
    if share < majority:
        # Genuinely mixed. For an androgynous or non-human character this is
        # the right answer, not a failure to reach one.
        return "unknown", "unknown", evidence
    gender = "male" if masc > fem else "female"
    confidence = "high" if total >= 10 and share >= 0.85 else "medium"
    return gender, confidence, evidence


def aliases_for(name, aliases):
    """Every spelling that resolves to the same character.

    character_aliases.json is {"ALIAS": "CANONICAL"} with inconsistent casing,
    so it is folded and searched in both directions - a canonical name needs
    its aliases, and an alias needs its siblings.
    """
    if not aliases:
        return set()
    target = (aliases.get(name) or aliases.get(name.upper())
              or aliases.get(name.lower()) or name).upper()
    out = {target}
    for alias, canonical in aliases.items():
        if str(canonical).upper() == target:
            out.add(alias)
    return {a for a in out if a and a.upper() != name.upper()}


def narration_text(entries, narrator_names=("NARRATOR",)):
    """Join the narration from an annotated script or chunk list.

    Dialogue is excluded on purpose - it is the source that produced the
    inverted readings this module exists to replace.
    """
    upper = {n.upper() for n in narrator_names}
    return " ".join(
        e["text"] for e in entries
        if isinstance(e, dict) and e.get("text")
        and str(e.get("speaker") or "").upper() in upper)
