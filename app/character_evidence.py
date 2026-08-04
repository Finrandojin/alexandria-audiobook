"""Read a character's gender out of the narration.

NOT WIRED INTO PRODUCTION. Nothing in `routers/voices.py` calls this; the
production change that removed dialogue as a trait source did not replace it
with narration evidence, so a character with no gender-bearing label or persona
still resolves "unknown" in casting. Connecting it is a deliberate behaviour
change to voice allocation and has not been made. Treat the numbers below as an
offline measurement, not as current system behaviour.


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

WHAT WORKS BETTER, and the honest description of it. Two constructions USUALLY
bind a pronoun to the clause subject:

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

AN EARLIER VERSION OF THIS DOCSTRING CLAIMED THESE CONSTRUCTIONS "cannot float
to another referent". That was wrong, and an external review reproduced it:
"Subaru watched Emilia raise her hand" matched `Subaru ... her hand` and scored
feminine for Subaru. The regex was proximity wearing the vocabulary of grammar.
What now separates it from plain proximity is the INTERVENING-NAME RULE - a
match is discarded when another known character is named between the target and
the construction, because the nearer name is the likelier subject. That needs a
roster; without one the function degrades to the old behaviour and callers
should pass one.

WHAT THIS IS NOT. It is not coreference and does not pretend to be. It is a
heuristic that abstains often, and it does not resolve subordinate clauses or
coordination. It also says nothing about characters who are androgynous, non-human
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


def _spans(narration, name, window, tail):
    """Text between `name` and a following construction, for each occurrence."""
    pattern = rf"\b{re.escape(name)}\b([^.!?]{{0,{window}}}?)\b(?:{tail})\b"
    return [m.group(1) for m in re.finditer(pattern, narration, re.I)]


def _count(narration, name, others=(), window_reflexive=REFLEXIVE_WINDOW,
           window_possessive=POSSESSIVE_WINDOW):
    """Count constructions that plausibly bind to `name`.

    THE INTERVENING-NAME RULE IS WHAT MAKES THIS MORE THAN PROXIMITY, and its
    absence was a real defect: "Subaru watched Emilia raise her hand" matched
    `Subaru ... her hand` and scored FEMININE for Subaru. An external review
    reproduced exactly that. A match is now discarded when another known
    character is named between the target and the construction, because the
    nearer name is the likelier subject.

    This is still not parsing. It is a heuristic that abstains more often, and
    it does not claim to resolve subordinate clauses or coordination - those
    are left to `others` catching the second name, and to the majority
    threshold when it does not.
    """
    others_re = "|".join(re.escape(o) for o in others if o and
                         o.casefold() != str(name).casefold())
    blocked = re.compile(rf"\b(?:{others_re})\b", re.I) if others_re else None

    def tally(window, tail):
        hits = 0
        for span in _spans(narration, name, window, tail):
            if blocked and blocked.search(span):
                continue
            hits += 1
        return hits

    masc = tally(window_reflexive, "himself")
    masc += tally(window_possessive, rf"his (?:{BODY_PARTS})")
    fem = tally(window_reflexive, "herself")
    fem += tally(window_possessive, rf"her (?:{BODY_PARTS})")
    return masc, fem


def gender_from_narration(narration, name, min_evidence=MIN_EVIDENCE,
                          majority=MAJORITY, aliases=None, roster=None):
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
    # Every OTHER known character blocks a match that would have to reach past
    # them. Without a roster this degrades to the old proximity behaviour, so
    # callers should pass one.
    others = [o for o in (roster or []) if o and o not in names]
    masc = fem = 0
    for candidate in names:
        m, f = _count(narration, candidate, others)
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
    # Normalise the WHOLE mapping once. Probing the original dict with three
    # spellings of the query missed any stored key whose casing matched none of
    # them: aliases_for('SUBARU', {'Subaru': 'NATSUKI SUBARU'}) returned an
    # empty set, so a character's evidence was not pooled across spellings.
    folded = {str(k).casefold(): str(v) for k, v in aliases.items()}
    target = folded.get(str(name).casefold(), str(name)).casefold()
    out = {target}
    for alias, canonical in folded.items():
        if canonical.casefold() == target:
            out.add(alias)
    # Return the ORIGINAL spellings, since they are what the narration uses.
    originals = {str(k) for k in aliases} | {str(v) for v in aliases.values()}
    return {o for o in originals
            if o.casefold() in out and o.casefold() != str(name).casefold()}


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
