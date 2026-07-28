"""One definition of "is this the right speaker", for every harness.

Eighteen harnesses in this directory each define their own `same()`. They agree
today because they were copy-pasted, which is exactly the situation Rule 15
warns about: two independently-maintained copies of one decision will drift,
and the drift is invisible because both keep producing plausible numbers. This
module is the single copy.

It also fixes a scoring bug the copies all share. Across every artifact in the
ledger, 268 of 21,190 wrong rows are not wrong:

    punctuation only       162   MR. PRIEST vs MR PRIEST, MS. SHORT HAIR
    romanization variant   106   ARUMANFI vs ALMANFI, RUDIUS vs RUDEUS
    genuinely different  20,922

The 162 are a period. `same()` upper-cased both sides and compared, so a model
that wrote the honorific with its full stop was marked wrong for it. That is a
defect in the instrument, not a finding about models, and it is folded into the
primary comparison here.

THE 106 ARE DELIBERATELY NOT. `attribution_accuracy.romaji_key` documents why
and the reasoning still holds: exact spelling measures whether a name is usable
downstream, because a misspelled speaker fragments the cast list and breaks
voice assignment, while phonetic matching measures whether the model identified
the right character. Those are different questions, and the penalty is
model-specific - magistral-small loses 7.9 points of oracle accuracy to
romanization where three other models lose nothing. Folding it into the primary
metric would hide a real, unevenly-distributed weakness. `same_speaker` reports
it separately so a caller can have both numbers.
"""
import re

_PUNCT = re.compile(r"[^A-Z0-9 ]")


def normalize(name):
    """Upper-case, strip punctuation, collapse whitespace.

    Punctuation only: this must not merge distinct characters, and the honorific
    cases it exists for (MR. TALL, MS. SHORT HAIR) differ from every other
    roster entry by more than a full stop.
    """
    return " ".join(_PUNCT.sub("", (name or "").upper()).split())


def alias_groups(*sources):
    """Collect alias groups from any number of fixture-shaped dicts or lists."""
    groups = []
    for source in sources:
        if not source:
            continue
        raw = source.get("aliases", []) if isinstance(source, dict) else source
        for group in raw or []:
            members = {normalize(n) for n in group if n}
            if len(members) > 1:
                groups.append(members)
    return groups


def same_speaker(expected, actual, groups=(), phonetic=False):
    """Is `actual` the speaker `expected` names?

    `phonetic=True` additionally accepts romanization variants. Leave it False
    for the headline number and call it twice if you want both - see the module
    docstring for why the two are not the same question.
    """
    a, b = normalize(expected), normalize(actual)
    if not b:
        return False
    if a == b:
        return True
    for group in groups:
        if a in group and b in group:
            return True
    if phonetic:
        # Imported lazily: this module is used by offline analyses that have no
        # reason to pull in the accuracy module's dependencies.
        from attribution_accuracy import romaji_key
        key = romaji_key(b)
        return bool(key) and key == romaji_key(a)
    return False
