"""Build a small candidate set per gold line, from evidence the pipeline has.

Instrument, not production logic. Sources are deliberately generous so the
result is an upper bound on recall; ablate afterwards to find the smallest
reliable set. Per the external review: measure whether the true speaker is even
available before asking whether the model can pick it.
"""
import re

SPEECH = (r"(?:said|says|asked|replied|answered|murmured|whispered|shouted|"
          r"cried|muttered|added|spoke|snapped|barked|laughed|sighed|"
          r"continued|declared|protested|demanded|called|put in|finished|"
          r"pouts|nods|shrugged|grinned|yelled|screamed|breathed)")


def _names_in(text, roster):
    """Roster names written in this text, matched case-insensitively."""
    found = []
    for name in roster:
        if re.search(r"\b" + re.escape(name) + r"\b", text or "", re.I):
            found.append(name.upper())
    return found


def tag_candidates(segmented, index, roster, window=2):
    """Names attached to a speech verb in the neighbouring narration.

    The only relation that is actual evidence of who is speaking: a vocative
    ("Nina, listen") names the listener, and a bare mention names neither.
    Measured on mushoku16, only 6.8% of wrong answers had one of these nearby,
    against 31% that merely had the name nearby somehow.
    """
    out = []
    for j in range(max(0, index - window), min(len(segmented), index + window + 1)):
        if j == index or segmented[j].get("type") != "NARRATOR":
            continue
        text = segmented[j].get("text") or ""
        for name in _names_in(text, roster):
            pattern = re.escape(name)
            if re.search(rf"\b{pattern}\b[^.!?]{{0,40}}\b{SPEECH}\b", text, re.I) or \
               re.search(rf"\b{SPEECH}\b[^.!?]{{0,25}}\b{pattern}\b", text, re.I):
                out.append(name)
    return out


def recent_speakers(named, index, depth=6):
    """Speakers of the last few attributed lines - an exchange usually alternates."""
    out = []
    for j in range(index - 1, max(-1, index - 1 - depth * 3), -1):
        if j < 0 or j >= len(named) or not named[j]:
            continue
        speaker = (named[j].get("speaker") or "").upper()
        if speaker and speaker not in ("NARRATOR", "UNKNOWN") and speaker not in out:
            out.append(speaker)
        if len(out) >= depth:
            break
    return out


def scene_names(segmented, index, roster, window=12):
    """Roster names appearing anywhere in the surrounding window."""
    out = []
    for j in range(max(0, index - window), min(len(segmented), index + window + 1)):
        for name in _names_in(segmented[j].get("text"), roster):
            if name not in out:
                out.append(name)
    return out


def build_candidates(segmented, named, index, roster, sources=None):
    """Ordered candidate set for one line. UNKNOWN is always available."""
    sources = sources or ("tag", "recent", "scene")
    parts = []
    if "tag" in sources:
        parts += tag_candidates(segmented, index, roster)
    if "recent" in sources:
        parts += recent_speakers(named, index)
    if "scene" in sources:
        parts += scene_names(segmented, index, roster)
    seen, ordered = set(), []
    for name in parts:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered
