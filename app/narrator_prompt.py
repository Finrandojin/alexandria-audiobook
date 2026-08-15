"""Prompt and validation helpers for explicit first-person narrator metadata."""
import re


def normalize_narrator_name(name):
    return " ".join((name or "").upper().split())


def get_valid_narrator_name(name):
    """Return a safe canonical narrator name, or raise for invalid metadata."""
    narrator = normalize_narrator_name(name)
    if not narrator:
        return None
    if len(narrator) > 100:
        raise ValueError("first-person narrator must be 100 characters or fewer")
    if narrator in {"NARRATOR", "UNKNOWN"}:
        raise ValueError("first-person narrator must be an exact character name")
    return narrator


def is_narrator_attested(name, source_text, minimum=3):
    """Require a meaningful name token to occur before seeding the roster."""
    tokens = [token for token in normalize_narrator_name(name).split()
              if len(token) >= 2]
    source = (source_text or "").upper()
    return bool(tokens) and any(len(re.findall(
        r"\b" + re.escape(token) + r"\b", source)) >= minimum
        for token in tokens)


def add_narrator_prior(base_system, narrator):
    """Return the attribution prompt with book-level narrator metadata."""
    narrator = normalize_narrator_name(narrator)
    return base_system.rstrip() + (
        f"\n\nThis book is narrated in the first person by {narrator}. The "
        f"narration is {narrator}'s own voice, so lines of interior thought or "
        f"unmarked commentary are usually {narrator} speaking. Other characters "
        f"are normally introduced by name or by who is being addressed.")


def add_first_person_awareness(base_system):
    """Add a generic rule that requires a roster identity for the narrator."""
    return base_system.rstrip() + (
        "\n\nA first-person narrator may also speak quoted dialogue. When the "
        "passage indicates that the first-person narrator speaks the line, "
        "return that narrator's CHARACTER NAME from the roster. Never answer "
        "THE NARRATOR; infer which roster character narrates from the passage "
        "and the other entries in the batch.")
