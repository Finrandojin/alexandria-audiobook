import os

_PROMPTS_FILE = os.path.join(os.path.dirname(__file__), "..", "review_prompts.txt")


def load_review_prompts():
    """Read review_prompts.txt from disk and return (system_prompt, user_prompt).

    Re-reads on every call so edits are picked up without restarting the app.
    """
    try:
        with open(_PROMPTS_FILE, "r", encoding="utf-8") as f:
            raw = f.read()
    except FileNotFoundError:
        raise RuntimeError(
            f"review_prompts.txt not found at {os.path.abspath(_PROMPTS_FILE)}. "
            "This file is required for the review pass."
        )

    parts = raw.split("---SEPARATOR---", maxsplit=1)
    if len(parts) != 2:
        raise RuntimeError(
            "review_prompts.txt is malformed: expected exactly one '---SEPARATOR---' delimiter."
        )

    return parts[0].strip(), parts[1].strip()


# Cached at import time — used by review_script.py (subprocess, fresh each run)
REVIEW_SYSTEM_PROMPT, REVIEW_USER_PROMPT = load_review_prompts()
