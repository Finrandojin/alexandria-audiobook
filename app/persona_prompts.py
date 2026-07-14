import os

_PROMPTS_FILE = os.path.join(os.path.dirname(__file__), "..", "persona_prompts.txt")


def load_persona_prompts():
    """Read persona_prompts.txt from disk and return (system_prompt, user_prompt, advanced_prompt).

    Re-reads on every call so edits are picked up without restarting the app.
    """
    try:
        with open(_PROMPTS_FILE, "r", encoding="utf-8") as f:
            raw = f.read()
    except FileNotFoundError:
        raise RuntimeError(
            f"persona_prompts.txt not found at {os.path.abspath(_PROMPTS_FILE)}. "
            "This file is required for persona generation."
        )

    parts = raw.split("---SEPARATOR---")
    if len(parts) != 3:
        raise RuntimeError(
            "persona_prompts.txt is malformed: expected exactly two '---SEPARATOR---' delimiters "
            "(system prompt, basic user prompt, advanced compilation prompt)."
        )

    return parts[0].strip(), parts[1].strip(), parts[2].strip()


# Cached at import time — used by generate_personas.py (subprocess, fresh each run)
PERSONA_SYSTEM_PROMPT, PERSONA_USER_PROMPT, PERSONA_ADVANCED_PROMPT = load_persona_prompts()
