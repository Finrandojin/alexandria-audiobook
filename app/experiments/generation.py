"""One way to render a segment, so failures cannot be mistaken for successes.

THE DEFECT THIS EXISTS TO PREVENT. `tts.py`'s generate_* methods return False
on failure rather than raising - there are 26 `return False` sites. Every
experiment harness ignored that boolean and tested `os.path.exists(wav)`
instead. If a WAV from an earlier run was already at that path, a FAILED
generation was counted as a success and STALE AUDIO WAS SCORED AS FRESH. The
harness cannot see it; the artifact looks complete; the number is wrong in a
direction nobody checks.

Six harnesses had this independently, which is the argument for one shared
implementation rather than six corrected copies - they would drift apart again.

WHAT THIS DOES, in order, because the order is the fix:

    1. delete any existing output first, so a stale file cannot survive
    2. dispatch through tts.voice_category, exactly as production does
    3. check the RETURNED BOOLEAN, not just the file
    4. check the file exists and is non-empty
    5. raise GenerationFailed naming which check failed

Raising rather than returning a flag is deliberate. A caller that forgets to
check a returned bool is how this class of bug started; an exception cannot be
ignored by omission.
"""
import os


class GenerationFailed(RuntimeError):
    """A segment did not render. Carries which check caught it."""


def render(engine, text, instruct, speaker, voice_config, voice_data, path):
    """Render one segment to `path`, or raise GenerationFailed.

    `voice_data` is passed separately from `voice_config` because the LoRA path
    takes the entry directly while the clone and custom paths take the whole
    config and look the speaker up themselves.
    """
    from tts import voice_category

    # Delete FIRST. This is the actual fix: without it, a False return plus a
    # leftover file from a previous run is indistinguishable from success.
    if os.path.exists(path):
        os.remove(path)

    category = voice_category(voice_data)
    if category == "lora":
        ok = engine.generate_lora_voice(text, instruct, voice_data, path)
    elif category == "clone":
        ok = engine.generate_clone_voice(text, speaker, voice_config, path)
    else:
        ok = engine.generate_custom_voice(text, instruct, speaker,
                                          voice_config, path)

    # `is False` rather than `not ok`: some paths return None on success, and
    # treating that as failure would discard good audio.
    if ok is False:
        raise GenerationFailed(f"{category} generation returned False for "
                               f"{speaker!r}")
    if not os.path.exists(path):
        raise GenerationFailed(f"{category} generation wrote no file for "
                               f"{speaker!r}")
    if os.path.getsize(path) == 0:
        raise GenerationFailed(f"{category} generation wrote an empty file for "
                               f"{speaker!r}")
    return path
