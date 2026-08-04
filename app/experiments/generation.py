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

    # NON-EMPTY IS NOT THE SAME AS USABLE. Existence and size were the only
    # checks here, so a truncated or malformed WAV - a run killed mid-write, a
    # disk that filled, a header with no samples - passed as a success and was
    # scored. That is the stale-audio defect one layer down: the bytes are
    # fresh, and still not audio.
    #
    # An external test plan asked for decodable-WAV validation on 2026-08-04
    # and it did not exist. Checked here rather than in each harness, for the
    # same reason the rest of this function exists.
    try:
        import soundfile as sf
        info = sf.info(path)
        frames, rate = info.frames, info.samplerate
    except Exception as exc:                            # noqa: BLE001
        raise GenerationFailed(f"{category} generation wrote an undecodable "
                               f"file for {speaker!r}: {type(exc).__name__} "
                               f"{str(exc)[:80]}") from exc
    # A header can be valid while describing nothing. Zero frames is silence
    # that no listener would accept and no metric would flag.
    if not frames or not rate:
        raise GenerationFailed(f"{category} generation wrote a header with no "
                               f"audio for {speaker!r} "
                               f"({frames} frames @ {rate} Hz)")

    # DECODING IS NOT ENOUGH EITHER, which the first version of this check got
    # wrong. Truncating a real 195,884-byte render to 5,000 bytes still decodes
    # - libsndfile returns the frames that are actually present (2,478) rather
    # than raising. So a run killed mid-write yields a short, valid, wrong file
    # and `sf.info` is happy with it.
    #
    # The RIFF header declares how many bytes of samples SHOULD follow. Compare
    # that against what is on disk: a file smaller than its own header claims
    # is truncated, whatever it decodes to.
    _check_riff_completeness(path, category, speaker)
    return path


# 8-byte RIFF preamble ("RIFF" + size) that the declared size excludes.
_RIFF_PREAMBLE = 8


def _check_riff_completeness(path, category="", speaker=""):
    """Raise if a RIFF file is shorter than its own header says it is.

    Deliberately tolerant in one direction: a file LARGER than declared is
    fine (trailing metadata chunks are legal and common), and anything that is
    not RIFF is left alone here because the decode check above already covered
    it. Only a shortfall is an error.
    """
    import struct
    with open(path, "rb") as fh:
        head = fh.read(12)
    if len(head) < 12 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        return                      # not RIFF; the decode check already ran
    declared = struct.unpack("<I", head[4:8])[0] + _RIFF_PREAMBLE
    actual = os.path.getsize(path)
    # The size field is 32-bit and wraps past 4 GiB. This repo has already been
    # bitten by that on long audiobook WAVs, where the wrapped value is
    # meaningless rather than a shortfall - so do not treat it as truncation.
    if declared >= 0xFFFFFFF0:
        return
    if actual < declared:
        raise GenerationFailed(
            f"{category} generation wrote a truncated file for {speaker!r}: "
            f"header declares {declared} bytes, file is {actual} "
            f"({declared - actual} missing)")
