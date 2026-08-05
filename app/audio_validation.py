"""Shared structural validation for newly generated audio files."""
import os
import struct


class GeneratedAudioError(RuntimeError):
    """Generated output is missing, empty, undecodable, or incomplete."""


def remove_stale_audio(path):
    """Remove an earlier output so it cannot masquerade as a fresh render."""
    if os.path.exists(path):
        os.remove(path)


def validate_generated_audio(path, context="audio generation"):
    """Return ``path`` after fully validating a newly generated audio file."""
    if not os.path.exists(path):
        raise GeneratedAudioError(f"{context} wrote no file: {path}")
    size = os.path.getsize(path)
    if size == 0:
        raise GeneratedAudioError(f"{context} wrote an empty file: {path}")

    try:
        import soundfile as sf
        info = sf.info(path)
        frames, rate = info.frames, info.samplerate
        # Reading the entire stream exercises decoding beyond the header.
        with sf.SoundFile(path) as audio:
            while audio.read(65536).size:
                pass
    except Exception as exc:  # noqa: BLE001
        raise GeneratedAudioError(
            f"{context} wrote undecodable audio: {type(exc).__name__}: "
            f"{str(exc)[:120]}") from exc
    if not frames or not rate:
        raise GeneratedAudioError(
            f"{context} wrote no audio frames ({frames} frames @ {rate} Hz)")

    with open(path, "rb") as handle:
        head = handle.read(12)
    if len(head) >= 12 and head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        declared = struct.unpack("<I", head[4:8])[0] + 8
        # RIFF uses a 32-bit size. Near the limit it may wrap for audiobook-
        # length files, so only ordinary declared sizes are comparable.
        if declared < 0xFFFFFFF0 and size < declared:
            raise GeneratedAudioError(
                f"{context} wrote truncated audio: header declares {declared} "
                f"bytes, file is {size} ({declared - size} missing)")
    return path
