"""Empirical verification tool for the optional TTS text-normalization flag.

Alexandria's audiobooks are verbatim by default: `config["tts"]
["enable_nemo_normalization"]` defaults to False and the TTS engine sends
narration text through unchanged (see app/tts_normalizer.py for the full
design rationale, including the nemo -> wetext -> none backend chain).
Whether turning normalization ON is worth it for a given book is an
empirical question -- this script exists to answer it without guessing.

Usage:
    python tools/verify_tts_normalization.py
        Text-only comparison: raw text vs. the active backend's
        normalized text (nemo if available, else wetext, else
        "unavailable") vs. pronunciation-dict-applied text. Works on any
        machine -- on Windows nemo is normally unavailable but wetext has
        prebuilt wheels, so this shows real normalized output there too.

    python tools/verify_tts_normalization.py --synthesize
        Additionally constructs a real TTSEngine from app/config.json and
        renders the fixed test sentence to two wav files -- one with the
        flag off, one with it on -- so you can listen to the difference.
        Requires the TTS runtime deps (torch/numpy/soundfile/pydub) and at
        least one configured voice in voice_config.json. On a dev machine
        missing those deps this prints a clear message and exits 2 rather
        than a traceback; on Linux/Colab (with deps + a configured voice)
        it actually synthesizes.
"""

import argparse
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.join(ROOT_DIR, "app")
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import tts_normalizer  # noqa: E402  (must come after sys.path tweak above)

FIXED_SENTENCE = "Dr. Halloway lived on Elm St."
PRONUNCIATION_DICT_PATH = os.path.join(ROOT_DIR, "pronunciation_dict.json")


def text_comparison():
    """Compare raw / backend-normalized / dict-applied text. Works everywhere."""
    print("=" * 70)
    print("TTS normalization check -- text-only comparison")
    print("=" * 70)
    print(f"Raw text:             {FIXED_SENTENCE!r}")

    normalizer_obj, backend_name = tts_normalizer._select_backend()
    if normalizer_obj is None:
        print("Normalized text:      <unavailable -- no backend>")
        print(
            "  Reason: neither nemo_text_processing nor wetext could be imported in\n"
            "  this Python environment. Install one of:\n"
            "      pip install nemo_text_processing   (Linux/Colab; highest fidelity)\n"
            "      pip install wetext                 (Windows-friendly, prebuilt wheels)\n"
            "  and re-run this script to see real normalized output."
        )
        backend_text = FIXED_SENTENCE
    else:
        backend_text = tts_normalizer._apply_backend_normalization(FIXED_SENTENCE)
        print(f"Normalized text ({backend_name}): {backend_text!r}")
        if backend_text != FIXED_SENTENCE:
            print("  (differs from raw text)")
        else:
            print("  (identical to raw text)")
        if backend_name == "wetext":
            print(
                "  Note: nemo_text_processing is unavailable in this environment (expected\n"
                "  on Windows -- its pynini dependency has no prebuilt wheel), so wetext is\n"
                "  carrying normalization here. wetext catches fewer cases than nemo (e.g. it\n"
                "  won't expand \"St. Peter's\" -> \"Saint Peter's\") but never over-expands."
            )

    pdict = tts_normalizer.load_pronunciation_dict(PRONUNCIATION_DICT_PATH)
    if pdict:
        dict_applied = tts_normalizer._apply_pronunciation_dict(backend_text, pdict)
        print(f"Pronunciation dict:   {PRONUNCIATION_DICT_PATH} ({len(pdict)} entries)")
        print(f"Dict-applied text:    {dict_applied!r}")
    else:
        print(f"Pronunciation dict:   none found at {PRONUNCIATION_DICT_PATH!r} -- skipping this step.")

    print("=" * 70)


def _find_first_voice(voice_config):
    for speaker, data in voice_config.items():
        if isinstance(data, dict) and data.get("type"):
            return speaker, data
    return None, None


def synthesize_comparison():
    """Render the fixed sentence with the flag off and on via a real TTSEngine."""
    config_path = os.environ.get("ALEXANDRIA_CONFIG_PATH") or os.path.join(APP_DIR, "config.json")
    voice_config_path = os.path.join(ROOT_DIR, "voice_config.json")
    out_dir = os.path.join(ROOT_DIR, "tools", "normalization_check")

    print("\n" + "=" * 70)
    print("TTS normalization check -- synthesis comparison")
    print("=" * 70)

    if not os.path.exists(config_path):
        print(f"ERROR: config file not found at {config_path!r}. Cannot synthesize.")
        print("(Set ALEXANDRIA_CONFIG_PATH, or create app/config.json, then re-run.)")
        return 2
    if not os.path.exists(voice_config_path):
        print(f"ERROR: voice_config.json not found at {voice_config_path!r}. Cannot synthesize.")
        print("(Configure at least one voice in the app first, then re-run.)")
        return 2

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        with open(voice_config_path, "r", encoding="utf-8") as f:
            voice_config = json.load(f)
    except Exception as exc:
        print(f"ERROR: could not read config/voice_config JSON: {exc}")
        return 2

    speaker, _voice_data = _find_first_voice(voice_config)
    if not speaker:
        print(f"ERROR: no usable voice entries found in {voice_config_path!r}.")
        print("Configure at least one voice before running --synthesize.")
        return 2

    try:
        from tts import TTSEngine
    except Exception as exc:
        print(f"ERROR: could not import TTSEngine -- a TTS runtime dependency is missing\n"
              f"in this Python environment (e.g. torch/numpy/soundfile/pydub): {exc}")
        print("This is expected on a dev box without the TTS deps installed; "
              "run --synthesize on Linux/Colab instead.")
        return 2

    os.makedirs(out_dir, exist_ok=True)
    print(f"Using voice: {speaker!r}")
    print(f"Output dir:  {out_dir}")

    wrote_all = True
    for flag_name, flag_value in (("flag_off", False), ("flag_on", True)):
        cfg = dict(config)
        tts_cfg = dict(cfg.get("tts", {}))
        tts_cfg["enable_nemo_normalization"] = flag_value
        cfg["tts"] = tts_cfg

        print(f"\n--- {flag_name} (enable_nemo_normalization={flag_value}) ---")
        try:
            engine = TTSEngine(cfg)
        except Exception as exc:
            print(f"ERROR: could not construct TTSEngine: {exc}")
            return 2

        sent_text = engine._normalizer.normalize(FIXED_SENTENCE)
        print(f"Text sent to engine: {sent_text!r}")

        output_path = os.path.join(out_dir, f"{flag_name}.wav")
        try:
            success = engine.generate_voice(FIXED_SENTENCE, None, speaker, voice_config, output_path)
        except Exception as exc:
            print(f"ERROR during generation for {flag_name}: {exc}")
            return 2

        if success:
            print(f"Wrote {output_path}")
        else:
            print(f"Generation reported failure for {flag_name} (speaker={speaker!r}).")
            wrote_all = False

    if not wrote_all:
        print("\nNot all wavs were generated successfully -- see messages above.")
        return 2

    print(f"\nDone. Listen to and compare the wavs in {out_dir}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Verify TTS text normalization output (text-only, or with --synthesize).")
    parser.add_argument(
        "--synthesize", action="store_true",
        help="Also render audio via TTSEngine for flag off/on (needs torch/soundfile + a configured voice).",
    )
    args = parser.parse_args()

    text_comparison()

    if args.synthesize:
        return synthesize_comparison()
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
