#!/usr/bin/env python3
"""
voice_profiler.py — Analyze narrator ref.wav clips and generate voice descriptions.

For each adapter in manifest.json:
  1. Extracts ref.wav from the source zip
  2. Runs librosa acoustic analysis (pitch, energy, rate, timbre)
  3. Sends feature summary + narrator name to local Qwen LLM
  4. Writes voice_profile back to manifest + exports voice_profiles.csv

Usage:
    python voice_profiler.py
    python voice_profiler.py --dry_run          # acoustic analysis only, no LLM
    python voice_profiler.py --overwrite        # re-profile already-profiled entries
    python voice_profiler.py --model /path/to/other.gguf
"""

import argparse
import csv
import io
import json
import os
import re
import sys
import tempfile
import zipfile

try:
    import numpy as np
    import librosa
    DEPENDENCY_ERROR = None
except ImportError as e:
    np = None
    librosa = None
    DEPENDENCY_ERROR = e

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(SCRIPT_DIR, "app")
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from voicelab_settings import get_profiler_paths

CSV_FIELDS = ("id", "narrator", "best_loss", "voice_profile", "gender_est",
              "mean_f0", "std_f0", "speaking_rate")


def atomic_json_write(data, target_path):
    """Write JSON via temp file + os.replace, so a crash mid-write (Ctrl+C,
    OOM-kill, etc.) during this script's per-narrator manifest checkpoint
    can't truncate/corrupt manifest.json - a shared file that
    batch_train_lora.py, name_voices.py, and the web app's LoRA listing all
    depend on, not just this script's own progress."""
    directory = os.path.dirname(target_path) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, target_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def atomic_csv_write(rows: list[dict], target_path: str) -> None:
    """Atomically replace the profile CSV with rows from the current manifest."""
    directory = os.path.dirname(target_path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".csv", dir=directory)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_path, target_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def get_preflight_report(manifest_path: str, model_path: str,
                         output_csv: str, epub_dirs: list[str]) -> dict:
    """Check profiler prerequisites without loading the GGUF or changing data."""
    errors = []
    warnings = []
    if DEPENDENCY_ERROR is not None:
        errors.append(f"acoustic dependency unavailable: {DEPENDENCY_ERROR}")
    try:
        from llama_cpp import Llama
        if Llama is None:
            errors.append("llama_cpp.Llama is unavailable")
    except (ImportError, OSError) as e:
        errors.append(f"llama_cpp unavailable: {e}")
    try:
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        if not isinstance(manifest, list):
            errors.append("manifest must contain a JSON list")
    except (OSError, json.JSONDecodeError) as e:
        errors.append(f"manifest unreadable: {e}")
    if not os.path.isfile(model_path):
        errors.append(f"model not found: {model_path}")
    output_dir = os.path.dirname(os.path.abspath(output_csv))
    probe_path = None
    try:
        fd, probe_path = tempfile.mkstemp(prefix=".voice_profiler_check_", dir=output_dir)
        os.close(fd)
    except OSError:
        errors.append(f"output directory is not writable: {output_dir}")
    finally:
        if probe_path:
            try:
                os.remove(probe_path)
            except OSError:
                errors.append(f"preflight probe could not be removed: {probe_path}")
    for path in epub_dirs:
        if not os.path.isdir(path) or not os.access(path, os.R_OK):
            warnings.append(f"EPUB directory is not readable: {path}")
    return {"status": "passed" if not errors else "failed",
            "errors": errors, "warnings": warnings}


def describe_model_init_error(error: Exception) -> str:
    """Turn common llama.cpp startup failures into an actionable one-line error."""
    detail = str(error).strip() or type(error).__name__
    lowered = detail.lower()
    if "out of memory" in lowered or "memory allocation" in lowered:
        return f"insufficient GPU memory while loading the profiler model: {detail}"
    if "gguf" in lowered or "magic" in lowered:
        return f"invalid or incompatible GGUF model: {detail}"
    return f"profiler model initialization failed: {detail}"
PROFILER_PATHS = get_profiler_paths(SCRIPT_DIR)
MANIFEST = PROFILER_PATHS["manifest"]
MODEL_PATH = PROFILER_PATHS["model"]
OUTPUT_CSV = PROFILER_PATHS["output_csv"]


# ── Acoustic analysis ─────────────────────────────────────────────────────────

def analyze_ref_wav(wav_bytes: bytes) -> dict:
    y, sr = librosa.load(io.BytesIO(wav_bytes), sr=22050, mono=True)
    duration = len(y) / sr

    # Pitch via probabilistic YIN.
    #
    # Plain librosa.yin returns a pitch for EVERY frame, including silence,
    # breaths and unvoiced consonants, and it has no voiced/unvoiced decision.
    # Measured on the eight lora_models reference samples: only 33% of frames
    # are actually voiced, so two thirds of what yin reported was noise that
    # the 60-380Hz filter could not remove - spurious estimates land inside
    # that band. The effect was a mean f0 disagreement of 17.6 Hz and a std_f0
    # inflated 1.4x, on figures that feed voice naming and dedup. 17.6 Hz is
    # the width of a voice category, not a rounding error.
    #
    # pyin makes the voiced/unvoiced decision explicitly and returns NaN where
    # there is no pitch. It is several times slower, which is irrelevant here:
    # this runs once per reference sample, not per generated line.
    #
    # Verified NOT to octave-double, which was the obvious way this could have
    # made things worse: energy at f0/2 relative to f0 is 0.01-0.08 across the
    # reference samples, so the fundamentals it reports are real.
    #
    # CONSEQUENCE WORTH KNOWING. name_voices.py assigns gender from
    # mean_f0 >= 165, and on these samples the corrected pitch moves three of
    # twelve voices across that line. That is not this function's error - a
    # bright male tenor really does sit near 180 Hz, and the threshold was
    # always too crude to separate him from a contralto. Better pitch removes
    # the noise that was hiding it.
    f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=400, sr=sr)
    voiced = f0[voiced_flag & ~np.isnan(f0)]
    mean_f0 = float(np.mean(voiced)) if len(voiced) > 0 else 0.0
    std_f0  = float(np.std(voiced))  if len(voiced) > 0 else 0.0

    # Energy
    rms = librosa.feature.rms(y=y)[0]
    mean_rms = float(np.mean(rms))

    # Brightness (spectral centroid)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_centroid = float(np.mean(centroid))

    # Spectral rolloff — where 85% of energy lives; low = dark/warm, high = bright
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    mean_rolloff = float(np.mean(rolloff))

    # Voice smoothness: harmonic component energy vs total energy
    # Calibrated for speech (speech median ~0.37, max ~0.75 — much lower than instruments)
    y_harmonic = librosa.effects.harmonic(y, margin=2.0)
    harm_rms  = float(np.mean(librosa.feature.rms(y=y_harmonic)[0]))
    total_rms = float(np.mean(librosa.feature.rms(y=y)[0])) + 1e-8
    smoothness = harm_rms / total_rms

    # Spectral flatness — noise-like quality; calibrated for speech (median ~0.037)
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)[0]))

    # Speaking rate
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
    rate = len(onsets) / duration if duration > 0 else 0.0

    return {
        'mean_f0':       mean_f0,
        'std_f0':        std_f0,
        'mean_rms':      mean_rms,
        'mean_centroid': mean_centroid,
        'mean_rolloff':  mean_rolloff,
        'smoothness':    smoothness,
        'flatness':      flatness,
        'speaking_rate': rate,
        'duration':      duration,
    }


def interpret_features(f: dict) -> str:
    """Rich human-readable feature summary for the LLM prompt."""
    f0  = f['mean_f0']
    std = f['std_f0']
    rms = f['mean_rms']
    rate = f['speaking_rate']
    rolloff  = f['mean_rolloff']
    smooth   = f['smoothness']
    flat     = f['flatness']

    # Pitch / gender / register + rough age decade estimate
    if f0 < 100:
        pitch_label = "very deep bass male"
        age_est = "likely 50s–70s"
    elif f0 < 125:
        pitch_label = "deep baritone male"
        age_est = "likely 40s–60s"
    elif f0 < 145:
        pitch_label = "baritone male"
        age_est = "likely 35s–55s"
    elif f0 < 165:
        pitch_label = "tenor / upper-register male"
        age_est = "likely 25s–45s"
    elif f0 < 185:
        pitch_label = "low alto / mature female"
        age_est = "likely 40s–60s"
    elif f0 < 210:
        pitch_label = "mezzo-soprano female"
        age_est = "likely 30s–50s"
    elif f0 < 250:
        pitch_label = "soprano female"
        age_est = "likely 20s–40s"
    else:
        pitch_label = "high soprano female"
        age_est = "likely teens–30s"

    # Pitch variation → delivery style
    if std < 18:
        variation = "nearly monotone, flat affect"
    elif std < 30:
        variation = "controlled, measured variation"
    elif std < 50:
        variation = "moderate natural variation"
    elif std < 70:
        variation = "expressive range"
    else:
        variation = "wide dramatic range, theatrical"

    # Voice texture — thresholds calibrated to speech (smoothness median ~0.37, max ~0.75)
    if smooth > 0.55:
        texture = "smooth, clean tone"
    elif smooth > 0.43 and flat < 0.04:
        texture = "warm, slightly rounded texture"
    elif smooth > 0.43:
        texture = "warm but somewhat airy"
    elif smooth > 0.30 and flat > 0.08:
        texture = "breathy, airy quality"
    elif smooth > 0.30 and flat < 0.04:
        texture = "moderately textured, warm"
    elif smooth > 0.30:
        texture = "slightly husky texture"
    elif flat > 0.08:
        texture = "breathy with rough edges"
    elif smooth < 0.20:
        texture = "gravelly, rough, husky"
    else:
        texture = "husky, noticeably textured"

    # Brightness
    if rolloff < 2500:
        brightness = "very dark/warm resonance"
    elif rolloff < 3500:
        brightness = "warm, rounded tone"
    elif rolloff < 5000:
        brightness = "balanced, full tone"
    elif rolloff < 7000:
        brightness = "bright, clear tone"
    else:
        brightness = "very bright, thin/crisp tone"

    # Energy
    if rms < 0.025:
        energy = "soft, intimate volume"
    elif rms < 0.05:
        energy = "moderate conversational volume"
    elif rms < 0.09:
        energy = "strong, projected delivery"
    else:
        energy = "loud, powerful projection"

    # Pace
    if rate < 2.5:
        pace = "very slow, deliberate"
    elif rate < 3.5:
        pace = "slow, measured"
    elif rate < 4.5:
        pace = "moderate pace"
    elif rate < 5.5:
        pace = "brisk pace"
    else:
        pace = "rapid-fire delivery"

    return (
        f"Pitch: {f0:.0f}Hz → {pitch_label}, {age_est}\n"
        f"Pitch variation: {std:.0f}Hz → {variation}\n"
        f"Voice texture: {texture}\n"
        f"Tone brightness: {brightness}\n"
        f"Volume: {energy}\n"
        f"Pace: {pace}"
    )


# ── EPUB extraction ──────────────────────────────────────────────────────────

from html.parser import HTMLParser

class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self._skip = True
        if tag in ('p', 'div', 'br', 'h1', 'h2', 'h3', 'h4'):
            self.parts.append('\n')

    def handle_endtag(self, tag):
        if tag in ('script', 'style'):
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)

    def text(self):
        return re.sub(r'\n{3,}', '\n\n', ''.join(self.parts)).strip()


def find_epub(dataset_id: str, epub_dirs: list[str]) -> str | None:
    """Find EPUB matching this adapter — first by ASIN, then by title tokens."""
    _, title, asin = get_dataset_identity(dataset_id)
    title_words = [word for word in re.sub(r"[^a-z0-9 ]", " ", title.lower()).split()
                   if len(word) >= 2]

    # Collect all candidates with scores, return best
    candidates = []
    for epub_dir in epub_dirs:
        if not os.path.isdir(epub_dir):
            continue
        for fname in sorted(os.listdir(epub_dir)):
            if not fname.lower().endswith('.epub'):
                continue
            fl = fname.lower()
            # Exact ASIN match wins immediately
            if asin and f"[{asin.lower()}]" in fl:
                return os.path.join(epub_dir, fname)
            if title_words:
                hits = sum(1 for w in title_words if re.search(rf'\b{re.escape(w)}\b', fl))
                ratio = hits / len(title_words)
                # Require >60% of title words to match
                if ratio > 0.6:
                    candidates.append((ratio, os.path.join(epub_dir, fname)))

    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]

    # Fallback: if any title word is a standalone number, try matching that alone
    number_words = [w for w in title_words if w.isdigit()]
    for num in number_words:
        for epub_dir in epub_dirs:
            if not os.path.isdir(epub_dir):
                continue
            for fname in sorted(os.listdir(epub_dir)):
                if fname.lower().endswith('.epub') and re.search(rf'\b{num}\b', fname.lower()):
                    return os.path.join(epub_dir, fname)
    return None


def extract_epub_passage(epub_path: str, target_chars: int = 600) -> str:
    """Pull a prose passage from ~20% into the book (past front matter)."""
    try:
        import zipfile as zmod
        with zmod.ZipFile(epub_path, 'r') as zf:
            names = set(zf.namelist())
            if 'META-INF/container.xml' not in names:
                return ""
            container = zf.read('META-INF/container.xml').decode('utf-8', errors='ignore')
            opf_m = re.search(r'full-path="([^"]+\.opf)"', container)
            if not opf_m:
                return ""
            opf_path = opf_m.group(1)
            opf_dir  = os.path.dirname(opf_path)
            opf      = zf.read(opf_path).decode('utf-8', errors='ignore')

            spine_ids = re.findall(r'<itemref\s+idref="([^"]+)"', opf)
            manifest  = {m.group(1): m.group(2)
                         for m in re.finditer(r'<item\b[^>]+\bid="([^"]+)"[^>]+href="([^"]+)"', opf)}

            start = max(1, len(spine_ids) // 5)  # skip first 20%
            collected = ""
            for item_id in spine_ids[start:]:
                href = manifest.get(item_id, "")
                if not href:
                    continue
                fpath = (opf_dir + "/" + href).lstrip("/") if opf_dir else href
                if fpath not in names:
                    fpath = href
                if fpath not in names:
                    continue
                html_bytes = zf.read(fpath).decode('utf-8', errors='ignore')
                p = _TextExtractor()
                p.feed(html_bytes)
                raw = p.text()
                paras = [x.strip() for x in raw.split('\n\n') if len(x.strip()) > 80]
                for para in paras[:4]:
                    collected += para + " "
                if len(collected) >= target_chars:
                    break
            return collected[:target_chars].strip()
    except Exception:
        return ""


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_dataset_identity(dataset_id: str) -> tuple[str, str, str | None]:
    """Return narrator, book title, and ASIN parsed from a training dataset ID."""
    s = dataset_id.removeprefix("narrator_")
    s = re.sub(r"_char\d+_vol\d+$", "", s)
    m = re.search(r"_([bB][a-z0-9]{9}|\d{10})(?:_|$)", s)
    asin = m.group(1).upper() if m else None
    if m:
        s = s[:m.start()]
    parts = [p for p in s.split("_") if p]
    name_count = 3 if len(parts) >= 3 and len(parts[1]) == 1 else min(2, len(parts))
    narrator = " ".join(part.title() for part in parts[:name_count])
    title = " ".join(part.title() for part in parts[name_count:])
    return narrator, title, asin


def parse_narrator_name(dataset_id: str) -> str:
    """Extract a readable narrator name from dataset_id."""
    narrator, _, _ = get_dataset_identity(dataset_id)
    return narrator


def get_ref_wav(zip_path: str) -> bytes | None:
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            if 'ref.wav' in zf.namelist():
                return zf.read('ref.wav')
    except Exception as e:
        print(f"  ERROR reading zip: {e}", flush=True)
    return None


def get_ref_text(zip_path: str) -> str:
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            if 'ref_text.txt' in zf.namelist():
                return zf.read('ref_text.txt').decode('utf-8').strip()
    except Exception:
        pass
    return ""


def parse_book_title(dataset_id: str) -> str:
    """Extract book title from dataset_id after stripping narrator name and suffixes."""
    _, title, _ = get_dataset_identity(dataset_id)
    return title


def profile_csv_row(entry: dict) -> dict | None:
    """Build one CSV row from a completed manifest profile."""
    profile = entry.get("voice_profile")
    features = entry.get("voice_features")
    if not profile or not isinstance(features, dict):
        return None
    dataset_id = entry.get("dataset_id", entry.get("name", ""))
    mean_f0 = features.get("mean_f0", 0)
    return {
        "id": entry.get("id", ""),
        "narrator": parse_narrator_name(dataset_id),
        "best_loss": entry.get("best_loss", entry.get("final_loss", 0)),
        "voice_profile": profile,
        "gender_est": "female" if mean_f0 >= 165 else "male",
        "mean_f0": mean_f0,
        "std_f0": features.get("std_f0", 0),
        "speaking_rate": features.get("speaking_rate", 0),
    }


# ── LLM description ───────────────────────────────────────────────────────────

SYSTEM_MSG = (
    "You are a voice casting director for audiobooks. "
    "You write precise slate cards telling producers what a voice sounds like "
    "and what character type it should portray. Never use generic filler."
)

USER_TEMPLATE = """\
Write a voice casting description for this audiobook narrator.

Narrator: {narrator}
Book: {book_title}
Narrator's ref clip: "{ref_text}"
Book prose sample: "{book_passage}"

Acoustic data:
{summary}

Write ONE line (12-18 words) with TWO parts separated by " — ":
Part 1: What makes this voice DISTINCTIVE — use the book's genre/tone and the prose sample to inform the voice character
Part 2: Best CHARACTER AGE AND TYPE for this voice, specific to the kind of story it suits

BANNED: highly expressive, dynamic, engaging, tone, delivery
Use register words: baritone / tenor / alto / soprano / bass / mezzo
Use texture words: weathered / crisp / silky / husky / breathy / warm / gravelly / rich / bright / velvety

Examples:
- Rich weathered baritone, battle-worn and deliberate — best for 40s–60s fantasy commanders and veterans
- Bright clear soprano, quick-witted and playful — best for teenage to 20s anime-style heroines
- Warm intimate alto, literary and introspective — best for 30s–40s literary fiction female leads
- Deep gravelly bass, slow gothic authority — best for 60s+ Dracula-era aristocrats and elders
- Crisp bright tenor, urban and irreverent — best for 20s–30s contemporary male protagonists
- Silky cool mezzo, politically sharp and composed — best for 30s–50s sci-fi diplomats and spies

Voice casting line (one line, no quotes):"""


def llm_describe(llm, narrator: str, summary: str,
                 book_title: str = "", ref_text: str = "",
                 book_passage: str = "") -> str:
    ref_preview     = (ref_text[:120]     + "…") if len(ref_text)     > 120 else ref_text
    passage_preview = (book_passage[:400] + "…") if len(book_passage) > 400 else book_passage
    prompt = USER_TEMPLATE.format(
        narrator=narrator,
        book_title=book_title or "unknown",
        ref_text=ref_preview or "(not available)",
        book_passage=passage_preview or "(not available)",
        summary=summary,
    )
    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=80,
        temperature=0.6,
        stop=["\n"],
    )
    return resp['choices'][0]['message']['content'].strip().strip('"').strip("'")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Profile narrator voices via acoustics + LLM")
    parser.add_argument("--manifest",   default=MANIFEST)
    parser.add_argument("--model",      default=MODEL_PATH,    help="Path to GGUF model")
    parser.add_argument("--output_csv", default=OUTPUT_CSV)
    parser.add_argument("--epub-dir", dest="epub_dirs", action="append", default=[],
                        help="Optional EPUB search directory (repeatable)")
    parser.add_argument("--dry_run",    action="store_true",   help="Acoustics only, skip LLM")
    parser.add_argument("--overwrite",  action="store_true",   help="Re-profile existing entries")
    parser.add_argument("--check", action="store_true",
                        help="Validate prerequisites without loading the model or writing files")
    args = parser.parse_args()

    if args.check:
        report = get_preflight_report(args.manifest, args.model, args.output_csv, args.epub_dirs)
        print(json.dumps(report), flush=True)
        return 0 if report["status"] == "passed" else 1

    if not os.path.exists(args.manifest):
        print(f"ERROR: manifest not found: {args.manifest} (run batch_train_lora.py first)")
        return 1
    try:
        with open(args.manifest, encoding='utf-8') as f:
            manifest = json.load(f)
        if not isinstance(manifest, list):
            raise ValueError("manifest must contain a JSON list")
    except (OSError, json.JSONDecodeError, ValueError) as e:
        print(f"ERROR: manifest unreadable: {e}", flush=True)
        return 1

    batch = [e for e in manifest if e.get('zip_source')]
    todo  = [e for e in batch if args.overwrite or not e.get('voice_profile')]

    print(f"{len(batch)} adapters with zip_source — {len(todo)} to profile")
    if not todo:
        if not args.dry_run:
            csv_rows = [row for entry in manifest
                        if (row := profile_csv_row(entry)) is not None]
            try:
                atomic_csv_write(csv_rows, args.output_csv)
            except OSError as e:
                print(f"ERROR: unable to write profile CSV: {e}", flush=True)
                return 1
            print(f"CSV: {args.output_csv}")
        print("All already profiled. Use --overwrite to redo.")
        return 0

    if not args.dry_run:
        report = get_preflight_report(args.manifest, args.model, args.output_csv, args.epub_dirs)
        if report["status"] != "passed":
            print("ERROR: " + "; ".join(report["errors"]), flush=True)
            return 1

    # Load LLM once (expensive — ~10-20s for 14B Q6)
    llm = None
    if not args.dry_run:
        if not os.path.exists(args.model):
            print(f"ERROR: model not found: {args.model}")
            return 1
        print(f"Loading LLM: {os.path.basename(args.model)} …", flush=True)
        try:
            from llama_cpp import Llama
            llm = Llama(
                model_path=args.model,
                n_ctx=2048,
                n_gpu_layers=-1,
                verbose=False,
            )
        except Exception as e:
            print(f"ERROR: {describe_model_init_error(e)}", flush=True)
            return 1
        print("LLM ready.\n", flush=True)

    preview_rows = []
    errors = 0

    for i, entry in enumerate(todo, 1):
        dataset_id = entry.get('dataset_id', entry.get('name', ''))
        zip_path   = entry['zip_source']
        best_loss  = entry.get('best_loss', entry.get('final_loss', 0))

        print(f"[{i:3d}/{len(todo)}] {dataset_id[:72]}", flush=True)

        wav_bytes = get_ref_wav(zip_path)
        if wav_bytes is None:
            print(f"  SKIP — no ref.wav", flush=True)
            errors += 1
            continue

        try:
            features = analyze_ref_wav(wav_bytes)
            summary  = interpret_features(features)
        except Exception as e:
            print(f"  ERROR in acoustic analysis: {e}", flush=True)
            errors += 1
            continue

        print(f"  {summary}", flush=True)

        narrator     = parse_narrator_name(dataset_id)
        book_title   = parse_book_title(dataset_id)
        ref_text     = get_ref_text(zip_path)
        description  = summary  # fallback for dry_run

        epub_path    = find_epub(dataset_id, args.epub_dirs)
        book_passage = extract_epub_passage(epub_path) if epub_path else ""

        if epub_path:
            print(f"  epub: {os.path.basename(epub_path)}", flush=True)
        else:
            print(f"  epub: not found", flush=True)

        llm_failed = False
        if llm is not None:
            try:
                description = llm_describe(llm, narrator, summary,
                                           book_title=book_title, ref_text=ref_text,
                                           book_passage=book_passage)
                print(f"  → {description}", flush=True)
            except Exception as e:
                print(f"  LLM error ({e}) — leaving profile pending", flush=True)
                llm_failed = True
                errors += 1

        if not args.dry_run:
            entry['voice_features'] = {
                'mean_f0':       round(features['mean_f0'], 1),
                'std_f0':        round(features['std_f0'], 1),
                'mean_rms':      round(features['mean_rms'], 4),
                'speaking_rate': round(features['speaking_rate'], 2),
                'mean_centroid': round(features['mean_centroid'], 0),
                'smoothness':    round(features['smoothness'], 3),
                'flatness':      round(features['flatness'], 4),
            }
            if not llm_failed:
                entry['voice_profile'] = description
            # Checkpoint manifest after each narrator
            try:
                atomic_json_write(manifest, args.manifest)
            except OSError as e:
                print(f"ERROR: unable to checkpoint profiler manifest: {e}", flush=True)
                return 1

        if not llm_failed:
            preview_rows.append({
                'id':            entry['id'],
                'narrator':      narrator,
                'best_loss':     best_loss,
                'voice_profile': description,
                'gender_est':    'female' if features['mean_f0'] >= 165 else 'male',
                'mean_f0':       round(features['mean_f0'], 1),
                'std_f0':        round(features['std_f0'], 1),
                'speaking_rate': round(features['speaking_rate'], 2),
            })

    if not args.dry_run:
        csv_rows = [row for entry in manifest if (row := profile_csv_row(entry)) is not None]
        try:
            atomic_csv_write(csv_rows, args.output_csv)
        except OSError as e:
            print(f"ERROR: unable to write profile CSV: {e}", flush=True)
            return 1
        print(f"\nCSV: {args.output_csv}")

    print(f"Done: {len(preview_rows)} profiles processed, {errors} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
