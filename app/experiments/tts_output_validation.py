"""Does what Qwen3-TTS actually emits match the text we asked it to say?

Nothing in this project inspects generated audio. tts.py and project.py have no
transcribe-back, no word-error check, no truncation detection, and no retry on a
bad segment. Every gate is upstream, on text. Whatever the model emits ships.

That is a gap with no measurement attached, and the measurement is cheap because
whisper.cpp is already built in this repo and its model is on disk - integrated
for the preparer and otherwise idle.

THE METHOD. Generate a segment, transcribe it back, align the transcript to the
source word by word, and count the errors. A segment fails when errors exceed a
threshold that scales with length, because one wrong word in six is a defect and
one in two hundred is ASR noise.

WHAT MAKES THIS USABLE RATHER THAN NOISY. A naive comparison fails constantly on
correct audio: ASR writes "twenty five" for "25" and homophones for names. So a
mismatch is only counted when it survives normalisation of numbers, punctuation
and case. The residual false-positive rate is itself reported - a gate nobody
trusts is a gate nobody keeps.

THE SECOND QUESTION IT SETTLES. tts.py's external clone path passes
max_chunk_chars=200; generate_lora_voice passes no cap at all, and the local
Qwen3TTSModel.generate_voice_clone does no internal splitting (verified: no
chunk or split logic in its source). On grimgar03, 12.6% of segments exceed 200
characters and 5.2% exceed 500. If long segments truncate or drift, word-error
rate against length shows it, and --by-length reports exactly that. This turns
an argument about a magic number into a measurement.

WHAT THIS IS NOT. Word errors measure CONTENT, not quality. A segment can score
perfectly and still be flat, mispronounced, or in the wrong voice. This catches
dropped, repeated, hallucinated and truncated speech, which are the failures
that make an audiobook unusable rather than merely imperfect.

Idea credit: zeropointnine/tts-audiobook-tool (MIT), whose validator established
transcribe-back with a length-scaled threshold as the shape of this check. The
alignment here is an independent implementation over stdlib difflib. No code
taken; see THIRD_PARTY_NOTICES.md.
"""
import argparse, difflib, json, os, re, subprocess, sys, tempfile

REPO = "/home/fakemitch/pinokio/api/alexandria-audiobook2.git"
APP = REPO + "/app/"
sys.path.insert(0, APP)

LEDGER = REPO + "/ab_test_runtime/experiments"
WHISPER_BIN = REPO + "/whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL = REPO + "/models/whisper.cpp/ggml-small.en.bin"

STRICTNESS = {"low": (10, 1), "moderate": (10, 0), "high": (10, -1),
              "intolerant": (0, 0)}

ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven",
        "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
        "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
        "eighty", "ninety"]


def say_number(digits):
    """Digits as the words a reader would say them.

    Needed because ASR transcribes '25' as 'twenty five'. Getting this wrong
    does not fail loudly - it silently charges an error to every segment
    containing a number, which is enough false positives to make the whole gate
    look broken. Covers 0-999; anything longer is read digit by digit, which is
    what a reader does with a phone number and close enough for a year that the
    surrounding words carry the match.
    """
    n = int(digits)
    if n < 20:
        return [ONES[n]]
    if n < 100:
        return [TENS[n // 10]] + ([ONES[n % 10]] if n % 10 else [])
    if n < 1000:
        rest = say_number(str(n % 100)) if n % 100 else []
        return [ONES[n // 100], "hundred"] + rest
    # Four digits are almost always a YEAR in a book, and a narrator reads
    # 2016 as "twenty sixteen", not "two zero one six". Reading it digitwise
    # charged a false error to every copyright line - observed on the first
    # real run, where "2016" scored 4 errors against a correct rendition.
    if len(digits) == 4 and 1100 <= n <= 9999:
        high, low = n // 100, n % 100
        if 2000 <= n <= 2009:
            # "two thousand five", not "twenty oh five".
            return ["two", "thousand"] + (say_number(str(low)) if low else [])
        if low == 0:
            return say_number(str(high)) + ["hundred"]
        if low < 10:
            return say_number(str(high)) + ["oh"] + say_number(str(low))
        return say_number(str(high)) + say_number(str(low))
    # Longer runs are identifiers - ISBNs, phone numbers - and are read out
    # digit by digit, which is also what a narrator does.
    return [ONES[int(d)] for d in digits]


def compute_threshold(num_words, strictness="moderate"):
    """Word errors tolerated before a segment is a failure.

    Scales with length: a two-word segment gets no slack, a two-hundred word one
    gets twenty. A flat threshold would either fail every long segment on ASR
    noise or pass a short one that came out as gibberish.
    """
    if strictness not in STRICTNESS:
        raise ValueError(f"unknown strictness: {strictness!r}")
    per, offset = STRICTNESS[strictness]
    if not per:
        return 0
    return max(0, -(-num_words // per) + offset)


# Books use typographic apostrophes; ASR emits ASCII ones. Leaving U+2019
# unmapped split "you've" into "you"+"ve" against the transcript's single
# token, charging a false error to every contraction in the corpus. This was
# the single largest source of false failures in the first real run.
APOSTROPHES = {"’": "'", "‘": "'", "ʼ": "'", "´": "'"}


def words(text):
    """Comparable word tokens: case, punctuation and digit spellings removed.

    Without this the gate fires on correct audio, because ASR writes '25' as
    'twenty five' and drops the comma we sent it.
    """
    text = (text or "")
    for bad, good in APOSTROPHES.items():
        text = text.replace(bad, good)
    out = []
    for raw in re.findall(r"[\w']+", text.lower()):
        if raw.isdigit():
            out.extend(say_number(raw))
        else:
            out.append(raw.strip("'"))
    return [w for w in out if w]


# Below this, two words are different words. Above it, they are the same word
# heard imperfectly - "shinichirou"/"shinichiro", "isbns"/"isbn". Proper nouns
# and romanised Japanese are where an English ASR model is weakest, and
# charging those to the TTS would make the gate unusable on this corpus.
NEAR_MATCH = 0.85


def sounds_like(a, b):
    return difflib.SequenceMatcher(a=a, b=b).ratio() >= NEAR_MATCH


def word_errors(source, transcript):
    """(error count, source word count, per-error detail).

    Substitutions, deletions and insertions all count as one error each, which
    is standard WER accounting. difflib gives the alignment; the detail is kept
    because a gate that only reports a number cannot be debugged.
    """
    a, b = words(source), words(transcript)
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    errors, detail = 0, []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace" and (i2 - i1) == (j2 - j1):
            # Same-length substitution: forgive the pairs that are the same
            # word misheard, charge the rest. Done pairwise so one genuine
            # error inside a run is still counted.
            for x, y in zip(a[i1:i2], b[j1:j2]):
                if sounds_like(x, y):
                    continue
                errors += 1
                detail.append({"kind": "replace", "expected": x, "heard": y})
            continue
        errors += max(i2 - i1, j2 - j1)
        detail.append({"kind": tag, "expected": " ".join(a[i1:i2]),
                       "heard": " ".join(b[j1:j2])})
    return errors, len(a), detail


def is_non_speech(transcript):
    """whisper.cpp emits '* * *' (or nothing) when the audio is not speech.

    This is the strongest signal the gate can get and deserves its own flag
    rather than being counted as N deletions. It found a real defect on the
    first run: a 349-character table of contents produced 24.2 seconds of
    audio at normal level that transcribed to '* * * * * * * *' - the model
    vocalising rather than reading. A pure error count reports that as "52
    words missing", which reads like a truncation and is a different bug.
    """
    stripped = re.sub(r"[\s*\-_.]+", "", transcript or "")
    return not stripped


def validate(source, transcript, strictness="moderate"):
    """-> dict. `failed` is the shippable verdict for one segment."""
    errors, n, detail = word_errors(source, transcript)
    threshold = compute_threshold(n, strictness)
    heard = len(words(transcript))
    non_speech = n >= 3 and is_non_speech(transcript)
    return {"errors": errors, "words": n, "heard_words": heard,
            "threshold": threshold,
            # Non-speech output is a failure at any error budget.
            "failed": errors > threshold or non_speech,
            "non_speech": non_speech,
            # Losing the tail is the failure mode a listener notices most and
            # the one a pure error count under-weights, so it is flagged apart.
            "possible_truncation": n >= 8 and heard < n * 0.6,
            "detail": detail}


def transcribe(wav_path, binary=WHISPER_BIN, model=WHISPER_MODEL):
    """whisper.cpp -> flat text. Raises if the toolchain is not built."""
    for p in (binary, model):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} missing - build whisper.cpp and fetch its model first")
    with tempfile.TemporaryDirectory() as td:
        stem = os.path.join(td, "out")
        subprocess.run([binary, "-m", model, "-f", wav_path, "-otxt", "-np",
                        "-nt", "-of", stem],
                       check=True, capture_output=True)
        with open(stem + ".txt", encoding="utf-8") as fh:
            return fh.read().strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", required=True,
                    help='JSON list of {"text": ..., "wav": ...}')
    ap.add_argument("--strictness", default="moderate", choices=STRICTNESS)
    ap.add_argument("--by-length", action="store_true",
                    help="break results down by source length, to test the "
                         "200-character chunk cap")
    ap.add_argument("--out", default=LEDGER + "/tts_output_validation.json")
    args = ap.parse_args()

    segments = json.load(open(args.manifest))
    rows = []
    for i, seg in enumerate(segments, 1):
        try:
            heard = transcribe(seg["wav"])
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"  [{i}/{len(segments)}] transcribe failed: {e}")
            continue
        r = validate(seg["text"], heard, args.strictness)
        r.update({"wav": seg["wav"], "chars": len(seg["text"]),
                  "transcript": heard})
        rows.append(r)
        mark = "FAIL" if r["failed"] else "ok  "
        print(f"  [{i}/{len(segments)}] {mark} {r['errors']}/{r['threshold']} "
              f"errors, {r['chars']} chars")

    if not rows:
        print("nothing validated")
        return

    failed = sum(r["failed"] for r in rows)
    trunc = sum(r["possible_truncation"] for r in rows)
    nonspeech = sum(r.get("non_speech") for r in rows)
    total_err = sum(r["errors"] for r in rows)
    total_words = sum(r["words"] for r in rows)
    print(f"\n{len(rows)} segments, strictness={args.strictness}")
    print(f"  failed              {failed} ({failed / len(rows) * 100:.1f}%)")
    print(f"  possible truncation {trunc}")
    print(f"  NON-SPEECH output   {nonspeech}")
    print(f"  word error rate     {total_err / max(total_words, 1) * 100:.2f}%")

    if args.by_length:
        print("\n  by source length - the 200-character cap question")
        print(f"  {'bucket':>14}{'n':>6}{'WER':>9}{'failed':>9}{'truncated':>11}")
        buckets = [("<=200", 0, 200), ("201-500", 201, 500),
                   ("501-1000", 501, 1000), (">1000", 1001, 10 ** 9)]
        for label, lo, hi in buckets:
            sel = [r for r in rows if lo <= r["chars"] <= hi]
            if not sel:
                continue
            e = sum(r["errors"] for r in sel)
            w = sum(r["words"] for r in sel)
            f = sum(r["failed"] for r in sel)
            t = sum(r["possible_truncation"] for r in sel)
            print(f"  {label:>14}{len(sel):6}{e / max(w, 1) * 100:8.2f}%"
                  f"{f / len(sel) * 100:8.1f}%{t:11}")
        print("\n  If WER and truncation climb past 200 characters, the local "
              "LoRA path\n  needs the cap the external path already enforces. "
              "If they are flat, the\n  cap is a request-size convention and "
              "carries no quality claim.")

    json.dump({"strictness": args.strictness, "n": len(rows),
               "failed": failed, "truncation": trunc, "non_speech": nonspeech,
               "wer": total_err / max(total_words, 1), "rows": rows},
              open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
