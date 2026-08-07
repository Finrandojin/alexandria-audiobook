#!/bin/bash
# The large-v3 arm of the ASR benchmark.
#
# WHY. Every CJK number in asr_backends__{kokoro,aishell3}.json carries the same
# caveat: they are `base`-size Whisper. 32% JA and 44% ZH CER may be the model
# size rather than the backend, and SenseVoice's 4x Chinese advantage may shrink
# or vanish against large-v3. Until this runs, "Whisper is poor at CJK" is a
# statement about base, not about Whisper.
#
# The English arm is included as the control. If large-v3 does NOT improve
# English much - it is already at 3.0% - while moving CJK a lot, that is
# evidence the gap was capacity on the harder languages rather than anything
# about the harness.
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
L="$REPO/ab_test_runtime/logs"
MODEL="$REPO/whisper.cpp/models/ggml-large-v3.bin"
PY="$REPO/app/env/bin/python"
export GPU_LOCK="${GPU_LOCK:-$HOME/.alexandria_gpu.lock}"
export GPU_QLOG="${GPU_QLOG:-$L/gpu_jobq.log}"
mkdir -p "$L"

# Wait for the download rather than racing it. A truncated GGUF fails in a way
# that looks like a model problem, so check size stability, not just existence.
echo "waiting for $MODEL"
last=0
for _ in $(seq 1 240); do
    [ -f "$MODEL" ] || { sleep 10; continue; }
    now=$(stat -c %s "$MODEL")
    if [ "$now" = "$last" ] && [ "$now" -gt 2000000000 ]; then
        echo "model ready: $((now/1048576)) MB"
        break
    fi
    last=$now
    sleep 10
done
if [ ! -f "$MODEL" ] || [ "$(stat -c %s "$MODEL")" -lt 2000000000 ]; then
    echo "MODEL_NEVER_ARRIVED" >&2
    exit 2
fi

cd "$REPO/app"
for arm in "ljspeech en" "kokoro ja" "aishell3 zh"; do
    # zsh does not word-split unquoted parameters; this script runs under bash
    # and does, but set -- is explicit and survives either shell.
    set -- $arm
    book=$1; lang=$2
    echo "=== $book ($lang) large-v3 $(date -u +%H:%M:%S) ==="
    "$REPO/gpu_job.sh" "asr_large_${book}" \
        timeout 5400 "$PY" -u experiments/asr_backends.py \
            --build "$REPO/ab_test_runtime/${book}_eval/build.json" \
            --lang "$lang" --backends whisper_cpp --limit 50 --align-clips 10 \
            --whisper-cpp-model "$MODEL" \
            --out "$REPO/ab_test_runtime/experiments/asr_backends_large__${book}.json" \
        > "$L/asr_large_${book}.log" 2>&1
    echo "  rc=$?"
    grep -E 'WER|median' "$L/asr_large_${book}.log" | tail -2
done
echo "ASR LARGE DONE $(date -u +%FT%TZ)"
