#!/bin/bash
# Close goal 2.6's Chinese cell by re-running the arm on a typical speaker.
#
# THE FINDING THIS TESTS. Chinese LoRA measures HNR 1.17x its narrator. Clip
# length was ruled out (moves it 0.0025) and the corpus was ruled out
# (AISHELL-3 medians 12.02 dB, CLEANER than LJSpeech's 10.83). What remains is
# the eval speaker herself: SSB1585 sits at 9.39 dB, 2.63 below her corpus
# median, at the 8th percentile of 40 sampled speakers.
#
# SSB0748 is the replacement, chosen as the closest speaker to the corpus
# median: 12.025 dB, +0.005 off. Nothing else about the pipeline changes.
#
# THE PREDICTION, WRITTEN DOWN BEFORE THE RUN. If the ratio is a property of
# speaker selection, SSB0748's arm lands near 0.93x - inside the 0.85-1.15
# band - because the generated side already measures 11.17 dB against a corpus
# that medians 12.02. If it lands near 1.17x again, speaker selection was the
# wrong explanation too and the adapter is implicated after all. Recording the
# prediction here so the result cannot be re-narrated afterwards either way.
set -uo pipefail
REPO=/home/fakemitch/pinokio/api/alexandria-audiobook2.git
L="$REPO/ab_test_runtime/logs"
PY="$REPO/app/env/bin/python"
SPK=SSB0748
export GPU_LOCK="${GPU_LOCK:-$HOME/.alexandria_gpu.lock}"
export GPU_QLOG="$L/gpu_jobq.log"
mkdir -p "$L"
cd "$REPO/app"

stage() {
    local name="$1"; shift
    echo ""
    echo "=== $name  $(date -u +%FT%TZ) ==="
    "$REPO/gpu_job.sh" "$name" "$@" > "$L/$name.log" 2>&1
    local rc=$?
    echo "  rc=$rc"
    tail -5 "$L/$name.log" | sed 's/^/  /' | cut -c1-115
    # Unlike the overnight chains, a failure here MUST stop the run: every
    # stage consumes the previous one's output, so continuing past a failure
    # would score whatever stale artifact was on disk and report it as new.
    if [ $rc -ne 0 ]; then
        echo "STOPPING - $name failed"
        exit $rc
    fi
}

stage cn_prepare timeout 3600 "$PY" -u experiments/aishell3_prepare.py \
    --speaker "$SPK" \
    --out "$REPO/ab_test_runtime/experiments/aishell3_${SPK}_prepare.json"

stage cn_build timeout 3600 "$PY" -u experiments/aishell3_build.py \
    --out "$REPO/ab_test_runtime/aishell3_${SPK}_eval"

echo ""
echo "PREPARED $(date -u +%FT%TZ) - generation and scoring are the next"
echo "stages and need the adapter trained on $SPK first."
