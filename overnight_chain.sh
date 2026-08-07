#!/bin/bash
# Queue for an ~8 hour window with the machine free.
#
# Ordered so the thing that has NEVER been measured runs first and on the most
# compute. If the window is cut short, the drift result still lands.
#
# Every stage goes through gpu_job.sh, so nothing overlaps on the card, and a
# stage that fails is recorded and skipped rather than stopping the queue - the
# point of a long unattended run is to come back to several answers, not to one
# failure with everything behind it unstarted.
set -uo pipefail
REPO=/home/fakemitch/pinokio/api/alexandria-audiobook2.git
L="$REPO/ab_test_runtime/logs"
PY="$REPO/app/env/bin/python"
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
    tail -6 "$L/$name.log" | sed 's/^/  /' | cut -c1-110
    return 0          # never let one stage strand the rest
}

# 1. DRIFT — the headline. Nothing in this project has ever measured whether a
#    voice survives a book-length run; every existing number is one line.
#    Three adapters so a result is not one adapter's quirk: the best scorer, a
#    mid scorer, and one whose training data is clean but whose adapter failed.
for a in husky_tenor_30s_m_literary warm_mezzo_30s_f_fantasy_2 husky_baritone_20s_m_anime; do
    stage "drift_$a" timeout 10800 "$PY" -u experiments/voice_drift.py \
        --adapter "$a" --lines 400
done

# 2. REF-CLIP SWEEP — three of the five worst adapters have a reference clip
#    that is not their own narrator, and one file sets speaker identity for all
#    200 training samples. One counter-example (warm_tenor_20s_m) means this
#    needs the full library before it is a finding. Cheap: no generation.
stage ref_clip_sweep timeout 7200 "$PY" -u experiments/ref_clip_match.py

# 3. LONGER FIDELITY PASS — the library scores used 4 val clips each. Ten gives
#    a median worth ranking on, and the earlier run is contaminated anyway
#    (adapters trained on their own val split), so this is about precision of
#    the ranking, not about fixing the bound.
stage fidelity_10 timeout 21600 "$PY" -u experiments/library_voice_fidelity.py \
    --lines 10 --out "$REPO/ab_test_runtime/experiments/library_voice_fidelity_n10.json"

echo ""
echo "OVERNIGHT CHAIN DONE $(date -u +%FT%TZ)"
