#!/bin/bash
# Overnight queue, ~4h. Two jobs, in this order for a reason.
#
# 1. GATE THE PROMOTION CANDIDATES. Seven retrained adapters beat their shipped
#    counterparts on held-out ECAPA. That score came from the retrain runs
#    themselves, so it is the number the training loop reported about its own
#    output. The gate is an independent re-measurement, and it is the thing
#    standing between here and overwriting seven shipped voices. It runs first
#    because it unblocks a decision, and it is cheap.
#
# 2. RE-MEASURE GOALS 2.5 AND 2.6 AT PROPER SAMPLE SIZE. Both are currently
#    recorded OPEN on twelve clips per language. Twelve is thin enough that the
#    finding could be sample size rather than signal, and an OPEN goal asserts a
#    failure - a claim that should not rest on twelve samples. At 100 clips
#    either the gap survives, and it is real, or it moves, and we retract a
#    false OPEN. Both outcomes are worth the card time; that is what makes this
#    worth running rather than merely unrun.
#
# Ordering: the gate first so a morning with a dead card still has the decision.
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
    echo "  rc=$?"
    tail -6 "$L/$name.log" | sed 's/^/  /' | cut -c1-115
    return 0          # a failed stage must not strand the rest of the night
}

# --- 1. Gate ------------------------------------------------------------
# Each adapter is gated against its OWN dataset. The candidates live under
# retrain_honest/; the three extra directories under reference_intervention/
# are the medoid/worst/foreign arms of the causal experiment and are
# deliberately excluded - they were built to be compared, not shipped.
CANDIDATES="husky_baritone_20s_m_anime warm_baritone_40s_m_fantasy \
husky_baritone_40s_m_military warm_tenor_25s_m_military silky_baritone_30s_m \
breathy_tenor_18s_m_supernatural warm_mezzo_30s_f_fantasy_2"

for name in $CANDIDATES; do
    base="$REPO/ab_test_runtime/retrain_honest/$name"
    if [ ! -d "$base/adapter" ]; then
        echo "  SKIP $name - no adapter at $base/adapter"
        continue
    fi
    stage "gate_$name" timeout 3600 "$PY" -u experiments/verify_adapter_identity.py \
        --adapter "$base/adapter" \
        --dataset "$base/data" \
        --lines 10 \
        --out "$REPO/ab_test_runtime/experiments/gate_promote__$name.json"
done

# --- 2. Pitch range and voice quality at n=100 --------------------------
# --lines 100 against the twelve-clip runs already on disk. voice_compare_view
# computes both the pitch statistics behind 2.5 and the quality measures behind
# 2.6 in one pass, so this is one job answering two goals.
stage pitch_quality_n100 timeout 21600 "$PY" -u experiments/voice_compare_view.py \
    --lines 100 \
    --out "$REPO/ab_test_runtime/experiments/voice_compare_n100.json"

echo ""
echo "OVERNIGHT DONE $(date -u +%FT%TZ)"
