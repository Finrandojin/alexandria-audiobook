#!/bin/bash
# Queued behind overnight_chain.sh for a longer unattended window.
#
# Waits on the PID of the running chain rather than a pgrep pattern - pgrep
# matches the shell that ran it, which is the mistake gpu_job.sh's header calls
# out and which cost two runs on 2026-08-04.
#
# Usage: ./extended_chain.sh [PID-to-wait-for]
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
L="$REPO/ab_test_runtime/logs"
PY="$REPO/app/env/bin/python"
export GPU_LOCK="${GPU_LOCK:-$HOME/.alexandria_gpu.lock}"
export GPU_QLOG="$L/gpu_jobq.log"
WAIT_PID="${1:-}"
mkdir -p "$L"

if [ -n "$WAIT_PID" ]; then
    echo "waiting on PID $WAIT_PID (overnight_chain)"
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi
echo "overnight chain finished $(date -u +%FT%TZ)"

cd "$REPO/app"
stage() {
    local name="$1"; shift
    echo ""
    echo "=== $name  $(date -u +%FT%TZ) ==="
    "$REPO/gpu_job.sh" "$name" "$@" > "$L/$name.log" 2>&1
    echo "  rc=$?"
    tail -8 "$L/$name.log" | sed 's/^/  /' | cut -c1-115
    return 0
}

# RETRAIN WITH AN HONEST SPLIT. Only possible now that train_lora.py prefers
# train/metadata.jsonl - before that every adapter trained on its own eval
# clips. Two questions in one run:
#
#   the five failures  did clean data + failed adapter mean training is
#                      stochastically unreliable, or is the dataset wrong in a
#                      way speaker consistency does not capture?
#   the three controls how big is the contamination bound on EVERY existing
#                      library number? Their contaminated-vs-honest gap is it.
#
# Writes to ab_test_runtime/retrain_honest/, never over lora_models/.
stage retrain_honest timeout 14400 "$PY" -u experiments/retrain_honest.py \
    --adapters husky_baritone_20s_m_anime velvety_mezzo_30s_f_gothic \
               warm_baritone_40s_m_fantasy breathy_tenor_18s_m_supernatural \
               warm_tenor_25s_m_military \
    --controls husky_tenor_30s_m_literary warm_baritone_40s_m_2 warm_tenor_20s_m

echo ""
echo "EXTENDED CHAIN DONE $(date -u +%FT%TZ)"
