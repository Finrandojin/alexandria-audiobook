#!/bin/bash
# Overnight queue, ~10 hours.
#
# The library retraining avenue is exhausted: 14 of 15 broken adapters have
# been retrained and the one remaining has a mixed-speaker dataset, where
# retraining is measured as pointless (+0.009, -0.076). So the night goes to
# the one question this project has never been able to answer.
#
# ORDER MATTERS. The cheap bookkeeping runs first so that if anything goes
# wrong overnight there is still a correct library ranking in the morning.
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

# 1. RE-SCORE THE LIBRARY (~1.5h). Thirteen adapters were rebuilt today and the
#    published ranking still reflects their broken scores. This is the number
#    anyone would quote, so it should be the true one.
stage library_fidelity_post_fix timeout 21600 "$PY" -u experiments/library_voice_fidelity.py \
    --lines 10 \
    --out "$REPO/ab_test_runtime/experiments/library_voice_fidelity_postfix.json"

# 2 & 3. DRIFT AT BOOK LENGTH (~4h each).
#
# THE OPEN QUESTION. Every voice measurement in this project was a single line
# until yesterday, when 400 consecutive lines showed a small downward drift:
# -0.018 on husky_tenor_30s_m_literary and -0.050 on warm_mezzo_30s_f_fantasy_2,
# with pitch rising while vocal tract length and HNR held. A real audiobook is
# five to twenty thousand lines, and the 400-line result explicitly does not
# license a claim about 5000: whether drift is linear, plateaus, or accelerates
# is unmeasured.
#
# 2000 lines is five times the previous run and enough to separate those three
# shapes. The same two adapters are used deliberately - extending a measurement
# is worth more than starting a third one, and voice_drift skips lines it has
# already generated, so the first 400 of each are already on disk.
stage drift_2000_husky_tenor timeout 36000 "$PY" -u experiments/voice_drift.py \
    --adapter husky_tenor_30s_m_literary --lines 2000 \
    --out "$REPO/ab_test_runtime/experiments/voice_drift_2000__husky_tenor_30s_m_literary.json"

stage drift_2000_warm_mezzo timeout 36000 "$PY" -u experiments/voice_drift.py \
    --adapter warm_mezzo_30s_f_fantasy_2 --lines 2000 \
    --out "$REPO/ab_test_runtime/experiments/voice_drift_2000__warm_mezzo_30s_f_fantasy_2.json"

# 4. A THIRD ADAPTER, NOT PREVIOUSLY DRIFT-TESTED (~4h). The first two extend
#    existing 400-line measurements; this one asks whether the shape holds on
#    an adapter that has never been run long. If drift is a property of long
#    generation rather than of particular voices - which is what all three
#    moving the same direction suggested - this should look like the others.
#    If it does not, "drift" is really "these two adapters drift".
stage drift_2000_warm_baritone timeout 36000 "$PY" -u experiments/voice_drift.py \
    --adapter warm_baritone_40s_m_2 --lines 2000 \
    --out "$REPO/ab_test_runtime/experiments/voice_drift_2000__warm_baritone_40s_m_2.json"

echo ""
echo "OVERNIGHT DONE $(date -u +%FT%TZ)"
