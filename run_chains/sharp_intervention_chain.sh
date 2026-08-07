#!/bin/bash
# The sharp version: three arms, including a reference from a DIFFERENT
# narrator. The two-arm run could only contrast 0.873 against 0.815 - the most
# and least typical clip of the same clean speaker - which is far smaller than
# the failure being explained. A reference scoring -0.026 against its dataset
# is not an atypical clip, it is the wrong person, and only a foreign clip
# reproduces that.
#
# Also retrains the seven adapters whose shipped reference is mismatched,
# using the medoid. That is the practical payoff: does fixing the reference
# actually recover them?
set -uo pipefail
REPO=/home/fakemitch/pinokio/api/alexandria-audiobook2.git
export GPU_LOCK="${GPU_LOCK:-$HOME/.alexandria_gpu.lock}"
export GPU_QLOG="$REPO/ab_test_runtime/logs/gpu_jobq.log"
while pgrep -f 'reference_interventio[n].py' >/dev/null; do sleep 30; done
cd "$REPO/app"

# 1. CAUSAL TEST with a real contrast. A female narrator's clip against a male
#    narrator's dataset, so "wrong person" is unambiguous.
"$REPO/gpu_job.sh" sharp_intervention \
  timeout 10800 "$REPO/app/env/bin/python" -u experiments/reference_intervention.py \
    --adapter husky_baritone_20s_m_anime \
    --foreign-from crisp_mezzo_30s_f \
    --out "$REPO/ab_test_runtime/experiments/reference_intervention_sharp.json" \
  > "$REPO/ab_test_runtime/logs/sharp_intervention.log" 2>&1
echo "sharp rc=$?"
grep -E 'medoid:|worst:|foreign:|VERDICT|delta' "$REPO/ab_test_runtime/logs/sharp_intervention.log" | tail -6

# 2. PRACTICAL PAYOFF: retrain the mismatched-reference adapters with a proper
#    reference. train_lora falls back to the first training clip when no
#    ref.wav is present, so retrain_honest already supplies a different
#    reference than the shipped one - this measures whether that is enough.
"$REPO/gpu_job.sh" retrain_bad_refs \
  timeout 14400 "$REPO/app/env/bin/python" -u experiments/retrain_honest.py \
    --adapters husky_baritone_20s_m_anime warm_baritone_40s_m_fantasy \
               velvety_mezzo_30s_f_gothic silky_baritone_45s_m \
               warm_tenor_20s_m warm_baritone_50s_m_gothic \
               husky_baritone_40s_m_military \
    --out "$REPO/ab_test_runtime/experiments/retrain_bad_refs.json" \
  > "$REPO/ab_test_runtime/logs/retrain_bad_refs.log" 2>&1
echo "retrain rc=$?"
grep -E 'held-out ecapa' "$REPO/ab_test_runtime/logs/retrain_bad_refs.log" | tail -8
echo "SHARP CHAIN DONE $(date -u +%FT%TZ)"
