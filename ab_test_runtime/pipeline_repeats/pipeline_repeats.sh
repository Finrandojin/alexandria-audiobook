#!/usr/bin/env bash
# Measure the PIPELINE's run-to-run variance properly.
#
# Section 7.0 reported 17.9% of per-line answers changing between two runs, and
# I derived a "+/-4.9 point noise band" from it. The reviewer was right to
# reject that: two runs cannot estimate a run-level distribution, and treating
# 291 gold lines as independent replicates ignores that they share one book, one
# prompt set and one run state.
#
# This runs the SAME model on the SAME book N times and scores each, so the
# spread across runs is measured rather than inferred. It is the number several
# ledger comparisons depend on - section 6.3's 5.5-point decomposition/pipeline
# gap in particular - and nothing else in the project supplies it.
#
# Deliberate deviation from the overnight runs: --pass2-on-exhaustion fallback
# rather than fail. `fail` stops pass 2 at the first unrecoverable batch, which
# happened at 99.96% in one run and 82% in another - different denominators,
# useless for comparing runs to each other. `fallback` completes every run.
# These are therefore comparable AMONG THEMSELVES and not directly to the
# overnight `fail` artifacts.
set -u
ROOT=/home/fakemitch/pinokio/api/alexandria-audiobook2.git
APP="$ROOT/app"; D="$ROOT/ab_test_runtime/pipeline_repeats"
OUT="$ROOT/ab_test_runtime/results/overnight_20260726-185022/day"
LOG="$D/repeats.log"
LMS=/home/fakemitch/.lmstudio/bin/lms
N=8

# Do not start while the grammar test still holds the card.
while ! grep -qE "local queue done|grammar test (exit|skipped)" "$OUT/local_queue.log" 2>/dev/null; do
  sleep 60
done
echo "=== local queue clear, starting $N pipeline repeats $(date -Is) ===" >> "$LOG"

"$LMS" unload --all >> "$LOG" 2>&1; sleep 5
cd "$APP" && ./env/bin/python - qwen/qwen3-14b <<'PY' >> "$LOG" 2>&1
import sys
from lmstudio_settings import ensure_ideal_settings
_, s, msg = ensure_ideal_settings("local", "http://localhost:1234/v1", sys.argv[1])
print(msg); raise SystemExit(0 if s.get("loaded") else 1)
PY
if [ $? -ne 0 ]; then echo "=== model would not load, aborting ===" >> "$LOG"; exit 1; fi

for i in $(seq 1 $N); do
  echo "=== repeat $i/$N start $(date -Is) ===" >> "$LOG"
  cd "$APP" && ALEXANDRIA_DATA_DIR="$D" ./env/bin/python three_pass_generate.py \
    "$ROOT/ab_test_runtime/results/matrix_20260725-115148/inputs/grimgar03.txt" \
    --reasoning-effort none --pass2-on-exhaustion fallback \
    --output "$D/run$i.json" >> "$D/run$i.log" 2>&1
  echo "=== repeat $i/$N exit=$? $(date -Is) ===" >> "$LOG"
done
echo "=== all repeats done $(date -Is) ===" >> "$LOG"

# Score every completed run against the gold set and report the spread.
cd "$APP" && ./env/bin/python "$D/score_repeats.py" >> "$LOG" 2>&1
echo "=== scored $(date -Is) ===" >> "$LOG"
