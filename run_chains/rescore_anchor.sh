#!/bin/bash
set -uo pipefail
REPO=/home/fakemitch/pinokio/api/alexandria-audiobook2.git
cd "$REPO/app"
for t in aishell3 kokoro ljspeech; do
  echo "=== $t $(date -u +%H:%M:%S) ==="
  timeout 7200 "$REPO/app/env/bin/python" -u experiments/ljspeech_score.py \
    --generated "$REPO/ab_test_runtime/experiments/${t}_generate.json" \
    --out "$REPO/ab_test_runtime/experiments/${t}_score.json" \
    > "$REPO/ab_test_runtime/logs/rescore_${t}.log" 2>&1
  echo "  rc=$?"
  grep -E 'human_vs_human|ANCHOR INVALID' "$REPO/ab_test_runtime/logs/rescore_${t}.log" | tail -2
done
echo "RESCORE DONE $(date -u +%FT%TZ)"
