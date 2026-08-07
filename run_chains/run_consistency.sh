#!/bin/bash
set -uo pipefail
REPO=/home/fakemitch/pinokio/api/alexandria-audiobook2.git
cd "$REPO/app"
timeout 10800 "$REPO/app/env/bin/python" -u experiments/dataset_speaker_consistency.py \
  > "$REPO/ab_test_runtime/logs/dataset_consistency.log" 2>&1
echo "rc=$? $(date -u +%FT%TZ)"
