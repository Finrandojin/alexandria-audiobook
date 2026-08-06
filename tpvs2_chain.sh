#!/bin/bash
# The three-pass comparison, with a server this job owns end to end.
#
# REPLACES an ad-hoc /tmp version that made two of the three mistakes
# gpu_job.sh's header calls out:
#
#   1. It waited with `while pgrep -f relaunch_chain.sh`. pgrep matches on a
#      pattern, which is a guess about what else is running; this waits on the
#      actual PID instead, passed in, and confirms the completion marker.
#   2. It started llama-server with nohup and never stopped it. The model is
#      12 GB on a 15.9 GB card, so it stayed resident after the run and would
#      have blocked every later TTS job. gpu_job.sh deliberately does not
#      manage servers - "those belong to the job" - so the job that starts the
#      server is the one that stops it, in the same shell, via a trap.
#
# Usage: ./tpvs2_chain.sh [PID-to-wait-for]
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
L="$REPO/ab_test_runtime/logs"
export GPU_LOCK="${GPU_LOCK:-$HOME/.alexandria_gpu.lock}"
export GPU_QLOG="${GPU_QLOG:-$L/gpu_jobq.log}"
WAIT_PID="${1:-}"

# Wait for the TTS arms: llama.cpp at -ngl 99 and Qwen3-TTS will not fit on
# this card together. The GPU lock alone is not enough, because relaunch_chain
# takes and releases it once per stage - tpvs2 would slip in between two arms.
if [ -n "$WAIT_PID" ]; then
    echo "waiting on PID $WAIT_PID (the TTS arms)"
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi
echo "TTS arms finished $(date -u +%FT%TZ)"
grep -q 'ALL ARMS DONE' "$L/relaunch_chain.log" 2>/dev/null \
    && echo "  completion marker present" \
    || echo "  WARNING: no ALL ARMS DONE marker; the arms may have died early"

cd "$REPO"
"$REPO/gpu_job.sh" three_pass_vs_single_v2 bash -c '
  set -uo pipefail
  REPO='"$REPO"'
  # This shell owns the server for its whole life and kills it on any exit
  # path - success, failure, or interrupt - so nothing is left on the card.
  "$REPO/start_llama_server.sh" || exit 2
  SRV=$(pgrep -f "llama-server.*--port 8090" | head -1)
  trap "[ -n \"$SRV\" ] && kill $SRV 2>/dev/null; sleep 2" EXIT INT TERM
  echo "server pid $SRV, owned by this job"
  cd "$REPO/app"
  "$REPO/app/env/bin/python" -u experiments/three_pass_vs_single.py
' > "$L/three_pass_vs_single_v2.log" 2>&1
rc=$?

echo "three_pass_vs_single rc=$rc"
[ $rc -ne 0 ] && echo "  rc!=0 means it compared nothing or a book failed"
tail -14 "$L/three_pass_vs_single_v2.log"

# The server must not outlive this job. Verified, not assumed.
if pgrep -f 'llama-server.*--port 8090' >/dev/null; then
    echo "WARNING: llama-server still up after the job; killing it"
    pkill -f 'llama-server.*--port 8090'
fi
exit $rc
