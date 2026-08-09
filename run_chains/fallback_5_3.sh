#!/bin/bash
# Goal 5.3's second scored book, with production exhaustion semantics.
#
# WHY THIS RERUN. Last night produced ONE scored comparison, not four.
# owarimonogatari3's three-pass arm died after 38m on a single one-entry batch
# it could not attribute, and the harness drops a book whole when one arm
# fails - correctly, since scoring one arm against gold while the other failed
# would publish a comparison that is not one.
#
# The abort itself was not a bug: three_pass_generate defaults to
# on_exhaustion='fail' so a failure rate stays visible. But 5.3 asks which
# METHOD is more accurate, and production runs 'fallback', where an
# unresolvable span becomes UNKNOWN instead of killing a 4454-entry book. That
# is the arm a user actually gets, so it is the arm to score.
#
# Only owarimonogatari3 runs. mushoku16 already has both arms from last night
# and its three-pass never exhausted, so fallback would change nothing there -
# rerunning it would cost two hours to reproduce a number already on disk.
# grimgar03 and index18 are excluded for reasons fallback cannot fix: the
# former fails chunk 1 of its SINGLE arm on qwen3-14b, the latter is refused by
# the source gate over 6,662 U+FFFD before any model sees it.
set -uo pipefail
REPO=/home/fakemitch/pinokio/api/alexandria-audiobook2.git
L="$REPO/ab_test_runtime/logs"
PY="$REPO/app/env/bin/python"
BACKUP="$L/config.json.pre_fallback_backup"
export GPU_LOCK="${GPU_LOCK:-$HOME/.alexandria_gpu.lock}"
export GPU_QLOG="$L/gpu_jobq.log"
cd "$REPO/app"

restore_config() {
    [ -f "$BACKUP" ] && cp -f "$BACKUP" "$REPO/app/config.json" && \
        echo "restored app/config.json"
}
trap restore_config EXIT INT TERM

if ! curl -s -m 20 http://127.0.0.1:8090/v1/models | grep -q qwen3; then
    echo "ABORT: no qwen3 server on 8090"; exit 1
fi
echo "endpoint ok $(date -u +%FT%TZ)"

"$REPO/gpu_job.sh" tpvs_fallback timeout 72000 "$PY" -u experiments/three_pass_vs_single.py \
    --books owarimonogatari3 \
    --pass2-on-exhaustion fallback \
    --reuse-complete --model qwen3-14b \
    --out "$REPO/ab_test_runtime/experiments/three_pass_vs_single_fallback.json" \
    > "$L/tpvs_fallback.log" 2>&1
echo "rc=$?"
tail -12 "$L/tpvs_fallback.log" | sed 's/^/  /' | cut -c1-115
echo "DONE $(date -u +%FT%TZ)"
