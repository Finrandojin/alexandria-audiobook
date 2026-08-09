#!/bin/bash
# Overnight: answer goals 5.3 and 3.1 in one run, on the shipped model.
#
# WHY ONE RUN ANSWERS BOTH. 5.3 has NO BASELINE because the only three-pass
# comparison on disk recorded timing but never accuracy, and its single arm
# failed outright on grimgar03 and index18. 3.1 is OPEN because no saved book
# was ever generated with qwen3-14b, so its failure figures come from a model
# that does not ship. Generating all four books through both arms produces the
# accuracy comparison 5.3 wants AND a chunk-completion record attributable to
# qwen3-14b, which is exactly what 3.1 is missing.
#
# THE MODEL SETUP, AND WHY IT IS NOT THE DEFAULT ONE.
#   - app/config.json pointed at qwen2.5-14b, the model 3.1 explicitly says is
#     not the shipped path. It now names qwen3-14b; the original is backed up
#     at ab_test_runtime/logs/config.json.pre_qwen3_backup and RESTORED at the
#     end of this script, including on failure.
#   - llama-server runs with --chat-template-kwargs '{"enable_thinking":false}'.
#     Qwen3 is a thinking model: asked for 16 tokens it spent all of them on
#     reasoning and returned empty content, which the pipeline would see as a
#     failed chunk. Disabling thinking server-side means no app code changes
#     and no token budget spent on reasoning. Verified before launching:
#     content 'ready', reasoning absent, finish_reason stop.
#
# WHAT WOULD MAKE THIS RUN WORTHLESS. If the server dies, every chunk fails and
# the morning shows a 0% completion rate that says nothing about qwen3-14b. So
# the endpoint is re-checked before the long stage starts, and the run refuses
# rather than producing a number that would be read as a model result.
set -uo pipefail
REPO=/home/fakemitch/pinokio/api/alexandria-audiobook2.git
L="$REPO/ab_test_runtime/logs"
PY="$REPO/app/env/bin/python"
BACKUP="$L/config.json.pre_qwen3_backup"
export GPU_LOCK="${GPU_LOCK:-$HOME/.alexandria_gpu.lock}"
export GPU_QLOG="$L/gpu_jobq.log"
mkdir -p "$L"
cd "$REPO/app"

restore_config() {
    if [ -f "$BACKUP" ]; then
        cp "$BACKUP" "$REPO/app/config.json"
        echo "restored app/config.json from backup"
    fi
}
# Runs on normal exit, on error, and on Ctrl-C. Leaving the config pointing at
# a model whose server is gone would break the app for the next person to open
# it, which is a worse outcome than the run not finishing.
trap restore_config EXIT INT TERM

echo "=== endpoint check $(date -u +%FT%TZ) ==="
if ! curl -s -m 20 http://127.0.0.1:8090/v1/models | grep -q qwen3; then
    echo "ABORT: no qwen3 server on 8090 - a run without it measures nothing"
    exit 1
fi
echo "  qwen3-14b responding"

# 4 books x 2 arms. The previous partial run took 103 and 177 minutes for the
# single arm on two books, so this is a genuinely long job; the timeout is
# per-book inside the harness, not for the whole chain.
echo ""
echo "=== three_pass_vs_single  $(date -u +%FT%TZ) ==="
"$REPO/gpu_job.sh" three_pass_qwen3 timeout 72000 "$PY" -u experiments/three_pass_vs_single.py \
    --books grimgar03 index18 mushoku16 owarimonogatari3 \
    --out "$REPO/ab_test_runtime/experiments/three_pass_vs_single_qwen3.json" \
    > "$L/three_pass_qwen3.log" 2>&1
echo "  rc=$?"
tail -12 "$L/three_pass_qwen3.log" | sed 's/^/  /' | cut -c1-115

# Re-read chunk completion including whatever the run above just wrote, so the
# morning has goal 3.1's number attributable to a named model.
echo ""
echo "=== chunk completion re-read  $(date -u +%FT%TZ) ==="
"$REPO/gpu_job.sh" chunk_recount "$PY" -u experiments/chunk_completion.py \
    --scripts "$REPO/ab_test_runtime/three_pass_vs_single" \
    --out "$REPO/ab_test_runtime/experiments/chunk_completion_qwen3.json" \
    > "$L/chunk_recount.log" 2>&1
echo "  rc=$?"
tail -14 "$L/chunk_recount.log" | sed 's/^/  /' | cut -c1-115

echo ""
echo "OVERNIGHT DONE $(date -u +%FT%TZ)"
