#!/bin/bash
# Idempotent llama-server for a given LoRA adapter.
#
# OWNERSHIP IS THE BUG THIS FIXES. Previously one script started the server and
# a different one pkilled it on exit, so whether a server existed depended on
# which script happened to run last. A retry died with SERVER_GONE for exactly
# that reason on 2026-08-04.
#
# The inversion: the CALLER never kills the server. This script reuses a
# healthy one already serving the requested adapter, and replaces it only when
# something different is needed. A server therefore outlives the job that
# started it, which is what lets consecutive evals share one load.
#
# Usage:
#   ./ensure_llama_server.sh [adapter.gguf]
#
# Environment:
#   LLAMA_BIN     llama-server binary      (default ~/llama.cpp/build/bin/llama-server)
#   LLAMA_MODEL   base GGUF                (required)
#   LLAMA_PORT    port                     (default 8090)
#   LLAMA_CTX     context length           (default 32768)
#   LLAMA_LOG     server log path          (default ~/llama_server.log)
set -uo pipefail

ADAPTER="${1:-}"
BIN="${LLAMA_BIN:-$HOME/llama.cpp/build/bin/llama-server}"
MODEL="${LLAMA_MODEL:-}"
PORT="${LLAMA_PORT:-8090}"
CTX="${LLAMA_CTX:-32768}"
LOG="${LLAMA_LOG:-$HOME/llama_server.log}"
STAMP="$HOME/.llama_server_adapter"
URL="http://127.0.0.1:${PORT}/v1/models"

[ -x "$BIN" ] || { echo "ensure_llama_server: no binary at $BIN" >&2; exit 2; }
[ -n "$MODEL" ] || { echo "ensure_llama_server: LLAMA_MODEL is required" >&2; exit 2; }
[ -f "$MODEL" ] || { echo "ensure_llama_server: no model at $MODEL" >&2; exit 2; }
[ -n "$ADAPTER" ] && [ ! -f "$ADAPTER" ] && {
    echo "ensure_llama_server: no adapter at $ADAPTER" >&2; exit 2; }

# -f is load-bearing. llama-server answers 503 while it loads weights, and
# `curl -s` exits 0 on a 503 - which is precisely how an eval once fired at a
# server that was not up yet and died on HTTP 503. Without -f this whole
# readiness check is decorative.
ready() { curl -sf --max-time 5 "$URL" >/dev/null 2>&1; }

if ready && [ "$(cat "$STAMP" 2>/dev/null)" = "$ADAPTER" ]; then
    echo "ensure_llama_server: reusing server already serving ${ADAPTER:-base}"
    exit 0
fi

pkill -f "llama-server" 2>/dev/null
sleep 5
ARGS=(-m "$MODEL" --port "$PORT" --host 127.0.0.1 -ngl 99 -c "$CTX" --parallel 1)
[ -n "$ADAPTER" ] && ARGS+=(--lora "$ADAPTER")
"$BIN" "${ARGS[@]}" > "$LOG" 2>&1 &

for _ in $(seq 1 120); do
    if ready; then
        printf '%s' "$ADAPTER" > "$STAMP"
        echo "ensure_llama_server: ready for ${ADAPTER:-base}"
        exit 0
    fi
    sleep 10
done

# Fail loudly rather than letting the caller run against nothing.
echo "ensure_llama_server: SERVER_NEVER_READY for ${ADAPTER:-base}; see $LOG" >&2
exit 1
