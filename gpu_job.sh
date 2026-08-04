#!/bin/bash
# Serialise GPU work and make its failures loud.
#
# WHY THIS EXISTS. Long experiment runs get chained - train, then evaluate,
# then serve and evaluate again - and every chain written ad hoc so far
# repeated the same three mistakes:
#
#   1. `while pgrep -f <something>; do sleep; done` as a wait. That guesses at
#      what else is running and races between "is it running?" and "start
#      mine", and pgrep also matches the shell that ran it.
#   2. Ambiguous server ownership. One script started llama-server and a
#      different one pkilled it on exit, so whether a server existed depended
#      on script ordering.
#   3. No mutual exclusion. Nothing prevented two GPU jobs overlapping except
#      the author sequencing them correctly by hand.
#
# Two runs died on that on 2026-08-04 - an eval that fired at a server still
# loading its weights, and a retry that found the server killed by its parent.
# Neither failure was scientific, and both cost GPU hours.
#
# Usage:
#   ./gpu_job.sh <name> <command...>
#
# Environment:
#   GPU_LOCK   lock file path      (default ~/.gpu.lock)
#   GPU_QLOG   queue log path      (default ~/gpu_jobq.log)
#
# Guarantees:
#   - exactly one job holds the GPU at a time, via flock on a real file. A
#     second invocation BLOCKS rather than racing.
#   - start, end and exit code are appended to the queue log, so what ran and
#     what it returned survives without terminal scrollback.
#   - a non-zero exit is recorded with a FAILED marker and propagated, instead
#     of being swallowed by the next command in a chain.
#
# It deliberately does NOT manage servers, retry, or interpret results. Those
# belong to the job.
set -uo pipefail

LOCK="${GPU_LOCK:-$HOME/.gpu.lock}"
QLOG="${GPU_QLOG:-$HOME/gpu_jobq.log}"

NAME="${1:-}"
[ -z "$NAME" ] && { echo "usage: gpu_job.sh <name> <command...>" >&2; exit 2; }
shift
[ "$#" -eq 0 ] && { echo "gpu_job.sh: no command given" >&2; exit 2; }

stamp() { date -u +%FT%TZ; }

echo "$(stamp) QUEUED   $NAME" >> "$QLOG"
exec 9>"$LOCK" || {
    echo "$(stamp) LOCK_FAILED $NAME (cannot open $LOCK)" >> "$QLOG"
    echo "gpu_job: cannot open lock file $LOCK" >&2
    exit 4
}
# The lock MUST be a gate, not a suggestion. `set -e` is deliberately not on
# here (the wrapped command's exit code has to survive), so an unchecked
# `flock` would fall through to running the command on failure - defeating the
# entire purpose of this script. A concurrent GPU job is exactly what cost 42
# minutes of training on 2026-08-04.
if ! flock 9; then
    echo "$(stamp) LOCK_FAILED $NAME (flock failed)" >> "$QLOG"
    echo "gpu_job: failed to acquire GPU lock; refusing to run $NAME" >&2
    exit 4
fi
echo "$(stamp) START    $NAME" >> "$QLOG"

"$@"
rc=$?

if [ "$rc" -eq 0 ]; then
    echo "$(stamp) OK       $NAME" >> "$QLOG"
else
    # Loud on purpose. A chained job that fails quietly gets read as a result.
    echo "$(stamp) FAILED   $NAME rc=$rc" >> "$QLOG"
    echo "gpu_job: $NAME FAILED rc=$rc" >&2
fi
exit "$rc"
