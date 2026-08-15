#!/bin/bash
# Re-run identity gates after repairing the medoid audio/transcript pairing.
# Serialized intentionally: each gate loads Qwen3-TTS and ECAPA, so parallel
# gates would compete for VRAM and invalidate the safety assumptions.
set -uo pipefail

REPO=/home/fakemitch/pinokio/api/alexandria-audiobook2.git
PY="$REPO/app/env/bin/python"
EXP="$REPO/ab_test_runtime/experiments"
WORK="$REPO/ab_test_runtime/decontaminate"
LOG="$REPO/ab_test_runtime/logs/regate_reference_text"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
BACKUP="$EXP/gate_reference_text_backup_$STAMP"
QUEUE="$REPO/ab_test_runtime/regate_reference_text.tsv"
mkdir -p "$LOG" "$BACKUP"

cd "$REPO"
"$PY" - "$QUEUE" <<'PYEOF'
import glob, json, os, re, sys
repo = os.getcwd()
remaining = {row["id"] for row in json.load(open("lora_models/manifest.json"))
             if row.get("sample_count") == 200}
found = {}
for artifact in sorted(glob.glob("ab_test_runtime/experiments/decontaminate_batch*.json")):
    batch = "batch" + re.search(r"batch(\d+)\.json$", artifact).group(1)
    for row in json.load(open(artifact)).get("results", []):
        name = row.get("adapter")
        if name not in remaining:
            continue
        base = os.path.join(repo, "ab_test_runtime", "decontaminate", batch, name)
        adapter, data = os.path.join(base, "adapter"), os.path.join(base, "data")
        if os.path.exists(os.path.join(adapter, "adapter_model.safetensors")):
            found[name] = (adapter, data)
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    for name in sorted(found):
        handle.write("\t".join((name, *found[name])) + "\n")
missing = sorted(remaining - found.keys())
print(f"queue={len(found)} missing_candidates={len(missing)}")
for name in missing:
    print(f"MISSING {name}")
PYEOF

while IFS=$'\t' read -r name adapter data; do
    old="$EXP/gate_promote__$name.json"
    if [ -f "$old" ]; then
        cp "$old" "$BACKUP/"
    fi
    echo "=== $name $(date -u +%FT%TZ) ==="
    "$PY" -u app/experiments/verify_adapter_identity.py \
        --adapter "$adapter" --dataset "$data" --lines 6 \
        --out "$old" > "$LOG/$name.log" 2>&1
    rc=$?
    echo "rc=$rc $(tail -2 "$LOG/$name.log" | head -1)"
done < "$QUEUE"

echo "REGATE COMPLETE $(date -u +%FT%TZ)"
"$PY" promote_adapters.py --dry-run
