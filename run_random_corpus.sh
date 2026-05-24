#!/usr/bin/env bash
# run_random_corpus.sh — verify ROCm pipeline on 3 random book pairs.

set -u

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
OUT_DIR="$SCRIPT_DIR/random_test_output"
MODEL="Qwen2.5-14B-Instruct-Q6_K.gguf"
mkdir -p "$OUT_DIR"

PAIRS=(
    "/home/fakemitch/Desktop/New folder/Cliff Kurt Mushoku Tensei-converted.wav|/home/fakemitch/Desktop/books/Mushoku Tensei - Volume 01.epub"
    "/home/fakemitch/Desktop/New folder/Paul Boehmer Vampire Hunter D-converted.wav|/home/fakemitch/Desktop/books/Vampire Hunter D - Volume 01.epub"
    "/home/fakemitch/Desktop/New folder/J Michael Tatum Spice and Wolf, Vol. 10-converted.wav|/home/fakemitch/Desktop/books/Spice and Wolf - Volume 10 [Yen Press][Kobo].epub"
)

start_epoch=$(date +%s)
i=0
for pair in "${PAIRS[@]}"; do
    i=$((i+1))
    audio="${pair%%|*}"
    source="${pair#*|}"
    stem=$(basename "$audio" | sed 's/\.[^.]*$//')
    output="$OUT_DIR/dataset.zip"

    echo ""
    echo "═════════════════════════════════════════════════════════════════"
    echo "[$i/${#PAIRS[@]}] $stem"
    echo "  audio  : $audio"
    echo "  source : $source"
    echo "═════════════════════════════════════════════════════════════════"

    # Running with --limit 5 to keep the test fast but verify the full GPU handoff
    "$SCRIPT_DIR/run_with_restart.sh" \
        --audio "$audio" \
        --model "$MODEL" \
        --source "$source" \
        --output "$output" \
        --limit 5 \
        --diarize \
        --hf-token YOUR_HF_TOKEN_HERE \
        --lang en

    rc=$?
    if (( rc == 130 )); then
        exit 130
    fi
done

elapsed=$(( $(date +%s) - start_epoch ))
echo ""
echo "Random corpus test complete in $(( elapsed / 60 ))m."
