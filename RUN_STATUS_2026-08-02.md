# Run status — leave/return notes (2026-08-02)

Two queues are running unattended. Nothing needs you until you're back.

## Read this first when you return

    cat LOCAL_QUEUE_STATUS.txt                       # local, one line per stage
    ssh -i ~/.ssh/thunder_alexandria -p 30179 \
        ubuntu@64.247.196.38 'cat ~/QUEUE2_STATUS.txt'   # cloud

Both files are timestamped and say `OK` or `FAIL` per stage. Stages are
independent: a failure logs and the next stage still runs, so a `FAIL` line
costs one result, not the night.

## Local queue (free, your card)

    script   scratchpad/local_queue.sh
    logs     scratchpad/q_*.log

1. shippable eval on index18 + mushoku16 (completes the four-book llama.cpp number)
2. contiguity on index18 + mushoku16 (makes that finding four-for-four)
3. re-score all offline analyses and regenerate the results index

## Cloud queue (A6000, ~$0.49/hr)

    scripts  ~/cloud_queue.sh then ~/cloud_queue2.sh (chained by ~/chain.sh)
    logs     ~/q_*.log and ~/q2_*.log

1. cascade expensive phases, index18 cascade, gated history (3rd design),
   learning curve 25% and 50%
2. teacher labels from arc4_volume10wn and mushoku23
3. **retrain on all four books' labels** (~2,000 rows vs the current 1,091) and
   evaluate on all four gold books — the one remaining lever that produces more
   ACCURACY rather than more understanding
4. learning curve 75%
5. 8B student training

## THE INSTANCE IS STILL BILLING

Roughly $0.49/hr. When the cloud queue finishes (~20h), delete it:

    tnr snapshot create --instance-id kwami7f2 --name alexandria-attribution-<date> -y
    # wait for READY
    tnr snapshot list
    tnr delete 0

Everything needed to resume lives in the snapshot. Restoring takes ~20 minutes.

## Headline results so far

- **Distillation works.** +11.7 in bf16/transformers, **+4.2 in the shippable
  Q4+LoRA/llama.cpp stack** on grimgar03. Read the second number: the first was
  measured against a baseline the stack itself depressed by ~11 points.
- **The 70B is needed to CREATE the adapter, not to RUN it.** Self-training on
  the model's own answers: +1.3, p=0.58.
- **The gain is audible** — +13.3 on lead characters vs +11.7 overall, 66 lead
  errors removed against 6 minor ones, zero narrator confusions.
- **Batch size works because of conversation, not context** (-16.6 and -18.5 on
  two books, two stacks). Batch *boundaries* are null.
- **Agreement predicts correctness by +47.7 points**, and gated substitution on
  those rows is worth +3.3 whole-book with no extra inference.
- **Routing is dead** (-0.96 vs a fixed choice; oracle ceiling only +2.42).

## Open, and needing you rather than a machine

- **More `NOT_DIALOGUE` labels.** There are 46. That single number blocks
  segmentation entirely — not the model.
- **More gold books.** Every per-book claim is a line through four points.
