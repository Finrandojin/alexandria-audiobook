"""Fine-tune the cheap model on the 70B's answers to the rows it gets wrong.

The cost curve made the cascade a 70B-class commitment: a 32B scored -2.2 on
routed rows and a 27B +3.0/+3.7, against the 70B's +11.1 to +22.0. This is the
attempt to move that capability into the 14B instead of renting it per book.

`distill_collect` produced 1,091 rows from two books with no gold - grimgar06
(498) and mushoku18 (593) - each one a line where two cheap passes disagreed
and the 70B answered. The overlap check said the teacher supplies an answer
neither cheap pass produced on 26% of grimgar06's rows and 45% of mushoku18's,
so there is something to learn rather than a re-weighting of existing guesses.

THE PROMPT MUST MATCH PRODUCTION. The adapter is only useful if it drops into
the shipped path, so training examples are built with the same system prompt
and the same batch JSON shape `attribute_batch` sends - one entry, since a
per-row teacher label cannot supervise a 25-entry batch response. That is a
real mismatch between training and inference and it is the most likely reason
this fails: the model would be trained one-at-a-time and used in batches.
Recorded here rather than discovered later.

EVALUATION IS CROSS-BOOK BY CONSTRUCTION. grimgar06 and mushoku18 share no
rows with the four gold books, so scoring the adapter on those four measures
transfer, not memorisation. Run `distill_eval.py` after training; do not read
the training loss as a result.

VRAM. A 14B LoRA in bf16 needs roughly 30-40GB with gradient checkpointing,
which fits the A6000 and does not fit a 16GB card without 4-bit quantisation
that ROCm makes awkward. Run this on the instance.
"""
import argparse, json, os, sys, glob

REPO = "/home/fakemitch/pinokio/api/alexandria-audiobook2.git"
sys.path.insert(0, REPO + "/app")


def build_examples(paths, tokenizer_name=None, label_field="teacher"):
    """One (prompt, completion) pair per teacher-labelled row.

    The prompt mirrors what attribute_batch builds so the adapter transfers to
    the shipped path: same system prompt, same entry shape, same roster line.
    """
    from default_prompts import load_attribute_prompts
    system, user_template = load_attribute_prompts()
    out = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for raw in fh:
                r = json.loads(raw)
                entry = {"n": 0, "type": "SPOKEN", "text": r["line"]}
                ctx = r.get("context") or []
                target = next((k for k, c in enumerate(ctx) if c.get("target")), None)
                if target is not None:
                    if target > 0:
                        entry["previous_context"] = {
                            "type": ctx[target - 1].get("type"),
                            "text": ctx[target - 1].get("text")}
                    if target + 1 < len(ctx):
                        entry["next_context"] = {
                            "type": ctx[target + 1].get("type"),
                            "text": ctx[target + 1].get("text")}
                user = user_template.format(
                    roster=", ".join(r["roster"]) or "(none yet)",
                    batch=json.dumps([entry], ensure_ascii=False))
                # `label_field` selects WHOSE answer is being learned. The
                # teacher is the 70B; cheap_a is the student's own b25 answer on
                # the same routed row. Training on cheap_a is the ablation that
                # separates "the 70B taught it something" from "it learned the
                # task format", and only the label differs between the two runs.
                label = r.get(label_field)
                if not label:
                    continue
                completion = json.dumps(
                    [{"n": 0, "speaker": label}], ensure_ascii=False)
                out.append({"system": system, "user": user,
                            "completion": completion, "book": r["book"]})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", nargs="+",
                    default=sorted(glob.glob(REPO + "/ab_test_runtime/distill/train__*.jsonl")))
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--out", default=REPO + "/ab_test_runtime/distill/adapter")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=2048)
    # Default to holding nothing back. The four gold books already provide a
    # clean cross-book evaluation, and with 1,091 rows total an internal
    # holdout of 593 would cost more than half the training data to answer a
    # question distill_eval already answers better.
    ap.add_argument("--holdout", default="",
                    help="optional book kept out of training; the real "
                         "evaluation is cross-book against the four gold sets")
    ap.add_argument("--label_field", default="teacher",
                    choices=("teacher", "cheap_a", "cheap_b"),
                    help="whose answers to learn; cheap_a is the self-training "
                         "ablation against the 70B teacher")
    ap.add_argument("--dry_run", action="store_true",
                    help="build and report the dataset without loading a model")
    args = ap.parse_args()

    rows = build_examples(args.data, label_field=args.label_field)
    train = [r for r in rows if r["book"] != args.holdout]
    held = [r for r in rows if r["book"] == args.holdout]
    lens = sorted(len(r["user"]) for r in rows)
    print(f"{len(rows)} examples from {len(args.data)} files "
          f"(labels: {args.label_field})")
    print(f"  train {len(train)}  holdout({args.holdout}) {len(held)}")
    print(f"  prompt chars: median {lens[len(lens)//2]}, max {lens[-1]}")
    import collections
    print(f"  {args.label_field} labels: "
          f"{len(collections.Counter(r['completion'] for r in rows))} distinct")
    if args.dry_run:
        print("\n--- example ---")
        print(rows[0]["user"][:700])
        print("--- completion ---")
        print(rows[0]["completion"])
        return

    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              TrainingArguments, Trainer, DataCollatorForSeq2Seq)
    from peft import LoraConfig, get_peft_model
    from datasets import Dataset

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def encode(batch):
        """Tokenise, and mask the prompt so loss falls only on the answer.

        The prompt is the roster plus the batch JSON and runs to ~740 chars,
        while the completion is about 30. Training on the whole sequence would
        spend almost all of the gradient teaching the model to reproduce its
        own input, and the speaker - the only thing the 70B actually taught us -
        would be a rounding error in the loss.
        """
        input_ids, labels = [], []
        for s, u, c in zip(batch["system"], batch["user"], batch["completion"]):
            messages = [{"role": "system", "content": s},
                        {"role": "user", "content": u}]
            # enable_thinking=False APPENDS an empty <think></think> block to
            # the prompt; it is not a no-op flag. Production runs with
            # reasoning suppressed, so training must use the same template or
            # the adapter is tuned for a prompt shape inference never sends.
            try:
                prompt = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False)
            except TypeError:
                prompt = tok.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True)
            prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            answer_ids = tok(c + tok.eos_token,
                             add_special_tokens=False)["input_ids"]
            ids = (prompt_ids + answer_ids)[:args.max_len]
            lab = ([-100] * len(prompt_ids) + answer_ids)[:args.max_len]
            input_ids.append(ids)
            labels.append(lab)
        return {"input_ids": input_ids, "labels": labels,
                "attention_mask": [[1] * len(x) for x in input_ids]}

    ds = Dataset.from_list(train).map(
        encode, batched=True, remove_columns=["system", "user", "completion", "book"])
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out, num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr, bf16=True, logging_steps=10,
            save_strategy="epoch", report_to=[], lr_scheduler_type="cosine",
            warmup_ratio=0.03),
        train_dataset=ds,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True,
                                             label_pad_token_id=-100),
    ).train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"\nwrote adapter to {args.out}")
    print("Now run distill_eval.py against the four gold books. The training "
          "loss is not a result.")


if __name__ == "__main__":
    main()
