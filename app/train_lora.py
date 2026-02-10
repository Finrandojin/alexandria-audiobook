#!/usr/bin/env python3
"""
train_lora.py - Standalone LoRA fine-tuning script for Qwen3-TTS Base model.

Runs as a subprocess with structured stdout for log capture by Alexandria.
Prints [DATA], [TRAIN], [EPOCH], [DONE], [ERROR] prefixed lines for progress tracking.

Targets the talker's attention layers with LoRA via PEFT. Training uses teacher forcing:
the full input sequence (text + ground-truth codec codes) is built, the talker forward
produces the main loss (first codec group prediction), and forward_sub_talker_finetune
produces the code predictor loss (remaining groups). Both losses backpropagate through
the LoRA-adapted talker.

Usage:
    python train_lora.py \
        --data_dir /path/to/dataset \
        --output_dir /path/to/output \
        --epochs 50 --lr 5e-6 --lora_r 64 --lora_alpha 128
"""

import argparse
import gc
import json
import os
import random
import shutil
import sys
import time
import traceback


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Qwen3-TTS Base model")
    parser.add_argument("--data_dir", required=True, help="Directory containing metadata.jsonl and audio files")
    parser.add_argument("--output_dir", required=True, help="Directory to save the LoRA adapter")
    parser.add_argument("--model_name", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                        help="Base model name or path")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (samples per step)")
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128, help="LoRA alpha scaling")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--device", default="auto", help="Device: auto, cuda, cpu")
    parser.add_argument("--max_audio_seconds", type=float, default=30.0,
                        help="Maximum audio duration in seconds (longer clips are skipped)")
    return parser.parse_args()


def resolve_device(device_str):
    if device_str != "auto":
        return device_str
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def enable_rocm_optimizations():
    """Apply ROCm-specific optimizations. No-op on NVIDIA/CPU."""
    import torch
    if not (hasattr(torch.version, "hip") and torch.version.hip):
        return
    os.environ.setdefault("MIOPEN_FIND_MODE", "2")
    os.environ.setdefault("MIOPEN_LOG_LEVEL", "4")
    os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "TRUE")
    try:
        from triton.compiler import compiler as triton_compiler
        if not hasattr(triton_compiler, "triton_key"):
            import triton
            triton_compiler.triton_key = lambda: f"pytorch-triton-rocm-{triton.__version__}"
    except ImportError:
        pass


# ── Data preparation ────────────────────────────────────────────────────

def load_dataset(data_dir, hf_model, processor, device, dtype, max_audio_seconds):
    """Load metadata.jsonl and prepare training samples.

    For each entry, encodes audio to codec IDs and tokenizes text.
    Speaker embedding is extracted from a consistent ref_audio (same for all
    samples) per the official Qwen3-TTS fine-tuning approach.

    Returns list of sample dicts with pre-computed tensors.
    """
    import librosa
    import numpy as np
    import torch
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

    metadata_path = os.path.join(data_dir, "metadata.jsonl")
    if not os.path.exists(metadata_path):
        print(f"[ERROR] metadata.jsonl not found in {data_dir}", flush=True)
        sys.exit(1)

    with open(metadata_path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    if not entries:
        print("[ERROR] metadata.jsonl is empty", flush=True)
        sys.exit(1)

    print(f"[DATA] Found {len(entries)} entries in metadata.jsonl", flush=True)

    # ── Extract speaker embedding from ref_audio (consistent across all samples) ──
    # Check for ref_audio field in entries, or fall back to ref.wav in dataset dir,
    # or use the first training sample as reference.
    ref_audio_path = None
    if entries[0].get("ref_audio"):
        ref_rel = entries[0]["ref_audio"]
        ref_audio_path = os.path.join(data_dir, ref_rel)
    elif os.path.exists(os.path.join(data_dir, "ref.wav")):
        ref_audio_path = os.path.join(data_dir, "ref.wav")

    if ref_audio_path is None:
        # Fall back to first training sample as reference
        first_audio_rel = entries[0].get("audio_filepath") or entries[0].get("audio", "")
        ref_audio_path = os.path.join(data_dir, first_audio_rel)

    if not os.path.exists(ref_audio_path):
        print(f"[ERROR] Reference audio not found: {ref_audio_path}", flush=True)
        sys.exit(1)

    print(f"[DATA] Using reference audio: {os.path.basename(ref_audio_path)}", flush=True)

    ref_audio, ref_sr = librosa.load(ref_audio_path, sr=24000, mono=True)
    ref_audio = ref_audio.astype(np.float32)

    with torch.no_grad():
        ref_mels = mel_spectrogram(
            torch.from_numpy(ref_audio).unsqueeze(0),
            n_fft=1024, num_mels=128, sampling_rate=24000,
            hop_size=256, win_size=1024, fmin=0, fmax=12000,
        ).transpose(1, 2).to(device).to(dtype)
        spk_embedding = hf_model.speaker_encoder(ref_mels).detach()

    print(f"[DATA] Speaker embedding extracted from reference audio", flush=True)

    samples = []
    skipped = 0

    for i, entry in enumerate(entries):
        audio_rel = entry.get("audio_filepath") or entry.get("audio", "")
        audio_path = os.path.join(data_dir, audio_rel)
        text = entry["text"]

        if not os.path.exists(audio_path):
            print(f"[DATA] SKIP {i+1}/{len(entries)}: {audio_rel} (file not found)", flush=True)
            skipped += 1
            continue

        print(f"[DATA] Tokenizing {i+1}/{len(entries)}: {os.path.basename(audio_path)}", flush=True)

        # Load audio
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        duration = len(audio) / sr
        if duration > max_audio_seconds:
            print(f"[DATA] SKIP {i+1}/{len(entries)}: {audio_rel} ({duration:.1f}s > {max_audio_seconds}s)", flush=True)
            skipped += 1
            continue

        # Encode audio to codec IDs via speech tokenizer
        with torch.no_grad():
            enc = hf_model.speech_tokenizer.encode(audio, sr=sr)
            # 12Hz tokenizer returns list of [T, num_code_groups] per sample
            codec_ids = enc.audio_codes[0]  # [T, num_code_groups]

        # Tokenize text with chat template: <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
        assistant_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        text_inputs = processor(text=assistant_text, return_tensors="pt", padding=True)
        text_ids = text_inputs["input_ids"].to(device)
        if text_ids.dim() == 1:
            text_ids = text_ids.unsqueeze(0)

        samples.append({
            "codec_ids": codec_ids.to(device),          # [T, num_code_groups]
            "spk_embedding": spk_embedding,             # shared ref embedding [1, enc_dim]
            "text_ids": text_ids,                        # [1, text_len]
            "audio_path": audio_path,
            "text": text,
            "duration": duration,
        })

    print(f"[DATA] Prepared {len(samples)} samples ({skipped} skipped)", flush=True)
    if not samples:
        print("[ERROR] No valid training samples", flush=True)
        sys.exit(1)

    return samples, ref_audio_path


# ── Input construction ──────────────────────────────────────────────────

def build_teacher_forcing_input(sample, hf_model, device, dtype):
    """Build the full teacher-forcing input sequence for one training sample.

    Replicates the generate() method's input construction but includes
    ground-truth codec embeddings at every audio timestep.

    Returns:
        inputs_embeds: [1, prefill_len + T, D] full input sequence
        labels: [1, prefill_len + T] with -100 for prefill, first codec group for audio
        all_codec_ids: [T, num_code_groups] ground truth for code predictor
        prefill_len: int, number of prefill positions
    """
    import torch

    talker = hf_model.talker
    config = hf_model.config
    tc = config.talker_config  # talker config

    codec_ids_2d = sample["codec_ids"]   # [T, num_code_groups]
    spk_embedding = sample["spk_embedding"]  # [1, enc_dim]
    text_ids = sample["text_ids"]         # [1, text_len]

    T = codec_ids_2d.shape[0]  # number of audio frames
    num_code_groups = tc.num_code_groups

    # ── Special token embeddings ──
    special_ids = torch.tensor(
        [[config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]],
        device=device, dtype=text_ids.dtype,
    )
    tts_bos_embed, tts_eos_embed, tts_pad_embed = talker.text_projection(
        talker.get_text_embeddings()(special_ids)
    ).chunk(3, dim=1)  # each [1, 1, D]

    # ── Build prefill sequence (mirrors generate method) ──
    parts = []

    # Role tokens: first 3 tokens of text_ids = <|im_start|>assistant\n
    role_embed = talker.text_projection(
        talker.get_text_embeddings()(text_ids[:, :3])
    )  # [1, 3, D]

    # Codec prefix: [think_id, think_bos_id, language_id, think_eos_id]
    language_id = tc.codec_language_id.get("english", None) if tc.codec_language_id else None
    if language_id is not None:
        codec_prefill_list = [[tc.codec_think_id, tc.codec_think_bos_id,
                               language_id, tc.codec_think_eos_id]]
    else:
        codec_prefill_list = [[tc.codec_nothink_id, tc.codec_think_bos_id,
                               tc.codec_think_eos_id]]

    codec_prefix_embed = talker.get_input_embeddings()(
        torch.tensor(codec_prefill_list, device=device, dtype=text_ids.dtype)
    )  # [1, 3-4, D]

    # Speaker embed + codec_pad + codec_bos
    codec_suffix_embed = talker.get_input_embeddings()(
        torch.tensor([[tc.codec_pad_id, tc.codec_bos_id]], device=device, dtype=text_ids.dtype)
    )  # [1, 2, D]

    codec_embed = torch.cat([
        codec_prefix_embed,
        spk_embedding.view(1, 1, -1),
        codec_suffix_embed,
    ], dim=1)  # [1, prefix_codec_len, D]  (e.g. 7 for english: think,bos,lang,eos,spk,pad,bos)

    prefix_codec_len = codec_embed.shape[1]

    # Build the text-layer + codec-layer combined prefix
    # tts_pad for (prefix_codec_len - 2) positions + tts_bos, added to codec_embed[:-1]
    tts_prefix = torch.cat([
        tts_pad_embed.expand(-1, prefix_codec_len - 2, -1),
        tts_bos_embed,
    ], dim=1)  # [1, prefix_codec_len - 1, D]

    prefix_embed = tts_prefix + codec_embed[:, :-1]  # [1, prefix_codec_len - 1, D]

    # Combine role + prefix
    role_prefix = torch.cat([role_embed, prefix_embed], dim=1)  # [1, 3 + prefix_codec_len - 1, D]
    parts.append(role_prefix)

    # Text content (non-streaming mode): text_content + eos, with codec_pad overlay
    # text_ids[:, 3:-5] is the actual text content (strip role prefix and chat suffix)
    text_content_ids = text_ids[:, 3:-5]
    text_content_len = text_content_ids.shape[1]

    text_content_embed = talker.text_projection(
        talker.get_text_embeddings()(text_content_ids)
    )  # [1, text_content_len, D]
    text_with_eos = torch.cat([text_content_embed, tts_eos_embed], dim=1)  # [1, text_content_len + 1, D]

    # Codec pad overlay for text portion
    text_pad_ids = torch.full(
        (1, text_content_len + 1), tc.codec_pad_id,
        device=device, dtype=text_ids.dtype,
    )
    text_codec_pad_embed = talker.get_input_embeddings()(text_pad_ids)
    text_portion = text_with_eos + text_codec_pad_embed  # [1, text_content_len + 1, D]
    parts.append(text_portion)

    # End of prefill: tts_pad + codec_bos
    codec_bos_embed = talker.get_input_embeddings()(
        torch.tensor([[tc.codec_bos_id]], device=device, dtype=text_ids.dtype)
    )
    end_embed = tts_pad_embed + codec_bos_embed  # [1, 1, D]
    parts.append(end_embed)

    # Concatenate full prefill
    prefill_embeds = torch.cat(parts, dim=1)  # [1, prefill_len, D]
    prefill_len = prefill_embeds.shape[1]

    # ── Build audio steps (teacher forcing with ground-truth codes) ──
    # For each audio timestep t, the input is: sum of all codec group embeddings + tts_pad
    # codec_ids_2d: [T, num_code_groups]
    codec_ids_per_step = codec_ids_2d  # [T, num_code_groups]

    # Embed each codec group and sum
    # Group 0 uses the main embedding
    group_0_embed = talker.get_input_embeddings()(
        codec_ids_per_step[:, :1]
    )  # [T, 1, D]

    # Groups 1..N-1 use code_predictor embeddings
    group_embeds = [group_0_embed]
    for g in range(1, num_code_groups):
        g_embed = talker.code_predictor.get_input_embeddings()[g - 1](
            codec_ids_per_step[:, g:g + 1]
        )  # [T, 1, D]
        group_embeds.append(g_embed)

    # Sum all groups: [T, 1, D] -> squeeze to [T, D] -> unsqueeze batch
    all_groups = torch.cat(group_embeds, dim=1)  # [T, num_code_groups, D]
    codec_sum = all_groups.sum(dim=1)  # [T, D]

    # Add tts_pad_embed (trailing_text_hidden for non-streaming x-vector mode)
    audio_embeds = codec_sum + tts_pad_embed.squeeze(0)  # [T, D] broadcast
    audio_embeds = audio_embeds.unsqueeze(0)  # [1, T, D]

    # ── Full input sequence ──
    full_input = torch.cat([prefill_embeds, audio_embeds], dim=1)  # [1, prefill_len + T, D]

    # ── Labels ──
    # First codec group at each audio timestep. -100 for prefill (ignored by loss).
    first_codec = codec_ids_2d[:, 0]  # [T] - first code group across all timesteps
    labels = torch.full((1, prefill_len + T), -100, device=device, dtype=torch.long)
    labels[0, prefill_len:] = first_codec

    return full_input, labels, codec_ids_per_step, prefill_len


# ── Training loop ───────────────────────────────────────────────────────

def train(args):
    import torch
    import torch.nn.functional as F
    from transformers import AutoProcessor

    device = resolve_device(args.device)
    dtype = torch.bfloat16 if "cuda" in device else torch.float32

    enable_rocm_optimizations()

    print(f"[TRAIN] Device: {device}, dtype: {dtype}", flush=True)
    print(f"[TRAIN] Config: epochs={args.epochs}, lr={args.lr}, lora_r={args.lora_r}, "
          f"lora_alpha={args.lora_alpha}, grad_accum={args.gradient_accumulation_steps}", flush=True)

    # ── Load model ──
    print("[TRAIN] Loading Base model...", flush=True)
    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        args.model_name,
        device_map=device if device != "cpu" else None,
        dtype=dtype,
        attn_implementation="eager",
    )
    processor = model.processor
    hf_model = model.model  # Qwen3TTSForConditionalGeneration

    print("[TRAIN] Base model loaded", flush=True)

    # ── Load data ──
    samples, ref_audio_path = load_dataset(args.data_dir, hf_model, processor, device, dtype, args.max_audio_seconds)

    # ── Apply LoRA ──
    print("[TRAIN] Applying LoRA to talker...", flush=True)
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        print("[ERROR] peft package not installed. Run: pip install peft", flush=True)
        sys.exit(1)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )

    talker = hf_model.talker  # Qwen3TTSTalkerForConditionalGeneration
    peft_talker = get_peft_model(talker, lora_config)
    hf_model.talker = peft_talker

    # Enable gradient checkpointing for memory efficiency
    peft_talker.enable_input_require_grads()
    peft_talker.base_model.model.model.gradient_checkpointing_enable()

    trainable_params = sum(p.numel() for p in peft_talker.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_talker.parameters())
    print(f"[TRAIN] LoRA applied: {trainable_params:,} trainable / {total_params:,} total "
          f"({100 * trainable_params / total_params:.2f}%)", flush=True)

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        [p for p in peft_talker.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    # ── Training ──
    os.makedirs(args.output_dir, exist_ok=True)
    peft_talker.train()

    total_steps_per_epoch = len(samples)
    best_loss = float("inf")
    training_start = time.time()

    # Access underlying model structure (stable references)
    base_talker = peft_talker.base_model.model  # original talker with LoRA layers
    transformer = base_talker.model  # Qwen3TTSTalkerModel

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        epoch_steps = 0
        optimizer.zero_grad()

        # Shuffle samples each epoch
        epoch_samples = samples.copy()
        random.shuffle(epoch_samples)

        for step_idx, sample in enumerate(epoch_samples, 1):
            try:
                # Build teacher-forcing input
                full_input, labels, all_codec_ids, prefill_len = build_teacher_forcing_input(
                    sample, hf_model, device, dtype
                )

                T = all_codec_ids.shape[0]  # number of audio frames

                # ── Forward pass through talker transformer ──
                # Position IDs are auto-created by the model (3D multi-rope)
                output = transformer(
                    inputs_embeds=full_input,
                    use_cache=False,
                )
                hidden_states = output.last_hidden_state  # [1, seq_len, hidden_size]

                # ── Talker main loss: predict first codec group ──
                # codec_head predictions at audio positions
                # With standard causal LM shift: logit at position i predicts label at position i+1
                # Position prefill_len-1 predicts first audio code (labels[prefill_len])
                logits = base_talker.codec_head(hidden_states)  # [1, seq_len, vocab_size]

                # Shift: logits[:-1] predict labels[1:]
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                talker_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

                # ── Code predictor loss: predict remaining codec groups ──
                # Extract hidden states at audio-predicting positions
                # Position prefill_len-1 predicts audio step 0,
                # position prefill_len predicts audio step 1, etc.
                audio_hidden = hidden_states[0, prefill_len - 1:prefill_len + T - 1, :]  # [T, hidden_size]

                # all_codec_ids: [T, num_code_groups]
                _, sub_loss = base_talker.forward_sub_talker_finetune(
                    all_codec_ids, audio_hidden
                )

                # Combined loss (0.3 weight on sub-talker per official Qwen3-TTS training)
                total_loss = talker_loss + 0.3 * sub_loss

                # Scale for gradient accumulation
                scaled_loss = total_loss / args.gradient_accumulation_steps
                scaled_loss.backward()

                # Capture loss values before freeing tensors
                step_loss = total_loss.item()
                step_talker_loss = talker_loss.item()
                step_sub_loss = sub_loss.item()

                epoch_loss += step_loss
                epoch_steps += 1

                # Free intermediate tensors
                del full_input, labels, all_codec_ids, hidden_states
                del logits, shift_logits, shift_labels, audio_hidden
                del talker_loss, sub_loss, total_loss, scaled_loss

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"[TRAIN] OOM at epoch={epoch} step={step_idx}, skipping sample", flush=True)
                    if "cuda" in device:
                        torch.cuda.empty_cache()
                    gc.collect()
                    optimizer.zero_grad()
                    continue
                raise

            # Gradient accumulation step
            if step_idx % args.gradient_accumulation_steps == 0 or step_idx == total_steps_per_epoch:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in peft_talker.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()
                optimizer.zero_grad()

                if "cuda" in device:
                    torch.cuda.empty_cache()

            print(f"[TRAIN] epoch={epoch}/{args.epochs} step={step_idx}/{total_steps_per_epoch} "
                  f"loss={step_loss:.4f} talker_loss={step_talker_loss:.4f} "
                  f"sub_loss={step_sub_loss:.4f} lr={args.lr:.2e}", flush=True)

        # Epoch summary
        avg_loss = epoch_loss / max(epoch_steps, 1)
        print(f"[EPOCH] {epoch}/{args.epochs} avg_loss={avg_loss:.4f}", flush=True)

        # Save best adapter
        if avg_loss < best_loss:
            best_loss = avg_loss
            peft_talker.save_pretrained(args.output_dir)
            print(f"[TRAIN] Best adapter saved (loss={best_loss:.4f})", flush=True)

    # ── Final save ──
    training_time = time.time() - training_start

    # Always save final adapter (overwrites best if last epoch is better)
    peft_talker.save_pretrained(args.output_dir)

    # Copy reference audio as ref_sample.wav for inference
    ref_dest = os.path.join(args.output_dir, "ref_sample.wav")
    shutil.copy2(ref_audio_path, ref_dest)

    # Determine ref_sample_text: use ref_text.txt if present (written by Dataset Builder),
    # otherwise find the sample whose audio matches ref.wav, fallback to first sample.
    ref_text_file = os.path.join(args.data_dir, "ref_text.txt")
    if os.path.exists(ref_text_file):
        with open(ref_text_file, "r", encoding="utf-8") as f:
            ref_sample_text = f.read().strip()
        print(f"[DATA] Using ref text from ref_text.txt: '{ref_sample_text[:60]}...'", flush=True)
    else:
        # Legacy datasets: ref.wav is typically the first sample
        ref_sample_text = samples[0]["text"]
        print(f"[DATA] Using first sample text as ref text: '{ref_sample_text[:60]}...'", flush=True)

    # Save training metadata
    meta = {
        "model_name": args.model_name,
        "epochs": args.epochs,
        "lr": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "batch_size": args.batch_size,
        "num_samples": len(samples),
        "final_loss": avg_loss,
        "best_loss": best_loss,
        "training_time_seconds": round(training_time, 1),
        "ref_sample_audio": ref_audio_path,
        "ref_sample_text": ref_sample_text,
    }
    with open(os.path.join(args.output_dir, "training_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Adapter saved to {args.output_dir} "
          f"(best_loss={best_loss:.4f}, time={training_time:.0f}s)", flush=True)


if __name__ == "__main__":
    args = parse_args()
    try:
        train(args)
    except Exception as e:
        print(f"[ERROR] {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
