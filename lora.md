# Qwen3-TTS LoRA Training Guide for Alexandria

## Quick Reference

| Dataset Size | Epochs | Learning Rate | LoRA r | LoRA Alpha | Grad Accum | Target Loss |
|-------------|--------|---------------|--------|------------|------------|-------------|
| ~30 samples | 10-15 | 5e-6 | 64 | 128 | 4 | 4.1-4.2 |
| ~60 samples | 5-8 | 3e-6 | 64 | 128 | 4 | 4.1-4.2 |
| ~120 samples | 3 | 2e-6 | 64 | 128 | 4 | 4.1-4.2 |

**Target loss: 4.1-4.2** — this is the sweet spot for voice identity + instruct following + clean audio. Loss 4.1 is the floor; below this, garbling becomes increasingly likely. Note that identical settings can produce slightly different losses between runs, so aim for 4.15-4.2 for a reliable margin.

## Key Principles

- **More data = fewer epochs.** Each epoch teaches more with a larger dataset, so fewer passes are needed before overfitting.
- **Total exposure matters.** Samples x epochs should land around 250-400 total forward passes. Going above 600 risks overfitting.
- **Loss below 4.1 = garble risk.** Run-to-run variance means the same config can land on either side of 4.1. Below 3.5, output is consistently garbled or fails to reach EOS.
- **Loss above 4.5 = undertrained.** Clear audio but weak voice identity and faint instruct following.

## What Each Setting Does

| Setting | Effect |
|---------|--------|
| **Epochs** | Number of full passes through the dataset. More = tighter fit. |
| **Learning Rate** | How much weights adjust per step. Higher = faster learning but riskier. |
| **LoRA Rank (r)** | Capacity of the adapter (number of trainable dimensions). 64 is a good default. |
| **LoRA Alpha** | Scaling factor. Alpha/r ratio controls effective adapter weight. 128/64 = 2x is the tested default. |
| **Grad Accumulation** | Simulates larger batch sizes. 4 is stable for most cases. |
| **Batch Size** | Samples per step. Keep at 1 (VRAM limited). |

## Overfitting Symptoms

| Loss | Audio Quality | Instruct Following | Verdict |
|------|--------------|-------------------|---------|
| 4.4+ | Clear, no garble | Slight/faint | Undertrained |
| 4.1-4.2 | Clear, expressive | Good | Sweet spot |
| 3.9-4.1 | Expressive but garble risk | Strong | Knife's edge — run-to-run variance may garble |
| 3.4-3.8 | Garbly but legible | Strong | Starting to overfit |
| 3.0-3.3 | Garbled / no EOS | N/A | Overfit, unusable |

## Dataset Preparation

### Using the Dataset Builder (recommended)

1. Go to the **Dataset** tab in Alexandria
2. Enter a voice description and add rows (emotion + text pairs)
3. Generate samples — each row produces a WAV via VoiceDesign
4. Pick a clear, representative line as the **reference sample** (used as `ref.wav` for speaker embedding during training)
5. Save as dataset — creates the training folder automatically

### Tips for Good Datasets

- **Include variety:** Mix emotions, pacing, volume levels, sentence lengths
- **Include short utterances:** "Oh!", "Hmm.", "Right." — helps the model learn EOS behavior on short inputs
- **End with a neutral passage:** A long, calm, descriptive paragraph makes an ideal reference sample
- **Use consistent seed** for the reference sample to keep the speaker embedding stable across regenerations
- **15-30 minutes** of total audio is the target for a premium voice profile

### Dataset Structure

```
lora_datasets/{name}/
├── metadata.jsonl      # {audio_filepath, text} per line
├── ref.wav             # Reference audio for speaker embedding
├── ref_text.txt        # Transcript of ref.wav (must match exactly)
└── sample_000.wav ...  # Training audio files
```

### Metadata Format

```json
{"audio_filepath": "sample_000.wav", "text": "I told you never to come back here!"}
{"audio_filepath": "sample_001.wav", "text": "I just don't know what to do anymore."}
```

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| Garbled audio on new text | Overfitting (loss too low) | Reduce epochs or lr |
| Generation hangs / no EOS | Severe overfitting | Retrain with fewer epochs |
| Clear but no voice identity | Undertrained (loss too high) | Increase epochs or lr |
| Fast/rushed speech | Training data had fast pacing | Use "slow, even narration" in instruct, or retune dataset |
| Short texts hang at max_new_tokens | Model never learned short-utterance EOS | Add short vocalizations to training data |
| Initial audio glitch | Clone prompt alignment artifact | Minor — usually not present in full audiobook generation |
| ref.wav mismatch | ref_text.txt doesn't match ref.wav content | Ensure ref_text.txt contains the exact transcript of ref.wav |

## Tested Configurations (Alexandria)

| Adapter | Samples | Epochs | LR | Alpha | Loss | Result |
|---------|---------|--------|----|-------|------|--------|
| female-lora-01 | 33 | 3 | 1e-5 | 128 | 3.93 | Working, slightly fast pacing |
| female-lora-02 | 121 | 15 | 3e-6 | 128 | 3.03 | Overfit, garbled |
| female-lora-03 | 121 | 5 | 5e-6 | 128 | 3.10 | Overfit, no EOS |
| female-lora-04 | 121 | 2 | 5e-6 | 128 | 3.86 | Understandable, garbles + weird tones |
| female-lora-05 | 121 | 1 | 5e-6 | 128 | 4.43 | Clear, weak instruct |
| female-lora-06 | 121 | 3 | 2e-6 | 64 | 3.46 | Garbly but legible |
| **female-lora-07** | **121** | **3** | **2e-6** | **128** | **4.11** | **Best — clear audio, good instruct** |
| male-lora-01 | 61 | 5 | 1e-6 | 128 | 4.44 | Clear but flat, minimal instruct following |
| male-lora-02 | 61 | 7 | 1e-6 | 128 | 4.31 | Emotive, responsive to instruct |
| **male-lora-03** | **61** | **10** | **1e-6** | **128** | **4.11** | **Best — expressive, rich, good instruct** |
| male-lora-04 | 61 | 10 | 1e-6 | 128 | 4.12 | Same config as 03, few garbled lines (run-to-run variance) |
| male-lora-05 | 61 | 9 | 1e-6 | 128 | 4.17 | Clean, expressive, safe margin |
| male-lora-06 | 61 | 12 | 1e-6 | 128 | 3.99 | Very expressive but 50% garbled |
| male-lora-07 | 61 | 14 | 1e-6 | 128 | 3.89 | Legible but overfit |
