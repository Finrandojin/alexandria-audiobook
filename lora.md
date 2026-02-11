# Qwen3-TTS LoRA Training Guide for Alexandria

## Quick Reference

| Dataset Size | Epochs | Learning Rate | LoRA r | LoRA Alpha | Grad Accum | Target Loss |
|-------------|--------|---------------|--------|------------|------------|-------------|
| ~30 samples | 10-15 | 5e-6 | 64 | 128 | 4 | 3.9-4.2 |
| ~60 samples | 5-8 | 3e-6 | 64 | 128 | 4 | 3.9-4.2 |
| ~120 samples | 3 | 2e-6 | 64 | 128 | 4 | 3.9-4.2 |

**Target loss: 3.9-4.2** — this is the sweet spot for voice identity + instruct following + clean audio.

## Key Principles

- **More data = fewer epochs.** Each epoch teaches more with a larger dataset, so fewer passes are needed before overfitting.
- **Total exposure matters.** Samples x epochs should land around 250-400 total forward passes. Going above 600 risks overfitting.
- **Loss below 3.5 = overfitting.** The model memorizes training data and produces garbled output or fails to reach EOS on new text.
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
| 3.9-4.2 | Clear, minimal glitches | Good | Sweet spot |
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
├── metadata.jsonl      # {audio_filepath, text, instruct} per line
├── ref.wav             # Reference audio for speaker embedding
├── ref_text.txt        # Transcript of ref.wav (must match exactly)
└── sample_000.wav ...  # Training audio files
```

### Metadata Format

```json
{"audio_filepath": "sample_000.wav", "text": "I told you never to come back here!", "instruct": "Angry, forceful shout."}
{"audio_filepath": "sample_001.wav", "text": "I just don't know what to do anymore.", "instruct": "Sad, quiet whispering."}
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
| Rose | 33 | 3 | 1e-5 | 128 | 3.93 | Working, slightly fast pacing |
| Laura v1 | 121 | 15 | 3e-6 | 128 | 3.03 | Overfit, garbled |
| Laura v2 | 121 | 5 | 5e-6 | 128 | 3.10 | Overfit, no EOS |
| Laura v3 | 121 | 2 | 5e-6 | 128 | 3.86 | Understandable, garbles + weird tones |
| Laura (1 epoch) | 121 | 1 | 5e-6 | 128 | 4.43 | Clear, weak instruct |
| Laura v4 | 121 | 3 | 2e-6 | 64 | 3.46 | Garbly but legible |
| **Laura v5** | **121** | **3** | **2e-6** | **128** | **4.11** | **Best — clear audio, good instruct** |
