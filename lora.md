# Qwen3-TTS LoRA Training Guide for Alexandria

## What Is LoRA? (Plain English)

LoRA (Low-Rank Adaptation) is a way to **teach the TTS model a new voice** without retraining the entire thing. Think of it like this:

- The base TTS model already knows how to speak — it just doesn't know **your specific voice**
- A LoRA adapter is a small file (usually under 100 MB) that teaches the model what your voice sounds like
- Once trained, you can use this adapter to make the TTS model speak in that voice consistently

### Who Is This For?

- **You want a persistent custom voice** that you can reuse across projects
- **You have clean audio samples** of the voice you want to clone (at least 30 short clips)
- **You have a GPU with 8+ GB VRAM** (16 GB+ recommended)

### How Does This Relate to the Rest of Alexandria?

1. **Dataset Builder tab** — Where you create the training dataset (audio clips + text)
2. **Training tab** — Where you run the LoRA training
3. **Voices tab** — Where you use the trained LoRA adapter by selecting "LoRA Voice" and pointing to the adapter file

### Quick Start: Train Your First Voice

#### Step 1: Prepare Your Audio

You need **30-120 short audio clips** (2-10 seconds each) of the voice you want to clone:
- Clear recording, minimal background noise
- Each clip should be a single spoken sentence or phrase
- Save them as WAV files (16-bit, 24 kHz preferred)

#### Step 2: Create a Dataset

In Alexandria's web UI:
1. Go to the **Dataset** tab
2. Upload your audio clips one by one
3. For each clip, type the text that was spoken
4. Click **Save Dataset** when done

#### Step 3: Train the LoRA Adapter

1. Go to the **Training** tab
2. Select your dataset
3. Use these settings for your first training:
   - **Epochs:** 5-8 (for ~60 samples)
   - **Learning Rate:** 3e-6
   - **LoRA Rank:** 64
   - **LoRA Alpha:** 128
   - **Gradient Accumulation:** 4
4. Click **Start Training**
5. Watch the **Loss** number — stop training when it reaches **4.1-4.2**

#### Step 4: Use Your New Voice

1. Go to the **Voices** tab
2. Select **LoRA Voice** as the voice type
3. Point to your trained adapter file
4. Your new voice is ready to use!

### Understanding the Loss Number

The **Loss** is how well the model is learning your voice:
- **Above 4.5:** The voice sounds clear but doesn't match your target well (undertrained)
- **4.1-4.2:** The sweet spot — clear audio, good voice match, follows instructions
- **Below 4.1:** Risk of garbled or broken audio (overtrained) — **stop training immediately**
- **Below 3.5:** Audio is consistently garbled — you trained too long

**Rule of thumb:** Check the loss every few minutes. When it hits 4.15, stop training. Don't push it lower.

---

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

## Multilingual

LoRA adapters trained on single-language data carry that language's accent and pronunciation. An adapter trained on English samples will render German text with English pronunciation — the base model occasionally provides enough guidance for acceptable results, but it's inconsistent.

**Train a separate LoRA per language for each speaker.** This produces clean pronunciation without fighting the adapter's learned phonology.

### New Language vs. Voice Identity

The loss targets in this guide (4.1-4.2) were established for **English voice identity** training, where the base model already knows the language and the adapter only learns a new voice. When training on a **language the base model doesn't natively support**, the adapter must learn both new phonology and voice identity simultaneously. This is a harder task that shifts the overfitting boundary upward:

| Training Goal | Target Loss | Garble Floor |
|--------------|-------------|--------------|
| Voice identity (supported language) | 4.1-4.2 | ~4.1 |
| New language + voice identity | 4.5-5.5 | ~4.1-4.5 (still garbles at similar absolute loss) |

For new languages, use much lower learning rates and expect higher final loss. A loss of 5.0-5.5 can produce clean, natural speech with good pronunciation — the adapter has learned the language but hasn't been pushed into overfitting. Instruct following will be weaker at these higher losses, but voice identity and pronunciation are the priority.

**Tested example:** Catalan female, 2321 samples, 1 epoch, lr=2e-7, r=64, alpha=128 — final loss 5.38, clean speech with proper Catalan pronunciation. Earlier attempts at lr=1e-5 (loss 3.78) produced garbled output despite being in the "normal" English voice identity range.

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
| catalan-f-01 | 2321 | 1 | 1e-5 | 128 | 3.78 | Garbled — lr too high for new language |
| **catalan-f-07** | **2321** | **1** | **2e-7** | **128** | **5.38** | **Clean Catalan, good pronunciation** |
