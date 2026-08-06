# Proposal: measure voice cloning against a human reading the same line

Your suggestion, worked into a design. Short version: **this is the only way to
get ground truth for voice quality, and nothing in this repo currently has
any.**

---

## Why it matters more than it first sounds

Every voice-similarity number measured here compares generated audio to
`ref_sample.wav` — the same short clip that was *also used as the generation
prompt*. That answers "does the output resemble its own prompt", which is close
to circular, and it can never answer "does this sound like the character should
sound", because nothing knows what the line should sound like.

With a public-domain audiobook, something does: **a human read that exact
line.** For any held-out line you get a paired comparison — same text, same
speaker, human against generated — which is a different kind of evidence from
anything in `results_index.csv`.

**And it supplies the two anchors every current number is missing.** Today
`clone_vs_lora` reports cosine 0.664 vs 0.634. Is 0.664 good? Nobody knows,
because there is no reference band. An audiobook gives both ends:

| anchor | what it is | what it tells you |
| --- | --- | --- |
| **ceiling** | same narrator, two different held-out lines | how similar audio can get when it genuinely IS the same person |
| **floor** | that narrator vs a different narrator | what "different voice" scores |

Generated-vs-human only becomes interpretable between those two. Reporting a
similarity score without them is the mistake already made once with acoustic
distance, and the reason `mean_f0` had to be marked indicative.

---

## Sources, with licences

Checked rather than recalled, because this repo credits what it uses.

| corpus | size | licence | fit |
| --- | --- | --- | --- |
| **[LJSpeech](https://keithito.com/LJ-Speech-Dataset/)** | ~24 h, **one** speaker, 13,100 clips | public domain (LibriVox + Gutenberg source) | **best starting point** — single narrator is exactly the LoRA use case |
| **[LibriTTS-R](https://www.openslr.org/141/)** | 585 h, 2,456 speakers, **24 kHz** | CC BY 4.0 | best for multi-speaker and for the floor anchor; restored audio quality |
| [LibriTTS](https://www.openslr.org/60/) | 585 h, same speakers | CC BY 4.0 | superseded by -R for audio quality |
| [LibriVox](https://librivox.org) | thousands of books | public domain | raw source if a specific narrator or non-English book is wanted |

**24 kHz matters** — LibriTTS-R is natively 24 kHz, which is what `_save_wav`
writes. LJSpeech is 22.05 kHz and needs resampling, which is a confound to
control rather than ignore: resample the *human* audio once, up front, and
compare like with like.

Attribution goes in `THIRD_PARTY_NOTICES.md` before any of this is run, not
after.

---

## Design

**Split by content, never by clip.** Train on chapters 1–N, hold out chapter
N+1 entirely. A random clip-level split leaks the narrator's rendering of
neighbouring sentences and inflates every number.

```
train   chapters 1..N of one narrator      -> LoRA + the ref_sample prompt
test    chapter N+1, same narrator, unseen -> generate each line
```

For every held-out line, generate at a **fixed seed** and compare to the human
audio of that line.

### Arms

| arm | what it isolates |
| --- | --- |
| `human` | ground truth |
| `lora` | trained adapter |
| `clone` | zero-shot from the same reference, no training |
| `human_other_line` | **ceiling** — same narrator, different held-out line |
| `different_narrator` | **floor** |

`clone` against `lora` is the same question `clone_vs_lora` asked at n=9 and
answered −0.0236 with the adapter ahead on 4 of 9. Here it can be asked at
n≈500 with ground truth, which is what that result actually needs.

### Metrics

Deliberately **not** `mean_f0` — this week measured 12.9 Hz error against a
32.4 Hz within-voice spread, so a mean over a line is too coarse to say
anything.

1. **ECAPA speaker embedding cosine** — `speaker_similarity.py` already does
   this; runs under the sibling interpreter.
2. **F0 contour correlation, time-warped** — the "hertz" comparison done
   properly. Correlate the *shape* of the pitch track after DTW alignment, not
   its average. Two readings of one sentence differ in timing; alignment is
   what makes the contours comparable, and a mean throws the contour away.
3. **Duration ratio** — generated ÷ human, per line. The instruction control
   showed 1.36–1.43× swings from wording alone, so pacing is measurable and
   already known to move.
4. **Mel-cepstral distortion (MCD)** — the standard TTS timbre metric, and it
   makes results comparable to published work rather than only to ourselves.
5. **Transcription WER of both** — the human read is not error-free either.
   Scoring generated audio against text without scoring the human against the
   same text hands the human an unearned advantage.

Report per line and per narrator. Never pool a single mean — book identity
already dominated method by 19 points in the attribution work, and there is no
reason to assume narrator identity behaves differently.

---

## What it would settle

- **Is the Voice Lab pipeline worth its cost?** Currently rests on n=9, one
  sentence, one seed, with the older comparison artifact gone from disk. This
  is the properly-powered version.
- **What does a similarity number mean?** Nothing, until the ceiling and floor
  exist.
- **Does more training data help?** `voice_data_saturation` asks this against a
  reference clip; against ground truth it becomes answerable.
- **Where does the model actually fail?** Per-line pairing points at specific
  sentences instead of a mean, so failures can be listened to.

## What it would not settle

Preference. A clone can score well on every metric here and still sound wrong,
and none of these numbers is a listener. That stays a blinded-listening
question, and this proposal feeds those materials rather than replacing them.

It also says nothing about **Japanese light novels**, which is what the product
actually generates. LJSpeech and LibriTTS are English non-fiction and classic
prose read in a measured audiobook register. A LoRA that clones an English
narrator well may not clone an anime-style performance well, and the reverse.
Treat the result as evidence about the *method*, not about the shipped voices.

---

## Cost

Measured locally today: **5.8 s/render** generation-only, **15.9 s** with ASR.

| stage | scale | estimate |
| --- | --- | --- |
| download LJSpeech | 2.6 GB | minutes |
| prepare + split | CPU | ~1 h |
| train one LoRA | existing `train_lora.py` | ~1-2 h |
| generate held-out lines | 500 lines × 3 arms | **~2.5 h** |
| scoring | ECAPA + F0 + MCD, CPU/GPU | ~1 h |

**Roughly 6 GPU-hours for the first narrator.** Three narrators — enough to say
anything general — is about a day, and the marginal narrator is ~4 h since
preparation is written once.

Cheaper than the cross-tradition run that just finished, against a question
that touches the most expensive subsystem in the repo.
