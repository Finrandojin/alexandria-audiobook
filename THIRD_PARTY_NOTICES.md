# Third-Party Notices

Alexandria Audiobook2 incorporates material from the projects below. Each entry
records what was taken and reproduces the licence terms that require it.

Adding an entry is not optional. MIT and Apache-2.0 both require the copyright
notice to travel with the code; a comment in the source naming the project is
good practice but does not by itself satisfy either licence. If you copy or
closely adapt third-party code, add it here in the same commit.

Projects with **no licence file grant no rights at all** and must not be copied
from, however permissive they look. Reading them for ideas is fine; reproducing
their code is not.

---

## p0n1/epub_to_audiobook

- Source: https://github.com/p0n1/epub_to_audiobook
- Licence: MIT
- Used in: `app/experiments/quote_aware_chunking.py`

The priority-ordered list of split points in `PUNCT_PRIORITY` is adapted from
that project's `split_long_sentence`. Two properties were the reason to adopt it
rather than write our own: CJK sentence-enders are tried before English ones, so
a language without spaces never falls through to a hard character slice; and
closing brackets and quote marks are themselves split points, so a cut lands
after a quotation closes rather than inside it.

```
MIT License

Copyright (c) 2023 p0n1

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Ideas adopted without code

These projects shaped a design but no code was copied from them. No licence
obligation attaches to an idea; they are recorded because the provenance is
worth keeping, and because if we later lift code from one of them, the entry
above is where the obligation gets written down.

| Project | Licence | What it prompted |
| --- | --- | --- |
| [zeropointnine/tts-audiobook-tool](https://github.com/zeropointnine/tts-audiobook-tool) | MIT | Transcribe-back validation of generated audio: word-error alignment against the source with a length-scaled fail threshold, plus truncation detection. Implemented independently in `app/experiments/tts_output_validation.py`. |
| [DrewThomasson/VoxNovel](https://github.com/DrewThomasson/VoxNovel) | MIT | That BookNLP is the field-standard baseline for quotation speaker attribution, and that our ledger had never measured against it. See `app/experiments/booknlp_baseline.py`. |
| [nazdridoy/kokoro-tts](https://github.com/nazdridoy/kokoro-tts) | MIT | Voice blending by weighted interpolation as a way to widen a cast's voice pool. See `app/experiments/voice_blending.py`. |
| [WhiskeyCoder/Qwen3-Audiobook-Converter](https://github.com/WhiskeyCoder/Qwen3-Audiobook-Converter) | MIT | An independent choice of a 200-character chunk cap for Qwen3-TTS voice cloning, matching the cap our own external path enforces and our local path does not. |

### Consulted and rejected

Recorded so the same repositories are not re-reviewed from scratch.

| Project | Reason |
| --- | --- |
| sudonitin/Audio-book-generator | No licence file — no grant of rights. |
| aedocw/epub2tts-chatterbox | No licence file — no grant of rights. |
| devsapp/ai-audiobook-flow | No licence file — no grant of rights. |
| OmniVoice-Studio, QuickPiperAudiobook | AGPL-3.0; copyleft reach is a poor fit for a shipped application. |
| transitive-bullshit/kindle-ai-export | Solves acquisition, not narration; drives the Kindle web reader with Playwright, which raises terms-of-service questions for no technical gain. |
| ReadAlongs/Studio | Licence is unresolved (`NOASSERTION`), and our preparer alignment already works. |
| aedocw/epub2tts, DrewThomasson/ebook2audiobookpiper-tts, pravinyo/AudioBook | Nothing structurally absent from this project. |

---

## Research and industry practice consulted

No code or text is taken from these; they are recorded because they changed
design decisions, and a decision whose reasoning cannot be traced is one that
gets re-argued from scratch. Where a finding contradicted something already
built, that is noted - those are the entries worth keeping.

### Audiobook casting

- [How Professional Audiobook Narrators Voice Characters — Jay Myers](https://www.jaymyersvoiceover.com/blog-ideas/how-audiobook-narrators-make-characters-sound-distinct-believable-and-consistent)
- [How to Give Every Character a Different Voice in Your Audiobook — Audie](https://www.audie.ai/how-to-give-every-character-a-different-voice-in-your-audiobook)
- [Narrator Voice vs Character Voices — Audie](https://www.audie.ai/narrator-voice-vs-character-voices-getting-dialogue-right)
- [Character Voices in Audiobooks: 7 Key Considerations — Malk Williams](https://www.linkedin.com/pulse/character-voices-audiobooks-7-key-considerations-malk-williams)
- [Creating Distinct Character Voices for Audiobooks — Vois](https://vois.so/blog/character-voices-audiobooks)

Two findings changed the design:

**4-8 distinct character voices plus a narrator is the working maximum; more
risk listener confusion.** This CONTRADICTS the direction the voice-blending
experiment was heading. `voice_blending.py` measured that blending an 8-voice
pool reaches 148 identities and framed that as the prize. By professional
practice 148 is far past the point where a listener stops tracking who is
speaking. The arithmetic in that experiment stands; the goal it implied does
not.

**Contrast matters between characters who SHARE SCENES, not globally.**
Narrators pick voices so that characters appearing together are
distinguishable. Measured on the live book: 69% of character pairs never
co-occur within a 20-line window, and a greedy colouring of the co-occurrence
graph needs only 9 voices for 32 characters. So `audible_errors.py`'s
assumption - that any two characters sharing a voice is a confusion - is
stricter than practice. Sharing is fine when the characters never meet.

### Anime and dub casting

- [Answerman — Why Do Dubs Cast Men As Boy Characters, while Japan Casts Women? (ANN)](https://www.animenewsnetwork.com/answerman/2018-08-22/.135762)
- [Cross-Dressing Voices / Anime (TV Tropes)](https://tvtropes.org/pmwiki/pmwiki.php/CrossDressingVoices/Anime)
- [How Are English Dub Voice Actors Cast? (ANN)](https://www.animenewsnetwork.com/answerman/2015-09-11/.92806)
- [Anime Voice Acting: From Start to Finish — Bunny Studio](https://bunnystudio.com/blog/voice-acting-anime-from-start-to-finish/)

Adult women routinely voice boys and young men in Japanese originals - Naruto,
Edward Elric, Luffy, Ash - because few adult men sound convincingly young and
child actors bring scheduling and content limits. This matters for a corpus of
translated light novels: an `_f_` adapter on a teenage boy is CONVENTION, not
an error. It corrected a conclusion about Subaru, where the real mismatch was
the `50s` age band rather than the `f` tag.

### Voice and perceived sexual orientation

- [The effect of sexual orientation on voice acoustic properties (Frontiers in Psychology, 2024)](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2024.1412372/full)
- [Speech Acoustic Features: Gay Men, Heterosexual Men, and Heterosexual Women (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7497419/)
- [Do gay-sounding men speak like women? — Smyth & Rogers (TWPL)](https://twpl.library.utoronto.ca/index.php/twpl/article/download/6168/3157/0)
- ["Do I Sound Straight?" — Acoustic Correlates (JSLHR)](https://pubs.asha.org/doi/abs/10.1044/2018_JSLHR-S-17-0125)

Checked because the folk model - "sounds more feminine, so higher pitch" - was
about to be encoded. The literature does not support it. Findings on mean pitch
are mixed and sometimes reversed; the reliable correlates are ARTICULATORY,
chiefly /s/ with higher spectral peaks and longer duration, longer voice-onset
times, and more alveolar /l/. One study reports sibilant variation as socially
rather than anatomically determined. Nothing here is representable by an
`_f_`/`_m_` tag, which is further evidence that the sex tag is the wrong
primary key for casting.

---

## Speech corpora for ground-truth voice evaluation

Added 2026-08-05, before use rather than after. These provide something no
measurement in this repo previously had: a human reading the *same line* a
model was asked to generate, so voice similarity can be scored against ground
truth instead of against the reference clip that was also the prompt.

### LJSpeech (The LJ Speech Dataset)

- <https://keithito.com/LJ-Speech-Dataset/>
- 13,100 clips, ~24 hours, a single speaker, 22.05 kHz.
- **Public domain.** The recordings come from LibriVox and the texts from
  seven non-fiction works in the public domain via Project Gutenberg.
- Used as the primary single-narrator corpus: one speaker across many hours is
  exactly the shape a voice LoRA is trained for.
- Note for anyone reusing our numbers: at 22.05 kHz it needs resampling to the
  24 kHz this project generates at. We resample the HUMAN audio once, up
  front, so both sides of every comparison share a rate.

### LibriTTS-R

- <https://www.openslr.org/141/> — Koizumi et al., 2023.
- 585 hours, 2,456 speakers, natively 24 kHz; a speech-restored version of
  LibriTTS with identical content.
- **CC BY 4.0.**
- Used for multi-speaker work and for the "different narrator" floor anchor.
  Native 24 kHz matches what `tts.py` writes, so no resampling confound.

### LibriTTS

- <https://www.openslr.org/60/> — Zen et al., Interspeech 2019.
- **CC BY 4.0.** Derived from LibriSpeech's original LibriVox mp3 audio and
  Project Gutenberg texts.
- Superseded by LibriTTS-R for audio quality; recorded here because -R is
  derived from it and the alignment work is theirs.

### LibriVox

- <https://librivox.org> — **public domain.**
- The upstream source of all of the above, and the fallback if a specific
  narrator, book or language is wanted that the derived corpora do not cover.

### Project Gutenberg

- <https://www.gutenberg.org> — the text side of every corpus above.

None of this audio is redistributed by us. The corpora are downloaded to
`ab_test_runtime/corpora/`, which is gitignored; `PROVENANCE.md` there records
what to fetch and from where.

### Non-English speech corpora (candidates, not yet used)

Recorded 2026-08-05 while answering "is there anything open source, not
necessarily English". Listed with licences so the choice is auditable before
any of it is downloaded.

**[Kokoro Speech Dataset](https://mozilladatacollective.com/datasets/cmmknsho4014wmf087kvq5rc6)**
— the closest match to what this project actually generates.

- 43,253 clips, a single speaker, reading 14 novels.
- **Public domain**: Aozora Bunko texts + LibriVox recordings, both PD.
- Japanese, single-speaker, audiobook register - the same shape as LJSpeech,
  so `ljspeech_prepare.py` / `ljspeech_build.py` apply with a different root.
- **We already hold the text side.** `ab_test_runtime/corpora/aozora/kokoro.txt`
  is Natsume Sōseki's こころ, downloaded for the Japanese quote-robustness work.
  This corpus is LibriVox audio of that same text.

**[Multilingual LibriSpeech (MLS)](https://arxiv.org/pdf/2012.03411)** — **CC0**,
public domain. English, German, Dutch, French, Spanish, Italian, Portuguese,
Polish, from LibriVox + Gutenberg. The multi-language equivalent of the
English work.

**[CSS10](https://github.com/Kyubyong/css10)** — single speaker in ten
languages including **Japanese**, Chinese, Russian, Greek, Finnish, Hungarian,
from LibriVox. Single-speaker-per-language is exactly the voice-LoRA shape.

**[Common Voice](https://commonvoice.mozilla.org)** — **CC0**, ~9,283 hours
across 60 languages. Read Wikipedia sentences rather than audiobook
performance, so useful for speaker variety and a floor anchor, less so for
narration register.

**[CML-TTS](https://github.com/freds0/CML-TTS-Dataset)** — **CC BY 4.0**,
MLS adapted for TTS: Dutch, French, German, Italian, Polish, Portuguese,
Spanish.

**[JSUT / JVS](https://www.jstage.jst.go.jp/article/ast/41/5/41_E1950/_pdf)** —
Japanese, 10 h single speaker and 30 h across 100 speakers. Check the current
licence terms before use; they are free for research but not obviously the same
blanket public domain as the LibriVox-derived sets.

**J-MAC** — Japanese multi-speaker audiobook corpus, ~150 audiobooks. Built
from commercial Audiobook.jp recordings with Aozora reference text, so the
AUDIO is not redistributable the way the above are. Noted for completeness and
deliberately not a candidate.
