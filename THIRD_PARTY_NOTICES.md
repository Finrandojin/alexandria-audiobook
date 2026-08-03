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
