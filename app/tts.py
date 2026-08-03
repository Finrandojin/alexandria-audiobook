import os
import re
import json
import tempfile
import threading
import shutil

import numpy as np
import soundfile as sf

import device_utils
from pydub import AudioSegment

try:
    from .utils import secure_filename as _secure_filename, get_runtime_data_dir as _get_runtime_data_dir
except ImportError:
    from utils import secure_filename as _secure_filename, get_runtime_data_dir as _get_runtime_data_dir


def _resolve_asset_path(rel_path):
    """Resolve a voice-config relative path (adapter_path / ref_audio) or an
    output subdir to an absolute path, honoring ALEXANDRIA_DATA_DIR.

    Built-in adapters ship read-only at the repo root; all other voice data
    (lora_models, clone_voices, designed_voices) lives under the runtime data
    dir. A no-op when the data dir IS the repo root (the default layout), so
    existing installs are unaffected — it only corrects paths once data is
    relocated (e.g. the Docker /alexandria/runtime volume)."""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    norm = str(rel_path).replace("\\", "/").lstrip("/")
    base = root_dir if norm.split("/", 1)[0] == "builtin_lora" else _get_runtime_data_dir(root_dir)
    return os.path.join(base, rel_path)


DEFAULT_PAUSE_MS = 500  # Pause between different speakers
SAME_SPEAKER_PAUSE_MS = 250  # Shorter pause for same speaker continuing


def voice_category(voice_data):
    """Normalize a voice config entry's type into a routing category.

    Collapses the "lora"/"builtin_lora" pair into a single "lora" category so
    the membership test isn't duplicated at every call site. Returns one of
    "clone", "lora", "design", "ensemble", or "custom".
    """
    voice_type = (voice_data or {}).get("type", "custom")
    if voice_type == "clone":
        return "clone"
    if voice_type in ("lora", "builtin_lora"):
        return "lora"
    if voice_type == "design":
        return "design"
    if voice_type == "ensemble":
        return "ensemble"
    return "custom"


# Characters that are not speech and have no pronunciation. Reaching the model
# with these is not cosmetic: a 349-character table of contents separated by
# U+2022 produced 24.2 seconds of audio at normal level that transcribed to
# "* * *" - the model vocalising instead of reading. Measured 2026-08-03 by
# experiments/tts_output_validation.py.
#
# Two classes, because they need opposite treatment. A bullet or a rule is
# STRUCTURE: it separates items and must become a sentence break, not a word,
# or the list runs together. A symbol like © is a WORD the writer expects read
# aloud. Deleting the first class and speaking the second is why this is not
# one mapping table.
SPEECH_BREAKS = "•·▪◦‣∙■□◆●▲─━―*_~"
SPEECH_WORDS = {
    "©": "copyright", "®": "registered trademark", "™": "trademark",
    "&": "and", "@": "at", "%": "percent", "°": "degrees",
    "№": "number", "§": "section", "†": "", "‡": "",
}
# Absorbs whitespace either side, so "one ■■■ two" becomes "one. two."
# rather than "one . two." with a floating full stop the model may voice.
_BREAK_RE = re.compile(f"\\s*[{re.escape(SPEECH_BREAKS)}]+\\s*")
_SPACE_RE = re.compile(r"[ \t]{2,}")
_ORPHAN_PUNCT_RE = re.compile(r"(?:\.\s*){2,}")
_DUPE_WORD_RE = re.compile(r"\b(\w+)(\s+\1)+\b", re.IGNORECASE)


def normalize_for_speech(text):
    """Make text speakable before it reaches the model.

    Idempotent, so applying it twice is harmless - which is what lets it sit at
    every public entry point without tracking whether an inner path already ran
    it. Deliberately conservative: it does not touch typographic quotes, which
    the model reads correctly and which account for 59,004 of the 62,000
    non-ASCII characters in the library.
    """
    if not text:
        return text
    for symbol, word in SPEECH_WORDS.items():
        if symbol in text:
            text = text.replace(symbol, f" {word} " if word else " ")
    # A structural mark becomes a full stop so the items either side are read
    # as separate phrases rather than run together.
    text = _BREAK_RE.sub(". ", text)
    text = _ORPHAN_PUNCT_RE.sub(". ", text)
    text = _SPACE_RE.sub(" ", text)
    # "Copyright © 2016" would otherwise be read "copyright copyright 2016",
    # because the writer already spelled out the word the symbol stands for.
    text = _DUPE_WORD_RE.sub(r"\1", text)
    return text.strip(" .\t\n") + "." if text.strip(" .\t\n") else ""


def mix_to_unison(wav_paths, output_path, max_stretch=1.35):
    """Align same-text clips to a common length and mix them into one track.

    For an "ensemble" speaker, each member renders the same line separately, so
    the clips differ in length and prosody. Time-stretching every clip to the
    LONGEST one makes them start and end together (nobody is sped up). Words
    still drift mid-line — this is a chorus/doubling effect, not true unison.
    Word-level alignment would need forced alignment and is out of scope.

    A clip needing more than `max_stretch` is left alone: stretching that far
    smears it audibly, and letting it drift sounds better than destroying it.

    Returns True on success.
    """
    import librosa

    clips, rates = [], set()
    for path in wav_paths:
        audio, sr = sf.read(path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # force mono before mixing
        clips.append(np.asarray(audio, dtype=np.float32))
        rates.add(sr)
    if not clips:
        raise ValueError("mix_to_unison needs at least one clip")
    if len(rates) > 1:
        raise ValueError(f"cannot mix clips with different sample rates: {sorted(rates)}")
    sample_rate = rates.pop()

    target_len = max(len(c) for c in clips)
    aligned = []
    for clip in clips:
        stretch = target_len / len(clip)
        if stretch > max_stretch:
            print(f"  Unison: clip is {stretch:.2f}x shorter than the longest "
                  f"(> {max_stretch}x); leaving it unstretched to avoid smearing")
            out = clip
        else:
            # librosa rate>1 speeds up; to reach target_len use len/target_len.
            out = librosa.effects.time_stretch(clip, rate=len(clip) / target_len)
        if len(out) < target_len:
            out = np.pad(out, (0, target_len - len(out)))
        aligned.append(out[:target_len])

    # Power-preserving sum: /N would duck the line, straight sum would clip.
    mixed = np.sum(aligned, axis=0) / np.sqrt(len(aligned))

    # Match the loudest input so the line neither jumps out nor hides.
    loudest = max(float(np.abs(c).max()) for c in clips)
    peak = float(np.abs(mixed).max())
    if peak > 0 and loudest > 0:
        mixed = mixed * (loudest / peak)

    sf.write(output_path, mixed, sample_rate)
    return True


def sanitize_filename(name):
    """Make a string safe for use in filenames. Uses secure_filename to prevent path
    traversal, then collapses remaining spaces/dots/etc. to underscores to match the
    naming convention existing on-disk files were written with."""
    safe = _secure_filename(name) or "unnamed"
    return re.sub(r'[^\w\-]', '_', safe).lower()


def combine_audio_with_pauses(audio_segments, speakers, pause_ms=DEFAULT_PAUSE_MS,
                              same_speaker_pause_ms=SAME_SPEAKER_PAUSE_MS,
                              pause_overrides=None):
    """Combine audio segments with pauses between them.

    Args:
        pause_overrides: Optional list aligned with audio_segments. Each entry is
            the pause (ms) to insert *after* that segment, or None to use the
            default speaker-change logic. The last entry is ignored.
    """
    if not audio_segments:
        return None

    combined = audio_segments[0]
    prev_speaker = speakers[0]

    for i, (segment, speaker) in enumerate(zip(audio_segments[1:], speakers[1:])):
        override = pause_overrides[i] if pause_overrides else None
        if override is not None:
            gap = AudioSegment.silent(duration=override)
        elif speaker == prev_speaker:
            gap = AudioSegment.silent(duration=same_speaker_pause_ms)
        else:
            gap = AudioSegment.silent(duration=pause_ms)
        combined += gap + segment
        prev_speaker = speaker

    return combined


def compute_timeline(chunks_with_audio, pause_ms=DEFAULT_PAUSE_MS,
                     same_speaker_pause_ms=SAME_SPEAKER_PAUSE_MS):
    """Compute a timeline of (chunk, segment, abs_start_ms) tuples.

    Args:
        chunks_with_audio: list of (chunk_dict, AudioSegment) tuples.
            Each chunk_dict may have an optional 'pause_after' key (int ms)
            that overrides the default pause inserted after that chunk.
        pause_ms: Default pause between different speakers.
        same_speaker_pause_ms: Default pause when same speaker continues.

    Returns:
        list of (chunk_dict, AudioSegment, abs_start_ms) tuples.
    """
    timeline = []
    cursor_ms = 0
    prev_speaker = None
    prev_chunk = None

    for chunk, segment in chunks_with_audio:
        if prev_speaker is not None:
            override = prev_chunk.get("pause_after")
            if override is not None:
                gap = int(override)
            elif chunk.get("speaker") == prev_speaker:
                gap = same_speaker_pause_ms
            else:
                gap = pause_ms
            cursor_ms += gap

        timeline.append((chunk, segment, cursor_ms))
        cursor_ms += len(segment)
        prev_speaker = chunk.get("speaker")
        prev_chunk = chunk

    return timeline


class TTSEngine:
    """TTS engine supporting local (qwen-tts) and external (Gradio) backends.

    Mode is determined by config["tts"]["mode"]:
      - "local": Loads Qwen3TTSModel directly. No external server needed.
      - "external": Connects via Gradio client to a running TTS server.

    Models and clients are lazily initialized on first use.
    """

    def __init__(self, config):
        tts_config = config.get("tts", {})
        self._mode = tts_config.get("mode", "external")
        self._url = tts_config.get("url", "http://127.0.0.1:7860")
        self._device = tts_config.get("device", "auto")
        self._compile_codec_enabled = tts_config.get("compile_codec", False)

        # Cap on generated audio tokens per sequence. One source for both the
        # VRAM estimate and every generate call — if they ever disagree, the
        # estimate under-counts memory per sequence and over-sizes the batch.
        self._max_new_tokens = tts_config.get("max_new_tokens", 2048)

        # Language setting (passed to Qwen3-TTS)
        self._language = tts_config.get("language", "English")

        # Sub-batching config
        self._sub_batch_enabled = tts_config.get("sub_batch_enabled", True)
        self._sub_batch_min_size = max(1, tts_config.get("sub_batch_min_size", 4))
        self._sub_batch_ratio = max(1.0, float(tts_config.get("sub_batch_ratio", 5)))
        self._sub_batch_max_items = int(tts_config.get("sub_batch_max_items", 0))  # 0 = auto

        # Lazy-loaded backends (guarded by _model_lock to prevent concurrent loads)
        self._model_lock = threading.Lock()
        self._local_custom_model = None
        self._local_clone_model = None
        self._local_design_model = None
        self._local_lora_model = None
        self._custom_warmup_needed = True
        self._lora_adapter_path = None  # track which adapter is currently loaded
        self._gradio_client = None

        # Clone prompt cache: speaker_name -> (ref_audio_path, reusable voice_clone_prompt)
        self._clone_prompt_cache = {}
        # LoRA clone prompt cache: adapter_path -> reusable voice_clone_prompt
        self._lora_prompt_cache = {}

    @property
    def mode(self):
        return self._mode

    @staticmethod
    def _concat_audio(wav):
        """Concatenate audio array(s) into a single numpy array."""
        if isinstance(wav, list):
            return np.concatenate(wav) if len(wav) > 1 else wav[0]
        return wav

    @staticmethod
    def _vram_snapshot(label=""):
        """Log and return current VRAM state in GB (allocated/reserved/free/total)."""
        import torch
        if not torch.cuda.is_available():
            return {}
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        free, total = torch.cuda.mem_get_info()
        free /= 1e9
        total /= 1e9
        snap = {"allocated_gb": round(alloc, 2), "reserved_gb": round(reserved, 2),
                "free_gb": round(free, 2), "total_gb": round(total, 1)}
        tag = f"[{label}] " if label else ""
        print(f"VRAM {tag}{alloc:.2f} alloc / {reserved:.2f} reserved / {free:.2f} free / {total:.1f} total GB")
        return snap

    @staticmethod
    def _clear_gpu_cache():
        """Free GPU memory: garbage-collect Python objects, then clear CUDA cache."""
        import gc
        gc.collect()
        import torch
        torch.cuda.empty_cache()

    @staticmethod
    def _reset_compile_cache():
        """Reset torch.compile dynamo state to prevent guard accumulation.

        torch.compile(dynamic=True) accumulates shape guards across calls.
        With varying batch sizes and sequence lengths, the guard list grows
        and CPU-side guard evaluation becomes a bottleneck, causing
        progressive throughput degradation.  Resetting clears all in-memory
        guards; the next call pays a one-time recompilation cost (fast due
        to inductor disk cache) but prevents the slowdown from compounding.

        Only applied on ROCm (AMD GPUs). On NVIDIA, max-autotune mode
        re-benchmarks all kernel variants after each reset, and the
        benchmarking cost scales with tensor size — causing worse slowdown
        than the guard accumulation it prevents.
        """
        import torch
        if not (hasattr(torch.version, "hip") and torch.version.hip):
            return  # skip on NVIDIA/CPU — recompilation cost outweighs benefit
        torch._dynamo.reset()

    def _estimate_max_batch_size(self, model, clone_prompt_tokens=0,
                                ref_text_chars=0, max_text_chars=0):
        """Estimate how many sequences fit in free VRAM based on KV cache math.

        Uses the talker's architecture (num_layers, num_kv_heads, head_dim) to
        calculate KV cache bytes per token, then estimates total tokens per
        sequence from clone prompt size + text length + max generation length.

        The generation length is read from self._max_new_tokens - the same value
        the generate calls use - rather than taken as an argument, so the two
        cannot drift apart and silently over-size the batch.

        Returns max batch size (>= 1).  Falls back to a large default on CPU
        or if the model config is inaccessible.
        """
        max_new_tokens = self._max_new_tokens
        import torch
        if not torch.cuda.is_available():
            return 9999

        try:
            config = model.model.talker.config
            num_layers = config.num_hidden_layers
            num_kv_heads = config.num_key_value_heads
            head_dim = config.hidden_size // config.num_attention_heads
        except AttributeError:
            return 9999  # can't read config, skip estimation

        dtype_bytes = 2  # bf16
        kv_per_token = num_layers * 2 * num_kv_heads * head_dim * dtype_bytes

        # Total tokens per sequence (worst case: padded to longest + full generation)
        overhead = 10  # role tokens + prefix + special tokens
        ref_text_tokens = ref_text_chars // 3 if ref_text_chars else 0
        text_tokens = max_text_chars // 3 if max_text_chars else 0
        total_tokens = overhead + clone_prompt_tokens + ref_text_tokens + text_tokens + max_new_tokens

        # Overhead factor covers prefill activations, codec, allocator fragmentation
        OVERHEAD_FACTOR = 2.0
        mem_per_seq = total_tokens * kv_per_token * OVERHEAD_FACTOR

        # Available = driver-level free + PyTorch reserved-but-unallocated
        free_driver, _ = torch.cuda.mem_get_info()
        reserved_unused = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        free_total = free_driver + reserved_unused

        budget = int(free_total * 0.8)
        max_batch = max(1, budget // mem_per_seq)

        print(f"VRAM estimate: {free_total / 1e9:.1f}GB free, "
              f"{total_tokens} tok/seq ({clone_prompt_tokens} prompt + "
              f"{ref_text_tokens + text_tokens} text + {max_new_tokens} gen), "
              f"{mem_per_seq / 1e6:.0f}MB/seq -> max_batch={max_batch}")

        return max_batch

    def _build_sub_batches(self, texts, max_items=None):
        """Split sorted-by-length texts into sub-batches.

        Splits on three criteria (checked in order):
        1. VRAM item limit: when max_items is set (from _estimate_max_batch_size)
        2. Length ratio: when longest/shortest > sub_batch_ratio
        3. Minimum size: ratio splits only happen after sub_batch_min_size items

        Returns list of (start, end) index tuples.
        """
        if len(texts) <= 1:
            return [(0, len(texts))]

        # Manual cap overrides VRAM estimate when set (take the stricter of the two)
        if self._sub_batch_max_items > 0:
            max_items = min(max_items, self._sub_batch_max_items) if max_items else self._sub_batch_max_items

        if not self._sub_batch_enabled:
            # Ratio-based splitting is disabled, but STILL honor the VRAM/manual
            # item cap so a huge render isn't issued as one OOM-ing batch.
            if max_items and max_items > 0 and len(texts) > max_items:
                return [(s, min(s + max_items, len(texts))) for s in range(0, len(texts), max_items)]
            return [(0, len(texts))]

        sub_batches = []
        batch_start = 0

        for i in range(1, len(texts)):
            shortest = max(len(texts[batch_start]), 1)
            should_split = False

            # VRAM-estimated item limit (highest priority — based on actual
            # free GPU memory and per-sequence KV cache cost)
            if max_items is not None and (i - batch_start) >= max_items:
                should_split = True
            # Ratio split: large length disparity wastes padding —
            # only split after min_size items to preserve parallelism
            elif (i - batch_start) >= self._sub_batch_min_size:
                if len(texts[i]) > self._sub_batch_ratio * shortest:
                    should_split = True

            if should_split:
                sub_batches.append((batch_start, i))
                batch_start = i

        sub_batches.append((batch_start, len(texts)))
        return sub_batches

    # ── Lazy initialization ──────────────────────────────────────

    def _warmup_model(self, model) -> bool:
        """Run a short warmup generation to pre-tune MIOpen/GPU solvers.

        First generation after model load is ~2x slower due to MIOpen autotuning.
        This warmup pays that cost upfront so real generations run at full speed.
        """
        import time
        t0 = time.time()
        try:
            model.generate_custom_voice(
                text="The ancient library stood at the crossroads of two forgotten paths, its weathered stone walls covered in ivy that had been growing for centuries.",
                language=self._language,
                speaker="serena",
                instruct="neutral",
                non_streaming_mode=True,
                max_new_tokens=self._max_new_tokens,
            )
            print(f"Warmup done in {time.time()-t0:.1f}s")
            return True
        except Exception as e:
            print(f"Warmup failed (non-fatal): {e}")
            return False

    def ensure_custom_warmup(self, model=None) -> bool:
        """Warm the CustomVoice backend once, retrying after non-fatal failure."""
        if not self._custom_warmup_needed:
            return True
        if model is None:
            model = self._init_local_custom()
        print("Running batch warmup generation...")
        succeeded = self._warmup_model(model)
        self._custom_warmup_needed = not succeeded
        return succeeded

    def _resolve_device(self):
        """Resolve 'auto' device to the best available."""
        return device_utils.resolve_device(self._device)

    def _enable_rocm_optimizations(self):
        """Apply ROCm-specific optimizations. No-op on NVIDIA/CPU. See
        device_utils.enable_rocm_optimizations for the per-step rationale
        (MIOpen fast-find, flash attention via Triton AMD, triton_key shim)."""
        import torch
        device_utils.enable_rocm_optimizations()

        # Correct under-reported GPU properties on consumer RDNA2/3.
        # ROCm reports half the CU count and warp size 32 instead of 64,
        # causing PyTorch to under-schedule work on RX 6000/7000 GPUs.
        self._patch_rdna_device_properties(torch)


    @staticmethod
    def _patch_rdna_device_properties(torch):
        """Monkey-patch torch.cuda.get_device_properties to report correct
        CU count and wavefront size for consumer RDNA2/3 GPUs.

        ROCm exposes these GPUs with half CU count and warp_size=32
        (matching the CDNA/MI convention). The actual hardware has the
        full CU count and native wavefront64. Under-reporting causes
        PyTorch to generate smaller kernel launches.

        Based on AMD-GPU-BOOST (github.com/Painter3000/AMD-GPU-BOOST).
        """
        if hasattr(torch.cuda, '_rdna_props_patched'):
            return

        # Known RDNA GPU corrections: {name_substring: (true_CUs, true_warp)}
        _rdna_corrections = {
            # RDNA4. Verified on this machine 2026-07-27: rocminfo reports
            # "Compute Unit: 64" while torch reports multi_processor_count=32,
            # i.e. torch is given the WGP count, the same halving this patch
            # exists to undo on RDNA2/3. Warp stays 32 because rocminfo reports
            # "Wavefront Size: 32" for gfx1201 - unlike the entries below, there
            # is no measured basis here for forcing 64.
            "9070 XT":  (64, 32),
            "7900 XTX": (96, 64),
            "7900 XT":  (84, 64),
            "7900 GRE": (80, 64),
            "7800 XT":  (60, 64),
            "7700 XT":  (54, 64),
            "7600":     (32, 64),
            "6950 XT":  (80, 64),
            "6900 XT":  (80, 64),
            "6800 XT":  (72, 64),
            "6800":     (60, 64),
            "6750 XT":  (40, 64),
            "6700 XT":  (40, 64),
            "6700":     (36, 64),
            "6650 XT":  (32, 64),
            "6600 XT":  (32, 64),
            "6600":     (28, 64),
        }

        original_fn = torch.cuda.get_device_properties
        _cache = {}

        def _patched_get_device_properties(device=None):
            if device is None:
                device = torch.cuda.current_device()
            if isinstance(device, int):
                key = device
            else:
                try:
                    dev = torch.device(device) if isinstance(device, str) else device
                    key = dev.index if dev.index is not None else torch.cuda.current_device()
                except Exception:
                    key = 0

            if key in _cache:
                return _cache[key]

            props = original_fn(device)

            # Find matching correction
            correction = None
            for substr, vals in _rdna_corrections.items():
                if substr in props.name:
                    correction = vals
                    break

            if correction:
                true_cus, true_warp = correction
                _ov = {"multi_processor_count": true_cus, "warp_size": true_warp}

                class _RDNADeviceProps:
                    __slots__ = ("_orig", "_ov")

                    def __init__(self, orig, ov):
                        object.__setattr__(self, "_orig", orig)
                        object.__setattr__(self, "_ov", ov)

                    def __getattr__(self, name):
                        ov = object.__getattribute__(self, "_ov")
                        if name in ov:
                            return ov[name]
                        try:
                            return getattr(object.__getattribute__(self, "_orig"), name)
                        except RuntimeError:
                            # Some ROCm/driver combos raise RuntimeError (not
                            # AttributeError) for certain C-extension device
                            # property fields. The SimpleNamespace this proxy
                            # replaced caught (AttributeError, RuntimeError)
                            # per-attribute and just omitted that attribute -
                            # re-raise as AttributeError so callers using
                            # getattr(obj, name, default)/hasattr still get
                            # that same "not available" signal instead of an
                            # unhandled RuntimeError propagating through.
                            raise AttributeError(name)

                    def __repr__(self):
                        return repr(object.__getattribute__(self, "_orig"))

                patched = _RDNADeviceProps(props, _ov)
                # Safely calculate old thread count - handle missing warp_size on unusual ROCm versions
                old_warp = getattr(props, 'warp_size', 32)  # Default to 32 if missing
                old_threads = props.multi_processor_count * old_warp
                new_threads = true_cus * true_warp
                print(f"  [RDNA fix] {props.name}: CUs {props.multi_processor_count}->{true_cus}, "
                      f"warp {old_warp}->{true_warp}, "
                      f"threads {old_threads}->{new_threads}")
                _cache[key] = patched
                return patched

            _cache[key] = props
            return props

        torch.cuda.get_device_properties = _patched_get_device_properties
        torch.cuda._rdna_props_patched = True

    def _compile_codec(self, model):
        """Apply torch.compile to the audio codec for faster decoding.

        The codec decoder has 136 attention modules and many small ops that
        benefit enormously from compilation.  Profiling shows the codec is
        47% of single-gen time and 85% of batch time uncompiled.  With
        torch.compile (dynamic=True, max-autotune), batch throughput
        improves from ~1.3x to ~4.3x real-time and single generation
        drops from ~14s to ~9s.

        max-autotune mode benchmarks GPU kernels to pick the fastest and
        handles varying batch sizes gracefully (unlike reduce-overhead
        which uses CUDA graphs that break on shape changes).
        """
        import torch, time
        vram_before = self._vram_snapshot("pre-compile_codec")
        t0 = time.time()
        try:
            codec = model.model.speech_tokenizer.model
            model.model.speech_tokenizer.model = torch.compile(
                codec, mode="max-autotune", dynamic=True,
            )
            compile_time = time.time() - t0
            vram_after = self._vram_snapshot("post-compile_codec")
            delta = vram_after.get("allocated_gb", 0) - vram_before.get("allocated_gb", 0)
            print(f"Codec compiled OK in {compile_time:.1f}s (VRAM delta: {delta:+.2f} GB).")
        except Exception as e:
            print(f"Codec compilation FAILED after {time.time()-t0:.1f}s: {type(e).__name__}: {e}")

    @staticmethod
    def _resolve_local_model_path(model_id):
        """Check if a HuggingFace model is cached locally and return its snapshot path.

        Uses try_to_load_from_cache to find the local snapshot directory.
        Returns the local path string if cached, or None if not cached.
        """
        from huggingface_hub import try_to_load_from_cache
        result = try_to_load_from_cache(model_id, "config.json")
        if isinstance(result, str):
            # result is the full path to config.json inside the snapshot dir
            return os.path.dirname(result)
        return None

    @staticmethod
    def _load_model(model_cls, model_id, load_kwargs):
        """Load a model, preferring local cache to avoid network issues.

        Checks if the model snapshot exists in the HF cache and loads from
        the local directory path directly, bypassing all HF Hub network calls.
        Falls back to normal download on first install when cache is empty.
        If loading from local cache fails (e.g. incomplete snapshot), retries
        with the model ID so HF Hub can download any missing files.
        """
        local_path = TTSEngine._resolve_local_model_path(model_id)
        if local_path:
            print(f"  Loading from local cache: {local_path}")
            try:
                return model_cls.from_pretrained(local_path, **load_kwargs)
            except Exception as e:
                import traceback
                print(f"  Warning: Failed to load from local cache: {e}")
                traceback.print_exc()
                print(f"  Retrying with model ID (may download missing files)...")
                return model_cls.from_pretrained(model_id, **load_kwargs)
        else:
            print(f"  Model not cached locally, downloading {model_id}...")
            return model_cls.from_pretrained(model_id, **load_kwargs)

    def _init_local_custom(self):
        """Load Qwen3-TTS CustomVoice model on demand."""
        if self._local_custom_model is not None:
            return self._local_custom_model

        with self._model_lock:
            if self._local_custom_model is not None:
                return self._local_custom_model

            self._enable_rocm_optimizations()

            import torch
            from qwen_tts import Qwen3TTSModel

            device = self._resolve_device()
            dtype = torch.bfloat16 if "cuda" in device else torch.float32

            print(f"Loading Qwen3-TTS CustomVoice model on {device} ({dtype})...")
            vram_before = self._vram_snapshot("pre-load")
            load_kwargs = {"dtype": dtype}
            if device != "cpu":
                load_kwargs["device_map"] = device
            self._local_custom_model = self._load_model(
                Qwen3TTSModel, "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", load_kwargs,
            )
            vram_after = self._vram_snapshot("post-load")
            model_gb = vram_after.get("allocated_gb", 0) - vram_before.get("allocated_gb", 0)
            print(f"CustomVoice model loaded: {model_gb:.2f} GB VRAM footprint")
            if self._compile_codec_enabled:
                self._compile_codec(self._local_custom_model)
            return self._local_custom_model

    def _init_local_clone(self):
        """Load Qwen3-TTS Base model (for voice cloning) on demand."""
        if self._local_clone_model is not None:
            return self._local_clone_model

        with self._model_lock:
            if self._local_clone_model is not None:
                return self._local_clone_model

            self._enable_rocm_optimizations()

            import torch
            from qwen_tts import Qwen3TTSModel

            device = self._resolve_device()
            dtype = torch.bfloat16 if "cuda" in device else torch.float32

            print(f"Loading Qwen3-TTS Base model (voice cloning) on {device} ({dtype})...")
            load_kwargs = {"dtype": dtype}
            if device != "cpu":
                load_kwargs["device_map"] = device
            self._local_clone_model = self._load_model(
                Qwen3TTSModel, "Qwen/Qwen3-TTS-12Hz-1.7B-Base", load_kwargs,
            )
            if self._compile_codec_enabled:
                self._compile_codec(self._local_clone_model)
            print("Base model (voice cloning) loaded.")
            return self._local_clone_model

    def _init_local_design(self):
        """Load Qwen3-TTS VoiceDesign model on demand."""
        if self._local_design_model is not None:
            return self._local_design_model

        with self._model_lock:
            if self._local_design_model is not None:
                return self._local_design_model

            self._enable_rocm_optimizations()

            import torch
            from qwen_tts import Qwen3TTSModel

            device = self._resolve_device()
            dtype = torch.bfloat16 if "cuda" in device else torch.float32

            print(f"Loading Qwen3-TTS VoiceDesign model on {device} ({dtype})...")
            load_kwargs = {"dtype": dtype}
            if device != "cpu":
                load_kwargs["device_map"] = device
            self._local_design_model = self._load_model(
                Qwen3TTSModel, "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", load_kwargs,
            )
            if self._compile_codec_enabled:
                self._compile_codec(self._local_design_model)
            print("VoiceDesign model loaded.")
            return self._local_design_model

    def _init_local_lora(self, adapter_path):
        """Load Qwen3-TTS Base model with a LoRA adapter on demand.

        Caches the model; if a different adapter is requested the old one
        is unloaded first to free VRAM.
        """
        if self._local_lora_model is not None and self._lora_adapter_path == adapter_path:
            return self._local_lora_model

        with self._model_lock:
            if self._local_lora_model is not None and self._lora_adapter_path == adapter_path:
                return self._local_lora_model

            # Unload previous adapter if switching
            if self._local_lora_model is not None:
                print(f"Unloading previous LoRA adapter ({self._lora_adapter_path})...")
                del self._local_lora_model
                self._local_lora_model = None
                self._lora_adapter_path = None
                self._lora_prompt_cache.clear()
                self._clear_gpu_cache()

            self._enable_rocm_optimizations()

            import torch
            from qwen_tts import Qwen3TTSModel
            from peft import PeftModel

            device = self._resolve_device()
            dtype = torch.bfloat16 if "cuda" in device else torch.float32

            print(f"Loading Qwen3-TTS Base model + LoRA adapter on {device} ({dtype})...")
            load_kwargs = {"dtype": dtype}
            if device != "cpu":
                load_kwargs["device_map"] = device

            model = self._load_model(
                Qwen3TTSModel, "Qwen/Qwen3-TTS-12Hz-1.7B-Base", load_kwargs,
            )

            # Wrap the talker with the LoRA adapter
            model.model.talker = PeftModel.from_pretrained(
                model.model.talker,
                adapter_path,
            )
            model.model.talker.eval()

            if self._compile_codec_enabled:
                self._compile_codec(model)

            self._local_lora_model = model
            self._lora_adapter_path = adapter_path
            print(f"LoRA adapter loaded from {adapter_path}")
            return model

    def _init_external(self):
        """Create Gradio client on demand."""
        if self._gradio_client is not None:
            return self._gradio_client

        from gradio_client import Client

        print(f"Connecting to TTS server at {self._url}...")
        self._gradio_client = Client(self._url)
        print("Connected to external TTS server.")
        return self._gradio_client

    # ── Clone prompt cache (local mode) ──────────────────────────

    def _get_clone_prompt(self, speaker, voice_config):
        """Get or create a cached voice clone prompt for a speaker."""
        voice_data = voice_config.get(speaker, {})
        ref_audio_path = voice_data.get("ref_audio")
        ref_text = voice_data.get("ref_text")

        if not ref_audio_path or not ref_text:
            raise ValueError(f"Clone voice for '{speaker}' missing ref_audio or ref_text")
        # Resolve relative paths against project root (parent of app/)
        if not os.path.isabs(ref_audio_path):
            ref_audio_path = _resolve_asset_path(ref_audio_path)
        if not os.path.exists(ref_audio_path):
            raise FileNotFoundError(f"Reference audio not found for '{speaker}': {ref_audio_path}")

        # Check cache — invalidate if ref_audio changed
        if speaker in self._clone_prompt_cache:
            cached_path, cached_prompt = self._clone_prompt_cache[speaker]
            if cached_path == ref_audio_path:
                return cached_prompt
            print(f"Voice changed for '{speaker}', rebuilding clone prompt...")

        model = self._init_local_clone()

        # Load reference audio as numpy array
        audio_array, sample_rate = sf.read(ref_audio_path)
        # Ensure mono
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)

        print(f"Creating clone prompt for '{speaker}'...")
        prompt = model.create_voice_clone_prompt(
            ref_audio=(audio_array, sample_rate),
            ref_text=ref_text,
        )
        self._clone_prompt_cache[speaker] = (ref_audio_path, prompt)
        print(f"Clone prompt cached for '{speaker}'.")
        return prompt

    # ── Core generation methods ──────────────────────────────────

    def generate_custom_voice(self, text, instruct_text, speaker, voice_config, output_path):
        """Generate audio using CustomVoice model. Returns True on success."""
        text = normalize_for_speech(text)
        if self._mode == "local":
            return self._local_generate_custom(text, instruct_text, speaker, voice_config, output_path)
        else:
            return self._external_generate_custom(text, instruct_text, speaker, voice_config, output_path)

    def generate_clone_voice(self, text, speaker, voice_config, output_path):
        """Generate audio using voice cloning. Returns True on success."""
        text = normalize_for_speech(text)
        if self._mode == "local":
            return self._local_generate_clone(text, speaker, voice_config, output_path)
        else:
            return self._external_generate_clone(text, speaker, voice_config, output_path)

    def generate_voice(self, text, instruct_text, speaker, voice_config, output_path):
        """Generate audio using the appropriate method based on voice type config."""
        text = normalize_for_speech(text)
        voice_data = voice_config.get(speaker)
        if not voice_data:
            print(f"Warning: No voice configuration for '{speaker}'. Skipping.")
            return False

        category = voice_category(voice_data)

        if category == "clone":
            return self.generate_clone_voice(text, speaker, voice_config, output_path)
        elif category == "lora":
            return self.generate_lora_voice(text, instruct_text, voice_data, output_path)
        elif category == "design":
            return self.generate_design_voice(text, instruct_text, voice_data, output_path)
        elif category == "ensemble":
            return self.generate_ensemble_voice(text, instruct_text, voice_data,
                                                voice_config, output_path)
        else:
            return self.generate_custom_voice(text, instruct_text, speaker, voice_config, output_path)

    def generate_ensemble_voice(self, text, instruct_text, voice_data, voice_config,
                                output_path):
        """Render one line as several characters speaking at once.

        Members are ordinary speaker keys, so each is rendered through
        generate_voice with whatever voice it already has — a member with a LoRA
        assigned keeps that LoRA. The clips are then aligned and mixed by
        mix_to_unison.
        """
        members = voice_data.get("members") or []
        if not members:
            raise ValueError("Ensemble voice has no members configured.")

        for member in members:
            member_data = voice_config.get(member)
            if not member_data:
                raise ValueError(
                    f"Ensemble member '{member}' has no voice configuration.")
            # One level only: an ensemble of ensembles would recurse forever.
            if voice_category(member_data) == "ensemble":
                raise ValueError(
                    f"Ensemble member '{member}' is itself an ensemble; "
                    f"ensembles cannot be nested.")

        temp_dir = tempfile.mkdtemp(prefix="unison_")
        try:
            member_paths = []
            for i, member in enumerate(members):
                member_path = os.path.join(temp_dir, f"member_{i}.wav")
                if not self.generate_voice(text, instruct_text, member,
                                           voice_config, member_path):
                    raise RuntimeError(
                        f"Ensemble member '{member}' failed to generate.")
                member_paths.append(member_path)

            print(f"  Unison: mixing {len(member_paths)} voices ({', '.join(members)})")
            return mix_to_unison(member_paths, output_path)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    # ── Voice design generation ──────────────────────────────────

    def generate_voice_design(self, description, sample_text, language=None, seed=-1):
        """Generate a voice from a text description using the VoiceDesign model.

        Args:
            description: Natural language description of the desired voice
            sample_text: Text to synthesize with the designed voice
            language: Language code (defaults to engine's configured language)
            seed: Random seed (-1 for random, >= 0 for reproducible)

        Returns:
            (wav_path, sample_rate) on success

        Raises:
            RuntimeError: If generation fails
        """
        import time
        import torch

        lang = language or self._language
        print(f"VoiceDesign: generating preview for description='{description[:80]}...'"
              f"{f', seed={seed}' if seed >= 0 else ''}")

        model = self._init_local_design()

        if seed >= 0:
            torch.manual_seed(seed)

        t_start = time.time()
        wavs, sr = model.generate_voice_design(
            text=sample_text,
            instruct=description,
            language=lang,
            non_streaming_mode=True,
            max_new_tokens=self._max_new_tokens,
        )
        gen_time = time.time() - t_start

        if wavs is None or len(wavs) == 0:
            raise RuntimeError("VoiceDesign model returned no audio")

        audio = np.concatenate(wavs) if len(wavs) > 1 else wavs[0]
        duration = len(audio) / sr
        print(f"VoiceDesign: done in {gen_time:.1f}s -> {duration:.1f}s audio")

        # Save to previews directory (under the runtime data dir, matching where
        # app.py's /api/voice_design/save reads it back from).
        previews_dir = _resolve_asset_path(os.path.join("designed_voices", "previews"))
        os.makedirs(previews_dir, exist_ok=True)

        filename = f"preview_{int(time.time() * 1000)}.wav"
        wav_path = os.path.join(previews_dir, filename)
        self._save_wav(audio, sr, wav_path)

        return wav_path, sr

    def generate_design_voice(self, text, instruct_text, voice_data, output_path):
        """Generate audio using VoiceDesign model with combined description + instruct.

        The voice_data 'description' field provides the base voice identity,
        and the per-line instruct_text is appended for delivery/emotion direction.
        """
        text = normalize_for_speech(text)
        import shutil

        base_desc = (voice_data.get("description") or "").strip()
        instruct = (instruct_text or "").strip()

        if base_desc and instruct:
            description = f"{base_desc}, {instruct}"
        elif base_desc:
            description = base_desc
        elif instruct:
            description = instruct
        else:
            print("Warning: Design voice has no description or instruct. Using generic.")
            description = "A clear, natural speaking voice"

        wav_path, sr = self.generate_voice_design(description=description, sample_text=text)
        shutil.copy2(wav_path, output_path)
        # Remove the throwaway preview WAV now that it's copied to output_path, so
        # chunk/batch design-voice generation doesn't leak one preview per chunk
        # into designed_voices/previews/ (only the Designer route caps that dir).
        try:
            if os.path.abspath(wav_path) != os.path.abspath(output_path):
                os.remove(wav_path)
        except OSError:
            pass
        return True

    # ── LoRA voice generation ────────────────────────────────────

    def generate_lora_voice(self, text, instruct_text, voice_data, output_path):
        """Generate audio using a LoRA-finetuned Base model.

        The adapter directory must contain:
          - PEFT adapter weights (adapter_model.safetensors / adapter_config.json)
          - ref_sample.wav (reference audio for voice cloning prompt)
          - training_meta.json (with ref_sample_text)

        The LoRA weights refine voice identity beyond what the reference alone provides.
        """
        text = normalize_for_speech(text)
        try:
            import time

            adapter_path = voice_data.get("adapter_path")
            if not adapter_path:
                print(f"Error: No adapter_path in voice_data")
                return False

            # Resolve relative paths against project root
            if not os.path.isabs(adapter_path):
                adapter_path = _resolve_asset_path(adapter_path)

            if not os.path.isdir(adapter_path):
                # Auto-download built-in adapters from HF
                adapter_id = os.path.basename(adapter_path)
                if adapter_id.startswith("builtin_"):
                    print(f"Adapter {adapter_id} not downloaded, attempting auto-download...")
                    try:
                        try:
                            from .hf_utils import download_builtin_adapter
                        except ImportError:
                            from hf_utils import download_builtin_adapter
                        builtin_dir = os.path.dirname(adapter_path)
                        download_builtin_adapter(adapter_id, builtin_dir)
                    except Exception as e:
                        print(f"Error: Auto-download failed for {adapter_id}: {e}")
                        return False
                else:
                    print(f"Error: LoRA adapter path not found: {adapter_path}")
                    return False

            # Load reference audio and text from adapter directory
            ref_wav_path = os.path.join(adapter_path, "ref_sample.wav")
            meta_path = os.path.join(adapter_path, "training_meta.json")

            if not os.path.exists(ref_wav_path):
                print(f"Error: ref_sample.wav not found in {adapter_path}")
                return False
            if not os.path.exists(meta_path):
                print(f"Error: training_meta.json not found in {adapter_path}")
                return False

            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            ref_text = meta.get("ref_sample_text", "")
            if not ref_text:
                print(f"Error: ref_sample_text missing from training_meta.json")
                return False

            print(f"TTS [local lora] generating for adapter={os.path.basename(adapter_path)}, "
                  f"text='{text[:50]}...'")

            model = self._init_local_lora(adapter_path)

            prompt = self._ensure_lora_prompt(adapter_path, model, ref_text)

            # Build instruct_ids so the Base model can follow style prompts
            gen_extra = {}
            instruct = instruct_text or ""
            character_style = voice_data.get("character_style", "") or voice_data.get("default_style", "")
            if character_style:
                instruct = f"{instruct} {character_style}".strip()
            if instruct:
                instruct_formatted = f"<|im_start|>user\n{instruct}<|im_end|>\n"
                gen_extra["instruct_ids"] = model._tokenize_texts([instruct_formatted])

            t_start = time.time()
            wavs, sr = model.generate_voice_clone(
                text=text,
                voice_clone_prompt=prompt,
                non_streaming_mode=True,
                max_new_tokens=self._max_new_tokens,
                **gen_extra,
            )
            gen_time = time.time() - t_start

            if wavs is None or len(wavs) == 0:
                print(f"Error: No audio generated for: '{text[:50]}...'")
                return False

            audio = np.concatenate(wavs) if len(wavs) > 1 else wavs[0]
            duration = len(audio) / sr
            rtf = duration / gen_time if gen_time > 0 else 0
            print(f"TTS [local lora] done: {gen_time:.1f}s -> {duration:.1f}s audio ({rtf:.2f}x real-time)")
            self._save_wav(audio, sr, output_path)
            return True

        except Exception as e:
            import traceback
            print(f"Error generating LoRA voice: {e}")
            traceback.print_exc()
            return False

    def _ensure_lora_prompt(self, adapter_path, model, ref_text):
        """Build and cache the clone prompt used by LoRA inference."""
        if adapter_path not in self._lora_prompt_cache:
            audio_array, sample_rate = sf.read(os.path.join(adapter_path, "ref_sample.wav"))
            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=1)
            print("Creating clone prompt for LoRA adapter...")
            prompt = model.create_voice_clone_prompt(
                ref_audio=(audio_array, sample_rate),
                ref_text=ref_text,
                x_vector_only_mode=True,
            )
            self._lora_prompt_cache[adapter_path] = prompt
            print("Clone prompt cached for LoRA adapter.")
        return self._lora_prompt_cache[adapter_path]

    # ── Batch generation ─────────────────────────────────────────

    def generate_batch(self, chunks, voice_config, output_dir, batch_seed=-1):
        """Generate multiple audio files.

        Local mode: uses native list-based batch API for custom voices.
        External mode: sequential individual calls.

        Args:
            chunks: List of dicts with 'text', 'instruct', 'speaker', 'index' keys
            voice_config: Voice configuration dict
            output_dir: Directory to save output files
            batch_seed: Single seed for all generations (-1 for random)

        Returns:
            dict with 'completed' (list of indices) and 'failed' (list of (index, error) tuples)
        """
        results = {"completed": [], "failed": []}

        if not chunks:
            return results

        # Normalised once here so every downstream path - including
        # _local_batch_lora, which does not go through the single-item
        # methods - sees speakable text. Copies rather than mutating the
        # caller's chunk dicts (Rule 17); the caller still owns the originals.
        chunks = [{**c, "text": normalize_for_speech(c.get("text"))}
                  for c in chunks]

        # Reset torch.compile state to prevent progressive slowdown
        # from dynamo guard accumulation across batches
        if self._compile_codec_enabled:
            self._reset_compile_cache()

        # Separate chunks by voice type
        custom_chunks = []
        clone_chunks = []
        lora_chunks = []
        design_chunks = []
        ensemble_chunks = []

        for chunk in chunks:
            speaker = chunk.get("speaker")
            voice_data = voice_config.get(speaker, {})
            category = voice_category(voice_data)

            if category == "clone":
                clone_chunks.append(chunk)
            elif category == "lora":
                lora_chunks.append(chunk)
            elif category == "design":
                design_chunks.append(chunk)
            elif category == "ensemble":
                ensemble_chunks.append(chunk)
            else:
                custom_chunks.append(chunk)

        # Ensemble chunks need several different voices each, so they can't join
        # a same-voice batch — render them one at a time.
        if ensemble_chunks:
            batch_results = self._sequential_ensemble(ensemble_chunks, voice_config, output_dir)
            results["completed"].extend(batch_results["completed"])
            results["failed"].extend(batch_results["failed"])
            self._clear_gpu_cache()

        # Process custom voice chunks
        if custom_chunks:
            if self._mode == "local":
                batch_results = self._local_batch_custom(custom_chunks, voice_config, output_dir, batch_seed)
            else:
                batch_results = self._sequential_custom(custom_chunks, voice_config, output_dir, batch_seed)
            results["completed"].extend(batch_results["completed"])
            results["failed"].extend(batch_results["failed"])
            self._clear_gpu_cache()

        # Process clone voice chunks (batched by speaker in local mode)
        if clone_chunks:
            if self._mode == "local":
                batch_results = self._local_batch_clone(clone_chunks, voice_config, output_dir, batch_seed)
            else:
                batch_results = {"completed": [], "failed": []}
                for chunk in clone_chunks:
                    idx = chunk["index"]
                    output_path = os.path.join(output_dir, f"temp_batch_{idx}.wav")
                    try:
                        success = self.generate_clone_voice(
                            chunk["text"], chunk["speaker"], voice_config, output_path
                        )
                        if success:
                            batch_results["completed"].append(idx)
                        else:
                            batch_results["failed"].append((idx, "Clone voice generation failed"))
                    except Exception as e:
                        batch_results["failed"].append((idx, str(e)))
            results["completed"].extend(batch_results["completed"])
            results["failed"].extend(batch_results["failed"])
            self._clear_gpu_cache()

        # Process LoRA voice chunks (batched by adapter in local mode)
        if lora_chunks:
            if self._mode == "local":
                batch_results = self._local_batch_lora(lora_chunks, voice_config, output_dir, batch_seed)
            else:
                batch_results = {"completed": [], "failed": []}
                for chunk in lora_chunks:
                    idx = chunk["index"]
                    output_path = os.path.join(output_dir, f"temp_batch_{idx}.wav")
                    speaker = chunk.get("speaker")
                    voice_data = voice_config.get(speaker, {})
                    try:
                        success = self.generate_lora_voice(
                            text=chunk["text"],
                            instruct_text=chunk.get("instruct", ""),
                            voice_data=voice_data,
                            output_path=output_path,
                        )
                        if success:
                            batch_results["completed"].append(idx)
                        else:
                            batch_results["failed"].append((idx, "LoRA voice generation failed"))
                    except Exception as e:
                        batch_results["failed"].append((idx, str(e)))
            results["completed"].extend(batch_results["completed"])
            results["failed"].extend(batch_results["failed"])
            self._clear_gpu_cache()

        # Process design voice chunks (sequential — each line has unique description)
        if design_chunks:
            for chunk in design_chunks:
                idx = chunk["index"]
                output_path = os.path.join(output_dir, f"temp_batch_{idx}.wav")
                speaker = chunk.get("speaker")
                voice_data = voice_config.get(speaker, {})
                try:
                    success = self.generate_design_voice(
                        text=chunk["text"],
                        instruct_text=chunk.get("instruct", ""),
                        voice_data=voice_data,
                        output_path=output_path,
                    )
                    if success:
                        results["completed"].append(idx)
                    else:
                        results["failed"].append((idx, "Design voice generation failed"))
                except Exception as e:
                    results["failed"].append((idx, str(e)))

        return results

    # ── Connection test ──────────────────────────────────────────

    def set_sub_batch_size(self, max_items: int):
        """Set the sub-batch size for benchmarking. Public wrapper for _sub_batch_max_items."""
        self._sub_batch_max_items = max_items

    def run_benchmark_batch(self, chunks, voice_config, output_dir, batch_seed=-1):
        """Run a benchmark batch generation. Public wrapper for _local_batch_custom."""
        return self._local_batch_custom(chunks, voice_config, output_dir, batch_seed)

    def run_clone_benchmark_batch(self, chunks, voice_config, output_dir,
                                  batch_seed=-1):
        """Run a benchmark batch through the production clone implementation."""
        return self._local_batch_clone(chunks, voice_config, output_dir, batch_seed)

    def enable_codec_compilation(self):
        """Enable torch.compile for codec. Public wrapper for internal compilation."""
        if hasattr(self, '_compile_codec_enabled') and hasattr(self, '_compile_codec'):
            self._compile_codec_enabled = True
            if getattr(self, '_local_custom_model', None) is not None:
                self._compile_codec(self._local_custom_model)

    # ── Local backend methods ────────────────────────────────────

    def _local_generate_custom(self, text, instruct_text, speaker, voice_config, output_path):
        """Generate custom voice audio using local Qwen3-TTS model."""
        try:
            import torch

            voice_data = voice_config.get(speaker)
            if not voice_data:
                print(f"Warning: No voice configuration for '{speaker}'. Skipping.")
                return False

            voice = voice_data.get("voice", "Ryan")
            default_style = voice_data.get("default_style", "")
            seed = int(voice_data.get("seed", -1))

            instruct = instruct_text if instruct_text else (default_style if default_style else "neutral")

            import time

            print(f"TTS [local] generating with instruct='{instruct}' for text='{text[:50]}...'")

            model = self._init_local_custom()

            if seed >= 0:
                torch.manual_seed(seed)

            t_start = time.time()
            wavs, sr = model.generate_custom_voice(
                text=text,
                language=self._language,
                speaker=voice,
                instruct=instruct,
                non_streaming_mode=True,
                max_new_tokens=self._max_new_tokens,
            )
            gen_time = time.time() - t_start

            if wavs is None or len(wavs) == 0:
                print(f"Error: No audio generated for: '{text[:50]}...'")
                return False

            # wavs is a list of numpy arrays; concatenate them
            audio = np.concatenate(wavs) if len(wavs) > 1 else wavs[0]
            duration = len(audio) / sr
            rtf = duration / gen_time if gen_time > 0 else 0
            print(f"TTS [local] done: {gen_time:.1f}s -> {duration:.1f}s audio ({rtf:.2f}x real-time)")
            self._save_wav(audio, sr, output_path)
            return True

        except Exception as e:
            import traceback
            print(f"Error generating custom voice for '{speaker}': {e}")
            traceback.print_exc()
            return False

    def _local_generate_clone(self, text, speaker, voice_config, output_path):
        """Generate voice-cloned audio using local Qwen3-TTS Base model."""
        try:
            import torch

            voice_data = voice_config.get(speaker)
            if not voice_data:
                print(f"Warning: No voice configuration for '{speaker}'. Skipping.")
                return False

            seed = int(voice_data.get("seed", -1))

            import time

            print(f"TTS [local clone] generating for speaker='{speaker}', text='{text[:50]}...'")

            prompt = self._get_clone_prompt(speaker, voice_config)
            model = self._init_local_clone()

            if seed >= 0:
                torch.manual_seed(seed)

            t_start = time.time()
            wavs, sr = model.generate_voice_clone(
                text=text,
                voice_clone_prompt=prompt,
                non_streaming_mode=True,
                max_new_tokens=self._max_new_tokens,
            )
            gen_time = time.time() - t_start

            if wavs is None or len(wavs) == 0:
                print(f"Error: No audio generated for: '{text[:50]}...'")
                return False

            audio = np.concatenate(wavs) if len(wavs) > 1 else wavs[0]
            duration = len(audio) / sr
            rtf = duration / gen_time if gen_time > 0 else 0
            print(f"TTS [local clone] done: {gen_time:.1f}s -> {duration:.1f}s audio ({rtf:.2f}x real-time)")
            self._save_wav(audio, sr, output_path)
            return True

        except Exception as e:
            import traceback
            print(f"Error generating clone voice for '{speaker}': {e}")
            traceback.print_exc()
            return False

    def _local_batch_custom(self, chunks, voice_config, output_dir, batch_seed=-1):
        """Batch generate custom voice using native list API with sub-batching.

        Autoregressive batch generation runs for as long as the longest sequence.
        Shorter sequences waste compute on padding. To minimize this, chunks are
        sorted by text length and split into sub-batches when the length ratio
        exceeds the configured threshold. Sub-batching can be disabled entirely
        via config, in which case everything runs as one batch.
        """
        import torch
        import time

        results = {"completed": [], "failed": []}
        batch_peak_vram_gb = 0.0

        texts = []
        speakers = []
        instructs = []
        indices = []

        for chunk in chunks:
            idx = chunk["index"]
            text = chunk.get("text", "")
            instruct_text = chunk.get("instruct", "")
            speaker_name = chunk.get("speaker", "")

            voice_data = voice_config.get(speaker_name, {})
            voice = voice_data.get("voice", "Ryan")
            character_style = voice_data.get("character_style", "") or voice_data.get("default_style", "")

            instruct = instruct_text if instruct_text else "neutral"
            if character_style:
                instruct = f"{instruct} {character_style}"

            texts.append(text)
            speakers.append(voice)
            instructs.append(instruct)
            indices.append(idx)

        total_text_chars = sum(len(t) for t in texts)

        # Sort by text length to group similar-length chunks together.
        # This reduces wasted padding during autoregressive generation
        # (the LLM runs until ALL sequences finish, so short chunks
        # waste compute waiting for long ones).
        sort_order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        texts = [texts[i] for i in sort_order]
        speakers = [speakers[i] for i in sort_order]
        instructs = [instructs[i] for i in sort_order]
        indices = [indices[i] for i in sort_order]

        model = self._init_local_custom()

        # Warmup on first batch to pre-tune MIOpen/GPU solvers
        self.ensure_custom_warmup(model)

        # Clear stale GPU cache from any prior generation to avoid
        # fragmented VRAM blocking large batch allocations (ROCm especially).
        self._clear_gpu_cache()


        max_items = self._estimate_max_batch_size(
            model, max_text_chars=len(texts[-1]),
        )
        sub_batches = self._build_sub_batches(texts, max_items=max_items)

        print(f"Batch [local]: generating {len(texts)} chunks ({total_text_chars} chars) "
              f"in {len(sub_batches)} sub-batch(es)...")

        t_total_start = time.time()
        total_audio_duration = 0.0

        for sb_idx, (start, end) in enumerate(sub_batches):
            sb_texts = texts[start:end]
            sb_speakers = speakers[start:end]
            sb_instructs = instructs[start:end]
            sb_indices = indices[start:end]
            sb_chars = sum(len(t) for t in sb_texts)

            print(f"  Sub-batch {sb_idx+1}/{len(sub_batches)}: {len(sb_texts)} chunks "
                  f"({sb_chars} chars, {len(sb_texts[0])}-{len(sb_texts[-1])} chars/chunk)")

            try:
                if batch_seed >= 0:
                    torch.manual_seed(batch_seed)

                torch.cuda.reset_peak_memory_stats()
                t_start = time.time()
                wavs_list, sr = model.generate_custom_voice(
                    text=sb_texts,
                    language=[self._language] * len(sb_texts),
                    speaker=sb_speakers,
                    instruct=sb_instructs,
                    non_streaming_mode=True,
                    max_new_tokens=self._max_new_tokens,
                )
                gen_time = time.time() - t_start
                peak_gb = torch.cuda.max_memory_allocated() / 1e9
                batch_peak_vram_gb = max(batch_peak_vram_gb, peak_gb)
                print(f"  Peak VRAM sub-batch {sb_idx+1}: {peak_gb:.2f} GB")

                if wavs_list is None:
                    for idx in sb_indices:
                        results["failed"].append((idx, "Batch returned None"))
                    continue

                for idx in sb_indices[len(wavs_list):]:
                    results["failed"].append((idx, "Batch returned no waveform for this chunk"))

                sb_audio_duration = 0.0
                for i, (wav, idx) in enumerate(zip(wavs_list, sb_indices)):
                    try:
                        output_path = os.path.join(output_dir, f"temp_batch_{idx}.wav")
                        audio = self._concat_audio(wav)
                        self._save_wav(audio, sr, output_path)
                        results["completed"].append(idx)
                        duration = len(audio) / sr
                        sb_audio_duration += duration
                        print(f"    Chunk {idx} saved: {os.path.getsize(output_path)} bytes ({duration:.1f}s audio)")
                    except Exception as e:
                        print(f"    Error saving chunk {idx}: {e}")
                        results["failed"].append((idx, str(e)))

                total_audio_duration += sb_audio_duration
                sb_rtf = sb_audio_duration / gen_time if gen_time > 0 else 0
                print(f"  Sub-batch {sb_idx+1} done: {gen_time:.1f}s -> {sb_audio_duration:.1f}s audio ({sb_rtf:.2f}x RT)")

            except Exception as e:
                print(f"  Sub-batch {sb_idx+1} failed: {e}")
                for idx in sb_indices:
                    results["failed"].append((idx, f"Batch error: {e}"))

            # Free GPU memory between sub-batches to prevent VRAM exhaustion
            self._clear_gpu_cache()

        total_time = time.time() - t_total_start
        rtf = total_audio_duration / total_time if total_time > 0 else 0
        print(f"Batch total: {total_time:.1f}s -> {total_audio_duration:.1f}s audio ({rtf:.2f}x real-time)")



        results["peak_vram_gb"] = round(batch_peak_vram_gb, 2)
        return results

    def _local_batch_clone(self, chunks, voice_config, output_dir, batch_seed=-1):
        """Batch generate clone voices, grouped by speaker.

        Chunks sharing the same speaker (same reference audio) are batched
        together through generate_voice_clone(text=[list], ...).
        Sub-batching by text length is applied within each speaker group.
        """
        import torch
        import time

        results = {"completed": [], "failed": []}
        batch_peak_vram_gb = 0.0

        # Group chunks by speaker
        speaker_groups = {}
        for chunk in chunks:
            speaker = chunk.get("speaker", "")
            speaker_groups.setdefault(speaker, []).append(chunk)

        model = self._init_local_clone()

        self._clear_gpu_cache()


        t_total_start = time.time()
        total_audio_duration = 0.0

        for speaker, group in speaker_groups.items():
            try:
                prompt = self._get_clone_prompt(speaker, voice_config)
            except Exception as e:
                print(f"  Error building clone prompt for '{speaker}': {e}")
                for chunk in group:
                    results["failed"].append((chunk["index"], str(e)))
                continue

            texts = [c["text"] for c in group]
            indices = [c["index"] for c in group]

            # Sort by text length for sub-batching efficiency
            sort_order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
            texts = [texts[i] for i in sort_order]
            indices = [indices[i] for i in sort_order]

            # Estimate max batch size from VRAM + clone prompt overhead
            clone_tokens = prompt[0].ref_code.shape[0] if prompt[0].ref_code is not None else 0
            ref_text_chars = len(prompt[0].ref_text) if prompt[0].ref_text else 0
            max_items = self._estimate_max_batch_size(
                model, clone_tokens, ref_text_chars, len(texts[-1]),
            )
            sub_batches = self._build_sub_batches(texts, max_items=max_items)

            print(f"Batch [clone] speaker='{speaker}': {len(texts)} chunks "
                  f"in {len(sub_batches)} sub-batch(es)")

            for sb_idx, (start, end) in enumerate(sub_batches):
                sb_texts = texts[start:end]
                sb_indices = indices[start:end]

                print(f"  Sub-batch {sb_idx+1}/{len(sub_batches)}: {len(sb_texts)} chunks "
                      f"({len(sb_texts[0])}-{len(sb_texts[-1])} chars/chunk)")

                try:
                    # Seed for reproducible batch output when batch_seed is set
                    # (matches _local_batch_custom); otherwise random each run.
                    if batch_seed >= 0:
                        torch.manual_seed(batch_seed)
                    torch.cuda.reset_peak_memory_stats()
                    t_start = time.time()
                    wavs_list, sr = model.generate_voice_clone(
                        text=sb_texts,
                        voice_clone_prompt=prompt,
                        non_streaming_mode=True,
                        max_new_tokens=self._max_new_tokens,
                    )
                    gen_time = time.time() - t_start
                    peak_gb = torch.cuda.max_memory_allocated() / 1e9
                    batch_peak_vram_gb = max(batch_peak_vram_gb, peak_gb)
                    print(f"  Peak VRAM clone sub-batch {sb_idx+1}: {peak_gb:.2f} GB")

                    if wavs_list is None:
                        for idx in sb_indices:
                            results["failed"].append((idx, "Batch returned None"))
                        continue

                    for idx in sb_indices[len(wavs_list):]:
                        results["failed"].append((idx, "Batch returned no waveform for this chunk"))

                    sb_audio_duration = 0.0
                    for wav, idx in zip(wavs_list, sb_indices):
                        try:
                            output_path = os.path.join(output_dir, f"temp_batch_{idx}.wav")
                            audio = self._concat_audio(wav)
                            self._save_wav(audio, sr, output_path)
                            results["completed"].append(idx)
                            duration = len(audio) / sr
                            sb_audio_duration += duration
                        except Exception as e:
                            print(f"    Error saving chunk {idx}: {e}")
                            results["failed"].append((idx, str(e)))

                    total_audio_duration += sb_audio_duration
                    sb_rtf = sb_audio_duration / gen_time if gen_time > 0 else 0
                    print(f"  Sub-batch {sb_idx+1} done: {gen_time:.1f}s -> {sb_audio_duration:.1f}s audio ({sb_rtf:.2f}x RT)")

                except Exception as e:
                    print(f"  Sub-batch {sb_idx+1} failed: {e}")
                    for idx in sb_indices:
                        results["failed"].append((idx, f"Batch error: {e}"))

                self._clear_gpu_cache()

        total_time = time.time() - t_total_start
        rtf = total_audio_duration / total_time if total_time > 0 else 0
        print(f"Batch [clone] total: {total_time:.1f}s -> {total_audio_duration:.1f}s audio ({rtf:.2f}x real-time)")



        results["peak_vram_gb"] = round(batch_peak_vram_gb, 2)
        return results

    def _local_batch_lora(self, chunks, voice_config, output_dir, batch_seed=-1):
        """Batch generate LoRA voices, grouped by adapter.

        Chunks sharing the same adapter are batched together through
        generate_voice_clone(text=[list], instruct_ids=[list], ...).
        Sub-batching by text length is applied within each adapter group.
        """
        import torch
        import time

        results = {"completed": [], "failed": []}

        # Group chunks by adapter_path (resolved to absolute). Two different
        # speakers can share the same adapter (e.g. aliases, or one trained
        # voice reused for two characters) while having different
        # character_style/default_style - so each chunk keeps its own
        # voice_data instead of the group inheriting just the first chunk's.
        adapter_groups = {}  # adapter_path -> [(chunk, voice_data), ...]
        for chunk in chunks:
            speaker = chunk.get("speaker", "")
            voice_data = voice_config.get(speaker, {})
            adapter_path = voice_data.get("adapter_path", "")

            if not adapter_path:
                results["failed"].append((chunk["index"], "No adapter_path"))
                continue

            if not os.path.isabs(adapter_path):
                adapter_path = _resolve_asset_path(adapter_path)

            adapter_groups.setdefault(adapter_path, []).append((chunk, voice_data))

        self._clear_gpu_cache()

        t_total_start = time.time()
        total_audio_duration = 0.0

        for adapter_path, group_entries in adapter_groups.items():
            group = [c for c, _ in group_entries]
            if not os.path.isdir(adapter_path):
                # Auto-download built-in adapters from HF, mirroring the
                # single-chunk generate_lora_voice path — otherwise a built-in
                # LoRA voice that hasn't been fetched yet fails every chunk of a
                # batch render while single-chunk regen of the same voice works.
                adapter_id = os.path.basename(adapter_path)
                downloaded = False
                if adapter_id.startswith("builtin_"):
                    print(f"  Adapter {adapter_id} not downloaded, attempting auto-download...")
                    try:
                        from hf_utils import download_builtin_adapter
                        download_builtin_adapter(adapter_id, os.path.dirname(adapter_path))
                        downloaded = os.path.isdir(adapter_path)
                    except Exception as e:
                        print(f"  Error: Auto-download failed for {adapter_id}: {e}")
                if not downloaded:
                    print(f"  Error: adapter path not found: {adapter_path}")
                    for chunk in group:
                        results["failed"].append((chunk["index"], f"Adapter not found: {adapter_path}"))
                    continue

            # Load adapter and build/get clone prompt
            try:
                ref_wav_path = os.path.join(adapter_path, "ref_sample.wav")
                meta_path = os.path.join(adapter_path, "training_meta.json")
                if not os.path.exists(ref_wav_path) or not os.path.exists(meta_path):
                    raise FileNotFoundError(f"Missing ref_sample.wav or training_meta.json in {adapter_path}")

                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                ref_text = meta.get("ref_sample_text", "")
                if not ref_text:
                    raise ValueError("ref_sample_text missing from training_meta.json")

                model = self._init_local_lora(adapter_path)

                prompt = self._ensure_lora_prompt(adapter_path, model, ref_text)
            except Exception as e:
                print(f"  Error loading LoRA adapter {os.path.basename(adapter_path)}: {e}")
                for chunk in group:
                    results["failed"].append((chunk["index"], str(e)))
                continue

            texts = [c["text"] for c in group]
            instructs_raw = [c.get("instruct", "") for c in group]
            character_styles = [(vd.get("character_style", "") or vd.get("default_style", ""))
                                for _, vd in group_entries]
            indices = [c["index"] for c in group]

            # Sort by text length
            sort_order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
            texts = [texts[i] for i in sort_order]
            instructs_raw = [instructs_raw[i] for i in sort_order]
            character_styles = [character_styles[i] for i in sort_order]
            indices = [indices[i] for i in sort_order]

            # Estimate max batch size from VRAM + clone prompt overhead
            clone_tokens = prompt[0].ref_code.shape[0] if prompt[0].ref_code is not None else 0
            ref_text_chars = len(prompt[0].ref_text) if prompt[0].ref_text else 0
            max_items = self._estimate_max_batch_size(
                model, clone_tokens, ref_text_chars, len(texts[-1]),
            )
            sub_batches = self._build_sub_batches(texts, max_items=max_items)

            print(f"Batch [lora] adapter='{os.path.basename(adapter_path)}': {len(texts)} chunks "
                  f"in {len(sub_batches)} sub-batch(es)")

            for sb_idx, (start, end) in enumerate(sub_batches):
                sb_texts = texts[start:end]
                sb_instructs = instructs_raw[start:end]
                sb_character_styles = character_styles[start:end]
                sb_indices = indices[start:end]

                print(f"  Sub-batch {sb_idx+1}/{len(sub_batches)}: {len(sb_texts)} chunks "
                      f"({len(sb_texts[0])}-{len(sb_texts[-1])} chars/chunk)")

                try:
                    # Seed for reproducible batch output when batch_seed is set
                    # (matches _local_batch_custom); otherwise random each run.
                    if batch_seed >= 0:
                        torch.manual_seed(batch_seed)
                    # Build instruct_ids list for this sub-batch
                    instruct_ids = []
                    for inst, character_style in zip(sb_instructs, sb_character_styles):
                        instruct = inst or ""
                        if character_style:
                            instruct = f"{instruct} {character_style}".strip()
                        if instruct:
                            instruct_formatted = f"<|im_start|>user\n{instruct}<|im_end|>\n"
                            instruct_ids.append(model._tokenize_texts([instruct_formatted])[0])
                        else:
                            instruct_ids.append(None)

                    gen_extra = {}
                    if any(iid is not None for iid in instruct_ids):
                        gen_extra["instruct_ids"] = instruct_ids

                    t_start = time.time()
                    wavs_list, sr = model.generate_voice_clone(
                        text=sb_texts,
                        voice_clone_prompt=prompt,
                        non_streaming_mode=True,
                        max_new_tokens=self._max_new_tokens,
                        **gen_extra,
                    )
                    gen_time = time.time() - t_start

                    if wavs_list is None:
                        for idx in sb_indices:
                            results["failed"].append((idx, "Batch returned None"))
                        continue

                    for idx in sb_indices[len(wavs_list):]:
                        results["failed"].append((idx, "Batch returned no waveform for this chunk"))

                    sb_audio_duration = 0.0
                    for wav, idx in zip(wavs_list, sb_indices):
                        try:
                            output_path = os.path.join(output_dir, f"temp_batch_{idx}.wav")
                            audio = self._concat_audio(wav)
                            self._save_wav(audio, sr, output_path)
                            results["completed"].append(idx)
                            duration = len(audio) / sr
                            sb_audio_duration += duration
                        except Exception as e:
                            print(f"    Error saving chunk {idx}: {e}")
                            results["failed"].append((idx, str(e)))

                    total_audio_duration += sb_audio_duration
                    sb_rtf = sb_audio_duration / gen_time if gen_time > 0 else 0
                    print(f"  Sub-batch {sb_idx+1} done: {gen_time:.1f}s -> {sb_audio_duration:.1f}s audio ({sb_rtf:.2f}x RT)")

                except Exception as e:
                    print(f"  Sub-batch {sb_idx+1} failed: {e}")
                    for idx in sb_indices:
                        results["failed"].append((idx, f"Batch error: {e}"))

                self._clear_gpu_cache()

        total_time = time.time() - t_total_start
        rtf = total_audio_duration / total_time if total_time > 0 else 0
        print(f"Batch [lora] total: {total_time:.1f}s -> {total_audio_duration:.1f}s audio ({rtf:.2f}x real-time)")



        return results

    # ── External backend methods ─────────────────────────────────

    def _external_generate_custom(self, text, instruct_text, speaker, voice_config, output_path):
        """Generate custom voice audio via external Gradio server."""
        try:
            voice_data = voice_config.get(speaker)
            if not voice_data:
                print(f"Warning: No voice configuration for '{speaker}'. Skipping.")
                return False

            voice = voice_data.get("voice", "Ryan")
            default_style = voice_data.get("default_style", "")
            seed = int(voice_data.get("seed", -1))

            instruct = instruct_text if instruct_text else (default_style if default_style else "neutral")

            print(f"TTS [external] generating with instruct='{instruct}' for text='{text[:50]}...'")

            client = self._init_external()

            result = client.predict(
                text=text,
                language=self._language,
                speaker=voice,
                instruct=instruct,
                model_size="1.7B",
                seed=seed,
                api_name="/generate_custom_voice"
            )

            generated_audio_filepath = result[0]
            if not generated_audio_filepath or not os.path.exists(generated_audio_filepath):
                print(f"Error: No audio file generated for: '{text[:50]}...'")
                return False

            if os.path.getsize(generated_audio_filepath) == 0:
                print(f"Error: Generated audio file is empty for: '{text[:50]}...'")
                return False

            shutil.copy(generated_audio_filepath, output_path)
            return True

        except Exception as e:
            import traceback
            print(f"Error generating custom voice for '{speaker}': {e}")
            traceback.print_exc()
            return False

    def _external_generate_clone(self, text, speaker, voice_config, output_path):
        """Generate voice-cloned audio via external Gradio server."""
        try:
            from gradio_client import handle_file

            voice_data = voice_config.get(speaker)
            if not voice_data:
                print(f"Warning: No voice configuration for '{speaker}'. Skipping.")
                return False

            ref_audio = voice_data.get("ref_audio")
            ref_text = voice_data.get("ref_text")
            seed = int(voice_data.get("seed", -1))

            if not ref_audio or not ref_text:
                print(f"Warning: Clone voice for '{speaker}' missing ref_audio or ref_text. Skipping.")
                return False

            # Resolve relative paths against project root
            if not os.path.isabs(ref_audio):
                ref_audio = _resolve_asset_path(ref_audio)

            if not os.path.exists(ref_audio):
                print(f"Warning: Reference audio not found for '{speaker}': {ref_audio}")
                return False

            client = self._init_external()

            result = client.predict(
                handle_file(ref_audio),
                ref_text,
                text,
                "Auto",
                False,       # use_xvector_only
                "1.7B",
                200,         # max_chunk_chars
                0,           # chunk_gap
                seed,
                api_name="/generate_voice_clone"
            )

            generated_audio_filepath = result[0]
            if not generated_audio_filepath or not os.path.exists(generated_audio_filepath):
                print(f"Error: No audio file generated for: '{text[:50]}...'")
                return False

            if os.path.getsize(generated_audio_filepath) == 0:
                print(f"Error: Generated audio file is empty for: '{text[:50]}...'")
                return False

            shutil.copy(generated_audio_filepath, output_path)
            return True

        except Exception as e:
            import traceback
            print(f"Error generating clone voice for '{speaker}': {e}")
            traceback.print_exc()
            return False

    def _sequential_ensemble(self, chunks, voice_config, output_dir):
        """Render ensemble chunks one at a time (each needs several voices)."""
        results = {"completed": [], "failed": []}

        for chunk in chunks:
            idx = chunk["index"]
            output_path = os.path.join(output_dir, f"temp_batch_{idx}.wav")
            try:
                success = self.generate_voice(
                    chunk.get("text", ""),
                    chunk.get("instruct", ""),
                    chunk.get("speaker"),
                    voice_config,
                    output_path,
                )
                if success:
                    results["completed"].append(idx)
                else:
                    results["failed"].append((idx, "Ensemble generation failed"))
            except Exception as e:
                results["failed"].append((idx, str(e)))

        return results

    def _sequential_custom(self, chunks, voice_config, output_dir, batch_seed=-1):
        """Sequential custom voice generation for external mode (no native batch)."""
        results = {"completed": [], "failed": []}

        for chunk in chunks:
            idx = chunk["index"]
            output_path = os.path.join(output_dir, f"temp_batch_{idx}.wav")
            try:
                success = self.generate_custom_voice(
                    chunk.get("text", ""),
                    chunk.get("instruct", ""),
                    chunk.get("speaker", ""),
                    voice_config,
                    output_path,
                )
                if success:
                    results["completed"].append(idx)
                    print(f"Batch chunk {idx} saved: {os.path.getsize(output_path)} bytes")
                else:
                    results["failed"].append((idx, "Custom voice generation failed"))
            except Exception as e:
                results["failed"].append((idx, str(e)))

        return results

    # ── Utility ──────────────────────────────────────────────────

    @staticmethod
    def _save_wav(audio_array, sample_rate, output_path):
        """Save a numpy audio array as a WAV file."""
        # Ensure numpy array
        if not isinstance(audio_array, np.ndarray):
            audio_array = np.array(audio_array)
        # Flatten if needed
        if audio_array.ndim > 1:
            audio_array = audio_array.flatten()
        sf.write(output_path, audio_array, sample_rate)
