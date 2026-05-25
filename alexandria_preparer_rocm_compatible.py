#!/usr/bin/env python3
"""
Alexandria Master Preparer - ROCm Compatible Version
Handles CUDA/ROCm version mismatches gracefully
"""

# Suppress known noisy warnings before any other imports
import warnings
warnings.filterwarnings("ignore", message=".*torchcodec is not installed correctly.*")
warnings.filterwarnings("ignore", message=".*expandable_segments not supported.*")
warnings.filterwarnings("ignore", message=".*Flash Efficient attention.*")
warnings.filterwarnings("ignore", message=".*Mem Efficient attention.*")

import os
import sys
import tempfile

# Force llama_cpp to load first to ensure system ROCm libs are prioritized over torch's bundled ones.
# Do NOT defer this import — llama_cpp's ggml_cuda_init() must bind to the system HIP libs before
# torch's bundled copies get loaded, otherwise ROCm detection fails at runtime.
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

# Add insanely-fast-whisper-rocm to path if available
script_dir = os.path.dirname(os.path.abspath(__file__))
ifw_path = os.path.join(script_dir, "insanely-fast-whisper-rocm")
if os.path.exists(ifw_path):
    sys.path.insert(0, ifw_path)

# ROCm environment fixes
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HSA_ENABLE_SDMA"] = "0"
os.environ["GPU_MAX_HW_QUEUES"] = "2"

import argparse
import gc
import time
import logging
import json
import re
import subprocess
import difflib

# Shared alignment primitives (load_source, lexicon, find_best_match, ...).
# Only used when --source is passed; preparer remains zero-dep on this module
# for the legacy ASR-only workflow because nothing in the chunker calls into
# it unless source_state is populated.
import alexandria_alignment as alignment
import zipfile
import shutil
import soundfile as sf
import numpy as np
import traceback
from collections import deque
from typing import List, Dict, Optional
from datetime import datetime

# Deferred imports to avoid HIP/CUDA context contamination between phases
torch = None
librosa = None

def _lazy_import_torch():
    global torch
    if torch is None:
        import torch as t
        torch = t
    return torch

def _lazy_import_librosa():
    global librosa
    if librosa is None:
        import librosa as l
        librosa = l
    return librosa

def _lazy_import_intervaltree():
    global IntervalTree, Interval
    if "IntervalTree" not in globals():
        from intervaltree import IntervalTree as IT, Interval as I
        IntervalTree = IT
        Interval = I
    return IntervalTree, Interval


# Setup logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"alexandria_preparer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logger = logging.getLogger("alexandria")
logger.setLevel(logging.DEBUG)

# File handler (detailed)
fh = logging.FileHandler(log_file)
fh.setLevel(logging.DEBUG)
file_format = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] %(message)s')
fh.setFormatter(file_format)

# Console handler (info and above)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
console_format = logging.Formatter('[%(levelname)s] %(message)s')
ch.setFormatter(console_format)

logger.addHandler(fh)
logger.addHandler(ch)

# Progress tracker
class ProgressTracker:
    def __init__(self):
        self.steps = []
        self.current_step = 0

    def add_step(self, name):
        self.steps.append(name)

    def start(self, step_name):
        self.current_step = next((i for i, s in enumerate(self.steps) if s == step_name), 0)
        progress = f"[{self.current_step + 1}/{len(self.steps)}]"
        logger.info(f"▶ {progress} {step_name}...")

    def complete(self):
        logger.info(f"✓ Step {self.current_step + 1}/{len(self.steps)} completed")

progress = ProgressTracker()
progress.add_step("Validate inputs")
progress.add_step("Load audio")
progress.add_step("Transcribe audio")
progress.add_step("Annotate chunks")
progress.add_step("Create output dataset")

logger.info(f"=== Alexandria Master Preparer Started ===")
logger.info(f"Log file: {log_file}")
logger.info(f"Python version: {sys.version}")

def log_torch_info():
    t = _lazy_import_torch()
    logger.info(f"PyTorch version: {t.__version__}")

# Check available ASR options
INSANELY_FAST_WHISPER_AVAILABLE = False
WHISPERX_AVAILABLE = False
TRANSFORMERS_WHISPER_AVAILABLE = False

INSANELY_FAST_WHISPER_AVAILABLE = os.path.exists(
    os.path.join(script_dir, "insanely-fast-whisper-rocm")
)
if INSANELY_FAST_WHISPER_AVAILABLE:
    logger.info("✓ Insanely Fast Whisper (ROCm) available")
else:
    logger.debug("Insanely Fast Whisper not found in project directory")

try:
    from whisperx import asr as whisperx_asr
    from whisperx import alignment as whisperx_alignment
    WHISPERX_AVAILABLE = True
    logger.info("✓ WhisperX-ROCm available")
except ImportError as e:
    logger.debug(f"WhisperX not available: {e}")

try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
    logger.info("✓ pyannote.audio available")
except ImportError as e:
    PYANNOTE_AVAILABLE = False
    logger.debug(f"pyannote.audio not available: {e}")

try:
    from intervaltree import IntervalTree, Interval
    INTERVALTREE_AVAILABLE = True
    logger.info("✓ intervaltree available")
except ImportError as e:
    INTERVALTREE_AVAILABLE = False
    logger.debug(f"intervaltree not available: {e}")

try:
    from transformers import pipeline
    TRANSFORMERS_WHISPER_AVAILABLE = True
    logger.info("✓ Transformers available")
except ImportError as e:
    logger.debug(f"Transformers not available: {e}")

if LLAMA_CPP_AVAILABLE:
    logger.info("✓ llama-cpp-python available")
else:
    logger.critical("llama-cpp-python required. Install with: pip install llama-cpp-python")
    sys.exit(1)

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("whisperx").setLevel(logging.ERROR)

def clear_vram():
    """Clear GPU memory and sync."""
    gc.collect()
    t = _lazy_import_torch()
    if t.cuda.is_available():
        t.cuda.empty_cache()
        t.cuda.synchronize()
        # Note: GPU cache clearing is logged silently to reduce spam

def get_gpu_stats():
    """Get current GPU memory and utilization stats."""
    t = _lazy_import_torch()
    if not t.cuda.is_available():
        return None

    stats = {}
    try:
        # Memory stats (works for both NVIDIA and AMD ROCm)
        allocated = t.cuda.memory_allocated() / 1e9  # GB
        reserved = t.cuda.memory_reserved() / 1e9    # GB
        total = t.cuda.get_device_properties(0).total_memory / 1e9  # GB

        stats['allocated_gb'] = allocated
        stats['reserved_gb'] = reserved
        stats['total_gb'] = total
        stats['allocated_percent'] = (allocated / total * 100) if total > 0 else 0

        # Try to get utilization via rocm-smi for AMD GPUs
        try:
            result = subprocess.run(
                ['/opt/rocm/bin/rocm-smi', '--showuse', '--json'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Filter out warning lines and parse JSON
                json_lines = [line for line in result.stdout.split('\n') if line.strip().startswith('{')]
                if json_lines:
                    data = json.loads(json_lines[0])
                    # rocm-smi format: {"card0": {"GPU use (%)": "value"}}
                    for card_key, card_data in data.items():
                        gpu_use_str = card_data.get('GPU use (%)', 'N/A')
                        if gpu_use_str != 'N/A':
                            stats['utilization_percent'] = float(gpu_use_str)
                        break  # Just get first GPU
            else:
                logger.debug(f"rocm-smi returned error: {result.returncode}, stderr: {result.stderr}")
                stats['utilization_percent'] = None
        except FileNotFoundError as e:
            logger.debug(f"rocm-smi not found: {e}")
            stats['utilization_percent'] = None
        except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError) as e:
            logger.debug(f"rocm-smi parse error: {e}")
            stats['utilization_percent'] = None
        except Exception as e:
            logger.debug(f"rocm-smi unexpected error: {e}")
            stats['utilization_percent'] = None

    except Exception as e:
        logger.debug(f"Could not get GPU stats: {e}")
        return None

    return stats

def log_gpu_stats(label=""):
    """Log GPU memory and utilization statistics."""
    stats = get_gpu_stats()
    if not stats:
        return

    label_str = f" ({label})" if label else ""
    logger.info(f"GPU Usage{label_str}:")
    logger.info(f"  ├─ Memory: {stats['allocated_gb']:.2f}GB / {stats['total_gb']:.2f}GB ({stats['allocated_percent']:.1f}%)")
    if stats.get('utilization_percent') is not None:
        logger.info(f"  └─ Utilization: {stats['utilization_percent']:.1f}%")
    else:
        logger.info(f"  └─ Utilization: (rocm-smi unavailable)")

def format_duration(seconds):
    """Format seconds as Xh Ym Zs (or smaller unit when applicable)."""
    seconds = max(0, int(seconds))
    hours = seconds // 3600
    mins = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours}h {mins}m"
    elif mins > 0:
        return f"{mins}m {secs}s"
    else:
        return f"{secs}s"

# ── Oversized-WAV handling (>4 GiB data-chunk header wrap) ───────────────────
# Standard WAV uses a 32-bit unsigned chunk-size field, so any WAV whose audio
# `data` chunk exceeds 4 GiB wraps that field and reports a bogus header
# duration (only the bytes after the modulus). `soundfile`/`librosa.load`
# honor the wrapped header and silently truncate. Audiobook WAVs at this
# project's rates (44.1 kHz stereo PCM_16) hit the wrap at ~6.8 hours — every
# full-length audiobook in the test corpus is affected.
#
# We detect the wrap by comparing on-disk file size against header-implied
# data size. When detected, we route the load through ffmpeg with
# `-ignore_length 1`, which makes the WAV demuxer ignore the chunk-size field
# and decode until EOF, giving us the full audio. Streaming via subprocess
# also avoids materialising the entire native-rate float32 array in RAM
# (a 27-hour 44.1 kHz mono float32 buffer is ~16 GB; the user has files
# that long).

def _wav_overflow_info(path):
    """Return (is_oversized, true_duration_s, header_duration_s) for a WAV
    file. `is_oversized` is True when the on-disk size implies more audio
    than the header reports (the >4 GiB data-chunk-size wrap). For non-WAV
    files the function returns (False, header_dur, header_dur).
    """
    try:
        info = sf.info(path)
    except Exception:
        return False, 0.0, 0.0
    header_dur = info.duration
    if info.format != 'WAV':
        return False, header_dur, header_dur
    # Bytes per sample frame. soundfile exposes subtypes like PCM_16/PCM_24/PCM_32/FLOAT.
    subtype_bytes = {'PCM_16': 2, 'PCM_24': 3, 'PCM_32': 4, 'FLOAT': 4, 'DOUBLE': 8}
    bps = subtype_bytes.get(info.subtype, 2)
    file_size = os.path.getsize(path)
    # Subtract a generous 1 MB for header/junk chunks — true audio bytes
    # is essentially file_size minus a kilobyte or two of metadata.
    audio_bytes_estimate = max(0, file_size - 1024 * 1024)
    true_dur = audio_bytes_estimate / (info.samplerate * info.channels * bps)
    is_oversized = file_size > 2**32 and true_dur > header_dur * 1.5
    return is_oversized, true_dur if is_oversized else header_dur, header_dur


def _ffmpeg_decode_to_wav(src_path, dst_wav_path, target_sr, mono=True):
    """Decode an audio file to a 16-bit PCM WAV via ffmpeg, ignoring any
    bogus WAV chunk-size header. Returns the resulting file's on-disk size.
    Raises subprocess.CalledProcessError on ffmpeg failure.
    """
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ignore_length", "1",
        "-i", src_path,
        "-ac", "1" if mono else "2",
        "-ar", str(target_sr),
        "-c:a", "pcm_s16le",
        dst_wav_path,
    ]
    logger.debug(f"  ffmpeg decode → {dst_wav_path} ({target_sr}Hz, {'mono' if mono else 'stereo'})")
    subprocess.run(cmd, check=True)
    return os.path.getsize(dst_wav_path)


def _ffmpeg_decode_to_numpy(src_path, target_sr, mono=True):
    """Decode an audio file to a float32 numpy array via ffmpeg piped to
    s16le PCM. Bypasses soundfile entirely, so it works on >4 GiB WAVs
    whose data-chunk header has overflowed. Returns a 1-D float32 array
    normalised to [-1.0, 1.0].
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ignore_length", "1",
        "-i", src_path,
        "-ac", "1" if mono else "2",
        "-ar", str(target_sr),
        "-f", "s16le",
        "-",
    ]
    logger.debug(f"  ffmpeg decode → numpy ({target_sr}Hz, {'mono' if mono else 'stereo'})")
    proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE)
    pcm = np.frombuffer(proc.stdout, dtype=np.int16)
    return (pcm.astype(np.float32) / 32768.0)


def validate_inputs(args):
    """Validate input files."""
    logger.info("Validating input files...")

    def _missing_path_hint(flag, path):
        """When a path isn't found, log an absolute resolved path + CWD so
        the user can see whether they hit a relative-path-vs-CWD problem.
        Reads as: 'we looked here, and our working dir is X, so try…'.
        """
        resolved = os.path.abspath(path)
        cwd = os.getcwd()
        logger.error(f"{flag}: file not found")
        logger.error(f"  requested        : {path}")
        if resolved != path:
            logger.error(f"  resolved to      : {resolved}")
        logger.error(f"  current dir      : {cwd}")
        if not os.path.isabs(path):
            logger.error(f"  hint: pass an absolute path, or 'cd' into the project "
                         f"directory before running this script "
                         f"(model/source paths are resolved relative to the "
                         f"working dir, not the script's location).")

    if not os.path.exists(args.audio):
        _missing_path_hint("--audio", args.audio)
        sys.exit(1)
    logger.debug(f"Audio file exists: {args.audio}")

    if not args.skip_annotation and not os.path.exists(args.model):
        _missing_path_hint("--model", args.model)
        sys.exit(1)
    if not args.skip_annotation:
        logger.debug(f"Model file exists: {args.model}")

    # Validate fallback eagerly so we fail fast on a typo'd path
    if not args.skip_annotation and args.fallback_model and not os.path.exists(args.fallback_model):
        _missing_path_hint("--fallback-model", args.fallback_model)
        logger.error("Either fix the path or omit --fallback-model")
        sys.exit(1)

    # Validate --source eagerly too, otherwise a typo only surfaces after
    # ASR transcription (potentially hours into the run).
    source_path = getattr(args, 'source', None)
    if source_path and not os.path.exists(source_path):
        _missing_path_hint("--source", source_path)
        sys.exit(1)
    if source_path:
        logger.debug(f"Source file exists: {source_path}")

    try:
        info = sf.info(args.audio)
        is_oversized, true_dur, header_dur = _wav_overflow_info(args.audio)
        if is_oversized:
            logger.warning(
                f"⚠ Oversized WAV detected: header says {header_dur:.1f}s "
                f"({header_dur/60:.1f} min) but file size implies "
                f"{true_dur:.1f}s ({true_dur/3600:.2f} hr). "
                f"WAV data-chunk-size field is 32-bit and has wrapped — "
                f"ffmpeg `-ignore_length 1` will be used to read the full audio."
            )
            logger.info(
                f"Audio file: {info.samplerate}Hz, {true_dur:.2f}s "
                f"(header reported {header_dur:.2f}s — wrapped), {info.channels}ch"
            )
            return true_dur
        logger.info(f"Audio file: {info.samplerate}Hz, {info.duration:.2f}s, {info.channels}ch")
        return info.duration
    except Exception as e:
        logger.error(f"Invalid audio file: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)

def transcribe_with_whisperx_cpu(audio_16k: np.ndarray, language: str = "en") -> tuple:
    """Transcribe using WhisperX on CPU (stable, no GPU conflicts)."""
    if not WHISPERX_AVAILABLE:
        raise ImportError("WhisperX not available")

    # Force CPU to avoid CUDA/ROCm driver conflicts
    device = "cpu"
    compute_type = "int8"

    logger.info(f"Starting WhisperX transcription on CPU (stable mode)...")

    try:
        logger.debug(f"Loading WhisperX base model (device={device}, compute_type={compute_type})...")
        model = whisperx_asr.load_model("base", device, compute_type=compute_type)
        logger.info("✓ WhisperX model loaded")

        logger.info("Transcribing audio (this may take a while on CPU)...")
        result = model.transcribe(audio_16k, batch_size=1, language=language)
        detected_lang = result.get("language", language)
        logger.info(f"✓ Transcription complete, detected language: {detected_lang}")

        del model
        clear_vram()

        logger.debug(f"Loading alignment model for language: {detected_lang}...")
        model_a, metadata_a = whisperx_alignment.load_align_model(
            language_code=detected_lang,
            device=device
        )
        logger.info("✓ Alignment model loaded")

        logger.debug("Running word-level alignment...")
        aligned = whisperx_alignment.align(
            result["segments"],
            model_a,
            metadata_a,
            audio_16k,
            device,
            return_char_alignments=False
        )
        logger.info("✓ Word-level alignment complete")

        del model_a
        clear_vram()

        # Extract word segments
        word_segments = []
        for segment in aligned["segments"]:
            if "words" in segment:
                for word_info in segment["words"]:
                    if "start" in word_info and "end" in word_info:
                        word_segments.append({
                            "word": word_info["word"].strip(),
                            "start": word_info["start"],
                            "end": word_info["end"],
                            "confidence": word_info.get("score", 1.0)
                        })

        logger.info(f"✓ WhisperX complete: {len(word_segments)} words extracted")
        return word_segments, detected_lang

    except Exception as e:
        logger.error(f"WhisperX transcription failed: {e}")
        logger.debug(traceback.format_exc())
        raise

def transcribe_with_wav2vec2(audio_16k: np.ndarray, language: str = "en", limit: int = None) -> tuple:
    """Use Wav2Vec2 for continuous context-aware transcription with CTC word alignment."""
    if not TRANSFORMERS_WHISPER_AVAILABLE:
        raise ImportError("Transformers not available")

    logger.info("▶ Initializing Wav2Vec2 ASR (CTC-aligned word timestamps)...")
    logger.info(f"  ├─ Model: facebook/wav2vec2-large-960h")
    logger.info(f"  ├─ Device: GPU (CUDA/ROCm)")
    logger.info(f"  └─ Language: {language}")

    try:
        from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
        import torch as torch_module

        logger.debug("Loading Wav2Vec2 processor and model...")
        device_str = "cuda" if torch_module.cuda.is_available() else "cpu"

        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
        model = model.to(device_str)
        model.eval()

        logger.info("✓ Wav2Vec2 model loaded to GPU")
        log_gpu_stats("after model load")

        # Frame rate: for wav2vec2-large-960h, CNN downsamples 16kHz audio by 320 → 50 frames/sec
        inputs_to_logits_ratio = getattr(model.config, "inputs_to_logits_ratio", 320)
        time_per_frame = inputs_to_logits_ratio / 16000.0  # seconds per logit frame (~0.02s)

        chunk_length_secs = 30
        chunk_length = chunk_length_secs * 16000
        overlap_secs = 3
        overlap = overlap_secs * 16000
        stride = chunk_length - overlap
        half_overlap_secs = overlap_secs / 2.0

        audio_duration = len(audio_16k) / 16000.0

        # Compute chunk start positions, ensuring last chunk reaches audio end
        chunk_starts = list(range(0, max(1, len(audio_16k) - chunk_length + 1), stride))
        # Append final chunk for any remaining audio
        if not chunk_starts or chunk_starts[-1] + chunk_length < len(audio_16k):
            tail_start = max(0, len(audio_16k) - chunk_length)
            if not chunk_starts or tail_start > chunk_starts[-1]:
                chunk_starts.append(tail_start)
        num_chunks = len(chunk_starts)

        logger.info(f"  ├─ Context window: {chunk_length_secs}s")
        logger.info(f"  ├─ Overlap: {overlap_secs}s ({1.0/time_per_frame:.0f} Hz frame rate)")
        logger.info(f"  ├─ Word timestamps: CTC frame alignment (true per-word timing)")
        logger.info(f"  └─ Processing {num_chunks} chunks...")
        log_gpu_stats("before chunk processing")

        word_segments = []
        chunk_times = deque(maxlen=10)  # rolling avg for ETA

        for chunk_idx, sample_start in enumerate(chunk_starts):
            if limit and chunk_idx >= limit:
                logger.info(f"Limit of {limit} chunks reached for transcription.")
                break
            chunk_t0 = time.monotonic()
            chunk_end = min(sample_start + chunk_length, len(audio_16k))
            chunk = audio_16k[sample_start:chunk_end]
            chunk_offset_secs = sample_start / 16000.0
            chunk_end_secs = chunk_end / 16000.0

            with torch_module.no_grad():
                inputs = processor(chunk, sampling_rate=16000, return_tensors="pt", padding=True)
                inputs = {k: v.to(device_str) for k, v in inputs.items()}
                logits = model(**inputs).logits
                predicted_ids = torch_module.argmax(logits, dim=-1)
                
                # Get probabilities and confidence scores
                probs = torch_module.nn.functional.softmax(logits, dim=-1)
                confidence = torch_module.max(probs, dim=-1).values.squeeze().cpu().numpy()

            # CTC decode with word-level frame offsets
            decoded = processor.batch_decode(predicted_ids, output_word_offsets=True)
            word_offsets = decoded.word_offsets[0] if decoded.word_offsets else []

            # Determine "owned" region for this chunk to avoid double-counting overlap:
            #   - first chunk owns [chunk_start, chunk_end - half_overlap]
            #   - middle chunks own [chunk_start + half_overlap, chunk_end - half_overlap]
            #   - last chunk owns [chunk_start + half_overlap, audio_end]
            is_first = (chunk_idx == 0)
            is_last = (chunk_idx == num_chunks - 1)
            owned_start = chunk_offset_secs if is_first else chunk_offset_secs + half_overlap_secs
            owned_end = chunk_end_secs if is_last else chunk_end_secs - half_overlap_secs

            for wo in word_offsets:
                word_start = chunk_offset_secs + wo["start_offset"] * time_per_frame
                word_end = chunk_offset_secs + wo["end_offset"] * time_per_frame
                # Use word center to decide ownership (avoids splitting across chunks)
                word_center = (word_start + word_end) / 2.0
                if owned_start <= word_center < owned_end:
                    # Calculate average confidence for the word
                    start_frame = wo["start_offset"]
                    end_frame = wo["end_offset"]
                    word_confidence = np.mean(confidence[start_frame:end_frame]) if end_frame > start_frame else confidence[start_frame]
                    
                    word_segments.append({
                        "word": wo["word"].strip(),
                        "start": word_start,
                        "end": word_end,
                        "confidence": float(word_confidence)
                    })

            chunk_times.append(time.monotonic() - chunk_t0)

            if (chunk_idx + 1) % 50 == 0 or chunk_idx == num_chunks - 1:
                avg_chunk_s = sum(chunk_times) / len(chunk_times)
                remaining = (num_chunks - chunk_idx - 1) * avg_chunk_s
                logger.info(f"  ↳ Chunk {chunk_idx + 1}/{num_chunks} | avg {avg_chunk_s:.2f}s/chunk | ETA {format_duration(remaining)}")
                log_gpu_stats(f"chunk {chunk_idx + 1}/{num_chunks}")

        del processor, model
        clear_vram()

        logger.info(f"✓ Wav2Vec2 complete: {len(word_segments)} words extracted with CTC-aligned timestamps")
        if word_segments:
            logger.debug(f"  First word: '{word_segments[0]['word']}' @ {word_segments[0]['start']:.3f}-{word_segments[0]['end']:.3f}s")
            logger.debug(f"  Last word:  '{word_segments[-1]['word']}' @ {word_segments[-1]['start']:.3f}-{word_segments[-1]['end']:.3f}s")
        return word_segments, language

    except Exception as e:
        logger.error(f"Wav2Vec2 transcription failed: {e}")
        logger.debug(traceback.format_exc())
        raise

def transcribe_with_whisper_v3(audio_16k: np.ndarray, device: str) -> tuple:
    """Use OpenAI Whisper-large-v3 (best model, CPU stable)."""
    if not TRANSFORMERS_WHISPER_AVAILABLE:
        raise ImportError("Transformers not available")

    logger.info("Starting Whisper-large-v3 transcription...")

    try:
        # Use CPU to avoid CUDA/ROCm issues, v3 is fast enough
        pipe_device = -1
        logger.debug(f"Loading Whisper-large-v3 pipeline (device=CPU)...")

        pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3",
            device=pipe_device,
            return_timestamps="word"
        )
        logger.info("✓ Whisper-large-v3 pipeline loaded")

        logger.info("Transcribing audio with Whisper-v3...")
        result = pipe(audio_16k, return_timestamps="word")

        word_segments = []
        if "chunks" in result:
            for chunk in result["chunks"]:
                if "timestamp" in chunk and chunk["timestamp"]:
                    start, end = chunk["timestamp"]
                    if start is not None and end is not None:
                        word_segments.append({
                            "word": chunk["text"].strip(),
                            "start": start,
                            "end": end
                        })

        del pipe
        clear_vram()

        logger.info(f"✓ Whisper-v3 complete: {len(word_segments)} words transcribed")
        return word_segments, "en"

    except Exception as e:
        logger.error(f"Whisper-v3 failed: {e}")
        logger.debug(traceback.format_exc())
        raise

def transcribe_with_insanely_fast_whisper(audio_16k: np.ndarray, language: str = "en") -> tuple:
    """Use Insanely Fast Whisper ROCm (optimized for AMD GPUs)."""
    if not INSANELY_FAST_WHISPER_AVAILABLE:
        raise ImportError("Insanely Fast Whisper not available")

    logger.info("▶ Initializing Insanely Fast Whisper (ROCm optimized)...")
    audio_duration = len(audio_16k) / 16000
    logger.info(f"  ├─ Audio: {audio_duration:.1f}s @ {audio_16k.shape[0]} samples")
    logger.info(f"  └─ Language: {language}")

    temp_audio_file = None
    temp_output_file = None
    temp_dir = None

    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="alexandria_audio_")
        logger.debug(f"Created temp directory: {temp_dir}")

        temp_audio_file = os.path.join(temp_dir, "temp_audio.wav")
        temp_output_file = os.path.join(temp_dir, "output.json")

        # Write audio to temporary file
        logger.debug(f"Writing audio to temporary file: {temp_audio_file}")
        sf.write(temp_audio_file, audio_16k, samplerate=16000)

        # Verify file was written
        if os.path.exists(temp_audio_file):
            file_size = os.path.getsize(temp_audio_file)
            logger.debug(f"✓ Audio file written successfully ({file_size} bytes)")
        else:
            raise RuntimeError(f"Failed to write audio file: {temp_audio_file}")

        logger.info("▶ Starting transcription...")
        logger.debug(f"  Expected duration: {audio_duration:.1f}s (this may take several minutes)")

        # Prepare CLI command
        ifw_module_path = os.path.join(script_dir, "insanely-fast-whisper-rocm")
        logger.debug(f"Insanely Fast Whisper module path: {ifw_module_path}")
        logger.debug(f"Module exists: {os.path.exists(ifw_module_path)}")

        # Use current Python executable (which has the correct ROCm environment)
        python_exe = sys.executable
        logger.debug(f"Python executable: {python_exe}")
        logger.debug(f"Working directory: {os.getcwd()}")
        logger.debug(f"sys.executable Python version: {sys.version}")

        # Use pre-downloaded model or HuggingFace model ID
        model_name = "openai/whisper-base"
        local_model_path = os.path.join(script_dir, "models", "whisper-base")
        if os.path.exists(local_model_path):
            model_name = local_model_path
            logger.debug(f"Using local model: {local_model_path}")
        else:
            logger.debug(f"Using HuggingFace model: {model_name}")

        cmd = [
            python_exe,
            "-m", "insanely_fast_whisper_rocm.cli",
            "transcribe",
            temp_audio_file,  # Positional argument for audio file
            "--model", model_name,
            "--language", language,
            "--timestamp-type", "word",
            "--output", temp_output_file
        ]

        logger.info(f"Command: {' '.join(cmd)}")

        # Set PYTHONPATH to include the insanely-fast-whisper-rocm module
        cmd_env = os.environ.copy()
        cmd_env["PYTHONPATH"] = f"{ifw_module_path}:{cmd_env.get('PYTHONPATH', '')}".rstrip(":")
        logger.debug(f"PYTHONPATH: {cmd_env.get('PYTHONPATH', 'not set')}")

        # Set LD_LIBRARY_PATH to use conda's FFmpeg 7 libraries (REPLACE system FFmpeg 8.1)
        conda_prefixes = [
            os.environ.get("CONDA_PREFIX", ""),
            os.environ.get("PINOKIO_CONDA_PREFIX", ""),
            os.path.join(os.environ.get("PINOKIO_HOME", ""), "bin", "miniconda"),
            sys.prefix,  # Current Python's conda prefix
        ]

        for prefix in conda_prefixes:
            if prefix and os.path.exists(prefix):
                conda_lib = os.path.join(prefix, "lib")
                if os.path.exists(conda_lib):
                    # REPLACE LD_LIBRARY_PATH entirely with conda libraries first
                    cmd_env["LD_LIBRARY_PATH"] = conda_lib
                    logger.info(f"  ├─ FFmpeg: Using conda FFmpeg 7 (v{conda_lib})")
                    logger.debug(f"  └─ LD_LIBRARY_PATH: {cmd_env['LD_LIBRARY_PATH']}")
                    break

        # Run subprocess
        logger.info("▶ Running Whisper transcription (this may take several minutes)...")
        logger.debug("  Subprocess environment: FFmpeg 7 + ROCm optimization")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=cmd_env)

        if result.returncode == 0:
            logger.info(f"✓ Transcription subprocess completed successfully")
        else:
            logger.warning(f"⚠ Transcription subprocess returned code: {result.returncode}")

        logger.debug(f"  Output length: {len(result.stdout)} chars, Error length: {len(result.stderr)} chars")

        # Parse GPU info from subprocess output
        if result.stdout:
            cuda_match = re.search(r'ASR using device:\s*(cuda:\d+)', result.stdout)
            if cuda_match:
                device_info = cuda_match.group(1)
                logger.info(f"  ├─ GPU Device: {device_info} (confirmed in subprocess)")

            torch_match = re.search(r'cuda_available=(\w+)', result.stdout)
            if torch_match:
                cuda_status = torch_match.group(1)
                logger.info(f"  ├─ CUDA Available: {cuda_status}")

            hip_match = re.search(r'hip=([0-9.]+)', result.stdout)
            if hip_match:
                hip_version = hip_match.group(1)
                logger.info(f"  └─ HIP (ROCm) Version: {hip_version}")

            logger.debug(f"Subprocess stdout:\n{result.stdout}")

        if result.stderr:
            logger.debug(f"Subprocess stderr:\n{result.stderr}")

        if result.returncode != 0:
            logger.error(f"Transcription command failed with return code {result.returncode}")
            logger.error(f"stderr: {result.stderr}")
            raise RuntimeError(f"Transcription failed: {result.stderr}")

        # The tool ignores --output and saves to transcripts/ with its own naming
        # Parse stdout to find where it actually saved the JSON
        import re
        actual_json_file = None

        if result.stdout:
            # Look for "Saved JSON to: <path>" in the output
            match = re.search(r'Saved JSON to:\s*(.+?)(?:\n|$)', result.stdout)
            if match:
                actual_json_file = match.group(1).strip()
                logger.debug(f"Found JSON path in stdout: {actual_json_file}")

        if not actual_json_file:
            logger.error(f"Could not find JSON output path in subprocess stdout")
            logger.debug(f"Temp directory contents: {os.listdir(temp_dir) if os.path.exists(temp_dir) else 'temp_dir not found'}")
            raise FileNotFoundError(f"Could not determine where insanely-fast-whisper saved JSON output")

        # The path in stdout is relative to where the subprocess was run
        # We need to check current working directory
        if not os.path.isabs(actual_json_file):
            actual_json_file = os.path.join(os.getcwd(), actual_json_file)

        if not os.path.exists(actual_json_file):
            logger.error(f"JSON file not found at expected path: {actual_json_file}")
            raise FileNotFoundError(f"JSON output file not found: {actual_json_file}")

        file_size = os.path.getsize(actual_json_file)
        logger.info(f"✓ JSON found: {actual_json_file} ({file_size} bytes)")

        # Load and parse JSON
        logger.debug(f"Loading JSON from: {actual_json_file}")
        with open(actual_json_file, 'r') as f:
            result_json = json.load(f)

        logger.debug(f"JSON keys: {list(result_json.keys())}")

        detected_lang = result_json.get("language", language)
        logger.info(f"✓ Detected language: {detected_lang}")

        # Extract word segments
        logger.info("▶ Extracting word segments...")
        word_segments = []
        chunks = result_json.get("chunks")
        chunk_count = len(chunks) if chunks else 0
        logger.debug(f"  Processing {chunk_count} chunks...")

        if chunks:
            for idx, chunk in enumerate(chunks):
                timestamp = chunk.get("timestamp")
                text = chunk.get("text", "").strip()

                if timestamp and len(timestamp) >= 2:
                    start, end = timestamp[0], timestamp[1]
                    if start is not None and end is not None:
                        word_segments.append({
                            "word": text,
                            "start": start,
                            "end": end
                        })
                        if idx < 5:  # Log first few entries
                            logger.debug(f"  Chunk {idx}: '{text}' [{start:.3f}-{end:.3f}]")
                else:
                    logger.debug(f"  Chunk {idx}: Missing timestamp - {chunk}")

        clear_vram()
        logger.info(f"✓ Insanely Fast Whisper complete: {len(word_segments)} words extracted")
        logger.debug(f"First 5 words: {word_segments[:5]}")
        logger.debug(f"Last 5 words: {word_segments[-5:]}")

        return word_segments, detected_lang

    except Exception as e:
        logger.error(f"Insanely Fast Whisper transcription failed: {e}")
        logger.debug(traceback.format_exc())
        raise
    finally:
        # Clean up temporary files
        logger.debug("Cleaning up temporary files...")

        if temp_audio_file and os.path.exists(temp_audio_file):
            try:
                os.remove(temp_audio_file)
                logger.debug(f"✓ Removed audio file: {temp_audio_file}")
            except Exception as e:
                logger.warning(f"Failed to clean up audio file: {e}")

        if temp_output_file and os.path.exists(temp_output_file):
            try:
                os.remove(temp_output_file)
                logger.debug(f"✓ Removed output file: {temp_output_file}")
            except Exception as e:
                logger.warning(f"Failed to clean up output file: {e}")

        # Clean up the JSON file created by the tool in transcripts/ directory
        if 'actual_json_file' in locals() and actual_json_file:
            try:
                if os.path.exists(actual_json_file):
                    os.remove(actual_json_file)
                    logger.debug(f"✓ Removed JSON output: {actual_json_file}")
                # Try to remove transcripts directory if empty
                transcripts_dir = os.path.dirname(actual_json_file)
                if os.path.exists(transcripts_dir) and not os.listdir(transcripts_dir):
                    os.rmdir(transcripts_dir)
                    logger.debug(f"✓ Removed empty transcripts directory")
            except Exception as e:
                logger.debug(f"Note: Could not clean up JSON file: {e}")

        # Clean up temp directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                remaining = os.listdir(temp_dir)
                if remaining:
                    logger.debug(f"Remaining files in {temp_dir}: {remaining}")
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.debug(f"✓ Removed temp directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory: {e}")

def diarize_audio(audio_path: str, hf_token: str = None, device: str = "cuda") -> list:
    """Perform speaker diarization using pyannote.audio.
    Requires a Hugging Face token with access to pyannote/speaker-diarization-3.1.
    """
    try:
        from pyannote.audio import Pipeline
        import torch as torch_module
    except ImportError:
        logger.error("pyannote.audio not installed. Diarization skipped.")
        return []

    if not hf_token:
        logger.warning("No Hugging Face token provided. Diarization requires a token for model access.")
        logger.warning("Pass --hf-token or set HF_TOKEN environment variable.")
        return []

    logger.info(f"▶ Initializing pyannote.audio diarization (device={device})...")
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token
        )
        if pipeline is None:
            raise ValueError("Failed to load pyannote pipeline. Check your token and model permissions.")
            
        pipeline.to(torch_module.device(device))
        
        logger.info(f"Running diarization on {audio_path}...")
        t0 = time.monotonic()
        diarization = pipeline(audio_path)
        elapsed = time.monotonic() - t0
        
        speaker_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
            
        logger.info(f"✓ Diarization complete in {format_duration(elapsed)}")
        unique_speakers = sorted(list(set(s["speaker"] for s in speaker_segments)))
        logger.info(f"  └─ Detected {len(unique_speakers)} unique speaker(s): {', '.join(unique_speakers)}")
        
        return speaker_segments

    except Exception as e:
        logger.error(f"✗ Diarization failed: {e}")
        logger.debug(traceback.format_exc())
        return []


def choose_and_transcribe(audio_16k: np.ndarray, device: str, language: str, limit: int = None) -> tuple:
    """Transcribe using Wav2Vec2 (continuous context-aware) as primary with fallbacks."""

    logger.info("=" * 70)
    logger.info("ASR Method Selection")
    logger.info("=" * 70)
    logger.info(f"Device: {device}")
    logger.info(f"Language: {language}")
    logger.info(f"Available ASR methods:")
    logger.info(f"  - Wav2Vec2 (GPU continuous): {TRANSFORMERS_WHISPER_AVAILABLE}")
    logger.info(f"  - InFastWhisper (ROCm): {INSANELY_FAST_WHISPER_AVAILABLE}")
    logger.info(f"  - WhisperX (CPU mode): {WHISPERX_AVAILABLE}")

    # Try Wav2Vec2 first (continuous context-aware)
    if TRANSFORMERS_WHISPER_AVAILABLE:
        logger.info("-" * 70)
        logger.info("▶ Method 1: Wav2Vec2 (Continuous context-aware) [GPU accelerated, 30s chunks with overlap]")
        logger.info("-" * 70)
        try:
            word_segments, detected_lang = transcribe_with_wav2vec2(audio_16k, language, limit=limit)
            logger.info(f"✓ SUCCESS with Wav2Vec2")
            logger.info(f"  ├─ Words extracted: {len(word_segments)}")
            logger.info(f"  ├─ Context preservation: Full audio (30s overlapping chunks)")
            logger.info(f"  └─ Detected language: {detected_lang}")
            return word_segments, detected_lang
        except Exception as e:
            logger.warning(f"✗ Wav2Vec2 failed: {e}")
            logger.debug(traceback.format_exc())
            logger.info("Falling back to Insanely Fast Whisper...")

    # Fallback to Insanely Fast Whisper (ROCm optimized)
    if INSANELY_FAST_WHISPER_AVAILABLE:
        logger.info("-" * 70)
        logger.info("▶ Method 2: Insanely Fast Whisper (ROCm optimized) [GPU accelerated, 30s chunks]")
        logger.info("-" * 70)
        try:
            word_segments, detected_lang = transcribe_with_insanely_fast_whisper(audio_16k, language)
            logger.info(f"✓ SUCCESS with Insanely Fast Whisper")
            logger.info(f"  ├─ Words extracted: {len(word_segments)}")
            logger.info(f"  └─ Detected language: {detected_lang}")
            return word_segments, detected_lang
        except Exception as e:
            logger.warning(f"✗ Insanely Fast Whisper failed: {e}")
            logger.debug(traceback.format_exc())
            logger.info("Falling back to WhisperX-CPU...")

    # Final fallback to WhisperX-CPU
    if WHISPERX_AVAILABLE:
        logger.info("-" * 70)
        logger.info("▶ Method 3: WhisperX-CPU (Stable fallback) [CPU mode, word-level alignment]")
        logger.info("-" * 70)
        try:
            word_segments, detected_lang = transcribe_with_whisperx_cpu(audio_16k, language)
            logger.info(f"✓ SUCCESS with WhisperX-CPU")
            logger.info(f"  ├─ Words extracted: {len(word_segments)}")
            logger.info(f"  └─ Detected language: {detected_lang}")
            return word_segments, detected_lang
        except Exception as e:
            logger.error(f"✗ WhisperX-CPU also failed: {e}")
            logger.debug(traceback.format_exc())

    logger.critical("=" * 70)
    logger.critical("✗ CRITICAL: No ASR method available!")
    logger.critical("Install with: pip install insanely-fast-whisper-rocm whisperx")
    logger.critical("=" * 70)
    sys.exit(1)

# ── Annotation output sanitisation ────────────────────────────────────────────
# These run on the LLM's annotation BEFORE we write it to metadata.jsonl, so
# downstream consumers (alexandria_compare.py and any TTS trainer that
# tokenises on whitespace) don't have to peel the same artefacts apart.
_EMPHASIS_PATTERN = re.compile(r'\*([^*]+)\*')
_DOTS_PATTERN     = re.compile(r'\.{3,}')
_WS_COLLAPSE      = re.compile(r'\s+')


def _sanitize_annotation(text: str) -> str:
    """Clean up common LLM annotation quirks before writing to JSONL.

    1) Multi-word emphasis '*Trull Sengar*' → '*Trull* *Sengar*' so per-word
       prosody markers survive whitespace-tokenisation downstream.
    2) Pad '...' / '....' pause runs so they sit as their own tokens rather
       than fusing into adjacent words ('YOU...*DERONDL*...THE' otherwise
       collapses to one garbled token in any naive parser).
    3) Collapse any doubled whitespace introduced by step 2.
    """
    text = _EMPHASIS_PATTERN.sub(
        lambda m: ' '.join(f'*{w}*' for w in m.group(1).split()) or m.group(0),
        text,
    )
    text = _DOTS_PATTERN.sub(r' \g<0> ', text)
    text = _WS_COLLAPSE.sub(' ', text).strip()
    return text


# ── Chunk-boundary selection ──────────────────────────────────────────────────
def _find_best_cut(word_starts, word_ends, words, chunk_start,
                   min_pause: float = 0.25,
                   lookback_frac: float = 0.30) -> tuple:
    """Pick the cut point for an over-size chunk. Returns
    `(last_word_idx_inclusive, strategy)` where strategy is one of
    'sentence_end', 'pause', 'too_few_words', or 'fallback' — the strategy
    is used by the caller for histogram logging so we can see which
    branch dominates and tune the parameters.

    The chunker previously cut at the first word that crossed the target
    duration — wherever that landed. This helper prefers natural boundaries
    within the last `lookback_frac` of the chunk's accumulated time so the
    emitted WAV ends on a breath/clause/sentence break instead of mid-phrase.

    Preference order:
      1. Word ending in sentence punctuation (.!?) within the lookback
         window — best for TTS coherence, take the LATEST one.
      2. Word followed by the longest pause ≥ `min_pause` seconds in the
         same window — natural breath/clause break.
      3. The final word (the pre-fix behaviour) when neither is available.
    """
    n = len(words)
    if n < 4:
        return n - 1, 'too_few_words'
    chunk_end   = word_ends[-1]
    chunk_dur   = chunk_end - chunk_start
    threshold_t = chunk_end - chunk_dur * lookback_frac

    # 1) Sentence-end cut — walk backward from end, take the latest in window
    for i in range(n - 1, -1, -1):
        if word_ends[i] < threshold_t:
            break
        bare = words[i].rstrip(') ”"\'')
        if bare.endswith(('.', '!', '?')):
            return i, 'sentence_end'

    # 2) Pause cut — largest gap in the window
    best_idx = None
    best_gap = min_pause
    for i in range(n - 1):
        if word_ends[i] < threshold_t:
            continue
        gap = word_starts[i + 1] - word_ends[i]
        if gap >= best_gap:
            best_gap = gap
            best_idx = i
    if best_idx is not None:
        return best_idx, 'pause'

    # 3) Fall back to current behaviour
    return n - 1, 'fallback'


def _provisional_entries_for_anchor(word_segments, chunk_size, max_entries=30):
    """Pack the first N chunks' worth of ASR words into the entry shape
    alignment.auto_anchor / alignment.estimate_alignment_quality expect.

    Auto-anchor needs to see actual chunk text to figure out where the audio
    first lines up with the source. But the real chunker hasn't run yet (it
    depends on the source-cursor we're trying to derive). Build provisional
    chunks via the same duration-threshold rule the real chunker uses — no
    pause-aware look-back, no LLM, just enough to feed the anchor.
    """
    entries = []
    current_words = []
    current_start = None
    for word_data in word_segments:
        if "start" not in word_data or "end" not in word_data:
            continue
        word = word_data.get("word", "").strip()
        if not word:
            continue
        if current_start is None:
            current_start = word_data["start"]
        current_words.append(word)
        current_end = word_data["end"]
        if current_end - current_start >= chunk_size:
            entries.append({
                'text':  " ".join(current_words),
                'start': current_start,
                'end':   current_end,
            })
            current_words = []
            current_start = None
            if len(entries) >= max_entries:
                break
    return entries


def _log_word_segment_stats(word_segments, label="ASR word segments"):
    """Summarise the ASR word_segments at INFO so the user has a feel for
    the audio's word density / pause structure before chunking starts.

    Pause percentiles especially matter: if the corpus has lots of >0.5s
    gaps, the pause-aware chunker will produce clean sentence-ending
    chunks; if it's all sub-0.2s gaps (rapid-fire dialogue), most cuts
    will land in the sentence-end or fallback branch.
    """
    if not word_segments:
        logger.info(f"  {label}: empty")
        return

    # Filter to entries that have valid start/end + non-empty word text
    valid = [
        w for w in word_segments
        if 'start' in w and 'end' in w and w.get('word', '').strip()
    ]
    if not valid:
        logger.info(f"  {label}: {len(word_segments)} entries, 0 valid")
        return

    first_t = valid[0]['start']
    last_t  = valid[-1]['end']
    total_t = last_t - first_t

    durations = [w['end'] - w['start'] for w in valid]
    gaps = [
        valid[i + 1]['start'] - valid[i]['end']
        for i in range(len(valid) - 1)
    ]
    gaps = [g for g in gaps if g >= 0]   # filter rare ASR overlaps

    logger.info(f"  {label}: {len(valid):,} words over {total_t:.1f}s "
                f"(rate {len(valid)/max(total_t,1e-9):.1f} words/s)")
    logger.info(f"    ├─ word duration  : median {_percentile(durations, 50):.3f}s, "
                f"p95 {_percentile(durations, 95):.3f}s, max {max(durations):.3f}s")
    if gaps:
        logger.info(f"    ├─ inter-word gap : median {_percentile(gaps, 50):.3f}s, "
                    f"p95 {_percentile(gaps, 95):.3f}s, max {max(gaps):.3f}s")
        long_gaps = sum(1 for g in gaps if g >= 0.5)
        logger.info(f"    └─ pauses ≥ 0.5s  : {long_gaps:,} "
                    f"({100*long_gaps/len(gaps):.1f}% of gaps)")
    else:
        logger.info(f"    └─ inter-word gap : n/a (single word)")


def _percentile(xs, p):
    """Tiny percentile helper used by the end-of-chunker summary."""
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round(p / 100.0 * (len(xs) - 1)))))
    return xs[k]


def _build_source_state(source_path: str,
                        source_start: int = None,
                        source_start_text: str = None,
                        no_auto_anchor: bool = False,
                        entries_for_anchor: list = None):
    """Load + clean the source, build the proper-noun lexicon, tokenise into
    parallel display/match word lists, and pick the initial cursor.

    Returns a dict with everything the chunker needs to align ASR chunks
    against the source:
      {
        'orig_display': [...],   # source words, original capitalisation
        'orig_match'  : [...],   # source words, normalised for fuzzy matching
        'cursor'      : N,       # current source-word index
      }

    `entries_for_anchor` should be the first few ASR chunks (already
    available at this point) so we can use auto_anchor to find where the
    audio's prose lines up with the source text. Audio intros (credits,
    narrator notes) often have no source equivalent — the anchor jumps past
    them to the prologue's first real sentence.
    """
    logger.info(f"▶ Loading source for guided chunking: {source_path}")
    source_text = alignment.load_source(source_path)
    source_text = alignment.clean_source_text(source_text)
    logger.info(f"  ├─ Source: {len(source_text):,} characters")

    # Build per-book proper-noun lexicon (character names + recurring
    # capitalised terms). Used by alignment._step_threshold to relax the
    # boundary acceptance bar for ASR-mangled Japanese romanisations like
    # 'coodo'↔'kudou' that sit far below the default 0.55 fuzzy bar.
    alignment._PROPER_NOUNS = alignment._build_proper_nouns(source_text)
    if alignment._PROPER_NOUNS:
        sample = ', '.join(sorted(alignment._PROPER_NOUNS)[:8])
        more = f' +{len(alignment._PROPER_NOUNS) - 8} more' if len(alignment._PROPER_NOUNS) > 8 else ''
        logger.info(f"  ├─ {len(alignment._PROPER_NOUNS)} recurring proper nouns ({sample}{more})")

    # Hyphenated compounds split into separate tokens so "twenty-minute" doesn't
    # become a single un-alignable word. Same logic as compare's main(); U+2500
    # appears in some EPUB→text conversions where em-dashes should be.
    compound_split = re.compile(r'[-‐‑‒–—―─━]')
    tokens = compound_split.sub(' ', source_text).split()
    orig_display, orig_match = [], []
    for w in tokens:
        m = alignment.normalize(w)
        if not m:
            continue
        orig_display.append(w)
        orig_match.append(m)
    logger.info(f"  ├─ {len(orig_display):,} source words")

    # Pick initial cursor
    anchor_entry_idx = 0
    if source_start is not None:
        cursor = max(0, min(source_start, len(orig_match)))
        logger.info(f"  └─ Starting at source word {cursor} (--source-start)")
    elif source_start_text:
        pos = alignment.find_text_in_source(source_start_text, orig_match)
        if pos < 0:
            sys.exit(
                f"--source-start-text: could not confidently locate "
                f"{source_start_text!r} in the source. Try a longer or more "
                f"distinctive phrase, or use --source-start N."
            )
        cursor = pos
        logger.info(f"  └─ Starting at source word {cursor} (matched --source-start-text)")
    elif no_auto_anchor:
        cursor = 0
        logger.info(f"  └─ Auto-anchor disabled; starting at source word 0")
    elif entries_for_anchor:
        # Use auto_anchor with the first ~20 chunks to find where the audio's
        # prose lines up with the source. Anchor entries are built from the
        # ASR word_segments accumulated so far (before chunking) — we pack
        # them into the same shape compare's auto_anchor expects.
        anchor_idx, anchor_pos, anchor_ratio = alignment.auto_anchor(
            entries_for_anchor, orig_match
        )
        if anchor_ratio > 0:
            logger.info(f"  └─ Auto-anchor: entry {anchor_idx} → source word {anchor_pos} "
                        f"({anchor_ratio:.1%} match)")
            cursor = anchor_pos
            anchor_entry_idx = anchor_idx
        else:
            logger.warning(f"  └─ Auto-anchor found no confident match in the first "
                           f"{min(20, len(entries_for_anchor))} chunks; starting at word 0")
            cursor = 0
    else:
        cursor = 0
        logger.info(f"  └─ No anchor data; starting at source word 0")

    return {
        'orig_display': orig_display,
        'orig_match':   orig_match,
        'cursor':       cursor,
        'anchor_entry_idx': anchor_entry_idx,
    }


def _read_audio_segment(audio_24k_source, start_s, end_s):
    """Read an audio segment by time range from either an in-memory array or a soundfile path."""
    start_samp = max(0, round(start_s * 24000))
    end_samp = round(end_s * 24000)

    if isinstance(audio_24k_source, (str, os.PathLike)):
        with sf.SoundFile(str(audio_24k_source)) as f:
            end_samp = min(end_samp, f.frames)
            if end_samp <= start_samp:
                return np.zeros(0, dtype=np.float32)
            f.seek(start_samp)
            data = f.read(end_samp - start_samp, dtype="float32", always_2d=False)
        return data
    else:
        end_samp = min(end_samp, len(audio_24k_source))
        slice_view = audio_24k_source[start_samp:end_samp]
        if slice_view.dtype != np.float32:
            return slice_view.astype(np.float32)
        return slice_view


def _load_existing_checkpoint(temp_dir):
    """Read existing metadata.jsonl checkpoint and return (entries, resume_time, next_segment_idx).

    Tolerates a truncated/corrupt trailing line (common after power loss while
    line-buffered append was in flight): keeps the good prefix and stops at
    the first bad line. Anything after a bad line is suspect — the file may be
    fsync-ordered with later writes that landed after a gap — so we discard
    the tail rather than trying to recover it.

    Critically: if a bad line is found, the checkpoint file is immediately
    rewritten with only the good prefix. Without this, every subsequent resume
    would hit the same bad line, truncate at the same point, and sweep the
    newly-appended WAVs — making recovery impossible after repeated crashes.
    """
    checkpoint_path = os.path.join(temp_dir, "metadata.jsonl")
    entries = []
    if not os.path.exists(checkpoint_path):
        return entries, 0.0, 0

    good_lines = []
    truncated = False
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for line_no, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                    good_lines.append(raw if raw.endswith("\n") else raw + "\n")
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Checkpoint line {line_no} unparseable ({e}); "
                        f"keeping {len(entries)} good entries and stopping."
                    )
                    truncated = True
                    break
    except Exception as e:
        logger.warning(f"Could not read checkpoint {checkpoint_path}: {e}")
        return [], 0.0, 0

    if truncated:
        try:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                f.writelines(good_lines)
            logger.info(f"  Checkpoint rewritten to {len(entries)} good entries (corrupt tail removed)")
        except Exception as e:
            logger.warning(f"Could not rewrite checkpoint after truncation ({e}); "
                           f"future resumes may re-truncate at the same line")

    if not entries:
        return [], 0.0, 0

    try:
        resume_time = max(e.get("end", 0.0) for e in entries)
        next_idx = max(
            int(os.path.splitext(e["audio_filepath"])[0].split("_")[-1])
            for e in entries
        ) + 1
    except Exception as e:
        logger.warning(f"Could not compute resume state from checkpoint ({e}); forcing fresh start")
        return [], 0.0, 0

    return entries, resume_time, next_idx


def _sweep_orphan_wavs(temp_dir, next_segment_idx):
    """Delete sample_NNNN.wav files at or above next_segment_idx.

    These are WAVs written before the matching metadata entry was committed
    (or before its kernel buffer flushed). Without this sweep, a multi-resume
    sequence can leave high-index orphan WAVs in dataset_temp/ that get
    packaged into the final ZIP with no metadata pointing at them.
    """
    if not os.path.isdir(temp_dir):
        return 0
    removed = 0
    for name in os.listdir(temp_dir):
        if not (name.startswith("sample_") and name.endswith(".wav")):
            continue
        try:
            idx = int(name[len("sample_"):-len(".wav")])
        except ValueError:
            continue
        if idx >= next_segment_idx:
            try:
                os.remove(os.path.join(temp_dir, name))
                removed += 1
            except Exception as e:
                logger.warning(f"Failed to remove orphan {name}: {e}")
    if removed:
        msg = f"  ├─ Swept {removed} orphan WAV(s) at idx ≥ {next_segment_idx}"
        if removed > 1:
            logger.warning(msg + " — unexpectedly large; verify next_segment_idx is correct")
        else:
            logger.info(msg)
    return removed


def _wipe_temp_dir(temp_dir):
    """Remove all preparer-generated files from temp_dir but keep the directory itself."""
    if not os.path.exists(temp_dir):
        return
    # Protect intermediate files used across phases
    protected = {
        "asr_segments.json",       # ASR phase output
        "audio_24k_scratch.wav",   # Audio scratch file
        "enriched_segments.json",  # LLM enrichment output
        "asr_chunks_for_enrich.json",  # Enrichment input chunks
        "diarization.json",        # Speaker diarization output
    }
    for name in os.listdir(temp_dir):
        if name in protected:
            continue
        full_path = os.path.join(temp_dir, name)
        try:
            if os.path.isfile(full_path) or os.path.islink(full_path):
                os.remove(full_path)
            elif os.path.isdir(full_path):
                shutil.rmtree(full_path)
        except Exception as e:
            logger.warning(f"Failed to remove {full_path}: {e}")


def _check_source_marker(temp_dir, audio_source_path):
    """Return True if temp_dir's .source marker matches audio_source_path."""
    marker_path = os.path.join(temp_dir, ".source")
    if not os.path.exists(marker_path):
        return False
    try:
        with open(marker_path, "r", encoding="utf-8") as f:
            stored = f.read().strip()
        return stored == os.path.abspath(audio_source_path)
    except Exception:
        return False


def _write_source_marker(temp_dir, audio_source_path):
    """Write the .source marker so future runs can verify this temp_dir's owner."""
    marker_path = os.path.join(temp_dir, ".source")
    with open(marker_path, "w", encoding="utf-8") as f:
        f.write(os.path.abspath(audio_source_path))


def _load_llm(model_path):
    """Load a GGUF LLM via llama-cpp-python, all layers on GPU.

    `verbose=True` surfaces llama-cpp's own offload count
    (e.g. `offloaded 65/65 layers to GPU`) into stderr so we can verify
    whether the model actually landed on GPU or silently fell back.

    NOTE: do NOT call torch.cuda.empty_cache() / synchronize() before this.
    On ROCm those calls leave the device in a state where llama-cpp's
    ggml_cuda_init() reports "no ROCm-capable device is detected" and
    falls back to pure CPU — observed regressing per-chunk from ~11s
    (partial GPU) to ~13s (no GPU). Just delete the GC-eligible Python
    refs to Wav2Vec2 (already done in the caller) and let HIP keep the
    device context warm.
    """
    logger.debug(f"Loading GGUF model from: {model_path}")
    gc.collect()  # let dead Wav2Vec2 tensor refs drop without touching torch.cuda
    try:
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=99,   # explicit count > total; -1 was misinterpreted on some HIP builds
            n_ctx=8192,
            verbose=True,      # let llama-cpp's own "offloaded N/M layers" line into the log
        )
    except BrokenPipeError:
        # llama_cpp prints system info to stderr during init; if stderr is
        # piped (e.g. | tee | head), the pipe may close before the print
        # completes. Retry with verbose=False — the model still loads fine.
        logger.debug("BrokenPipeError during Llama init (stderr pipe closed); retrying with verbose=False")
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=99,
            n_ctx=8192,
            verbose=False,
        )
    logger.info(f"✓ LLM loaded: {os.path.basename(model_path)}")
    if hasattr(llm, 'n_gpu_layers'):
        logger.info(f"  ├─ GPU Layers Loaded: {llm.n_gpu_layers}")
    logger.info(f"  └─ Model device: {llm.metadata.get('device', 'cuda (via n_gpu_layers=-1)')}")

    # Verify GPU usage with a tiny test inference
    logger.debug("Verifying GPU inference capability with test prompt...")
    llm.create_chat_completion(
        messages=[{"role": "user", "content": "test"}],
        max_tokens=1
    )
    logger.info(f"✓ GPU inference verified - model responding on GPU")
    return llm


# System prompt tuned for terse, structured TTS annotation output.
# Works well with instruction-following models (Qwen, Llama Instruct, Gemma Instruct).
TTS_ANNOTATION_SYSTEM_PROMPT = (
    "You are a TTS annotation tool. Given a text segment from an audiobook, "
    "output ONLY the annotated text with these markers and nothing else:\n"
    "- Pauses: use ... for natural pauses, .... for longer pauses\n"
    "- Emphasis: wrap stressed words in *asterisks*\n"
    "- Tone: punctuation conveys prosody (?, !, ,, .)\n"
    "Output the annotated text directly with no preamble, no explanation, "
    "no alternatives, no quotation marks around the output."
)


def _calculate_chunk_snr(chunk_audio: np.ndarray) -> float:
    """Calculate the Signal-to-Noise Ratio (SNR) for an audio chunk.

    Uses a pure-numpy approach: signal power = mean of squared samples,
    noise power = mean energy of quietest 10% of 2048-sample blocks.
    Avoids librosa.util.frame() which creates expensive 2D arrays.
    """
    if chunk_audio.size == 0:
        return -100.0  # Represents silent or empty chunk

    signal_power = np.mean(chunk_audio ** 2)
    if signal_power == 0:
        return -100.0

    # Block-based noise estimation (2048-sample blocks)
    block_size = 2048
    n_blocks = len(chunk_audio) // block_size
    if n_blocks == 0:
        return 100.0  # Too short to estimate noise

    # Compute per-block energy efficiently using reshape (no loop, no 2D copy)
    n_samples = n_blocks * block_size
    blocks = chunk_audio[:n_samples].reshape(n_blocks, block_size)
    block_energies = np.mean(blocks ** 2, axis=1)

    # Noise = mean energy of quietest 10% of blocks
    n_quiet = max(1, int(len(block_energies) * 0.1))
    noise_power = np.mean(np.partition(block_energies, n_quiet)[:n_quiet])

    if noise_power <= 0:
        return 100.0  # No detectable noise

    return 10 * np.log10(signal_power / noise_power)

def annotate_chunks(word_segments, model_path, chunk_size, audio_24k_source,
                    resume=False, audio_source_path=None, fallback_model_path=None,
                    source_state=None, source_threshold=0.65, keep_unaligned=False,
                    min_chunk_duration=2.0, min_confidence=0.85, min_snr=25,
                    book_title=None, character=None, narrator_style=None):
    """Create and annotate chunks with periodic checkpointing and resume support.

    audio_24k_source: either a numpy array (in-memory) or a path to a 24kHz WAV file.
    audio_source_path: the original input audio path, used to validate resume safety.
    fallback_model_path: optional secondary GGUF to load if model_path fails.

    source_state: when provided (from --source), each chunk is fuzzy-aligned
    against the source text BEFORE the LLM annotates. High-confidence matches
    (>= source_threshold) have their text replaced with the source's spelling
    so character names and dialect spellings come out correct. Below-threshold
    chunks are dropped (audio-only material) unless keep_unaligned=True.
    Pass None to run the pre-source legacy ASR-only flow with no behaviour
    change.
    """
    temp_dir = "dataset_temp"
    os.makedirs(temp_dir, exist_ok=True)
    checkpoint_path = os.path.join(temp_dir, "metadata.jsonl")

    # Determine if existing dataset_temp/ belongs to this audio file
    marker_matches = (
        audio_source_path is not None
        and _check_source_marker(temp_dir, audio_source_path)
    )

    if resume and marker_matches:
        existing_entries, resume_time, next_segment_idx = _load_existing_checkpoint(temp_dir)
        if existing_entries:
            logger.info(f"▶ Resuming from checkpoint: {len(existing_entries)} segments already processed")
            logger.info(f"  ├─ Source verified: {audio_source_path}")
            logger.info(f"  ├─ Resume time: {resume_time:.2f}s")
            logger.info(f"  ├─ Next segment index: {next_segment_idx}")
            _sweep_orphan_wavs(temp_dir, next_segment_idx)
            logger.info(f"  └─ Resume state clean")
        else:
            logger.info("▶ --resume specified but no checkpoint found, starting fresh")
            _sweep_orphan_wavs(temp_dir, 0)  # wipe any stale WAVs from a prior run
            existing_entries, resume_time, next_segment_idx = [], 0.0, 0
    else:
        if resume and not marker_matches:
            logger.warning(
                "▶ --resume specified, but dataset_temp/ belongs to a different source file "
                "(or has no marker). Wiping and starting fresh to avoid corrupting another run."
            )
        elif os.listdir(temp_dir):
            logger.info("▶ Wiping stale dataset_temp/ contents for fresh start")
        _wipe_temp_dir(temp_dir)
        existing_entries, resume_time, next_segment_idx = [], 0.0, 0

    # Always (re)write the source marker for the current run
    if audio_source_path is not None:
        _write_source_marker(temp_dir, audio_source_path)

    logger.info("▶ Loading LLM for annotations...")
    logger.info(f"  ├─ Primary model: {os.path.basename(model_path)}")
    if fallback_model_path:
        logger.info(f"  ├─ Fallback model: {os.path.basename(fallback_model_path)}")
    logger.info("  ├─ Device: GPU (CUDA/ROCm acceleration)")
    logger.info("  ├─ GPU Layers: All (-1 = fully loaded to GPU)")
    logger.info("  └─ Checkpoint: fsync per chunk (durable across power loss)")

    active_model_path = model_path
    try:
        llm = _load_llm(model_path)
    except Exception as primary_err:
        logger.error(f"✗ Failed to load primary model {model_path}: {primary_err}")
        logger.debug(traceback.format_exc())
        if fallback_model_path and os.path.exists(fallback_model_path):
            logger.warning(f"▶ Falling back to: {fallback_model_path}")
            try:
                llm = _load_llm(fallback_model_path)
                active_model_path = fallback_model_path
            except Exception as fallback_err:
                logger.error(f"✗ Fallback model also failed: {fallback_err}")
                logger.debug(traceback.format_exc())
                raise
        else:
            raise

    log_gpu_stats(f"after LLM load ({os.path.basename(active_model_path)})")

    # ── Pre-chunk diagnostic: word density, gap distribution ─────────────────
    # Lets the user see what kind of audio they're working with before the
    # 60+ hour annotation starts. If gaps are uniformly tiny, expect the
    # pause-cut branch to be useless and most chunks to fall back. If gaps
    # span a wide range, expect sentence-end and pause cuts to dominate.
    _log_word_segment_stats(word_segments, label="ASR word segments")
    logger.info(f"  ├─ Source mode    : {'enabled' if source_state else 'disabled (ASR-only)'}")
    if source_state:
        logger.info(f"  ├─ Threshold      : {source_threshold:.2f} "
                    f"({'keep-unaligned' if keep_unaligned else 'strict-drop'})")
        logger.info(f"  └─ Initial cursor : source word {source_state['cursor']}")

    # ── Per-run summary metrics (logged at the end of annotate_chunks) ───────
    from collections import Counter
    stats = {
        'cut_strategy':    Counter(),   # which look-back path picked the cut
        'source_action':   Counter(),   # 'replace' / 'keep_asr' / 'dropped' / 'dropped_short' / 'deduplicated' / 'dropped_low_quality'
        'llm_success':     0,
        'llm_fail':        0,
        'sanitize_changed':0,           # times _sanitize_annotation altered text
        'chunk_durations': [],          # for end-of-run distribution stats
        'audio_short':     0,           # times audio slice was shorter than expected
        'reanchor_backward': 0,         # large backward re-anchor jumps (source/audio mismatch signal)
    }

    metadata = list(existing_entries)
    segment_idx = next_segment_idx
    current_words = []
    current_word_starts = []   # parallel to current_words — for pause-aware cuts
    current_word_ends   = []   # parallel to current_words — for pause-aware cuts
    current_start = resume_time  # Start fresh after the resume point
    context = deque(maxlen=5)

    # Pre-populate context from last 5 resumed entries for continuity
    for prior in metadata[-5:]:
        context.append(prior.get("text", ""))

    prev_raw_text = context[-1] if context else ""

    total_words = len(word_segments)
    logger.info(f"▶ Creating and annotating chunks (target: {chunk_size}s per chunk)...")
    logger.info(f"  Processing {total_words} word segments...")

    # Estimate based on remaining audio
    estimated_chunks_total = max(1, int((len(word_segments) / 12) * (chunk_size / 10)))
    estimated_chunks_remaining = max(1, estimated_chunks_total - segment_idx)
    logger.info(f"  ├─ Estimated total chunks: ~{estimated_chunks_total}")
    if segment_idx > 0:
        logger.info(f"  ├─ Already completed: {segment_idx}")
        logger.info(f"  └─ Remaining to process: ~{estimated_chunks_remaining}")
    else:
        logger.info(f"  └─ Initial ETA will appear after first chunk completes")
    log_gpu_stats("before annotation loop")

    annotation_start_time = time.monotonic()
    chunk_times = deque(maxlen=20)  # rolling window for dynamic ETA

    # Open checkpoint in line-buffered mode (buffering=1) so each entry hits
    # the kernel buffer immediately, then fsync per chunk so power loss can't
    # lose work the code thinks was persisted. fsync is ~10-50ms on SSD vs
    # ~12s per chunk, so the overhead is <0.5%.
    checkpoint_file = open(checkpoint_path, "a", encoding="utf-8", buffering=1)

    resume_point = current_start  # marker for "skip words before this"
    started = False  # True once we've consumed the first qualifying word

    try:
        for idx, word_data in enumerate(word_segments):
            if "start" not in word_data or "end" not in word_data:
                continue

            word_start_time = word_data["start"]
            # Skip words before the resume point
            if word_start_time < resume_point:
                continue

            word = word_data.get("word", "").strip()
            if not word:
                continue

            # Pin chunk start to the first qualifying word so resumed runs
            # don't include leading silence between resume_point and the first word.
            if not started:
                current_start = word_start_time
                started = True

            current_words.append(word_data)
            current_word_starts.append(word_start_time)
            current_word_ends.append(word_data["end"])
            current_end = word_data["end"]
            duration = current_end - current_start

            is_final = (idx == len(word_segments) - 1)
            
            # ── Smart Clip Length ───────────────────────────────────────────────
            # Dynamically split on natural speech boundaries instead of rigid caps.
            # Start evaluating boundaries at 70% of target chunk size. If we don't
            # find a natural break by 150%, force a cut using the lookback heuristic.
            target_dur = chunk_size * 0.70
            max_dur = chunk_size * 1.50
            
            is_good_break = False
            cut_strategy = None
            
            if duration >= target_dur and not is_final:
                bare_word = word.rstrip(') ”"\'')
                if bare_word.endswith(('.', '!', '?')):
                    is_good_break = True
                    cut_strategy = 'sentence_end'
                elif idx < len(word_segments) - 1:
                    next_start = word_segments[idx + 1].get("start", current_end)
                    if next_start - current_end >= 0.4:
                        is_good_break = True
                        cut_strategy = 'pause'

            if is_final or is_good_break or duration >= max_dur:
                if is_final:
                    cut_at = len(current_words) - 1
                    cut_strategy = 'is_final'
                elif is_good_break:
                    cut_at = len(current_words) - 1
                else:
                    cut_at, cut_strategy = _find_best_cut(
                        current_word_starts, current_word_ends,
                        [w['word'] for w in current_words], current_start,
                    )
                stats['cut_strategy'][cut_strategy] += 1

                chunk_word_data = current_words[:cut_at + 1]
                chunk_words    = [w['word'] for w in chunk_word_data]
                chunk_end_time = current_word_ends[cut_at]
                chunk_duration = chunk_end_time - current_start
                trimmed_tail   = len(current_words) - 1 - cut_at  # words carried forward
                logger.debug(
                    f"chunk-emit idx={segment_idx} "
                    f"t={current_start:.2f}-{chunk_end_time:.2f}s "
                    f"dur={chunk_duration:.2f}s words={len(chunk_words)} "
                    f"cut={cut_strategy} carry_tail={trimmed_tail}"
                )

                if chunk_words and chunk_duration >= 1.0:
                    chunk_t0 = time.monotonic()
                    text = " ".join(chunk_words)

                    drop_chunk = False
                    reason_rejected = None

                    # 1. Quality Filtering: Check duration
                    if chunk_duration < min_chunk_duration:
                        drop_chunk = True
                        reason_rejected = f"too_short ({chunk_duration:.2f}s < {min_chunk_duration}s)"
                        stats['source_action']['dropped_short'] += 1
                        logger.info(f"  ↪ DROPPED chunk at {current_start:.2f}s (too short: {chunk_duration:.2f}s)")
                    
                    if not drop_chunk:
                        # Read audio for confidence and SNR checks
                        audio_slice = _read_audio_segment(audio_24k_source, current_start, chunk_end_time)

                        # Check confidence
                        confidences = [w.get("confidence", 1.0) for w in chunk_word_data]
                        avg_confidence = np.mean(confidences) if confidences else 1.0
                        if avg_confidence < min_confidence:
                            drop_chunk = True
                            reason_rejected = f"low_confidence ({avg_confidence:.2f} < {min_confidence})"
                            stats['source_action']['dropped_low_quality'] += 1
                            logger.info(f"  ↪ DROPPED chunk at {current_start:.2f}s ({reason_rejected})")

                        # Check SNR
                        if not drop_chunk:
                            snr = _calculate_chunk_snr(audio_slice)
                            if snr < min_snr:
                                drop_chunk = True
                                reason_rejected = f"low_snr ({snr:.1f}dB < {min_snr}dB)"
                                stats['source_action']['dropped_low_quality'] += 1
                                logger.info(f"  ↪ DROPPED chunk at {current_start:.2f}s ({reason_rejected})")

                    # 2. Deduplication: Check for narrator retakes
                    if not drop_chunk and prev_raw_text:
                        # Use SequenceMatcher for a fuzzy text similarity check.
                        # Narrator retakes often vary slightly in wording or ASR noise.
                        sm = difflib.SequenceMatcher(None, prev_raw_text.lower(), text.lower())
                        similarity = sm.ratio()
                        if similarity > 0.85:
                            drop_chunk = True
                            reason_rejected = f"duplicate (similarity {similarity:.2f} > 0.85)"
                            stats['source_action']['deduplicated'] += 1
                            logger.info(f"  ↪ DROPPED chunk at {current_start:.2f}s (duplicate/retake detected)")

                    # ── Source-guided alignment (only when --source is set) ──
                    source_words_for_merge = None
                    if not drop_chunk and source_state is not None:
                        chunk_match_words = alignment.to_words(text)
                        cursor_before = source_state['cursor']
                        sa_start, sa_end, sa_ratio = alignment.find_best_match(
                            chunk_match_words,
                            source_state['orig_match'],
                            cursor_before,
                        )
                        # Three-tier recovery, mirroring compare's run() loop:
                        #
                        #   tier 0: find_best_match (already done above) —
                        #     narrow ±200 word window around cursor.
                        #   tier 1: realign — wide forward search up to 3000
                        #     source words past cursor. Cheap, catches the
                        #     common case where audio skipped a paragraph or
                        #     two of source.
                        #   tier 2: find_anchor_position — full-source scan.
                        #     Expensive but rare; catches catastrophic loss
                        #     where audio jumped chapters, or the EPUB's
                        #     front-matter order put content far from where
                        #     the cursor expected it.
                        #
                        # Each tier only fires when the previous one's result
                        # is too weak to be confident, and each requires the
                        # new ratio to clear a tier-specific bar that's higher
                        # than what `--source-threshold` would otherwise require.
                        # That keeps a chunk from being rescued by a low-
                        # confidence wider match when the local match was
                        # just noise.
                        if sa_ratio < 0.45 and len(chunk_match_words) >= 5:
                            r_start, r_end, r_ratio = alignment.realign(
                                chunk_match_words,
                                source_state['orig_match'],
                                cursor_before,
                            )
                            if r_ratio >= 0.55 and r_ratio > sa_ratio + 0.15:
                                logger.debug(
                                    f"source-realign idx={segment_idx} "
                                    f"local {sa_ratio:.3f} → wide {r_ratio:.3f} "
                                    f"cursor {cursor_before}→{r_end} "
                                    f"(jumped {r_end - cursor_before} words)"
                                )
                                sa_start, sa_end, sa_ratio = r_start, r_end, r_ratio
                            elif r_ratio < 0.30:
                                # Tier 2: full-source scan. Same logic compare
                                # uses for catastrophic alignment loss. Requires
                                # both an absolute bar (>=0.60) AND a clear
                                # improvement over the local ratio (+0.40) so
                                # we don't false-positive on chunks that
                                # genuinely have no source equivalent (audio-
                                # only credits, narrator inserts, etc.) —
                                # those should still be dropped.
                                a_start, a_end, a_ratio = alignment.find_anchor_position(
                                    chunk_match_words,
                                    source_state['orig_match'],
                                    min_ratio=0.6,
                                )
                                if a_ratio >= 0.6 and a_ratio > sa_ratio + 0.4:
                                    # Trim the wide-anchor window down to the
                                    # actual aligned region so the source span
                                    # we use is tight, not the full +slop window.
                                    t_start, t_end = alignment.trim_span_to_alignment(
                                        chunk_match_words,
                                        source_state['orig_match'],
                                        a_start, a_end,
                                    )
                                    if t_end > t_start:
                                        a_start, a_end = t_start, t_end
                                        a_ratio = alignment._ratio(
                                            chunk_match_words,
                                            source_state['orig_match'][a_start:a_end],
                                        )
                                    _jump = a_end - cursor_before
                                    logger.info(
                                        f"  ↪ chunk {segment_idx}: full-source re-anchor "
                                        f"ratio={a_ratio:.3f} cursor "
                                        f"{cursor_before}→{a_end} (jumped "
                                        f"{_jump:+d} words)"
                                    )
                                    if _jump < -5000:
                                        stats['reanchor_backward'] += 1
                                        logger.warning(
                                            f"  ⚠ Large backward re-anchor ({_jump:+d} words) — "
                                            f"source structure may not match this audio file"
                                        )
                                    sa_start, sa_end, sa_ratio = a_start, a_end, a_ratio
                        if sa_ratio >= source_threshold:
                            asr_preview = (text[:60] + '…') if len(text) > 60 else text
                            source_words_for_merge = source_state['orig_display'][sa_start:sa_end]
                            text = ' '.join(source_words_for_merge)
                            source_state['cursor'] = sa_end
                            stats['source_action']['replace'] += 1
                            src_preview = (text[:60] + '…') if len(text) > 60 else text
                            logger.debug(
                                f"source-replace idx={segment_idx} ratio={sa_ratio:.3f} "
                                f"cursor {cursor_before}→{sa_end} (+{sa_end - cursor_before}) "
                                f"asr={asr_preview!r} src={src_preview!r}"
                            )
                        elif keep_unaligned:
                            stats['source_action']['keep_asr'] += 1
                            logger.info(
                                f"  ↪ chunk {segment_idx} kept (ASR text); "
                                f"source ratio {sa_ratio:.2f} < {source_threshold} "
                                f"(cursor stays at {cursor_before})"
                            )
                            # Cursor stays put — don't advance through source
                            # we couldn't confidently match.
                        else:
                            stats['source_action']['dropped'] += 1
                            logger.info(
                                f"  ↪ DROPPED chunk at {current_start:.2f}s "
                                f"(source ratio {sa_ratio:.2f} < {source_threshold}, "
                                f"cursor stays at {cursor_before})"
                            )
                            asr_preview = (text[:80] + '…') if len(text) > 80 else text
                            logger.debug(f"dropped chunk asr={asr_preview!r}")
                            drop_chunk = True
                            reason_rejected = f"low_source_ratio ({sa_ratio:.2f} < {source_threshold})"

                if drop_chunk and reason_rejected:
                    # Log rejected chunk separately for review
                    with open(os.path.join(temp_dir, "rejected_chunks.jsonl"), "a", encoding="utf-8") as rf:
                        json.dump({
                            "segment_idx": segment_idx,
                            "reason": reason_rejected,
                            "text": text,
                            "start": current_start,
                            "end": chunk_end_time,
                            "duration": chunk_duration
                        }, rf)
                        rf.write("\n")

                if chunk_words and chunk_duration >= 1.0 and not drop_chunk:
                    # Build user prompt with optional preceding context for continuity
                    ctx = " ".join(list(context)[-2:]) if context else ""
                    if ctx:
                        user_prompt = f"Previous context: {ctx}\n\nAnnotate this segment:\n{text}"
                    else:
                        user_prompt = f"Annotate this segment:\n{text}"

                    try:
                        response = llm.create_chat_completion(
                            messages=[
                                {"role": "system", "content": TTS_ANNOTATION_SYSTEM_PROMPT},
                                {"role": "user", "content": user_prompt},
                            ],
                            max_tokens=512,
                            temperature=0.3,  # Lower temp for more deterministic structured output
                        )
                        annotated_raw = response["choices"][0]["message"]["content"].strip()
                        # If we have source words for this chunk, run the LLM's
                        # output through compare's merge logic to GUARANTEE the
                        # saved text uses source-words + LLM-markers — even when
                        # the LLM paraphrased, added, or dropped words while
                        # adding prosody. Without --source we keep the legacy
                        # sanitiser path (multi-word emph + dot-pad + fused-
                        # punct strip, no source-word enforcement).
                        if source_words_for_merge is not None:
                            annotated = alignment.merge_annotations_with_source(
                                annotated_raw, source_words_for_merge
                            )
                        else:
                            annotated = _sanitize_annotation(annotated_raw)
                        if annotated != annotated_raw:
                            stats['sanitize_changed'] += 1
                            logger.debug(
                                f"sanitize idx={segment_idx} "
                                f"raw_len={len(annotated_raw)} clean_len={len(annotated)} "
                                f"raw={annotated_raw[:120]!r}"
                            )
                        stats['llm_success'] += 1
                        logger.debug(
                            f"llm-ok idx={segment_idx} "
                            f"prompt_chars={len(user_prompt)} response_chars={len(annotated_raw)}"
                        )
                        if segment_idx == next_segment_idx:
                            logger.info(f"✓ LLM GPU inference confirmed - {os.path.basename(active_model_path)} responding on GPU")
                    except Exception as e:
                        stats['llm_fail'] += 1
                        logger.warning(f"Annotation failed for segment {segment_idx}, using original text: {e}")
                        logger.debug(f"llm-fail idx={segment_idx}: {traceback.format_exc()}")
                        annotated = text

                    # Read audio segment (in-memory slice or disk seek)
                    audio_slice = _read_audio_segment(audio_24k_source, current_start, chunk_end_time)
                    actual_duration = len(audio_slice) / 24000.0
                    expected_duration = chunk_end_time - current_start
                    if abs(actual_duration - expected_duration) > 0.1:
                        # 0.1s mismatch is normally just rounding; bigger means
                        # we hit end-of-file or _read_audio_segment ran short.
                        stats['audio_short'] += 1
                        logger.warning(
                            f"audio slice mismatch idx={segment_idx} "
                            f"expected {expected_duration:.3f}s got {actual_duration:.3f}s "
                            f"(req {current_start:.3f}-{chunk_end_time:.3f}s)"
                        )

                    if len(audio_slice) > 0:
                        stats['chunk_durations'].append(actual_duration)
                        seg_name = f"sample_{segment_idx:04d}.wav"
                        wav_path = os.path.join(temp_dir, seg_name)
                        sf.write(wav_path, audio_slice, 24000)
                        # Force the WAV to disk before recording metadata that
                        # references it — otherwise power loss can leave a
                        # truncated WAV with a metadata line claiming the full
                        # duration.
                        wav_fd = os.open(wav_path, os.O_RDWR)
                        try:
                            os.fsync(wav_fd)
                        finally:
                            os.close(wav_fd)

                        # ── Metadata Tagging ──
                        try:
                            import mutagen.wave
                            import mutagen.id3
                            audio_tags = mutagen.wave.WAVE(wav_path)
                            if audio_tags.tags is None:
                                audio_tags.add_tags()
                            if book_title:
                                audio_tags.tags.add(mutagen.id3.TALB(encoding=3, text=book_title))
                            if character:
                                audio_tags.tags.add(mutagen.id3.TPE1(encoding=3, text=character))
                            if narrator_style:
                                audio_tags.tags.add(mutagen.id3.COMM(encoding=3, text=narrator_style, lang="eng", desc=""))
                            audio_tags.tags.add(mutagen.id3.TIT2(encoding=3, text=seg_name))
                            audio_tags.save()
                        except ImportError:
                            logger.debug("mutagen not installed, skipping metadata tagging")
                        except Exception as tag_e:
                            logger.warning(f"Failed to write metadata tags to {seg_name}: {tag_e}")

                        entry = {
                            "audio_filepath": seg_name,
                            "text": annotated,
                            "duration": len(audio_slice) / 24000,
                            "start": current_start,
                            "end": chunk_end_time,
                            "speaker": Counter(w.get("speaker", "UNKNOWN") for w in chunk_word_data).most_common(1)[0][0]
                        }
                        if character:
                            entry["character"] = character
                        if narrator_style:
                            entry["narrator_style"] = narrator_style
                        metadata.append(entry)

                        # Append to checkpoint and fsync immediately so power
                        # loss can lose at most the in-progress chunk.
                        checkpoint_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        checkpoint_file.flush()
                        os.fsync(checkpoint_file.fileno())

                        chunk_times.append(time.monotonic() - chunk_t0)

                        # Dynamic ETA from rolling average
                        if (segment_idx + 1) % 10 == 0:
                            avg_chunk_s = sum(chunk_times) / len(chunk_times)
                            completed_this_run = segment_idx - next_segment_idx + 1
                            elapsed_s = time.monotonic() - annotation_start_time
                            remaining_chunks = max(0, estimated_chunks_total - segment_idx - 1)
                            remaining_s = remaining_chunks * avg_chunk_s
                            logger.info(
                                f"  ↳ Progress: {segment_idx + 1}/{estimated_chunks_total} chunks "
                                f"| Avg: {avg_chunk_s:.1f}s/chunk "
                                f"| Elapsed: {format_duration(elapsed_s)} "
                                f"| ETA: {format_duration(remaining_s)}"
                            )
                            log_gpu_stats(f"annotation segment {segment_idx + 1}/{estimated_chunks_total}")

                        segment_idx += 1
                        prev_raw_text = text  # Record for next chunk deduplication check

                    context.append(text)
                # Carry the post-cut tail forward as the start of the next chunk.
                # The chunker previously dropped everything; now any words that
                # the look-back trimmed off become the seed of the next chunk.
                current_words       = current_words[cut_at + 1:]
                current_word_starts = current_word_starts[cut_at + 1:]
                current_word_ends   = current_word_ends[cut_at + 1:]
                current_start = current_word_starts[0] if current_word_starts else chunk_end_time
    finally:
        checkpoint_file.close()
        del llm
        clear_vram()

    # ── Comprehensive end-of-chunker summary ──────────────────────────────────
    # Most of what's useful for iterating on the chunker / source mode is in
    # the histograms below. Compare run-to-run to see which cut strategy is
    # firing, how many chunks the source mode dropped vs. replaced, whether
    # the LLM is failing repeatedly, etc.
    chunks_emitted_this_run = segment_idx - next_segment_idx
    logger.info(f"✓ Annotation complete: {segment_idx} total segments "
                f"({chunks_emitted_this_run} new this run)")

    # Cut-strategy histogram (per chunk-emit decision, includes dropped chunks)
    if stats['cut_strategy']:
        total_cuts = sum(stats['cut_strategy'].values())
        logger.info(f"  Cut strategy distribution ({total_cuts} chunks):")
        for strategy in ('sentence_end', 'pause', 'fallback', 'is_final',
                         'too_few_words', 'undersized'):
            count = stats['cut_strategy'].get(strategy, 0)
            if count:
                pct = 100 * count / total_cuts
                logger.info(f"    {strategy:<14} : {count:>6} ({pct:5.1f}%)")

    # Source-mode histogram (only when --source was used)
    if source_state is not None:
        sa_replace = stats['source_action']['replace']
        sa_keep    = stats['source_action']['keep_asr']
        sa_drop    = stats['source_action']['dropped']
        sa_short   = stats['source_action']['dropped_short']
        sa_dedup   = stats['source_action']['deduplicated']
        sa_low_qual = stats['source_action']['dropped_low_quality']
        sa_total   = sa_replace + sa_keep + sa_drop + sa_short + sa_dedup + sa_low_qual
        if sa_total:
            logger.info(f"  Source-guided actions ({sa_total} chunks aligned):")
            logger.info(f"    replace        : {sa_replace:>6} ({100*sa_replace/sa_total:5.1f}%)")
            logger.info(f"    keep_asr       : {sa_keep:>6} ({100*sa_keep/sa_total:5.1f}%)")
            logger.info(f"    dropped        : {sa_drop:>6} ({100*sa_drop/sa_total:5.1f}%)")
            if sa_short:
                logger.info(f"    dropped_short  : {sa_short:>6} ({100*sa_short/sa_total:5.1f}%)")
            if sa_dedup:
                logger.info(f"    deduplicated   : {sa_dedup:>6} ({100*sa_dedup/sa_total:5.1f}%)")
            if sa_low_qual:
                logger.info(f"    dropped_low_qual: {sa_low_qual:>6} ({100*sa_low_qual/sa_total:5.1f}%)")
        _src_cursor = source_state['cursor']
        _src_total  = len(source_state['orig_match'])
        _src_pct    = 100 * _src_cursor / _src_total if _src_total else 0
        logger.info(f"  Source cursor finished at word "
                    f"{_src_cursor:,} / {_src_total:,} ({_src_pct:.0f}% coverage)")
        if _src_pct < 60:
            logger.warning(
                f"  ⚠ Low source coverage ({_src_pct:.0f}%): the EPUB likely contains "
                f"more content than this audio file (e.g. multiple volumes)."
            )
        if stats['reanchor_backward'] >= 2:
            logger.warning(
                f"  ⚠ {stats['reanchor_backward']} large backward re-anchors detected — "
                f"source text structure does not align well with this audio."
            )

    # LLM + sanitisation
    llm_total = stats['llm_success'] + stats['llm_fail']
    if llm_total:
        _san_pct = 100 * stats['sanitize_changed'] / llm_total if llm_total else 0
        logger.info(f"  LLM annotations: {stats['llm_success']} ok, "
                    f"{stats['llm_fail']} failed "
                    f"({stats['sanitize_changed']} cleaned by sanitiser, {_san_pct:.0f}%)")
        if _san_pct > 80:
            logger.warning(
                f"  ⚠ High sanitiser rate ({_san_pct:.0f}%): LLM output is consistently "
                f"reformatted — review prompt template or model output format."
            )

    # Duration distribution of emitted chunks (audio-side, not source-side)
    durs = stats['chunk_durations']
    if durs:
        logger.info(f"  Emitted chunk durations (n={len(durs)}):")
        logger.info(f"    min  {min(durs):.2f}s   p50 {_percentile(durs, 50):.2f}s   "
                    f"p95 {_percentile(durs, 95):.2f}s   max {max(durs):.2f}s")
        # Useful flag: if many chunks are way under chunk_size, the cut
        # logic is picking earlier breaks than the user expects.
        short_count = sum(1 for d in durs if d < chunk_size * 0.5)
        if short_count:
            logger.info(f"    chunks < 50% target ({chunk_size * 0.5:.1f}s): "
                        f"{short_count} ({100*short_count/len(durs):.1f}%)")

    if stats['audio_short']:
        logger.warning(f"  ⚠ Audio-slice mismatches: {stats['audio_short']} chunk(s) "
                       f"shorter than expected — likely end-of-file or read truncation")

    if metadata:
        total_duration = sum(m["duration"] for m in metadata)
        logger.info(f"  Total audio in dataset    : {total_duration:.1f}s "
                    f"({total_duration/60:.1f} min)")
        logger.info(f"  Average segment duration  : {total_duration/len(metadata):.2f}s")

    return metadata

# ── Output zip naming from source metadata ────────────────────────────────────
# When --output is left at the default (or another well-known placeholder), try
# to derive a self-describing zip name from the source ePub's title/author, or
# fall back to the source file's stem. Keeps generated dataset names useful
# without forcing the caller to compute one.
_NAME_GENERIC_OUTPUTS = {"alexandria_dataset.zip", "dataset.zip", "output.zip"}
_NAME_MAX_PART_LEN = 80  # per-token cap; long ebook titles can run 100+ chars
_NAME_DASH_TRANSLATE = str.maketrans({"—": "-", "–": "-", "−": "-"})
_NAME_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_-]")
_NAME_COLLAPSE_RE = re.compile(r"([_-])[_-]+")  # _-_ → -, ___ → _


def _sanitize_name_part(text) -> str:
    """Make a single naming token filesystem-safe: normalize unicode dashes,
    spaces→_, strip everything else, collapse runs of _ and -.

    Coerce to str() because pathological ePubs can yield BeautifulSoup tags or
    tuples from Dublin Core fields, and we'd rather degrade to a sanitized
    repr than crash the pipeline before processing starts.
    """
    text = str(text).strip().translate(_NAME_DASH_TRANSLATE).replace(" ", "_")
    text = _NAME_SANITIZE_RE.sub("", text)
    text = _NAME_COLLAPSE_RE.sub(r"\1", text)
    return text[:_NAME_MAX_PART_LEN].strip("_-")


def extract_metadata_for_naming(source_path: str) -> tuple[Optional[str], Optional[str]]:
    """Return (title, author) from an ePub, or (None, None) on any failure.

    Non-epub sources and missing/parse-failing ePubs return (None, None) so the
    caller can fall through to the filename-stem branch without special-casing.
    """
    if not source_path or not os.path.exists(source_path):
        return None, None
    if not source_path.lower().endswith(".epub"):
        return None, None
    if not getattr(alignment, "EPUB_AVAILABLE", False):
        return None, None
    try:
        book = alignment.epub.read_epub(source_path, options={"ignore_ncx": True})
        title_md = book.get_metadata("DC", "title")
        author_md = book.get_metadata("DC", "creator")
        title = title_md[0][0] if title_md else None
        author = author_md[0][0] if author_md else None
        return title, author
    except Exception as e:
        logger.warning(f"Could not read ePub metadata from {source_path}: {e}")
        return None, None


def generate_zip_filename(source_path: Optional[str],
                          title: Optional[str],
                          author: Optional[str]) -> str:
    """Derive a zip filename from ePub metadata, source stem, or fall back to
    'alexandria_dataset.zip'. Sanitization keeps only [A-Za-z0-9_-]."""
    safe_title = _sanitize_name_part(title) if title else ""
    safe_author = _sanitize_name_part(author) if author else ""
    if safe_title and safe_author:
        return f"{safe_title}_{safe_author}.zip"
    if safe_title:
        return f"{safe_title}.zip"
    if source_path:
        stem = os.path.splitext(os.path.basename(source_path))[0]
        safe_stem = _sanitize_name_part(stem)
        if safe_stem:
            return f"{safe_stem}.zip"
    return "alexandria_dataset.zip"


def maybe_autoname_output(output: str, source_path: Optional[str], title: Optional[str] = None, author: Optional[str] = None) -> str:
    """Replace a generic --output value with a derived name, preserving the
    user-supplied directory if one was given. Returns the original `output`
    unchanged when the caller pinned a non-generic name (anything outside
    _NAME_GENERIC_OUTPUTS) so an explicit choice is always respected."""
    basename = os.path.basename(output) if output else ""
    if basename not in _NAME_GENERIC_OUTPUTS:
        return output
    derived = generate_zip_filename(source_path, title, author)
    parent = os.path.dirname(output) if output else ""
    return os.path.join(parent, derived) if parent else derived


def _assign_speakers_to_words(word_segments, speaker_segments):
    """Assign a speaker to each word based on diarization results."""
    if not speaker_segments:
        for word in word_segments:
            word["speaker"] = "UNKNOWN"
        return word_segments, {"UNKNOWN"}

    IntervalTree, Interval = _lazy_import_intervaltree()
    tree = IntervalTree()
    for seg in speaker_segments:
        tree.add(Interval(seg["start"], seg["end"], seg["speaker"]))

    unassigned_words = 0
    speaker_counts = Counter()

    for word in word_segments:
        word_mid_point = (word["start"] + word["end"]) / 2
        overlapping_intervals = tree.at(word_mid_point)
        
        if overlapping_intervals:
            speaker = next(iter(overlapping_intervals)).data
            word["speaker"] = speaker
            speaker_counts[speaker] += 1
        else:
            word["speaker"] = "UNKNOWN"
            unassigned_words += 1

    logger.info("▶ Speaker assignment summary:")
    for speaker, count in speaker_counts.most_common():
        logger.info(f"  - {speaker}: {count} words")
    if unassigned_words > 0:
        logger.warning(f"  - UNKNOWN: {unassigned_words} words (no speaker segment overlap)")
        
    unique_speakers = set(speaker_counts.keys())
    if unassigned_words > 0:
        unique_speakers.add("UNKNOWN")
        
    return word_segments, unique_speakers


def _create_zip_dataset(metadata: List[Dict], output_path: str, val_split: float = 0.10, zip_max_files: int = 200, unique_speakers: set = None):
    """Bundle annotated chunks and metadata into segmented ZIP files (volumes),
    grouped by speaker, character, and narrator style."""
    temp_dir = "dataset_temp"
    
    if not metadata:
        logger.warning("No metadata to save to ZIP.")
        return

    # 1. Group metadata by speaker, then (character, style)
    groups = {}
    for entry in metadata:
        speaker = entry.get("speaker", "UNKNOWN")
        char = entry.get("character", "narrator")
        style = entry.get("narrator_style", "default")
        key = (speaker, char, style)
        groups.setdefault(key, []).append(entry)

    logger.info(f"▶ Creating segmented ZIP volumes (max {zip_max_files} files/vol, val_split={val_split:.0%})")
    logger.info(f"  ├─ Unique speaker/character/style combinations: {len(groups)}")

    import random
    base, ext = os.path.splitext(output_path)
    
    total_train = 0
    total_val = 0
    total_vols = 0

    for (speaker, char, style), group_metadata in groups.items():
        num_vols = (len(group_metadata) + zip_max_files - 1) // zip_max_files
        
        # Sanitise for filenames
        safe_speaker = _sanitize_name_part(speaker)
        safe_char = _sanitize_name_part(char)
        safe_style = _sanitize_name_part(style)
        
        for vol_idx in range(num_vols):
            start_idx = vol_idx * zip_max_files
            end_idx = min(start_idx + zip_max_files, len(group_metadata))
            vol_metadata = group_metadata[start_idx:end_idx]
            
            # Generate volume path: base_Speaker_Character_Style_volNN.zip
            parts = [base]
            if unique_speakers and len(unique_speakers) > 1:
                parts.append(safe_speaker)
            if safe_char != "narrator":
                parts.append(safe_char)
            if safe_style != "default":
                parts.append(safe_style)
            if num_vols > 1:
                parts.append(f"vol{vol_idx + 1:02d}")
            
            if len(parts) == 1:
                vol_path = output_path
            else:
                vol_path = "_".join(parts) + ext

            # Partition this volume into train/val
            indices = list(range(len(vol_metadata)))
            random.seed(42 + vol_idx) # stable per volume
            random.shuffle(indices)
            
            v_count = int(len(vol_metadata) * val_split)
            v_indices = set(indices[:v_count])
            
            train_meta = []
            val_meta = []

            with zipfile.ZipFile(vol_path, "w", zipfile.ZIP_DEFLATED) as z:
                for i, entry in enumerate(vol_metadata):
                    wav_name = entry["audio_filepath"]
                    src_path = os.path.join(temp_dir, wav_name)

                    if not os.path.exists(src_path):
                        logger.warning(f"  ⚠ Audio file not found for ZIP {vol_path}: {wav_name}")
                        continue

                    is_val = (i in v_indices)
                    folder = "val" if is_val else "train"
                    zip_wav_path = f"{folder}/{wav_name}"

                    zip_entry = entry.copy()
                    zip_entry["audio_filepath"] = zip_wav_path

                    if is_val:
                        val_meta.append(zip_entry)
                        total_val += 1
                    else:
                        train_meta.append(zip_entry)
                        total_train += 1

                    z.write(src_path, zip_wav_path)

                # Write partitioned metadata.jsonl files
                if train_meta:
                    train_jsonl = "\n".join([json.dumps(e, ensure_ascii=False) for e in train_meta]) + "\n"
                    z.writestr("train/metadata.jsonl", train_jsonl)
                if val_meta:
                    val_jsonl = "\n".join([json.dumps(e, ensure_ascii=False) for e in val_meta]) + "\n"
                    z.writestr("val/metadata.jsonl", val_jsonl)

                # Volume manifest
                vol_manifest = sorted(train_meta + val_meta, key=lambda x: x["audio_filepath"])
                if vol_manifest:
                    manifest_jsonl = "\n".join([json.dumps(e, ensure_ascii=False) for e in vol_manifest]) + "\n"
                    z.writestr("metadata.jsonl", manifest_jsonl)
            
            total_vols += 1
            logger.info(f"  ✓ Volume {total_vols} saved: {vol_path} ({len(vol_metadata)} segments)")

    durations = [m["duration"] for m in metadata]
    logger.info("=" * 70)
    logger.info(f"Total segments across {total_vols} volume(s): {len(metadata)}")
    if durations:
        logger.info(f"  ├─ Total Train: {total_train} segments")
        logger.info(f"  ├─ Total Val  : {total_val} segments")
        logger.info(f"  ├─ Average duration: {np.mean(durations):.2f}s")
        logger.info(f"  └─ Total audio: {sum(durations)/60:.1f} minutes")
    logger.info("=" * 70)
    logger.info(f"✓ ALL VOLUMES COMPLETED.")
def main():
    parser = argparse.ArgumentParser(
        description="Alexandria Master Preparer - ROCm Compatible"
    )

    parser.add_argument("--audio", required=True, help="Input audio file")
    parser.add_argument("--model",
                        help="Primary GGUF model (recommended: Qwen2.5-14B-Instruct-Q6_K.gguf)")
    parser.add_argument("--fallback-model",
                        help="Optional fallback GGUF model if --model fails to load "
                             "(e.g., Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf)")
    parser.add_argument("--skip-annotation", action="store_true")
    parser.add_argument("--chunk-size", type=float, default=10.0)
    parser.add_argument("--min-chunk-duration", type=float, default=2.0, help="Minimum duration for an audio chunk to be kept (default: 2.0s)")
    parser.add_argument("--min-confidence", type=float, default=0.85, help="Minimum average word confidence for a chunk to be kept (default: 0.85)")
    parser.add_argument("--min-snr", type=int, default=25, help="Minimum Signal-to-Noise Ratio (SNR) in dB for a chunk to be kept (default: 25)")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of chunks to process")
    parser.add_argument("--phase", choices=["asr", "enrich", "annotate"], help="Run only a specific phase (internal use for ROCm isolation)")
    parser.add_argument("--asr-output", help="Path to save/load ASR word segments (default: dataset_temp/asr_segments.json)")
    parser.add_argument("--scratch-audio", help="Path to 24k scratch WAV (default: dataset_temp/audio_24k.wav)")
    parser.add_argument("--output", default="alexandria_dataset.zip",
                        help="Output ZIP path. If left at the default — or set to "
                             "'dataset.zip' / 'output.zip' — a name is auto-derived "
                             "from --source (ePub title+author, or filename stem). "
                             "Any other value is used verbatim.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing dataset_temp/ instead of starting over")
    parser.add_argument("--val-split", type=float, default=0.10,
                        help="Ratio of segments to carve out for validation (default: 0.10)")
    parser.add_argument("--zip-max-files", type=int, default=200,
                        help="Maximum files per ZIP volume (default: 200)")

    # ── Source-guided mode ────────────────────────────────────────────────────
    # When --source is provided, each ASR chunk is fuzzy-aligned against the
    # source text. Chunks whose alignment ratio meets --source-threshold get
    # their text replaced with the source spelling before the LLM annotates,
    # so character names and dialect spellings come out correct. Chunks below
    # threshold are dropped (likely audio-only material — credits, narrator
    # intros, content the source doesn't have) unless --keep-unaligned is set.
    parser.add_argument("--source", metavar="PATH",
                        help="Optional source EPUB or TXT. Enables source-guided "
                             "chunking: chunk text is replaced with the source's "
                             "spelling for high-confidence alignments, audio-only "
                             "passages are dropped.")
    parser.add_argument("--source-threshold", type=float, default=0.65, metavar="N",
                        help="Minimum alignment ratio to keep a chunk in source-guided "
                             "mode (default: 0.65). Chunks below this are dropped unless "
                             "--keep-unaligned is set.")
    parser.add_argument("--keep-unaligned", action="store_true",
                        help="When --source is set, keep chunks that fall below "
                             "--source-threshold and use their ASR text instead of "
                             "dropping them (default: strict-drop).")
    parser.add_argument("--source-start", type=int, metavar="N",
                        help="Start source alignment at word N (skip auto-anchor)")
    parser.add_argument("--source-start-text", metavar="TEXT",
                        help="Search source for TEXT and start alignment there")
    parser.add_argument("--no-auto-anchor", action="store_true",
                        help="When --source is set, disable auto-anchor (start at word 0)")
    
    # ── Metadata Tagging ──────────────────────────────────────────────────────
    parser.add_argument("--book-title", help="Book title for metadata tagging (overrides ePub title)")
    parser.add_argument("--character", help="Character name for metadata tagging")
    parser.add_argument("--narrator-style", help="Narrator style for metadata tagging")
    
    # ── Speaker Diarization ───────────────────────────────────────────────────
    parser.add_argument("--diarize", action="store_true", help="Enable speaker diarization using pyannote.audio")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"), help="Hugging Face token for pyannote access")
    parser.add_argument("--auto-detect-speakers", action="store_true", help="Auto-detect narrator count by analyzing audio and logging the number of unique speakers.")
    
    # ── LLM Enrichment ────────────────────────────────────────────────────────
    parser.add_argument("--enrich-with-llm", action="store_true", help="Enable LLM-based transcript enrichment.")
    parser.add_argument("--llm-model-path", help="Path to the GGUF LLM model file for transcript enrichment. Required if --enrich-with-llm is set.")
    parser.add_argument("--enrich-speaker-attribution", action="store_true", help="Instruct LLM to extract speaker attribution.")
    parser.add_argument("--enrich-narration-style", action="store_true", help="Instruct LLM to extract narration style.")
    parser.add_argument("--enrich-emotional-tone", action="store_true", help="Instruct LLM to extract emotional tone.")

    args = parser.parse_args()

    if not args.skip_annotation and not args.model:
        parser.error("--model required unless --skip-annotation")

    if args.enrich_with_llm and not args.llm_model_path:
        parser.error("--llm-model-path is required when --enrich-with-llm is set.")

    # Auto-derive --output filename from --source metadata when the caller left
    # the default (or another generic placeholder). Pinned names pass through.
    epub_title, epub_author = extract_metadata_for_naming(args.source) if args.source else (None, None)
    book_title = args.book_title or epub_title
    derived_output = maybe_autoname_output(args.output, args.source, epub_title, epub_author)
    if derived_output != args.output:
        logger.info(f"Auto-derived output filename: {derived_output} "
                    f"(was: {args.output})")
        args.output = derived_output

    # ── Phase Orchestration ──────────────────────────────────────────────────
    # ROCm HIP contexts from PyTorch (Wav2Vec2) and llama-cpp often conflict
    # if initialized in the same process. We split them into separate phases.
    if args.phase is None:
        logger.info("=" * 70)
        logger.info("Alexandria Master Preparer - Phase Orchestrator (ROCm Isolation)")
        logger.info("=" * 70)
        
        # 1. Run ASR Phase (if not already completed and resuming)
        asr_output_path = args.asr_output or os.path.join("dataset_temp", "asr_segments.json")
        should_run_asr = True
        if args.resume and os.path.exists(asr_output_path):
            logger.info(f"▶ ASR output found at {asr_output_path}, skipping ASR phase due to --resume")
            should_run_asr = False

        if should_run_asr:
            asr_cmd = [sys.executable, __file__, "--phase", "asr"]
            # Pass all original arguments except potentially conflicting ones
            for arg in sys.argv[1:]:
                if arg not in ["--phase", "asr", "enrich", "annotate"]:
                    asr_cmd.append(arg)

            logger.info("▶ Launching ASR Phase...")
            res = subprocess.run(asr_cmd)
            if res.returncode != 0:
                logger.error(f"ASR Phase failed with exit code {res.returncode}")
                sys.exit(res.returncode)

        # 2. Run LLM Enrichment Phase (if requested)
        enriched_output_path = os.path.join("dataset_temp", "enriched_segments.json")
        should_run_enrich = args.enrich_with_llm
        if should_run_enrich and args.resume and os.path.exists(enriched_output_path):
            logger.info(f"▶ Enriched output found at {enriched_output_path}, skipping enrichment phase due to --resume")
            should_run_enrich = False

        if should_run_enrich:
            if not args.llm_model_path:
                logger.error("▶ LLM enrichment requested but --llm-model-path not provided")
                sys.exit(1)

            # Run enrichment as a subprocess to reuse the same chunking logic
            enrich_cmd = [sys.executable, __file__, "--phase", "enrich"]
            for arg in sys.argv[1:]:
                if arg not in ["--phase", "asr", "enrich", "annotate"]:
                    enrich_cmd.append(arg)

            logger.info("▶ Launching LLM Enrichment Phase...")
            res = subprocess.run(enrich_cmd)
            if res.returncode != 0:
                logger.error(f"LLM Enrichment Phase failed with exit code {res.returncode}")
                sys.exit(res.returncode)

        # 3. Run Annotation Phase
        ann_cmd = [sys.executable, __file__, "--phase", "annotate"]
        for arg in sys.argv[1:]:
            if arg not in ["--phase", "asr", "enrich", "annotate"]:
                ann_cmd.append(arg)

        logger.info("▶ Launching Annotation Phase...")
        res = subprocess.run(ann_cmd)
        if res.returncode == 0:
            # Clean up scratch audio and temp dir on success
            audio_24k_scratch = args.scratch_audio or os.path.join("dataset_temp", "audio_24k_scratch.wav")
            if os.path.exists(audio_24k_scratch):
                try:
                    os.remove(audio_24k_scratch)
                    logger.debug(f"Removed scratch audio: {audio_24k_scratch}")
                except Exception:
                    pass
            if os.path.exists("dataset_temp"):
                try:
                    shutil.rmtree("dataset_temp")
                    logger.debug("Cleaned up dataset_temp/")
                except Exception:
                    pass
            logger.info(f"Log file saved to: {log_file}")
        sys.exit(res.returncode)

    # ── Individual Phase Execution ───────────────────────────────────────────
    
    # Standard paths for intermediate files
    temp_dir = "dataset_temp"
    os.makedirs(temp_dir, exist_ok=True)
    asr_output_path = args.asr_output or os.path.join(temp_dir, "asr_segments.json")
    audio_24k_path = args.scratch_audio or os.path.join(temp_dir, "audio_24k_scratch.wav")

    audio_24k_scratch = audio_24k_path  # for the finally block cleanup

    try:
        if args.phase == "asr":
            t = _lazy_import_torch()
            device = "cuda" if t.cuda.is_available() else "cpu"

            logger.info("-" * 70)
            logger.info(f"PHASE: ASR (Device: {device})")
            logger.info("-" * 70)
            log_torch_info()

            progress.start("Validate inputs")
            validate_inputs(args)
            progress.complete()

            progress.start("Load audio")
            logger.debug(f"Loading audio from {args.audio} (single read)...")
            load_t0 = time.monotonic()
            is_oversized, _, _ = _wav_overflow_info(args.audio)

            if is_oversized:
                logger.info("  Using ffmpeg loader (oversized WAV)")
                _ffmpeg_decode_to_wav(args.audio, audio_24k_path, 24000, mono=True)
                sf_info_24k = sf.info(audio_24k_path)
                duration_secs = sf_info_24k.duration
                logger.info(f"  Audio: {duration_secs:.1f}s @ {sf_info_24k.frames} samples (loaded in {time.monotonic()-load_t0:.1f}s)")
                logger.debug(f"  Decoding 16 kHz stream for ASR via ffmpeg...")
                audio_16k = _ffmpeg_decode_to_numpy(args.audio, 16000, mono=True)
            else:
                l = _lazy_import_librosa()
                audio_native, native_sr = l.load(args.audio, sr=None, mono=True)
                logger.debug(f"  Native sample rate: {native_sr}Hz, duration: {len(audio_native)/native_sr:.1f}s")

                if native_sr == 16000:
                    audio_16k = audio_native
                else:
                    logger.debug(f"  Resampling to 16kHz (in memory)...")
                    audio_16k = l.resample(audio_native, orig_sr=native_sr, target_sr=16000)

                if native_sr == 24000:
                    audio_24k = audio_native
                else:
                    logger.debug(f"  Resampling to 24kHz (in memory)...")
                    audio_24k = l.resample(audio_native, orig_sr=native_sr, target_sr=24000)

                if native_sr not in (16000, 24000):
                    del audio_native
                    gc.collect()

                duration_secs = len(audio_24k) / 24000
                logger.info(f"  Audio: {duration_secs:.1f}s @ {len(audio_24k)} samples (loaded in {time.monotonic()-load_t0:.1f}s)")

                logger.debug(f"  Spilling 24kHz audio to scratch file: {audio_24k_scratch}")
                sf.write(audio_24k_scratch, audio_24k, 24000, subtype="FLOAT")
                del audio_24k
                gc.collect()

                scratch_size_mb = os.path.getsize(audio_24k_scratch) / (1024 * 1024)
                logger.info(f"  ├─ Scratch audio: {audio_24k_scratch} ({scratch_size_mb:.1f} MB) - freed from RAM")

            progress.complete()

            # Diarize speakers if requested
            if args.diarize:
                progress.start("Diarize speakers")
                device_str = "cuda" if _lazy_import_torch().cuda.is_available() else "cpu"
                speaker_segments = diarize_audio(audio_24k_path, args.hf_token, device=device_str)
                if speaker_segments:
                    diarization_path = os.path.join(temp_dir, "diarization.json")
                    with open(diarization_path, "w") as f:
                        json.dump(speaker_segments, f, indent=2)
                    logger.info(f"  ✓ Found {len(speaker_segments)} speaker segments. Saved to {diarization_path}")
                else:
                    logger.warning("  ⚠ Diarization produced no segments. Continuing without speaker data.")
                progress.complete()

            progress.start("Transcribe audio")
            word_segments, detected_lang = choose_and_transcribe(audio_16k, device, args.lang, limit=args.limit)
            logger.info(f"  Detected language: {detected_lang}")

            # Save ASR results for next phase
            logger.info(f"▶ Saving ASR segments to {asr_output_path}...")
            with open(asr_output_path, "w", encoding="utf-8") as f:
                json.dump({
                    "detected_lang": detected_lang,
                    "word_segments": word_segments,
                    "audio_duration": duration_secs
                }, f)

            del audio_16k
            clear_vram()
            progress.complete()
            logger.info("✓ ASR Phase completed successfully.")
            return 0

        elif args.phase == "enrich":
            logger.info("-" * 70)
            logger.info("PHASE: LLM Enrichment")
            logger.info("-" * 70)

            if not os.path.exists(asr_output_path):
                logger.error(f"ASR results not found at {asr_output_path}. Run ASR phase first.")
                sys.exit(1)

            if not args.llm_model_path:
                logger.error("--llm-model-path is required for enrichment phase")
                sys.exit(1)

            enriched_output_path = os.path.join("dataset_temp", "enriched_segments.json")
            
            logger.info(f"▶ Loading ASR results from {asr_output_path}...")
            with open(asr_output_path, "r", encoding="utf-8") as f:
                asr_data = json.load(f)

            # Extract word_segments and group into chunks for enrichment
            word_segments = asr_data.get("word_segments", [])
            
            # Group words into chunks (e.g., 10 seconds per chunk)
            chunk_duration = 10.0  # seconds
            chunks = []
            current_chunk_words = []
            chunk_start = None
            chunk_end = None
            
            for word in word_segments:
                if chunk_start is None:
                    chunk_start = word.get("start", 0)
                    chunk_end = chunk_start + chunk_duration
                
                if word.get("start", 0) <= chunk_end:
                    current_chunk_words.append(word)
                else:
                    # Save current chunk and start new one
                    if current_chunk_words:
                        chunk_text = " ".join(w.get("word", "") for w in current_chunk_words)
                        chunks.append({
                            "text": chunk_text,
                            "start": current_chunk_words[0].get("start", 0),
                            "end": current_chunk_words[-1].get("end", 0),
                            "speaker": "UNKNOWN",
                            "words": current_chunk_words
                        })
                    current_chunk_words = [word]
                    chunk_start = word.get("start", 0)
                    chunk_end = chunk_start + chunk_duration
            
            # Don't forget the last chunk
            if current_chunk_words:
                chunk_text = " ".join(w.get("word", "") for w in current_chunk_words)
                chunks.append({
                    "text": chunk_text,
                    "start": current_chunk_words[0].get("start", 0),
                    "end": current_chunk_words[-1].get("end", 0),
                    "speaker": "UNKNOWN",
                    "words": current_chunk_words
                })
            
            logger.info(f"  Created {len(chunks)} chunks for LLM enrichment")
            
            # Save chunks for llm_enricher.py
            asr_chunks_path = os.path.join("dataset_temp", "asr_chunks_for_enrich.json")
            with open(asr_chunks_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, indent=2)

            # Build command for llm_enricher.py
            enrich_cmd = [sys.executable, "llm_enricher.py",
                         "--model-path", args.llm_model_path,
                         "--input-file", asr_chunks_path,
                         "--output-file", enriched_output_path]

            logger.info(f"▶ Running LLM enrichment: {' '.join(enrich_cmd)}")
            res = subprocess.run(enrich_cmd)
            if res.returncode != 0:
                logger.error(f"LLM enrichment failed with exit code {res.returncode}")
                sys.exit(res.returncode)

            if not os.path.exists(enriched_output_path):
                logger.error(f"Enrichment output file not found at {enriched_output_path}")
                sys.exit(1)

            logger.info("✓ LLM Enrichment Phase completed successfully.")
            return 0

        elif args.phase == "annotate":
            logger.info("-" * 70)
            logger.info(f"PHASE: Annotation")
            logger.info("-" * 70)

            if not os.path.exists(asr_output_path):
                logger.error(f"ASR results not found at {asr_output_path}. Run ASR phase first.")
                sys.exit(1)

            # Check if enriched data exists (from LLM enrichment phase)
            enriched_output_path = os.path.join("dataset_temp", "enriched_segments.json")
            use_enriched = os.path.exists(enriched_output_path)

            if use_enriched:
                logger.info(f"▶ Loading enriched results from {enriched_output_path}...")
                with open(enriched_output_path, "r", encoding="utf-8") as f:
                    enriched_data = json.load(f)
                # The enriched data should contain word_segments with additional metadata
                if isinstance(enriched_data, dict) and "word_segments" in enriched_data:
                    word_segments = enriched_data["word_segments"]
                    detected_lang = enriched_data.get("detected_lang", "en")
                else:
                    # Assume it's a list of enriched segments
                    word_segments = enriched_data
                    # Try to load detected_lang from original ASR output
                    with open(asr_output_path, "r", encoding="utf-8") as f:
                        asr_data = json.load(f)
                        detected_lang = asr_data.get("detected_lang", "en")
                logger.info("  Using LLM-enriched transcript data")
            else:
                logger.info(f"▶ Loading ASR results from {asr_output_path}...")
                with open(asr_output_path, "r", encoding="utf-8") as f:
                    asr_data = json.load(f)
                    word_segments = asr_data["word_segments"]
                    detected_lang = asr_data["detected_lang"]

            # Load diarization results if they exist
            diarization_path = os.path.join(temp_dir, "diarization.json")
            speaker_segments = []
            if os.path.exists(diarization_path):
                logger.info(f"▶ Loading diarization results from {diarization_path}...")
                with open(diarization_path, "r") as f:
                    speaker_segments = json.load(f)
                word_segments, unique_speakers = _assign_speakers_to_words(word_segments, speaker_segments)
            else:
                unique_speakers = {"UNKNOWN"}
                for word in word_segments:
                    word["speaker"] = "UNKNOWN"

            # ── Optional: source-guided chunking ──────────────────────────────────
            source_state = None
            if args.source:
                entries_for_anchor = _provisional_entries_for_anchor(
                    word_segments, args.chunk_size, max_entries=30
                )
                source_state = _build_source_state(
                    args.source,
                    source_start=args.source_start,
                    source_start_text=args.source_start_text,
                    no_auto_anchor=args.no_auto_anchor,
                    entries_for_anchor=entries_for_anchor,
                )
                avg, n_sampled, low_ct, review_ct = alignment.estimate_alignment_quality(
                    entries_for_anchor, source_state['orig_match'], source_state['cursor'],
                    start_entry_idx=source_state['anchor_entry_idx']
                )
                if n_sampled >= 10:
                    pct_low = low_ct / n_sampled
                    if avg < 0.50 or pct_low > 0.40:
                        sys.exit(
                            f"\n⚠ Source/audio divergence too high to proceed:\n"
                            f"  Sampled {n_sampled} chunks — avg alignment {avg:.0%}, "
                            f"{low_ct} ({pct_low:.0%}) below 60%.\n"
                            f"  Usually means a wrong edition or different translation.\n"
                            f"  Re-run without --source, or pass --keep-unaligned to "
                            f"accept the ASR text for low-confidence chunks."
                        )

            progress.start("Annotate chunks")
            if not args.skip_annotation:
                metadata = annotate_chunks(
                    word_segments,
                    args.model,
                    args.chunk_size,
                    audio_24k_scratch,
                    resume=args.resume,
                    audio_source_path=args.audio,
                    fallback_model_path=args.fallback_model,
                    source_state=source_state,
                    source_threshold=args.source_threshold,
                    keep_unaligned=args.keep_unaligned,
                    min_chunk_duration=args.min_chunk_duration,
                    min_confidence=args.min_confidence,
                    min_snr=args.min_snr,
                    book_title=book_title,
                    character=args.character,
                    narrator_style=args.narrator_style,
                )
                logger.info(f"  Chunks annotated: {len(metadata)}")
            else:
                logger.error("--skip-annotation not yet implemented")
                sys.exit(1)
            progress.complete()

            progress.start("Create output dataset")
            _create_zip_dataset(metadata, args.output, val_split=args.val_split, zip_max_files=args.zip_max_files, unique_speakers=unique_speakers)
            progress.complete()

            logger.info("✓ Annotation Phase completed successfully.")
            return 0

    except KeyboardInterrupt:
        logger.warning("⚠ Process interrupted by user")
        logger.info(f"Partial results preserved in dataset_temp/ - rerun with --resume to continue")
        return 130
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        logger.debug(traceback.format_exc())
        logger.info(f"Partial results preserved in dataset_temp/ - rerun with --resume to continue")
        return 1
    finally:
        # Only clean up the scratch audio file after the final phase (annotation)
        # or if we are not using the phase orchestration.
        # Preserve it during the 'asr' phase so 'annotate' can use it.
        if args.phase != "asr" and os.path.exists(audio_24k_scratch):
            try:
                os.remove(audio_24k_scratch)
                logger.debug(f"Removed scratch audio: {audio_24k_scratch}")
            except Exception as e:
                logger.warning(f"Failed to remove scratch audio: {e}")

        logger.info(f"Log file saved to: {log_file}")

if __name__ == "__main__":
    sys.exit(main())
