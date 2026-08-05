"""One isolated SIMD benchmark arm; invoked by simd_benchmark.py."""
import argparse
import gc
import json
import math
import os
import statistics
import time

import numpy as np
from scipy.signal import resample_poly


def scalar_audio_metrics(samples):
    peak = total = silence = clipping = 0.0
    for raw in samples:
        value = float(raw) / 32768.0
        absolute = abs(value)
        peak = max(peak, absolute)
        total += value * value
        silence += absolute < 0.001
        clipping += absolute >= 0.999
    count = len(samples)
    return (peak, math.sqrt(total / count), silence / count, clipping / count)


def numpy_audio_metrics(samples):
    values = samples.astype(np.float32) / 32768.0
    return (float(np.max(np.abs(values))),
            float(np.sqrt(np.mean(np.square(values)))),
            float(np.mean(np.abs(values) < 0.001)),
            float(np.mean(np.abs(values) >= 0.999)))


def _signature(value):
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise RuntimeError("benchmark produced a non-finite result")
    return {"shape": list(array.shape), "mean": float(np.mean(array)),
            "rms": float(np.sqrt(np.mean(np.square(array)))),
            "minimum": float(np.min(array)), "maximum": float(np.max(array))}


def _time_case(function, warmups, repeats):
    for _ in range(warmups):
        function()
    gc.disable()
    try:
        timings, result = [], None
        for _ in range(repeats):
            started = time.perf_counter_ns()
            result = function()
            timings.append(time.perf_counter_ns() - started)
    finally:
        gc.enable()
    return result, timings


def run_arm(arm, profile):
    if hasattr(os, "sched_getaffinity"):
        allowed = os.sched_getaffinity(0)
        os.sched_setaffinity(0, {min(allowed)})
    rng = np.random.default_rng(20260805)
    rate = 24000
    seconds = (10, 300, 1800) if profile == "full" else (2, 10)
    repeats = 31 if profile == "full" else 5
    warmups = 5 if profile == "full" else 2
    cases = []
    for duration in seconds:
        count = rate * duration
        pcm = rng.integers(-32768, 32767, count, dtype=np.int16)
        stereo = rng.normal(0, 0.1, (count, 2)).astype(np.float32)
        tracks = rng.normal(0, 0.05, (8, count)).astype(np.float32)
        definitions = {
            f"pcm_metrics_{duration}s": lambda p=pcm: numpy_audio_metrics(p),
            f"stereo_downmix_{duration}s": lambda s=stereo: s.mean(axis=1),
            f"mix_8_tracks_{duration}s": lambda t=tracks: np.sum(t, axis=0) / math.sqrt(8),
        }
        if duration <= 300:
            signal = stereo[:, 0]
            definitions[f"fft_{duration}s"] = lambda s=signal: np.abs(np.fft.rfft(s))
            definitions[f"resample_24k_to_16k_{duration}s"] = \
                lambda s=signal: resample_poly(s, 2, 3)
        for name, function in definitions.items():
            result, timings = _time_case(function, warmups, repeats)
            cases.append({"name": name, "samples": count, "timings_ns": timings,
                          "median_ns": statistics.median(timings),
                          "result_signature": _signature(result)})

    embeddings = rng.normal(size=(200000, 192)).astype(np.float32)
    reference = rng.normal(size=192).astype(np.float32)
    result, timings = _time_case(
        lambda: embeddings @ reference / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(reference) + 1e-9),
        warmups, repeats)
    cases.append({"name": "speaker_cosine_200k_x_192", "samples": len(embeddings),
                  "timings_ns": timings, "median_ns": statistics.median(timings),
                  "result_signature": _signature(result)})

    scalar_pcm = rng.integers(-32768, 32767, rate * (10 if profile == "full" else 2),
                              dtype=np.int16)
    scalar_result, scalar_timings = _time_case(
        lambda: scalar_audio_metrics(scalar_pcm), 1, 5 if profile == "full" else 2)
    numpy_result = numpy_audio_metrics(scalar_pcm)
    if not np.allclose(scalar_result, numpy_result, rtol=1e-5, atol=1e-7):
        raise RuntimeError("scalar and NumPy PCM metrics disagree")
    return {"arm": arm, "profile": profile, "numpy": np.__version__,
            "cpu_features": {key: bool(value) for key, value in
                             np._core._multiarray_umath.__cpu_features__.items()},
            "affinity": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity") else None,
            "cases": cases,
            "scalar_control": {"name": "pcm_metrics_scalar_10s",
                               "timings_ns": scalar_timings,
                               "median_ns": statistics.median(scalar_timings),
                               "result_signature": _signature(scalar_result),
                               "numpy_result_signature": _signature(numpy_result)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("baseline", "native"), required=True)
    parser.add_argument("--profile", choices=("pilot", "full"), default="pilot")
    args = parser.parse_args()
    print(json.dumps(run_arm(args.arm, args.profile), sort_keys=True))


if __name__ == "__main__":
    main()
