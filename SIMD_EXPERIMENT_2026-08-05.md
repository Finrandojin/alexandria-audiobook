# SIMD experiment — 2026-08-05

## Question

Does the SIMD dispatch already present in this machine's NumPy/SciPy build make
Alexandria-relevant CPU work faster?

## Controlled method

- Host: AMD Ryzen 7 7800X3D.
- Native arm: NumPy's detected AVX2/AVX-512 dispatch enabled.
- Baseline arm: advanced NumPy SIMD disabled before importing NumPy.
- Order: baseline, native, native, baseline (ABBA), in separate processes.
- Each process was pinned to one CPU and BLAS/OpenMP thread counts were set to 1.
- Full workloads used 10-second, 5-minute, and 30-minute 24 kHz arrays with 5
  warmups and 31 measured repetitions per process.
- The harness rejected an arm if its actual NumPy feature state was wrong, if a
  result was non-finite, or if output shapes/numerical signatures disagreed.
- Verdicts use a deterministic 10,000-draw bootstrap 95% interval for the ratio
  of baseline/native medians. An interval crossing 1 is `not_proven`.

The complete machine-readable artifact is
`ab_test_runtime/experiments/simd_benchmark.json`.

## Full-run results

| Workload | Size | Native speedup | 95% interval | Verdict |
|---|---:|---:|---:|---|
| PCM health metrics | 10 s | 1.072x | 1.028–1.167 | faster |
| PCM health metrics | 5 min | 1.055x | 1.024–1.087 | faster |
| PCM health metrics | 30 min | 1.051x | 1.026–1.084 | faster |
| Stereo downmix | 10 s | 1.009x | 1.007–1.011 | faster |
| Stereo downmix | 5 min | 1.012x | 1.009–1.014 | faster |
| Stereo downmix | 30 min | 1.012x | 1.010–1.013 | faster |
| Eight-track sum | 10 s | 0.951x | 0.919–0.972 | slower |
| Eight-track sum | 5 min | 0.963x | 0.904–0.979 | slower |
| Eight-track sum | 30 min | 1.031x | 1.006–1.064 | faster |
| Real FFT | 10 s | 1.040x | 1.024–1.048 | faster |
| Real FFT | 5 min | 1.022x | 1.014–1.029 | faster |
| 24 kHz to 16 kHz resampling | 10 s | 0.986x | 0.981–0.995 | slower |
| 24 kHz to 16 kHz resampling | 5 min | 1.012x | 1.005–1.027 | faster |
| 200,000 × 192 cosine scoring | fixed | 0.915x | 0.885–0.952 | slower |

## What the evidence supports

Alexandria already receives SIMD through NumPy automatically. Keeping native
dispatch enabled helps `tts_benchmark.measure_wav`-style PCM metrics by about
5–7% and the mono conversion used by `tts.mix_to_unison` by about 1% on this
host. The exact `np.sum(..., axis=0)` core used by `mix_to_unison` only benefits
at the 30-minute scale in this matrix. The SciPy resampling used by
`experiments/speaker_similarity.py` changes by roughly 1% and changes direction
with input size.

This does **not** support writing custom SIMD code. The measured gains already
come from the installed NumPy/SciPy binaries, are small relative to model
inference, and some workloads regress. It also does not prove gains on another
CPU, NumPy/SciPy build, thread setting, or end-to-end audiobook run. FFT and the
large cosine matrix are controls, not current production hot paths.

## Reproduce

Pilot:

```sh
app/env/bin/python app/experiments/simd_benchmark.py \
  --profile pilot --out /tmp/alexandria_simd_pilot.json
```

Full (refuses to overwrite an existing artifact):

```sh
app/env/bin/python app/experiments/simd_benchmark.py --profile full \
  --out ab_test_runtime/experiments/simd_benchmark.json
```

The four arms run sequentially; no GPU lock is needed because the harness is
CPU-only. Preserve the JSON artifact when comparing machines or dependency
upgrades rather than copying this host's conclusion to them.
