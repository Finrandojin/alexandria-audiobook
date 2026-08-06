# Space reclaimed 2026-08-05

Disk was at 95% (40 GB free of 773 GB); now 92% (65 GB free) - 25 GB reclaimed. Everything below was verified
unreferenced before deletion; nothing here is unique data.

## Deleted

| item | size | why it was safe | how to restore |
| --- | ---: | --- | --- |
| `.analysis_env/` | 23 GB | Python venv, untouched since 2026-05-28. No launcher script uses it (`install.js` builds `env` and `preparer_env` only). Its one "reference" was `test_no_machine_paths.py`, which merely EXCLUDES it from scanning. | `uv venv .analysis_env` + reinstall |
| `venv/` | 177 MB | Python 3.14.6 while all live work uses 3.10. Not referenced by any launcher script. | `python -m venv venv` |
| HF `microsoft/VibeVoice-ASR-HF` | 16 GB | Zero references in the repo. | re-downloads on demand |
| HF `Qwen/Qwen3-ASR-1.7B` | 4.4 GB | Zero references. | re-downloads on demand |
| HF `google/bert_uncased_L-12_H-768_A-12` | 421 MB | Zero references; BookNLP's, and `~/booknlp_models` exists separately. | re-downloads on demand |
| HF `google/bert_uncased_L-6_H-768_A-12` | 516 MB | as above | re-downloads on demand |
| `UV_CACHE_DIR` | 17 GB | Package download cache, rebuilt on next install. | rebuilds automatically |

## Deliberately kept

- `app/env` (16 GB) and `preparer_env` (14 GB) — both referenced by
  `install.js`; `preparer_env` used 2026-07-23.
- `Qwen2.5-14B-Instruct-Q6_K.gguf` (12 GB) — referenced by three batch
  scripts and NOT duplicated in the HF cache.
- `ab_test_runtime/corpora` (3.6 GB) — LJSpeech, in active use.
- `.git` (6.4 GB) — history.
- `~/Desktop` (321 GB) and `~/audiobooks` (45 GB) — the user's own files,
  and by far the largest remaining targets. Not touched.
