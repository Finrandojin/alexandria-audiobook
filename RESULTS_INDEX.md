# Results index

Generated 2026-08-05 03:43 from `ab_test_runtime/experiments/` — 212 artifacts, 471 arms.

Regenerate with `python3 collect_results.py`. Machine-readable copy in `results_index.csv`.

`evidence_status` comes from the committed audit snapshots. `supported_structure` validates provenance shape only; `supported_measurement` is the strongest attribution classification. `not_audited` is explicit and must not be treated as support.

`dirty=True` means tracked files were modified when the artifact was written: the numbers are inspectable but the run is not reproducible from its recorded commit.

**`closed-oracle` arms are invalidated.** Their candidate sets were built from the pre-gold labels, so the arm was shown shortlists derived from answers that have since changed. `valid=ok` on those rows means internally consistent, NOT trustworthy — do not read them as results.

`closed_set.json` and `two_by_two.json` predate the environment contract and captured no `context_length` or `parallel`, which cannot be reconstructed. Their rows and summaries were recomputed and are internally consistent, but the runs are not comparable to artifacts that record an environment: inspectable, not citable.


## batch_alignment

| book | model | env | backend | ctx | arm | n | acc | evidence | valid | dirty | elapsed |
|---|---|---|---|---:|---|---:|---:|---|---|---|---:|
| grimgar03 | qwen3-14b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda | 32768 | aligned | 385 | 70.9% | exploratory | ok | False | 1295.5s |
| grimgar03 | qwen3-14b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda | 32768 | fixed | 385 | 71.4% | exploratory | ok | False | 1295.5s |

## batch_contiguity

| book | model | env | backend | ctx | arm | n | acc | evidence | valid | dirty | elapsed |
|---|---|---|---|---:|---|---:|---:|---|---|---|---:|
| grimgar03 | qwen3-14b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda | 32768 | contiguous | 385 | 74.5% | exploratory | ok | False | 5518.1s |
| grimgar03 | qwen3-14b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda | 32768 | scattered | 385 | 57.9% | exploratory | ok | False | 5518.1s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | contiguous | 162 | 53.7% | supported_measurement | ok | False | 19624.4s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | scattered | 162 | 35.2% | supported_measurement | ok | False | 19624.4s |

## batch_size

| book | model | env | backend | ctx | arm | n | acc | evidence | valid | dirty | elapsed |
|---|---|---|---|---:|---|---:|---:|---|---|---|---:|
| grimgar03 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | b1 | 400 | 60.5% | exploratory | ok | False | 3853.3s |
| grimgar03 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | b25 | 400 | 79.2% | exploratory | ok | False | 3853.3s |
| grimgar03 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | b5 | 400 | 69.0% | exploratory | ok | False | 3853.3s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | b1 | 400 | 54.0% | historical_only | ok | False | 2298.0s |
| grimgar03 | qwen3-14b | cloud-a6000-LM | LM Studio loopback | 16384 | b1 | 400 | 52.8% | exploratory | ok | False | 2647.3s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | b10 | 400 | 54.0% | historical_only | ok | False | 2298.0s |
| grimgar03 | qwen3-14b | cloud-a6000-LM | LM Studio loopback | 16384 | b10 | 400 | 52.0% | exploratory | ok | False | 2647.3s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | b100 | 400 | 62.3% | historical_only | ok | True | 2990.1s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | b25 | 400 | 59.8% | historical_only | ok | True | 2990.1s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | b25 | 400 | 55.5% | historical_only | ok | False | 2298.0s |
| grimgar03 | qwen3-14b | cloud-a6000-LM | LM Studio loopback | 16384 | b25 | 400 | 55.8% | exploratory | ok | False | 2647.3s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | b5 | 400 | 54.0% | historical_only | ok | False | 2298.0s |
| grimgar03 | qwen3-14b | cloud-a6000-LM | LM Studio loopback | 16384 | b5 | 400 | 52.0% | exploratory | ok | False | 2647.3s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | b50 | 400 | 63.7% | historical_only | ok | True | 2990.1s |
| index18 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | b10 | 99 | 68.7% | supported_measurement | ok | False | 1851.3s |
| index18 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | b25 | 99 | 66.7% | supported_measurement | ok | False | 1851.3s |
| index18 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | b5 | 99 | 54.5% | supported_measurement | ok | False | 1851.3s |
| index18 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | b50 | 99 | 54.5% | supported_measurement | ok | False | 1851.3s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | b1 | 139 | 30.2% | provisional | ok | False | 952.5s |
| mushoku16 | qwen3-14b | cloud-a6000-LM | LM Studio loopback | 16384 | b1 | 139 | 28.8% | exploratory | ok | False | 1396.6s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | b10 | 139 | 46.0% | provisional | ok | False | 952.5s |
| mushoku16 | qwen3-14b | cloud-a6000-LM | LM Studio loopback | 16384 | b10 | 139 | 38.1% | exploratory | ok | False | 1396.6s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | b100 | 139 | 49.6% | provisional | ok | True | 3425.6s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | b25 | 139 | 51.8% | provisional | ok | True | 3425.6s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | b25 | 139 | 49.6% | provisional | ok | False | 952.5s |
| mushoku16 | qwen3-14b | cloud-a6000-LM | LM Studio loopback | 16384 | b25 | 139 | 46.8% | exploratory | ok | False | 1396.6s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | b5 | 139 | 40.3% | provisional | ok | False | 952.5s |
| mushoku16 | qwen3-14b | cloud-a6000-LM | LM Studio loopback | 16384 | b5 | 139 | 37.4% | exploratory | ok | False | 1396.6s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | b50 | 139 | 48.9% | provisional | ok | True | 3425.6s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | b10 | 162 | 36.4% | supported_measurement | ok | False | 2716.3s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | b100 | 162 | 46.3% | provisional | ok | True | 8526.1s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | b100 | 162 | 46.3% | provisional | ok | True | 8230.1s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | b25 | 162 | 40.7% | provisional | ok | True | 8526.1s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | b25 | 162 | 40.7% | provisional | ok | True | 8230.1s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | b25 | 162 | 39.5% | supported_measurement | ok | False | 2716.3s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | b5 | 162 | 26.5% | supported_measurement | ok | False | 2716.3s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | b50 | 162 | 47.5% | provisional | ok | True | 8526.1s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | b50 | 162 | 47.5% | provisional | ok | True | 8230.1s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | b50 | 162 | 46.3% | supported_measurement | ok | False | 2716.3s |

## because_production

| book | model | env | backend | ctx | arm | n | acc | evidence | valid | dirty | elapsed |
|---|---|---|---|---:|---|---:|---:|---|---|---|---:|
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | baseline | 400 | 55.5% | historical_only | ok | False | 4766.2s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | thinking | 400 | 63.7% | historical_only | ok | False | 4766.2s |
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | baseline | 139 | 49.6% | provisional | ok | False | 2668.6s |
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | baseline | 139 | 49.6% | provisional | ok | True | 4564.3s |
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | because | 139 | 42.4% | provisional | ok | True | 4564.3s |
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | scaffold_thinking | 139 | 43.2% | provisional | ok | True | 4564.3s |
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | thinking | 139 | 52.5% | provisional | ok | False | 2668.6s |

## booknlp_baseline

| book | model | env | backend | ctx | arm | n | acc | evidence | valid | dirty | elapsed |
|---|---|---|---|---:|---|---:|---:|---|---|---|---:|
| ? | booknlp-big | local-lmstudio | lmstudio |  | booknlp | 1226 | 54.2% | exploratory | None | None | s |

## candidate_id

| book | model | env | backend | ctx | arm | n | acc | evidence | valid | dirty | elapsed |
|---|---|---|---|---:|---|---:|---:|---|---|---|---:|
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | id | 147 | 35.4% | historical_only | ok | False | 68.3s |
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | name | 147 | 49.0% | historical_only | ok | False | 68.3s |

## cascade

| book | model | env | backend | ctx | arm | n | acc | evidence | valid | dirty | elapsed |
|---|---|---|---|---:|---|---:|---:|---|---|---|---:|
| grimgar03 | qwen3-14b + llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | cascade | 396 | 73.7% | exploratory | ok | False | 0.1s |
| grimgar03 | qwen3-14b + llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | cascade | 400 | 77.8% | exploratory | ok | False | 0.1s |
| grimgar03 | qwen3-14b + llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda | 16384 | cascade | 396 | 84.6% | exploratory | ok | False | 0.2s |
| grimgar03 | qwen3-14b + llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | cheap-w1 | 396 | 57.3% | exploratory | ok | False | 0.1s |
| grimgar03 | qwen3-14b + llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | cheap-w1 | 400 | 55.8% | exploratory | ok | False | 0.1s |
| grimgar03 | qwen3-14b + llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda | 16384 | cheap-w1 | 396 | 82.3% | exploratory | ok | False | 0.2s |
| index18 | gemma-3-27b | cloud-a6000-LM | LM Studio loopback | 16384 | cascade | 99 | 65.7% | exploratory | ok | False | 0.1s |
| index18 | gemma-3-27b | cloud-a6000-LM | LM Studio loopback | 16384 | cheap-w1 | 99 | 62.6% | exploratory | ok | False | 0.1s |
| index18 | qwen3-14b + llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | cascade | 99 | 69.7% | exploratory | ok | False | 0.1s |
| index18 | qwen3-14b + llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | cascade | 99 | 73.7% | exploratory | ok | False | 0.1s |
| index18 | qwen3-14b + llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda | 16384 | cascade | 99 | 63.6% | exploratory | ok | False | 0.1s |
| index18 | qwen3-14b + llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | cheap-w1 | 99 | 62.6% | exploratory | ok | False | 0.1s |
| index18 | qwen3-14b + llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | cheap-w1 | 99 | 62.6% | exploratory | ok | False | 0.1s |
| index18 | qwen3-14b + llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda | 16384 | cheap-w1 | 99 | 57.6% | exploratory | ok | False | 0.1s |
| mushoku16 | qwen3-14b + llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | cascade | 136 | 64.0% | exploratory | ok | False | 0.1s |
| mushoku16 | qwen3-14b + llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | cascade | 139 | 64.0% | exploratory | ok | False | 0.1s |
| mushoku16 | qwen3-14b + llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda | 16384 | cascade | 136 | 61.8% | exploratory | ok | False | 0.1s |
| mushoku16 | qwen3-14b + llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | cheap-w1 | 136 | 50.0% | exploratory | ok | False | 0.1s |
| mushoku16 | qwen3-14b + llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | cheap-w1 | 139 | 47.5% | exploratory | ok | False | 0.1s |
| mushoku16 | qwen3-14b + llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda | 16384 | cheap-w1 | 136 | 55.1% | exploratory | ok | False | 0.1s |
| mushoku16 | qwen3-32b | cloud-a6000-LM | LM Studio loopback | 16384 | cascade | 139 | 45.3% | exploratory | ok | False | 0.1s |
| mushoku16 | qwen3-32b | cloud-a6000-LM | LM Studio loopback | 16384 | cheap-w1 | 139 | 47.5% | exploratory | ok | False | 0.1s |
| owarimonogatari3 | gemma-3-27b | cloud-a6000-LM | LM Studio loopback | 16384 | cascade | 162 | 46.3% | exploratory | ok | False | 0.1s |
| owarimonogatari3 | gemma-3-27b | cloud-a6000-LM | LM Studio loopback | 16384 | cheap-w1 | 162 | 42.6% | exploratory | ok | False | 0.1s |
| owarimonogatari3 | qwen3-14b + llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | cascade | 162 | 54.9% | exploratory | ok | False | 0.1s |
| owarimonogatari3 | qwen3-14b + llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | cascade | 162 | 56.2% | exploratory | ok | False | 0.1s |
| owarimonogatari3 | qwen3-14b + llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda | 16384 | cascade | 162 | 59.9% | exploratory | ok | False | 0.1s |
| owarimonogatari3 | qwen3-14b + llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | cheap-w1 | 162 | 40.1% | exploratory | ok | False | 0.1s |
| owarimonogatari3 | qwen3-14b + llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | cheap-w1 | 162 | 42.0% | exploratory | ok | False | 0.1s |
| owarimonogatari3 | qwen3-14b + llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda | 16384 | cheap-w1 | 162 | 54.3% | exploratory | ok | False | 0.1s |

## closed_set

| book | model | env | backend | ctx | arm | n | acc | evidence | valid | dirty | elapsed |
|---|---|---|---|---:|---|---:|---:|---|---|---|---:|
| grimgar03 | gemma-4-e4b-uncensored-hau | local-lmstudio | lmstudio | 32768 | closed-6 | 400 | 59.8% | historical_only | ok | False | 198.1s |
| grimgar03 | gemma-4-e4b-uncensored-hau | local-lmstudio | lmstudio | 32768 | closed-oracle | 400 | 65.5% | historical_only | ok | False | 198.1s |
| grimgar03 | gemma-4-e4b-uncensored-hau | local-lmstudio | lmstudio | 32768 | open | 400 | 57.2% | historical_only | ok | False | 198.1s |
| grimgar03 | gemma-3-27b | cloud-a6000-lmstudio | lmstudio | 16384 | closed-6 | 400 | 62.3% | historical_only | ok | False | 1206.5s |
| grimgar03 | gemma-3-27b | cloud-a6000-lmstudio | lmstudio | 16384 | closed-oracle | 400 | 70.5% | historical_only | ok | False | 1206.5s |
| grimgar03 | gemma-3-27b | cloud-a6000-lmstudio | lmstudio | 16384 | open | 400 | 61.5% | historical_only | ok | False | 1206.5s |
| grimgar03 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda b10 | 8192 | closed-6 | 400 | 76.5% | exploratory | ok | False | 1489.6s |
| grimgar03 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda b10 | 8192 | closed-oracle | 400 | 83.0% | exploratory | ok | False | 1489.6s |
| grimgar03 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda b10 | 8192 | open | 400 | 75.8% | exploratory | ok | False | 1489.6s |
| grimgar03 | ministral-3-14b-instruct-2 | local-lmstudio | lmstudio | 16384 | closed-6 | 400 | 52.5% | historical_only | ok | False | 285.2s |
| grimgar03 | ministral-3-14b-instruct-2 | local-lmstudio | lmstudio | 16384 | closed-oracle | 400 | 58.2% | historical_only | ok | False | 285.2s |
| grimgar03 | ministral-3-14b-instruct-2 | local-lmstudio | lmstudio | 16384 | open | 400 | 51.5% | historical_only | ok | False | 285.2s |
| grimgar03 | magistral-small | local-llamacpp-hip | llama.cpp-hip | 16384 | closed-6 | 396 | 64.4% | supported_measurement | ok | False | 470.2s |
| grimgar03 | magistral-small | cloud-a6000-lmstudio | lmstudio | 16384 | closed-6 | 400 | 58.0% | historical_only | ok | False | 854.3s |
| grimgar03 | magistral-small | local-llamacpp-hip | llama.cpp-hip | 16384 | closed-oracle | 396 | 77.3% | supported_measurement | ok | False | 470.2s |
| grimgar03 | magistral-small | cloud-a6000-lmstudio | lmstudio | 16384 | closed-oracle | 400 | 70.8% | historical_only | ok | False | 854.3s |
| grimgar03 | magistral-small | local-llamacpp-hip | llama.cpp-hip | 16384 | open | 396 | 64.1% | supported_measurement | ok | False | 470.2s |
| grimgar03 | magistral-small | cloud-a6000-lmstudio | lmstudio | 16384 | open | 400 | 61.3% | historical_only | ok | False | 854.3s |
| grimgar03 | qwen3-14b | cloud-a6000-lmstudio | lmstudio | 98304 | closed-6 | 400 | 60.8% | historical_only | ok | False | 915.3s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | closed-6 | 400 | 60.5% | historical_only | ok | False | 291.8s |
| grimgar03 | qwen3-14b | cloud-a6000-lmstudio | lmstudio | 98304 | closed-6 | 400 | 60.8% | historical_only | ok | False | 915.3s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | closed-6 | 396 | 64.4% | supported_measurement | ok | False | 261.7s |
| grimgar03 | qwen3-14b | cloud-a6000-lmstudio | lmstudio | 98304 | closed-oracle | 400 | 72.8% | historical_only | ok | False | 915.3s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | closed-oracle | 400 | 72.5% | historical_only | ok | False | 291.8s |
| grimgar03 | qwen3-14b | cloud-a6000-lmstudio | lmstudio | 98304 | closed-oracle | 400 | 72.8% | historical_only | ok | False | 915.3s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | closed-oracle | 396 | 73.7% | supported_measurement | ok | False | 261.7s |
| grimgar03 | qwen3-14b | cloud-a6000-lmstudio | lmstudio | 98304 | open | 400 | 61.3% | historical_only | ok | False | 915.3s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | open | 400 | 60.8% | historical_only | ok | False | 291.8s |
| grimgar03 | qwen3-14b | cloud-a6000-lmstudio | lmstudio | 98304 | open | 400 | 61.3% | historical_only | ok | False | 915.3s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | open | 396 | 62.6% | supported_measurement | ok | False | 261.7s |
| grimgar03 | qwen3-32b | cloud-a6000-lmstudio | lmstudio | 16384 | closed-6 | 400 | 61.5% | historical_only | ok | False | 1430.8s |
| grimgar03 | qwen3-32b | cloud-a6000-lmstudio | lmstudio | 16384 | closed-oracle | 400 | 76.2% | historical_only | ok | False | 1430.8s |
| grimgar03 | qwen3-32b | cloud-a6000-lmstudio | lmstudio | 16384 | open | 400 | 61.8% | historical_only | ok | False | 1430.8s |
| index18 | magistral-small | local-llamacpp-hip | llama.cpp-hip | 16384 | closed-6 | 99 | 70.7% | supported_measurement | ok | False | 97.5s |
| index18 | magistral-small | local-llamacpp-hip | llama.cpp-hip | 16384 | closed-oracle | 99 | 78.8% | supported_measurement | ok | False | 97.5s |
| index18 | magistral-small | local-llamacpp-hip | llama.cpp-hip | 16384 | open | 99 | 65.7% | supported_measurement | ok | False | 97.5s |
| index18 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | closed-6 | 99 | 60.6% | supported_measurement | ok | False | 62.6s |
| index18 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | closed-oracle | 99 | 72.7% | supported_measurement | ok | False | 62.6s |
| index18 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | open | 99 | 66.7% | supported_measurement | ok | False | 62.6s |
| mushoku16 | gemma-4-e4b-uncensored-hau | local-lmstudio | lmstudio | 32768 | closed-6 | 147 | 38.8% | provisional | ok | False | 81.9s |
| mushoku16 | gemma-4-e4b-uncensored-hau | local-lmstudio | lmstudio | 32768 | closed-oracle | 147 | 49.7% | provisional | ok | False | 81.9s |
| mushoku16 | gemma-4-e4b-uncensored-hau | local-lmstudio | lmstudio | 32768 | open | 147 | 39.5% | provisional | ok | False | 81.9s |
| mushoku16 | gemma-3-27b | cloud-a6000-lmstudio | lmstudio | 16384 | closed-6 | 139 | 44.6% | historical_only | ok | False | 421.3s |
| mushoku16 | gemma-3-27b | cloud-a6000-lmstudio | lmstudio | 16384 | closed-oracle | 139 | 59.0% | historical_only | ok | False | 421.3s |
| mushoku16 | gemma-3-27b | cloud-a6000-lmstudio | lmstudio | 16384 | open | 139 | 55.4% | historical_only | ok | False | 421.3s |
| mushoku16 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda b10 | 8192 | closed-6 | 139 | 60.4% | exploratory | ok | False | 445.8s |
| mushoku16 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda b10 | 8192 | closed-oracle | 139 | 74.8% | exploratory | ok | False | 445.8s |
| mushoku16 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda b10 | 8192 | open | 139 | 59.7% | exploratory | ok | False | 445.8s |
| mushoku16 | phi-4 | local-lmstudio | lmstudio | 16384 | closed-6 | 147 | 32.7% | provisional | ok | False | 112.6s |
| mushoku16 | phi-4 | local-lmstudio | lmstudio | 16384 | closed-oracle | 147 | 59.2% | provisional | ok | False | 112.6s |
| mushoku16 | phi-4 | local-lmstudio | lmstudio | 16384 | open | 147 | 45.6% | provisional | ok | False | 112.6s |
| mushoku16 | ministral-3-14b-instruct-2 | local-lmstudio | lmstudio | 16384 | closed-6 | 147 | 41.5% | historical_only | ok | False | 64.0s |
| mushoku16 | ministral-3-14b-instruct-2 | local-lmstudio | lmstudio | 16384 | closed-oracle | 147 | 61.2% | historical_only | ok | False | 64.0s |
| mushoku16 | ministral-3-14b-instruct-2 | local-lmstudio | lmstudio | 16384 | open | 147 | 47.6% | historical_only | ok | False | 64.0s |
| mushoku16 | ministral-3-14b-instruct-2 | local-lmstudio | lmstudio | 16384 | closed-6 | 147 | 40.8% | historical_only | ok | False | 84.9s |
| mushoku16 | ministral-3-14b-instruct-2 | local-lmstudio | lmstudio | 16384 | closed-oracle | 147 | 59.2% | historical_only | ok | False | 84.9s |
| mushoku16 | ministral-3-14b-instruct-2 | local-lmstudio | lmstudio | 16384 | open | 147 | 46.9% | historical_only | ok | False | 84.9s |
| mushoku16 | magistral-small | local-llamacpp-hip | llama.cpp-hip b101 | 8192 | closed-6 | 139 | 45.3% | historical_only | ok | False | 100.5s |
| mushoku16 | magistral-small | cloud-a6000-lmstudio | lmstudio | 16384 | closed-6 | 139 | 45.3% | historical_only | ok | False | 254.3s |
| mushoku16 | magistral-small | local-llamacpp-hip | llama.cpp-hip b101 | 8192 | closed-oracle | 139 | 57.6% | historical_only | ok | False | 100.5s |
| mushoku16 | magistral-small | cloud-a6000-lmstudio | lmstudio | 16384 | closed-oracle | 139 | 56.8% | historical_only | ok | False | 254.3s |
| mushoku16 | magistral-small | local-llamacpp-hip | llama.cpp-hip b101 | 8192 | open | 139 | 53.2% | historical_only | ok | False | 100.5s |
| mushoku16 | magistral-small | cloud-a6000-lmstudio | lmstudio | 16384 | open | 139 | 52.5% | historical_only | ok | False | 254.3s |
| mushoku16 | qwen3-14b | cloud-a6000-lmstudio | lmstudio | 98304 | closed-6 | 139 | 38.8% | historical_only | ok | False | 317.6s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | closed-6 | 136 | 40.4% | supported_measurement | ok | False | 65.2s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip b101 | 16384 | closed-6 | 139 | 38.8% | historical_only | ok | False | 81.7s |
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | closed-6 | 139 | 39.6% | historical_only | ok | False | 90.6s |
| mushoku16 | qwen3-14b | local-llamacpp-vulkan | llama.cpp-vulkan b | 16384 | closed-6 | 139 | 39.6% | historical_only | ok | False | 87.6s |
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | closed-6 | 147 | 36.7% | historical_only | ok | False | 92.7s |
| mushoku16 | qwen3-14b | cloud-a6000-lmstudio | lmstudio | 98304 | closed-oracle | 139 | 66.9% | historical_only | ok | False | 317.6s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | closed-oracle | 136 | 66.2% | supported_measurement | ok | False | 65.2s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip b101 | 16384 | closed-oracle | 139 | 66.9% | historical_only | ok | False | 81.7s |
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | closed-oracle | 139 | 66.2% | historical_only | ok | False | 90.6s |
| mushoku16 | qwen3-14b | local-llamacpp-vulkan | llama.cpp-vulkan b | 16384 | closed-oracle | 139 | 66.9% | historical_only | ok | False | 87.6s |
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | closed-oracle | 147 | 66.0% | historical_only | ok | False | 92.7s |
| mushoku16 | qwen3-14b | cloud-a6000-lmstudio | lmstudio | 98304 | open | 139 | 50.4% | historical_only | ok | False | 317.6s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | open | 136 | 52.9% | supported_measurement | ok | False | 65.2s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip b101 | 16384 | open | 139 | 48.2% | historical_only | ok | False | 81.7s |
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | open | 139 | 49.6% | historical_only | ok | False | 90.6s |
| mushoku16 | qwen3-14b | local-llamacpp-vulkan | llama.cpp-vulkan b | 16384 | open | 139 | 49.6% | historical_only | ok | False | 87.6s |
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | open | 147 | 48.3% | historical_only | ok | False | 92.7s |
| mushoku16 | qwen3-32b | cloud-a6000-lmstudio | lmstudio | 16384 | closed-6 | 139 | 46.8% | historical_only | ok | False | 496.9s |
| mushoku16 | qwen3-32b | cloud-a6000-lmstudio | lmstudio | 16384 | closed-oracle | 139 | 64.0% | historical_only | ok | False | 496.9s |
| mushoku16 | qwen3-32b | cloud-a6000-lmstudio | lmstudio | 16384 | open | 139 | 52.5% | historical_only | ok | False | 496.9s |
| mushoku16 | qwen3.5-9b-uncensored-hauh | local-lmstudio | lmstudio |  | closed-6 | 147 | 34.7% | exploratory | ['no LM Studio load state recorded', 'en | True | 138.3s |
| mushoku16 | qwen3.5-9b-uncensored-hauh | local-lmstudio | lmstudio | 32768 | closed-6 | 147 | 34.7% | historical_only | ok | False | 150.0s |
| mushoku16 | qwen3.5-9b-uncensored-hauh | local-lmstudio | lmstudio |  | closed-oracle | 147 | 49.0% | exploratory | ['no LM Studio load state recorded', 'en | True | 138.3s |
| mushoku16 | qwen3.5-9b-uncensored-hauh | local-lmstudio | lmstudio | 32768 | closed-oracle | 147 | 49.0% | historical_only | ok | False | 150.0s |
| mushoku16 | qwen3.5-9b-uncensored-hauh | local-lmstudio | lmstudio |  | open | 147 | 35.4% | exploratory | ['no LM Studio load state recorded', 'en | True | 138.3s |
| mushoku16 | qwen3.5-9b-uncensored-hauh | local-lmstudio | lmstudio | 32768 | open | 147 | 35.4% | historical_only | ok | False | 150.0s |
| owarimonogatari3 | magistral-small | local-llamacpp-hip | llama.cpp-hip | 16384 | closed-6 | 162 | 41.4% | supported_measurement | ok | False | 165.9s |
| owarimonogatari3 | magistral-small | local-llamacpp-hip | llama.cpp-hip | 16384 | closed-oracle | 162 | 57.4% | supported_measurement | ok | False | 165.9s |
| owarimonogatari3 | magistral-small | local-llamacpp-hip | llama.cpp-hip | 16384 | open | 162 | 46.9% | supported_measurement | ok | False | 165.9s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | closed-6 | 162 | 42.6% | supported_measurement | ok | False | 110.4s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | closed-oracle | 162 | 51.9% | supported_measurement | ok | False | 110.4s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | open | 162 | 47.5% | supported_measurement | ok | False | 110.4s |

## committed_history

| book | model | env | backend | ctx | arm | n | acc | evidence | valid | dirty | elapsed |
|---|---|---|---|---:|---|---:|---:|---|---|---|---:|
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip b101 | 16384 | none | 400 | 63.5% | historical_only | ok | False | 337.5s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip b101 | 16384 | oracle | 400 | 63.5% | historical_only | ok | False | 337.5s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip b101 | 16384 | predicted | 400 | 62.3% | historical_only | ok | False | 337.5s |
| index18 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | none | 99 | 63.6% | supported_measurement | ok | False | 80.8s |
| index18 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | oracle | 99 | 60.6% | supported_measurement | ok | False | 80.8s |
| index18 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | predicted | 99 | 63.6% | supported_measurement | ok | False | 80.8s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | none | 136 | 50.7% | supported_measurement | ok | False | 77.8s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | oracle | 136 | 54.4% | supported_measurement | ok | False | 77.8s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | predicted | 136 | 47.8% | supported_measurement | ok | False | 77.8s |
| owarimonogatari3 | qwen3-14b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda | 32768 | gated | 162 | 48.8% | exploratory | ok | False | 417.6s |
| owarimonogatari3 | qwen3-14b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda | 32768 | gated | 162 | 48.8% | exploratory | ok | False | 579.9s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | none | 162 | 50.0% | supported_measurement | ok | False | 116.4s |
| owarimonogatari3 | qwen3-14b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda | 32768 | none | 162 | 50.0% | exploratory | ok | False | 417.6s |
| owarimonogatari3 | qwen3-14b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda | 32768 | none | 162 | 50.0% | exploratory | ok | False | 579.9s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | oracle | 162 | 59.3% | supported_measurement | ok | False | 116.4s |
| owarimonogatari3 | qwen3-14b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda | 32768 | oracle | 162 | 59.3% | exploratory | ok | False | 417.6s |
| owarimonogatari3 | qwen3-14b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda | 32768 | oracle | 162 | 59.3% | exploratory | ok | False | 579.9s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | predicted | 162 | 46.9% | supported_measurement | ok | False | 116.4s |
| owarimonogatari3 | qwen3-14b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda | 32768 | predicted | 162 | 48.1% | exploratory | ok | False | 417.6s |
| owarimonogatari3 | qwen3-14b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda | 32768 | predicted | 162 | 48.1% | exploratory | ok | False | 579.9s |

## context_width

| book | model | env | backend | ctx | arm | n | acc | evidence | valid | dirty | elapsed |
|---|---|---|---|---:|---|---:|---:|---|---|---|---:|
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip b101 | 16384 | w1 | 400 | 55.8% | historical_only | ok | False | 1003.4s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip b101 | 16384 | w15 | 400 | 61.5% | historical_only | ok | False | 1003.4s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip b101 | 16384 | w4 | 400 | 62.0% | historical_only | ok | False | 1003.4s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip b101 | 16384 | w40 | 400 | 60.5% | historical_only | ok | False | 1003.4s |
| index18 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | w1 | 99 | 48.5% | supported_measurement | ok | False | 146.2s |
| index18 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | w15 | 99 | 63.6% | supported_measurement | ok | False | 146.2s |
| index18 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | w4 | 99 | 65.7% | supported_measurement | ok | False | 146.2s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | w1 | 136 | 38.2% | supported_measurement | ok | False | 140.1s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | w15 | 136 | 45.6% | supported_measurement | ok | False | 140.1s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | w4 | 136 | 55.9% | supported_measurement | ok | False | 140.1s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | w1 | 162 | 41.4% | supported_measurement | ok | False | 238.7s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | w15 | 162 | 46.3% | supported_measurement | ok | False | 238.7s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | w4 | 162 | 46.9% | supported_measurement | ok | False | 238.7s |

## context_width_production

| book | model | env | backend | ctx | arm | n | acc | evidence | valid | dirty | elapsed |
|---|---|---|---|---:|---|---:|---:|---|---|---|---:|
| grimgar03 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | w1 | 400 | 79.2% | exploratory | ok | False | 4704.6s |
| grimgar03 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | w4 | 400 | 76.8% | exploratory | ok | False | 4704.6s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | w1 | 400 | 59.8% | historical_only | ok | False | 1914.8s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | w1 | 400 | 58.0% | historical_only | ok | False | 1897.4s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | w1 | 400 | 58.0% | historical_only | ok | False | 1882.6s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | w1 | 400 | 57.0% | historical_only | ok | False | 1426.2s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | w4 | 400 | 69.8% | historical_only | ok | False | 1914.8s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | w4 | 400 | 69.8% | historical_only | ok | False | 1897.4s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | w4 | 400 | 69.8% | historical_only | ok | False | 1882.6s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | w4 | 400 | 67.5% | historical_only | ok | False | 1426.2s |
| index18 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | w1 | 99 | 70.7% | exploratory | ok | False | 4411.0s |
| index18 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | w4 | 99 | 72.7% | exploratory | ok | False | 4411.0s |
| index18 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | w1 | 99 | 65.7% | supported_measurement | ok | False | 1317.4s |
| index18 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | w4 | 99 | 68.7% | supported_measurement | ok | False | 1317.4s |
| mushoku16 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | w1 | 139 | 66.9% | exploratory | ok | False | 2904.1s |
| mushoku16 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | w4 | 139 | 59.0% | exploratory | ok | False | 2904.1s |
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | w1 | 139 | 49.6% | provisional | ok | False | 826.8s |
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | w4 | 139 | 44.6% | provisional | ok | False | 826.8s |
| owarimonogatari3 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | w1 | 162 | 59.3% | exploratory | ok | False | 5596.8s |
| owarimonogatari3 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | w4 | 162 | 62.3% | exploratory | ok | False | 5596.8s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | w1 | 162 | 40.1% | supported_measurement | ok | False | 2955.3s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | w4 | 162 | 40.1% | supported_measurement | ok | False | 2955.3s |

## distill_eval

| book | model | env | backend | ctx | arm | n | acc | evidence | valid | dirty | elapsed |
|---|---|---|---|---:|---|---:|---:|---|---|---|---:|
| grimgar03 | Qwen3-14B | local-lmstudio | lmstudio | 32768 | base | 547 | 60.3% | exploratory | ok | False | 23636.2s |
| grimgar03 | Qwen3-14B | local-lmstudio | lmstudio | 32768 | base | 772 | 60.0% | exploratory | ok | False | 50138.1s |
| grimgar03 | Qwen3-14B | local-lmstudio | lmstudio | 32768 | tuned | 547 | 61.6% | exploratory | ok | False | 23636.2s |
| grimgar03 | Qwen3-14B | local-lmstudio | lmstudio | 32768 | tuned | 772 | 71.6% | exploratory | ok | False | 50138.1s |

## grammar_constraint

| book | model | env | backend | ctx | arm | n | acc | evidence | valid | dirty | elapsed |
|---|---|---|---|---:|---|---:|---:|---|---|---|---:|
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | open-free | 396 | 62.6% | supported_measurement | ok | False | 360.2s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | open-grammar | 396 | 62.6% | supported_measurement | ok | False | 360.2s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | oracle-free | 396 | 74.2% | supported_measurement | ok | False | 360.2s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | oracle-grammar | 396 | 76.8% | supported_measurement | ok | False | 360.2s |
| index18 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | open-free | 99 | 66.7% | supported_measurement | ok | False | 71.3s |
| index18 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | open-grammar | 99 | 65.7% | supported_measurement | ok | False | 71.3s |
| index18 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | oracle-free | 99 | 73.7% | supported_measurement | ok | False | 71.3s |
| index18 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | oracle-grammar | 99 | 74.7% | supported_measurement | ok | False | 71.3s |
| mushoku16 | magistral-small | local-llamacpp-hip | llama.cpp-hip b101 | 8192 | open-free | 139 | 53.2% | provisional | ok | False | 114.0s |
| mushoku16 | magistral-small | local-llamacpp-hip | llama.cpp-hip b101 | 8192 | open-grammar | 139 | 51.8% | provisional | ok | False | 114.0s |
| mushoku16 | magistral-small | local-llamacpp-hip | llama.cpp-hip b101 | 8192 | oracle-free | 139 | 58.3% | provisional | ok | False | 114.0s |
| mushoku16 | magistral-small | local-llamacpp-hip | llama.cpp-hip b101 | 8192 | oracle-grammar | 139 | 66.2% | provisional | ok | False | 114.0s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | open-free | 162 | 47.5% | supported_measurement | ok | False | 135.1s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | open-grammar | 162 | 46.3% | supported_measurement | ok | False | 135.1s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | oracle-free | 162 | 51.9% | supported_measurement | ok | False | 135.1s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | oracle-grammar | 162 | 51.2% | supported_measurement | ok | False | 135.1s |

## joint_scene

| book | model | env | backend | ctx | arm | n | acc | evidence | valid | dirty | elapsed |
|---|---|---|---|---:|---|---:|---:|---|---|---|---:|
| grimgar03 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | independent | 400 | 71.5% | exploratory | ok | False | 2149.6s |
| grimgar03 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | joint-chrono | 400 | 63.2% | exploratory | ok | False | 2149.6s |
| grimgar03 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | joint-shuffled | 400 | 47.8% | exploratory | ok | False | 2149.6s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | independent | 400 | 60.5% | historical_only | ok | False | 498.9s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | joint-chrono | 400 | 57.0% | historical_only | ok | False | 498.9s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | joint-shuffled | 400 | 50.2% | historical_only | ok | False | 498.9s |
| index18 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | independent | 99 | 76.8% | exploratory | ok | False | 957.2s |
| index18 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | joint-chrono | 99 | 66.7% | exploratory | ok | False | 957.2s |
| index18 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | joint-shuffled | 99 | 58.6% | exploratory | ok | False | 957.2s |
| index18 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | independent | 99 | 61.6% | supported_measurement | ok | False | 199.5s |
| index18 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | joint-chrono | 99 | 54.5% | supported_measurement | ok | False | 199.5s |
| index18 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | joint-shuffled | 99 | 54.5% | supported_measurement | ok | False | 199.5s |
| mushoku16 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | independent | 139 | 59.7% | exploratory | ok | False | 996.8s |
| mushoku16 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | joint-chrono | 139 | 58.3% | exploratory | ok | False | 996.8s |
| mushoku16 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | joint-shuffled | 139 | 43.9% | exploratory | ok | False | 996.8s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | independent | 139 | 51.8% | historical_only | ok | False | 213.6s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | joint-chrono | 139 | 47.5% | historical_only | ok | False | 213.6s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | joint-shuffled | 139 | 48.9% | historical_only | ok | False | 213.6s |
| owarimonogatari3 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | independent | 162 | 59.3% | exploratory | ok | False | 1996.3s |
| owarimonogatari3 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | joint-chrono | 162 | 69.8% | exploratory | ok | False | 1996.3s |
| owarimonogatari3 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | joint-shuffled | 162 | 43.8% | exploratory | ok | False | 1996.3s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | independent | 162 | 48.8% | supported_measurement | ok | False | 538.3s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | joint-chrono | 162 | 45.7% | supported_measurement | ok | False | 538.3s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | joint-shuffled | 162 | 45.7% | supported_measurement | ok | False | 538.3s |

## lora_serving_eval

| book | model | env | backend | ctx | arm | n | acc | evidence | valid | dirty | elapsed |
|---|---|---|---|---:|---|---:|---:|---|---|---|---:|
| grimgar03 | qwen3-14b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda | 32768 | base | 772 | 58.7% | exploratory | ok | False | 6206.6s |
| grimgar03 | qwen3-14b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda | 32768 | base | 385 | 64.4% | exploratory | ok | False | 2253.6s |
| grimgar03 | qwen3-14b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda | 32768 | base | 385 | 64.4% | exploratory | ok | False | 1984.6s |
| grimgar03 | qwen3-14b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda | 32768 | base | 385 | 64.4% | exploratory | ok | False | 1929.7s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | base | 547 | 72.0% | historical_only | ok | True | 28194.5s |
| grimgar03 | qwen3-14b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda | 32768 | lora | 772 | 68.5% | exploratory | ok | False | 6206.6s |
| grimgar03 | qwen3-14b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda | 32768 | lora | 385 | 68.1% | exploratory | ok | False | 2253.6s |
| grimgar03 | qwen3-14b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda | 32768 | lora | 385 | 72.7% | exploratory | ok | False | 1984.6s |
| grimgar03 | qwen3-14b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda | 32768 | lora | 385 | 76.6% | exploratory | ok | False | 1929.7s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | lora | 547 | 76.2% | historical_only | ok | True | 28194.5s |
| index18 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | base | 225 | 60.9% | historical_only | ok | False | 19529.9s |
| index18 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | lora | 225 | 69.3% | historical_only | ok | False | 19529.9s |

## narrator_prior

| book | model | env | backend | ctx | arm | n | acc | evidence | valid | dirty | elapsed |
|---|---|---|---|---:|---|---:|---:|---|---|---|---:|
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | baseline | 400 | 57.2% | historical_only | ok | False | 2180.7s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | narrator | 400 | 60.2% | historical_only | ok | False | 2180.7s |
| index18 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | baseline | 99 | 64.6% | supported_measurement | ok | False | 1021.5s |
| index18 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | narrator | 99 | 65.7% | supported_measurement | ok | False | 1021.5s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | baseline | 139 | 51.8% | provisional | ok | False | 842.4s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | narrator | 139 | 56.1% | provisional | ok | False | 842.4s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | baseline | 162 | 38.3% | supported_measurement | ok | False | 1783.8s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | narrator | 162 | 38.9% | supported_measurement | ok | False | 1783.8s |

## reasoning_arms

| book | model | env | backend | ctx | arm | n | acc | evidence | valid | dirty | elapsed |
|---|---|---|---|---:|---|---:|---:|---|---|---|---:|
| grimgar03 | gemma-3-27b | cloud-a6000-lmstudio | lmstudio | 16384 | baseline | 400 | 71.5% | historical_only | ok | False | 9454.9s |
| grimgar03 | gemma-3-27b | cloud-a6000-lmstudio | lmstudio | 16384 | because | 400 | 72.2% | historical_only | ok | False | 9454.9s |
| grimgar03 | gemma-3-27b | cloud-a6000-lmstudio | lmstudio | 16384 | scaffold | 400 | 68.2% | historical_only | ok | False | 9454.9s |
| grimgar03 | gemma-3-27b | cloud-a6000-lmstudio | lmstudio | 16384 | scaffold_thinking | 400 | 68.2% | historical_only | ok | False | 9454.9s |
| grimgar03 | gemma-3-27b | cloud-a6000-lmstudio | lmstudio | 16384 | thinking | 400 | 71.2% | historical_only | ok | False | 9454.9s |
| grimgar03 | magistral-small | cloud-a6000-lmstudio | lmstudio | 16384 | baseline | 400 | 64.0% | exploratory | ok | False | 8256.2s |
| grimgar03 | magistral-small | cloud-a6000-lmstudio | lmstudio | 16384 | because | 400 | 61.8% | exploratory | ok | False | 8256.2s |
| grimgar03 | magistral-small | cloud-a6000-lmstudio | lmstudio | 16384 | scaffold | 400 | 52.0% | exploratory | ok | False | 8256.2s |
| grimgar03 | magistral-small | cloud-a6000-lmstudio | lmstudio | 16384 | scaffold_thinking | 400 | 52.0% | exploratory | ok | False | 8256.2s |
| grimgar03 | magistral-small | cloud-a6000-lmstudio | lmstudio | 16384 | thinking | 400 | 63.0% | exploratory | ok | False | 8256.2s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | baseline | 400 | 56.5% | historical_only | ok | True | 8588.7s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | because | 400 | 54.8% | historical_only | ok | True | 8588.7s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | scaffold | 400 | 52.5% | historical_only | ok | True | 8588.7s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | scaffold_thinking | 400 | 56.5% | historical_only | ok | True | 8588.7s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | thinking | 400 | 66.2% | historical_only | ok | True | 8588.7s |
| grimgar03 | qwen3-32b | cloud-a6000-lmstudio | lmstudio | 16384 | baseline | 400 | 67.2% | exploratory | ok | False | 21565.9s |
| grimgar03 | qwen3-32b | cloud-a6000-lmstudio | lmstudio | 16384 | because | 400 | 67.5% | exploratory | ok | False | 21565.9s |
| grimgar03 | qwen3-32b | cloud-a6000-lmstudio | lmstudio | 16384 | scaffold | 400 | 57.8% | exploratory | ok | False | 21565.9s |
| grimgar03 | qwen3-32b | cloud-a6000-lmstudio | lmstudio | 16384 | scaffold_thinking | 400 | 68.5% | exploratory | ok | False | 21565.9s |
| grimgar03 | qwen3-32b | cloud-a6000-lmstudio | lmstudio | 16384 | thinking | 400 | 72.2% | exploratory | ok | False | 21565.9s |
| index18 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | baseline | 99 | 61.6% | supported_measurement | ok | False | 8280.2s |
| index18 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | because | 99 | 62.6% | supported_measurement | ok | False | 8280.2s |
| index18 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | scaffold | 99 | 56.6% | supported_measurement | ok | False | 8280.2s |
| index18 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | scaffold_thinking | 99 | 68.7% | supported_measurement | ok | False | 8280.2s |
| index18 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | thinking | 99 | 68.7% | supported_measurement | ok | False | 8280.2s |
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | baseline | 139 | 39.6% | historical_only | ok | True | 5021.5s |
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | because | 139 | 50.4% | historical_only | ok | True | 5021.5s |
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | scaffold | 139 | 41.0% | historical_only | ok | True | 5021.5s |
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | scaffold_thinking | 139 | 48.2% | historical_only | ok | True | 5021.5s |
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | thinking | 139 | 41.7% | historical_only | ok | True | 5021.5s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | baseline | 162 | 43.2% | supported_measurement | ok | False | 12654.2s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | because | 162 | 47.5% | supported_measurement | ok | False | 12654.2s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | scaffold | 162 | 38.9% | supported_measurement | ok | False | 12654.2s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | scaffold_thinking | 162 | 43.2% | supported_measurement | ok | False | 12654.2s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | thinking | 162 | 38.9% | supported_measurement | ok | False | 12654.2s |

## reasoning_check

| book | model | env | backend | ctx | arm | n | acc | evidence | valid | dirty | elapsed |
|---|---|---|---|---:|---|---:|---:|---|---|---|---:|
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | plain | 396 | 58.1% | historical_only | ok | False | 3748.4s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | reasoned | 396 | 59.3% | historical_only | ok | False | 3748.4s |
| index18 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | plain | 99 | 62.6% | supported_measurement | ok | False | 1966.4s |
| index18 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | reasoned | 99 | 58.6% | supported_measurement | ok | False | 1966.4s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | plain | 139 | 51.1% | supported_measurement | ok | False | 1970.5s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | reasoned | 139 | 48.2% | supported_measurement | ok | False | 1970.5s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | plain | 162 | 40.7% | supported_measurement | ok | False | 2902.2s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | reasoned | 162 | 41.4% | supported_measurement | ok | False | 2902.2s |

## reexamine

| book | model | env | backend | ctx | arm | n | acc | evidence | valid | dirty | elapsed |
|---|---|---|---|---:|---|---:|---:|---|---|---|---:|
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | baseline | 139 | 49.6% | provisional | ok | True | 2737.5s |
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | narration | 139 | 34.5% | provisional | ok | True | 2737.5s |
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | narrator | 139 | 51.8% | provisional | ok | True | 2737.5s |
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | prose | 139 | 47.5% | provisional | ok | True | 2737.5s |
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | voting | 139 | 49.6% | provisional | ok | True | 2737.5s |

## roster_quality

| book | model | env | backend | ctx | arm | n | acc | evidence | valid | dirty | elapsed |
|---|---|---|---|---:|---|---:|---:|---|---|---|---:|
| grimgar03 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | augmented | 385 | 85.7% | exploratory | ok | False | 8051.1s |
| grimgar03 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | generated | 385 | 84.9% | exploratory | ok | False | 8051.1s |
| grimgar03 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | gold | 385 | 83.9% | exploratory | ok | False | 8051.1s |
| grimgar03 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | inflated | 385 | 83.6% | exploratory | ok | False | 8051.1s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | augmented | 385 | 63.6% | provisional | ok | True | 4837.9s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | generated | 385 | 59.7% | provisional | ok | True | 4837.9s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | gold | 385 | 61.6% | provisional | ok | True | 4837.9s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | inflated | 385 | 61.0% | provisional | ok | True | 4837.9s |
| index18 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | augmented | 92 | 79.3% | exploratory | ok | False | 7535.3s |
| index18 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | generated | 92 | 80.4% | exploratory | ok | False | 7535.3s |
| index18 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | gold | 92 | 82.6% | exploratory | ok | False | 7535.3s |
| index18 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | inflated | 92 | 79.3% | exploratory | ok | False | 7535.3s |
| index18 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | augmented | 92 | 71.7% | provisional | ok | True | 2056.7s |
| index18 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | generated | 92 | 67.4% | provisional | ok | True | 2056.7s |
| index18 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | gold | 92 | 69.6% | provisional | ok | True | 2056.7s |
| index18 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | inflated | 92 | 66.3% | provisional | ok | True | 2056.7s |
| mushoku16 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | augmented | 133 | 69.2% | exploratory | ok | False | 4930.5s |
| mushoku16 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | generated | 133 | 69.2% | exploratory | ok | False | 4930.5s |
| mushoku16 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | gold | 133 | 69.9% | exploratory | ok | False | 4930.5s |
| mushoku16 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | inflated | 133 | 68.4% | exploratory | ok | False | 4930.5s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | augmented | 133 | 51.9% | provisional | ok | True | 2800.3s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | generated | 133 | 48.9% | provisional | ok | True | 2800.3s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | gold | 133 | 48.9% | provisional | ok | True | 2800.3s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | inflated | 133 | 48.1% | provisional | ok | True | 2800.3s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | augmented | 162 | 45.1% | provisional | ok | True | 7321.2s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | generated | 162 | 40.7% | provisional | ok | True | 7321.2s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | gold | 162 | 42.6% | provisional | ok | True | 7321.2s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | inflated | 162 | 30.9% | provisional | ok | True | 7321.2s |

## roster_warmup

| book | model | env | backend | ctx | arm | n | acc | evidence | valid | dirty | elapsed |
|---|---|---|---|---:|---|---:|---:|---|---|---|---:|
| mushoku16 | ministral-3-14b-instruct-2 | local-lmstudio | lmstudio | 16384 | incremental | 139 | 41.0% | provisional | ok | False | 8408.8s |
| mushoku16 | ministral-3-14b-instruct-2 | local-lmstudio | lmstudio | 16384 | oracle | 139 | 46.8% | provisional | ok | False | 8408.8s |
| mushoku16 | ministral-3-14b-instruct-2 | local-lmstudio | lmstudio | 16384 | warm | 139 | 44.6% | provisional | ok | False | 8408.8s |
| mushoku16 | qwen3.5-9b-uncensored-hauh | local-lmstudio | lmstudio | 32768 | incremental | 139 | 27.3% | provisional | ok | False | 1039.6s |
| mushoku16 | qwen3.5-9b-uncensored-hauh | local-lmstudio | lmstudio | 32768 | oracle | 139 | 35.3% | provisional | ok | False | 1039.6s |
| mushoku16 | qwen3.5-9b-uncensored-hauh | local-lmstudio | lmstudio | 32768 | warm | 139 | 32.4% | provisional | ok | False | 1039.6s |

## scene_cast

| book | model | env | backend | ctx | arm | n | acc | evidence | valid | dirty | elapsed |
|---|---|---|---|---:|---|---:|---:|---|---|---|---:|
| grimgar03 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | full | 385 | 85.7% | exploratory | ok | False | 6197.0s |
| grimgar03 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | scene | 385 | 84.9% | exploratory | ok | False | 6197.0s |
| grimgar03 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | scene+2 | 385 | 86.8% | exploratory | ok | False | 6197.0s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | full | 385 | 63.6% | provisional | ok | True | 2178.2s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | full | 385 | 63.6% | provisional | ok | True | 2284.5s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | full | 385 | 66.0% | provisional | ok | True | 2094.8s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | scene | 385 | 66.0% | provisional | ok | True | 2178.2s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | scene | 385 | 62.3% | provisional | ok | True | 2284.5s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | scene | 385 | 63.1% | provisional | ok | True | 2094.8s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | scene+2 | 385 | 65.5% | provisional | ok | True | 2178.2s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | scene+2 | 385 | 63.1% | provisional | ok | True | 2284.5s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | scene+2 | 385 | 63.6% | provisional | ok | True | 2094.8s |
| index18 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | full | 92 | 81.5% | exploratory | ok | False | 5653.6s |
| index18 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | scene | 92 | 80.4% | exploratory | ok | False | 5653.6s |
| index18 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | scene+2 | 92 | 80.4% | exploratory | ok | False | 5653.6s |
| index18 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | full | 92 | 71.7% | provisional | ok | True | 1518.9s |
| index18 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | scene | 92 | 68.5% | provisional | ok | True | 1518.9s |
| index18 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | scene+2 | 92 | 68.5% | provisional | ok | True | 1518.9s |
| mushoku16 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | full | 133 | 69.2% | exploratory | ok | False | 4011.9s |
| mushoku16 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | scene | 133 | 62.4% | exploratory | ok | False | 4011.9s |
| mushoku16 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | scene+2 | 133 | 70.7% | exploratory | ok | False | 4011.9s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | full | 133 | 51.9% | provisional | ok | True | 1690.7s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | full | 133 | 51.9% | provisional | ok | True | 1714.3s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | scene | 133 | 51.9% | provisional | ok | True | 1690.7s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | scene | 133 | 52.6% | provisional | ok | True | 1714.3s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | scene+2 | 133 | 52.6% | provisional | ok | True | 1690.7s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | scene+2 | 133 | 51.9% | provisional | ok | True | 1714.3s |
| owarimonogatari3 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | full | 162 | 64.2% | exploratory | ok | False | 7587.4s |
| owarimonogatari3 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | scene | 162 | 54.9% | exploratory | ok | False | 7587.4s |
| owarimonogatari3 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | scene+2 | 162 | 62.3% | exploratory | ok | False | 7587.4s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | full | 162 | 45.1% | provisional | ok | True | 3728.1s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | scene | 162 | 38.9% | provisional | ok | True | 3728.1s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | scene+2 | 162 | 40.7% | provisional | ok | True | 3728.1s |

## segmentation_crossover

| book | model | env | backend | ctx | arm | n | acc | evidence | valid | dirty | elapsed |
|---|---|---|---|---:|---|---:|---:|---|---|---|---:|
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | seg=gemma,attr=gemma,t=0.0,rep=1 | 399 | 58.6% | historical_only | ok | False | 1832.5s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | seg=gemma,attr=gemma,t=0.0,rep=2 | 399 | 58.6% | historical_only | ok | False | 1832.5s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | seg=gemma,attr=gemma,t=0.6,rep=1 | 399 | 57.1% | historical_only | ok | False | 1832.5s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | seg=gemma,attr=gemma,t=0.6,rep=2 | 399 | 57.9% | historical_only | ok | False | 1832.5s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | seg=gemma,attr=gemma,t=0.6,rep=3 | 399 | 59.4% | historical_only | ok | False | 1832.5s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | seg=gemma,attr=qwen,t=0.0,rep=1 | 399 | 60.9% | historical_only | ok | False | 1832.5s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | seg=gemma,attr=qwen,t=0.0,rep=2 | 399 | 60.9% | historical_only | ok | False | 1832.5s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | seg=gemma,attr=qwen,t=0.6,rep=1 | 399 | 60.7% | historical_only | ok | False | 1832.5s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | seg=gemma,attr=qwen,t=0.6,rep=2 | 399 | 60.7% | historical_only | ok | False | 1832.5s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | seg=gemma,attr=qwen,t=0.6,rep=3 | 399 | 59.9% | historical_only | ok | False | 1832.5s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | seg=qwen,attr=gemma,t=0.0,rep=1 | 399 | 56.6% | historical_only | ok | False | 1832.5s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | seg=qwen,attr=gemma,t=0.0,rep=2 | 399 | 56.6% | historical_only | ok | False | 1832.5s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | seg=qwen,attr=gemma,t=0.6,rep=1 | 399 | 55.9% | historical_only | ok | False | 1832.5s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | seg=qwen,attr=gemma,t=0.6,rep=2 | 399 | 56.6% | historical_only | ok | False | 1832.5s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | seg=qwen,attr=gemma,t=0.6,rep=3 | 399 | 57.4% | historical_only | ok | False | 1832.5s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | seg=qwen,attr=qwen,t=0.0,rep=1 | 399 | 58.4% | historical_only | ok | False | 1832.5s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | seg=qwen,attr=qwen,t=0.0,rep=2 | 399 | 58.4% | historical_only | ok | False | 1832.5s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | seg=qwen,attr=qwen,t=0.6,rep=1 | 399 | 58.6% | historical_only | ok | False | 1832.5s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | seg=qwen,attr=qwen,t=0.6,rep=2 | 399 | 57.9% | historical_only | ok | False | 1832.5s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | seg=qwen,attr=qwen,t=0.6,rep=3 | 399 | 58.1% | historical_only | ok | False | 1832.5s |

## tag_priority

| book | model | env | backend | ctx | arm | n | acc | evidence | valid | dirty | elapsed |
|---|---|---|---|---:|---|---:|---:|---|---|---|---:|
| grimgar03 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | baseline | 396 | 82.6% | exploratory | ok | False | 4007.5s |
| grimgar03 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | tagfirst | 396 | 82.8% | exploratory | ok | False | 4007.5s |
| grimgar03 | magistral-small | local-llamacpp-hip | llama.cpp-hip | 16384 | baseline | 396 | 70.7% | provisional | ok | True | 2667.2s |
| grimgar03 | magistral-small | local-llamacpp-hip | llama.cpp-hip | 16384 | tagfirst | 396 | 68.9% | provisional | ok | True | 2667.2s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | baseline | 396 | 63.4% | supported_measurement | ok | False | 1803.9s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | baseline | 396 | 62.4% | supported_measurement | ok | False | 1782.1s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | baseline | 396 | 62.4% | supported_measurement | ok | False | 1781.9s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | baseline | 400 | 56.5% | historical_only | ok | True | 1887.0s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | tagfirst | 396 | 62.4% | supported_measurement | ok | False | 1803.9s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | tagfirst | 396 | 62.4% | supported_measurement | ok | False | 1782.1s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | tagfirst | 396 | 62.4% | supported_measurement | ok | False | 1781.9s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | tagfirst | 400 | 63.0% | historical_only | ok | True | 1887.0s |
| index18 | magistral-small | local-llamacpp-hip | llama.cpp-hip | 16384 | baseline | 99 | 68.7% | provisional | ok | True | 1642.5s |
| index18 | magistral-small | local-llamacpp-hip | llama.cpp-hip | 16384 | tagfirst | 99 | 64.6% | provisional | ok | True | 1642.5s |
| index18 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | baseline | 99 | 64.6% | supported_measurement | ok | False | 913.0s |
| index18 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | tagfirst | 99 | 65.7% | supported_measurement | ok | False | 913.0s |
| mushoku16 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | baseline | 136 | 67.6% | exploratory | ok | False | 2362.9s |
| mushoku16 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | tagfirst | 136 | 66.2% | exploratory | ok | False | 2362.9s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | baseline | 139 | 51.1% | provisional | ok | True | 1073.2s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | tagfirst | 139 | 45.3% | provisional | ok | True | 1073.2s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | baseline | 162 | 40.7% | supported_measurement | ok | False | 1891.9s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | tagfirst | 162 | 43.2% | supported_measurement | ok | False | 1891.9s |

## two_by_two

| book | model | env | backend | ctx | arm | n | acc | evidence | valid | dirty | elapsed |
|---|---|---|---|---:|---|---:|---:|---|---|---|---:|
| mushoku16 | qwen3.5-9b-uncensored-hauh | local-lmstudio | lmstudio |  | A | 139 | 19.4% | exploratory | ['no LM Studio load state recorded', 'en | True | 698.2s |
| mushoku16 | qwen3.5-9b-uncensored-hauh | local-lmstudio | lmstudio |  | B | 139 | 2.2% | exploratory | ['no LM Studio load state recorded', 'en | True | 698.2s |
| mushoku16 | qwen3.5-9b-uncensored-hauh | local-lmstudio | lmstudio |  | C | 139 | 34.5% | exploratory | ['no LM Studio load state recorded', 'en | True | 698.2s |
| mushoku16 | qwen3.5-9b-uncensored-hauh | local-lmstudio | lmstudio |  | D | 139 | 18.7% | exploratory | ['no LM Studio load state recorded', 'en | True | 698.2s |

## voting

| book | model | env | backend | ctx | arm | n | acc | evidence | valid | dirty | elapsed |
|---|---|---|---|---:|---|---:|---:|---|---|---|---:|
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | greedy | 400 | 55.8% | historical_only | ok | False | 6517.4s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | vote3 | 400 | 58.0% | historical_only | ok | False | 6517.4s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | vote5 | 400 | 57.8% | historical_only | ok | False | 6517.4s |
| index18 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | greedy | 99 | 64.6% | supported_measurement | ok | False | 1607.3s |
| index18 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | vote3 | 99 | 63.6% | supported_measurement | ok | False | 1607.3s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | greedy | 139 | 47.5% | provisional | ok | False | 3221.3s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | vote3 | 139 | 48.9% | provisional | ok | False | 3221.3s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | vote5 | 139 | 48.9% | provisional | ok | False | 3221.3s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | greedy | 162 | 40.7% | supported_measurement | ok | False | 2872.6s |
| owarimonogatari3 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | vote3 | 162 | 38.3% | supported_measurement | ok | False | 2872.6s |

## Not indexed

These artifacts exist and hold real results; this table only represents per-arm attribution accuracy, so they cannot be rendered as rows. Read them directly.

| artifact | why |
|---|---|
| `audible_errors.json` | NOT INDEXED: no 'rows' list - this table only represents per-arm attribution results |
| `cascade_state__grimgar03__a6000-batchtrig.json` | NOT INDEXED: no 'rows' list - this table only represents per-arm attribution results |
| `cascade_state__grimgar03__a6000-contig.json` | NOT INDEXED: no 'rows' list - this table only represents per-arm attribution results |
| `cascade_state__grimgar03__local-batchtrig.json` | NOT INDEXED: no 'rows' list - this table only represents per-arm attribution results |
| `cascade_state__grimgar03__local-bt-rep1.json` | NOT INDEXED: no 'rows' list - this table only represents per-arm attribution results |
| `cascade_state__grimgar03__local-bt-rep2.json` | NOT INDEXED: no 'rows' list - this table only represents per-arm attribution results |
| `cascade_state__grimgar03__thunder-a6000.json` | NOT INDEXED: no 'rows' list - this table only represents per-arm attribution results |
| `cascade_state__grimgar03__tuned-cheap-arm.json` | NOT INDEXED: no 'rows' list - this table only represents per-arm attribution results |
| `cascade_state__index18__a6000-batchtrig.json` | NOT INDEXED: no 'rows' list - this table only represents per-arm attribution results |
| `cascade_state__index18__a6000-newbook.json` | NOT INDEXED: no 'rows' list - this table only represents per-arm attribution results |
| `cascade_state__index18__local-batchtrig.json` | NOT INDEXED: no 'rows' list - this table only represents per-arm attribution results |
| `cascade_state__index18__tuned-cheap-arm.json` | NOT INDEXED: no 'rows' list - this table only represents per-arm attribution results |
| `cascade_state__mushoku16__a6000-batchtrig.json` | NOT INDEXED: no 'rows' list - this table only represents per-arm attribution results |
| `cascade_state__mushoku16__a6000-contig.json` | NOT INDEXED: no 'rows' list - this table only represents per-arm attribution results |
| `cascade_state__mushoku16__local-batchtrig.json` | NOT INDEXED: no 'rows' list - this table only represents per-arm attribution results |
| `cascade_state__mushoku16__thunder-a6000.json` | NOT INDEXED: no 'rows' list - this table only represents per-arm attribution results |
| `cascade_state__mushoku16__tuned-cheap-arm.json` | NOT INDEXED: no 'rows' list - this table only represents per-arm attribution results |
| `cascade_state__owarimonogatari3__a6000-batchtrig.json` | NOT INDEXED: no 'rows' list - this table only represents per-arm attribution results |
| `cascade_state__owarimonogatari3__a6000-newbook.json` | NOT INDEXED: no 'rows' list - this table only represents per-arm attribution results |
| `cascade_state__owarimonogatari3__local-batchtrig.json` | NOT INDEXED: no 'rows' list - this table only represents per-arm attribution results |
| `cascade_state__owarimonogatari3__tuned-cheap-arm.json` | NOT INDEXED: no 'rows' list - this table only represents per-arm attribution results |
| `chapter_manifest.json` | SKIPPED: not a result object (list) |
| `chapter_validation.json` | SKIPPED: 'rows' is not a list of scored arms |
| `chinese_attribution_jy_fixed.json` | NOT INDEXED: no 'rows' list - this table only represents per-arm attribution results |
| `chinese_attribution_wp_fixed.json` | NOT INDEXED: no 'rows' list - this table only represents per-arm attribution results |
| `cluster_vs_name.json` | NOT INDEXED: no 'rows' list - this table only represents per-arm attribution results |
| `fallback_policy.json` | NOT INDEXED: no 'rows' list - this table only represents per-arm attribution results |
| `fix_verification.json` | SKIPPED: not a result object (list) |
| `instruct_value.json` | SKIPPED: 'rows' is not a list of scored arms |
| `japanese_quote_robustness.json` | NOT INDEXED: no 'rows' list - this table only represents per-arm attribution results |
| `listener_impact.json` | NOT INDEXED: no 'rows' list - this table only represents per-arm attribution results |
| `nonprose_gate.json` | NOT INDEXED: no 'rows' list - this table only represents per-arm attribution results |
| `nonprose_split.json` | SKIPPED: 'rows' is not a list of scored arms |
| `offbyone_turns.json` | NOT INDEXED: no 'rows' list - this table only represents per-arm attribution results |
| `pdnc_eval_full_summary.json` | NOT INDEXED: no 'rows' list - this table only represents per-arm attribution results |
| `prose_vs_nonprose.json` | SKIPPED: 'rows' is not a list of scored arms |
| `prose_vs_nonprose_v2.json` | SKIPPED: 'rows' is not a list of scored arms |
| `realizable_router.json` | NOT INDEXED: no 'rows' list - this table only represents per-arm attribution results |
| `residual_errors.json` | NOT INDEXED: no 'rows' list - this table only represents per-arm attribution results |
| `run_lengths.json` | NOT INDEXED: no 'rows' list - this table only represents per-arm attribution results |
| `scene_aware_casting.json` | NOT INDEXED: no 'rows' list - this table only represents per-arm attribution results |
| `segmentation_classifier.json` | NOT INDEXED: no 'rows' list - this table only represents per-arm attribution results |
| `shipping_readiness.json` | NOT INDEXED: no 'rows' list - this table only represents per-arm attribution results |
| `stack_overlap.json` | NOT INDEXED: no 'rows' list - this table only represents per-arm attribution results |
| `training_composition.json` | NOT INDEXED: no 'rows' list - this table only represents per-arm attribution results |
| `trivial_baselines.json` | NOT INDEXED: no 'rows' list - this table only represents per-arm attribution results |
| `tuned_disagreement.json` | NOT INDEXED: no 'rows' list - this table only represents per-arm attribution results |
| `validation_baseline.json` | SKIPPED: 'rows' is not a list of scored arms |
| `validation_smoke.json` | SKIPPED: 'rows' is not a list of scored arms |
| `voice_adapter_health.json` | SKIPPED: 'rows' is not a list of scored arms |
| `voice_blending.json` | NOT INDEXED: no 'rows' list - this table only represents per-arm attribution results |
| `weak_supervision.json` | SKIPPED: 'rows' is not a list of scored arms |
