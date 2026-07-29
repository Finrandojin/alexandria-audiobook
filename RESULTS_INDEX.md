# Results index

Generated 2026-07-28 19:09 from `ab_test_runtime/experiments/` — 72 artifacts, 240 arms.

Regenerate with `python3 collect_results.py`. Machine-readable copy in `results_index.csv`.

`dirty=True` means tracked files were modified when the artifact was written: the numbers are inspectable but the run is not reproducible from its recorded commit.


## batch_size

| book | model | env | backend | ctx | arm | n | acc | valid | dirty | elapsed |
|---|---|---|---|---:|---|---:|---:|---|---|---:|
| grimgar03 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | b1 | 400 | 60.5% | ok | False | 3853.3s |
| grimgar03 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | b25 | 400 | 79.2% | ok | False | 3853.3s |
| grimgar03 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | b5 | 400 | 69.0% | ok | False | 3853.3s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | b1 | 400 | 54.0% | ok | False | 2298.0s |
| grimgar03 | qwen3-14b | cloud-a6000-LM | LM Studio loopback | 16384 | b1 | 400 | 52.8% | ok | False | 2647.3s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | b10 | 400 | 54.0% | ok | False | 2298.0s |
| grimgar03 | qwen3-14b | cloud-a6000-LM | LM Studio loopback | 16384 | b10 | 400 | 52.0% | ok | False | 2647.3s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | b100 | 400 | 62.3% | ok | True | 2990.1s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | b25 | 400 | 59.8% | ok | True | 2990.1s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | b25 | 400 | 55.5% | ok | False | 2298.0s |
| grimgar03 | qwen3-14b | cloud-a6000-LM | LM Studio loopback | 16384 | b25 | 400 | 55.8% | ok | False | 2647.3s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | b5 | 400 | 54.0% | ok | False | 2298.0s |
| grimgar03 | qwen3-14b | cloud-a6000-LM | LM Studio loopback | 16384 | b5 | 400 | 52.0% | ok | False | 2647.3s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | b50 | 400 | 63.7% | ok | True | 2990.1s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | b1 | 139 | 30.2% | ok | False | 952.5s |
| mushoku16 | qwen3-14b | cloud-a6000-LM | LM Studio loopback | 16384 | b1 | 139 | 28.8% | ok | False | 1396.6s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | b10 | 139 | 46.0% | ok | False | 952.5s |
| mushoku16 | qwen3-14b | cloud-a6000-LM | LM Studio loopback | 16384 | b10 | 139 | 38.1% | ok | False | 1396.6s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | b100 | 139 | 49.6% | ok | True | 3425.6s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | b25 | 139 | 51.8% | ok | True | 3425.6s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | b25 | 139 | 49.6% | ok | False | 952.5s |
| mushoku16 | qwen3-14b | cloud-a6000-LM | LM Studio loopback | 16384 | b25 | 139 | 46.8% | ok | False | 1396.6s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | b5 | 139 | 40.3% | ok | False | 952.5s |
| mushoku16 | qwen3-14b | cloud-a6000-LM | LM Studio loopback | 16384 | b5 | 139 | 37.4% | ok | False | 1396.6s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 32768 | b50 | 139 | 48.9% | ok | True | 3425.6s |

## because_production

| book | model | env | backend | ctx | arm | n | acc | valid | dirty | elapsed |
|---|---|---|---|---:|---|---:|---:|---|---|---:|
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | baseline | 400 | 55.5% | ok | False | 4766.2s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | thinking | 400 | 63.7% | ok | False | 4766.2s |
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | baseline | 139 | 49.6% | ok | False | 2668.6s |
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | baseline | 139 | 49.6% | ok | True | 4564.3s |
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | because | 139 | 42.4% | ok | True | 4564.3s |
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | scaffold_thinking | 139 | 43.2% | ok | True | 4564.3s |
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | thinking | 139 | 52.5% | ok | False | 2668.6s |

## candidate_id

| book | model | env | backend | ctx | arm | n | acc | valid | dirty | elapsed |
|---|---|---|---|---:|---|---:|---:|---|---|---:|
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | id | 147 | 35.4% | ok | False | 68.3s |
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | name | 147 | 49.0% | ok | False | 68.3s |

## cascade

| book | model | env | backend | ctx | arm | n | acc | valid | dirty | elapsed |
|---|---|---|---|---:|---|---:|---:|---|---|---:|
| grimgar03 | qwen3-14b + llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | cascade | 400 | 77.8% | ok | False | 0.1s |
| grimgar03 | qwen3-14b + llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | cheap-w1 | 400 | 55.8% | ok | False | 0.1s |
| index18 | qwen3-14b + llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | cascade | 99 | 73.7% | ok | False | 0.1s |
| index18 | qwen3-14b + llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | cheap-w1 | 99 | 62.6% | ok | False | 0.1s |
| mushoku16 | qwen3-14b + llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | cascade | 139 | 64.0% | ok | False | 0.1s |
| mushoku16 | qwen3-14b + llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | cheap-w1 | 139 | 47.5% | ok | False | 0.1s |
| owarimonogatari3 | qwen3-14b + llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | cascade | 162 | 56.2% | ok | False | 0.1s |
| owarimonogatari3 | qwen3-14b + llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | cheap-w1 | 162 | 42.0% | ok | False | 0.1s |

## closed_set

| book | model | env | backend | ctx | arm | n | acc | valid | dirty | elapsed |
|---|---|---|---|---:|---|---:|---:|---|---|---:|
| grimgar03 | gemma-4-e4b-uncensored-hau | local-lmstudio | lmstudio | 32768 | closed-6 | 400 | 59.8% | ok | False | 198.1s |
| grimgar03 | gemma-4-e4b-uncensored-hau | local-lmstudio | lmstudio | 32768 | closed-oracle | 400 | 65.5% | ok | False | 198.1s |
| grimgar03 | gemma-4-e4b-uncensored-hau | local-lmstudio | lmstudio | 32768 | open | 400 | 57.2% | ok | False | 198.1s |
| grimgar03 | gemma-3-27b | cloud-a6000-lmstudio | lmstudio | 16384 | closed-6 | 400 | 62.3% | ok | False | 1206.5s |
| grimgar03 | gemma-3-27b | cloud-a6000-lmstudio | lmstudio | 16384 | closed-oracle | 400 | 70.5% | ok | False | 1206.5s |
| grimgar03 | gemma-3-27b | cloud-a6000-lmstudio | lmstudio | 16384 | open | 400 | 61.5% | ok | False | 1206.5s |
| grimgar03 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda b10 | 8192 | closed-6 | 400 | 76.5% | ok | False | 1489.6s |
| grimgar03 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda b10 | 8192 | closed-oracle | 400 | 83.0% | ok | False | 1489.6s |
| grimgar03 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda b10 | 8192 | open | 400 | 75.8% | ok | False | 1489.6s |
| grimgar03 | ministral-3-14b-instruct-2 | local-lmstudio | lmstudio | 16384 | closed-6 | 400 | 52.5% | ok | False | 285.2s |
| grimgar03 | ministral-3-14b-instruct-2 | local-lmstudio | lmstudio | 16384 | closed-oracle | 400 | 58.2% | ok | False | 285.2s |
| grimgar03 | ministral-3-14b-instruct-2 | local-lmstudio | lmstudio | 16384 | open | 400 | 51.5% | ok | False | 285.2s |
| grimgar03 | magistral-small | local-llamacpp-hip | llama.cpp-hip b101 | 8192 | closed-6 | 400 | 61.5% | ok | False | 478.5s |
| grimgar03 | magistral-small | cloud-a6000-lmstudio | lmstudio | 16384 | closed-6 | 400 | 58.0% | ok | False | 854.3s |
| grimgar03 | magistral-small | local-llamacpp-hip | llama.cpp-hip b101 | 8192 | closed-oracle | 400 | 74.8% | ok | False | 478.5s |
| grimgar03 | magistral-small | cloud-a6000-lmstudio | lmstudio | 16384 | closed-oracle | 400 | 70.8% | ok | False | 854.3s |
| grimgar03 | magistral-small | local-llamacpp-hip | llama.cpp-hip b101 | 8192 | open | 400 | 59.8% | ok | False | 478.5s |
| grimgar03 | magistral-small | cloud-a6000-lmstudio | lmstudio | 16384 | open | 400 | 61.3% | ok | False | 854.3s |
| grimgar03 | qwen3-14b | cloud-a6000-lmstudio | lmstudio | 98304 | closed-6 | 400 | 60.8% | ok | False | 915.3s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | closed-6 | 400 | 60.5% | ok | False | 291.8s |
| grimgar03 | qwen3-14b | cloud-a6000-lmstudio | lmstudio | 98304 | closed-6 | 400 | 60.8% | ok | False | 915.3s |
| grimgar03 | qwen3-14b | cloud-a6000-lmstudio | lmstudio | 98304 | closed-oracle | 400 | 72.8% | ok | False | 915.3s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | closed-oracle | 400 | 72.5% | ok | False | 291.8s |
| grimgar03 | qwen3-14b | cloud-a6000-lmstudio | lmstudio | 98304 | closed-oracle | 400 | 72.8% | ok | False | 915.3s |
| grimgar03 | qwen3-14b | cloud-a6000-lmstudio | lmstudio | 98304 | open | 400 | 61.3% | ok | False | 915.3s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | open | 400 | 60.8% | ok | False | 291.8s |
| grimgar03 | qwen3-14b | cloud-a6000-lmstudio | lmstudio | 98304 | open | 400 | 61.3% | ok | False | 915.3s |
| grimgar03 | qwen3-32b | cloud-a6000-lmstudio | lmstudio | 16384 | closed-6 | 400 | 61.5% | ok | False | 1430.8s |
| grimgar03 | qwen3-32b | cloud-a6000-lmstudio | lmstudio | 16384 | closed-oracle | 400 | 76.2% | ok | False | 1430.8s |
| grimgar03 | qwen3-32b | cloud-a6000-lmstudio | lmstudio | 16384 | open | 400 | 61.8% | ok | False | 1430.8s |
| mushoku16 | gemma-4-e4b-uncensored-hau | local-lmstudio | lmstudio | 32768 | closed-6 | 147 | 38.8% | ok | False | 81.9s |
| mushoku16 | gemma-4-e4b-uncensored-hau | local-lmstudio | lmstudio | 32768 | closed-oracle | 147 | 49.7% | ok | False | 81.9s |
| mushoku16 | gemma-4-e4b-uncensored-hau | local-lmstudio | lmstudio | 32768 | open | 147 | 39.5% | ok | False | 81.9s |
| mushoku16 | gemma-3-27b | cloud-a6000-lmstudio | lmstudio | 16384 | closed-6 | 139 | 44.6% | ok | False | 421.3s |
| mushoku16 | gemma-3-27b | cloud-a6000-lmstudio | lmstudio | 16384 | closed-oracle | 139 | 59.0% | ok | False | 421.3s |
| mushoku16 | gemma-3-27b | cloud-a6000-lmstudio | lmstudio | 16384 | open | 139 | 55.4% | ok | False | 421.3s |
| mushoku16 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda b10 | 8192 | closed-6 | 139 | 60.4% | ok | False | 445.8s |
| mushoku16 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda b10 | 8192 | closed-oracle | 139 | 74.8% | ok | False | 445.8s |
| mushoku16 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda b10 | 8192 | open | 139 | 59.7% | ok | False | 445.8s |
| mushoku16 | phi-4 | local-lmstudio | lmstudio | 16384 | closed-6 | 147 | 32.7% | ok | False | 112.6s |
| mushoku16 | phi-4 | local-lmstudio | lmstudio | 16384 | closed-oracle | 147 | 59.2% | ok | False | 112.6s |
| mushoku16 | phi-4 | local-lmstudio | lmstudio | 16384 | open | 147 | 45.6% | ok | False | 112.6s |
| mushoku16 | ministral-3-14b-instruct-2 | local-lmstudio | lmstudio | 16384 | closed-6 | 147 | 41.5% | ok | False | 64.0s |
| mushoku16 | ministral-3-14b-instruct-2 | local-lmstudio | lmstudio | 16384 | closed-oracle | 147 | 61.2% | ok | False | 64.0s |
| mushoku16 | ministral-3-14b-instruct-2 | local-lmstudio | lmstudio | 16384 | open | 147 | 47.6% | ok | False | 64.0s |
| mushoku16 | ministral-3-14b-instruct-2 | local-lmstudio | lmstudio | 16384 | closed-6 | 147 | 40.8% | ok | False | 84.9s |
| mushoku16 | ministral-3-14b-instruct-2 | local-lmstudio | lmstudio | 16384 | closed-oracle | 147 | 59.2% | ok | False | 84.9s |
| mushoku16 | ministral-3-14b-instruct-2 | local-lmstudio | lmstudio | 16384 | open | 147 | 46.9% | ok | False | 84.9s |
| mushoku16 | magistral-small | local-llamacpp-hip | llama.cpp-hip b101 | 8192 | closed-6 | 139 | 45.3% | ok | False | 100.5s |
| mushoku16 | magistral-small | cloud-a6000-lmstudio | lmstudio | 16384 | closed-6 | 139 | 45.3% | ok | False | 254.3s |
| mushoku16 | magistral-small | local-llamacpp-hip | llama.cpp-hip b101 | 8192 | closed-oracle | 139 | 57.6% | ok | False | 100.5s |
| mushoku16 | magistral-small | cloud-a6000-lmstudio | lmstudio | 16384 | closed-oracle | 139 | 56.8% | ok | False | 254.3s |
| mushoku16 | magistral-small | local-llamacpp-hip | llama.cpp-hip b101 | 8192 | open | 139 | 53.2% | ok | False | 100.5s |
| mushoku16 | magistral-small | cloud-a6000-lmstudio | lmstudio | 16384 | open | 139 | 52.5% | ok | False | 254.3s |
| mushoku16 | qwen3-14b | cloud-a6000-lmstudio | lmstudio | 98304 | closed-6 | 139 | 38.8% | ok | False | 317.6s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip b101 | 16384 | closed-6 | 139 | 38.8% | ok | False | 81.7s |
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | closed-6 | 139 | 39.6% | ok | False | 90.6s |
| mushoku16 | qwen3-14b | local-llamacpp-vulkan | llama.cpp-vulkan b | 16384 | closed-6 | 139 | 39.6% | ok | False | 87.6s |
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | closed-6 | 147 | 36.7% | ok | False | 92.7s |
| mushoku16 | qwen3-14b | cloud-a6000-lmstudio | lmstudio | 98304 | closed-oracle | 139 | 66.9% | ok | False | 317.6s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip b101 | 16384 | closed-oracle | 139 | 66.9% | ok | False | 81.7s |
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | closed-oracle | 139 | 66.2% | ok | False | 90.6s |
| mushoku16 | qwen3-14b | local-llamacpp-vulkan | llama.cpp-vulkan b | 16384 | closed-oracle | 139 | 66.9% | ok | False | 87.6s |
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | closed-oracle | 147 | 66.0% | ok | False | 92.7s |
| mushoku16 | qwen3-14b | cloud-a6000-lmstudio | lmstudio | 98304 | open | 139 | 50.4% | ok | False | 317.6s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip b101 | 16384 | open | 139 | 48.2% | ok | False | 81.7s |
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | open | 139 | 49.6% | ok | False | 90.6s |
| mushoku16 | qwen3-14b | local-llamacpp-vulkan | llama.cpp-vulkan b | 16384 | open | 139 | 49.6% | ok | False | 87.6s |
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | open | 147 | 48.3% | ok | False | 92.7s |
| mushoku16 | qwen3-32b | cloud-a6000-lmstudio | lmstudio | 16384 | closed-6 | 139 | 46.8% | ok | False | 496.9s |
| mushoku16 | qwen3-32b | cloud-a6000-lmstudio | lmstudio | 16384 | closed-oracle | 139 | 64.0% | ok | False | 496.9s |
| mushoku16 | qwen3-32b | cloud-a6000-lmstudio | lmstudio | 16384 | open | 139 | 52.5% | ok | False | 496.9s |
| mushoku16 | qwen3.5-9b-uncensored-hauh | local-lmstudio | lmstudio |  | closed-6 | 147 | 34.7% | None | True | 138.3s |
| mushoku16 | qwen3.5-9b-uncensored-hauh | local-lmstudio | lmstudio | 32768 | closed-6 | 147 | 34.7% | ok | False | 150.0s |
| mushoku16 | qwen3.5-9b-uncensored-hauh | local-lmstudio | lmstudio |  | closed-oracle | 147 | 49.0% | None | True | 138.3s |
| mushoku16 | qwen3.5-9b-uncensored-hauh | local-lmstudio | lmstudio | 32768 | closed-oracle | 147 | 49.0% | ok | False | 150.0s |
| mushoku16 | qwen3.5-9b-uncensored-hauh | local-lmstudio | lmstudio |  | open | 147 | 35.4% | None | True | 138.3s |
| mushoku16 | qwen3.5-9b-uncensored-hauh | local-lmstudio | lmstudio | 32768 | open | 147 | 35.4% | ok | False | 150.0s |

## committed_history

| book | model | env | backend | ctx | arm | n | acc | valid | dirty | elapsed |
|---|---|---|---|---:|---|---:|---:|---|---|---:|
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip b101 | 16384 | none | 400 | 63.5% | ok | False | 337.5s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip b101 | 16384 | oracle | 400 | 63.5% | ok | False | 337.5s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip b101 | 16384 | predicted | 400 | 62.3% | ok | False | 337.5s |

## context_width

| book | model | env | backend | ctx | arm | n | acc | valid | dirty | elapsed |
|---|---|---|---|---:|---|---:|---:|---|---|---:|
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip b101 | 16384 | w1 | 400 | 55.8% | ok | False | 1003.4s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip b101 | 16384 | w15 | 400 | 61.5% | ok | False | 1003.4s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip b101 | 16384 | w4 | 400 | 62.0% | ok | False | 1003.4s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip b101 | 16384 | w40 | 400 | 60.5% | ok | False | 1003.4s |

## context_width_production

| book | model | env | backend | ctx | arm | n | acc | valid | dirty | elapsed |
|---|---|---|---|---:|---|---:|---:|---|---|---:|
| grimgar03 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | w1 | 400 | 79.2% | ok | False | 4704.6s |
| grimgar03 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | w4 | 400 | 76.8% | ok | False | 4704.6s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | w1 | 400 | 59.8% | ok | False | 1914.8s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | w1 | 400 | 58.0% | ok | False | 1897.4s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | w1 | 400 | 58.0% | ok | False | 1882.6s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | w1 | 400 | 57.0% | ok | False | 1426.2s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | w4 | 400 | 69.8% | ok | False | 1914.8s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | w4 | 400 | 69.8% | ok | False | 1897.4s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | w4 | 400 | 69.8% | ok | False | 1882.6s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | w4 | 400 | 67.5% | ok | False | 1426.2s |
| mushoku16 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | w1 | 139 | 66.9% | ok | False | 2904.1s |
| mushoku16 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | w4 | 139 | 59.0% | ok | False | 2904.1s |
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | w1 | 139 | 49.6% | ok | False | 826.8s |
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | w4 | 139 | 44.6% | ok | False | 826.8s |

## grammar_constraint

| book | model | env | backend | ctx | arm | n | acc | valid | dirty | elapsed |
|---|---|---|---|---:|---|---:|---:|---|---|---:|
| mushoku16 | magistral-small | local-llamacpp-hip | llama.cpp-hip b101 | 8192 | open-free | 139 | 53.2% | ok | False | 114.0s |
| mushoku16 | magistral-small | local-llamacpp-hip | llama.cpp-hip b101 | 8192 | open-grammar | 139 | 51.8% | ok | False | 114.0s |
| mushoku16 | magistral-small | local-llamacpp-hip | llama.cpp-hip b101 | 8192 | oracle-free | 139 | 58.3% | ok | False | 114.0s |
| mushoku16 | magistral-small | local-llamacpp-hip | llama.cpp-hip b101 | 8192 | oracle-grammar | 139 | 66.2% | ok | False | 114.0s |

## joint_scene

| book | model | env | backend | ctx | arm | n | acc | valid | dirty | elapsed |
|---|---|---|---|---:|---|---:|---:|---|---|---:|
| grimgar03 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | independent | 400 | 71.5% | ok | False | 2149.6s |
| grimgar03 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | joint-chrono | 400 | 63.2% | ok | False | 2149.6s |
| grimgar03 | llama-3.3-70b | cloud-a6000-llamacpp-cuda | llama.cpp-cuda on- | 16384 | joint-shuffled | 400 | 47.8% | ok | False | 2149.6s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | independent | 400 | 60.5% | ok | False | 498.9s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | joint-chrono | 400 | 57.0% | ok | False | 498.9s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | joint-shuffled | 400 | 50.2% | ok | False | 498.9s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | independent | 139 | 51.8% | ok | False | 213.6s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | joint-chrono | 139 | 47.5% | ok | False | 213.6s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | joint-shuffled | 139 | 48.9% | ok | False | 213.6s |

## narrator_prior

| book | model | env | backend | ctx | arm | n | acc | valid | dirty | elapsed |
|---|---|---|---|---:|---|---:|---:|---|---|---:|
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | baseline | 400 | 57.2% | ok | False | 2180.7s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | narrator | 400 | 60.2% | ok | False | 2180.7s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | baseline | 139 | 51.8% | ok | False | 842.4s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | narrator | 139 | 56.1% | ok | False | 842.4s |

## pipeline_repeat

| book | model | env | backend | ctx | arm | n | acc | valid | dirty | elapsed |
|---|---|---|---|---:|---|---:|---:|---|---|---:|
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | run1 | 396 | 56.1% | n/a (pipeline output, not an ExperimentRecord) |  | s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | run2 | 396 | 56.1% | n/a (pipeline output, not an ExperimentRecord) |  | s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | run3 | 396 | 56.1% | n/a (pipeline output, not an ExperimentRecord) |  | s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | run4 | 396 | 56.1% | n/a (pipeline output, not an ExperimentRecord) |  | s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | run5 | 396 | 56.1% | n/a (pipeline output, not an ExperimentRecord) |  | s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | run6 | 396 | 56.1% | n/a (pipeline output, not an ExperimentRecord) |  | s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | run7 | 396 | 56.1% | n/a (pipeline output, not an ExperimentRecord) |  | s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | run8 | 396 | 56.1% | n/a (pipeline output, not an ExperimentRecord) |  | s |

## reasoning_arms

| book | model | env | backend | ctx | arm | n | acc | valid | dirty | elapsed |
|---|---|---|---|---:|---|---:|---:|---|---|---:|
| grimgar03 | gemma-3-27b | cloud-a6000-lmstudio | lmstudio | 16384 | baseline | 400 | 71.5% | ok | False | 9454.9s |
| grimgar03 | gemma-3-27b | cloud-a6000-lmstudio | lmstudio | 16384 | because | 400 | 72.2% | ok | False | 9454.9s |
| grimgar03 | gemma-3-27b | cloud-a6000-lmstudio | lmstudio | 16384 | scaffold | 400 | 68.2% | ok | False | 9454.9s |
| grimgar03 | gemma-3-27b | cloud-a6000-lmstudio | lmstudio | 16384 | scaffold_thinking | 400 | 68.2% | ok | False | 9454.9s |
| grimgar03 | gemma-3-27b | cloud-a6000-lmstudio | lmstudio | 16384 | thinking | 400 | 71.2% | ok | False | 9454.9s |
| grimgar03 | magistral-small | cloud-a6000-lmstudio | lmstudio | 16384 | baseline | 400 | 64.0% | ok | False | 8256.2s |
| grimgar03 | magistral-small | cloud-a6000-lmstudio | lmstudio | 16384 | because | 400 | 61.8% | ok | False | 8256.2s |
| grimgar03 | magistral-small | cloud-a6000-lmstudio | lmstudio | 16384 | scaffold | 400 | 52.0% | ok | False | 8256.2s |
| grimgar03 | magistral-small | cloud-a6000-lmstudio | lmstudio | 16384 | scaffold_thinking | 400 | 52.0% | ok | False | 8256.2s |
| grimgar03 | magistral-small | cloud-a6000-lmstudio | lmstudio | 16384 | thinking | 400 | 63.0% | ok | False | 8256.2s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | baseline | 400 | 56.5% | ok | True | 8588.7s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | because | 400 | 54.8% | ok | True | 8588.7s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | scaffold | 400 | 52.5% | ok | True | 8588.7s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | scaffold_thinking | 400 | 56.5% | ok | True | 8588.7s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | thinking | 400 | 66.2% | ok | True | 8588.7s |
| grimgar03 | qwen3-32b | cloud-a6000-lmstudio | lmstudio | 16384 | baseline | 400 | 67.2% | ok | False | 21565.9s |
| grimgar03 | qwen3-32b | cloud-a6000-lmstudio | lmstudio | 16384 | because | 400 | 67.5% | ok | False | 21565.9s |
| grimgar03 | qwen3-32b | cloud-a6000-lmstudio | lmstudio | 16384 | scaffold | 400 | 57.8% | ok | False | 21565.9s |
| grimgar03 | qwen3-32b | cloud-a6000-lmstudio | lmstudio | 16384 | scaffold_thinking | 400 | 68.5% | ok | False | 21565.9s |
| grimgar03 | qwen3-32b | cloud-a6000-lmstudio | lmstudio | 16384 | thinking | 400 | 72.2% | ok | False | 21565.9s |
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | baseline | 139 | 39.6% | ok | True | 5021.5s |
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | because | 139 | 50.4% | ok | True | 5021.5s |
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | scaffold | 139 | 41.0% | ok | True | 5021.5s |
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | scaffold_thinking | 139 | 48.2% | ok | True | 5021.5s |
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | thinking | 139 | 41.7% | ok | True | 5021.5s |

## reasoning_check

| book | model | env | backend | ctx | arm | n | acc | valid | dirty | elapsed |
|---|---|---|---|---:|---|---:|---:|---|---|---:|
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | plain | 396 | 58.1% | ok | False | 3748.4s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | reasoned | 396 | 59.3% | ok | False | 3748.4s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | plain | 139 | 51.1% | ok | False | 1970.5s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | reasoned | 139 | 48.2% | ok | False | 1970.5s |

## reexamine

| book | model | env | backend | ctx | arm | n | acc | valid | dirty | elapsed |
|---|---|---|---|---:|---|---:|---:|---|---|---:|
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | baseline | 139 | 49.6% | ok | True | 2737.5s |
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | narration | 139 | 34.5% | ok | True | 2737.5s |
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | narrator | 139 | 51.8% | ok | True | 2737.5s |
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | prose | 139 | 47.5% | ok | True | 2737.5s |
| mushoku16 | qwen3-14b | local-lmstudio | lmstudio | 16384 | voting | 139 | 49.6% | ok | True | 2737.5s |

## roster_warmup

| book | model | env | backend | ctx | arm | n | acc | valid | dirty | elapsed |
|---|---|---|---|---:|---|---:|---:|---|---|---:|
| mushoku16 | ministral-3-14b-instruct-2 | local-lmstudio | lmstudio | 16384 | incremental | 139 | 41.0% | ok | False | 8408.8s |
| mushoku16 | ministral-3-14b-instruct-2 | local-lmstudio | lmstudio | 16384 | oracle | 139 | 46.8% | ok | False | 8408.8s |
| mushoku16 | ministral-3-14b-instruct-2 | local-lmstudio | lmstudio | 16384 | warm | 139 | 44.6% | ok | False | 8408.8s |
| mushoku16 | qwen3.5-9b-uncensored-hauh | local-lmstudio | lmstudio | 32768 | incremental | 139 | 27.3% | ok | False | 1039.6s |
| mushoku16 | qwen3.5-9b-uncensored-hauh | local-lmstudio | lmstudio | 32768 | oracle | 139 | 35.3% | ok | False | 1039.6s |
| mushoku16 | qwen3.5-9b-uncensored-hauh | local-lmstudio | lmstudio | 32768 | warm | 139 | 32.4% | ok | False | 1039.6s |

## segmentation_crossover

| book | model | env | backend | ctx | arm | n | acc | valid | dirty | elapsed |
|---|---|---|---|---:|---|---:|---:|---|---|---:|
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | seg=gemma,attr=gemma,t=0.0,rep=1 | 399 | 58.6% | ok | False | 1832.5s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | seg=gemma,attr=gemma,t=0.0,rep=2 | 399 | 58.6% | ok | False | 1832.5s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | seg=gemma,attr=gemma,t=0.6,rep=1 | 399 | 57.1% | ok | False | 1832.5s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | seg=gemma,attr=gemma,t=0.6,rep=2 | 399 | 57.9% | ok | False | 1832.5s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | seg=gemma,attr=gemma,t=0.6,rep=3 | 399 | 59.4% | ok | False | 1832.5s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | seg=gemma,attr=qwen,t=0.0,rep=1 | 399 | 60.9% | ok | False | 1832.5s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | seg=gemma,attr=qwen,t=0.0,rep=2 | 399 | 60.9% | ok | False | 1832.5s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | seg=gemma,attr=qwen,t=0.6,rep=1 | 399 | 60.7% | ok | False | 1832.5s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | seg=gemma,attr=qwen,t=0.6,rep=2 | 399 | 60.7% | ok | False | 1832.5s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | seg=gemma,attr=qwen,t=0.6,rep=3 | 399 | 59.9% | ok | False | 1832.5s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | seg=qwen,attr=gemma,t=0.0,rep=1 | 399 | 56.6% | ok | False | 1832.5s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | seg=qwen,attr=gemma,t=0.0,rep=2 | 399 | 56.6% | ok | False | 1832.5s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | seg=qwen,attr=gemma,t=0.6,rep=1 | 399 | 55.9% | ok | False | 1832.5s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | seg=qwen,attr=gemma,t=0.6,rep=2 | 399 | 56.6% | ok | False | 1832.5s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | seg=qwen,attr=gemma,t=0.6,rep=3 | 399 | 57.4% | ok | False | 1832.5s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | seg=qwen,attr=qwen,t=0.0,rep=1 | 399 | 58.4% | ok | False | 1832.5s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | seg=qwen,attr=qwen,t=0.0,rep=2 | 399 | 58.4% | ok | False | 1832.5s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | seg=qwen,attr=qwen,t=0.6,rep=1 | 399 | 58.6% | ok | False | 1832.5s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | seg=qwen,attr=qwen,t=0.6,rep=2 | 399 | 57.9% | ok | False | 1832.5s |
| grimgar03 | qwen3-14b | local-lmstudio | lmstudio | 16384 | seg=qwen,attr=qwen,t=0.6,rep=3 | 399 | 58.1% | ok | False | 1832.5s |

## tag_priority

| book | model | env | backend | ctx | arm | n | acc | valid | dirty | elapsed |
|---|---|---|---|---:|---|---:|---:|---|---|---:|
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | baseline | 400 | 56.5% | ok | True | 1887.0s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | tagfirst | 400 | 63.0% | ok | True | 1887.0s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | baseline | 139 | 51.1% | ok | True | 1073.2s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | tagfirst | 139 | 45.3% | ok | True | 1073.2s |

## two_by_two

| book | model | env | backend | ctx | arm | n | acc | valid | dirty | elapsed |
|---|---|---|---|---:|---|---:|---:|---|---|---:|
| mushoku16 | qwen3.5-9b-uncensored-hauh | local-lmstudio | lmstudio |  | A | 139 | 19.4% | None | True | 698.2s |
| mushoku16 | qwen3.5-9b-uncensored-hauh | local-lmstudio | lmstudio |  | B | 139 | 2.2% | None | True | 698.2s |
| mushoku16 | qwen3.5-9b-uncensored-hauh | local-lmstudio | lmstudio |  | C | 139 | 34.5% | None | True | 698.2s |
| mushoku16 | qwen3.5-9b-uncensored-hauh | local-lmstudio | lmstudio |  | D | 139 | 18.7% | None | True | 698.2s |

## voting

| book | model | env | backend | ctx | arm | n | acc | valid | dirty | elapsed |
|---|---|---|---|---:|---|---:|---:|---|---|---:|
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | greedy | 400 | 55.8% | ok | False | 6517.4s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | vote3 | 400 | 58.0% | ok | False | 6517.4s |
| grimgar03 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | vote5 | 400 | 57.8% | ok | False | 6517.4s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | greedy | 139 | 47.5% | ok | False | 3221.3s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | vote3 | 139 | 48.9% | ok | False | 3221.3s |
| mushoku16 | qwen3-14b | local-llamacpp-hip | llama.cpp-hip | 16384 | vote5 | 139 | 48.9% | ok | False | 3221.3s |
