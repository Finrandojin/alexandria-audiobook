# Legacy attribution audit — 2026-08-05

All 110 legacy-metadata artifacts are listed exactly once. Classification describes whether the recorded measurement can be used with today's fixtures; it does not turn accuracy into a product or perceptual conclusion.

## Counts

- `exploratory`: 9
- `historical_only`: 38
- `provisional`: 26
- `supported_measurement`: 37

`historical_only` means current-gold rescoring changes at least one judgment or cannot map at least one row. Original files remain preserved; their saved summaries were not rewritten.

## Per-artifact audit

| artifact | family | class | rows | changed scores | unmapped | dirty | problems |
|---|---|---|---:|---:|---:|---|---|
| `batch_contiguity__owarimonogatari3__qwen__qwen3-14b__local-rocm-contig.json` | batch_contiguity | supported_measurement | 324 | 0 | 0 | False |  |
| `batch_size__grimgar03__qwen__qwen3-14b__local-llamacpp-wide.json` | batch_size | historical_only | 1200 | 41 | 12 | True |  |
| `batch_size__grimgar03__qwen__qwen3-14b__local-llamacpp.json` | batch_size | historical_only | 1600 | 27 | 16 | False |  |
| `batch_size__index18__qwen__qwen3-14b__local-llamacpp.json` | batch_size | supported_measurement | 396 | 0 | 0 | False |  |
| `batch_size__mushoku16__qwen__qwen3-14b__local-llamacpp-wide.json` | batch_size | provisional | 417 | 0 | 0 | True |  |
| `batch_size__mushoku16__qwen__qwen3-14b__local-llamacpp.json` | batch_size | provisional | 556 | 0 | 0 | False |  |
| `batch_size__owarimonogatari3__qwen__qwen3-14b__local-llamacpp-rep1.json` | batch_size | provisional | 486 | 0 | 0 | True |  |
| `batch_size__owarimonogatari3__qwen__qwen3-14b__local-llamacpp-rep2.json` | batch_size | provisional | 486 | 0 | 0 | True |  |
| `batch_size__owarimonogatari3__qwen__qwen3-14b__local-llamacpp.json` | batch_size | supported_measurement | 648 | 0 | 0 | False |  |
| `because_production__grimgar03__qwen__qwen3-14b__local.json` | because_production | historical_only | 800 | 33 | 8 | False |  |
| `because_production__mushoku16__qwen__qwen3-14b__local.json` | because_production | provisional | 278 | 0 | 0 | False |  |
| `because_production__qwen__qwen3-14b.json` | because_production | provisional | 417 | 0 | 0 | True |  |
| `candidate_id__qwen__qwen3-14b.json` | candidate_id | historical_only | 294 | 1 | 0 | False |  |
| `closed_set.json` | closed_set | exploratory | 441 | 2 | 0 | True | artifact validation is not ok; environment is missing context_length; environment is missing parallel; no LM Studio load state recorded; no harness fingerprint: the code that ran is unidentified |
| `closed_set__gemma-4-e4b-uncensored-hauhaucs-aggressive.json` | closed_set | exploratory | 441 | 0 | 0 | False | recorded commit is unavailable from current history |
| `closed_set__grimgar03__gemma-4-e4b-uncensored-hauhaucs-aggressive.json` | closed_set | historical_only | 1200 | 50 | 12 | False |  |
| `closed_set__grimgar03__google__gemma-3-27b__thunder-a6000.json` | closed_set | historical_only | 1200 | 54 | 12 | False |  |
| `closed_set__grimgar03__ministral-3-14b-instruct-2512.json` | closed_set | historical_only | 1200 | 52 | 12 | False |  |
| `closed_set__grimgar03__mistralai__magistral-small__local-llamacpp.json` | closed_set | supported_measurement | 1188 | 0 | 0 | False |  |
| `closed_set__grimgar03__mistralai__magistral-small__thunder-a6000.json` | closed_set | historical_only | 1200 | 58 | 12 | False |  |
| `closed_set__grimgar03__qwen__qwen3-14b.CLOUD-a6000.json` | closed_set | historical_only | 1200 | 53 | 12 | False |  |
| `closed_set__grimgar03__qwen__qwen3-14b.LOCAL-9070xt.json` | closed_set | historical_only | 1200 | 53 | 12 | False |  |
| `closed_set__grimgar03__qwen__qwen3-14b.json` | closed_set | historical_only | 1200 | 53 | 12 | False |  |
| `closed_set__grimgar03__qwen__qwen3-14b__local-llamacpp-regold.json` | closed_set | supported_measurement | 1188 | 0 | 0 | False |  |
| `closed_set__grimgar03__qwen__qwen3-32b__thunder-a6000.json` | closed_set | historical_only | 1200 | 59 | 12 | False |  |
| `closed_set__index18__mistralai__magistral-small__local-llamacpp.json` | closed_set | supported_measurement | 297 | 0 | 0 | False |  |
| `closed_set__index18__qwen__qwen3-14b__local-llamacpp-regold.json` | closed_set | supported_measurement | 297 | 0 | 0 | False |  |
| `closed_set__microsoft__phi-4.json` | closed_set | exploratory | 441 | 0 | 0 | False | recorded commit is unavailable from current history |
| `closed_set__ministral-3-14b-instruct-2512-absolute-heresy-i1.json` | closed_set | exploratory | 441 | 2 | 0 | False | recorded commit is unavailable from current history |
| `closed_set__ministral-3-14b-instruct-2512.json` | closed_set | exploratory | 441 | 3 | 0 | False | recorded commit is unavailable from current history |
| `closed_set__mushoku16__google__gemma-3-27b__thunder-a6000.json` | closed_set | historical_only | 417 | 3 | 0 | False |  |
| `closed_set__mushoku16__mistralai__magistral-small__local-llamacpp.json` | closed_set | historical_only | 417 | 1 | 0 | False |  |
| `closed_set__mushoku16__mistralai__magistral-small__thunder-a6000.json` | closed_set | historical_only | 417 | 1 | 0 | False |  |
| `closed_set__mushoku16__qwen__qwen3-14b.json` | closed_set | historical_only | 417 | 2 | 0 | False |  |
| `closed_set__mushoku16__qwen__qwen3-14b__local-llamacpp-regold.json` | closed_set | supported_measurement | 408 | 0 | 0 | False |  |
| `closed_set__mushoku16__qwen__qwen3-14b__local-llamacpp.json` | closed_set | historical_only | 417 | 2 | 0 | False |  |
| `closed_set__mushoku16__qwen__qwen3-14b__local-lmstudio.json` | closed_set | historical_only | 417 | 2 | 0 | False |  |
| `closed_set__mushoku16__qwen__qwen3-14b__local-vulkan.json` | closed_set | historical_only | 417 | 2 | 0 | False |  |
| `closed_set__mushoku16__qwen__qwen3-32b__thunder-a6000.json` | closed_set | historical_only | 417 | 3 | 0 | False |  |
| `closed_set__owarimonogatari3__mistralai__magistral-small__local-llamacpp.json` | closed_set | supported_measurement | 486 | 0 | 0 | False |  |
| `closed_set__owarimonogatari3__qwen__qwen3-14b__local-llamacpp-regold.json` | closed_set | supported_measurement | 486 | 0 | 0 | False |  |
| `closed_set__qwen3.5-9b-uncensored-hauhaucs-aggressive.json` | closed_set | exploratory | 441 | 2 | 0 | False | recorded commit is unavailable from current history |
| `closed_set__qwen__qwen3-14b.json` | closed_set | exploratory | 441 | 2 | 0 | False | recorded commit is unavailable from current history |
| `committed_history__grimgar03__qwen__qwen3-14b__local-llamacpp.json` | committed_history | historical_only | 1200 | 43 | 12 | False |  |
| `committed_history__index18__qwen__qwen3-14b__local-llamacpp.json` | committed_history | supported_measurement | 297 | 0 | 0 | False |  |
| `committed_history__mushoku16__qwen__qwen3-14b__local-llamacpp.json` | committed_history | supported_measurement | 408 | 0 | 0 | False |  |
| `committed_history__owarimonogatari3__qwen__qwen3-14b__local-llamacpp.json` | committed_history | supported_measurement | 486 | 0 | 0 | False |  |
| `context_width__grimgar03__qwen__qwen3-14b__local-llamacpp.json` | context_width | historical_only | 1600 | 49 | 16 | False |  |
| `context_width__index18__qwen__qwen3-14b__local-llamacpp.json` | context_width | supported_measurement | 297 | 0 | 0 | False |  |
| `context_width__mushoku16__qwen__qwen3-14b__local-llamacpp.json` | context_width | supported_measurement | 408 | 0 | 0 | False |  |
| `context_width__owarimonogatari3__qwen__qwen3-14b__local-llamacpp.json` | context_width | supported_measurement | 486 | 0 | 0 | False |  |
| `context_width_production__grimgar03__qwen__qwen3-14b__local-llamacpp-rep1.json` | context_width_production | historical_only | 800 | 26 | 8 | False |  |
| `context_width_production__grimgar03__qwen__qwen3-14b__local-llamacpp-rep2.json` | context_width_production | historical_only | 800 | 23 | 8 | False |  |
| `context_width_production__grimgar03__qwen__qwen3-14b__local-llamacpp-rep3.json` | context_width_production | historical_only | 800 | 23 | 8 | False |  |
| `context_width_production__grimgar03__qwen__qwen3-14b__local.json` | context_width_production | historical_only | 800 | 26 | 8 | False |  |
| `context_width_production__index18__qwen__qwen3-14b__local-llamacpp.json` | context_width_production | supported_measurement | 198 | 0 | 0 | False |  |
| `context_width_production__mushoku16__qwen__qwen3-14b__local.json` | context_width_production | provisional | 278 | 0 | 0 | False |  |
| `context_width_production__owarimonogatari3__qwen__qwen3-14b__local-llamacpp.json` | context_width_production | supported_measurement | 324 | 0 | 0 | False |  |
| `crossover__grimgar03__local.json` | segmentation_crossover | historical_only | 7980 | 320 | 60 | False |  |
| `grammar_constraint__grimgar03__qwen__qwen3-14b__local-llamacpp.json` | grammar_constraint | supported_measurement | 1584 | 0 | 0 | False |  |
| `grammar_constraint__index18__qwen__qwen3-14b__local-llamacpp.json` | grammar_constraint | supported_measurement | 396 | 0 | 0 | False |  |
| `grammar_constraint__mushoku16__mistralai__magistral-small__local-llamacpp.json` | grammar_constraint | provisional | 556 | 0 | 0 | False |  |
| `grammar_constraint__owarimonogatari3__qwen__qwen3-14b__local-llamacpp.json` | grammar_constraint | supported_measurement | 648 | 0 | 0 | False |  |
| `joint_scene__grimgar03__qwen__qwen3-14b__local-llamacpp.json` | joint_scene | historical_only | 1200 | 30 | 12 | False |  |
| `joint_scene__index18__qwen__qwen3-14b__local-llamacpp.json` | joint_scene | supported_measurement | 297 | 0 | 0 | False |  |
| `joint_scene__mushoku16__qwen__qwen3-14b__local-llamacpp.json` | joint_scene | historical_only | 417 | 2 | 0 | False |  |
| `joint_scene__owarimonogatari3__qwen__qwen3-14b__local-llamacpp.json` | joint_scene | supported_measurement | 486 | 0 | 0 | False |  |
| `lora_serving_eval__local-rocm-lora-b2.json` | lora_serving_eval | historical_only | 450 | 0 | 266 | False |  |
| `lora_serving_eval__local-rocm-lora.json` | lora_serving_eval | historical_only | 1094 | 0 | 324 | True |  |
| `narrator_prior__grimgar03__qwen__qwen3-14b__local-llamacpp.json` | narrator_prior | historical_only | 800 | 29 | 8 | False |  |
| `narrator_prior__index18__qwen__qwen3-14b__local-llamacpp.json` | narrator_prior | supported_measurement | 198 | 0 | 0 | False |  |
| `narrator_prior__mushoku16__qwen__qwen3-14b__local-llamacpp.json` | narrator_prior | provisional | 278 | 0 | 0 | False |  |
| `narrator_prior__owarimonogatari3__qwen__qwen3-14b__local-llamacpp.json` | narrator_prior | supported_measurement | 324 | 0 | 0 | False |  |
| `reasoning_arms__grimgar03__google__gemma-3-27b__thunder-a6000.json` | reasoning_arms | historical_only | 2000 | 111 | 20 | False |  |
| `reasoning_arms__grimgar03__qwen__qwen3-14b.json` | reasoning_arms | historical_only | 2000 | 79 | 20 | True |  |
| `reasoning_arms__index18__qwen__qwen3-14b__local-llamacpp.json` | reasoning_arms | supported_measurement | 495 | 0 | 0 | False |  |
| `reasoning_arms__owarimonogatari3__qwen__qwen3-14b__local-llamacpp.json` | reasoning_arms | supported_measurement | 810 | 0 | 0 | False |  |
| `reasoning_arms__qwen__qwen3-14b.json` | reasoning_arms | historical_only | 695 | 4 | 0 | True |  |
| `reasoning_check__grimgar03__qwen__qwen3-14b__local-llamacpp.json` | reasoning_check | historical_only | 792 | 1 | 0 | False |  |
| `reasoning_check__index18__qwen__qwen3-14b__local-llamacpp.json` | reasoning_check | supported_measurement | 198 | 0 | 0 | False |  |
| `reasoning_check__mushoku16__qwen__qwen3-14b__local-llamacpp.json` | reasoning_check | supported_measurement | 278 | 0 | 0 | False |  |
| `reasoning_check__owarimonogatari3__qwen__qwen3-14b__local-llamacpp.json` | reasoning_check | supported_measurement | 324 | 0 | 0 | False |  |
| `reexamine__qwen__qwen3-14b.json` | reexamine | provisional | 695 | 0 | 0 | True |  |
| `roster_quality__grimgar03__qwen__qwen3-14b__local-llamacpp.json` | roster_quality | provisional | 1540 | 0 | 0 | True |  |
| `roster_quality__index18__qwen__qwen3-14b__local-llamacpp.json` | roster_quality | provisional | 368 | 0 | 0 | True |  |
| `roster_quality__mushoku16__qwen__qwen3-14b__local-llamacpp.json` | roster_quality | provisional | 532 | 0 | 0 | True |  |
| `roster_quality__owarimonogatari3__qwen__qwen3-14b__local-llamacpp.json` | roster_quality | provisional | 648 | 0 | 0 | True |  |
| `roster_warmup.json` | roster_warmup | exploratory | 417 | 0 | 0 | False | recorded commit is unavailable from current history |
| `roster_warmup__ministral-3-14b-instruct-2512.json` | roster_warmup | provisional | 417 | 0 | 0 | False |  |
| `scene_cast__grimgar03__qwen__qwen3-14b__local-llamacpp-look1.json` | scene_cast | provisional | 1155 | 0 | 0 | True |  |
| `scene_cast__grimgar03__qwen__qwen3-14b__local-llamacpp-look6.json` | scene_cast | provisional | 1155 | 0 | 0 | True |  |
| `scene_cast__grimgar03__qwen__qwen3-14b__local-llamacpp.json` | scene_cast | provisional | 1155 | 0 | 0 | True |  |
| `scene_cast__index18__qwen__qwen3-14b__local-llamacpp.json` | scene_cast | provisional | 276 | 0 | 0 | True |  |
| `scene_cast__mushoku16__qwen__qwen3-14b__local-llamacpp-look6.json` | scene_cast | provisional | 399 | 0 | 0 | True |  |
| `scene_cast__mushoku16__qwen__qwen3-14b__local-llamacpp.json` | scene_cast | provisional | 399 | 0 | 0 | True |  |
| `scene_cast__owarimonogatari3__qwen__qwen3-14b__local-llamacpp.json` | scene_cast | provisional | 486 | 0 | 0 | True |  |
| `tag_priority__grimgar03__mistralai__magistral-small__local-llamacpp.json` | tag_priority | provisional | 792 | 0 | 0 | True |  |
| `tag_priority__grimgar03__qwen__qwen3-14b__local-llamacpp-rep1.json` | tag_priority | supported_measurement | 792 | 0 | 0 | False |  |
| `tag_priority__grimgar03__qwen__qwen3-14b__local-llamacpp-rep2.json` | tag_priority | supported_measurement | 792 | 0 | 0 | False |  |
| `tag_priority__grimgar03__qwen__qwen3-14b__local-llamacpp-rep3.json` | tag_priority | supported_measurement | 792 | 0 | 0 | False |  |
| `tag_priority__grimgar03__qwen__qwen3-14b__local-llamacpp.json` | tag_priority | historical_only | 800 | 22 | 8 | True |  |
| `tag_priority__index18__mistralai__magistral-small__local-llamacpp.json` | tag_priority | provisional | 198 | 0 | 0 | True |  |
| `tag_priority__index18__qwen__qwen3-14b__local-llamacpp.json` | tag_priority | supported_measurement | 198 | 0 | 0 | False |  |
| `tag_priority__mushoku16__qwen__qwen3-14b__local-llamacpp.json` | tag_priority | provisional | 278 | 0 | 0 | True |  |
| `tag_priority__owarimonogatari3__qwen__qwen3-14b__local-llamacpp.json` | tag_priority | supported_measurement | 324 | 0 | 0 | False |  |
| `two_by_two.json` | two_by_two | exploratory | 556 | 0 | 556 | True | artifact validation is not ok; environment is missing context_length; environment is missing parallel; no LM Studio load state recorded; no harness fingerprint: the code that ran is unidentified |
| `voting__grimgar03__qwen__qwen3-14b__local-llamacpp.json` | voting | historical_only | 1200 | 30 | 12 | False |  |
| `voting__index18__qwen__qwen3-14b__local-llamacpp.json` | voting | supported_measurement | 198 | 0 | 0 | False |  |
| `voting__mushoku16__qwen__qwen3-14b__local-llamacpp.json` | voting | provisional | 417 | 0 | 0 | False |  |
| `voting__owarimonogatari3__qwen__qwen3-14b__local-llamacpp.json` | voting | supported_measurement | 324 | 0 | 0 | False |  |

## Family-level interpretation limits

- `batch_contiguity`: Isolates companion ordering, not end-to-end production quality.
- `batch_size`: Accuracy and throughput must be considered together; books differ.
- `because_production`: A justification field test; explanations are not confidence estimates.
- `candidate_id`: One model/corpus comparison; opaque IDs do not prove general naming gains.
- `closed_set`: Oracle candidate arms are invalid for current claims because their lists used superseded labels.
- `committed_history`: Oracle history is an upper bound and is not shippable state.
- `context_width`: A harness diagnostic; production-path confirmation is separate.
- `context_width_production`: Book-specific repeats; report each book/repeat rather than pooling.
- `grammar_constraint`: Roster-valid output does not establish correct speaker identity.
- `joint_scene`: Joint and shuffled controls answer ordering only within the tested fixtures.
- `lora_serving_eval`: Two gold books and one serving stack; not a universal adapter claim.
- `narrator_prior`: A predeclared book-contrast test, not a general narrator rule.
- `reasoning_arms`: Reasoning/justification settings are model- and serving-stack-specific.
- `reasoning_check`: Justification disagreement is a routing signal, not calibrated confidence.
- `reexamine`: Selected previously negative results; selection prevents broad inference.
- `roster_quality`: Gold-roster arms are upper bounds and not deployable inputs.
- `roster_warmup`: Book-quartile diagnostic; oracle roster is not deployable.
- `scene_cast`: Scene-cast extraction and attribution effects cannot be conflated.
- `segmentation_crossover`: Factorial diagnostic on one book; retain repeat-level uncertainty.
- `tag_priority`: Prompt rule effects vary by book/model and require per-book reporting.
- `two_by_two`: The two factors are not independent; this prices context, not batching.
- `voting`: Voting cost and routing coverage accompany accuracy; no pooled policy claim.
