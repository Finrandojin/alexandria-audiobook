# Turning on the attribution adapter

The measured gain, on the model this project ships (qwen3-14b):

| | | |
| --- | --- | --- |
| `distill_eval__pdnc_only` | **+9.8** cross-book | positive on all four gold books; disagreement 149-73 |
| `lora_serving_eval__mixed-shippable` | **+14.6** | 58.9% -> 73.5%, Q4_K_M base + f16 LoRA through llama.cpp |

Both artifacts carry `validation=ok`. For scale: repeated runs of an identical
configuration spread **1.6-3.7 points**, so this is three to six times the
noise floor - unlike the scene-narrowing arms, whose differences sit inside it.

## It is a serving change, not an application change

The adapter is applied by `llama-server`. Nothing in `app/` computes with it.

```bash
llama-server \
  -m /path/to/Qwen3-14B-Q4_K_M.gguf \
  --lora /path/to/adapter_mixed.gguf \
  --host 127.0.0.1 --port 8090 -c 32768 --parallel 1
```

`ab_test_runtime/distill/gguf/adapter_mixed.gguf` is the one that measured
+14.6. It was downloaded from the Thunder instance before deletion and its
SHA-256 verified against the source.

Then point the app at it, in `app/config.json`, inside the block for the mode
you are running:

```json
"llm_local": {
  "base_url": "http://127.0.0.1:8090/v1",
  "model_name": "qwen/qwen3-14b",
  "attribution_adapter": {
    "path": "ab_test_runtime/distill/gguf/adapter_mixed.gguf",
    "scale": 1.0,
    "require": false
  }
}
```

The adapter lives in the same block as `base_url` on purpose: it belongs to
that endpoint, and a second location could disagree with it about which server
is meant.

## What the check does

`generate_script.py` prints, at startup:

```
LLM adapter: attribution_adapter=adapter_mixed.gguf scale=1.0 require=False
```

and warns if the server is not actually serving it. **This is the point of the
whole thing.** Without it, a server started without `--lora`, or with the scale
toggled to zero, answers every request perfectly happily at base quality.
Nothing fails. The book generates. It is simply 9.8 points worse and no
artifact records why - the same shape as the seed bug, which cost six
contaminated comparisons before somebody noticed by ear.

`require: true` turns the warning into a refusal. The default warns, because a
book half-generated overnight should not die when a server is restarted - but
the run must carry the fact.

## Three honest limits

**The endpoint must be llama.cpp.** LM Studio and Ollama do not implement
`/lora-adapters`, so the check reports "cannot verify" rather than pretending
the adapter is absent. That is deliberate: a warning that fires on every setup
is one nobody reads.

**Scale is not tuned.** 1.0 is what the eval used. `POST /lora-adapters` can
set it anywhere, and nothing here has measured whether 0.7 or 1.3 is better.

**The gain is measured on light novels.** The four gold books are Japanese
light novels in English translation. Nothing measures whether it helps or hurts
on other material, and the training data was English public-domain classics -
so transfer in the other direction is unmeasured, not assumed absent.
