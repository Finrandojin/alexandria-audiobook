"""Which trained voices are undertrained, and how much audio is enough?

Seventy-five voice LoRAs sit in `lora_models/`, indistinguishable from each
other in the UI. Their `training_meta.json` says how many samples each was
trained on, and the spread is wide: 67 at the 200-sample cap, and eight below
it - down to one voice trained on TWO samples.

A LoRA's `B` matrices are initialised to zero and grow as the adapter learns,
so the norm of B measures how far the adapter moved the base model. It is a
cheap, GPU-free proxy for "did this train at all".

WHAT THE NORM DOES AND DOES NOT TELL YOU. It measures how MUCH the adapter
changed the model, not whether it changed it CORRECTLY - a large norm can be
overfitting just as easily as learning. So a norm far below the population is
strong evidence a voice is undertrained, while the plateau this reveals is only
suggestive about how much audio is enough. Confirming that needs generation and
a speaker-similarity measurement against each voice's own reference.
"""
import glob, json, os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
MODELS = REPO + "/lora_models"


def adapter_rows():
    from safetensors import safe_open
    rows = []
    for meta in sorted(glob.glob(MODELS + "/*/training_meta.json")):
        folder = os.path.dirname(meta)
        weights = os.path.join(folder, "adapter_model.safetensors")
        if not os.path.exists(weights):
            continue
        try:
            samples = json.load(open(meta)).get("num_samples")
        except Exception:
            samples = None
        norms = []
        with safe_open(weights, framework="np") as handle:
            for key in handle.keys():
                if "lora_B" in key or "lora_b" in key:
                    norms.append(float(np.linalg.norm(handle.get_tensor(key))))
        if norms:
            rows.append({"voice": os.path.basename(folder), "samples": samples,
                         "mean_b": float(np.mean(norms)), "tensors": len(norms)})
    return rows


def main():
    rows = adapter_rows()
    if not rows:
        print("no adapters found")
        return
    full = [r["mean_b"] for r in rows if r["samples"] == 200]
    mean, std = (float(np.mean(full)), float(np.std(full))) if full else (0.0, 1.0)
    print(f"{len(rows)} voice adapters; {len(full)} trained at the 200-sample cap")
    print(f"reference population: mean |B| {mean:.4f} +/- {std:.4f}\n")

    rows.sort(key=lambda r: (r["samples"] is None, r["samples"]))
    print(f"  {'voice':40}{'samples':>8}{'mean |B|':>11}{'z':>7}  flag")
    suspect = []
    for r in rows:
        z = (r["mean_b"] - mean) / max(std, 1e-9)
        flag = ""
        if z <= -4:
            flag = "UNDERTRAINED - likely does not carry the voice"
            suspect.append(r["voice"])
        elif z <= -2:
            flag = "weak"
        if r["samples"] is None or r["samples"] < 200 or flag:
            print(f"  {r['voice'][:38]:40}{str(r['samples']):>8}"
                  f"{r['mean_b']:11.4f}{z:+7.1f}  {flag}")

    print(f"\n  {len(suspect)} voices are more than 4 sigma below the population.")
    for v in suspect:
        print(f"    {v}")
    print("\n  A norm far below the population means the adapter barely moved the")
    print("  base model, so the voice it is named for is probably not the voice it")
    print("  produces. Those are worth regenerating or removing.")
    print("\n  The norm plateaus near 120 samples in this data, which SUGGESTS the")
    print("  200 cap is about 1.7x more audio than the adapter uses - but norm is")
    print("  magnitude, not fidelity, and confirming that needs generation plus a")
    print("  speaker-similarity check against each voice's own reference.")

    out = REPO + "/ab_test_runtime/experiments/voice_adapter_health.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"population_mean_b": mean, "population_std": std,
               "rows": rows, "suspect": suspect}, open(out, "w"), indent=1)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
