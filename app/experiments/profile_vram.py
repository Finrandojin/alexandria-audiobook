"""Measure real VRAM cost per model so _VERIFIED_LOCAL_PROFILES entries are
derived from this machine, not guessed from on-disk size.

Without a verified profile, ensure_ideal_settings falls back to the static
8192-token default. Comparing a model loaded at 8192 against one at 16384 or
32768 confounds capability with available context, so any new model must be
profiled before it is benchmarked.

Set PROFILE_MODELS to a comma-separated list to profile specific models.

For each model: load at the 8192 baseline and at 32768, reading live VRAM after
each. model_vram_bytes comes from the baseline load; the delta across the extra
24576 tokens gives bytes_per_extra_context_token.
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lmstudio_settings import get_local_vram_bytes

LMS = os.environ.get("LMS_BIN",
                     os.path.expanduser("~/.lmstudio/bin/lms"))
BASELINE_CONTEXT = 8192
TARGET_CONTEXT = 32768
MODELS = [m for m in os.environ.get("PROFILE_MODELS","").split(",") if m] or [
    "ministral-3-14b-instruct-2512",
    "ministral-3-14b-instruct-2512-absolute-heresy-i1",
    "gemma-4-12b-coder-fable5-composer2.5-v1",
    "qwen3.6-27b-uncensored-hauhaucs-aggressive",
]
GIB = 1024 ** 3


def run(args, timeout=600):
    return subprocess.run([LMS] + args, capture_output=True, text=True, timeout=timeout)


def unload_all():
    run(["unload", "--all"])
    time.sleep(3)


def vram():
    memory = get_local_vram_bytes()
    return memory if memory else (0, 0)


def load(model, context, parallel=1):
    result = run(["load", model, "--context-length", str(context),
                  "--gpu", "max", "-y"])
    if result.returncode != 0:
        return False, (result.stderr or result.stdout).strip()[:200]
    time.sleep(4)
    return True, ""


results = {}
unload_all()
total, idle_used = vram()
print(f"card total={total / GIB:.2f} GiB, idle used={idle_used / GIB:.2f} GiB\n", flush=True)

for model in MODELS:
    print("=" * 72, flush=True)
    print(model, flush=True)
    row = {}
    unload_all()
    _, before = vram()

    ok, err = load(model, BASELINE_CONTEXT)
    if not ok:
        print(f"  FAILED baseline load: {err}", flush=True)
        results[model] = {"error": err}
        continue
    _, after_base = vram()
    model_bytes = after_base - before
    row["model_vram_bytes"] = model_bytes
    print(f"  @{BASELINE_CONTEXT}: used={after_base / GIB:6.2f} GiB "
          f"model={model_bytes / GIB:5.2f} GiB", flush=True)

    unload_all()
    ok, err = load(model, TARGET_CONTEXT)
    if not ok:
        print(f"  32768 load FAILED: {err}", flush=True)
        row["target_ok"] = False
        results[model] = row
        continue
    _, after_target = vram()
    row["target_ok"] = True
    row["target_used_bytes"] = after_target
    extra = after_target - after_base
    per_token = max(0, extra) / (TARGET_CONTEXT - BASELINE_CONTEXT)
    row["bytes_per_extra_context_token"] = per_token
    headroom = total - after_target
    row["headroom_bytes"] = headroom
    print(f"  @{TARGET_CONTEXT}: used={after_target / GIB:6.2f} GiB "
          f"extra={extra / GIB:5.2f} GiB  per_token={per_token / 1024:6.2f} KiB "
          f"headroom={headroom / GIB:5.2f} GiB "
          f"{'OK (>=2 GiB reserve)' if headroom >= 2 * GIB else 'TOO TIGHT'}",
          flush=True)
    results[model] = row

unload_all()
out = ('/home/fakemitch/pinokio/cache/TMPDIR/claude-1000/'
       '-home-fakemitch-pinokio-api-alexandria-audiobook2-git/'
       'e5db5129-c65a-459a-82cf-736dd0a173e7/scratchpad/vram_profiles.json')
json.dump({"total_bytes": total, "models": results}, open(out, "w"), indent=2)
print(f"\nwrote {out}", flush=True)

print("\n\nSUGGESTED _VERIFIED_LOCAL_PROFILES ENTRIES", flush=True)
for model, row in results.items():
    if not row.get("target_ok") or row.get("headroom_bytes", 0) < 2 * GIB:
        print(f'    # {model}: 32768 does not leave the 2 GiB reserve; '
              f'left on the 8192 fallback deliberately.', flush=True)
        continue
    # Round the measured per-token cost up to a safe multiple, mirroring the
    # existing entries' deliberate overestimate.
    per_token_kib = max(16, int(row["bytes_per_extra_context_token"] / 1024) + 4)
    print(f'    "{model}": {{', flush=True)
    print(f'        "context_length": 32768,', flush=True)
    print(f'        "parallel": 1,', flush=True)
    print(f'        "model_vram_bytes": int({row["model_vram_bytes"] / GIB:.2f} * 1024 ** 3),', flush=True)
    print(f'        "bytes_per_extra_context_token": {per_token_kib} * 1024,', flush=True)
    print(f'    }},', flush=True)
