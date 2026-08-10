"""Benchmark an OpenAI-compatible LLM endpoint to find the concurrency level
(simultaneous in-flight requests) where throughput stops improving.

Single-request decoding leaves most GPUs mostly idle - the fix is sending
multiple requests at once, not making one request faster. This module finds
how many is "at once" for a given endpoint/model/hardware combination, so
review_script.py and find_nicknames.py can use real concurrency instead of
sending batches one at a time.

Standalone use: `python llm_bench.py --model <name> [--base-url ...]`
"""
import argparse
import platform
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

import lmstudio_settings
from config_settings import load_app_config
from utils import atomic_json_write, file_lock

_BENCH_SYSTEM_PROMPT = "You are a creative writing assistant."

# A few hundred words of representative dialogue-shaped text, repeated to land
# around ~4000 prompt tokens - matching the real review-batch prompt size
# measured in production (~3900-4200 prompt tokens).
_BENCH_PASSAGE = (
    "The room had gone quiet. Elena stood by the window, arms folded, "
    "watching the last light drain from the sky. \"Tell me the truth,\" she "
    "said, her voice low and controlled. Marcus could not meet her gaze. "
    "He had always been a poor liar, and they both knew it. \"There is "
    "nothing to tell,\" he said, forcing a calm he did not feel.\n"
)
_BENCH_USER_PROMPT = (
    "Continue this scene for several more paragraphs, in the same style:\n\n"
    + _BENCH_PASSAGE * 40
)

# Production review batches generate ~1500-2000 completion tokens (measured:
# "tokens: prompt=3939 completion=1564" and similar in review_responses.log).
# A short benchmark completion under-counts the GPU's decode-phase cost - a
# server limited to one generation slot at a time can look like it benefits
# from concurrency on a SHORT completion (prefill of the next request can
# overlap the current one's brief decode) while actually being neutral-to-worse
# on the LONG completions production actually sends, once decode dominates.
# Confirmed by a real regression: a 600-token benchmark wrongly picked
# concurrency=2 for a parallel=1 local server, which then ran ~2x slower
# per-batch in production than concurrency=1 - this default fixes that.
_BENCH_MAX_TOKENS = 1600


def _one_call(client, model, max_tokens, timeout):
    t0 = time.time()
    resp = client.with_options(timeout=timeout).chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _BENCH_SYSTEM_PROMPT},
            {"role": "user", "content": _BENCH_USER_PROMPT},
        ],
        max_tokens=max_tokens,
        temperature=0.7,
    )
    usage = getattr(resp, "usage", None)
    completion = getattr(usage, "completion_tokens", None) if usage else None
    if completion is not None:
        return completion, time.time() - t0
    # Some OpenAI-compatible servers omit `usage`; estimate from the reply length
    # so the benchmark doesn't silently settle on concurrency=1 with no signal.
    content = (resp.choices[0].message.content or "") if getattr(resp, "choices", None) else ""
    return max(1, len(content) // 4), time.time() - t0


def measure_throughput(client, model, concurrency, max_tokens=_BENCH_MAX_TOKENS, timeout=180):
    """Fire `concurrency` simultaneous requests against `client`/`model`.

    Returns aggregate completion tokens/sec across the whole concurrent
    group, or None if any request failed or timed out (the caller treats
    that as "this concurrency level isn't safe here").
    """
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(_one_call, client, model, max_tokens, timeout)
                   for _ in range(concurrency)]
        t_start = time.time()
        results = []
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"  [bench] request failed at concurrency={concurrency}: {type(e).__name__}: {e}")
                return None
        wall = time.time() - t_start
    if wall <= 0:
        return None
    total_tokens = sum(tokens for tokens, _ in results)
    return total_tokens / wall


def _measure_robust(client, model, concurrency, trials=3):
    """Median of `trials` measurements at this concurrency level.

    A single sample is too noisy to base a concurrency decision on - it's
    what let a 2x-slower-in-production setting through in testing. Returns
    None if any trial fails (same "not safe at this level" meaning as
    `measure_throughput`).
    """
    results = []
    for _ in range(trials):
        throughput = measure_throughput(client, model, concurrency)
        if throughput is None:
            return None
        results.append(throughput)
    results.sort()
    return results[len(results) // 2]


def find_optimal_concurrency(client, model, max_concurrency=16, trials=3,
                              server_parallel_limit=None):
    """Sweep concurrency 1, 2, 4, 8, 16... and return the level just before
    throughput improvement drops below 10% over the previous level, or before
    a level fails outright. Always returns at least 1.

    `server_parallel_limit`, when known (from get_lmstudio_status /
    get_remote_lmstudio_status), caps how far the sweep goes - a server can't
    truly run more concurrent generations than it's configured for, so
    benchmarking past that just measures queueing noise. If it's <= 1, the
    sweep is skipped entirely and 1 is returned immediately: this is exactly
    what would have caught the parallel=1-server regression instead of
    measuring a noisy "improvement" that didn't hold up in production.
    """
    if server_parallel_limit is not None and server_parallel_limit <= 1:
        return 1
    if server_parallel_limit is not None:
        max_concurrency = min(max_concurrency, server_parallel_limit)

    best_level = 1
    best_throughput = _measure_robust(client, model, 1, trials)
    if best_throughput is None:
        return 1

    level = 2
    while level <= max_concurrency:
        throughput = _measure_robust(client, model, level, trials)
        if throughput is None or throughput < best_throughput * 1.10:
            break
        best_level, best_throughput = level, throughput
        level *= 2
    return best_level


def get_cached_or_benchmarked_concurrency(config_path, llm_mode, base_url, model_name,
                                           client, ssh_alias=None, max_concurrency=16,
                                           status=None):
    """Return a concurrency level for (base_url, model_name).

    Reuses the cached value from config.json's llm_local/llm_remote section
    (whichever `llm_mode` selects) if it was measured against this exact
    base_url+model - logging just a one-line confirmation, no SSH/subprocess
    calls, so re-checking the cache every book in a batch stays cheap.

    On a cache miss, logs an environment fingerprint (hostname, GPU
    name/backend, LM Studio's actual loaded parallel/context) *before*
    benchmarking, so the decision that follows is traceable instead of a bare
    number with no explanation - this is what would have made the
    parallel=1-server regression obvious immediately instead of requiring a
    manual investigation. The server's reported `parallel` value caps the
    benchmark sweep (see find_optimal_concurrency); persists the result back
    to config.json under `file_lock`, since review_script.py/find_nicknames.py
    run as separate subprocesses that could race the main app or each other.

    `status`: an already-fetched get_lmstudio_status/get_remote_lmstudio_status
    result, if the caller already has one (e.g. from a just-run self-heal
    check via ensure_ideal_settings) - avoids a redundant SSH/subprocess
    round-trip on a cache miss.
    """
    is_remote = lmstudio_settings.is_remote_llm(llm_mode, base_url)
    profile_key = "llm_remote" if is_remote else "llm_local"
    cache_key = f"{base_url}::{model_name}"

    config = load_app_config(config_path)
    profile = config.get(profile_key) or {}
    if profile.get("concurrency_for") == cache_key and profile.get("concurrency"):
        concurrency = profile["concurrency"]
        print(f"  Using cached concurrency={concurrency} for {base_url}::{model_name}")
        return concurrency

    hostname = platform.node()
    if status is None:
        if is_remote:
            status = lmstudio_settings.get_remote_lmstudio_status(ssh_alias, model_name)
        else:
            status = lmstudio_settings.get_lmstudio_status(model_name)
    if is_remote:
        gpu_name, backend = lmstudio_settings.get_remote_gpu_name_and_backend(ssh_alias)
        where = f"remote via '{ssh_alias}'" if ssh_alias else "remote (no SSH alias configured)"
    else:
        gpu_name, backend = lmstudio_settings.get_gpu_name_and_backend()
        where = f"local ({hostname})"
    print(f"  Environment: {where} | GPU: {gpu_name or 'unknown'} ({backend or 'unknown backend'}) "
          f"| LM Studio: available={status.get('available')} loaded={status.get('loaded')} "
          f"parallel={status.get('parallel')} context={status.get('context_length')}")

    server_parallel = status.get("parallel") if status.get("available") else None
    if not status.get("available"):
        print("  Server status unknown (unreachable or no SSH alias configured) - "
              "defaulting to concurrency=1 rather than guessing at a safe limit.")
        concurrency = 1
    elif server_parallel is not None and server_parallel <= 1:
        print(f"  Server reports parallel={server_parallel} - concurrency can't help here "
              f"(this is the local VRAM-safe default unless changed), skipping benchmark.")
        concurrency = 1
    else:
        effective_max = min(max_concurrency, server_parallel) if server_parallel else max_concurrency
        print(f"  Benchmarking {model_name} at {base_url} to find optimal concurrency "
              f"(cap={effective_max})...")
        concurrency = find_optimal_concurrency(client, model_name, effective_max,
                                                server_parallel_limit=server_parallel)
        print(f"  Optimal concurrency: {concurrency}")

    try:
        with file_lock(config_path):
            config = load_app_config(config_path)
            profile = dict(config.get(profile_key) or {})
            profile["concurrency"] = concurrency
            profile["concurrency_for"] = cache_key
            config[profile_key] = profile
            atomic_json_write(config, config_path)
    except (OSError, TimeoutError, TypeError, ValueError):
        # Best-effort really means best-effort. This used to catch only OSError
        # and TimeoutError, so a config the sanitizer had validated into
        # objects raised TypeError here and killed the whole run - a cache
        # write, which the next line says is optional, aborting a book.
        pass

    return concurrency


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:1234/v1")
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-concurrency", type=int, default=16)
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    print(f"Benchmarking {args.model} at {args.base_url}\n")

    level = 1
    while level <= args.max_concurrency:
        t0 = time.time()
        throughput = measure_throughput(client, args.model, level)
        dt = time.time() - t0
        status = f"{throughput:.1f} tok/s" if throughput is not None else "FAILED"
        print(f"  concurrency={level:3d}  {status:>12s}  ({dt:.1f}s)")
        if throughput is None:
            break
        level *= 2

    optimal = find_optimal_concurrency(client, args.model, args.max_concurrency)
    print(f"\nOptimal concurrency: {optimal}")


if __name__ == "__main__":
    _main()
