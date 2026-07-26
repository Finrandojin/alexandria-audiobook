"""Helpers for applying VRAM-safe LM Studio load settings via the `lms` CLI.

Background: a long batch_review run keeps the LLM loaded with a large KV
cache. Loading the model with a high `parallel` value multiplies that KV
cache and can exhaust GPU VRAM, crashing the display server (see the
VRAM watchdog in review_script.py for the runtime safety net). These
helpers let both the CLI script and the web UI force/inspect the
VRAM-safe load configuration (context 8192, parallel 1, full GPU offload).

Also covers the remote case: a LM Studio on a Thunder Compute GPU instance,
managed over SSH instead of the local `lms` CLI (see apply_remote_lmstudio_settings).
"""

import json
import shlex
import shutil
import subprocess
import threading
import time
from collections import defaultdict
from urllib.parse import urlparse

from utils import run_rocm_smi_json

IDEAL_SETTINGS = {"context_length": 8192, "parallel": 1, "gpu": "max"}
DEFAULT_SETTINGS = {"context_length": 4096, "parallel": 4, "gpu": "max"}

# Profiles are enabled only after the exact model/profile has passed a real
# concurrent near-limit run on this machine. Unknown models and unreadable GPU
# metrics retain IDEAL_SETTINGS rather than extrapolating from another model.
_VERIFIED_LOCAL_PROFILES = {
    "gemma-4-e4b-uncensored-hauhaucs-aggressive": {
        "context_length": 32768,
        "parallel": 2,
        "model_vram_bytes": int(8.50 * 1024 ** 3),
        # Measured context growth was smaller; 16 KiB/token deliberately
        # overestimates it so transient backend workspaces remain covered.
        "bytes_per_extra_context_token": 16 * 1024,
    },
    # A 9B (6.55 GB on disk) that leaves ample room for a full 32768 context on
    # a ~16 GB card - added for the script-gen A/B so it runs at the same
    # context as gemma instead of the 8192 fallback the 27B is stuck at. The
    # get_safe_local_settings VRAM guard still gates the actual load, so an
    # over-tight moment falls back to 8192 rather than risking OOM.
    "qwen3.5-9b-uncensored-hauhaucs-aggressive": {
        "context_length": 32768,
        "parallel": 1,
        "model_vram_bytes": int(7.20 * 1024 ** 3),
        "bytes_per_extra_context_token": 16 * 1024,
    },
    # Both ministrals measured 2026-07-24 on this card (15.92 GiB total):
    # ~8.8 GiB of weights, but ~160 KiB/token of KV cache - ten times the
    # gemma/qwen figure, so context is what constrains them, not weights.
    # 32768 was measured and left only ~1.5 GiB, under the 2 GiB reserve;
    # 16384 projects to ~11.9 GiB used with ~4.0 GiB spare. Kept at 16384
    # rather than the 24576 that also fits, because this is a load-time
    # measurement rather than the sustained near-limit run this table wants.
    "ministral-3-14b-instruct-2512": {
        "context_length": 16384,
        "parallel": 1,
        "model_vram_bytes": int(8.90 * 1024 ** 3),
        # Measured 158.90 KiB/token; rounded up to cover transient workspaces.
        "bytes_per_extra_context_token": 176 * 1024,
    },
    # Dense 14B/15B pairs measured 2026-07-26 on this card (15.92 GiB total,
    # 2.25 GiB resident at idle). Both cost ~160 KiB per context token, ten
    # times the compressed-KV models above, so 32768 leaves under 1 GiB of
    # headroom and is refused; 16384 matches ministral and keeps ~2.5-3 GiB.
    "microsoft/phi-4": {
        "context_length": 16384,
        "parallel": 1,
        "model_vram_bytes": int(9.91 * 1024 ** 3),
        # Measured 159.95 KiB/token; rounded up to cover transient workspaces.
        "bytes_per_extra_context_token": 176 * 1024,
    },
    "qwen/qwen3-14b": {
        "context_length": 16384,
        "parallel": 1,
        "model_vram_bytes": int(9.33 * 1024 ** 3),
        # Measured 159.78 KiB/token; rounded up as above.
        "bytes_per_extra_context_token": 176 * 1024,
    },
    # mistralai/magistral-small is deliberately absent. Its weights alone take
    # 13.51 GiB, leaving 0.16 GiB at the 8192 minimum - below any usable
    # reserve. It cannot run here without weakening the VRAM guard, so it gets
    # no profile and falls back to the conservative default rather than being
    # made to fit.
    "ministral-3-14b-instruct-2512-absolute-heresy-i1": {
        "context_length": 16384,
        "parallel": 1,
        "model_vram_bytes": int(8.80 * 1024 ** 3),
        # Measured 163.05 KiB/token; rounded up to cover transient workspaces.
        "bytes_per_extra_context_token": 176 * 1024,
    },
    # gemma-4-12b-coder-fable5-composer2.5-v1 (13.45 GiB) and
    # qwen3.6-27b-uncensored-hauhaucs-aggressive (12.67 GiB) are deliberately
    # absent: both exceed the 2 GiB reserve at the 8192 baseline already
    # (0.53 and 1.33 GiB spare), so no context setting makes them safe here.
    # They stay on the conservative fallback.
}
_LOCAL_VRAM_RESERVE_BYTES = 2 * 1024 ** 3

# "Best settings" for a remote LM Studio on a big cloud GPU (e.g. Thunder A6000,
# 48GB): a large context with headroom under the model's max, vs LM Studio's
# small defaults. Applied over SSH since the forwarded /v1 port can't set these.
REMOTE_IDEAL_SETTINGS = {"context_length": 98304, "parallel": 2}
REMOTE_DEFAULT_SETTINGS = {"context_length": 4096, "parallel": 4}


class TokenBudgetError(ValueError):
    """The prompt and safety reserve do not fit the verified context."""


def get_effective_max_tokens(fallback, context_length=None, messages=None,
                             hard_max=None, reserve=512, scale_to_context=True):
    """Return a context-safe completion budget for a production LLM call.

    ``fallback`` is the conservative allowance used when live model context is
    unknown.  A verified larger context may raise it, but never beyond the
    call-specific ``hard_max`` or the space left after prompt and reserve.
    Character-based prompt estimation is deliberately conservative because
    this helper must work without loading a model-specific tokenizer.
    """
    fallback = max(1, int(fallback))
    hard_max = max(1, int(hard_max or fallback))
    fallback = min(fallback, hard_max)
    if not context_length:
        return min(fallback, hard_max)
    context_length = max(1, int(context_length))
    prompt_chars = sum(len(str(message.get("content") or ""))
                       for message in (messages or []))
    prompt_tokens = (prompt_chars + 2) // 3
    available = context_length - prompt_tokens - max(0, int(reserve))
    if available < 1:
        raise TokenBudgetError(
            f"Prompt estimate ({prompt_tokens}) plus reserve ({reserve}) exceeds "
            f"the loaded context ({context_length}). Reduce the input or increase LM Studio context.")
    scaled_target = max(fallback, context_length // 4) if scale_to_context else fallback
    return max(1, min(hard_max, scaled_target, available))


def get_next_retry_max_tokens(current, retry_reason, hard_max, multiplier=1.5):
    """Return the next requested completion budget for evidence of truncation."""
    current = max(1, int(current))
    hard_max = max(1, int(hard_max or current))
    if retry_reason not in {"token_truncated", "incomplete_output"}:
        return min(current, hard_max)
    increased = max(current + 1, int(current * multiplier + 0.5))
    return min(increased, hard_max)

_LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "::1", "")


def is_local_llm_endpoint(base_url):
    """Return True if base_url points at a local LM Studio (so the local `lms`
    CLI and the local GPU VRAM watchdog are meaningful). Returns False for a
    remote endpoint (e.g. LM Studio on a Thunder Compute instance), where the
    model is loaded/managed elsewhere and local `lms`/nvidia-smi calls are moot.
    """
    if not base_url:
        return True
    return (urlparse(base_url).hostname or "").lower() in _LOCAL_HOSTS


def is_remote_llm(llm_mode, base_url):
    """Single source of truth for "is this LLM endpoint remote?" - use this
    everywhere instead of checking is_local_llm_endpoint alone, which misses
    an explicit llm_mode="remote" pointed at a URL that happens to resolve as
    local (or vice versa after a save_config edge case where llm_mode and the
    active base_url have drifted out of sync).
    """
    return llm_mode == "remote" or not is_local_llm_endpoint(base_url)


def find_lms_binary():
    """Return the path to the `lms` CLI, or None if it isn't available."""
    return shutil.which("lms")


def get_local_vram_bytes():
    """Return (total_bytes, used_bytes), or None when no reliable probe works."""
    data = run_rocm_smi_json(["--showmeminfo", "vram"],
                             rocm_smi_path="/opt/rocm/bin/rocm-smi", timeout=2)
    if data:
        for card_data in data.values():
            try:
                total = int(card_data["VRAM Total Memory (B)"])
                used = int(card_data["VRAM Total Used Memory (B)"])
            except (KeyError, TypeError, ValueError):
                continue
            if total > 0 and 0 <= used <= total:
                return total, used

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        total_mib, used_mib = (int(part.strip())
                               for part in result.stdout.splitlines()[0].split(","))
        if result.returncode == 0 and total_mib > 0 and 0 <= used_mib <= total_mib:
            return total_mib * 1024 ** 2, used_mib * 1024 ** 2
    except (OSError, IndexError, ValueError, subprocess.TimeoutExpired):
        pass
    return None


def get_safe_local_settings(model_name, model_loaded, vram_bytes=None):
    """Select a verified profile from live VRAM, otherwise return the fallback."""
    fallback = dict(IDEAL_SETTINGS)
    fallback["reason"] = "conservative fallback"
    profile = _VERIFIED_LOCAL_PROFILES.get(model_name)
    memory = vram_bytes if vram_bytes is not None else get_local_vram_bytes()
    if not profile or not memory:
        fallback["reason"] = ("model has no verified dynamic profile" if not profile
                              else "GPU memory could not be measured")
        return fallback

    total, used = memory
    baseline = used if model_loaded else used + profile["model_vram_bytes"]
    extra_tokens = max(0, profile["context_length"] - IDEAL_SETTINGS["context_length"])
    projected = baseline + extra_tokens * profile["bytes_per_extra_context_token"]
    if projected + _LOCAL_VRAM_RESERVE_BYTES > total:
        fallback["reason"] = "live VRAM lacks the 2 GiB safety reserve"
        return fallback
    return {"context_length": profile["context_length"],
            "parallel": profile["parallel"], "gpu": "max",
            "reason": "verified profile fits live VRAM with 2 GiB reserved"}


def _validate_ssh_alias(ssh_alias):
    """Raise OSError if ssh_alias is empty or could be parsed by `ssh` as an
    option rather than a literal hostname (e.g. starts with '-'). Raises
    OSError (not ValueError) specifically because every SSH-driving caller in
    this module already catches OSError for connection failures.
    """
    if not ssh_alias or ssh_alias.startswith("-"):
        raise OSError(f"Invalid SSH host alias: {ssh_alias!r}")


def _ssh_run(ssh_alias, remote_cmd, timeout, connect_timeout=10):
    """Run remote_cmd on ssh_alias via `bash -lc`, returning the CompletedProcess.

    Shared by every remote helper below instead of each one re-building its
    own ssh argv list. Pre-quoting remote_cmd into a single argv element
    (rather than passing "bash", "-lc", remote_cmd as 3 separate argv
    elements) is required: ssh joins all args after the host with bare
    spaces and hands the result to the remote shell as one line, so 3 separate
    elements would let that join split remote_cmd's own `;`-separated
    statements apart, breaking any command after the first.
    """
    _validate_ssh_alias(ssh_alias)
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={connect_timeout}",
         ssh_alias, "bash -lc " + shlex.quote(remote_cmd)],
        capture_output=True, text=True, timeout=timeout,
    )


def _parse_lms_ps_output(stdout, model_name, ideal_settings):
    """Parse `lms ps --json` output and return the status dict for model_name.

    Result dict: {available, loaded, context_length, parallel, optimized}.
    Tries json.loads on the whole string first (the local, no-login-shell
    case always looks like this); only on failure falls back to scanning
    non-empty lines from the end, since a remote login shell (bash -lc)
    prints a decorative banner ahead of the real JSON output - this fallback
    keeps the local path's behavior identical while making the remote path
    robust to one stray trailing non-JSON line too.
    """
    models = None
    try:
        models = json.loads(stdout or "")
    except json.JSONDecodeError:
        for line in reversed([l for l in (stdout or "").splitlines() if l.strip()]):
            try:
                models = json.loads(line)
                break
            except json.JSONDecodeError:
                continue

    if not isinstance(models, list):
        return {"available": True, "loaded": False, "context_length": None,
                "parallel": None, "optimized": False}

    for m in models:
        if m.get("identifier") == model_name or m.get("modelKey") == model_name:
            context_length = m.get("contextLength")
            parallel = m.get("parallel")
            optimized = (context_length == ideal_settings["context_length"]
                         and parallel == ideal_settings["parallel"])
            return {"available": True, "loaded": True, "context_length": context_length,
                    "parallel": parallel, "optimized": optimized}

    return {"available": True, "loaded": False, "context_length": None,
            "parallel": None, "optimized": False}


def get_lmstudio_status(model_name):
    """Return current load status for model_name via `lms ps --json`.

    Result dict: {available, loaded, context_length, parallel, optimized}
    - available: whether the `lms` CLI could be found/run
    - loaded: whether the model is currently loaded
    - optimized: whether the loaded settings match IDEAL_SETTINGS
    """
    lms = find_lms_binary()
    if not lms:
        return {"available": False, "loaded": False, "context_length": None,
                "parallel": None, "optimized": False}

    try:
        result = subprocess.run([lms, "ps", "--json"], capture_output=True,
                                 text=True, timeout=15)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError, UnicodeDecodeError):
        return {"available": True, "loaded": False, "context_length": None,
                "parallel": None, "optimized": False}

    initial = _parse_lms_ps_output(result.stdout, model_name, IDEAL_SETTINGS)
    ideal = get_safe_local_settings(model_name, initial["loaded"])
    status = _parse_lms_ps_output(result.stdout, model_name, ideal)
    status.update({"ideal_context_length": ideal["context_length"],
                   "ideal_parallel": ideal["parallel"],
                   "settings_reason": ideal["reason"]})
    return status


def _remote_server_bound(ssh_alias, port, timeout=10):
    """True if a process on ssh_alias is listening on 0.0.0.0:<port>
    specifically (not 127.0.0.1:<port> - that distinction IS the bug this
    guards against: an LM Studio server bound to localhost only is
    unreachable through the forwarded HTTPS tunnel, but still looks
    "loaded" to `lms ps`). Returns None on SSH failure - callers should
    treat that as "unknown", not "unreachable".

    Pure read (Rule 16) - never starts anything; see
    ensure_remote_server_running for the mutating counterpart.
    """
    try:
        result = _ssh_run(ssh_alias, "ss -tlnp", timeout=timeout, connect_timeout=10)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    needle = f"0.0.0.0:{port} "
    return any(needle in line for line in result.stdout.splitlines())


def ensure_remote_server_running(ssh_alias, port, timeout=30):
    """Start LM Studio's server bound to 0.0.0.0:<port> on ssh_alias if it
    isn't already - `apply_remote_lmstudio_settings`'s `lms load` only
    loads a model into an already-running server; it never starts the
    server itself, so a stopped or localhost-only-bound server "succeeds"
    while staying unreachable through the forwarded tunnel.

    Returns (success, message). Never raises.
    """
    if _remote_server_bound(ssh_alias, port, timeout=10):
        return True, f"Server already bound on 0.0.0.0:{port}"
    remote_cmd = f"lms server start --port {port} --bind 0.0.0.0"
    try:
        result = _ssh_run(ssh_alias, remote_cmd, timeout=timeout, connect_timeout=15)
    except (subprocess.TimeoutExpired, OSError) as e:
        return False, f"SSH to '{ssh_alias}' failed: {e}"
    if result.returncode != 0:
        return False, (result.stderr.strip() or result.stdout.strip()
                       or f"ssh exited {result.returncode}")
    return True, f"Started server on 0.0.0.0:{port}"


def get_remote_lmstudio_status(ssh_alias, model_name, timeout=20, port=None):
    """Like get_lmstudio_status, but for a remote LM Studio reached over SSH
    (ssh_alias e.g. "tnr-0", from config.json's llm_remote_ssh).

    Result dict: {available, loaded, context_length, parallel, optimized} -
    "optimized" compares against REMOTE_IDEAL_SETTINGS, mirroring
    get_lmstudio_status's local IDEAL_SETTINGS comparison. Best-effort: never
    raises, returns available=False on any SSH/parse failure or if ssh_alias
    isn't configured.

    `loaded: True` here only means `lms ps` sees the model in memory - it
    does NOT mean the HTTP server is reachable (a server bound to
    127.0.0.1 or not running at all still reports a loaded model). Pass
    `port` to also get a `server_reachable` field; omitted by default so
    existing callers that don't have a port handy are unaffected.
    """
    if not ssh_alias:
        return {"available": False, "loaded": False, "context_length": None,
                "parallel": None, "optimized": False}

    try:
        result = _ssh_run(ssh_alias, "lms ps --json", timeout=timeout, connect_timeout=10)
    except (subprocess.TimeoutExpired, OSError):
        return {"available": False, "loaded": False, "context_length": None,
                "parallel": None, "optimized": False}

    status = _parse_lms_ps_output(result.stdout, model_name, REMOTE_IDEAL_SETTINGS)
    if port is not None:
        status["server_reachable"] = _remote_server_bound(ssh_alias, port, timeout=timeout)
    return status


_remote_status_cache = {}  # (ssh_alias, model_name) -> (timestamp, status_dict)
_REMOTE_STATUS_CACHE_TTL = 10  # seconds - shorter than the UI's 30s poll interval
_remote_status_cache_lock = threading.Lock()  # protects _remote_status_cache itself (fast dict ops only)
_remote_status_key_locks = defaultdict(threading.Lock)  # one lock per (ssh_alias, model_name)

def get_remote_lmstudio_status_cached(ssh_alias, model_name, timeout=20):
    """Like get_remote_lmstudio_status, but reuses a result younger than
    _REMOTE_STATUS_CACHE_TTL seconds instead of making a fresh SSH round-trip.

    Multiple browser tabs each poll independently every 30s; without this,
    each poll across every open tab triggers its own live SSH call to the
    remote host just to refresh a status badge. This caps it to at most one
    SSH call per TTL window regardless of how many tabs are open.

    Keyed by (ssh_alias, model_name) - not just ssh_alias - so switching the
    configured model doesn't return a stale status computed for the
    previous one. The actual SSH call runs under a per-key lock (not the
    global _remote_status_cache_lock, which only ever guards the dict's own
    get/set) so concurrent requests for the SAME key block on one real SSH
    call, while requests for a DIFFERENT key aren't held up behind someone
    else's slow round-trip.
    """
    key = (ssh_alias, model_name)
    now = time.time()
    with _remote_status_cache_lock:
        cached = _remote_status_cache.get(key)
    if cached and (now - cached[0]) < _REMOTE_STATUS_CACHE_TTL:
        return cached[1]

    with _remote_status_key_locks[key]:
        now = time.time()
        with _remote_status_cache_lock:
            cached = _remote_status_cache.get(key)
        if cached and (now - cached[0]) < _REMOTE_STATUS_CACHE_TTL:
            return cached[1]
        status = get_remote_lmstudio_status(ssh_alias, model_name, timeout=timeout)
        with _remote_status_cache_lock:
            _remote_status_cache[key] = (time.time(), status)
        return status


def invalidate_remote_status_cache(ssh_alias=None):
    """Drop cached remote status so the next poll makes a fresh SSH call.

    Call this after any action that changes what 'lms ps' would report
    (e.g. apply_remote_lmstudio_settings) - otherwise a poll within the TTL
    window can show pre-change status right after a successful change.
    ssh_alias=None clears every cached entry; pass a specific alias to
    clear just that one.
    """
    with _remote_status_cache_lock:
        if ssh_alias is None:
            _remote_status_cache.clear()
        else:
            for key in [k for k in _remote_status_cache if k[0] == ssh_alias]:
                del _remote_status_cache[key]


def _gpu_name_from_probes(run):
    """Shared logic for get_gpu_name_and_backend/get_remote_gpu_name_and_backend.

    `run` is a callable(argv_list) -> CompletedProcess-like object with
    .returncode/.stdout, so the same probe logic works for a local
    subprocess.run and an SSH-wrapped one.

    Takes the LAST non-empty output line, not the first: a remote login
    shell (bash -lc) prints a decorative banner ahead of any command's real
    output (confirmed live - 4 lines of box-drawing before the actual
    "NVIDIA RTX A6000"), so the real answer is always the most recent line,
    never the first. Harmless for the local (banner-free) case too, since
    there's only one line of real output there either way.
    """
    try:
        result = run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
        lines = [l.strip() for l in (result.stdout or "").splitlines() if l.strip()]
        name = lines[-1] if lines else ""
        if result.returncode == 0 and name:
            return name, "cuda"
    except (OSError, IndexError, subprocess.TimeoutExpired):
        pass

    try:
        result = run(["rocm-smi", "--showproductname"])
    except (OSError, subprocess.TimeoutExpired):
        result = None
    if result is not None and result.returncode == 0 and result.stdout:
        # rocm-smi's JSON mode is handled by run_rocm_smi_json for local calls;
        # for the shared SSH-compatible path we just scan the plain-text output.
        for line in result.stdout.splitlines():
            if "Card series" in line or "Card model" in line:
                name = line.split(":", 1)[-1].strip()
                if name:
                    return name, "rocm"
    return None, None


def get_gpu_name_and_backend():
    """Return (gpu_name, backend) for the local machine, backend in
    ("cuda", "rocm", None). No torch dependency (same subprocess-based style
    as review_script.py's get_vram_usage) so this works even in environments
    without torch installed. Prefers rocm-smi's JSON output (more reliable
    than scraping --showproductname's text) before falling back to the
    shared text-scraping probe.
    """
    name_data = run_rocm_smi_json(["--showproductname"], rocm_smi_path="/opt/rocm/bin/rocm-smi", timeout=2)
    if name_data:
        for card_data in name_data.values():
            if isinstance(card_data, dict):
                name = card_data.get("Card Series") or card_data.get("Card Model")
                if name and name != "N/A":
                    return name, "rocm"

    return _gpu_name_from_probes(lambda argv: subprocess.run(argv, capture_output=True, text=True, timeout=5))


def get_remote_gpu_name_and_backend(ssh_alias):
    """Same as get_gpu_name_and_backend, but probes a remote host over SSH."""
    if not ssh_alias:
        return None, None

    def _run(argv):
        remote_cmd = " ".join(shlex.quote(a) for a in argv)
        return _ssh_run(ssh_alias, remote_cmd, timeout=15, connect_timeout=10)

    return _gpu_name_from_probes(_run)


def apply_lmstudio_settings(model_name, ideal=True, ttl=3600):
    """Reload model_name with either the VRAM-safe (ideal) or default settings.

    Best-effort: returns (success, message). Never raises.
    """
    lms = find_lms_binary()
    if not lms:
        return False, "lms CLI not found on PATH"

    if ideal:
        status = get_lmstudio_status(model_name)
        settings = get_safe_local_settings(model_name, status["loaded"])
    else:
        settings = DEFAULT_SETTINGS

    # `lms load` refuses to load if a model is already loaded under the same
    # identifier, so drop any existing instance first. If unload fails (e.g.
    # the model is busy), the load below will likely fail too - remember that
    # so the failure message can explain why the old settings may still be
    # active instead of just reporting the load error in isolation.
    unload_failed = False
    try:
        unload_result = subprocess.run([lms, "unload", model_name], capture_output=True,
                                        text=True, timeout=60)
        unload_failed = unload_result.returncode != 0
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired):
        unload_failed = True

    try:
        result = subprocess.run(
            [lms, "load", model_name,
             "--context-length", str(settings["context_length"]),
             "--parallel", str(settings["parallel"]),
             "--gpu", settings["gpu"],
             "--identifier", model_name,
             "--ttl", str(ttl),
             "-y"],
            capture_output=True, text=True, timeout=180
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError, FileNotFoundError) as e:
        return False, f"Failed to run lms load: {e}"

    if result.returncode != 0:
        msg = result.stderr.strip() or result.stdout.strip() or "lms load failed"
        if unload_failed:
            msg += (" (unloading the previously-loaded model also failed - "
                    "it may still be running with different settings)")
        return False, msg

    label = "VRAM-safe" if ideal else "default"
    detail = (f" ({settings['context_length']} context, parallel {settings['parallel']}; "
              f"{settings.get('reason', 'fixed profile')})")
    return True, f"Reloaded {model_name} with {label} settings{detail}"


def apply_remote_lmstudio_settings(ssh_alias, model_name, ideal=True, port=1234):
    """Reload model_name on a remote LM Studio host via SSH (`lms` over `tnr-N`).

    Returns (success, message). The forwarded OpenAI /v1 port can't change load
    settings, so we drive the remote `lms` CLI directly. Never raises. Unlike
    apply_lmstudio_settings, this intentionally does not pass `--ttl` - that's
    today's existing remote semantics, not an oversight.

    Ensures the server itself is bound to 0.0.0.0:<port> before loading -
    `lms load` only loads a model into an already-running server, so without
    this a stopped or localhost-only-bound server used to "succeed" here
    while staying unreachable through the forwarded tunnel.
    """
    server_ok, server_msg = ensure_remote_server_running(ssh_alias, port)
    if not server_ok:
        return False, f"Could not start remote server: {server_msg}"
    settings = REMOTE_IDEAL_SETTINGS if ideal else REMOTE_DEFAULT_SETTINGS
    # model_name is shlex.quote()'d (not hand-wrapped in '...') since it comes
    # from user-editable config with no character restrictions - a bare
    # f"'{model_name}'" would let an embedded single quote break out of the
    # intended argument and inject additional shell commands once bash -lc
    # evaluates remote_cmd on the remote host.
    quoted_model = shlex.quote(model_name)
    remote_cmd = (
        f"lms unload {quoted_model} >/dev/null 2>&1; "
        f"lms load {quoted_model} --context-length {settings['context_length']} "
        f"--parallel {settings['parallel']} --gpu max --identifier {quoted_model} -y"
    )
    try:
        result = _ssh_run(ssh_alias, remote_cmd, timeout=200, connect_timeout=15)
    except (subprocess.TimeoutExpired, OSError) as e:
        return False, f"SSH to '{ssh_alias}' failed: {e}"
    if result.returncode != 0:
        return False, (result.stderr.strip() or result.stdout.strip()
                       or f"ssh exited {result.returncode}")
    label = f"best ({REMOTE_IDEAL_SETTINGS['context_length']} ctx)" if ideal else "default"
    invalidate_remote_status_cache(ssh_alias)
    return True, f"Reloaded {model_name} on '{ssh_alias}' with {label} settings"


def get_current_status(llm_mode, base_url, model_name, ssh_alias=None, use_cache=False):
    """Fetch live status (local or remote) with no self-heal side effect -
    just the is_remote_llm dispatch shared by ensure_ideal_settings's own
    status checks, the /api/lmstudio/status route, and any later
    re-verification that doesn't want to trigger a reload.

    use_cache=True (only for the polling /api/lmstudio/status route, which
    gets hit every ~30s per open browser tab) reuses a recent remote-status
    result instead of making a fresh SSH round-trip every call. Callers that
    need a guaranteed-fresh read before acting on it (e.g. ensure_ideal_settings
    deciding whether to reload) must leave this False.
    """
    if is_remote_llm(llm_mode, base_url):
        if use_cache:
            return get_remote_lmstudio_status_cached(ssh_alias, model_name)
        return get_remote_lmstudio_status(ssh_alias, model_name)
    return get_lmstudio_status(model_name)


def get_planned_ideal_settings(llm_mode, base_url, model_name, ssh_alias=None):
    """Return the settings a subsequent ``ensure_ideal_settings`` will target.

    This is a pure sizing helper for preflight UIs: it reports the verified
    target without reloading a model. Runtime callers must still call
    ``ensure_ideal_settings`` before dispatching work and size against its
    fresh post-heal status.
    """
    if is_remote_llm(llm_mode, base_url):
        return {"context_length": REMOTE_IDEAL_SETTINGS["context_length"],
                "parallel": REMOTE_IDEAL_SETTINGS["parallel"],
                "settings_reason": "remote ideal profile"}
    status = get_current_status(llm_mode, base_url, model_name, ssh_alias)
    return {
        "context_length": (status.get("ideal_context_length")
                           or IDEAL_SETTINGS["context_length"]),
        "parallel": status.get("ideal_parallel") or IDEAL_SETTINGS["parallel"],
        "settings_reason": status.get("settings_reason") or "conservative fallback",
    }


def ensure_ideal_settings(llm_mode, base_url, model_name, ssh_alias=None):
    """Self-heal LM Studio's load settings (local or remote) toward the ideal
    config (VRAM-safe locally, large-context remotely), then return the FRESH
    post-heal status so callers can size chunking/context decisions and the
    concurrency benchmark off live truth instead of each re-fetching it.

    Shared by review_script.py and find_nicknames.py instead of each
    hand-rolling its own copy of this branch.

    Returns (is_remote, status, message). status always has the
    {available, loaded, context_length, parallel, optimized} shape. Never
    raises - every call it makes is itself best-effort/non-raising.
    """
    is_remote = is_remote_llm(llm_mode, base_url)

    if is_remote and not ssh_alias:
        return (True, get_remote_lmstudio_status(None, model_name),
                "Remote LLM endpoint - no SSH alias configured, cannot verify/apply ideal settings.")

    if is_remote:
        get_status = lambda: get_current_status(llm_mode, base_url, model_name, ssh_alias)
        apply_settings = lambda: apply_remote_lmstudio_settings(ssh_alias, model_name, ideal=True)
        label = "Remote LM Studio"
        ok_warning = ("Proceeding with whatever is currently loaded, which may truncate "
                      "responses or fail outright if the context is too small.")
    else:
        get_status = lambda: get_current_status(llm_mode, base_url, model_name, ssh_alias)
        apply_settings = lambda: apply_lmstudio_settings(model_name, ideal=True)
        label = "LM Studio"
        ok_warning = ("The model may be running with a higher 'parallel'/context-length "
                      "configuration, which uses more VRAM per request and increases the "
                      "risk of an out-of-memory crash. The VRAM watchdog below will still "
                      "pause batches if usage gets too high, but if you hit OOM, restart "
                      "LM Studio and re-run.")

    status = get_status()
    if status["loaded"] and status["optimized"]:
        return is_remote, status, f"{label}: {model_name} already loaded with ideal settings."

    ok, msg = apply_settings()
    status = get_status()
    if ok:
        return is_remote, status, f"{label}: {msg}"
    if status["loaded"] and status["optimized"]:
        return (is_remote, status,
                f"{label}: could not reload ({msg}), but {model_name} is "
                f"already loaded with ideal settings - continuing.")
    return is_remote, status, f"{label}: WARNING - could not apply ideal settings ({msg}). {ok_warning}"
