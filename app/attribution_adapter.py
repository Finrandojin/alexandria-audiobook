"""Is the attribution LoRA the server claims to have the one we configured?

WHAT THIS IS FOR. A distilled adapter measured +9.8 points cross-book on all
four gold sets (disagreement 149-73 in its favour) and +14.6 through the actual
deployment path - Q4_K_M base plus f16 LoRA served by llama.cpp. That is three
to six times the run-to-run spread of ~1.6-3.7 points seen on repeated
identical configurations, and it is the largest measured attribution gain in
this project.

The adapter lives in the SERVING layer. `llama-server` is started with `--lora`
and the scale is toggled through `POST /lora-adapters`, so the application does
not change - the server it points at does. What the application must gain is
the ability to say which adapter it expects and to NOTICE when that expectation
is not met.

WHY A LOUD CHECK RATHER THAN A QUIET DEFAULT. Without one, a server started
without the adapter, or with its scale at zero, answers every request perfectly
happily at BASE quality. Nothing fails. The book generates. It is simply 9.8
points worse and no artifact records why. That is the same shape as the seed
bug - a configured field silently ignored - which cost six contaminated
comparisons before anyone noticed by ear.

DELIBERATELY NOT A HARD GATE BY DEFAULT. `require=True` raises; the default
warns and returns the finding. A book half-generated overnight should not die
because a server was restarted without its adapter, but the run must carry the
fact.
"""
import json
import os
import urllib.error
import urllib.request


class AdapterError(RuntimeError):
    """The configured attribution adapter is not the one being served."""


def adapter_config(config):
    """-> {path, scale, expect} from app config, or None if not configured.

    Read from the same `llm_local` / `llm_remote` block as base_url, so the
    adapter travels with the endpoint it belongs to rather than sitting in a
    second place that can disagree with it.
    """
    if not isinstance(config, dict):
        return None
    mode = config.get("llm_mode", "local")
    block = config.get(f"llm_{mode}") or config.get("llm") or {}
    spec = block.get("attribution_adapter")
    if not isinstance(spec, dict):
        return None
    path = str(spec.get("path") or "").strip()
    if not path:
        return None
    return {"path": path,
            "scale": float(spec.get("scale", 1.0)),
            "expect": bool(spec.get("require", False))}


def served_adapters(base_url, timeout=10):
    """-> list of adapters llama.cpp reports, or None if it cannot be asked.

    None means UNKNOWN, not absent. An endpoint that does not implement
    /lora-adapters - LM Studio, Ollama - is not evidence that the adapter is
    missing, and treating it as such would cry wolf on every non-llama.cpp
    setup.
    """
    root = (base_url or "").rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    try:
        with urllib.request.urlopen(root + "/lora-adapters",
                                    timeout=timeout) as fh:
            data = json.loads(fh.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None
    if isinstance(data, dict):
        data = data.get("lora_adapters") or data.get("adapters") or []
    return data if isinstance(data, list) else None


def check_adapter(config, base_url, require=None):
    """-> (ok, message). Never raises unless `require` is set and it fails.

    ok is True when the configured adapter is served at a non-zero scale, and
    also when no adapter is configured at all - that is a valid choice, not a
    fault.
    """
    spec = adapter_config(config)
    if spec is None:
        return True, "no attribution adapter configured"
    require = spec["expect"] if require is None else require

    served = served_adapters(base_url)
    if served is None:
        msg = (f"cannot verify the attribution adapter: {base_url} does not "
               f"answer /lora-adapters. Expected {os.path.basename(spec['path'])} "
               f"at scale {spec['scale']}.")
        if require:
            raise AdapterError(msg)
        return False, msg

    want = os.path.basename(spec["path"])
    for entry in served:
        if not isinstance(entry, dict):
            continue
        name = os.path.basename(str(entry.get("path") or entry.get("id") or ""))
        if name != want:
            continue
        scale = float(entry.get("scale", entry.get("weight", 0)) or 0)
        if scale <= 0:
            msg = (f"attribution adapter {want} is loaded but at scale "
                   f"{scale} - generation will run at BASE quality, which "
                   f"measured 9.8 points worse cross-book.")
            if require:
                raise AdapterError(msg)
            return False, msg
        return True, f"attribution adapter {want} serving at scale {scale}"

    names = [os.path.basename(str(e.get("path") or e.get("id") or "?"))
             for e in served if isinstance(e, dict)] or ["none"]
    msg = (f"attribution adapter {want} is NOT loaded; the server has "
           f"{', '.join(names)}. Generation would run at base quality.")
    if require:
        raise AdapterError(msg)
    return False, msg


def describe(config):
    """One line for an artifact or a log, so a run records what it expected."""
    spec = adapter_config(config)
    if spec is None:
        return "attribution_adapter=none"
    return (f"attribution_adapter={os.path.basename(spec['path'])} "
            f"scale={spec['scale']} require={spec['expect']}")
