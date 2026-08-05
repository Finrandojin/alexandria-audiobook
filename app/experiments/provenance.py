"""Make a TTS artifact say how it was produced.

THE ARTIFACT THAT COULD NOT IDENTIFY ITSELF. On 2026-08-04 the work log quoted
`instruct_value.json` as the seeded run. It was the unseeded one. The seeded
result was in `instruct_value_seeded.json`, written three hours later, and
NOTHING INSIDE EITHER FILE DISTINGUISHED THEM:

    instruct_value.json          rows=18  arms=[per_line, per_char, none]
    instruct_value_seeded.json   rows=18  arms=[per_line, per_char, none]

Same shape, same arms, same row count, and neither recorded a seed - even
though `instruct_value.py` takes `--seed` and its own docstring is about the
seed bug. The only thing separating them was a filename suffix and an mtime: a
naming convention and a filesystem accident, neither of which is data.

An external reviewer caught it. Building a second index over those files was
considered and rejected, because an index can only show what an artifact
recorded - it would have printed `seed=-` on both rows and laundered the
ambiguity into a table.

WHY THIS IS NOT `ExperimentRecord`. `manifest.py` already does this properly
for the attribution experiments, and Rule 15 says one answer per question. But
`ExperimentRecord` requires a `gold_path` it can hash and queries LM Studio for
the model environment, and a TTS run has neither - there is no gold file and no
LLM endpoint. So the GIT capture is imported from `manifest` rather than
rewritten (that is the part that would drift), and the rest is what a
generation run actually has: seed, script, arguments.

USE IT LIKE THIS, at the point of writing the artifact:

    from experiments.provenance import provenance
    json.dump({"summary": summary, "rows": rows,
               "provenance": provenance(__file__, args)}, fh, indent=1)

`seed` is pulled from args automatically when present, because the seed is the
single field whose absence caused this.
"""
import hashlib
import os
import platform
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def file_sha256(path):
    """Return the content identity used by experiment provenance."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_sha256(paths):
    """Return repo-relative content identities, failing on missing inputs."""
    return {os.path.relpath(path, REPO): file_sha256(path) for path in paths}


def provenance(script_file, args=None, **extra):
    """-> a dict describing how this artifact was produced.

    `script_file` should be the caller's `__file__`. `args` is the argparse
    Namespace, recorded whole: an experiment's arguments ARE its identity, and
    guessing which ones matter is how `--seed` came to be omitted from a file
    whose entire purpose was to be seeded.

    Never raises. A provenance block that can fail is a provenance block that
    gets wrapped in try/except and quietly dropped, which returns us to
    artifacts that cannot identify themselves.
    """
    block = {"script": os.path.basename(str(script_file)),
             "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "host": platform.node()}
    try:
        from experiments.manifest import _git_state
        block["git"] = _git_state(REPO)
    except Exception as exc:                            # noqa: BLE001
        block["git"] = {"error": str(exc)[:120]}

    if args is not None:
        try:
            recorded = {k: v for k, v in vars(args).items()
                        if not k.startswith("_")}
            # Absolute paths differ per machine and say nothing about the run;
            # their basenames identify the inputs, which is the useful part.
            for k, v in list(recorded.items()):
                if isinstance(v, str) and os.path.isabs(v):
                    recorded[k] = os.path.relpath(v, REPO) \
                        if v.startswith(REPO) else os.path.basename(v)
            block["args"] = recorded
            # Promoted out of args as well as left inside it. This is the field
            # whose absence made two artifacts indistinguishable, and a reader
            # scanning for it should not have to know it lives under `args`.
            if "seed" in recorded:
                block["seed"] = recorded["seed"]
        except Exception as exc:                        # noqa: BLE001
            block["args"] = {"error": str(exc)[:120]}

    block.update(extra)
    return block
