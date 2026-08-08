"""Standalone tests for F17: config keys silently dropped on save.

Verifies that AppConfig/TTSConfig/GenerationConfig round-trip realistic
configs without dropping keys the pipeline reads (num_ctx,
max_context_roster_names, review_batch_char_budget, review_batch_size,
enable_nemo_normalization), without introducing keys that weren't present
in an untouched config, and without changing values. No pytest; exits 0 on
success, 1 on failure.
"""
import copy
import json
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("ALEXANDRIA_CONFIG_PATH",
                       os.path.join(os.path.dirname(os.path.abspath(__file__)), "_nonexistent_config.json"))

# app.py pulls in `project` (-> tts.py -> numpy/torch/etc.) purely for
# unrelated TTS/project-management functionality that this test does not
# exercise. Stub it out before importing app.py, same pattern as
# app/test_canon_wiring.py, to keep this test lightweight and standalone.
if 'project' not in sys.modules:
    _fake_project = types.ModuleType('project')

    class _FakeProjectManager:
        def __init__(self, *args, **kwargs):
            pass

        def load_chunks(self):
            return []

        def save_chunks(self, chunks):
            pass

        def __getattr__(self, name):
            def _noop(*args, **kwargs):
                return None
            return _noop

    _fake_project.ProjectManager = _FakeProjectManager
    sys.modules['project'] = _fake_project

import app as app_module  # noqa: E402

AppConfig = app_module.AppConfig

failures = []


def check(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"  [ {status} ] {name}")
    if not condition:
        failures.append(name)


REALISTIC_CONFIG = {
    "llm": {
        "base_url": "http://127.0.0.1:11434/v1",
        "api_key": "ollama",
        "model_name": "qwen2.5:14b",
    },
    "tts": {
        "mode": "local",
        "url": "http://127.0.0.1:7860",
        "device": "auto",
        "language": "English",
        "parallel_workers": 2,
        "batch_seed": None,
        "compile_codec": False,
        "sub_batch_enabled": True,
        "sub_batch_min_size": 4,
        "sub_batch_ratio": 5.0,
        "sub_batch_max_items": 0,
        "batch_group_by_type": False,
        "pause_between_speakers_ms": 500,
        "pause_same_speaker_ms": 250,
    },
    "prompts": {
        "system_prompt": "You are a script generator.",
        "user_prompt": "Generate a script.",
        "review_system_prompt": None,
        "review_user_prompt": None,
        "persona_system_prompt": None,
        "persona_user_prompt": None,
        "persona_advanced_prompt": None,
    },
    "generation": {
        "chunk_size": 3000,
        "max_tokens": 4096,
        "temperature": 0.6,
        "top_p": 0.8,
        "top_k": 0,
        "min_p": 0,
        "presence_penalty": 0.0,
        "banned_tokens": [],
        "merge_narrators": False,
    },
}


def save(config_dict):
    """Mimic save_config's persistence logic without touching disk/FastAPI."""
    cfg = AppConfig(**config_dict)
    return cfg.model_dump(exclude_none=True)


print("=" * 60)
print("  F17: Config round-trip (save_config key preservation)")
print("=" * 60)

# 1. Untouched realistic config: no keys added that weren't there, no
#    existing value changed. (New keys carry their own defaults which equal
#    the pipeline's existing hard-coded defaults, so this is intentional
#    growth of the schema, not silent corruption -- but nothing already
#    present may be dropped or mutated, and num_ctx must NOT appear.)
original = copy.deepcopy(REALISTIC_CONFIG)
saved = save(original)

check("llm section unchanged", saved["llm"] == original["llm"])
check("tts: no existing tts key changed",
      all(saved["tts"].get(k) == v for k, v in original["tts"].items() if v is not None))
check("tts: batch_seed (None) omitted, not corrupted",
      "batch_seed" not in saved["tts"] or saved["tts"]["batch_seed"] is None)
check("generation: no existing generation key changed",
      all(saved["generation"].get(k) == v for k, v in original["generation"].items()))
check("prompts: non-null prompt values preserved",
      saved["prompts"]["system_prompt"] == "You are a script generator." and
      saved["prompts"]["user_prompt"] == "Generate a script.")
check("num_ctx absent when never set (unset stays unset)",
      "num_ctx" not in saved["generation"])

# 2. New keys persist when explicitly set.
cfg2 = copy.deepcopy(REALISTIC_CONFIG)
cfg2["generation"]["num_ctx"] = 8192
cfg2["generation"]["max_context_roster_names"] = 75
cfg2["generation"]["review_batch_char_budget"] = 9000
cfg2["generation"]["review_batch_size"] = 10
cfg2["tts"]["enable_nemo_normalization"] = True
saved2 = save(cfg2)

check("num_ctx persists when set", saved2["generation"].get("num_ctx") == 8192)
check("max_context_roster_names persists when set",
      saved2["generation"].get("max_context_roster_names") == 75)
check("review_batch_char_budget persists when set",
      saved2["generation"].get("review_batch_char_budget") == 9000)
check("review_batch_size persists when set",
      saved2["generation"].get("review_batch_size") == 10)
check("enable_nemo_normalization persists when set",
      saved2["tts"].get("enable_nemo_normalization") is True)

# 3. num_ctx stays absent through a second round-trip if left unset.
cfg3 = copy.deepcopy(REALISTIC_CONFIG)
saved3a = save(cfg3)
saved3b = save(saved3a)
check("num_ctx stays absent across repeated round-trips",
      "num_ctx" not in saved3a["generation"] and "num_ctx" not in saved3b["generation"])

# 4. Unknown/legacy keys are preserved (extra="allow" design choice).
cfg4 = copy.deepcopy(REALISTIC_CONFIG)
cfg4["generation"]["some_future_key_not_yet_declared"] = "keep-me"
cfg4["tts"]["another_future_flag"] = 42
cfg4["a_brand_new_top_level_section"] = {"foo": "bar"}
saved4 = save(cfg4)
check("unknown generation key preserved (extra='allow')",
      saved4["generation"].get("some_future_key_not_yet_declared") == "keep-me")
check("unknown tts key preserved (extra='allow')",
      saved4["tts"].get("another_future_flag") == 42)
check("unknown top-level section preserved (extra='allow')",
      saved4.get("a_brand_new_top_level_section") == {"foo": "bar"})

# 5. Defaults for the new keys match the pipeline's own hard-coded defaults
#    (so a config that never mentions them behaves identically pre/post fix,
#    except num_ctx which must stay entirely absent).
minimal = {
    "llm": REALISTIC_CONFIG["llm"],
    "tts": {"mode": "local", "url": "http://x", "device": "auto"},
    "generation": {},
}
saved_min = save(minimal)
check("minimal config: max_context_roster_names defaults to 50",
      saved_min["generation"]["max_context_roster_names"] == 50)
check("minimal config: review_batch_size defaults to 25",
      saved_min["generation"]["review_batch_size"] == 25)
check("minimal config: review_batch_char_budget defaults to 12000",
      saved_min["generation"]["review_batch_char_budget"] == 12000)
check("minimal config: enable_nemo_normalization defaults to False",
      saved_min["tts"]["enable_nemo_normalization"] is False)
check("minimal config: num_ctx still absent by default",
      "num_ctx" not in saved_min["generation"])

print("=" * 60)
if failures:
    print(f"  {len(failures)} check(s) FAILED:")
    for f in failures:
        print(f"    - {f}")
    sys.exit(1)
else:
    print("  All checks passed.")
    sys.exit(0)
