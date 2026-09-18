#!/usr/bin/env python3
"""Automated API test script for Alexandria audiobook generator.

Usage:
    python test_api.py                    # Quick tests only
    python test_api.py --full             # Include TTS/LLM-dependent tests
    python test_api.py --offline          # Offline pipeline-invariant tests only
                                           # (no server, no LLM; alias: --offline-only)
    python test_api.py --url http://host:port
"""

import argparse
import io
import json
import os
import re
import sys
import tempfile
import time
from contextlib import redirect_stdout

import requests

# ── Offline pipeline invariant tests: optional local imports ────────────────
#
# Section 15 below (Pipeline Invariants) exercises the span-classifier
# pipeline modules directly -- no live server, no live LLM. Those modules
# pull in heavier local/optional dependencies (openai, rapidfuzz, fastapi,
# aiofiles, ...) that the rest of this file has never required: historically
# test_api.py only needs `requests` and can run from a minimal environment
# against a remote deployed server. To avoid regressing that use case, these
# imports are best-effort: if they fail, Section 15's tests SKIP individually
# (via the usual "SKIP:" TestFailure convention) instead of the whole script
# refusing to import.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_PIPELINE_IMPORT_ERROR = None
try:
    from span_tokenizer import tokenize, reassemble, validate_spans
    from generate_script import process_chunk, split_into_chunks
    from review_script import normalize_text, check_text_loss
    from default_prompts import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT
    from test_span_integration import FakeClient, FakeResponse, labels_for
    from test_epub_extract import make_epub, opf, extract_epub_text

    # app.py pulls in `project` (-> tts.py -> numpy/torch/etc.) purely for
    # unrelated TTS/project-management functionality this file's offline
    # test does not exercise. Stub it out before importing, same pattern as
    # test_canon_wiring.py, so this best-effort import block degrades
    # gracefully rather than requiring torch just to test a read-only helper.
    import types as _types
    if 'project' not in sys.modules:
        _fake_project = _types.ModuleType('project')

        class _FakeProjectManager:
            def __init__(self, *args, **kwargs):
                pass

            def __getattr__(self, name):
                def _noop(*args, **kwargs):
                    return None
                return _noop

        _fake_project.ProjectManager = _FakeProjectManager
        sys.modules['project'] = _fake_project

    import app as app_module
except Exception as _e:  # pragma: no cover - environment-dependent
    _PIPELINE_IMPORT_ERROR = _e

# ── Global state ─────────────────────────────────────────────

BASE_URL = ""
FULL_MODE = False
OFFLINE_ONLY = False
TEST_PREFIX = "_test_"

results = {"passed": 0, "failed": 0, "skipped": 0}
failures = []
shared = {}  # state shared between dependent tests


# ── Helpers ──────────────────────────────────────────────────

class TestFailure(Exception):
    pass


def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def run_test(name, func, requires_full=False):
    if requires_full and not FULL_MODE:
        print(f"  [ SKIP ] {name} (requires --full)")
        results["skipped"] += 1
        return
    try:
        func()
        print(f"  [ PASS ] {name}")
        results["passed"] += 1
    except TestFailure as e:
        msg = str(e)
        if msg.startswith("SKIP:"):
            print(f"  [ SKIP ] {name} ({msg[5:].strip()})")
            results["skipped"] += 1
        else:
            print(f"  [ FAIL ] {name}")
            print(f"           {msg}")
            results["failed"] += 1
            failures.append((name, msg))
    except Exception as e:
        print(f"  [ FAIL ] {name}")
        print(f"           {type(e).__name__}: {e}")
        results["failed"] += 1
        failures.append((name, str(e)))


def assert_status(resp, expected=200, msg=""):
    if resp.status_code != expected:
        body = resp.text[:500]
        raise TestFailure(
            f"Expected {expected}, got {resp.status_code}. {msg}\n"
            f"           Body: {body}"
        )


def assert_key(data, key):
    if key not in data:
        raise TestFailure(f"Missing key '{key}' in: {json.dumps(data)[:300]}")


def wait_for_task(task, timeout=120, poll_interval=2):
    """Poll /api/status/{task} until it stops running or timeout is reached."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(f"{BASE_URL}/api/status/{task}", timeout=10)
        if r.status_code == 200 and not r.json().get("running"):
            return True
        time.sleep(poll_interval)
    return False


def get(path, **kwargs):
    return requests.get(f"{BASE_URL}{path}", timeout=30, **kwargs)


def post(path, **kwargs):
    return requests.post(f"{BASE_URL}{path}", timeout=kwargs.pop("timeout", 30), **kwargs)


def delete(path, **kwargs):
    return requests.delete(f"{BASE_URL}{path}", timeout=30, **kwargs)


# ── Section 1: Server ───────────────────────────────────────

def test_server_reachable():
    r = get("/")
    assert_status(r, 200)
    if "text/html" not in r.headers.get("content-type", ""):
        raise TestFailure(f"Expected HTML, got {r.headers.get('content-type')}")


# ── Section 2: Config ───────────────────────────────────────

def test_get_config():
    r = get("/api/config")
    assert_status(r, 200)
    data = r.json()
    assert_key(data, "llm")
    assert_key(data, "tts")
    # current_file should always be present (may be null)
    assert_key(data, "current_file")


def test_save_config_roundtrip():
    # Read original
    r = get("/api/config")
    assert_status(r, 200)
    original = r.json()
    shared["original_config"] = original

    # Build test config with modified language
    test_config = {
        "llm": original["llm"],
        "tts": {**original.get("tts", {}), "language": "_test_roundtrip_lang"},
        "prompts": original.get("prompts"),
        "generation": original.get("generation"),
    }
    test_config["tts"].setdefault("mode", "external")
    test_config["tts"].setdefault("url", "http://127.0.0.1:7860")
    test_config["tts"].setdefault("device", "auto")

    # Save modified
    r = post("/api/config", json=test_config)
    assert_status(r, 200)

    # Read back and verify
    r = get("/api/config")
    assert_status(r, 200)
    readback = r.json()
    if readback.get("tts", {}).get("language") != "_test_roundtrip_lang":
        raise TestFailure("Config round-trip failed: language not persisted")

    # Verify generation section persists
    if original.get("generation") and not readback.get("generation"):
        raise TestFailure("Config round-trip failed: generation section dropped")

    # Verify review prompts persist through config save
    readback_prompts = readback.get("prompts", {})
    if original.get("prompts", {}).get("review_system_prompt"):
        if not readback_prompts.get("review_system_prompt"):
            raise TestFailure("Config round-trip failed: review_system_prompt dropped")

    # Verify persona prompts persist through config save
    if original.get("prompts", {}).get("persona_system_prompt"):
        if not readback_prompts.get("persona_system_prompt"):
            raise TestFailure("Config round-trip failed: persona_system_prompt dropped")

    # Restore original
    restore = {
        "llm": original["llm"],
        "tts": original.get("tts", {"mode": "external", "url": "http://127.0.0.1:7860", "device": "auto"}),
        "prompts": original.get("prompts"),
        "generation": original.get("generation"),
    }
    post("/api/config", json=restore)


def test_save_pause_config_roundtrip():
    # Read original
    r = get("/api/config")
    assert_status(r, 200)
    original = r.json()

    # Save with custom pause values
    test_config = {
        "llm": original["llm"],
        "tts": {
            **original.get("tts", {}),
            "pause_between_speakers_ms": 1000,
            "pause_same_speaker_ms": 400,
        },
        "prompts": original.get("prompts"),
        "generation": original.get("generation"),
    }
    test_config["tts"].setdefault("mode", "external")
    test_config["tts"].setdefault("url", "http://127.0.0.1:7860")
    test_config["tts"].setdefault("device", "auto")

    r = post("/api/config", json=test_config)
    assert_status(r, 200)

    # Read back and verify
    r = get("/api/config")
    assert_status(r, 200)
    readback = r.json()
    tts = readback.get("tts", {})
    if tts.get("pause_between_speakers_ms") != 1000:
        raise TestFailure(f"pause_between_speakers_ms not persisted: {tts.get('pause_between_speakers_ms')}")
    if tts.get("pause_same_speaker_ms") != 400:
        raise TestFailure(f"pause_same_speaker_ms not persisted: {tts.get('pause_same_speaker_ms')}")

    # Restore original
    restore = {
        "llm": original["llm"],
        "tts": original.get("tts", {"mode": "external", "url": "http://127.0.0.1:7860", "device": "auto"}),
        "prompts": original.get("prompts"),
        "generation": original.get("generation"),
    }
    post("/api/config", json=restore)


def test_pause_config_defaults():
    """Verify pause fields have sensible defaults when not explicitly set."""
    r = get("/api/config")
    assert_status(r, 200)
    tts = r.json().get("tts", {})
    pause_between = tts.get("pause_between_speakers_ms")
    pause_same = tts.get("pause_same_speaker_ms")
    if pause_between is None:
        raise TestFailure("pause_between_speakers_ms missing from config response")
    if pause_same is None:
        raise TestFailure("pause_same_speaker_ms missing from config response")
    if not isinstance(pause_between, int) or pause_between < 0:
        raise TestFailure(f"Invalid pause_between_speakers_ms: {pause_between}")
    if not isinstance(pause_same, int) or pause_same < 0:
        raise TestFailure(f"Invalid pause_same_speaker_ms: {pause_same}")


def test_save_review_prompts_roundtrip():
    # Read current config
    r = get("/api/config")
    assert_status(r, 200)
    original = r.json()

    # Save config with custom review prompts
    test_config = {
        "llm": original["llm"],
        "tts": original.get("tts", {"mode": "local", "url": "http://127.0.0.1:7860", "device": "auto"}),
        "prompts": {
            **(original.get("prompts") or {}),
            "review_system_prompt": f"{TEST_PREFIX}review_sys",
            "review_user_prompt": f"{TEST_PREFIX}review_usr",
        },
        "generation": original.get("generation"),
    }
    r = post("/api/config", json=test_config)
    assert_status(r, 200)

    # Read back and verify
    r = get("/api/config")
    assert_status(r, 200)
    readback = r.json()
    prompts = readback.get("prompts", {})
    if prompts.get("review_system_prompt") != f"{TEST_PREFIX}review_sys":
        raise TestFailure(f"review_system_prompt not persisted: {prompts.get('review_system_prompt')}")
    if prompts.get("review_user_prompt") != f"{TEST_PREFIX}review_usr":
        raise TestFailure(f"review_user_prompt not persisted: {prompts.get('review_user_prompt')}")

    # Restore original
    restore = {
        "llm": original["llm"],
        "tts": original.get("tts", {"mode": "local", "url": "http://127.0.0.1:7860", "device": "auto"}),
        "prompts": original.get("prompts"),
        "generation": original.get("generation"),
    }
    post("/api/config", json=restore)


def test_save_persona_prompts_roundtrip():
    # Read current config
    r = get("/api/config")
    assert_status(r, 200)
    original = r.json()

    # Save config with custom persona prompts
    test_config = {
        "llm": original["llm"],
        "tts": original.get("tts", {"mode": "local", "url": "http://127.0.0.1:7860", "device": "auto"}),
        "prompts": {
            **(original.get("prompts") or {}),
            "persona_system_prompt": f"{TEST_PREFIX}persona_sys",
            "persona_user_prompt": f"{TEST_PREFIX}persona_usr",
            "persona_advanced_prompt": f"{TEST_PREFIX}persona_adv",
        },
        "generation": original.get("generation"),
    }
    r = post("/api/config", json=test_config)
    assert_status(r, 200)

    # Read back and verify
    r = get("/api/config")
    assert_status(r, 200)
    readback = r.json()
    prompts = readback.get("prompts", {})
    if prompts.get("persona_system_prompt") != f"{TEST_PREFIX}persona_sys":
        raise TestFailure(f"persona_system_prompt not persisted: {prompts.get('persona_system_prompt')}")
    if prompts.get("persona_user_prompt") != f"{TEST_PREFIX}persona_usr":
        raise TestFailure(f"persona_user_prompt not persisted: {prompts.get('persona_user_prompt')}")
    if prompts.get("persona_advanced_prompt") != f"{TEST_PREFIX}persona_adv":
        raise TestFailure(f"persona_advanced_prompt not persisted: {prompts.get('persona_advanced_prompt')}")

    # Restore original
    restore = {
        "llm": original["llm"],
        "tts": original.get("tts", {"mode": "local", "url": "http://127.0.0.1:7860", "device": "auto"}),
        "prompts": original.get("prompts"),
        "generation": original.get("generation"),
    }
    post("/api/config", json=restore)


def test_get_default_prompts():
    r = get("/api/default_prompts")
    assert_status(r, 200)
    data = r.json()
    assert_key(data, "system_prompt")
    assert_key(data, "user_prompt")
    if not data["system_prompt"]:
        raise TestFailure("system_prompt is empty")
    assert_key(data, "review_system_prompt")
    assert_key(data, "review_user_prompt")
    if not data["review_system_prompt"]:
        raise TestFailure("review_system_prompt is empty")
    if not data["review_user_prompt"]:
        raise TestFailure("review_user_prompt is empty")
    assert_key(data, "persona_system_prompt")
    assert_key(data, "persona_user_prompt")
    assert_key(data, "persona_advanced_prompt")
    if not data["persona_system_prompt"]:
        raise TestFailure("persona_system_prompt is empty")
    if not data["persona_user_prompt"]:
        raise TestFailure("persona_user_prompt is empty")
    if not data["persona_advanced_prompt"]:
        raise TestFailure("persona_advanced_prompt is empty")


# ── Section 2b: System Stats ───────────────────────────────

def test_system_stats():
    r = get("/api/system/stats")
    assert_status(r, 200)
    data = r.json()
    assert_key(data, "gpu")
    assert_key(data, "disk")
    disk = data["disk"]
    assert_key(disk, "free_gb")
    assert_key(disk, "low_space")
    if not isinstance(disk["free_gb"], (int, float)):
        raise TestFailure(f"disk.free_gb should be numeric, got {type(disk['free_gb']).__name__}")
    if not isinstance(disk["low_space"], bool):
        raise TestFailure(f"disk.low_space should be bool, got {type(disk['low_space']).__name__}")


# ── Section 3: Upload ───────────────────────────────────────

def test_upload_file():
    content = b"Chapter One\nIt was a dark and stormy night.\nThe end."
    files = {"file": (f"{TEST_PREFIX}upload.txt", io.BytesIO(content), "text/plain")}
    r = post("/api/upload", files=files)
    assert_status(r, 200)
    data = r.json()
    assert_key(data, "filename")
    assert_key(data, "path")
    if data["filename"] != f"{TEST_PREFIX}upload.txt":
        raise TestFailure(f"Unexpected filename: {data['filename']}")


# ── Section 4: Annotated Script ─────────────────────────────

def test_get_annotated_script():
    r = get("/api/annotated_script")
    if r.status_code == 404:
        shared["has_script"] = False
        return  # acceptable — no script loaded
    assert_status(r, 200)
    data = r.json()
    if not isinstance(data, list):
        raise TestFailure(f"Expected list, got {type(data).__name__}")
    shared["has_script"] = True


# ── Section 5: Scripts CRUD ─────────────────────────────────

def test_save_script():
    if not shared.get("has_script"):
        raise TestFailure("SKIP: no annotated script loaded")
    r = post("/api/scripts/save", json={"name": f"{TEST_PREFIX}script"})
    assert_status(r, 200)
    data = r.json()
    if data.get("status") != "saved":
        raise TestFailure(f"Expected status=saved, got {data}")


def test_list_scripts():
    r = get("/api/scripts")
    assert_status(r, 200)
    data = r.json()
    if not isinstance(data, list):
        raise TestFailure(f"Expected list, got {type(data).__name__}")
    if shared.get("has_script"):
        names = [s["name"] for s in data]
        if f"{TEST_PREFIX}script" not in names:
            raise TestFailure(f"Saved script not in list: {names}")


def test_load_script():
    if not shared.get("has_script"):
        raise TestFailure("SKIP: no annotated script loaded")
    r = post("/api/scripts/load", json={"name": f"{TEST_PREFIX}script"})
    assert_status(r, 200)
    data = r.json()
    if data.get("status") != "loaded":
        raise TestFailure(f"Expected status=loaded, got {data}")


def test_delete_script():
    if not shared.get("has_script"):
        raise TestFailure("SKIP: no annotated script loaded")
    r = delete(f"/api/scripts/{TEST_PREFIX}script")
    assert_status(r, 200)
    data = r.json()
    if data.get("status") != "deleted":
        raise TestFailure(f"Expected status=deleted, got {data}")


def test_delete_script_404():
    r = delete(f"/api/scripts/{TEST_PREFIX}nonexistent_xyz")
    assert_status(r, 404)


# ── Section 6: Voices ───────────────────────────────────────

def test_get_voices():
    r = get("/api/voices")
    assert_status(r, 200)
    data = r.json()
    if not isinstance(data, list):
        raise TestFailure(f"Expected list, got {type(data).__name__}")


def test_save_voice_config():
    r = post("/api/save_voice_config", json={
        f"{TEST_PREFIX}voice": {
            "type": "custom",
            "voice": "Ryan",
            "character_style": "",
            "seed": "-1"
        }
    })
    assert_status(r, 200)
    data = r.json()
    if data.get("status") != "saved":
        raise TestFailure(f"Expected status=saved, got {data}")


# ── Section 7: Chunks ───────────────────────────────────────

def test_get_chunks():
    r = get("/api/chunks")
    assert_status(r, 200)
    data = r.json()
    if not isinstance(data, list):
        raise TestFailure(f"Expected list, got {type(data).__name__}")
    shared["has_chunks"] = len(data) > 0
    if data:
        shared["chunk0_original"] = {
            "text": data[0].get("text", ""),
            "instruct": data[0].get("instruct", ""),
            "speaker": data[0].get("speaker", ""),
        }


def test_update_chunk():
    if not shared.get("has_chunks"):
        raise TestFailure("SKIP: no chunks available")

    r = post("/api/chunks/0", json={
        "text": f"{TEST_PREFIX}updated_text",
        "instruct": f"{TEST_PREFIX}instruct"
    })
    assert_status(r, 200)
    data = r.json()
    if data.get("text") != f"{TEST_PREFIX}updated_text":
        raise TestFailure(f"Chunk text not updated: {data.get('text')}")

    # Restore original
    orig = shared.get("chunk0_original", {})
    post("/api/chunks/0", json=orig)


def test_update_chunk_pause_after():
    """Setting pause_after on a chunk persists and does not reset status."""
    if not shared.get("has_chunks"):
        raise TestFailure("SKIP: no chunks available")

    # Read current chunk 0 status
    r = get("/api/chunks")
    assert_status(r, 200)
    original_status = r.json()[0].get("status")

    # Set pause_after
    r = post("/api/chunks/0", json={"pause_after": 3000})
    assert_status(r, 200)
    data = r.json()
    if data.get("pause_after") != 3000:
        raise TestFailure(f"pause_after not set: {data.get('pause_after')}")

    # Verify status was NOT reset (pause_after is merge-time only)
    if data.get("status") != original_status:
        raise TestFailure(
            f"Status changed from '{original_status}' to '{data.get('status')}' "
            f"— pause_after should not reset status"
        )

    # Read back via GET to confirm persistence
    r = get("/api/chunks")
    assert_status(r, 200)
    chunk0 = r.json()[0]
    if chunk0.get("pause_after") != 3000:
        raise TestFailure(f"pause_after not persisted on read-back: {chunk0.get('pause_after')}")

    # Clear pause_after by sending null
    r = post("/api/chunks/0", json={"pause_after": None})
    assert_status(r, 200)
    data = r.json()
    if data.get("pause_after") is not None:
        raise TestFailure(f"pause_after not cleared: {data.get('pause_after')}")

    # Verify key is removed from JSON (not just set to null)
    r = get("/api/chunks")
    assert_status(r, 200)
    chunk0 = r.json()[0]
    if "pause_after" in chunk0:
        raise TestFailure(f"pause_after key should be removed after clearing, got: {chunk0.get('pause_after')}")


def test_update_chunk_pause_after_zero():
    """pause_after=0 is a valid override (no silence)."""
    if not shared.get("has_chunks"):
        raise TestFailure("SKIP: no chunks available")

    r = post("/api/chunks/0", json={"pause_after": 0})
    assert_status(r, 200)
    data = r.json()
    if data.get("pause_after") != 0:
        raise TestFailure(f"pause_after=0 not set correctly: {data.get('pause_after')}")

    # Clean up
    post("/api/chunks/0", json={"pause_after": None})


def test_update_chunk_pause_after_negative():
    """Negative pause_after should be clamped to 0."""
    if not shared.get("has_chunks"):
        raise TestFailure("SKIP: no chunks available")

    r = post("/api/chunks/0", json={"pause_after": -500})
    assert_status(r, 200)
    data = r.json()
    if data.get("pause_after") != 0:
        raise TestFailure(f"Negative pause_after should clamp to 0, got: {data.get('pause_after')}")

    # Clean up
    post("/api/chunks/0", json={"pause_after": None})


def test_update_chunk_404():
    r = post("/api/chunks/99999", json={"text": "nope"})
    assert_status(r, 404)


def test_insert_chunk():
    if not shared.get("has_chunks"):
        raise TestFailure("SKIP: no chunks available")

    # Get initial count
    r = get("/api/chunks")
    assert_status(r, 200)
    initial_chunks = r.json()
    initial_count = len(initial_chunks)

    # Insert after index 0
    r = post("/api/chunks/0/insert")
    assert_status(r, 200)
    data = r.json()
    if data.get("status") != "ok":
        raise TestFailure(f"Expected status=ok, got {data}")
    if data.get("total") != initial_count + 1:
        raise TestFailure(f"Expected total={initial_count + 1}, got {data.get('total')}")

    # Verify the new chunk exists at index 1 with empty text
    r = get("/api/chunks")
    assert_status(r, 200)
    chunks = r.json()
    if len(chunks) != initial_count + 1:
        raise TestFailure(f"Chunk count mismatch: expected {initial_count + 1}, got {len(chunks)}")
    if chunks[1].get("text") != "":
        raise TestFailure(f"Inserted chunk should have empty text, got: {chunks[1].get('text')}")

    # Store index for cleanup in delete test
    shared["inserted_chunk_index"] = 1


def test_insert_chunk_404():
    r = post("/api/chunks/99999/insert")
    assert_status(r, 404)


def test_delete_chunk():
    if not shared.get("has_chunks"):
        raise TestFailure("SKIP: no chunks available")

    idx = shared.get("inserted_chunk_index")
    if idx is None:
        raise TestFailure("SKIP: no inserted chunk to delete")

    # Get count before delete
    r = get("/api/chunks")
    assert_status(r, 200)
    before_count = len(r.json())

    r = delete(f"/api/chunks/{idx}")
    assert_status(r, 200)
    data = r.json()
    assert_key(data, "deleted")
    assert_key(data, "total")
    if data["total"] != before_count - 1:
        raise TestFailure(f"Expected total={before_count - 1}, got {data['total']}")

    # Save deleted chunk for restore test
    shared["deleted_chunk"] = data["deleted"]
    shared["deleted_chunk_index"] = idx


def test_delete_chunk_invalid():
    r = delete("/api/chunks/99999")
    assert_status(r, 400)


def test_restore_chunk():
    if not shared.get("deleted_chunk"):
        raise TestFailure("SKIP: no deleted chunk to restore")

    r = get("/api/chunks")
    assert_status(r, 200)
    before_count = len(r.json())

    r = post("/api/chunks/restore", json={
        "chunk": shared["deleted_chunk"],
        "at_index": shared["deleted_chunk_index"]
    })
    assert_status(r, 200)
    data = r.json()
    if data.get("status") != "ok":
        raise TestFailure(f"Expected status=ok, got {data}")
    if data.get("total") != before_count + 1:
        raise TestFailure(f"Expected total={before_count + 1}, got {data.get('total')}")

    # Clean up: delete the restored chunk so we leave chunks as we found them
    delete(f"/api/chunks/{shared['deleted_chunk_index']}")


# ── Section 8: Status Polling ────────────────────────────────

def test_status_known_tasks():
    task_names = [
        "script", "audio", "audacity_export",
        "review", "lora_training", "dataset_gen", "dataset_builder",
        "persona",
        "preparer", "batch_preparer"
    ]
    for name in task_names:
        r = get(f"/api/status/{name}")
        assert_status(r, 200, msg=f"task={name}")
        data = r.json()
        if "running" not in data:
            raise TestFailure(f"Missing 'running' key for task '{name}'")
        if "logs" not in data:
            raise TestFailure(f"Missing 'logs' key for task '{name}'")


def test_status_unknown_task():
    r = get(f"/api/status/{TEST_PREFIX}fake_task")
    assert_status(r, 404)


# ── Section: Preparer ─────────────────────────────────────────

def test_features_endpoint():
    r = get("/api/features")
    assert_status(r, 200)
    data = r.json()
    # Preparer flag must exist and be a boolean so the frontend can
    # decide whether to show the tab.
    if not isinstance(data.get("preparer"), bool):
        raise TestFailure(f"features.preparer must be bool, got {data}")


def test_preparer_status():
    r = get("/api/status/preparer")
    assert_status(r, 200)
    data = r.json()
    assert_key(data, "running")
    assert_key(data, "logs")
    assert_key(data, "status")


def test_batch_preparer_status():
    r = get("/api/status/batch_preparer")
    assert_status(r, 200)
    data = r.json()
    assert_key(data, "running")
    assert_key(data, "logs")
    assert_key(data, "tasks")


def test_preparer_cancel_when_idle():
    r = post("/api/preparer/cancel", json={})
    assert_status(r, 400)


def test_preparer_list_outputs():
    r = get("/api/preparer/list")
    assert_status(r, 200)
    data = r.json()
    assert_key(data, "files")


def test_preparer_download_404():
    r = get("/api/preparer/download/nonexistent_xyz.zip")
    assert_status(r, 404)


def test_batch_preparer_start_schema():
    r = post("/api/preparer/batch/start", json={"tasks": [
        {"audio_filename": "test.wav", "output_filename": "test.zip"}
    ]})
    # 200 = started (script present), 400 = already running, 503 = script absent
    if r.status_code not in (200, 400, 503):
        raise TestFailure(f"Unexpected status {r.status_code}: {r.text[:200]}")


def test_batch_preparer_cancel():
    r = post("/api/preparer/batch/cancel", json={})
    assert_status(r, 200)
    data = r.json()
    assert_key(data, "status")


# ── Section 9: Voice Design ─────────────────────────────────

def test_voice_design_list():
    r = get("/api/voice_design/list")
    assert_status(r, 200)
    data = r.json()
    if not isinstance(data, list):
        raise TestFailure(f"Expected list, got {type(data).__name__}")


def test_voice_design_delete_404():
    r = delete(f"/api/voice_design/{TEST_PREFIX}fake_id")
    assert_status(r, 404)


def test_voice_design_preview():
    r = post("/api/voice_design/preview", json={
        "description": "A clear young male voice with a steady tone",
        "sample_text": "This is a test of voice design.",
    })
    assert_status(r, 200)
    data = r.json()
    assert_key(data, "audio_url")
    shared["preview_file"] = data["audio_url"].split("/")[-1]


def test_voice_design_save_and_delete():
    preview_file = shared.get("preview_file")
    if not preview_file:
        raise TestFailure("SKIP: no preview file from previous test")

    r = post("/api/voice_design/save", json={
        "name": f"{TEST_PREFIX}voice_design",
        "description": "Test voice",
        "sample_text": "Test text",
        "preview_file": preview_file
    })
    assert_status(r, 200)
    data = r.json()
    assert_key(data, "voice_id")
    voice_id = data["voice_id"]

    # Delete it
    r = delete(f"/api/voice_design/{voice_id}")
    assert_status(r, 200)


# ── Section 9b: Clone Voices ────────────────────────────────

def test_clone_voices_list():
    r = get("/api/clone_voices/list")
    assert_status(r, 200)
    data = r.json()
    if not isinstance(data, list):
        raise TestFailure(f"Expected list, got {type(data).__name__}")


def test_clone_voices_upload_bad_format():
    files = {"file": ("test.txt", b"not audio", "text/plain")}
    r = requests.post(f"{BASE_URL}/api/clone_voices/upload", files=files)
    assert_status(r, 400)


def test_clone_voices_delete_404():
    r = delete(f"/api/clone_voices/{TEST_PREFIX}fake_id")
    assert_status(r, 404)


def test_clone_voices_upload_and_delete():
    # Create a minimal WAV file (44-byte header + silence)
    import struct
    sample_rate = 16000
    num_samples = 16000  # 1 second
    data_size = num_samples * 2
    wav_header = struct.pack('<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + data_size, b'WAVE',
        b'fmt ', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
        b'data', data_size)
    wav_bytes = wav_header + b'\x00' * data_size

    files = {"file": (f"{TEST_PREFIX}clone_test.wav", wav_bytes, "audio/wav")}
    r = requests.post(f"{BASE_URL}/api/clone_voices/upload", files=files)
    assert_status(r, 200)
    data = r.json()
    assert_key(data, "voice_id")
    assert_key(data, "filename")
    voice_id = data["voice_id"]

    # Verify it appears in list
    r = get("/api/clone_voices/list")
    assert_status(r, 200)
    found = any(v["id"] == voice_id for v in r.json())
    if not found:
        raise TestFailure(f"Uploaded voice {voice_id} not found in list")

    # Delete it
    r = delete(f"/api/clone_voices/{voice_id}")
    assert_status(r, 200)

    # Verify it's gone
    r = get("/api/clone_voices/list")
    found = any(v["id"] == voice_id for v in r.json())
    if found:
        raise TestFailure(f"Deleted voice {voice_id} still in list")


# ── Section 10: LoRA Datasets ───────────────────────────────

def test_lora_list_datasets():
    r = get("/api/lora/datasets")
    assert_status(r, 200)
    data = r.json()
    if not isinstance(data, list):
        raise TestFailure(f"Expected list, got {type(data).__name__}")


def test_lora_delete_dataset_404():
    r = delete(f"/api/lora/datasets/{TEST_PREFIX}fake_ds")
    assert_status(r, 404)


def test_lora_upload_bad_file():
    files = {"file": (f"{TEST_PREFIX}bad.txt", io.BytesIO(b"not a zip"), "text/plain")}
    r = post("/api/lora/upload_dataset", files=files)
    # Should fail — not a valid zip
    if r.status_code < 400:
        raise TestFailure(f"Expected error for non-zip upload, got {r.status_code}")


# ── Section 11: LoRA Models ─────────────────────────────────

def test_lora_list_models():
    r = get("/api/lora/models")
    assert_status(r, 200)
    data = r.json()
    if not isinstance(data, list):
        raise TestFailure(f"Expected list, got {type(data).__name__}")
    # Verify built-in adapters have 'downloaded' field
    for m in data:
        if m.get("builtin"):
            if "downloaded" not in m:
                raise TestFailure(f"Built-in adapter {m['id']} missing 'downloaded' field")
    shared["lora_models"] = data


def test_lora_download_invalid():
    r = post(f"/api/lora/download/{TEST_PREFIX}fake_adapter", json={})
    if r.status_code < 400:
        raise TestFailure(f"Expected error for invalid adapter, got {r.status_code}")


def test_lora_delete_model_404():
    r = delete(f"/api/lora/models/{TEST_PREFIX}fake_model")
    assert_status(r, 404)


def test_lora_train_bad_dataset():
    r = post("/api/lora/train", json={
        "name": f"{TEST_PREFIX}model",
        "dataset_id": f"{TEST_PREFIX}nonexistent_ds"
    })
    # Should fail — dataset does not exist
    if r.status_code < 400:
        raise TestFailure(f"Expected error for bad dataset, got {r.status_code}")


def test_lora_preview_404():
    r = post(f"/api/lora/preview/{TEST_PREFIX}fake_adapter")
    assert_status(r, 404)


def test_lora_preview():
    models = shared.get("lora_models", [])
    if not models:
        raise TestFailure("SKIP: no LoRA models available")
    adapter = models[0]
    r = post(f"/api/lora/preview/{adapter['id']}", timeout=120)
    assert_status(r, 200)
    data = r.json()
    assert_key(data, "audio_url")


# ── Section 12: Dataset Builder CRUD ────────────────────────

def test_dataset_builder_list():
    r = get("/api/dataset_builder/list")
    assert_status(r, 200)
    data = r.json()
    if not isinstance(data, list):
        raise TestFailure(f"Expected list, got {type(data).__name__}")


def test_dataset_builder_create():
    r = post("/api/dataset_builder/create", json={
        "name": f"{TEST_PREFIX}builder_proj"
    })
    assert_status(r, 200)
    data = r.json()
    assert_key(data, "name")


def test_dataset_builder_update_meta():
    r = post("/api/dataset_builder/update_meta", json={
        "name": f"{TEST_PREFIX}builder_proj",
        "description": "A test voice description",
        "global_seed": "42"
    })
    assert_status(r, 200)


def test_dataset_builder_update_rows():
    r = post("/api/dataset_builder/update_rows", json={
        "name": f"{TEST_PREFIX}builder_proj",
        "rows": [
            {"emotion": "neutral", "text": "Hello world.", "seed": ""},
            {"emotion": "happy", "text": "Great to see you!", "seed": ""}
        ]
    })
    assert_status(r, 200)
    data = r.json()
    if data.get("sample_count") != 2:
        raise TestFailure(f"Expected sample_count=2, got {data.get('sample_count')}")


def test_dataset_builder_status():
    r = get(f"/api/dataset_builder/status/{TEST_PREFIX}builder_proj")
    assert_status(r, 200)
    data = r.json()
    assert_key(data, "description")
    assert_key(data, "samples")
    assert_key(data, "running")
    assert_key(data, "logs")
    if len(data["samples"]) != 2:
        raise TestFailure(f"Expected 2 samples, got {len(data['samples'])}")


def test_dataset_builder_cancel():
    r = post("/api/dataset_builder/cancel")
    assert_status(r, 200)
    data = r.json()
    if data.get("status") not in ("not_running", "cancelling"):
        raise TestFailure(f"Unexpected cancel status: {data}")


def test_dataset_builder_save_no_samples():
    r = post("/api/dataset_builder/save", json={
        "name": f"{TEST_PREFIX}builder_proj",
        "ref_index": 0
    })
    # Should fail — no completed samples
    if r.status_code < 400:
        raise TestFailure(f"Expected error for save with no samples, got {r.status_code}")


def test_dataset_builder_delete():
    r = delete(f"/api/dataset_builder/{TEST_PREFIX}builder_proj")
    assert_status(r, 200)
    data = r.json()
    if data.get("status") != "deleted":
        raise TestFailure(f"Expected status=deleted, got {data}")


def test_dataset_builder_delete_404():
    r = delete(f"/api/dataset_builder/{TEST_PREFIX}nonexistent")
    assert_status(r, 404)


# ── Section 13: Persona Generation ──────────────────────────

def test_cancel_persona_not_running():
    """Cancel endpoint returns idle when not running."""
    r = post("/api/cancel_persona", json={})
    assert_status(r, 200)
    data = r.json()
    if data.get("status") not in ("idle", "cancelling"):
        raise TestFailure(f"Expected status idle or cancelling, got {data}")


# ── Section 14: Merge / Export ──────────────────────────────

def test_get_audiobook():
    r = get("/api/audiobook")
    if r.status_code == 404:
        return  # acceptable — no audiobook generated yet
    assert_status(r, 200)


def test_get_audiobook_m4b():
    r = get("/api/audiobook_m4b")
    if r.status_code == 404:
        return  # acceptable — no M4B generated yet
    assert_status(r, 200)


def test_get_audacity_export():
    r = get("/api/export_audacity")
    if r.status_code == 404:
        return  # acceptable — no export generated yet
    assert_status(r, 200)


# ── Section 14: Full Tests — Generation ─────────────────────

def test_generate_script():
    r = post("/api/generate_script")
    if r.status_code == 400:
        raise TestFailure("SKIP: prerequisite not met (no uploaded file or already running)")
    assert_status(r, 200)
    data = r.json()
    if data.get("status") != "started":
        raise TestFailure(f"Expected status=started, got {data}")


def test_generate_script_single_speaker():
    r = post("/api/generate_script", json={
        "single_speaker": True,
        "speaker_name": "Narrator",
        "instruct": "Neutral narration."
    })
    if r.status_code == 400:
        raise TestFailure("SKIP: prerequisite not met (no uploaded file or already running)")
    assert_status(r, 200)
    data = r.json()
    if data.get("status") != "started":
        raise TestFailure(f"Expected status=started, got {data}")


def test_review_script():
    if not shared.get("has_script"):
        raise TestFailure("SKIP: no annotated script loaded")
    r = post("/api/review_script")
    if r.status_code == 400:
        raise TestFailure("SKIP: already running")
    assert_status(r, 200)
    data = r.json()
    if data.get("status") != "started":
        raise TestFailure(f"Expected status=started, got {data}")



def test_generate_chunk():
    if not shared.get("has_chunks"):
        raise TestFailure("SKIP: no chunks available")
    r = post("/api/chunks/0/generate")
    assert_status(r, 200)


def test_generate_batch():
    if not shared.get("has_chunks"):
        raise TestFailure("SKIP: no chunks available")
    r = post("/api/generate_batch", json={"indices": [0]})
    if r.status_code == 400:
        raise TestFailure("SKIP: audio generation already running")
    assert_status(r, 200)
    data = r.json()
    if data.get("status") != "started":
        raise TestFailure(f"Expected status=started, got {data}")
    # Wait for batch to finish so subsequent tests don't conflict
    if not wait_for_task("audio", timeout=120):
        raise TestFailure("generate_batch did not complete within 120s")


def test_generate_batch_fast():
    if not shared.get("has_chunks"):
        raise TestFailure("SKIP: no chunks available")
    # Wait for any prior generation to finish
    if not wait_for_task("audio", timeout=120):
        raise TestFailure("SKIP: prior audio generation did not finish in time")
    r = post("/api/generate_batch_fast", json={"indices": [0]})
    if r.status_code == 400:
        raise TestFailure("SKIP: audio generation already running")
    assert_status(r, 200)
    data = r.json()
    if data.get("status") != "started":
        raise TestFailure(f"Expected status=started, got {data}")


def test_cancel_audio():
    """Cancel endpoint works when nothing is running (resets stuck chunks)."""
    r = post("/api/cancel_audio", json={})
    assert_status(r, 200)
    data = r.json()
    if data.get("status") not in ("not_running", "cancelling"):
        raise TestFailure(f"Expected status not_running or cancelling, got {data}")


def test_export_audacity():
    r = post("/api/export_audacity")
    if r.status_code == 400:
        raise TestFailure("SKIP: already running")
    assert_status(r, 200)
    data = r.json()
    if data.get("status") != "started":
        raise TestFailure(f"Expected status=started, got {data}")


def test_lora_test_model():
    models = shared.get("lora_models", [])
    if not models:
        raise TestFailure("SKIP: no LoRA models available")
    adapter = models[0]
    r = post("/api/lora/test", json={
        "adapter_id": adapter["id"],
        "text": "This is a test of the LoRA voice.",
        "instruct": "Neutral, even delivery."
    }, timeout=120)
    assert_status(r, 200)
    data = r.json()
    assert_key(data, "audio_url")


def test_lora_generate_dataset():
    r = post("/api/lora/generate_dataset", json={
        "name": f"{TEST_PREFIX}dataset",
        "description": "A clear young male voice",
        "samples": [
            {"emotion": "neutral", "text": "Hello, this is a test sample."},
            {"emotion": "happy", "text": "Great to see you today!"}
        ]
    })
    if r.status_code == 400:
        raise TestFailure("SKIP: already running or bad request")
    assert_status(r, 200)
    data = r.json()
    if data.get("status") != "started":
        raise TestFailure(f"Expected status=started, got {data}")


def test_dataset_builder_generate_sample():
    # Create a temp project for this test
    post("/api/dataset_builder/create", json={"name": f"{TEST_PREFIX}gen_proj"})
    post("/api/dataset_builder/update_rows", json={
        "name": f"{TEST_PREFIX}gen_proj",
        "rows": [{"emotion": "neutral", "text": "Hello world.", "seed": ""}]
    })

    r = post("/api/dataset_builder/generate_sample", json={
        "description": "A clear male voice",
        "text": "Hello world.",
        "dataset_name": f"{TEST_PREFIX}gen_proj",
        "sample_index": 0,
        "seed": -1
    })
    assert_status(r, 200)
    data = r.json()
    assert_key(data, "status")

    # Cleanup
    delete(f"/api/dataset_builder/{TEST_PREFIX}gen_proj")


# ── Section 15: Pipeline Invariants (Offline, no server/LLM) ───────────────
#
# The span-classifier audiobook pipeline (span_tokenizer.py, generate_script.py,
# app.extract_epub_text) makes three promises that must never regress:
#
#   1. REASSEMBLY: span reassembly is byte-identical to the source text.
#   2. SPINE YIELD (>200 chars): every EPUB spine item yields more than 200
#      characters of extracted text.
#   3. WORD COVERAGE (== 1.0): source -> annotated_script.json word coverage
#      is exactly 1.0 -- not a tolerance band.
#
# These tests build their own fixtures with tempfile/zipfile and a fake LLM
# client (see test_span_integration.FakeClient), so they need neither a live
# server nor a live LLM, and MUST pass even when every server-dependent test
# above is failing/skipping for lack of a running server.

def _require_pipeline_modules():
    if _PIPELINE_IMPORT_ERROR is not None:
        raise TestFailure(f"SKIP: offline pipeline modules unavailable ({_PIPELINE_IMPORT_ERROR})")


# --- fixtures ----------------------------------------------------------------

PIPELINE_FIXTURE_MIXED_QUOTES = (
    '"You are certain?" Elena asked.\n'
    '\n'
    '“I am,” Marcus replied, “completely certain.”\n'
)

PIPELINE_FIXTURE_ATTRIBUTION_TAGS = (
    '"We should leave," he said, pulling on his coat, "now, before they notice."'
)

PIPELINE_FIXTURE_EM_DASH_DIALOGUE = (
    "—Are you coming with us? Elena asked, not turning around.\n"
    "\n"
    "—Not yet, Marcus said, still watching the door.\n"
)

PIPELINE_FIXTURE_MULTI_PARAGRAPH = (
    "The rain had not stopped for three days.\n"
    "\n"
    '"I told you it would flood," Elena said, arms crossed.\n'
    "\n"
    "Marcus said nothing. He watched the water climb the third step, then the fourth.\n"
    "\n"
    '"We should go," she said again, quieter this time.\n'
)

# At least 4 fixtures: mixed straight/curly quotes, attribution tags,
# em-dash dialogue, multi-paragraph.
PIPELINE_FIXTURES = {
    "mixed_quotes": PIPELINE_FIXTURE_MIXED_QUOTES,
    "attribution_tags": PIPELINE_FIXTURE_ATTRIBUTION_TAGS,
    "em_dash_dialogue": PIPELINE_FIXTURE_EM_DASH_DIALOGUE,
    "multi_paragraph": PIPELINE_FIXTURE_MULTI_PARAGRAPH,
}

PIPELINE_DIGITS_ABBREV_FIXTURE = (
    'Dr. Smith owed $1,000 on Elm St. by noon, or so Mrs. Vance claimed.'
)


# --- helpers -------------------------------------------------------------

def _sc_run_chunk(client, chunk, **kwargs):
    """process_chunk with the real default prompts, stdout suppressed.

    `client` stands in for the LLM (test_span_integration.FakeClient) --
    no network I/O happens here.
    """
    buffer = io.StringIO()
    options = dict(system_prompt=DEFAULT_SYSTEM_PROMPT, user_prompt_template=DEFAULT_USER_PROMPT, max_retries=1)
    options.update(kwargs)
    with redirect_stdout(buffer):
        entries, stats = process_chunk(client, "fake-model", chunk, 1, 1, **options)
    return entries, stats


def _sc_modes_for(chunk):
    """The three LLM behaviors the span classifier must survive without losing prose."""
    full = json.dumps(labels_for(chunk))
    return {
        "full_labels": FakeResponse(full),
        "truncated_finish_length": FakeResponse(full[:max(1, len(full) // 2)], finish_reason="length"),
        "total_failure": RuntimeError("connection refused (fake)"),
    }


def _sc_coverage(source_text, entries):
    """(ratio, sequence_identical, orig_words, corr_words) using review_script's
    OWN normalize_text/check_text_loss (lowercase, strip [^\\w\\s], collapse
    whitespace, split to words) -- so this enforces the exact same definition
    the review stage uses, rather than a reimplementation that could drift.
    """
    _passed, orig_joined, corr_joined, ratio = check_text_loss(
        [{"text": source_text}], entries, threshold=1.0, upper_bound=1.0
    )
    return ratio, orig_joined == corr_joined, orig_joined, corr_joined


# --- Invariant 1: reassembly byte-identity --------------------------------

def test_span_tokenizer_reassembly_is_byte_identical():
    """INVARIANT (reassembly): tokenize()+reassemble() reproduce the source
    byte-for-byte for every fixture, independent of the LLM entirely."""
    _require_pipeline_modules()
    bad = []
    for name, text in PIPELINE_FIXTURES.items():
        spans = tokenize(text)
        try:
            validate_spans(spans, text)
        except ValueError as e:
            bad.append(f"{name}: span tiling invalid ({e})")
            continue
        if reassemble(spans, text) != text:
            bad.append(f"{name}: reassemble() != source")
    if bad:
        raise TestFailure("REASSEMBLY INVARIANT VIOLATED: " + "; ".join(bad))


def test_process_chunk_reassembly_byte_identical_across_fixtures_and_modes():
    """INVARIANT (reassembly): whatever the LLM does -- labels everything,
    truncates mid-response, or fails every attempt -- process_chunk's entries
    concatenate back to the exact source chunk. Covers >=4 distinct fixtures
    (mixed straight/curly quotes, attribution tags, em-dash dialogue,
    multi-paragraph) x 3 failure modes = 12 cases.
    """
    _require_pipeline_modules()
    bad = []
    for fixture_name, chunk in PIPELINE_FIXTURES.items():
        for mode_name, response in _sc_modes_for(chunk).items():
            client = FakeClient(response)
            entries, stats = _sc_run_chunk(client, chunk)
            rebuilt = "".join(e["text"] for e in entries)
            if rebuilt != chunk:
                bad.append(f"{fixture_name}/{mode_name}: reassembled text != source chunk")
            elif stats["labelled"] + stats["fallback"] != stats["spans"]:
                bad.append(
                    f"{fixture_name}/{mode_name}: span accounting mismatch "
                    f"(labelled={stats['labelled']} fallback={stats['fallback']} spans={stats['spans']})"
                )
    if bad:
        raise TestFailure("REASSEMBLY INVARIANT VIOLATED: " + "; ".join(bad))


# --- Invariant 2: EPUB spine yield > 200 chars ----------------------------

def test_epub_spine_items_yield_over_200_chars():
    """INVARIANT (spine yield > 200 chars): every EPUB spine item yields more
    than 200 characters of extracted text.

    Reads the per-chapter char-count report app.extract_epub_text() prints to
    stdout, rather than re-deriving counts by splitting the joined return
    value on '\\n\\n': that join is not a reliable per-chapter delimiter,
    since a chapter's own paragraphs are blank-line-separated too. Capturing
    stdout instead exercises the actual reporting mechanism a human/CI
    consumer of this function reads.
    """
    _require_pipeline_modules()
    chapter_bodies = {
        "ch1.xhtml": (
            "The lighthouse keeper had not seen another soul in eleven weeks, "
            "and the silence had started to sound like company. Every morning "
            "he counted the gulls circling the rocks below, and every evening "
            "he wrote the count in a ledger nobody would ever read."
        ),
        "ch2.xhtml": (
            '"You are certain no one is coming?" she asked, not for the first time. '
            '"Certain," he said. "The last supply boat won\'t be back until spring, '
            'and even then only if the ice breaks early enough to matter. We will '
            'simply have to wait it out, the way we always do."'
        ),
        "ch3.xhtml": (
            "Three winters had passed since the keeper first climbed the tower "
            "stairs, and each one had carved a little more silence into him. "
            "He no longer minded it. The lamp still turned, the gulls still "
            "circled, and somewhere past the horizon, ships still needed the light."
        ),
    }
    for href, body in chapter_bodies.items():
        if len(body) <= 200:
            raise TestFailure(f"test fixture bug: '{href}' body is only {len(body)} chars, need > 200")

    with tempfile.TemporaryDirectory() as tmp:
        epub_path = os.path.join(tmp, "book.epub")
        manifest_items = [
            (f"ch{i}", href, "application/xhtml+xml", None)
            for i, href in enumerate(chapter_bodies, start=1)
        ]
        spine = [item_id for item_id, _, _, _ in manifest_items]
        opf_xml = opf(manifest_items, spine)
        files = {
            f"OEBPS/{href}": f"<html><body><p>{body}</p></body></html>"
            for href, body in chapter_bodies.items()
        }
        make_epub(epub_path, opf_xml, files)

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            text = extract_epub_text(epub_path)
        report = buffer.getvalue()

    missing = [href for href, body in chapter_bodies.items() if body not in text]
    if missing:
        raise TestFailure(f"SPINE YIELD INVARIANT: chapter text missing from extracted output for: {missing}")

    counts = dict(re.findall(r'chapter \d+: (.+?) -> (\d+) chars', report))
    if len(counts) != len(chapter_bodies):
        raise TestFailure(
            f"SPINE YIELD INVARIANT (>200 chars): expected {len(chapter_bodies)} per-chapter "
            f"report lines, parsed {len(counts)} from stdout: {report!r}"
        )
    at_or_under_200 = {href: int(n) for href, n in counts.items() if int(n) <= 200}
    if at_or_under_200:
        raise TestFailure(
            f"SPINE YIELD INVARIANT (>200 chars) VIOLATED: spine item(s) at/under the "
            f"200-char floor: {at_or_under_200}"
        )


def test_epub_spine_item_under_200_chars_would_be_flagged():
    """INVARIANT (spine yield > 200 chars) counter-case: a legitimate short
    front-matter page (a bare title) yields fewer than 200 chars. This
    documents that the floor is a real, failable signal for genuine front
    matter -- not a tautology every fixture trivially satisfies -- by
    asserting the fixture's own reported count is <= 200, i.e. exactly the
    condition a real ">200 chars per spine item" gate would flag.
    """
    _require_pipeline_modules()
    short_body = "Moonrise"  # a bare, single-word title page: far under 200 chars

    with tempfile.TemporaryDirectory() as tmp:
        epub_path = os.path.join(tmp, "book.epub")
        manifest_items = [("front", "front.xhtml", "application/xhtml+xml", None)]
        opf_xml = opf(manifest_items, ["front"])
        files = {"OEBPS/front.xhtml": f"<html><body><p>{short_body}</p></body></html>"}
        make_epub(epub_path, opf_xml, files)

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            extract_epub_text(epub_path)
        report = buffer.getvalue()

    matches = re.findall(r'chapter \d+: (.+?) -> (\d+) chars', report)
    if not matches:
        raise TestFailure(
            f"SPINE YIELD INVARIANT: expected a per-chapter report line for the front-matter "
            f"fixture, got: {report!r}"
        )
    _href, count = matches[0]
    count = int(count)
    if count > 200:
        raise TestFailure(
            f"SPINE YIELD INVARIANT fixture is broken: front-matter body yielded {count} chars "
            "(> 200); this test needs a genuinely short fixture to demonstrate the floor is failable"
        )
    # `count` (<= 200) IS the FAIL signal a real ">200 chars per spine item"
    # gate would raise for this legitimate short front-matter page.


# --- Invariant 3: word coverage == 1.0 exact -------------------------------

def test_word_coverage_exact_one_across_failure_modes():
    """INVARIANT (word coverage == 1.0): source -> script word coverage is
    exactly 1.0 in every LLM failure mode (full labels, truncated, total
    failure) -- the entire point of the span-classifier design is that
    labels can be lost, but not a single source word can be. Word SEQUENCE
    identity is checked too, not just counts.
    """
    _require_pipeline_modules()
    source_text = "\n\n".join(PIPELINE_FIXTURES.values())
    bad = []
    for mode in ("full_labels", "truncated_finish_length", "total_failure"):
        chunks = split_into_chunks(source_text, max_size=300)
        if len(chunks) < 2:
            raise TestFailure(f"test fixture bug: expected multiple chunks, got {len(chunks)}")
        all_entries = []
        for chunk in chunks:
            response = _sc_modes_for(chunk)[mode]
            client = FakeClient(response)
            entries, _stats = _sc_run_chunk(client, chunk)
            all_entries.extend(entries)

        ratio, seq_match, _orig_words, _corr_words = _sc_coverage(source_text, all_entries)
        if ratio != 1.0:
            bad.append(f"{mode}: coverage ratio={ratio!r} (expected exactly 1.0)")
        if not seq_match:
            bad.append(f"{mode}: word sequence differs from source (counts may match but order/content does not)")
    if bad:
        raise TestFailure("WORD COVERAGE INVARIANT (== 1.0) VIOLATED: " + "; ".join(bad))


def test_no_character_entry_mixes_quotation_and_narration():
    """INVARIANT (one entry, one kind of text): the renderer gives an entry a
    single voice, so a character-voiced entry holding both a quotation and the
    narration around it means the narration is spoken by the character.

    Measured on one 8,083-entry artifact before the fix: 13,337 chars of
    narration read in a character voice, 73 entries mixing the two kinds. The
    model is simulated at its worst here -- EVERY span, quoted or not, labelled
    with the same character -- which is exactly how attribution tags ("said
    Marcus") got swallowed into the preceding line.
    """
    _require_pipeline_modules()
    from span_tokenizer import tokenize as _tokenize

    bad = []
    for name, chunk in PIPELINE_FIXTURES.items():
        labels = [
            {"id": span.id, "speaker": "ELENA", "role": "dialogue", "instruct": "Flat."}
            for span in _tokenize(chunk)
        ]
        entries, _stats = _sc_run_chunk(
            FakeClient(FakeResponse(json.dumps(labels))), chunk)

        if "".join(entry["text"] for entry in entries) != chunk:
            bad.append(f"{name}: text is no longer byte-identical")
        for entry in entries:
            if entry["speaker"] == "NARRATOR":
                continue
            kinds = {span.kind for span in _tokenize(entry["text"])
                     if entry["text"][span.start:span.end].strip()}
            if len(kinds) > 1:
                bad.append(f"{name}: character entry mixes {sorted(kinds)}: {entry['text']!r}")
    if bad:
        raise TestFailure("ONE-ENTRY-ONE-KIND INVARIANT VIOLATED: " + "; ".join(bad))


def _legacy_split_into_chunks(text, max_size=3000):
    """The pre-fix chunker, verbatim, kept ONLY so the test below can prove the
    byte-fidelity assertion is failable (it drops a paragraph break at every
    chunk seam). Not used by the pipeline."""
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current_chunk) + len(para) + 2 > max_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            if len(para) > max_size:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 1 > max_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence
                    else:
                        current_chunk += " " + sentence if current_chunk else sentence
            else:
                current_chunk = para
        else:
            current_chunk += "\n\n" + para if current_chunk else para
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


# Synthetic, corpus-independent document exercising every whitespace edge the
# chunker can meet: leading/trailing blank lines, a whitespace-only "paragraph"
# (NBSP), CRLF, an indented paragraph, and an oversized paragraph that forces
# the sentence-split path.
CHUNK_FIDELITY_DOC = (
    "\n\n  \n"
    "Chapter One\n\n"
    "He turned away.\n\n"
    '"That\'s it," she said.\n\n'
    " \n\n"
    "   An indented paragraph follows a blank-ish one.\r\n\r\n"
    + " ".join(f"Sentence number {i} runs on for a while here." for i in range(60))
    + "\n\n"
    "The end.\n \n\n"
)


def test_chunking_is_byte_lossless_over_source():
    """INVARIANT (byte-verbatim, whole-document half of contract 4): the
    concatenation of split_into_chunks() output must reproduce the source
    file BYTE-FOR-BYTE. The existing word-coverage test cannot see a
    violation here, because review_script.normalize_text() maps punctuation
    to spaces before tokenizing and every chunk seam sits at punctuation --
    so a lost paragraph break still scores 1.0. This asserts the raw bytes.

    The paired counter-case runs the pre-fix chunker on the same document and
    requires it to LOSE characters, proving the assertion is failable rather
    than tautological.
    """
    _require_pipeline_modules()
    doc = CHUNK_FIDELITY_DOC
    bad = []

    for max_size in (120, 300, 3000):
        chunks = split_into_chunks(doc, max_size=max_size)
        rejoined = "".join(chunks)
        if rejoined != doc:
            bad.append(
                f"max_size={max_size}: rejoined chunks differ from source "
                f"({len(rejoined)} chars vs {len(doc)}); "
                f"first divergence at offset "
                f"{next((i for i, (a, b) in enumerate(zip(rejoined, doc)) if a != b), min(len(doc), len(rejoined)))}"
            )
        if any(c == "" for c in chunks):
            bad.append(f"max_size={max_size}: produced an empty chunk")

    # Degenerate inputs must round-trip too.
    for edge in ("", "   ", "\n\n\n", "one paragraph"):
        if "".join(split_into_chunks(edge, max_size=50)) != edge:
            bad.append(f"edge input {edge!r} did not round-trip")

    if bad:
        raise TestFailure("CHUNK BYTE-FIDELITY INVARIANT VIOLATED: " + "; ".join(bad))

    # Counter-case: the old chunker must fail this same assertion.
    legacy = "".join(_legacy_split_into_chunks(doc, max_size=300))
    if legacy == doc:
        raise TestFailure(
            "CHUNK BYTE-FIDELITY test is not failable: the pre-fix chunker "
            "round-tripped this fixture, so the fixture no longer exercises the bug"
        )


def test_word_coverage_digits_and_abbreviations_survive_verbatim():
    """INVARIANT (word coverage == 1.0) + verbatim: digits, currency, and
    abbreviations ("Dr. Smith owed $1,000 on Elm St.") survive into the
    script UNTOUCHED in every failure mode, and still contribute exactly to
    word coverage.
    """
    _require_pipeline_modules()
    source_text = PIPELINE_DIGITS_ABBREV_FIXTURE
    bad = []
    for mode in ("full_labels", "truncated_finish_length", "total_failure"):
        client = FakeClient(_sc_modes_for(source_text)[mode])
        entries, _stats = _sc_run_chunk(client, source_text)

        rebuilt = "".join(e["text"] for e in entries)
        if rebuilt != source_text:
            bad.append(f"{mode}: reassembly not verbatim: {rebuilt!r} != {source_text!r}")
            continue
        for literal in ("Dr. Smith", "$1,000", "Elm St.", "Mrs. Vance"):
            if literal not in rebuilt:
                bad.append(f"{mode}: '{literal}' altered or lost")

        ratio, seq_match, _orig_words, _corr_words = _sc_coverage(source_text, entries)
        if ratio != 1.0 or not seq_match:
            bad.append(f"{mode}: coverage ratio={ratio!r} sequence_match={seq_match} (expected ratio==1.0 and match)")
    if bad:
        raise TestFailure(
            "WORD COVERAGE INVARIANT (== 1.0) / REASSEMBLY INVARIANT VIOLATED on "
            "digits/abbreviations fixture: " + "; ".join(bad)
        )


def test_real_annotated_script_word_coverage_if_present():
    """Opportunistic: if a real annotated_script.json + its resolvable source
    input exist on disk (the live-artifact path from a prior
    /api/generate_script run), check INVARIANT (word coverage == 1.0)
    against them. Skips cleanly when either artifact is absent -- this is
    the common case for a fresh checkout.
    """
    _require_pipeline_modules()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_path = os.path.join(repo_root, "annotated_script.json")
    state_path = os.path.join(repo_root, "state.json")

    if not os.path.exists(script_path):
        raise TestFailure("SKIP: no annotated_script.json at repo root")
    if not os.path.exists(state_path):
        raise TestFailure("SKIP: no state.json to locate the source input file")

    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    input_file = state.get("input_file_path")
    if not input_file or not os.path.exists(input_file):
        raise TestFailure("SKIP: state.json has no resolvable input_file_path")

    with open(script_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    with open(input_file, "r", encoding="utf-8") as f:
        source_text = f.read()

    ratio, seq_match, _orig_words, _corr_words = _sc_coverage(source_text, entries)
    if ratio != 1.0 or not seq_match:
        raise TestFailure(
            f"WORD COVERAGE INVARIANT (== 1.0) VIOLATED on real artifacts: "
            f"ratio={ratio!r} sequence_match={seq_match}"
        )


def test_label_flags_endpoint_logic_never_writes_files():
    """The advisory GET /api/voices/label_flags endpoint (app.py's
    _compute_label_flags, called directly here rather than over HTTP so this
    stays offline) must never write or mutate voice_config.json or
    annotated_script.json on disk -- it is READ-ONLY / advisory tooling per
    CLAUDE.md's frozen contracts.
    """
    _require_pipeline_modules()

    script_path = app_module.SCRIPT_PATH
    voice_config_path = app_module.VOICE_CONFIG_PATH

    def _snapshot(path):
        if not os.path.exists(path):
            return None
        return (os.path.getmtime(path), os.path.getsize(path))

    before_script = _snapshot(script_path)
    before_voice_config = _snapshot(voice_config_path)

    source_text = (
        'Elena walked into the hall. "We should go," Elena said quietly. '
        "A ghostly figure watched from the shadows without speaking."
    )
    script_data = [
        {"speaker": "Elena", "text": "We should go,", "instruct": "quiet"},
        # PHANTASMAGORIA never appears in source_text -> exercises the
        # "flagged unattested" path too, not just the happy path.
        {"speaker": "Phantasmagoria", "text": "A ghostly figure watched from the shadows", "instruct": "eerie"},
    ]

    flags = app_module._compute_label_flags(script_data, source_text)

    after_script = _snapshot(script_path)
    after_voice_config = _snapshot(voice_config_path)

    if before_script != after_script:
        raise TestFailure(
            f"_compute_label_flags touched annotated_script.json on disk: "
            f"before={before_script!r} after={after_script!r}"
        )
    if before_voice_config != after_voice_config:
        raise TestFailure(
            f"_compute_label_flags touched voice_config.json on disk: "
            f"before={before_voice_config!r} after={after_voice_config!r}"
        )

    names = {f["name"] for f in flags}
    if names != {"ELENA", "PHANTASMAGORIA"}:
        raise TestFailure(f"expected flags for ELENA and PHANTASMAGORIA, got: {names!r}")

    by_name = {f["name"]: f for f in flags}
    if by_name["ELENA"]["attested"] is not True:
        raise TestFailure(f"ELENA should be attested (appears near its own line): {by_name['ELENA']!r}")
    if by_name["PHANTASMAGORIA"]["attested"] is not False:
        raise TestFailure(
            f"PHANTASMAGORIA should be flagged unattested (name never appears in source): "
            f"{by_name['PHANTASMAGORIA']!r}"
        )


# ── Run all tests ────────────────────────────────────────────

def run_all_tests():
    section("Server")
    run_test("server_reachable", test_server_reachable)

    section("Config")
    run_test("get_config", test_get_config)
    run_test("save_config_roundtrip", test_save_config_roundtrip)
    run_test("save_pause_config_roundtrip", test_save_pause_config_roundtrip)
    run_test("pause_config_defaults", test_pause_config_defaults)
    run_test("save_review_prompts_roundtrip", test_save_review_prompts_roundtrip)
    run_test("save_persona_prompts_roundtrip", test_save_persona_prompts_roundtrip)
    run_test("get_default_prompts", test_get_default_prompts)

    section("System Stats")
    run_test("system_stats", test_system_stats)

    section("Upload")
    run_test("upload_file", test_upload_file)

    section("Annotated Script")
    run_test("get_annotated_script", test_get_annotated_script)

    section("Scripts CRUD")
    run_test("save_script", test_save_script)
    run_test("list_scripts", test_list_scripts)
    run_test("load_script", test_load_script)
    run_test("delete_script", test_delete_script)
    run_test("delete_script_404", test_delete_script_404)

    section("Voices")
    run_test("get_voices", test_get_voices)
    run_test("save_voice_config", test_save_voice_config)

    section("Chunks")
    run_test("get_chunks", test_get_chunks)
    run_test("update_chunk", test_update_chunk)
    run_test("update_chunk_pause_after", test_update_chunk_pause_after)
    run_test("update_chunk_pause_after_zero", test_update_chunk_pause_after_zero)
    run_test("update_chunk_pause_after_negative", test_update_chunk_pause_after_negative)
    run_test("update_chunk_404", test_update_chunk_404)
    run_test("insert_chunk", test_insert_chunk)
    run_test("insert_chunk_404", test_insert_chunk_404)
    run_test("delete_chunk", test_delete_chunk)
    run_test("delete_chunk_invalid", test_delete_chunk_invalid)
    run_test("restore_chunk", test_restore_chunk)

    section("Status Polling")
    run_test("status_known_tasks", test_status_known_tasks)
    run_test("status_unknown_task", test_status_unknown_task)

    section("Preparer")
    run_test("features_endpoint", test_features_endpoint)
    run_test("preparer_status", test_preparer_status)
    run_test("batch_preparer_status", test_batch_preparer_status)
    run_test("preparer_cancel_when_idle", test_preparer_cancel_when_idle)
    run_test("preparer_list_outputs", test_preparer_list_outputs)
    run_test("preparer_download_404", test_preparer_download_404)
    run_test("batch_preparer_start_schema", test_batch_preparer_start_schema)
    run_test("batch_preparer_cancel", test_batch_preparer_cancel)

    section("Voice Design")
    run_test("voice_design_list", test_voice_design_list)
    run_test("voice_design_delete_404", test_voice_design_delete_404)
    run_test("voice_design_preview", test_voice_design_preview, requires_full=True)
    run_test("voice_design_save_and_delete", test_voice_design_save_and_delete, requires_full=True)

    section("Clone Voices")
    run_test("clone_voices_list", test_clone_voices_list)
    run_test("clone_voices_upload_bad_format", test_clone_voices_upload_bad_format)
    run_test("clone_voices_delete_404", test_clone_voices_delete_404)
    run_test("clone_voices_upload_and_delete", test_clone_voices_upload_and_delete)

    section("LoRA Datasets")
    run_test("lora_list_datasets", test_lora_list_datasets)
    run_test("lora_delete_dataset_404", test_lora_delete_dataset_404)
    run_test("lora_upload_bad_file", test_lora_upload_bad_file)

    section("LoRA Models")
    run_test("lora_list_models", test_lora_list_models)
    run_test("lora_download_invalid", test_lora_download_invalid)
    run_test("lora_delete_model_404", test_lora_delete_model_404)
    run_test("lora_train_bad_dataset", test_lora_train_bad_dataset)
    run_test("lora_preview_404", test_lora_preview_404)
    run_test("lora_preview", test_lora_preview, requires_full=True)

    section("Dataset Builder")
    run_test("dataset_builder_list", test_dataset_builder_list)
    run_test("dataset_builder_create", test_dataset_builder_create)
    run_test("dataset_builder_update_meta", test_dataset_builder_update_meta)
    run_test("dataset_builder_update_rows", test_dataset_builder_update_rows)
    run_test("dataset_builder_status", test_dataset_builder_status)
    run_test("dataset_builder_cancel", test_dataset_builder_cancel)
    run_test("dataset_builder_save_no_samples", test_dataset_builder_save_no_samples)
    run_test("dataset_builder_delete", test_dataset_builder_delete)
    run_test("dataset_builder_delete_404", test_dataset_builder_delete_404)

    section("Persona Generation")
    run_test("cancel_persona_not_running", test_cancel_persona_not_running)

    section("Merge / Export")
    run_test("get_audiobook", test_get_audiobook)
    run_test("get_audiobook_m4b", test_get_audiobook_m4b)
    run_test("get_audacity_export", test_get_audacity_export)

    section("Generation (TTS/LLM)")
    run_test("generate_script", test_generate_script, requires_full=True)
    run_test("generate_script_single_speaker", test_generate_script_single_speaker, requires_full=True)
    run_test("review_script", test_review_script, requires_full=True)
    run_test("generate_chunk", test_generate_chunk, requires_full=True)
    run_test("generate_batch", test_generate_batch, requires_full=True)
    run_test("generate_batch_fast", test_generate_batch_fast, requires_full=True)
    run_test("cancel_audio", test_cancel_audio)
    run_test("export_audacity", test_export_audacity, requires_full=True)

    section("LoRA (TTS)")
    run_test("lora_test_model", test_lora_test_model, requires_full=True)
    run_test("lora_generate_dataset", test_lora_generate_dataset, requires_full=True)

    section("Dataset Builder Generate (TTS)")
    run_test("dataset_builder_generate_sample", test_dataset_builder_generate_sample, requires_full=True)

    run_offline_invariant_tests()


def run_offline_invariant_tests():
    """The genuinely server-free subset of the suite: Section 15's pipeline
    invariants. No network I/O happens in here -- safe (and fast) to run with
    `--offline` / `--offline-only` when no server is up, and also called from
    run_all_tests() above so default behavior is unchanged.
    """
    section("Pipeline Invariants (Offline, no server/LLM)")
    run_test("span_tokenizer_reassembly_byte_identical", test_span_tokenizer_reassembly_is_byte_identical)
    run_test("process_chunk_reassembly_byte_identical_across_fixtures_and_modes",
              test_process_chunk_reassembly_byte_identical_across_fixtures_and_modes)
    run_test("epub_spine_items_yield_over_200_chars", test_epub_spine_items_yield_over_200_chars)
    run_test("epub_spine_item_under_200_chars_would_be_flagged",
              test_epub_spine_item_under_200_chars_would_be_flagged)
    run_test("word_coverage_exact_one_across_failure_modes", test_word_coverage_exact_one_across_failure_modes)
    run_test("no_character_entry_mixes_quotation_and_narration", test_no_character_entry_mixes_quotation_and_narration)
    run_test("chunking_is_byte_lossless_over_source", test_chunking_is_byte_lossless_over_source)
    run_test("word_coverage_digits_and_abbreviations_survive_verbatim",
              test_word_coverage_digits_and_abbreviations_survive_verbatim)
    run_test("real_annotated_script_word_coverage_if_present", test_real_annotated_script_word_coverage_if_present)
    run_test("label_flags_endpoint_logic_never_writes_files", test_label_flags_endpoint_logic_never_writes_files)


# ── Cleanup ──────────────────────────────────────────────────

def cleanup():
    print(f"\n--- Cleanup ---")
    items = []

    try:
        delete(f"/api/scripts/{TEST_PREFIX}script")
        items.append("test script")
    except Exception:
        pass

    try:
        delete(f"/api/dataset_builder/{TEST_PREFIX}builder_proj")
        items.append("builder project")
    except Exception:
        pass

    try:
        delete(f"/api/dataset_builder/{TEST_PREFIX}gen_proj")
        items.append("gen project")
    except Exception:
        pass

    try:
        delete(f"/api/lora/datasets/{TEST_PREFIX}dataset")
        items.append("test dataset")
    except Exception:
        pass

    try:
        r = get("/api/voice_design/list")
        if r.status_code == 200:
            for v in r.json():
                if v.get("id", "").startswith(TEST_PREFIX):
                    delete(f"/api/voice_design/{v['id']}")
                    items.append(f"voice {v['id']}")
    except Exception:
        pass

    if items:
        print(f"  Cleaned: {', '.join(items)}")
    else:
        print(f"  Nothing to clean")


# ── Main ─────────────────────────────────────────────────────

def main():
    global BASE_URL, FULL_MODE, OFFLINE_ONLY

    parser = argparse.ArgumentParser(description="Alexandria API test suite")
    parser.add_argument("--url", default="http://127.0.0.1:4200",
                        help="Server URL (default: http://127.0.0.1:4200)")
    parser.add_argument("--full", action="store_true",
                        help="Include TTS/LLM-dependent tests")
    parser.add_argument("--offline", "--offline-only", dest="offline", action="store_true",
                        help="Run ONLY the offline pipeline-invariant tests (Section 15): "
                             "no server, no LLM, no network calls at all. Ignores --full and "
                             "--url. Use this to check the span-classifier invariants in "
                             "seconds instead of waiting through connection-refused timeouts "
                             "on every server-dependent test.")
    args = parser.parse_args()

    BASE_URL = args.url.rstrip("/")
    FULL_MODE = args.full
    OFFLINE_ONLY = args.offline

    print(f"Alexandria API Tests")
    if OFFLINE_ONLY:
        print(f"Mode:   OFFLINE ONLY (pipeline invariants; no server, no LLM, no network)")
    else:
        print(f"Server: {BASE_URL}")
        print(f"Mode:   {'FULL (includes TTS/LLM tests)' if FULL_MODE else 'QUICK (no TTS/LLM)'}")

    if OFFLINE_ONLY:
        # Genuinely server-free: skip run_all_tests() (which would otherwise
        # dial out to BASE_URL for ~90 tests first) and skip cleanup() too,
        # since it only deletes server-side test fixtures via HTTP.
        run_offline_invariant_tests()
    else:
        try:
            run_all_tests()
        finally:
            cleanup()

    # Summary
    total = results["passed"] + results["failed"] + results["skipped"]
    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {results['passed']} passed, {results['failed']} failed, "
          f"{results['skipped']} skipped  (total: {total})")
    print(f"{'=' * 60}")

    if failures:
        print(f"\nFailed tests:")
        for name, err in failures:
            # Truncate long error messages
            short = err.split("\n")[0][:200]
            print(f"  - {name}: {short}")

    sys.exit(1 if results["failed"] > 0 else 0)


if __name__ == "__main__":
    main()
