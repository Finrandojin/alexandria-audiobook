#!/usr/bin/env python3
"""Automated API test script for Alexandria audiobook generator.

Usage:
    python test_api.py                    # Quick tests only
    python test_api.py --full             # Include TTS/LLM-dependent tests
    python test_api.py --url http://host:port
"""

import argparse
import io
import json
import sys
import time
import requests

# ── Global state ─────────────────────────────────────────────

BASE_URL = ""
FULL_MODE = False
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
    global BASE_URL, FULL_MODE

    parser = argparse.ArgumentParser(description="Alexandria API test suite")
    parser.add_argument("--url", default="http://127.0.0.1:4200",
                        help="Server URL (default: http://127.0.0.1:4200)")
    parser.add_argument("--full", action="store_true",
                        help="Include TTS/LLM-dependent tests")
    args = parser.parse_args()

    BASE_URL = args.url.rstrip("/")
    FULL_MODE = args.full

    print(f"Alexandria API Tests")
    print(f"Server: {BASE_URL}")
    print(f"Mode:   {'FULL (includes TTS/LLM tests)' if FULL_MODE else 'QUICK (no TTS/LLM)'}")

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
