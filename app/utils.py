import errno
import os
import json
import shutil
import time
import tempfile
import contextlib
import re
import sys
import logging
import hashlib
import hmac
import base64
import uuid

logger = logging.getLogger(__name__)


def is_nonverbal_text(text):
    """Return whether text contains no speakable letter or number."""
    return not any(char.isalnum() for char in str(text or ""))


def get_unique_id(prefix: str) -> str:
    """Return a readable, collision-resistant identifier for a saved artifact."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def get_runtime_data_dir(root_dir: str) -> str:
    """Return the single mutable-data root for this application instance."""
    configured = os.environ.get("ALEXANDRIA_DATA_DIR", "").strip()
    return os.path.abspath(configured or root_dir)


def get_app_config_path(data_dir: str, root_dir: str, app_dir: str) -> str:
    """Keep legacy local config placement while isolating configured runtimes."""
    if os.path.abspath(data_dir) == os.path.abspath(root_dir):
        return os.path.join(app_dir, "config.json")
    return os.path.join(data_dir, "config.json")

# --- GPU stats (rocm-smi) ---
# Canonical implementation lives in gpu_stats.py at the repo root, shared
# with the standalone alexandria_*.py scripts which can't import from
# inside this package. Re-exported here so existing `from utils import
# run_rocm_smi_json` call sites in app/ keep working unchanged.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpu_stats import run_rocm_smi_json, system_has_gpu, is_oom_failure, rocm_smi_utilization  # noqa: F401


# --- Path containment ---

def is_path_inside(path: str, base_dir: str) -> bool:
    """True if the realpath of `path` is base_dir itself or somewhere under it.

    Canonical realpath-containment check, shared by every caller that needs
    to confirm a path doesn't escape (or doesn't merely resolve inside) a
    given directory - app.py's traversal/denylist guards and train_lora.py's
    dataset-path guards previously each carried their own copy of this same
    comparison.
    """
    base = os.path.realpath(base_dir)
    target = os.path.realpath(path)
    return target == base or target.startswith(base + os.sep)


# --- Balanced-bracket text extraction ---

def extract_balanced(text, open_char, close_char, search_from=0):
    """Find the first `open_char ... close_char`-balanced span in `text`
    starting the search at or after `search_from`, tracking string-escaping
    so a quoted brace/bracket doesn't desync the depth count. Returns the
    matched substring, or None if `open_char` never appears (at or after
    `search_from`) or never balances back to depth 0.

    Shared by clean_json_string ([...]) and extract_json_object ({...}) -
    both need the same escape-aware bracket-matching, just for a different
    delimiter pair.

    `search_from` lets a caller retry past a span that turned out not to be
    the real value (see extract_json_object) - it does NOT mean "assume
    we're inside a string at this position"; in_string tracking still always
    starts fresh at `open_char`'s position, same as before, since a caller
    only ever retries from a point known to be outside any string (right
    after a previous open_char that was itself found outside a string).

    A backslash only escapes the next character while inside a string
    (real JSON has no escape meaning outside one) - this is stricter than
    clean_json_string's original bracket-loop, which treated any backslash
    as an escape everywhere. That's intentional: it matches actual JSON
    semantics, so a stray unescaped backslash in malformed LLM output
    outside a string no longer causes a real closing bracket to be missed.
    """
    start = text.find(open_char, search_from)
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        ch = text[i]

        if escape_next:
            escape_next = False
            continue

        if ch == '\\':
            if in_string:
                escape_next = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def extract_json_object(text):
    """Extract the first JSON object from text using robust parsing.

    Tries standard json.loads first, then falls back to escape-aware
    brace-matching (extract_balanced) for free-form LLM output that wraps
    the object in other text.

    The first '{' in the text isn't necessarily the real object's start -
    free-form LLM prose can contain an earlier, incidental balanced
    brace-pair (e.g. "I'll use {category} mapping: {...}") that bracket-
    matches cleanly but isn't JSON. If a candidate span fails to parse,
    retry from just past that span's opening brace instead of giving up,
    so a real object later in the text still gets found.
    """
    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    search_from = 0
    while True:
        start = text.find('{', search_from)
        if start == -1:
            return None
        span = extract_balanced(text, '{', '}', search_from=start)
        if span is None:
            return None
        try:
            return json.loads(span)
        except json.JSONDecodeError:
            search_from = start + 1


def warn_unparseable_llm_json(what: str, raw: str, fallback_action: str) -> None:
    """Print a consistent warning when extract_json_object found no JSON
    object in an LLM response, for CLI scripts (find_nicknames.py,
    review_script.py) whose convention is print() rather than logging."""
    print(f"  Warning: could not parse a JSON object from the LLM's {what} "
          f"response ({len(raw)} chars); {fallback_action}.")


# --- Filename Sanitization ---

def secure_filename(filename: str) -> str:
    """Sanitize a filename to prevent path-traversal attacks.

    Removes path separators and null bytes, keeps only safe characters,
    and caps length well under the ~255-byte filesystem component limit
    (leaving room for a caller-appended suffix, e.g. "sample_001.wav").
    Two different inputs that only differ after the cap get a short hash
    of the full original appended, so truncation can't make them collide
    on the same output.
    """
    if not filename:
        return ""
    for sep in ("/", "\\", "\0"):
        filename = filename.replace(sep, "_")
    filename = filename.lstrip(". ")
    filename = re.sub(r"[^\w\-. ]", "_", filename)
    if len(filename) > 150:
        suffix = hashlib.sha1(filename.encode("utf-8")).hexdigest()[:8]
        filename = filename[:150 - len(suffix) - 1] + "_" + suffix
    if not filename:
        return ""
    return filename


# --- Atomic JSON write (write-to-temp + rename) ---

def safe_load_json(path, default=None):
    """Load JSON from `path`, returning `default` if missing, empty, or corrupted.

    If `default` is a dict/list, a successfully-parsed value of a different
    type (e.g. a config.json truncated to "null" or "[]") is also treated as
    corrupted. Most callers pass a dict/list default and call .get()/iterate
    on the result immediately with no type check of their own - without this,
    a type mismatch surfaces as an uncaught AttributeError/TypeError deep in
    the caller instead of the same graceful fallback every other corruption
    case already gets. Callers that want the raw value regardless of type
    (e.g. to distinguish a missing file from a non-dict one themselves) should
    pass default=None, which skips this check entirely.
    """
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Corrupted/unreadable JSON at {path}, using default: {e}")
        return default
    if default is not None and not isinstance(data, type(default)):
        logger.warning(
            f"Unexpected JSON shape at {path} (expected {type(default).__name__}, "
            f"got {type(data).__name__}), using default"
        )
        return default
    return data


def atomic_json_write(data, target_path, max_retries=5):
    """Atomically write JSON data using a temp file and os.replace.

    Includes retry logic with exponential backoff for Windows file locking
    (Access is denied / file in use errors).  On cross-device paths (e.g.
    Linux bind mounts or NAS shares) os.replace raises EXDEV; in that case
    the function falls back to shutil.move (copy + delete) so the write
    succeeds rather than raising an unhandled OSError.
    """
    directory = os.path.dirname(target_path) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        for attempt in range(max_retries):
            try:
                os.replace(tmp_path, target_path)
                # Also fsync the directory so the rename itself is durable across
                # power loss (POSIX) — the file fsync above doesn't cover the
                # directory entry. Best-effort; unsupported/needless on Windows.
                try:
                    dir_fd = os.open(os.path.dirname(target_path) or ".", os.O_RDONLY)
                    try:
                        os.fsync(dir_fd)
                    finally:
                        os.close(dir_fd)
                except (OSError, AttributeError):
                    pass
                return
            except OSError as e:
                if e.errno == errno.EXDEV:
                    # Cross-device rename is not supported by the kernel, so
                    # os.replace can't be used across filesystems - fall
                    # back to copy+delete, which is not atomic but avoids a
                    # hard failure. Refuse if target_path is already a
                    # symlink: unlike os.replace (which atomically retargets
                    # the symlink itself), shutil.move's copy step follows a
                    # symlink and writes through to whatever it points at -
                    # not a complete guarantee against a same-instant TOCTOU
                    # swap, but a real mitigation against a symlink already
                    # sitting there.
                    if os.path.islink(target_path):
                        raise OSError(
                            f"Refusing cross-device fallback write through a symlink at {target_path}"
                        ) from e
                    try:
                        shutil.move(tmp_path, target_path)
                        return
                    except OSError as move_err:
                        # Let the same retry/backoff check below apply to a
                        # transient failure during the fallback copy too
                        # (e.g. destination momentarily locked) - tmp_path
                        # is untouched unless copy_function fully succeeded,
                        # so retrying shutil.move again is safe.
                        e = move_err
                # ERROR_ACCESS_DENIED (5) / ERROR_SHARING_VIOLATION (32) are raw
                # Windows error codes, which only ever show up on e.winerror -
                # e.errno holds the CRT's translated POSIX-equivalent (EACCES=13
                # for both), so checking e.errno here would never match. The
                # string checks below already catch this in practice, but
                # checking winerror directly doesn't depend on the exception's
                # message text staying in this exact wording.
                if attempt < max_retries - 1 and (
                    getattr(e, "winerror", None) in (5, 32)
                    or "Access is denied" in str(e)
                    or "being used by another process" in str(e)
                    or "The process cannot access the file" in str(e)
                ):
                    delay = 0.05 * (2 ** attempt)
                    time.sleep(delay)
                    continue
                raise
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def atomic_json_write_pair(first_data, first_path, second_data, second_path):
    """Replace two locked JSON files, rolling both back if replacement fails."""
    staged, backups = [], []
    try:
        for data, path in ((first_data, first_path), (second_data, second_path)):
            directory = os.path.dirname(os.path.abspath(path))
            os.makedirs(directory, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=".pair-", suffix=".json", dir=directory)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            staged.append(tmp)
            backup = path + ".pair-backup"
            existed = os.path.exists(path)
            if existed:
                shutil.copy2(path, backup)
            backups.append((path, backup, existed))
        for index, (_data, path) in enumerate(((first_data, first_path), (second_data, second_path))):
            os.replace(staged[index], path)
            staged[index] = None
    except Exception:
        for path, backup, existed in backups:
            if existed and os.path.exists(backup):
                os.replace(backup, path)
            elif not existed and os.path.exists(path):
                os.remove(path)
        raise
    finally:
        for tmp in staged:
            if tmp and os.path.exists(tmp):
                os.remove(tmp)
        for _path, backup, _existed in backups:
            if os.path.exists(backup):
                os.remove(backup)


def backup_file_with_timestamp(path):
    """Copy ``path`` to a collision-resistant timestamped sibling backup."""
    stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 1_000_000_000:09d}"
    backup = f"{path}.bak-{stamp}"
    shutil.copy2(path, backup)
    return backup


@contextlib.contextmanager
def file_lock(target_path, timeout=10, stale_after=120):
    """Advisory cross-process lock for read-modify-write access to target_path.

    Coordinates against a sibling `<target_path>.lock` marker file, created via
    an atomic exclusive open (works on POSIX and Windows). Without this, two
    processes that each read-modify-write the same JSON file (e.g. a batch
    review remapping speaker names in a voice_config.json while the UI applies
    a saved cast to it) can silently lose one side's update.

    This is advisory only: if the lock can't be acquired within `timeout`
    seconds, this raises `TimeoutError` so the caller can decide how to
    handle contention (e.g. skip the operation, return a "busy" error, or
    retry) rather than silently proceeding without the lock. A lock file
    older than `stale_after` seconds is treated as abandoned (e.g. left
    behind by a crashed process) and removed.
    
    Args:
        target_path: Path to the file being protected
        timeout: Maximum seconds to wait for lock acquisition (default: 10)
        stale_after: Seconds after which a lock file is considered abandoned
            (default: 120 — kept well above any plausible critical-section
            duration so a still-live holder isn't reaped and double-held)
    """
    lock_path = target_path + ".lock"
    deadline = time.time() + timeout
    acquired = False
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            acquired = True
            break
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(lock_path) > stale_after:
                    os.remove(lock_path)
                    continue
            except OSError:
                pass
            if time.time() >= deadline:
                raise TimeoutError(f"Could not acquire file lock on {lock_path} within {timeout} seconds.")
            time.sleep(0.05)
    try:
        yield
    finally:
        # Only remove the lock file if we created it - a timed-out waiter that
        # never acquired the lock must not delete another process's active lock.
        if acquired:
            try:
                os.remove(lock_path)
            except OSError:
                pass


# --- Generic speaker de-collision ---

# Common generic character labels that collide across books (a voice assigned to
# "man" in one book would otherwise bleed into "man" in another).
_GENERIC_SPEAKER_WORDS = {
    "man", "woman", "old man", "young woman", "boy", "girl", "child", "person",
    "someone", "stranger", "soldier", "guard", "voice", "villager", "crowd",
    "villagers", "guards", "soldiers", "police", "officer", "doctor", "nurse",
    "waiter", "driver", "maid", "servant", "priest",
}


def is_generic_speaker(name):
    """Return whether name is a book-local generic character label."""
    value = (name or "").strip().lower()
    if value.startswith("the "):
        value = value[4:].strip()
    # Annotation models commonly disambiguate incidental roles as "Man 1" or
    # "Guard #2".  Only accept a numeric suffix on a known generic base so
    # proper names containing digits are not accidentally scoped to one book.
    value = re.sub(r"\s+#?\d+$", "", value).strip()
    return value in _GENERIC_SPEAKER_WORDS


def check_basic_auth(header_value, username, password):
    """Constant-time verify an 'Authorization: Basic <base64>' header value
    against the expected username/password. Returns False on any missing or
    malformed input (wrong scheme, bad base64, no colon)."""
    if not header_value or not header_value.startswith("Basic "):
        return False
    try:
        decoded = base64.b64decode(header_value[6:].strip()).decode("utf-8")
    except Exception:
        return False
    supplied_user, sep, supplied_pw = decoded.partition(":")
    if not sep:
        return False
    # Evaluate both comparisons before combining so the response time does not
    # reveal which of username/password mismatched.
    user_ok = hmac.compare_digest(supplied_user, username)
    pw_ok = hmac.compare_digest(supplied_pw, password)
    return user_ok and pw_ok


# ── Voice seeding ────────────────────────────────────────────────────────

def character_voice_seed(character):
    """A stable, per-character generation seed derived from the name.

    WHY THIS EXISTS. `generate_lora_voice` ignored the seed field entirely
    until 2026-08-04, so every line of every LoRA voice was an independent
    draw. The user identified it by ear as "multiple narrators" before any
    metric did. Fixing the plumbing made the field WORK; it did not make
    anything SET one, and 70 of 71 characters in the shipped voice_config
    still carried "-1" - meaning a character's voice was still redrawn per
    line, just now on purpose rather than by accident.

    Measured: seeded generation is byte-identical across fresh processes;
    unseeded, one adapter's pitch moves across a 32.4 Hz median band, which is
    larger than the separation between many pairs of distinct voices in the
    pool.

    DERIVED FROM THE NAME rather than allocated, for three reasons: the same
    character sounds the same across regenerations and across books; nothing
    has to be stored or migrated; and two characters cannot collide onto one
    draw the way a single global seed would make everybody collide.

    Returns a positive int; callers store it as a string, matching the
    existing config shape. Seeds are compared as `int(...) >= 0`, so this must
    stay non-negative - -1 is the sentinel meaning "draw randomly".
    """
    import hashlib
    name = str(character or "").strip().casefold()
    if not name:
        return 0
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    # 31 bits: comfortably inside torch.manual_seed's accepted range and
    # never negative, so it cannot be mistaken for the -1 sentinel.
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
