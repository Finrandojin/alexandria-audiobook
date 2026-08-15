import asyncio
import hashlib
import json
import logging
import os
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
import xml.etree.ElementTree as ET
from math import ceil
from typing import Dict, List, Literal, Optional

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from config_settings import load_app_config
from default_prompts import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT
from generate_script import (build_book_request_preflight, fix_mojibake,
                             split_into_chunks)
from lmstudio_settings import (ensure_ideal_settings, get_planned_ideal_settings)
from script_preflight import audit_unicode_text
from source_normalization import normalize_known_source_corruptions

from core import (
    BASE_DIR,
    CHARACTER_ALIASES_PATH,
    CONFIG_PATH,
    DATA_DIR,
    REPORTS_DIR,
    ROOT_DIR,
    SCRIPTS_DIR,
    SCRIPT_PATH,
    UPLOADS_DIR,
    VOICE_CONFIG_PATH,
    _batch_cancel_helper,
    _cancel_task,
    _combine_pass_stats,
    _combine_pass_totals,
    _extract_diff_highlights,
    _extract_failed_sections,
    _extract_new_aliases,
    _extract_review_stats,
    _format_book_summary,
    _format_pass_summary,
    _init_batch_state,
    _init_task_log,
    _insert_llm_summary,
    _markdown_aliases_lines,
    _markdown_book_pass_lines,
    _markdown_diff_highlights_lines,
    _markdown_heads_up_lines,
    _markdown_stats_table,
    _new_review_totals,
    _pause_task,
    _require_safe_filename,
    _resume_task,
    _run_claimed_background_task,
    _save_upload_limited,
    _stream_subprocess_to_logs,
    _task_log_path,
    _warn_corrupted_json,
    check_global_gpu_lock,
    claim_gpu_task,
    process_state,
    run_process,
)
from review_script import clear_checkpoint
from utils import atomic_json_write, backup_file_with_timestamp, safe_load_json, secure_filename


logger = logging.getLogger("AlexandriaUI")
router = APIRouter()
_upload_hash_cache = {}
_upload_hash_lock = threading.Lock()
_upload_dedupe_lock = threading.Lock()


class ReviewRequest(BaseModel):
    dedupe_speakers: bool = True

class ContextualReviewRequest(BaseModel):
    window_size: int = 4
    dedupe_speakers: bool = True

class BatchReviewRequest(BaseModel):
    script_names: List[str]            # names from the Scripts library (without .json)
    context_window: int = 0            # >0 enables contextual review
    dedupe_speakers: bool = True       # merge same-character aliases, consistent across the batch
    find_nicknames: bool = True        # run nickname discovery per book first, into the shared series alias file
    bidirectional: bool = False        # after the forward pass, re-scan in reverse so early books get
                                       # discovery seeded with full-series hindsight (requires find_nicknames)



def _write_batch_review_report(state: dict, names: List[str], bidirectional: bool, discover: bool) -> Optional[str]:
    """Write one plain-language Markdown summary covering an entire batch review run
    (whether it was 1 book or many).

    Returns the path to the written file, or None if it couldn't be written.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(REPORTS_DIR, f"batch_review_{timestamp}.md")

    tasks = state.get("tasks", [])
    total_books = len(names)
    if bidirectional:
        # The bare "stats"/"diffs" keys hold whichever pass ran last, so a book
        # that only completed the forward pass before cancellation would still
        # look "done" via the bare key. Require both passes' stats and a
        # "done" status (not "incomplete", which means VRAM cut a pass short).
        done = [t for t in tasks if t.get("stats_fwd") and t.get("stats_bwd") and t.get("status") == "done"]
    else:
        done = [t for t in tasks if t.get("stats_fwd") and t.get("status") == "done"]
    incomplete = [t for t in tasks if t.get("status") == "incomplete"]
    failed = [t for t in tasks if t.get("status") == "failed"]
    cancelled = [t for t in tasks if t.get("status") == "cancelled"]

    book_word = "book" if total_books == 1 else "books"
    intro = [
        "# Batch Review Report",
        "",
        f"*Generated {time.strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        f"The AI reviewer checked **{total_books} {book_word}** for possible mistakes — like "
        "the wrong character speaking a line, awkward wording, or repeated narration — and "
        "recorded its changes.",
    ]

    if bidirectional:
        intro += [
            "",
            "It went through the books twice: once in reading order, then a second "
            '"hindsight" pass from the last book back to the first, so things learned about '
            "characters later in the series could also be applied to earlier books.",
        ]

    if cancelled or state.get("cancel"):
        intro += ["", f"**Note:** this run was stopped early — {len(done)} of {total_books} "
                       f"{book_word} finished before it was cancelled."]
    if failed:
        names_list = ", ".join(f"*{t['name']}*" for t in failed)
        intro += ["", f"**Note:** {len(failed)} {'book' if len(failed) == 1 else 'books'} "
                       f"could not be reviewed (an error occurred): {names_list}"]
    if incomplete:
        names_list = ", ".join(f"*{t['name']}*" for t in incomplete)
        intro += ["", f"**Note:** {len(incomplete)} {'book' if len(incomplete) == 1 else 'books'} "
                       f"{'was' if len(incomplete) == 1 else 'were'} only partially reviewed: "
                       f"{names_list}. See the warnings below for the recorded reason and retry guidance."]

    if bidirectional:
        overall = _combine_pass_totals(state)
    else:
        overall = state["totals_fwd"]

    lines = list(intro)
    lines += ["", "## Overall totals", ""]
    lines += _markdown_stats_table(overall)

    if bidirectional:
        lines += ["", "### First pass (reading order)", ""]
        lines += _markdown_stats_table(state["totals_fwd"])
        lines += ["", "### Second pass (hindsight)", ""]
        lines += _markdown_stats_table(state["totals_bwd"])

    diff_pool = state.get("diff_pool", {"text": [], "speaker": []})
    overall_highlights = {
        "text_rewrites": sorted(diff_pool["text"], key=lambda h: h["magnitude"], reverse=True)[:5],
        "speaker_changes": diff_pool["speaker"][:5],
    }
    hl_lines = _markdown_diff_highlights_lines(overall_highlights, max_each=5)
    if hl_lines:
        lines += ["", "## Highlights", ""]
        lines += hl_lines

    heads_up = _markdown_heads_up_lines(overall)
    if heads_up:
        lines += ["", "## Things to check", ""]
        lines += heads_up

    if discover:
        aliases_fwd = state.get("aliases_fwd", [])
        aliases_bwd = state.get("aliases_bwd", [])
        lines += ["", "## New character names discovered", ""]
        if not aliases_fwd and not aliases_bwd:
            lines.append("- No new character names were found.")
        elif bidirectional:
            if aliases_fwd:
                lines += _markdown_aliases_lines(aliases_fwd, pass_label=" — first pass")
            if aliases_bwd:
                lines += _markdown_aliases_lines(aliases_bwd, pass_label=" — second/hindsight pass")
        else:
            lines += _markdown_aliases_lines(aliases_fwd)

    # Ask the LLM for a plain-English summary of the report so far, before appending
    # the (potentially very long) book-by-book breakdown.
    partial = bool(cancelled or failed or incomplete or state.get("cancel") or len(done) < total_books)
    lines = _insert_llm_summary(lines, len(intro), overall, incomplete=partial)

    if total_books > 1:
        lines += ["", "## Book-by-book breakdown", ""]
        for t in tasks:
            name = t.get("name", "?")
            status = t.get("status")
            lines += [f"### {name}", ""]
            if bidirectional:
                stats_fwd = t.get("stats_fwd")
                stats_bwd = t.get("stats_bwd")
                if stats_fwd or stats_bwd:
                    if stats_fwd:
                        lines += ["#### First pass (reading order)", ""]
                        lines += _markdown_book_pass_lines(
                            stats_fwd, t.get("diffs_fwd"), t.get("failures_fwd"), heading="#####")
                    if stats_bwd:
                        if stats_fwd:
                            lines.append("")
                        lines += ["#### Second pass (hindsight)", ""]
                        lines += _markdown_book_pass_lines(
                            stats_bwd, t.get("diffs_bwd"), t.get("failures_bwd"), heading="#####")
                elif status == "cancelled":
                    lines.append("- Not reviewed — the run was cancelled before reaching this book.")
                elif status == "failed":
                    lines.append("- Not reviewed — an error occurred for this book.")
                else:
                    lines.append("- Not reviewed.")
            else:
                stats = t.get("stats_fwd") or t.get("stats")
                if stats:
                    lines += _markdown_book_pass_lines(stats, t.get("diffs"), t.get("failures"))
                elif status == "cancelled":
                    lines.append("- Not reviewed — the run was cancelled before reaching this book.")
                elif status == "failed":
                    lines.append("- Not reviewed — an error occurred for this book.")
                else:
                    lines.append("- Not reviewed.")
            lines.append("")

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except OSError:
        return None
    return path


class _HTMLTextExtractor(HTMLParser):
    """Strip HTML tags from EPUB content, preserving block-level structure."""
    BLOCK_TAGS = frozenset({
        'p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
        'li', 'blockquote', 'br', 'hr', 'tr', 'section', 'article',
    })
    SKIP_TAGS = frozenset({'style', 'script'})

    def __init__(self):
        super().__init__()
        self.parts = []
        self._pending_newline = False
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
        elif tag in self.BLOCK_TAGS:
            self._pending_newline = True

    def handle_endtag(self, tag):
        if tag.lower() in self.SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth > 0:
            return
        if self._pending_newline and self.parts:
            self.parts.append('\n')
            self._pending_newline = False
        self.parts.append(data)

    def get_text(self):
        return ''.join(self.parts)


def extract_epub_text(epub_path: str) -> str:
    """Extract plain text from an EPUB file, ordered by spine (reading order).

    Parses the EPUB ZIP structure directly using stdlib only:
    META-INF/container.xml -> .opf manifest+spine -> XHTML content files.
    """
    with zipfile.ZipFile(epub_path, 'r') as zf:
        # 1. Find the OPF file path from container.xml
        container_xml = zf.read('META-INF/container.xml')
        container = ET.fromstring(container_xml)
        ns = {'c': 'urn:oasis:names:tc:opendocument:xmlns:container'}
        rootfile_el = container.find('.//c:rootfile', ns)
        if rootfile_el is None:
            raise ValueError("Invalid EPUB: no rootfile found in container.xml")
        opf_path = rootfile_el.get('full-path')

        # 2. Parse the OPF to get manifest (id->href) and spine (reading order)
        opf_xml = zf.read(opf_path)
        try:
            opf = ET.fromstring(opf_xml)
        except ET.ParseError:
            # Two verified Sigil-generated books contain this exact duplicated,
            # malformed metadata attribute. It is unrelated to the manifest or
            # spine; remove only that known fragment and otherwise fail closed.
            repaired_opf = opf_xml.replace(b' refines">"#pub-i"', b'')
            if repaired_opf == opf_xml:
                raise
            opf = ET.fromstring(repaired_opf)
        # Detect OPF namespace (varies between EPUB 2 and 3)
        opf_ns = opf.tag.split('}')[0] + '}' if '}' in opf.tag else ''

        # Build manifest: id -> href (resolve relative to OPF directory)
        opf_dir = opf_path.rsplit('/', 1)[0] + '/' if '/' in opf_path else ''
        manifest = {}
        for item in opf.findall(f'.//{opf_ns}item'):
            item_id = item.get('id')
            href = item.get('href')
            media_type = item.get('media-type', '')
            if item_id and href and 'html' in media_type:
                manifest[item_id] = opf_dir + href

        # Get spine order
        spine_ids = []
        for itemref in opf.findall(f'.//{opf_ns}itemref'):
            idref = itemref.get('idref')
            if idref:
                spine_ids.append(idref)

        # 3. Extract text from each spine item in order
        chapters = []
        for item_id in spine_ids:
            href = manifest.get(item_id)
            if href is None:
                continue
            try:
                html_bytes = zf.read(href)
            except KeyError:
                continue
            html_content = html_bytes.decode('utf-8', errors='replace')
            extractor = _HTMLTextExtractor()
            extractor.feed(html_content)
            text = extractor.get_text().strip()
            if text:
                chapters.append(text)

    return '\n\n'.join(chapters)


def _claim_unique_path(directory: str, filename: str) -> str:
    """Atomically reserve a unique path in directory for filename, returning the
    path to a newly-created empty file the caller should now write/truncate into.

    A directory scan picks a good starting candidate (avoiding O(n) O_EXCL
    failures when the directory is large), then os.O_EXCL claims it -
    closing the TOCTOU race a scan-then-write approach has under concurrent
    uploads of the same filename. Caps at 1000 attempts to prevent a DoS
    from a maliciously pre-populated directory.
    """
    existing = {e.name for e in os.scandir(directory) if e.is_file()}
    base, ext = os.path.splitext(filename)
    candidate = filename
    counter = 1
    while True:
        if candidate not in existing:
            path = os.path.join(directory, candidate)
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                os.close(fd)
                return path
            except FileExistsError:
                pass  # lost the race - fall through and try the next candidate
        if counter > 1000:
            raise RuntimeError(f"Too many collisions for filename: {filename}")
        counter += 1
        candidate = f"{base}_{counter}{ext}"


def _get_upload_hash(path: str) -> str:
    """Return a cached SHA-256 keyed by path, size, and modification time."""
    stat = os.stat(path)
    key = (path, stat.st_size, stat.st_mtime_ns)
    with _upload_hash_lock:
        cached = _upload_hash_cache.get(key)
    if cached:
        return cached
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    with _upload_hash_lock:
        _upload_hash_cache[key] = value
    return value


def _reuse_duplicate_upload(path: str) -> tuple[str, bool]:
    """Remove path and return an existing identical text upload when present."""
    # Serialize the scan-and-remove decision. Without this, two identical
    # concurrent uploads can each select the other as canonical and remove
    # both files, or one scan can stat a path the other just removed.
    with _upload_dedupe_lock:
        size = os.path.getsize(path)
        digest = _get_upload_hash(path)
        for entry in sorted(os.scandir(UPLOADS_DIR), key=lambda item: item.name):
            try:
                if (not entry.is_file() or entry.path == path or
                        os.path.splitext(entry.name)[1].lower() not in {".txt", ".md"} or
                        entry.stat().st_size != size):
                    continue
                if _get_upload_hash(entry.path) == digest:
                    os.remove(path)
                    return entry.path, True
            except FileNotFoundError:
                continue
    return path, False


def _get_reusable_uploads() -> List[dict]:
    uploads = []
    for entry in sorted(os.scandir(UPLOADS_DIR), key=lambda item: item.name.casefold()):
        if not entry.is_file() or os.path.splitext(entry.name)[1].lower() not in {".txt", ".md"}:
            continue
        stat = entry.stat()
        uploads.append({
            "filename": entry.name, "size": stat.st_size, "modified": stat.st_mtime,
            "sha256": _get_upload_hash(entry.path),
        })
    return uploads


def _select_upload(filename: str) -> str:
    safe_name = _require_safe_filename(filename, "Invalid upload filename")
    path = os.path.join(UPLOADS_DIR, safe_name)
    if not os.path.isfile(path) or os.path.splitext(path)[1].lower() not in {".txt", ".md"}:
        raise HTTPException(status_code=404, detail=f"Reusable upload '{filename}' not found.")
    state_path = os.path.join(DATA_DIR, "state.json")
    state = safe_load_json(state_path, default={})
    state["input_file_path"] = path
    state["active_book_id"] = secure_filename(os.path.splitext(safe_name)[0])
    atomic_json_write(state, state_path)
    return path


@router.get("/api/uploads")
async def list_reusable_uploads():
    return await asyncio.to_thread(_get_reusable_uploads)


class ExistingUploadRequest(BaseModel):
    filename: str


@router.post("/api/uploads/select")
async def select_existing_upload(request: ExistingUploadRequest):
    path = await asyncio.to_thread(_select_upload, request.filename)
    return {"status": "selected", "stored_filename": os.path.basename(path), "path": path}










@router.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    # Validate and sanitize filename to prevent path traversal
    safe_name = _require_safe_filename(file.filename or "", "Invalid or empty filename")
    file_path = await asyncio.to_thread(_claim_unique_path, UPLOADS_DIR, safe_name)
    await _save_upload_limited(file, file_path, 512 * 1024**2)

    # Convert EPUB to plain text
    if file_path.lower().endswith('.epub'):
        try:
            text = extract_epub_text(file_path)
        except Exception as e:
            os.remove(file_path)
            raise HTTPException(status_code=400, detail=f"Failed to process EPUB: {e}")
        if not text.strip():
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="No readable text content found in EPUB.")
        txt_name = os.path.basename(file_path).rsplit('.', 1)[0] + '.txt'
        txt_path = await asyncio.to_thread(_claim_unique_path, UPLOADS_DIR, txt_name)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)
        # The original .epub is no longer needed once its text is extracted;
        # leaving it behind leaks disk space as books accumulate.
        try:
            os.remove(file_path)
        except OSError:
            pass
        file_path = txt_path

    file_path, reused = await asyncio.to_thread(_reuse_duplicate_upload, file_path)

    # Save input path to state.json to be compatible with original scripts if needed
    state_path = os.path.join(DATA_DIR, "state.json")
    state = {}
    if os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as f:
            try:
                state = json.load(f)
            except (json.JSONDecodeError, ValueError) as e:
                _warn_corrupted_json("state", state_path, "overwriting with new data", e)

    state["input_file_path"] = file_path
    state["active_book_id"] = secure_filename(os.path.splitext(os.path.basename(file_path))[0])
    atomic_json_write(state, state_path)

    return {"filename": file.filename, "stored_filename": os.path.basename(file_path),
            "path": file_path, "reused": reused}

class GenerateScriptRequest(BaseModel):
    strip_front_matter: bool = True


def build_generate_script_command(input_file: str, output_path: Optional[str] = None,
                                  strip_front_matter: bool = True) -> List[str]:
    """Build the one production command used by single and batch generation."""
    command = [sys.executable, "-u", os.path.join(BASE_DIR, "three_pass_generate.py"),
               input_file, "--pass2-on-exhaustion", "fallback"]
    if output_path is not None:
        command.extend(["--output", output_path])
    if not strip_front_matter:
        command.append("--no-strip-front-matter")
    return command


@router.post("/api/generate_script")
async def generate_script(background_tasks: BackgroundTasks,
                           request: Optional[GenerateScriptRequest] = None):
    # Get input file from state.json
    state_path = os.path.join(DATA_DIR, "state.json")
    if not os.path.exists(state_path):
        raise HTTPException(status_code=400, detail="No input file selected")

    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
        input_file = state.get("input_file_path")

    if not input_file:
         raise HTTPException(status_code=400, detail="No input file found in state")

    check_global_gpu_lock("script")

    claim_gpu_task("script")
    command = build_generate_script_command(
        input_file,
        strip_front_matter=request is None or request.strip_front_matter,
    )
    background_tasks.add_task(run_process, command, "script")
    return {"status": "started"}

@router.post("/api/generate_script/cancel")
async def generate_script_cancel():
    return _cancel_task("script", "No script generation is currently running.", "Script generation process already exited.")



@router.post("/api/generate_script/pause")
async def generate_script_pause():
    return _pause_task("script", "No script generation is currently running.",
                        "Script generation is starting up, retry in a moment.",
                        "Script generation")

@router.post("/api/generate_script/resume")
async def generate_script_resume():
    return _resume_task("script", "No script generation is currently running.",
                         "Script generation")


@router.post("/api/review_script")
async def review_script(background_tasks: BackgroundTasks, request: Optional[ReviewRequest] = None):
    """Review the current annotated script. Accepts empty POST or JSON body."""
    if request is None:
        request = ReviewRequest()  # Use defaults
    if not os.path.exists(SCRIPT_PATH):
        raise HTTPException(status_code=400, detail="No annotated script found. Generate a script first.")

    check_global_gpu_lock("review")

    cmd = [sys.executable, "-u", "review_script.py"]
    if request.dedupe_speakers:
        cmd += ["--dedupe-speakers", "--remap-voice-config", VOICE_CONFIG_PATH,
                "--alias-registry", CHARACTER_ALIASES_PATH]
    claim_gpu_task("review")
    background_tasks.add_task(run_process, cmd, "review")
    return {"status": "started", "dedupe_speakers": request.dedupe_speakers}

@router.post("/api/review_script_contextual")
async def review_script_contextual(request: ContextualReviewRequest, background_tasks: BackgroundTasks):
    if not os.path.exists(SCRIPT_PATH):
        raise HTTPException(status_code=400, detail="No annotated script found. Generate a script first.")

    check_global_gpu_lock("review")

    window_size = max(1, min(int(request.window_size or 4), 12))
    total_entries = 0
    try:
        with open(SCRIPT_PATH, "r", encoding="utf-8") as f:
            total_entries = len(json.load(f))
    except (json.JSONDecodeError, ValueError, OSError) as e:
        _warn_corrupted_json("script", SCRIPT_PATH, "estimated_calls will read 0", e)
        total_entries = 0

    review_batch_size = 25
    try:
        cfg = load_app_config(CONFIG_PATH)
        generation = cfg.get("generation") or {}
        review_batch_size = max(1, int(generation.get("review_batch_size", 25)))
    except (ValueError, TypeError) as e:
        _warn_corrupted_json("config", CONFIG_PATH, "using default review_batch_size", e)

    estimated_calls = ceil(total_entries / review_batch_size) if total_entries else 0
    cmd = [sys.executable, "-u", "review_script.py", "--context-window", str(window_size)]
    if request.dedupe_speakers:
        cmd += ["--dedupe-speakers", "--remap-voice-config", VOICE_CONFIG_PATH,
                "--alias-registry", CHARACTER_ALIASES_PATH]
    claim_gpu_task("review")
    background_tasks.add_task(
        run_process,
        cmd,
        "review"
    )
    return {
        "status": "started",
        "mode": "contextual",
        "window_size": window_size,
        "batch_size": review_batch_size,
        "total_entries": total_entries,
        "estimated_calls": estimated_calls,
        "dedupe_speakers": request.dedupe_speakers,
    }


@router.post("/api/review_script/cancel")
async def review_script_cancel():
    return _cancel_task("review", "No script review is currently running.", "Script review process already exited.")


@router.post("/api/review_script/pause")
async def review_script_pause():
    return _pause_task("review", "No script review is currently running.",
                        "Script review is starting up, retry in a moment.",
                        "Script review")


@router.post("/api/review_script/resume")
async def review_script_resume():
    return _resume_task("review", "No script review is currently running.",
                         "Script review")


@router.post("/api/find_nicknames")
async def find_nicknames_endpoint(background_tasks: BackgroundTasks):
    """Scan the working script for character nicknames/aliases and write character_aliases.json."""
    if not os.path.exists(SCRIPT_PATH):
        raise HTTPException(status_code=400, detail="No annotated script found. Generate a script first.")
    # nicknames runs the LLM, so it must claim the GPU lock (this also guards
    # against a duplicate start, replacing the old running-flag check).
    claim_gpu_task("nicknames")
    cmd = [sys.executable, "-u", "find_nicknames.py",
           "--aliases-file", CHARACTER_ALIASES_PATH, "--append"]
    background_tasks.add_task(run_process, cmd, "nicknames")
    return {"status": "started"}


@router.post("/api/find_nicknames/cancel")
async def find_nicknames_cancel():
    return _cancel_task("nicknames", "No nickname discovery is currently running.", "Nickname discovery already exited.")


@router.post("/api/find_nicknames/pause")
async def find_nicknames_pause():
    return _pause_task("nicknames", "No nickname discovery is currently running.",
                        "Nickname discovery is starting up, retry in a moment.",
                        "Nickname discovery")


@router.post("/api/find_nicknames/resume")
async def find_nicknames_resume():
    return _resume_task("nicknames", "No nickname discovery is currently running.",
                         "Nickname discovery")


@router.get("/api/character_aliases")
async def get_character_aliases():
    """Return the current alias map { alias: canonical }."""
    aliases = safe_load_json(CHARACTER_ALIASES_PATH, default={})
    if not isinstance(aliases, dict):
        return {}
    # Hide identity rows (NAME -> NAME) — they're inert and only clutter the editor.
    # Exact-match only, so a legitimate case-fix alias (kenji -> KENJI) stays visible.
    return {k: v for k, v in aliases.items()
            if isinstance(k, str) and isinstance(v, str) and k.strip() != v.strip()}


@router.post("/api/character_aliases")
async def save_character_aliases(aliases: Dict[str, str]):
    """Overwrite the alias map (lets the user correct discovered nicknames before review)."""
    cleaned = {k.strip(): v.strip() for k, v in aliases.items() if k.strip() and v.strip()}
    atomic_json_write(cleaned, CHARACTER_ALIASES_PATH)
    return {"status": "saved", "count": len(cleaned)}


@router.post("/api/review_script/batch/start")
async def review_script_batch_start(request: BatchReviewRequest, background_tasks: BackgroundTasks):
    """Review multiple saved scripts from the Scripts library, in place.
    A shared alias registry keeps merged character names consistent across the batch."""
    check_global_gpu_lock("batch_review")
    if not request.script_names:
        raise HTTPException(status_code=400, detail="No scripts selected.")

    window = max(0, min(int(request.context_window or 0), 12))
    dedupe = bool(request.dedupe_speakers)
    discover = bool(request.find_nicknames) and dedupe
    # A backward pass only adds value when discovery is on (it re-scans early books with the
    # now-complete registry as hindsight context). With discovery off it would be a pure re-apply.
    bidirectional = bool(request.bidirectional) and discover

    names = request.script_names
    total = len(names)

    def _run():
        state = process_state["batch_review"]
        prefix = "bidirectional " if bidirectional else ""
        _init_batch_state(state,
                          [f"Starting {prefix}batch review of {total} script(s)..."],
                          [{"name": n, "status": "pending"} for n in names])
        state["bidirectional"] = bidirectional
        state["totals_fwd"] = _new_review_totals()
        state["totals_bwd"] = _new_review_totals()
        state["aliases_fwd"] = []
        state["aliases_bwd"] = []
        state["diff_pool"] = {"text": [], "speaker": []}

        # One full on-disk log for the whole batch (in-memory list is a capped tail)
        log_path = _init_task_log("batch_review")

        # One shared registry for the whole batch so canonical names align across books
        registry_path = os.path.join(SCRIPTS_DIR, ".series_aliases.json") if dedupe else None

        def _process_book(i: int, name: str, tag: str = "") -> bool:
            """Discover + review one book in place. Returns False to stop the batch (cancel)."""
            state["current_task_idx"] = i
            orig_status = state["tasks"][i].get("status")
            state["tasks"][i]["status"] = "running"

            safe_name = secure_filename(name)
            if not safe_name:
                state["logs"].append(f"--- [{i+1}/{total}]{tag} Skipping — invalid name: {name} ---")
                state["tasks"][i]["status"] = "failed"
                return True
            script_path = os.path.join(SCRIPTS_DIR, f"{safe_name}.json")
            if not os.path.exists(script_path):
                state["logs"].append(f"--- [{i+1}/{total}]{tag} Skipping — not found: {name} ---")
                state["tasks"][i]["status"] = "failed"
                return True

            state["logs"].append(f"--- [{i+1}/{total}]{tag} Reviewing '{name}' ---")

            # Optional nickname discovery first, accumulating into the shared series registry
            if discover and registry_path:
                state["logs"].append(f"[{i+1}]{tag} Discovering nicknames...")
                nick_cmd = [
                    sys.executable, "-u",
                    os.path.join(BASE_DIR, "find_nicknames.py"),
                    "--input", script_path,
                    "--aliases-file", registry_path,
                    "--append",
                ]
                _, nick_lines = _stream_subprocess_to_logs(nick_cmd, BASE_DIR, state, log_prefix=f"[{i+1}] ", log_file=log_path)
                if state.get("cancel"):
                    state["tasks"][i]["status"] = "cancelled"
                    return False
                new_aliases = _extract_new_aliases(nick_lines)
                if new_aliases:
                    state["tasks"][i]["aliases_found"] = new_aliases
                    bucket = state["aliases_bwd"] if tag == " [bwd]" else state["aliases_fwd"]
                    for a in new_aliases:
                        bucket.append({**a, "book": name})
                    state["logs"].append(
                        f"[{i+1}]{tag} New alias(es): " +
                        ", ".join(f"'{a['variant']}' -> '{a['canonical']}'" for a in new_aliases)
                    )

            # Only clear checkpoint at the start of the first pass (forward).
            # For bidirectional reviews, preserve the forward pass checkpoint
            # so if the backward pass crashes, we can resume from where forward left off.
            should_clear = True
            if state.get("bidirectional") and state.get("current_pass") == "bwd":
                # Don't clear checkpoint during backward pass - preserve forward progress
                should_clear = False
                if orig_status == "incomplete":
                    # The forward pass on this book was VRAM-aborted and left behind
                    # its own partial checkpoint (forward-pass progress/aliases). That
                    # checkpoint isn't valid for the backward pass - reusing it would
                    # silently splice forward-pass output into the backward result.
                    should_clear = True

            if should_clear:
                clear_checkpoint(script_path)

            cmd = [
                sys.executable, "-u",
                os.path.join(BASE_DIR, "review_script.py"),
                "--input", script_path,
                "--output", script_path,
            ]
            if window > 0:
                cmd += ["--context-window", str(window)]
            if dedupe:
                cmd += ["--dedupe-speakers", "--alias-registry", registry_path]
                companion = os.path.join(SCRIPTS_DIR, f"{safe_name}.voice_config.json")
                if os.path.exists(companion):
                    cmd += ["--remap-voice-config", companion]

            rc, own_lines = _stream_subprocess_to_logs(cmd, BASE_DIR, state, log_prefix=f"[{i+1}] ", log_file=log_path)

            if state.get("cancel"):
                state["tasks"][i]["status"] = "cancelled"
                return False
            elif rc == 0:
                # Bidirectional runs review each book twice (forward, then backward); keep
                # each pass's stats/diffs separate so the per-book breakdown doesn't lose
                # the first pass's results when the second pass overwrites them.
                pass_key = "bwd" if tag == " [bwd]" else "fwd"
                stats = _extract_review_stats(own_lines)
                if stats is None:
                    # rc == 0 but the summary line is missing/malformed - we
                    # genuinely don't know whether this book finished cleanly
                    # or hit a VRAM abort with no recorded summary. Treat as
                    # incomplete rather than silently calling it "done".
                    state["tasks"][i]["status"] = "incomplete"
                elif (stats.get("batches_failed", 0) > 0 or
                      stats.get("batches_skipped_vram", 0) > 0):
                    # The reviewer bailed out early to avoid an OOM; entries past the
                    # abort point or failed batches were left unreviewed, and a checkpoint
                    # may remain on disk for a future resume. Don't report this book as done.
                    state["tasks"][i]["status"] = "incomplete"
                else:
                    state["tasks"][i]["status"] = "done"
                if stats:
                    state["tasks"][i][f"stats_{pass_key}"] = stats
                    # "stats" is the combined fwd+bwd total used by the per-book
                    # badge tooltip — recompute it from whichever pass(es) have
                    # run so far rather than letting the last pass overwrite it.
                    # Only combine stats_bwd in if this run is actually
                    # bidirectional - otherwise stats_bwd is never populated
                    # by design (no backward pass ever runs), and combining
                    # it in would mark every single-pass book "partial" even
                    # when that book's one-and-only pass succeeded cleanly.
                    pass_stats = [state["tasks"][i].get("stats_fwd")]
                    if state.get("bidirectional"):
                        pass_stats.append(state["tasks"][i].get("stats_bwd"))
                    state["tasks"][i]["stats"] = _combine_pass_stats(*pass_stats)
                    totals = state["totals_bwd"] if tag == " [bwd]" else state["totals_fwd"]
                    for key in totals:
                        if key != "books_done":
                            totals[key] += stats[key]
                    totals["books_done"] += 1
                    state["logs"].append(_format_book_summary(i, total, tag, name, stats))
                else:
                    # The subprocess exited 0 but its "Review complete: X -> Y
                    # entries" summary line wasn't found - surface this rather
                    # than silently recording no stats for an otherwise "done" book.
                    state["logs"].append(
                        f"[{i+1}]{tag} Warning: '{name}' finished but no summary "
                        "stats were found in its output."
                    )
                highlights = _extract_diff_highlights(own_lines)
                failures = _extract_failed_sections(own_lines)
                if failures["sections"]:
                    state["tasks"][i][f"failures_{pass_key}"] = failures
                    state["tasks"][i]["failures"] = failures
                if highlights["text_rewrites"] or highlights["speaker_changes"]:
                    state["tasks"][i][f"diffs_{pass_key}"] = highlights
                    # Combine diffs from both passes for the UI badge tooltip
                    existing_diffs = state["tasks"][i].get("diffs", {})
                    combined = {
                        "text_rewrites": existing_diffs.get("text_rewrites", []) + highlights["text_rewrites"],
                        "speaker_changes": existing_diffs.get("speaker_changes", []) + highlights["speaker_changes"],
                    }
                    state["tasks"][i]["diffs"] = combined
                    for item in highlights["text_rewrites"]:
                        state["diff_pool"]["text"].append({**item, "book": name})
                    for item in highlights["speaker_changes"]:
                        state["diff_pool"]["speaker"].append({**item, "book": name})
            else:
                state["tasks"][i]["status"] = "failed"
                state["logs"].append(f"[{i+1}]{tag} Failed (exit {rc}): {name}")
            return True

        # Forward pass (reading order)
        state["current_pass"] = "fwd"
        if bidirectional:
            state["logs"].append("=== Forward pass (reading order) ===")
        for i, name in enumerate(names):
            if state["cancel"]:
                state["logs"].append("Batch review cancelled.")
                break
            if not _process_book(i, name, tag=" [fwd]" if bidirectional else ""):
                break

        state["logs"].append(_format_pass_summary(
            "Forward pass" if bidirectional else "Batch review",
            state["totals_fwd"], state["aliases_fwd"], show_aliases=discover))

        # Backward pass — re-scan from the end so early books get discovery seeded with the
        # now-complete series registry (catches references that only resolve later in the series).
        if bidirectional and not state["cancel"]:
            state["logs"].append("=== Backward pass (hindsight: re-scanning from the end) ===")
            state["current_pass"] = "bwd"
            for i in range(total - 1, -1, -1):
                if state["cancel"]:
                    state["logs"].append("Batch review cancelled.")
                    break
                if not _process_book(i, names[i], tag=" [bwd]"):
                    break

            state["logs"].append(_format_pass_summary(
                "Backward pass (hindsight)", state["totals_bwd"], state["aliases_bwd"], show_aliases=discover))

            overall_totals = _combine_pass_totals(state)
            overall_aliases = state["aliases_fwd"] + state["aliases_bwd"]
            state["logs"].append(_format_pass_summary("Overall", overall_totals, overall_aliases, show_aliases=discover))

        report_path = _write_batch_review_report(state, names, bidirectional, discover)
        if report_path:
            state["artifacts"].append({
                "artifact_path": report_path,
                "kind": "batch_review_report",
                "source_paths": [os.path.join(SCRIPTS_DIR, f"{name}.json") for name in names],
                "config_path": CONFIG_PATH,
            })
            state["logs"].append(f"Wrote batch review report: {os.path.relpath(report_path, ROOT_DIR)}")

        state["running"] = False
        state["logs"].append("Batch review finished.")

    claim_gpu_task("batch_review")
    background_tasks.add_task(_run_claimed_background_task, "batch_review", _run)
    return {"status": "started", "task_count": total, "bidirectional": bidirectional}




@router.post("/api/review_script/batch/cancel")
async def review_script_batch_cancel():
    return _batch_cancel_helper("batch_review")


@router.post("/api/review_script/batch/pause")
async def review_script_batch_pause():
    return _pause_task("batch_review", "No batch review is currently running.",
                        "Batch review is starting up, retry in a moment.",
                        "Batch review")


@router.post("/api/review_script/batch/resume")
async def review_script_batch_resume():
    return _resume_task("batch_review", "No batch review is currently running.",
                         "Batch review")


class BatchScriptTask(BaseModel):
    filename: str  # filename inside uploads/

class BatchScriptRequest(BaseModel):
    tasks: List[BatchScriptTask]
    collision_policy: Literal["cancel", "version", "replace"] = "cancel"
    strip_front_matter: bool = True


def _get_versioned_script_path(path: str) -> str:
    base, ext = os.path.splitext(path)
    counter = 2
    candidate = path
    while os.path.exists(candidate):
        candidate = f"{base}_{counter}{ext}"
        counter += 1
    return candidate


def _resolve_batch_output_path(output_path, policy, reserved_outputs):
    """Decide what happens to a batch job's output path given a collision.

    Returns (path, action) where action is one of:
      - "ok": no collision; use path as-is.
      - "skip": collision policy is "cancel"; caller should skip the job.
      - "version": path was suffixed to a versioned, non-colliding path.
      - "backup": path is unchanged but the existing on-disk file should be
        backed up before being overwritten.

    A collision with a *reserved* (in-batch) output is never resolved by
    "replace" -- replacing a file this same batch is about to produce is
    never what "replace" means, so it is versioned instead, exactly like
    the explicit "version" policy.
    """
    is_reserved = output_path in reserved_outputs
    exists_on_disk = os.path.exists(output_path)
    if not is_reserved and not exists_on_disk:
        return output_path, "ok"
    if policy == "cancel":
        return output_path, "skip"
    if policy == "version" or is_reserved:
        candidate = _get_versioned_script_path(output_path) if exists_on_disk else output_path
        base, extension = os.path.splitext(candidate)
        counter = 2
        while candidate in reserved_outputs:
            candidate = f"{base}_{counter}{extension}"
            counter += 1
        return candidate, "version"
    # policy == "replace" and the collision is disk-only (not reserved)
    return output_path, "backup"


def _resolve_batch_script_input(filename: str) -> str:
    safe_filename = secure_filename(filename)
    if not safe_filename:
        raise ValueError(f"Invalid filename: {filename}")
    input_path = os.path.join(UPLOADS_DIR, safe_filename)
    if not os.path.exists(input_path) and os.path.splitext(safe_filename)[1].lower() == ".epub":
        input_path = os.path.join(UPLOADS_DIR, os.path.splitext(safe_filename)[0] + ".txt")
    if not os.path.exists(input_path):
        raise ValueError(f"File not found: {filename}")
    return input_path


def build_batch_script_preflight(jobs):
    """Build the shared read-only sizing report used by the UI and dispatcher."""
    config = load_app_config(CONFIG_PATH)
    llm = config.get("llm") or {}
    status = get_planned_ideal_settings(
        config.get("llm_mode", "local"), llm.get("base_url", ""),
        llm.get("model_name", ""), config.get("llm_remote_ssh"))
    parallel = max(1, int(status.get("parallel") or 1))
    context = int(status.get("context_length") or 0)
    generation = config.get("generation") or {}
    prompts = config.get("prompts") or {}
    books = []
    for job in jobs:
        with open(job["input_path"], encoding="utf-8") as source:
            text = fix_mojibake(source.read())
        text, normalization_count = normalize_known_source_corruptions(text)
        chunks = split_into_chunks(text, generation.get("chunk_size", 3000))
        report = build_book_request_preflight(
            chunks, prompts.get("system_prompt") or DEFAULT_SYSTEM_PROMPT,
            prompts.get("user_prompt") or DEFAULT_USER_PROMPT,
            generation.get("max_tokens", 4096), context, 1)
        unicode_report = audit_unicode_text(text)
        books.append({
            "filename": job["filename"],
            "chunk_count": report["chunk_count"],
            "worst_predicted_tokens": report["worst_predicted_tokens"],
            "p95_predicted_tokens": report["p95_predicted_tokens"],
            "average_predicted_tokens": report["average_predicted_tokens"],
            "scripts": unicode_report["scripts"],
            "is_nfc": unicode_report["is_nfc"],
            "known_normalizations": normalization_count,
        })
    worst = max((book["worst_predicted_tokens"] for book in books), default=0)
    safe = min(parallel, len(jobs))
    while safe > 1 and worst * safe > context:
        safe -= 1
    safe = max(1, safe)
    per_slot = context // safe if context else 0
    for book in books:
        book["fits_selected_slot"] = bool(per_slot and book["worst_predicted_tokens"] <= per_slot)
    fallback = None
    if safe < min(parallel, len(jobs)):
        fallback = (f"Reduced concurrency from {min(parallel, len(jobs))} to {safe} because "
                    f"the largest predicted request needs {worst} tokens.")
    return {"book_count": len(books), "workers": safe, "loaded_parallel": parallel,
            "context_length": context, "per_slot_context": per_slot,
            "worst_request_tokens": worst, "fallback_reason": fallback, "books": books}


def _get_batch_script_workers(jobs):
    report = build_batch_script_preflight(jobs)
    return report["workers"], report["worst_request_tokens"], report["context_length"]


@router.post("/api/generate_script/batch/preflight")
async def generate_script_batch_preflight(request: BatchScriptRequest):
    if not request.tasks:
        raise HTTPException(status_code=400, detail="No files provided.")
    try:
        jobs = [{"filename": task.filename,
                 "input_path": _resolve_batch_script_input(task.filename)}
                for task in request.tasks]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return await asyncio.to_thread(build_batch_script_preflight, jobs)


def _run_batch_script_job(job, state, log_path, total):
    index = job["index"]
    if state.get("cancel"):
        state["tasks"][index]["status"] = "cancelled"
        return
    state["tasks"][index]["status"] = "running"
    state["logs"].append(f"--- [{index + 1}/{total}] {job['filename']} ---")
    env = os.environ.copy()
    if state.get("run_id"):
        env["ALEXANDRIA_RUN_ID"] = state["run_id"]
    command = build_generate_script_command(
        job["input_path"], output_path=job["output_path"],
        strip_front_matter=job.get("strip_front_matter", True),
    )
    rc, _ = _stream_subprocess_to_logs(
        command, BASE_DIR, state, log_prefix=f"[{index + 1}] ",
        log_file=log_path, env=env)
    if state.get("cancel"):
        state["tasks"][index]["status"] = "cancelled"
    elif rc == 0:
        state["tasks"][index].update({"status": "done", "saved_as": job["safe_stem"]})
        state["logs"].append(f"[{index + 1}] Saved as '{job['safe_stem']}' in Scripts library.")
    else:
        state["tasks"][index]["status"] = "failed"
        state["logs"].append(f"[{index + 1}] Failed (exit {rc}): {job['filename']}")


@router.post("/api/generate_script/batch/start")
async def generate_script_batch_start(request: BatchScriptRequest, background_tasks: BackgroundTasks):
    """Process multiple text/EPUB files sequentially through generate_script.py."""
    check_global_gpu_lock("batch_script")
    if not request.tasks:
        raise HTTPException(status_code=400, detail="No files provided.")

    def _run():
        state = process_state["batch_script"]
        _init_batch_state(state,
                          [f"Starting batch of {len(request.tasks)} file(s)..."],
                          [{"filename": t.filename, "status": "pending"} for t in request.tasks])

        # One full on-disk log for the whole batch (in-memory list is a capped tail)
        log_path = _init_task_log("batch_script")

        jobs = []
        reserved_outputs = set()
        for i, task in enumerate(request.tasks):
            if state["cancel"]:
                state["logs"].append("Batch cancelled.")
                break
            try:
                input_path = _resolve_batch_script_input(task.filename)
            except ValueError as exc:
                state["logs"].append(f"[{i+1}/{len(request.tasks)}] Skipping — {exc}")
                state["tasks"][i]["status"] = "failed"
                continue

            stem = os.path.splitext(os.path.basename(input_path))[0]
            safe_stem = secure_filename(stem) or f"batch_{i+1}"
            output_path = os.path.join(SCRIPTS_DIR, f"{safe_stem}.json")
            was_reserved = output_path in reserved_outputs
            output_path, action = _resolve_batch_output_path(
                output_path, request.collision_policy, reserved_outputs)
            if action == "skip":
                state["logs"].append(
                    f"[{i+1}] Skipping — saved script '{safe_stem}' already exists. "
                    "Choose version or replace explicitly.")
                state["tasks"][i]["status"] = "failed"
                continue
            if action == "version":
                if was_reserved and request.collision_policy == "replace":
                    state["logs"].append(
                        f"[{i+1}] '{safe_stem}.json' collides with an output already "
                        "produced by this batch — writing "
                        f"'{os.path.basename(output_path)}' instead of replacing.")
                safe_stem = os.path.splitext(os.path.basename(output_path))[0]
            elif action == "backup":
                backup = backup_file_with_timestamp(output_path)
                state["logs"].append(
                    f"[{i+1}] Backed up existing script as '{os.path.basename(backup)}'.")

            reserved_outputs.add(output_path)
            jobs.append({"index": i, "filename": task.filename, "input_path": input_path,
                         "output_path": output_path, "safe_stem": safe_stem,
                         "strip_front_matter": request.strip_front_matter})

        if jobs and not state.get("cancel"):
            config = load_app_config(CONFIG_PATH)
            llm = config.get("llm") or {}
            _, _, settings_message = ensure_ideal_settings(
                config.get("llm_mode", "local"), llm.get("base_url", ""),
                llm.get("model_name", ""), config.get("llm_remote_ssh"))
            state["logs"].append(f"Batch LM Studio preflight: {settings_message}")
            workers, worst, context = _get_batch_script_workers(jobs)
            state["workers"] = workers
            state["logs"].append(
                f"Batch preflight: workers={workers}, worst_request={worst}, context={context}")
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_run_batch_script_job, job, state, log_path,
                                           len(request.tasks)) for job in jobs]
                for future in futures:
                    future.result()

        state["running"] = False
        state["logs"].append("Batch script generation finished.")

    claim_gpu_task("batch_script")
    background_tasks.add_task(_run_claimed_background_task, "batch_script", _run)
    return {"status": "started", "task_count": len(request.tasks)}


@router.post("/api/generate_script/batch/cancel")
async def generate_script_batch_cancel():
    return _batch_cancel_helper("batch_script")


@router.post("/api/generate_script/batch/pause")
async def generate_script_batch_pause():
    return _pause_task("batch_script", "No batch script generation is currently running.",
                        "Batch script generation is starting up, retry in a moment.",
                        "Batch script generation")


@router.post("/api/generate_script/batch/resume")
async def generate_script_batch_resume():
    return _resume_task("batch_script", "No batch script generation is currently running.",
                         "Batch script generation")


@router.get("/api/annotated_script")
async def get_annotated_script():
    """Return the current working annotated_script.json.

    No SPA caller - intentionally kept as a programmatic/curl-accessible
    read endpoint (exercised by test_api.py's test_get_annotated_script).
    """
    if not os.path.exists(SCRIPT_PATH):
        raise HTTPException(status_code=404, detail="No annotated script found")
    with open(SCRIPT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@router.get("/api/status/{task_name}")
async def get_status(task_name: str):
    if task_name not in process_state:
        raise HTTPException(status_code=404, detail="Task not found")
    state = dict(process_state[task_name])
    state.pop("process", None)
    state.pop("processes", None)
    return state


@router.get("/api/logs/{task_name}")
async def get_task_log(task_name: str, download: bool = False):
    """Serve the complete on-disk log for a task (the in-memory status only keeps a
    capped tail). Use ?download=true to download the file."""
    if task_name not in process_state:
        raise HTTPException(status_code=404, detail="Task not found")
    log_path = _task_log_path(task_name)
    if not os.path.exists(log_path):
        raise HTTPException(status_code=404, detail="No log file for this task yet.")
    filename = f"{task_name}.log"
    return FileResponse(
        log_path,
        media_type="text/plain",
        filename=filename if download else None,
    )
