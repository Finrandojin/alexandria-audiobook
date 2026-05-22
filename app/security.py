"""Security helpers for path handling, uploads, URLs, and secrets."""
import ipaddress
import os
import re
import socket
import zipfile
from typing import Optional
from urllib.parse import urlparse

from fastapi import HTTPException, UploadFile

# Default max upload size: 100 MB (override with ALEXANDRIA_MAX_UPLOAD_MB)
MAX_UPLOAD_BYTES = int(os.environ.get("ALEXANDRIA_MAX_UPLOAD_MB", "100")) * 1024 * 1024

# Optional API token for exposed deployments (ALEXANDRIA_API_TOKEN)
API_TOKEN = os.environ.get("ALEXANDRIA_API_TOKEN", "").strip()

SECRET_MASK = "***"

_BLOCKED_HOSTS = frozenset({
    "metadata.google.internal",
})

_BLOCKED_NETS = (
    ipaddress.ip_network("169.254.0.0/16"),  # link-local / cloud metadata
    ipaddress.ip_network("fd00::/8"),        # ULA IPv6 metadata endpoints
)


def sanitize_resource_name(name: str) -> str:
    """Make a string safe for use as a directory or script filename stem."""
    name = re.sub(r"[^\w\- ]", "", name).strip()
    name = re.sub(r"\s+", "_", name)
    return name.lower()


def require_resource_name(name: str, field: str = "name") -> str:
    """Validate and return a sanitized resource name, or raise HTTP 400."""
    safe = sanitize_resource_name(name)
    if not safe:
        raise HTTPException(status_code=400, detail=f"Invalid {field}.")
    return safe


def safe_join(base_dir: str, *parts: str) -> str:
    """Join path parts and ensure the result stays inside base_dir."""
    base = os.path.realpath(base_dir)
    path = os.path.realpath(os.path.join(base, *parts))
    if path != base and not path.startswith(base + os.sep):
        raise HTTPException(status_code=400, detail="Invalid path.")
    return path


def safe_upload_filename(original: str) -> str:
    """Return a safe basename for uploaded files; reject traversal attempts."""
    if not original:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    normalized = original.replace("\\", "/")
    if ".." in normalized.split("/"):
        raise HTTPException(status_code=400, detail="Invalid filename.")
    base = os.path.basename(normalized)
    if not base or base in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid filename.")
    stem, ext = os.path.splitext(base)
    safe_stem = sanitize_resource_name(stem)
    if not safe_stem:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    if ext and not re.match(r"^\.[\w.]+$", ext.lower()):
        raise HTTPException(status_code=400, detail="Invalid file extension.")
    return safe_stem + ext.lower()


def safe_basename(filename: str, field: str = "filename") -> str:
    """Allow only a plain basename (no directories) for file references."""
    if not filename or filename != os.path.basename(filename.replace("\\", "/")):
        raise HTTPException(status_code=400, detail=f"Invalid {field}.")
    if ".." in filename:
        raise HTTPException(status_code=400, detail=f"Invalid {field}.")
    return filename


async def read_upload_limited(
    upload_file: UploadFile,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> bytes:
    """Read an upload in chunks, rejecting payloads above max_bytes."""
    chunks = []
    total = 0
    while True:
        chunk = await upload_file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large (max {max_bytes // (1024 * 1024)} MB).",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def safe_extract_zip(zip_path: str, dest_dir: str) -> None:
    """Extract a ZIP archive, rejecting Zip Slip paths."""
    dest = os.path.realpath(dest_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            target = os.path.realpath(os.path.join(dest, member.filename))
            if target != dest and not target.startswith(dest + os.sep):
                raise HTTPException(
                    status_code=400,
                    detail="ZIP archive contains unsafe paths.",
                )
        zf.extractall(dest)


def _resolve_host_ips(hostname: str):
    """Resolve hostname to IP addresses for SSRF checks."""
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise HTTPException(status_code=400, detail=f"Cannot resolve host: {hostname}") from exc
    ips = []
    for info in infos:
        ip_str = info[4][0]
        try:
            ips.append(ipaddress.ip_address(ip_str))
        except ValueError:
            continue
    if not ips:
        raise HTTPException(status_code=400, detail=f"Cannot resolve host: {hostname}")
    return ips


def validate_http_url(url: str, field: str = "url") -> str:
    """Validate an HTTP(S) URL and block known SSRF targets."""
    if not url or not isinstance(url, str):
        raise HTTPException(status_code=400, detail=f"Invalid {field}.")
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail=f"{field} must use http or https.")
    if not parsed.hostname:
        raise HTTPException(status_code=400, detail=f"Invalid {field}: missing host.")

    hostname = parsed.hostname.lower()
    if hostname in _BLOCKED_HOSTS:
        raise HTTPException(status_code=400, detail=f"Blocked {field} host.")

    for ip in _resolve_host_ips(hostname):
        if any(ip in net for net in _BLOCKED_NETS):
            raise HTTPException(status_code=400, detail=f"Blocked {field} host.")

    return url.strip()


def mask_secret(value: Optional[str]) -> Optional[str]:
    """Mask a secret for API responses."""
    if not value or value == "local":
        return value
    if len(value) <= 4:
        return SECRET_MASK
    return value[:2] + SECRET_MASK + value[-2:]


def is_masked_secret(value: Optional[str]) -> bool:
    """Return True if value looks like a masked placeholder from the UI."""
    return bool(value and SECRET_MASK in value)


def merge_preserved_secret(new_value: Optional[str], existing_value: Optional[str]) -> Optional[str]:
    """Keep the stored secret when the client submits a masked placeholder."""
    if is_masked_secret(new_value) and existing_value:
        return existing_value
    return new_value


def mask_config_secrets(config: dict) -> dict:
    """Return a copy of config with secrets masked for GET responses."""
    masked = dict(config)
    llm = dict(masked.get("llm") or {})
    if "api_key" in llm:
        llm["api_key"] = mask_secret(llm.get("api_key"))
    masked["llm"] = llm
    return masked


def validate_input_file_path(path: str, allowed_dirs) -> str:
    """Ensure an input file path resolves inside one of the allowed directories."""
    if not path:
        raise HTTPException(status_code=400, detail="No input file found in state")
    resolved = os.path.realpath(path)
    for base in allowed_dirs:
        base_resolved = os.path.realpath(base)
        if resolved == base_resolved or resolved.startswith(base_resolved + os.sep):
            if os.path.isfile(resolved):
                return resolved
            raise HTTPException(status_code=400, detail="Input file not found")
    raise HTTPException(status_code=400, detail="Invalid input file path")
