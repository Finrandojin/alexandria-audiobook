#!/usr/bin/env python3
"""Unit tests for security helpers."""

import os
import tempfile
import zipfile

from fastapi import HTTPException

from security import (
    mask_secret,
    merge_preserved_secret,
    safe_basename,
    safe_extract_zip,
    safe_join,
    safe_upload_filename,
    sanitize_resource_name,
    validate_http_url,
)


def _expect_http_error(func):
    try:
        func()
    except HTTPException:
        return
    raise AssertionError("Expected HTTPException")


def test_sanitize_resource_name():
    assert sanitize_resource_name("My Script!") == "my_script"
    assert sanitize_resource_name("../../../etc") == "etc"


def test_safe_join_blocks_traversal():
    with tempfile.TemporaryDirectory() as base:
        _expect_http_error(lambda: safe_join(base, "..", "secret.txt"))


def test_safe_upload_filename_rejects_traversal():
    _expect_http_error(lambda: safe_upload_filename("../../secret.txt"))


def test_safe_upload_filename_accepts_simple_name():
    assert safe_upload_filename("_test_upload.txt") == "_test_upload.txt"


def test_safe_basename_rejects_paths():
    _expect_http_error(lambda: safe_basename("../preview.wav"))


def test_safe_extract_zip_blocks_slip():
    with tempfile.TemporaryDirectory() as dest:
        zip_path = os.path.join(dest, "evil.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("../outside.txt", "pwnd")
        _expect_http_error(lambda: safe_extract_zip(zip_path, os.path.join(dest, "out")))


def test_mask_and_merge_secret():
    assert mask_secret("supersecretkey") == "su***ey"
    assert merge_preserved_secret("su***ey", "supersecretkey") == "supersecretkey"


def test_validate_http_url_blocks_metadata():
    _expect_http_error(
        lambda: validate_http_url("http://169.254.169.254/latest/meta-data/")
    )


def test_validate_http_url_allows_localhost():
    assert validate_http_url("http://127.0.0.1:11434/v1").startswith("http://")


if __name__ == "__main__":
    tests = [
        test_sanitize_resource_name,
        test_safe_join_blocks_traversal,
        test_safe_upload_filename_rejects_traversal,
        test_safe_upload_filename_accepts_simple_name,
        test_safe_basename_rejects_paths,
        test_safe_extract_zip_blocks_slip,
        test_mask_and_merge_secret,
        test_validate_http_url_blocks_metadata,
        test_validate_http_url_allows_localhost,
    ]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"[ PASS ] {test.__name__}")
        except Exception as exc:
            failed += 1
            print(f"[ FAIL ] {test.__name__}: {exc}")
    raise SystemExit(1 if failed else 0)
