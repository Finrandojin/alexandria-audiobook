"""Unit tests for MiniMax provider support in Alexandria."""

import sys
import os

# Allow importing from app/ without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from generate_script import is_minimax_url, clamp_temperature_for_minimax


# ── is_minimax_url ────────────────────────────────────────────


def test_is_minimax_url_io():
    assert is_minimax_url("https://api.minimax.io/v1") is True


def test_is_minimax_url_chat():
    assert is_minimax_url("https://api.minimax.chat/v1") is True


def test_is_minimax_url_subdomain():
    assert is_minimax_url("https://something.minimax.io/v1") is True


def test_is_minimax_url_localhost():
    assert is_minimax_url("http://localhost:1234/v1") is False


def test_is_minimax_url_openai():
    assert is_minimax_url("https://api.openai.com/v1") is False


def test_is_minimax_url_ollama():
    assert is_minimax_url("http://localhost:11434/v1") is False


def test_is_minimax_url_empty():
    assert is_minimax_url("") is False


def test_is_minimax_url_none():
    assert is_minimax_url(None) is False


# ── clamp_temperature_for_minimax ────────────────────────────


def test_clamp_zero_temperature_for_minimax():
    result = clamp_temperature_for_minimax(0.0, "https://api.minimax.io/v1")
    assert result == 0.01


def test_clamp_negative_temperature_for_minimax():
    result = clamp_temperature_for_minimax(-0.5, "https://api.minimax.io/v1")
    assert result == 0.01


def test_no_clamp_positive_temperature_for_minimax():
    result = clamp_temperature_for_minimax(0.6, "https://api.minimax.io/v1")
    assert result == 0.6


def test_no_clamp_for_non_minimax():
    result = clamp_temperature_for_minimax(0.0, "http://localhost:1234/v1")
    assert result == 0.0


def test_no_clamp_for_openai():
    result = clamp_temperature_for_minimax(0.0, "https://api.openai.com/v1")
    assert result == 0.0


def test_clamp_boundary_for_minimax():
    # Exactly 0.01 should pass through unchanged
    result = clamp_temperature_for_minimax(0.01, "https://api.minimax.io/v1")
    assert result == 0.01


def test_clamp_high_temperature_for_minimax():
    # Temperature > 1.0 is user's responsibility; we only clamp zero
    result = clamp_temperature_for_minimax(1.5, "https://api.minimax.io/v1")
    assert result == 1.5


def test_clamp_with_minimax_chat_url():
    result = clamp_temperature_for_minimax(0.0, "https://api.minimax.chat/v1")
    assert result == 0.01


def test_default_temperature_passes_through_for_minimax():
    # Default temperature 0.6 should be unmodified
    result = clamp_temperature_for_minimax(0.6, "https://api.minimax.io/v1")
    assert result == 0.6


if __name__ == "__main__":
    import unittest

    # Run all tests in this file via simple assertion loop
    test_functions = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed = 0
    failed = 0
    for fn in test_functions:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
