import os
import json
import time
import tempfile


def uses_json_mode(model_name):
    """Return True when the LLM provider needs response_format=json_object.

    DeepSeek (and Zhipu/GLM) APIs require the explicit
    response_format={"type": "json_object"} parameter to reliably produce
    valid JSON — prompt-only hints are not enough and the model may emit
    prose, markdown fences, or whitespace instead. Local servers (Ollama,
    LM Studio, etc.) may reject unknown response_format values, so this is
    gated on the model name rather than applied globally.
    """
    name = (model_name or "").lower()
    return "deepseek" in name or "glm" in name


def atomic_json_write(data, target_path, max_retries=5):
    """Atomically write JSON data using a temp file and os.replace.

    Includes retry logic with exponential backoff for Windows file locking
    (Access is denied / file in use errors).
    """
    directory = os.path.dirname(target_path) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        for attempt in range(max_retries):
            try:
                os.replace(tmp_path, target_path)
                return
            except OSError as e:
                if attempt < max_retries - 1 and (
                    e.errno == 5
                    or "Access is denied" in str(e)
                    or "being used by another process" in str(e)
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
