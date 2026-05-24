# Handoff Report: LLM Pre-Processing Stage

## Project Roadmap Context

The project roadmap, `rocm_roadmap.md`, is located at `/home/fakemitch/Desktop/rocm_roadmap.md`. It outlines the development plan for the ROCm Audio Pipeline. Always refer to this document first to understand the overall project goals and prioritized tasks.

### Where We Left Off
I was working on the "LLM Pre-Processing Stage" task. This involves using a local LLM to enrich transcript data with metadata such as speaker attribution, narration style, and emotional tone.

## Current Task
Implementing the "LLM Pre-Processing Stage" from the roadmap. This involves using a local LLM to enrich transcript data with metadata such as speaker attribution, narration style, and emotional tone.

## Work Completed
1.  **`llm_enricher.py` (Draft):** I initially drafted a separate `llm_enricher.py` script. However, upon further investigation, it was determined that modifying the existing `app/generate_script.py` script to include enrichment functionality is the preferred approach, leveraging existing LLM integration patterns.
2.  **Arguments to be added to `app/generate_script.py`:** I have attempted to add the following command-line arguments to `app/generate_script.py` to control the enrichment process:
    *   `--enrich-with-llm`
    *   `--llm-model-path`
    *   `--enrich-speaker-attribution`
    *   `--enrich-narration-style`
    *   `--enrich-emotional-tone`

## Problem Encountered
I have encountered a persistent issue using the `replace` tool to insert these new arguments into `app/generate_script.py`. Despite multiple attempts to `read_file` and precisely identify the `old_string` (including line numbers and surrounding context), the `replace` tool consistently reports "0 occurrences found." This suggests a highly sensitive mismatch, possibly due to hidden whitespace, line endings, or subtle formatting differences that the tool is unable to reconcile.

The intended insertion point is after the existing `--banned-tokens` argument (line 408) and before the `args = parser.parse_args()` call (line 410) in `app/generate_script.py`.

## Next Steps (Manual Intervention Required)

**Please manually modify `app/generate_script.py` as follows:**

1.  **Locate the `main()` function in `app/generate_script.py`.**
2.  **Find the line:** `parser.add_argument("--banned-tokens", help="Comma-separated list of tokens to ban from LLM output")` (currently around line 408).
3.  **Insert the following arguments directly after it, and before `args = parser.parse_args()` (currently around line 410):**

    ```python
        # LLM Enrichment arguments
        parser.add_argument("--enrich-with-llm", action="store_true", help="Enable LLM-based transcript enrichment.")
        parser.add_argument("--llm-model-path", help="Path to the GGUF LLM model file for transcript enrichment. Required if --enrich-with-llm is set.")
        parser.add_argument("--enrich-speaker-attribution", action="store_true", help="Instruct LLM to extract speaker attribution.")
        parser.add_argument("--enrich-narration-style", action="store_true", help="Instruct LLM to extract narration style.")
        parser.add_argument("--enrich-emotional-tone", action="store_true", help="Instruct LLM to extract emotional tone.")
    ```

4.  **Also, ensure the following validation is present after `args = parser.parse_args()`:**
    ```python
        if args.enrich_with_llm and not args.llm_model_path:
            parser.error("--llm-model-path is required when --enrich-with-llm is set.")
    ```

### Once `app/generate_script.py` is manually updated:

The **next automated task** would be to continue integrating the LLM enrichment into `alexandria_preparer_rocm_compatible.py`. This involves:

1.  **Modifying `alexandria_preparer_rocm_compatible.py`:**
    *   Adding a new 'enrich' phase to its `main` function orchestration (after ASR, before annotation).
    *   Calling `app/generate_script.py` within this 'enrich' phase using `subprocess.run`, passing the appropriate `--llm-model-path`, `--input-file` (the ASR output), `--output-file` (for enriched segments), and the `--enrich-*` flags.
    *   Updating the 'annotate' phase to consume the enriched transcript data instead of the raw ASR output.

## Testing and Logs

### How to Test
To test the changes, you would typically run the `run_random_corpus.sh` script, located in the project root: `/home/fakemitch/pinokio/api/alexandria-audiobook2.git/run_random_corpus.sh`.

After the LLM enrichment arguments are correctly integrated, you would run it with the `--enrich-with-llm` flag, and potentially other `--enrich-*` flags and `--llm-model-path`. For example:

```bash
./run_random_corpus.sh --enrich-with-llm --llm-model-path /path/to/your/gemma-model.gguf
```

### What Logs to Read

All detailed execution logs are stored in the `logs/` directory, specifically within `/home/fakemitch/pinokio/api/alexandria-audiobook2.git/logs/`. The most relevant logs for debugging the preparer script will be named `alexandria_preparer_YYYYMMDD_HHMMSS.log`.

When testing the LLM enrichment, you should also check `app/logs/llm_responses.log` (relative to `app/generate_script.py`) for the raw LLM API calls and responses. This will help verify if the prompts are correctly formatted and if the LLM's output is as expected. Also, look for `[INFO]` messages related to the "LLM Enrichment Phase" in the main preparer log. 