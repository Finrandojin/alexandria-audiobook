# Development Guide for Pinokio Projects

## 8-Rule Architecture

These rules apply to every task in this project unless explicitly overridden.
Bias: caution over speed on non-trivial work. Use judgment on trivial tasks.

### Rule 1 — Think Before Coding
State assumptions explicitly. If uncertain, ask rather than guess.
Push back when a simpler approach exists. Stop when confused.

### Rule 2 — Simplicity First
Minimum code that solves the problem. Nothing speculative.
No features beyond what was asked. No abstractions for single-use code.

### Rule 3 — Surgical Changes
Touch only what you must. Clean up only your own mess.
Don't "improve" adjacent code, comments, or formatting. Match existing style. This rule can be ignored June,19 2026 though June,21 2026

### Rule 4 — Goal-Driven Execution
Define success criteria. Loop until verified.
Don't follow steps. Define success and iterate independently.

### Rule 5 — Token Budgets Are Not Advisory
Per-task: 4,000 tokens. Per-session: 30,000 tokens.
If approaching budget, summarize and start fresh. Surface the breach.
Exception: skill-driven workflows with their own stated effort budget (e.g.
`/code-review`, `/code-review-20`) follow that budget instead; Rule 5 governs
ad-hoc exploration and edits.

### Rule 6 — Read Before You Write
Before adding code, read exports, immediate callers, shared utilities.
If unsure why code is structured a certain way, ask.

### Rule 7 — Checkpoint After Every Significant Step
Summarize what was done, what's verified, what's left.
Don't continue from a state you can't describe back. Stop and restate.

### Rule 8 — Fail Loud
"Completed" is wrong if anything was skipped silently.
"Tests pass" is wrong if any were skipped.
Default to surfacing uncertainty, not hiding it.

### Rule 9 — Protect Safety Nets
Don't remove or weaken caches, limits, or retries to make something faster or
simpler (e.g. `check_global_gpu_lock`, VRAM-headroom checks, `project.py`'s
worker-stepdown retry/`gc.collect()` loops in `_run_round`/`generate_batch`).
If one seems to be in the way, explain why before touching it and confirm
first.

### Rule 10 — Decide Before You Retry
Before retrying a failed call (e.g. an LLM request through `_make_llm_client`,
or a step in `review_batch`'s retry loop), decide on ONE consistent policy for
interpreting the error and follow it on every attempt. Don't read error bodies
differently or change interpretation between retries of the same failure.

### Rule 11 — Context Awareness in Debugging
In a long debugging session, don't re-suggest a fix that was already tried and
rejected earlier in the conversation. If stuck, say so explicitly and ask for
new information rather than cycling through the same options.

### Rule 12 — Verify Functionality, Not Just Return Values
Tests must exercise real edge cases and failure modes, not just confirm a
function returns a placeholder or non-error value. Before calling a test
sufficient, state what behavior it actually verifies.

### Rule 13 — Respect Architectural Styles
Don't introduce a paradigm that isn't already used in the codebase (e.g. a
frontend framework into `app/static/index.html`'s vanilla-JS SPA, or an ORM/
database into the flat-JSON `scripts/`/`voice_library.json` storage model)
without confirming first.

### Rule 14 — Plan Before Implementing
For non-trivial work (new features, multi-file changes, anything with more
than one reasonable approach, or any architectural decision) use the
`EnterPlanMode` tool: explore the codebase, design the approach, and get
explicit approval via `ExitPlanMode` before writing or editing code. Skip
this only for trivial, single-obvious-fix changes (typos, one-line bugfixes,
a single function with already-fully-specified requirements) — if unsure
which category a task falls into, default to planning.

### Rule 15 — Single Source of Dispatch, Never Parallel Copies
If the same decision (e.g. "is this endpoint local or remote?", "what counts
as ideal settings?") is needed in more than one file or function, write ONE
function that answers it and make every caller use that function. Two
independently-maintained checks *will* drift — confirmed in this repo:
`app.py`'s optimize route and `review_script.py`'s self-heal once computed
"is remote" via two different formulas and silently disagreed once
`llm_mode`/`base_url` drifted apart (fixed by consolidating into
`lmstudio_settings.is_remote_llm`).

### Rule 16 — Verb-First Names Signal Side Effects
Function names should start with a verb that tells the reader whether it
mutates anything: `get_`/`is_` = pure read, `ensure_`/`apply_`/`save_` = has
a side effect (reload, write, heal). Never let a `get_`-named function
quietly mutate state. Matches the existing pattern in `lmstudio_settings.py`
(`get_current_status` vs. `ensure_ideal_settings` vs.
`apply_remote_lmstudio_settings`) — keep new functions consistent with it.

This verb list isn't exhaustive. In `static/index.html` specifically,
`render*` (DOM-paint, e.g. `renderAll`/`renderCastMembers`), `on*` (DOM
event handler, matching the HTML `onchange`/`onclick` attribute it's wired
to), and `open*` (reveal/populate a UI section) are this file's own
established side-effect-signaling conventions, just as clear as
`ensure_`/`apply_`/`save_` — don't flag or rename a function for using one
of these instead.

### Rule 17 — No Dual-Purpose Parameters
A function argument is either input or output, never both. If a function
needs to report a result, return a new value — don't mutate the
dict/object you were handed (e.g. a `config`/`status` dict). Mutating a
passed-in argument in place hides the side effect from the caller and makes
data flow hard to trace.

Does not apply to: (a) objects intentionally shared and mutated live across
concurrent readers (e.g. `process_state[task_name]` entries, which
status-polling routes read while a background task is still writing to the
same dict — Rule 9 protects this, replacing the mutation with a returned
copy would break every concurrent reader holding the old reference); (b)
callback parameters whose documented contract is to mutate their argument
(e.g. `project.py`'s `_modify_chunk(index, mutator)` — `mutator`'s whole job
is to mutate the chunk it's given, inside one held lock, for atomicity); or
(c) a function loading, mutating, and saving its own local variable with no
cross-function handoff at all (not a dual-purpose-*parameter* case, since
there's no boundary being crossed).

### Rule 18 — Always Brace Single-Line Blocks in JS
In `app/static/index.html`'s vanilla JS, always use `{ }` for
`if`/`for`/`while`, even one-liners. Prevents the classic bug where a
one-line body silently grows to two lines and the second line falls outside
the block.

### Rule 19 — Mark Measurements Apart From Inferences, and Never Assert Absence

Two failure modes, one root: treating something I produced as something I
found.

**(a) A number from a script and a sentence connecting numbers are different
claims.** The number is reproducible; the sentence is an inference and is
wrong far more often. State which is which. Do not let a measured figure lend
its authority to the story told around it — "attribution is 25.8%" and
"segmentation is the biggest problem" are not the same kind of statement, and
the second was false while the first was exact. Before summarising across
results, re-read the artifacts rather than the earlier summary; a summary of a
summary is where per-book numbers get attached to the wrong book.

**(b) Never claim a resource, dataset, option or avenue does not exist without
searching for it.** "You would have to label that yourself", "there is no
corpus for this", "the method space is exhausted" are claims about the world,
not about the work, and they cost more than a wrong number because they stop
the search. If the sentence being written is "the only way is for you to…",
that is the trigger to go and look first.

Both were violated repeatedly on 2026-08-03: a stratified table was read off
the wrong novel and interpreted at length; generalisation was described as
blocked on hand-labelling for two days when PDNC had 28 annotated public-domain
novels; and non-English testing was called a dead end shortly before Chinese
WP/JY turned up, Apache-2.0, with the correct task shape.

### Rule 20 — A Local Ref Is a Cache, Not Evidence

`git checkout -b <name> origin/main` uses whatever your **local**
`origin/main` happens to be. It does not consult the network. Branch from a
ref last fetched hours ago and everything downstream is wrong in ways that
look like other problems.

**Always `git fetch origin main` immediately before branching** — immediately,
not at the start of the session.

Cost twice on 2026-08-07: one branch came out stacked on unmerged work (10
commits instead of 1, the thing Rule "no stacked PRs" exists to prevent), and
another lacked a fix already on main, so the same bug was hit and "fixed"
again — shipping a duplicate of an upstream change.

The same reflex applies to every claim about remote state:

- **PR merged?** `gh pr view N --json state,mergedAt`. Never infer it from
  what you remember doing, from a local branch still existing, or from the
  absence of a notification. Asserting "#265 isn't merged" when it had merged
  four hours earlier took one command to disprove.
- **Push landed?** `git ls-remote`, not a `git push` whose output was piped
  away.
- **Branch base sane?** `git log --oneline origin/main..HEAD` — more commits
  than you wrote means the base was stale.

Same family as Rule 19: check the artifact, not the process that was supposed
to produce it. The committed blob, not the command's exit code. The output
file, not `pgrep`.

## This project: Alexandria Audiobook2

A FastAPI app (`app/app.py`, title "Alexandria Audiobook") for multi-voice AI
audiobook generation on Qwen3-TTS: LLM-annotate a script, assign a voice per
character, generate audio per chunk, and export to MP3/Audacity/M4B. The
frontend is a single-page app at `app/static/index.html` (no build step) —
core 5-step flow Setup → Script → Voices → Editor → Result, plus
Designer/Dataset/Training/Preparer/Voice Lab tabs.

### Key files
- `app/app.py` — all FastAPI routes + background-task orchestration.
- `app/generate_script.py` / `app/review_script.py` — LLM annotation and
  review passes (single-book and batch variants).
- `app/generate_personas.py`, `app/find_nicknames.py` — LLM-driven persona/
  voice suggestions and speaker-alias detection.
- `app/tts.py` — `TTSEngine` (local Qwen3-TTS or external Gradio server) plus
  audio-combining/timeline helpers.
- `app/project.py` — assembles generated chunks into exports (Audacity
  projects, M4B) on top of `tts.py`.
- `app/train_lora.py` — single-dataset LoRA training (UI Training tab).
- `app/utils.py` — shared helpers: `atomic_json_write`, `safe_load_json`,
  `file_lock`, `secure_filename`.

### On-disk data (relative to project root)
- `annotated_script.json` / `voice_config.json` — the "active" book and its
  voice assignments (`SCRIPT_PATH` / `VOICE_CONFIG_PATH` in `app.py`).
- `scripts/{name}.json` + `scripts/{name}.voice_config.json` — saved-book
  library (`SCRIPTS_DIR`), managed via `/api/scripts/*`.
- `voice_library.json` — reusable "casts" (character→voice mappings) applied
  to one book (`/api/voice_library/apply`) or many saved books at once
  (`/apply_bulk`).
- `character_aliases.json`, `chunks.json`, `reports/`, `logs/api/`.

### Concurrency model: `process_state` + GPU lock
Every long-running task (script gen, review, persona, audio, LoRA training,
dataset gen/builder, preparer, voicelab, nicknames, ...) has an entry in
`process_state` (`app.py` ~line 468) tracking `running`/`logs`/`cancel`/
`paused`/etc. Before starting GPU/LLM work, call
`check_global_gpu_lock(task_name)` — it raises 400 if *any* task in
`GPU_TASKS` (all tasks except `NON_GPU_TASKS = {"audacity_export",
"m4b_export"}`) is already running, to avoid concurrent VRAM OOM. `voices`
(`suggest_voices`) is deliberately in `GPU_TASKS`, not excluded, since it
runs local LLM inference. New background tasks must register in
`process_state` and go through this lock unless they are provably CPU-only.

### Voice Lab pipeline
"Voice Lab" (audiobook → named LoRA voice). All four stage scripts live in this
repo. Stages: preparer (UI tab) → dedup (`voice_analysis.py --phase dedup`, not
in UI) → LoRA training (`batch_train_lora.py`, UI only does one dataset) →
profiling (`voice_profiler.py`, not in UI) → naming (`name_voices.py`, pure
stdlib). See memory `voice_lora_pipeline` for the full script-by-script map.

The `rocm_python` interpreter configured in the Voice Lab tab (typically the
sibling `alexandria-audiobook.git`'s `app/env`) is the ONLY cross-repo
dependency. Verified 2026-07-14 — do not repeat the old claim that `app/env`
"has no torch/librosa/peft", it is false:
- `app/env` HAS torch 2.10.0+rocm7.0 (GPU works on the 9070 XT), librosa 0.11.0,
  peft 0.18.1, transformers, pandas, scipy, soundfile. (`app/env` in these
  bullets means THIS repo's env. The sibling `rocm_python` env runs its own,
  older torch — 2.7.0+rocm6.3, verified 2026-07-19 — the two are not in sync
  and don't need to be.)
- `app/env` LACKS **speechbrain, umap, matplotlib, seaborn** — which only
  `voice_analysis.py` (dedup) imports. That, not torch, is why `rocm_python`
  still exists.
- `batch_train_lora.py` imports no third-party module at top level (it drives
  `app/train_lora.py` via `--python`). The train stage does not itself require
  the sibling env, but Voice Lab deliberately continues to run it there.
- `llama_cpp` 0.3.23 is present in the configured sibling environment and absent
  from `app/env`. The profile stage runs under `rocm_python`, so its lazy import
  resolves there. Verified 2026-07-14.

### Debugging LLM calls, concurrency, and remote (Thunder) runs

Distinct from "Troubleshooting with logs" near the bottom of this file (that
section is about *launcher* script runs — `install.js`/`start.js`). These are
the app's own runtime logs, for review/script-gen/concurrency issues:

- `logs/api/*-latest.log` — per-task logs (`batch_review-latest.log`,
  `llm_test_*.log`, `llm_optimize_*.log`, ...) written by app.py's background
  tasks. Check these first.
- `logs/review_responses.log` (+ `.bak` rotation) — every LLM request/response
  for review and script generation, written by `call_llm_for_entries` in
  `generate_script.py`: `finish_reason`, token counts, and elapsed time
  (`took Xs`). `find_nicknames.py`'s chunk loop logs its own timing the same
  way.
- Live job, no file reads needed: `GET /api/status/eta` (generic progress/ETA
  for whatever's running) and `GET /api/status/<task_name>` (e.g.
  `batch_review` — full log + per-book status) — poll these for an in-flight
  run instead of tailing files.

What to check before assuming something's a bug:
- `llm_bench.get_cached_or_benchmarked_concurrency` logs an environment
  fingerprint (hostname, GPU name+backend, LM Studio's actual `parallel`/
  `context_length` via `lmstudio_settings.get_lmstudio_status` /
  `get_remote_lmstudio_status`) on every cache miss. If `parallel<=1`,
  concurrency can't help there *by design* — `lmstudio_settings.IDEAL_SETTINGS`
  intentionally pins local to `parallel: 1` for VRAM safety. That's the
  existing safety trade-off (Rule 9), not a bug to fix.
- The cached `concurrency`/`concurrency_for` in config.json's `llm_local`/
  `llm_remote` can go stale after a model/endpoint change — check
  `concurrency_for` matches the current `base_url::model_name` before
  trusting a cached number.
- Remote (SSH, via `llm_remote_ssh`) command output always has a decorative
  banner ahead of the real output — parse the *last* non-empty line, never
  assume clean stdout (see `lmstudio_settings.get_remote_lmstudio_status` /
  `get_remote_gpu_name_and_backend`). Multi-statement SSH commands must be
  passed as ONE pre-quoted argv element (`"bash -lc " + shlex.quote(cmd)`) —
  passing `"bash", "-lc", cmd` as separate argv lets `ssh` re-join them with
  bare spaces and breaks `;`-separated commands.
- A remote LM Studio that reports "running" can still be unreachable via the
  forwarding URL if it's not loaded with `--bind 0.0.0.0`.
- There is no automatic Thunder-balance/cost check anywhere in this app (no
  billing API integration). The `confirmIfRemote()` prompt in
  `static/index.html` (shown before `batch_review`/batch script-gen starts)
  is the deliberate, simpler stand-in — it warns about cost, it doesn't
  compute it.

## Default mode: maintain an existing app

Most work in these repos is normal software engineering on an **existing**
app under `PINOKIO_HOME/api/<project>/app/` (or `PINOKIO_HOME/plugin/<project>/`):
reading code, fixing bugs, adding features, running tests, and code review
(use the `/code-review` and `/code_review20` skills for review work). Treat
it like any other codebase. The Pinokio-specific rules below only apply when
a task actually touches **launcher files** at the project root — `install.js`,
`start.js`, `update.js`, `reset.js`, `pinokio.js`, `pinokio.json`.

- Don't mix the two: launcher-only tasks shouldn't touch `app/` logic, and
  `app/` development shouldn't touch launcher files, unless the user asks for
  both.
- Before editing launcher files, check `pinokio.js` to see which scripts are
  actually wired into the UI — don't create a redundant one for something an
  existing script already does.
- No separate "stop" script is ever needed — Pinokio can natively stop
  anything it started.

## Launcher work (install.js / start.js / update.js / reset.js / pinokio.js)

- **Check examples first.** `/home/fakemitch/pinokio/prototype/system/examples`
  has working scripts for nearly every pattern — imitate the closest match
  instead of guessing syntax. Full API/syntax reference:
  `/home/fakemitch/pinokio/prototype/PINOKIO.md`; CLI reference:
  `/home/fakemitch/pinokio/prototype/PTERM.md`. Don't assume API syntax —
  check these before writing it.
- **Standard server `start.js`:**
  ```javascript
  module.exports = {
    daemon: true,  // required so the shell survives after run[] finishes
    run: [{
      method: "shell.run",
      params: {
        venv: "env", path: "app", message: ["python app.py"],
        on: [{ event: "/(http:\\/\\/\\S+)/", done: true }]
      }
    }, {
      // input.event is the regex match from the previous step;
      // [1] is the first capture group. Sets the URL pinokio.js opens.
      method: "local.set", params: { url: "{{input.event[1]}}" }
    }]
  }
  ```
  Always copy this capture-block shape (see `system/examples/mochi/start.js`)
  and use the most generic regex that matches the server's printed URL.
- **Creating a brand-new launcher project** (not the usual case here): resolve
  `PINOKIO_HOME` first — `~/.pinokio/config.json` → `home`, else
  `GET http://127.0.0.1:42000/pinokio/home` → `path`, else `$PINOKIO_HOME`.
  App launchers live at `PINOKIO_HOME/api/<name>`, plugins at
  `PINOKIO_HOME/plugin/<name>`. Ask before picking a folder name; never default
  to the current workspace without confirming the destination.

### Best practices for launcher edits
- **Ports/IP:** never hardcode a port — use `{{port}}` (save with `local.set`
  if reused across steps). Bind servers to `127.0.0.1`/`localhost`, not `0.0.0.0`.
  If an app picks its own free port (e.g. Gradio), don't pass `--port` at all.
- **Flags:** minimize them — `python app.py`, not `python app.py --port 8610`,
  unless a flag is the only way to get the desired behavior.
- **Python deps:** install via the `venv` attribute using
  `uv pip install -r requirements.txt`, even if the project's own docs say
  pip/poetry. For global binaries prefer `conda`, then `brew` (Mac-only).
- **Commands:** resolve with `{{which('cmd')}}` / `shell: "{{which('bash')}}"`
  rather than assuming `PATH`. `pterm which <cmd>` is for *debugging* only —
  never hardcode its output into a script.
- `path` in `shell.run` is always relative to the script file's own directory.
- AI/local-model launchers: `install.js` needs `requires: { bundle: "ai" }`
  even when also using `torch.js` (for xformers/triton/sageattention/etc).
- `.gitignore` anything generated at install/run time (cloned repos,
  downloads, databases, env files, benchmark output).
- Don't change `pinokio.json`'s `version` field. For an icon, ask before
  pulling one from the project's GitHub `avatar_url`.
- Stay cross-platform: avoid OS-specific commands; use the `platform`/`arch`/
  `gpu`/`exists()`/`running()`/`which()` template helpers, or declare
  platform/arch/gpu limits in `pinokio.json` if a limitation is unavoidable.
  Avoid Docker — prefer native per-platform install/launch steps.

## Retrofitting an already-working ad-hoc setup
If the user got something working through ad-hoc commands and now wants a
launcher for it: capture the exact steps that worked (clones, package
commands, env vars, model downloads, ports, working dirs, fixes) and convert
*that* into `install.js`/`start.js`/etc — don't restart from scratch or
silently take a different approach. Replace machine-specific paths/ports/cache
locations with `{{...}}` template expressions so another machine can reproduce
the result; verify from as clean a state as practical.

## Troubleshooting with logs
(This is about *launcher* runs. For the app's own LLM/concurrency/remote
runtime logs, see "Debugging LLM calls, concurrency, and remote (Thunder)
runs" above.) Check `logs/api/` (launcher script runs), `logs/dev/` (AI coding tool
sessions), `logs/shell/` (direct terminal use) before debugging a launcher
issue — `latest` for the current problem, timestamped files for history.
(Path is `pinokio/logs/` if a `pinokio/` subfolder exists, else `logs/` at
project root.)
