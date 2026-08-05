import base64
import importlib.util
import os
from pathlib import Path
import re
import unittest
from unittest.mock import patch

import config_settings
import utils
from fastapi.testclient import TestClient

# Frontend JS was split out of index.html's one inline <script> into a few
# static/js/app-*.js files (kept in their original relative order) to reduce
# merge-conflict surface on high-churn areas (training/review, voice lab).
# This helper reproduces the pre-split "one big string" view so every existing
# string-presence assertion below keeps working unchanged against the split
# files, without each test needing to know about the new layout.
_STATIC_DIR = Path(__file__).resolve().parent / "static"
_FRONTEND_JS_FILES = (
    "app-core.js", "app-scripts.js", "app-training.js",
    "app-workbench.js", "app-voicelab.js", "app-reports.js",
)


def _read_frontend_source():
    html = (_STATIC_DIR / "index.html").read_text(encoding="utf-8")
    js = "".join((_STATIC_DIR / "js" / name).read_text(encoding="utf-8")
                 for name in _FRONTEND_JS_FILES)
    return html + js


class FrontendJsSplitTests(unittest.TestCase):
    def test_split_js_files_exist_are_referenced_in_order_and_non_empty(self):
        # The split files must exist, be non-empty, and be referenced by
        # index.html as plain <script src> tags (no defer/async/module — those
        # would change execution-order semantics relative to the original
        # single inline block) in the exact original relative order.
        html = (_STATIC_DIR / "index.html").read_text(encoding="utf-8")
        positions = []
        for name in _FRONTEND_JS_FILES:
            path = _STATIC_DIR / "js" / name
            self.assertTrue(path.is_file(), f"missing {path}")
            self.assertGreater(path.stat().st_size, 0, f"{path} is empty")
            tag = f'src="/static/js/{name}?v=__APP_BUILD__"'
            self.assertIn(tag, html)
            self.assertNotIn(f'defer src="/static/js/{name}',
                             html.replace(tag, tag))  # no defer on this tag
            positions.append(html.index(tag))
        self.assertEqual(positions, sorted(positions),
                         "script tags must appear in the original file order")
        for name in _FRONTEND_JS_FILES:
            self.assertNotIn(f'<script src="/static/js/{name}" defer', html)
            self.assertNotIn(f'<script src="/static/js/{name}" async', html)
            self.assertNotIn(f'type="module" src="/static/js/{name}"', html)

    def test_cross_file_forward_reference_lands_in_a_later_script_tag(self):
        # reattachRunningPollers (app-workbench.js, called at page-init time,
        # behind an awaited network fetch) calls pollVoicelab, which is defined
        # in app-voicelab.js — a chunk loaded AFTER app-workbench.js. This is
        # safe only because the call happens after the await resolves, by which
        # point every <script> tag has already finished loading (see plan for
        # the full reasoning) — and only because app-voicelab.js is still
        # ordered after app-workbench.js. This test pins that ordering so a
        # future reshuffle can't silently break the forward reference.
        workbench = (_STATIC_DIR / "js" / "app-workbench.js").read_text(encoding="utf-8")
        voicelab = (_STATIC_DIR / "js" / "app-voicelab.js").read_text(encoding="utf-8")
        self.assertIn("pollVoicelab();", workbench)
        self.assertIn("function pollVoicelab()", voicelab)
        html = (_STATIC_DIR / "index.html").read_text(encoding="utf-8")
        self.assertLess(html.index("app-workbench.js"), html.index("app-voicelab.js"))


class FrontendTests(unittest.TestCase):
    def test_saved_script_audit_surfaces_nonprose_validation_state(self):
        frontend = _read_frontend_source()
        for required in ("saved-script-preflight", "auditSavedScript",
                         "nonprose_speech_risk", "details.validation",
                         "/preflight`, {}"):
            self.assertIn(required, frontend)

    def test_voicelab_requires_visible_preflight_before_start(self):
        frontend = _read_frontend_source()
        for required in ("vl-preflight", "/api/voicelab/preflight",
                         "renderVoicelabPreflight", "preflight.preflight_id"):
            self.assertIn(required, frontend)

    def test_voicelab_health_dashboard_is_rendered_and_wired(self):
        frontend = _read_frontend_source()
        for required in ("vl-health", "vl-health-body", "renderVoicelabHealth",
                         "refreshVoicelabHealth", "/api/voicelab/health",
                         "openLoraModelsTab", "pending_recovery"):
            self.assertIn(required, frontend)

    def test_voicelab_diagnostics_actions_are_present_and_sanitized(self):
        frontend = _read_frontend_source()
        for required in ("copyVoicelabDiagnostics", "downloadVoicelabDiagnostics",
                         "/api/voicelab/diagnostics", "Copy diagnostics",
                         "Download diagnostics"):
            self.assertIn(required, frontend)

    def test_review_summary_and_health_eta_are_rendered(self):
        frontend = _read_frontend_source()
        for required in ("renderReviewSummary", "review_summary", "human review",
                         "prefer candidate", "active.eta_seconds", "left"):
            self.assertIn(required, frontend)

    def test_stale_build_banner_is_wired_without_a_new_timer(self):
        frontend = _read_frontend_source()
        for required in ('name="app-build"', "__APP_BUILD__", "stale-build-banner",
                         "PAGE_BUILD", "checkStaleBuild", "location.reload()"):
            self.assertIn(required, frontend)
        # The check must piggyback on the existing stats poll, adding no timer:
        # checkStaleBuild is called from updateSystemStats, and there is no new
        # setInterval referencing it.
        self.assertIn("checkStaleBuild(runtime.short_revision)", frontend)
        self.assertNotIn("setInterval(checkStaleBuild", frontend)

    def test_runtime_build_and_package_versions_are_visible_in_navbar(self):
        frontend = _read_frontend_source()
        for required in ("sys-build", "sys-build-val", "stats.runtime",
                         "runtime.short_revision", "runtime.packages",
                         "Revision unavailable"):
            self.assertIn(required, frontend)

    def test_lora_candidate_comparison_is_advisory_and_separate_from_promotion(self):
        frontend = _read_frontend_source()
        for required in (
                "lora-comparison-panel", "openLoraCandidateComparison",
                "/comparison", "Advisory only", "Production", "Candidate",
                "promoteLoraCandidate"):
            self.assertIn(required, frontend)

    def test_blind_human_review_is_present_and_kept_separate_from_automated(self):
        frontend = _read_frontend_source()
        for required in ("openLoraBlindReview", "submitLoraBlindReview",
                         "renderBlindReviewResult", "openLoraReviewHistory",
                         "clearLoraReviewHistory", "/review/session",
                         "/reviews", "Identities are hidden until you submit",
                         "Automated recommendation (separate)",
                         # Submit disables itself so a double-click can't fire twice.
                         'id="btn-blind-submit"', "submitBtn.disabled = true"):
            self.assertIn(required, frontend)

    def test_lora_candidate_lifecycle_summary_is_rendered(self):
        frontend = _read_frontend_source()
        for required in ("renderCandidateSummary", "candidate_summary",
                         "awaiting evaluation", "candidate promoted",
                         "duplicate skipped", "production unchanged"):
            self.assertIn(required, frontend)

    def test_script_ui_reuses_uploads_and_sends_collision_policy(self):
        frontend = _read_frontend_source()
        for required in (
                "existing-upload-select", "script-existing-uploads", "/api/uploads/select",
                "script-collision-policy", "collision_policy: collisionPolicy",
                "/api/generate_script/batch/preflight", "Largest predicted request",
                "scriptBatchSelectAll(true)", "scriptBatchSelectAll(false)",
                "scriptBatchSort('num-asc')", "scriptBatchSort('reverse')",
                "script-batch-selected-count", "Failed books keep validated checkpoints",
                "completed, ${failed} failed, ${cancelled} cancelled",
                # Checkboxes instead of a native <select multiple> - selecting a
                # subset normally requires ctrl/shift-click, unusable over a
                # remote-desktop session with no modifier keys.
                "script-batch-upload-check", "renderScriptBatchUploads",
                # Toggle for stripping known non-narrative compiler front matter
                # (translator's notes / tables of contents) before generation.
                "script-strip-front-matter", "_isStripFrontMatterChecked",
                "strip_front_matter: _isStripFrontMatterChecked()"):
            self.assertIn(required, frontend)

        # Both the single and batch generate calls must send the toggle -
        # not just one of them (a bare count check catches a copy-paste that
        # only wires up one call site).
        self.assertEqual(2, frontend.count("strip_front_matter: _isStripFrontMatterChecked()"))

        sort_start = frontend.index("window.scriptBatchSort =")
        sort_end = frontend.index("window.cancelBatchScript", sort_start)
        sort_function = frontend[sort_start:sort_end]
        self.assertIn("_sortScriptList(sortable, mode)", sort_function)
        self.assertIn("renderScriptBatchUploads();", sort_function)
        self.assertIn("replace(/_\\d+$/, '')", sort_function)
        self.assertIn("onScriptBatchFilesChange();", sort_function)

    def test_launcher_contracts_cover_dynamic_ports_failures_and_rocm_constraints(self):
        root = Path(__file__).resolve().parent.parent
        start = (root / "start.js").read_text(encoding="utf-8")
        start_llm = (root / "start_llm.js").read_text(encoding="utf-8")
        install = (root / "install.js").read_text(encoding="utf-8")
        self.assertIn('port: "{{port}}"', start)
        self.assertIn('ALEXANDRIA_PORT: "{{local.port}}"', start)
        self.assertIn('PYTHONUNBUFFERED: "1"', start)
        self.assertIn('url: "{{input.event[1]}}"', start)
        self.assertIn("ModuleNotFoundError", start)
        self.assertIn("Address already in use", start)
        self.assertIn("Application startup failed", start)
        self.assertIn("break: true", start)
        self.assertIn('method: "script.return"', start_llm)
        self.assertIn("triton-rocm", install)
        torch_script = (root / "torch.js").read_text(encoding="utf-8")
        for rocm_pin in ("torch==2.10.0", "torchaudio==2.10.0",
                         "triton-rocm==3.6.0", "/whl/rocm7.0"):
            self.assertIn(rocm_pin, torch_script)

        readme = (root / "README.md").read_text(encoding="utf-8")
        for diagnostic in ("logs/api/start.js/latest", "logs/sessions/",
                           "Get Help", "navbar build label"):
            self.assertIn(diagnostic, readme)

    def test_readme_api_examples_do_not_reference_removed_routes(self):
        readme = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
        for removed in ("/api/parse_voices", "/api/lora/generate_dataset",
                        "/api/voice_design/delete/", "/api/status/script_generation"):
            self.assertNotIn(removed, readme)

    def test_docker_image_includes_root_runtime_dependencies(self):
        root = Path(__file__).resolve().parent.parent
        dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
        for required in ("gpu_stats.py", "persona_prompts.txt", "alexandria_alignment.py",
                         "alexandria_preparer_rocm_compatible.py",
                         "llm_enricher.py", "audit_voice_datasets.py", "voice_analysis.py", "batch_train_lora.py",
                         "evaluate_lora.py", "voice_profiler.py", "name_voices.py"):
            self.assertIn(required, dockerfile)

    def test_docker_mounts_single_persistent_runtime_root(self):
        root = Path(__file__).resolve().parent.parent
        dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
        compose = (root / "docker-compose.yml").read_text(encoding="utf-8")
        self.assertIn("ALEXANDRIA_DATA_DIR=/alexandria/runtime", dockerfile)
        self.assertIn("./data/runtime:/alexandria/runtime", compose)
        self.assertIn('127.0.0.1:4200:4200', compose)
        self.assertNotIn('- "4200:4200"', compose)

    def test_basic_auth_middleware_guards_api_and_mounted_files(self):
        app_path = Path(__file__).resolve().parent / "app.py"
        spec = importlib.util.spec_from_file_location("authenticated_test_app", app_path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(os.environ, {
            "ALEXANDRIA_AUTH_USERNAME": "reviewer",
            "ALEXANDRIA_AUTH_PASSWORD": "secret",
        }):
            spec.loader.exec_module(module)

        client = TestClient(module.app)
        credentials = base64.b64encode(b"reviewer:secret").decode()
        headers = {"Authorization": f"Basic {credentials}"}
        for path in ("/api/status/eta", "/static/index.html",
                     "/voicelines/missing.wav", "/lora_models/missing.wav"):
            with self.subTest(path=path):
                self.assertEqual(401, client.get(path).status_code)
                self.assertNotEqual(401, client.get(path, headers=headers).status_code)

    def test_frontend_wires_preparer_duration_and_lora_cancel(self):
        html = _read_frontend_source()
        self.assertIn("min_chunk_duration: getNumFieldValue('prep-min-chunk-duration', 2)", html)
        self.assertIn("id=\"btn-lora-cancel\"", html)
        self.assertIn("/api/lora/train/cancel", html)
        self.assertIn("promoteLoraCandidate", html)
        self.assertIn("/promote`, {}", html)
        self.assertIn("rollbackLoraPromotion", html)
        self.assertIn("/rollback-promotion`, {}", html)
        self.assertIn("recoverLoraCheckpointSwap", html)
        self.assertIn("/recover-checkpoint-swap`, {}", html)
        self.assertIn("deleteLoraRollbackBackup", html)
        self.assertIn("/api/lora/backups", html)
        self.assertIn("/rollback-backup`, { method: 'DELETE' }", html)
        self.assertNotIn("id=\"prep-skip-annotation\"", html)

    def test_frontend_renders_config_warnings_as_text_and_refreshes_after_save(self):
        html = _read_frontend_source()
        self.assertIn('id="config-warning-banner"', html)
        self.assertIn('id="config-warning-msg"', html)
        start = html.index("function renderConfigWarnings(config)")
        end = html.index("function populateLlmInputs", start)
        renderer = html[start:end]
        self.assertIn("message.textContent", renderer)
        self.assertNotIn("message.innerHTML", renderer)
        self.assertIn("renderConfigWarnings(config);", html)
        self.assertIn("renderConfigWarnings(savedConfig);", html)

    def test_frontend_config_controls_match_backend_schema(self):
        html = _read_frontend_source()
        input_tags = {
            match.group(1): match.group(0)
            for match in re.finditer(r'<input\b[^>]*\bid="([^"]+)"[^>]*>', html)
        }
        mappings = {
            "parallel-workers": (config_settings.TTSConfig, "parallel_workers"),
            "tts-max-new-tokens": (config_settings.TTSConfig, "max_new_tokens"),
            "sub-batch-min-size": (config_settings.TTSConfig, "sub_batch_min_size"),
            "sub-batch-ratio": (config_settings.TTSConfig, "sub_batch_ratio"),
            "sub-batch-max-items": (config_settings.TTSConfig, "sub_batch_max_items"),
            "pause-between-speakers": (config_settings.TTSConfig, "pause_between_speakers_ms"),
            "pause-same-speaker": (config_settings.TTSConfig, "pause_same_speaker_ms"),
            "chunk-size": (config_settings.GenerationConfig, "chunk_size"),
            "max-tokens": (config_settings.GenerationConfig, "max_tokens"),
            "temperature": (config_settings.GenerationConfig, "temperature"),
            "top-p": (config_settings.GenerationConfig, "top_p"),
            "top-k": (config_settings.GenerationConfig, "top_k"),
            "min-p": (config_settings.GenerationConfig, "min_p"),
            "presence-penalty": (config_settings.GenerationConfig, "presence_penalty"),
        }
        for control_id, (model, field_name) in mappings.items():
            with self.subTest(control_id=control_id):
                tag = input_tags[control_id]
                attrs = dict(re.findall(r'([\w-]+)="([^"]*)"', tag))
                schema = model.model_json_schema()["properties"][field_name]
                self.assertEqual(float(schema["default"]), float(attrs["value"]))
                for schema_key, attr_name in (("minimum", "min"), ("maximum", "max")):
                    if schema_key in schema:
                        self.assertEqual(float(schema[schema_key]), float(attrs[attr_name]))
                    else:
                        self.assertNotIn(attr_name, attrs)
                step = float(attrs.get("step", "1"))
                self.assertGreater(step, 0)
                if schema["type"] == "integer":
                    self.assertTrue(step.is_integer())

        selects = {
            match.group(1): set(re.findall(r'<option\s+value="([^"]+)"', match.group(2)))
            for match in re.finditer(
                r'<select\b[^>]*\bid="([^"]+)"[^>]*>(.*?)</select>', html, re.DOTALL
            )
        }
        app_mode = config_settings.AppConfig.model_json_schema()["properties"]["llm_mode"]
        tts_mode = config_settings.TTSConfig.model_json_schema()["properties"]["mode"]
        self.assertEqual(set(app_mode["enum"]), selects["llm-mode"])
        self.assertEqual(set(tts_mode["enum"]), selects["tts-mode"])
        self.assertIn(app_mode["default"], selects["llm-mode"])
        self.assertIn(tts_mode["default"], selects["tts-mode"])

    def test_check_basic_auth_accepts_and_rejects_credentials(self):
        make = lambda raw: "Basic " + base64.b64encode(raw.encode()).decode()
        # Correct credentials pass.
        self.assertTrue(utils.check_basic_auth(
            make("alexandria:secret"), "alexandria", "secret"))
        # Wrong password and wrong username both fail.
        self.assertFalse(utils.check_basic_auth(
            make("alexandria:wrong"), "alexandria", "secret"))
        self.assertFalse(utils.check_basic_auth(
            make("intruder:secret"), "alexandria", "secret"))
        # Malformed / missing inputs fail closed rather than raising.
        self.assertFalse(utils.check_basic_auth(
            make("nocolonhere"), "alexandria", "secret"))
        self.assertFalse(utils.check_basic_auth(
            "Basic !!!not-base64!!!", "alexandria", "secret"))
        self.assertFalse(utils.check_basic_auth(
            "Bearer " + base64.b64encode(b"alexandria:secret").decode(),
            "alexandria", "secret"))
        self.assertFalse(utils.check_basic_auth("", "alexandria", "secret"))
