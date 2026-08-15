import tempfile
import asyncio
import unittest
from pathlib import Path
from unittest.mock import patch

from routers import script
import three_pass_generate as tp


class BatchScriptConcurrencyTests(unittest.TestCase):
    def _preflight(self, context, parallel, worst):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "book.txt")
            path.write_text("Book text.", encoding="utf-8")
            jobs = [{"filename": "book-1.txt", "input_path": str(path)},
                    {"filename": "book-2.txt", "input_path": str(path)}]
            with patch.object(script, "load_app_config", return_value={
                    "llm": {"model_name": "model"}, "generation": {}, "prompts": {}}), \
                 patch.object(script, "get_planned_ideal_settings", return_value={
                    "context_length": context, "parallel": parallel}), \
                 patch.object(script, "build_three_pass_request_preflight",
                    return_value={"chunk_count": 1, "worst_predicted_tokens": worst,
                                  "p95_predicted_tokens": worst,
                                  "average_predicted_tokens": float(worst)}):
                return script.build_batch_script_preflight(jobs)

    def test_two_workers_when_every_book_fits(self):
        report = self._preflight(32768, 2, 9000)
        self.assertEqual(2, report["workers"])
        self.assertEqual(16384, report["per_slot_context"])
        self.assertIsNone(report["fallback_reason"])
        self.assertTrue(all(book["fits_selected_slot"] for book in report["books"]))

    def test_serializes_when_per_slot_context_is_too_small(self):
        report = self._preflight(16384, 2, 9000)
        self.assertEqual(1, report["workers"])
        self.assertEqual(16384, report["per_slot_context"])
        self.assertIn("Reduced concurrency", report["fallback_reason"])

    def test_worker_compatibility_helper_uses_shared_report(self):
        expected = {"workers": 2, "worst_request_tokens": 9000, "context_length": 32768}
        with patch.object(script, "build_batch_script_preflight", return_value=expected):
            self.assertEqual((2, 9000, 32768), script._get_batch_script_workers([]))

    def test_status_omits_all_live_process_objects(self):
        state = script.process_state["batch_script"]
        original = dict(state)
        try:
            state.update({"running": True, "process": object(),
                          "processes": [object()], "tasks": []})
            public = asyncio.run(script.get_status("batch_script"))
        finally:
            state.clear()
            state.update(original)

        self.assertNotIn("process", public)
        self.assertNotIn("processes", public)

    def test_preflight_uses_planned_runtime_profile(self):
        report = self._preflight(32768, 2, 9441)
        self.assertEqual(32768, report["context_length"])
        self.assertEqual(16384, report["per_slot_context"])

    def test_batch_source_rejects_unattested_narrator_before_gpu_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "book.txt")
            path.write_text("Alexis entered. Alexis spoke.", encoding="utf-8")
            job = {"filename": "book.txt", "input_path": str(path),
                   "first_person_narrator": "ALEXIS"}

            with self.assertRaisesRegex(ValueError, "at least three times"):
                script._read_and_validate_batch_script_source(job)

    def test_batch_source_accepts_attested_narrator(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "book.txt")
            path.write_text(
                "Alexis entered. Alexis spoke. Alexis left.", encoding="utf-8")
            job = {"filename": "book.txt", "input_path": str(path),
                   "first_person_narrator": "ALEXIS"}

            text, normalization_count = (
                script._read_and_validate_batch_script_source(job))

        self.assertIn("Alexis left", text)
        self.assertEqual([], normalization_count)

    def test_preflight_uses_three_pass_model_profile_settings(self):
        observed = {}

        def estimate(text, settings, context, parallel, context_windows=None):
            observed.update(settings)
            return {"chunk_count": 1, "worst_predicted_tokens": 1000,
                    "p95_predicted_tokens": 1000,
                    "average_predicted_tokens": 1000.0}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "book.txt")
            path.write_text("Book text.", encoding="utf-8")
            jobs = [{"filename": "book.txt", "input_path": str(path)}]
            config = {
                "llm": {"model_name": "model"},
                "generation": {
                    "three_pass_chunk_size": 4000,
                    "three_pass_presegment_quotes": True,
                    "three_pass_model_profiles": {
                        "model": {"chunk_size": 2200,
                                  "presegment_quotes": False},
                    },
                },
            }
            with patch.object(script, "load_app_config", return_value=config), \
                 patch.object(script, "get_planned_ideal_settings", return_value={
                     "context_length": 32768, "parallel": 1}), \
                 patch.object(script, "build_three_pass_request_preflight",
                              side_effect=estimate):
                script.build_batch_script_preflight(jobs)

        self.assertEqual(2200, observed["chunk_size"])
        self.assertFalse(observed["presegment_quotes"])

    def test_three_pass_estimator_covers_each_llm_stage(self):
        settings = {
            "chunk_size": 6000, "max_tokens": 4096,
            "segment_output_ratio": 3.0, "presegment_quotes": True,
        }
        report = tp.build_three_pass_request_preflight(
            'Narration. "Spoken words." More narration.', settings,
            context_length=32768, parallel=2)

        stages = {request["stage"] for request in report["requests"]}
        self.assertIn("attribute", stages)
        self.assertIn("instruct", stages)
        self.assertEqual(16384, report["per_slot_context"])

    def test_three_pass_estimator_includes_context_rescue_for_unknown_split(self):
        settings = {
            "chunk_size": 6000, "max_tokens": 4096,
            "segment_output_ratio": 3.0, "presegment_quotes": True,
        }
        report = tp.build_three_pass_request_preflight(
            "Unquoted source text.", settings, context_length=32768,
            parallel=1, context_windows=[2000, 4000])

        stages = [request["stage"] for request in report["requests"]]
        self.assertIn("segment", stages)
        self.assertIn("segment_context_rescue", stages)


class ResolveBatchOutputPathTests(unittest.TestCase):
    """Covers the Area 6 fix: a `replace`-policy collision with a *reserved*
    (in-batch) output must never share a path with the job that reserved it."""

    def test_no_collision_is_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp, "book.json"))
            resolved, action = script._resolve_batch_output_path(path, "replace", set())
            self.assertEqual(path, resolved)
            self.assertEqual("ok", action)

    def test_cancel_policy_skips_on_reserved_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp, "book.json"))
            reserved = {path}
            resolved, action = script._resolve_batch_output_path(path, "cancel", reserved)
            self.assertEqual(path, resolved)
            self.assertEqual("skip", action)

    def test_version_policy_suffixes_on_reserved_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp, "book.json"))
            reserved = {path}
            resolved, action = script._resolve_batch_output_path(path, "version", reserved)
            self.assertEqual("version", action)
            self.assertEqual(str(Path(tmp, "book_2.json")), resolved)

    def test_replace_policy_versions_reserved_collision_instead_of_overwriting(self):
        # Two same-stem inputs under collision_policy="replace": task 2's collision
        # is with task 1's *reserved* output, not a disk file. Prior behavior fell
        # through and returned the same path, so task 2 would silently overwrite
        # task 1's output once both jobs ran.
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp, "book.json"))
            reserved = {path}
            resolved, action = script._resolve_batch_output_path(path, "replace", reserved)
            self.assertEqual("version", action)
            self.assertNotEqual(path, resolved)
            self.assertEqual(str(Path(tmp, "book_2.json")), resolved)

    def test_replace_policy_backs_up_disk_only_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp, "book.json"))
            Path(path).write_text("{}", encoding="utf-8")
            resolved, action = script._resolve_batch_output_path(path, "replace", set())
            self.assertEqual(path, resolved)
            self.assertEqual("backup", action)


if __name__ == "__main__":
    unittest.main()
