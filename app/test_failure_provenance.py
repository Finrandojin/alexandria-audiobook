"""A failed generation must record which model failed. Goal 3.1.

WHAT THIS COSTS WHEN IT IS MISSING. 15 of the 34 saved books failed generation
with `chunk_failed_after_retries`, and not one records `model_name`. Their
response logs do not name a model either, so the worst failures this app has
produced can never be attributed. Goal 3.1 asks for a completion rate *on the
shipped model* and cannot be answered for them at all.

IT WAS PROVEN, NOT INFERRED. On 2026-08-09 all four books ran against a server
that was verified to be serving qwen3-14b. The two books that completed
recorded `model_name: qwen3-14b`. grimgar03, which failed chunk 1, recorded
`model_name: None` - same run, same server, same minutes. So the omission is in
the failure path, not in the environment.

Both failure call sites in `main` now pass `model_name`. These tests
assert the manifest carries it whatever the status, because the value of a
provenance field is exactly zero on the runs that succeed - a successful run
can be re-attributed from its output. Only the failures need it.
"""
import unittest

from generate_script import build_generation_quality_manifest

FINGERPRINT = {"source_sha256": "abc123"}


def manifest(status, **details):
    return build_generation_quality_manifest(
        status, FINGERPRINT, [], [], **details)


class FailureProvenanceTest(unittest.TestCase):

    def test_failed_manifest_keeps_the_model_name(self):
        record = manifest("failed", total_chunks=49, failed_chunk=1,
                          failure="chunk_failed_after_retries",
                          model_name="qwen3-14b")
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["model_name"], "qwen3-14b")

    def test_post_return_failure_keeps_the_model_name(self):
        record = manifest("failed", total_chunks=49, failed_chunk=7,
                          failure="post_return_validation_failed",
                          model_name="qwen3-14b")
        self.assertEqual(record["model_name"], "qwen3-14b")

    def test_a_none_model_is_not_silently_acceptable(self):
        """The exact shape of the 15 unattributable books.

        A manifest may legitimately be built without the field - the helper
        takes arbitrary details - but a record whose status is failed and whose
        model is None is the state that made goal 3.1 unanswerable, and this
        names it so a future reader does not have to rediscover why it matters.
        """
        record = manifest("failed", total_chunks=49,
                          failure="chunk_failed_after_retries")
        self.assertIsNone(record.get("model_name"),
                          "this test documents the BROKEN shape; if a default "
                          "was added, update it to assert the default")

    def test_the_two_call_sites_pass_it(self):
        """Reads the source, because the manifest helper cannot enforce this.

        `build_generation_quality_manifest` accepts **details, so a caller that
        forgets `model_name` produces a valid manifest with the field absent -
        which is precisely how this defect survived. The guarantee lives at the
        call sites, so that is what is checked.
        """
        import inspect
        import generate_script
        source = inspect.getsource(generate_script.main)
        failures = [block for block in source.split(
            "save_generation_quality_manifest(")[1:]
            if '"failed"' in block.split(")")[0] + block[:400]]
        self.assertGreaterEqual(len(failures), 2,
                                "expected both failure call sites")
        for index, block in enumerate(failures):
            with self.subTest(call_site=index):
                self.assertIn("model_name", block[:400],
                              "a failure manifest without model_name makes "
                              "the run unattributable, as 15 saved books are")


if __name__ == "__main__":
    unittest.main()
