import unittest

import numpy as np

from experiments.simd_benchmark import summarize
from experiments.simd_benchmark_worker import numpy_audio_metrics, scalar_audio_metrics


class SimdBenchmarkTests(unittest.TestCase):
    def test_scalar_and_numpy_audio_metrics_match(self):
        values = np.array([-32768, -32, 0, 32, 32767], dtype=np.int16)
        self.assertTrue(np.allclose(
            scalar_audio_metrics(values), numpy_audio_metrics(values),
            rtol=1e-5, atol=1e-7))

    def test_summary_requires_disabled_baseline_and_equivalent_outputs(self):
        def run(arm, timings, mean=1.0):
            return {"arm": arm,
                    "cpu_features": {"AVX2": arm == "native",
                                     "AVX512F": arm == "native"},
                    "cases": [{"name": "case", "result_signature": {
                        "shape": [2], "mean": mean, "rms": 2.0,
                        "minimum": 0.0, "maximum": 3.0},
                               "timings_ns": timings}]}
        runs = [run("baseline", [200, 210, 220]),
                run("native", [100, 105, 110], mean=1.000001),
                run("native", [101, 106, 111]),
                run("baseline", [201, 211, 221])]
        self.assertEqual("native_faster", summarize(runs)[0]["verdict"])
        runs[1]["cases"][0]["result_signature"]["mean"] = 1.1
        with self.assertRaisesRegex(RuntimeError, "output mismatch"):
            summarize(runs)

    def test_summary_rejects_fake_baseline(self):
        run = {"arm": "baseline", "cpu_features": {"AVX2": True},
               "cases": [{"name": "case", "result_signature": {"shape": [1]},
                          "timings_ns": [1]}]}
        native = {"arm": "native", "cpu_features": {"AVX2": True},
                  "cases": [{"name": "case", "result_signature": {"shape": [1]},
                             "timings_ns": [1]}]}
        with self.assertRaisesRegex(RuntimeError, "did not disable"):
            summarize([run, native])


if __name__ == "__main__":
    unittest.main()
