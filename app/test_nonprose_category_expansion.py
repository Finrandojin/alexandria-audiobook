import argparse
import collections
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.nonprose_category_expansion import (
    CATEGORIES, build_pair_manifest, get_category_run_fingerprint,
    is_ordinary_prose, load_locked_probes, load_locked_pairs, match_controls,
    matching_cost, summarize, surface_features)


class NonproseCategoryExpansionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.probes = load_locked_probes()
        cls.locked_pairs = load_locked_pairs()
        cls.pool = [{**control, "_features": surface_features(control["text"])}
                    for _, control in cls.locked_pairs]
        cls.pairs = match_controls(cls.probes, cls.pool)

    def test_fixture_has_four_unique_locked_probes_per_category(self):
        counts = collections.Counter(probe["category"] for probe in self.probes)
        self.assertEqual({category: 4 for category in CATEGORIES}, dict(counts))
        self.assertEqual(24, len({probe["source_sha256"]
                                  for probe in self.probes}))

    def test_pilot_limit_still_spans_every_category(self):
        pilot = load_locked_probes(limit_per_category=1)
        self.assertEqual(set(CATEGORIES), {probe["category"] for probe in pilot})
        self.assertEqual(6, len(pilot))

    def test_fingerprint_maps_category_limit_without_mutating_args(self):
        args = argparse.Namespace(
            source=__file__, config=__file__, seeds=[1234],
            limit_per_category=1)
        fingerprint = get_category_run_fingerprint(args, [], {})
        self.assertEqual(1, fingerprint["limit"])
        self.assertFalse(hasattr(args, "limit"))

    def test_controls_are_distinct_ordinary_prose_and_deterministic(self):
        repeated = match_controls(self.probes, self.pool)
        first_ids = [control["source_sha256"] for _, control in self.pairs]
        second_ids = [control["source_sha256"] for _, control in repeated]
        self.assertEqual(first_ids, second_ids)
        self.assertEqual(first_ids, [control["source_sha256"]
                                     for _, control in self.locked_pairs])
        self.assertEqual(24, len(set(first_ids)))
        self.assertTrue(all(is_ordinary_prose(control["text"])
                            for _, control in self.pairs))
        self.assertFalse(set(first_ids) &
                         {probe["source_sha256"] for probe in self.probes})

    def test_pair_manifest_reports_matching_cost_and_feature_gaps(self):
        manifest = build_pair_manifest(self.locked_pairs)
        self.assertEqual(24, len(manifest))
        for item, (probe, control) in zip(manifest, self.locked_pairs):
            self.assertEqual(probe["category"], item["category"])
            self.assertEqual(matching_cost(probe["text"], control["text"]),
                             item["matching_cost"])
            self.assertEqual({"chars", "words", "digit_fraction",
                              "uppercase_word_fraction",
                              "punctuation_fraction"},
                             set(item["absolute_feature_gap"]))

    def test_summary_never_pools_categories(self):
        base = {"adapter": "a", "seed": 1, "class": "probe", "words": 10,
                "errors": 2, "failed": True, "substitutions": 1,
                "deletions": 0, "insertions": 1}
        rows = [{**base, "category": "urls"},
                {**base, "category": "copyright", "errors": 0,
                 "failed": False, "substitutions": 0, "insertions": 0}]
        result = summarize(rows)
        self.assertEqual(2, len(result))
        self.assertEqual({"urls", "copyright"},
                         {item["category"] for item in result})


if __name__ == "__main__":
    unittest.main()
