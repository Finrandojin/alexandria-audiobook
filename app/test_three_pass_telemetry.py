import unittest

from three_pass_generate import build_failure_record


class BuildFailureRecordTest(unittest.TestCase):

    def test_record_carries_causal_fields(self):
        record = build_failure_record(
            pass_name="attribute", index=7, text="He said hello.",
            last_attempt={"finish_reason": "length", "prompt_tokens": 2328,
                          "completion_tokens": 10000, "reasoning_tokens": 9987,
                          "effective_max_tokens": 10000, "attempt": 3,
                          "failure_codes": ["missing_json_array"]})
        self.assertEqual(record["pass"], "attribute")
        self.assertEqual(record["entry"], 7)
        self.assertEqual(record["finish_reason"], "length")
        self.assertEqual(record["reasoning_tokens"], 9987)
        self.assertEqual(record["effective_max_tokens"], 10000)
        self.assertEqual(record["attempt"], 3)
        self.assertEqual(record["reason"], "missing_json_array")
        self.assertEqual(record["text_preview"], "He said hello.")
        self.assertEqual(len(record["text_sha256"]), 64)

    def test_record_tolerates_missing_attempt_data(self):
        record = build_failure_record(
            pass_name="instruct", index=0, text="x", last_attempt=None)
        self.assertEqual(record["pass"], "instruct")
        self.assertIsNone(record["finish_reason"])
        self.assertIsNone(record["reasoning_tokens"])
        self.assertEqual(record["reason"], "unknown")


class ModelProfileMappingTest(unittest.TestCase):
    """load_app_config validates profiles into ThreePassModelProfile objects,
    but this module reads them with .get(). A configured profile previously
    crashed the run with AttributeError."""

    def test_pydantic_profile_is_readable(self):
        from config_settings import ThreePassModelProfile
        from three_pass_generate import as_profile_mapping
        profile = as_profile_mapping(
            ThreePassModelProfile(segment_output_ratio=5.0))
        self.assertEqual(profile.get("segment_output_ratio"), 5.0)

    def test_unset_fields_fall_back_to_the_default(self):
        from config_settings import ThreePassModelProfile
        from three_pass_generate import as_profile_mapping
        profile = as_profile_mapping(
            ThreePassModelProfile(segment_output_ratio=5.0))
        # model_dump() emits None for unset fields; those must not shadow the
        # caller's default or chunk_size would become None.
        self.assertEqual(profile.get("chunk_size", 3000), 3000)

    def test_plain_dict_still_works(self):
        from three_pass_generate import as_profile_mapping
        self.assertEqual(
            as_profile_mapping({"segment_output_ratio": 4.0}),
            {"segment_output_ratio": 4.0})

    def test_missing_profile_is_empty(self):
        from three_pass_generate import as_profile_mapping
        self.assertEqual(as_profile_mapping(None), {})


class SegmentFailureReasonTest(unittest.TestCase):
    """Pass-1 fails per source chunk, not per entry, so it has its own record
    builder. It must still expose "reason" or the manifest rollup buckets every
    segment failure as unknown."""

    def test_reason_comes_from_the_failure_codes(self):
        from three_pass_generate import build_segment_failure_record
        record = build_segment_failure_record(
            4, "some source text", ["low_recall", "adjacent_duplicate"])
        self.assertEqual(record["pass"], "segment")
        self.assertEqual(record["chunk"], 4)
        self.assertEqual(record["reason"], "adjacent_duplicate")
        self.assertEqual(record["failure_codes"],
                         ["adjacent_duplicate", "low_recall"])
        self.assertEqual(record["source_characters"], len("some source text"))
        self.assertEqual(len(record["source_sha256"]), 64)

    def test_reason_falls_back_when_no_codes(self):
        from three_pass_generate import build_segment_failure_record
        record = build_segment_failure_record(1, "text", [])
        self.assertEqual(record["reason"], "segment_exhausted")

    def test_rollup_counts_segment_failures(self):
        from three_pass_generate import build_segment_failure_record
        from collections import Counter
        failures = [build_segment_failure_record(1, "a", ["low_recall"]),
                    build_segment_failure_record(2, "b", [])]
        reasons = Counter(f.get("reason") or "unknown" for f in failures)
        self.assertEqual(reasons["low_recall"], 1)
        self.assertEqual(reasons["segment_exhausted"], 1)
        self.assertNotIn("unknown", reasons)


class ObserverPlumbingTest(unittest.TestCase):

    def test_attribute_batch_accepts_an_observer(self):
        import inspect

        from three_pass_generate import attribute_batch, instruct_batch
        self.assertIn("attempt_observer",
                      inspect.signature(attribute_batch).parameters)
        self.assertIn("attempt_observer",
                      inspect.signature(instruct_batch).parameters)



class CheckedInProfileDefaultsTest(unittest.TestCase):
    """Measured per-model profiles ship in the repo so a run is reproducible
    on another machine; app/config.json is gitignored and machine-local."""

    def test_checked_in_defaults_load(self):
        from three_pass_generate import load_default_model_profiles
        defaults = load_default_model_profiles()
        self.assertIsInstance(defaults, dict)

    def test_config_profile_overrides_checked_in_default(self):
        from three_pass_generate import resolve_model_profile
        profile = resolve_model_profile(
            "m", config_profiles={"m": {"segment_output_ratio": 9.9}},
            defaults={"m": {"segment_output_ratio": 5.0, "chunk_size": 4000}})
        self.assertEqual(profile["segment_output_ratio"], 9.9)
        # Keys the config does not set still come from the checked-in default.
        self.assertEqual(profile["chunk_size"], 4000)

    def test_checked_in_default_used_when_config_is_silent(self):
        from three_pass_generate import resolve_model_profile
        profile = resolve_model_profile(
            "m", config_profiles={}, defaults={"m": {"segment_output_ratio": 5.0}})
        self.assertEqual(profile["segment_output_ratio"], 5.0)

    def test_unknown_model_is_empty(self):
        from three_pass_generate import resolve_model_profile
        self.assertEqual(resolve_model_profile("nope", {}, {}), {})


class RosterAttestationTest(unittest.TestCase):
    """A single hallucinated speaker enters the roster and is then fed back to
    every later batch as a known character. Measured on mushoku16: WEARING was
    invented at entry 11 and propagated to 1,106 - 89 entries in one arm, and
    it reproduced in all three arms.

    Real character names appear as capitalized proper nouns in the prose
    (28-291 times each, never lowercase). WEARING appears twice capitalized at
    sentence start and 15 times lowercase."""

    # Realistic proportions: a named character is written capitalized many
    # times over, a common word mostly lowercase. A fixture with only two or
    # three mentions cannot exercise the ratio.
    SOURCE = (("Roxy smiled. Eris laughed at Roxy. " * 6)
              + ("He was wearing a coat, wearing it badly. " * 8)
              + "Wearing thin, the day ended.")

    def test_attested_names_are_admitted(self):
        from three_pass_generate import build_roster
        entries = [{"speaker": "ROXY"}, {"speaker": "ERIS"}]
        self.assertEqual(build_roster(entries, self.SOURCE), ["ROXY", "ERIS"])

    def test_a_name_that_is_also_a_word_still_passes(self):
        # Rose is named far more often than "a rose" is used, so the ratio
        # keeps her even though the lowercase form occurs.
        from three_pass_generate import build_roster
        source = ("Rose spoke. " * 20) + "He picked up a rose. "
        self.assertEqual(build_roster([{"speaker": "ROSE"}], source), ["ROSE"])

    def test_common_word_is_rejected(self):
        from three_pass_generate import build_roster
        entries = [{"speaker": "ROXY"}, {"speaker": "WEARING"}]
        self.assertEqual(build_roster(entries, self.SOURCE), ["ROXY"])

    def test_without_source_every_name_is_kept(self):
        # Callers that have no source text must keep today's behaviour.
        from three_pass_generate import build_roster
        entries = [{"speaker": "ROXY"}, {"speaker": "WEARING"}]
        self.assertEqual(build_roster(entries), ["ROXY", "WEARING"])

    def test_narrator_and_unknown_still_excluded(self):
        from three_pass_generate import build_roster
        entries = [{"speaker": "NARRATOR"}, {"speaker": "UNKNOWN"},
                   {"speaker": "ROXY"}]
        self.assertEqual(build_roster(entries, self.SOURCE), ["ROXY"])


if __name__ == "__main__":
    unittest.main()
