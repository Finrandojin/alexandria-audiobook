import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import core as core_module
from routers import voice_library as voice_library_module
from routers import voices as voices_module


class VoicesTests(unittest.TestCase):
    def test_pitch_is_not_used_as_a_gender_classifier(self):
        low = {"voice_features": {"mean_f0": 90}}
        high = {"voice_features": {"mean_f0": 260}}
        self.assertEqual("unknown", voices_module._infer_lora_gender(low))
        self.assertEqual("unknown", voices_module._infer_lora_gender(high))
        self.assertEqual(
            "female", voices_module._infer_lora_gender(
                {"description": "warm mezzo voice",
                 "voice_features": {"mean_f0": 90}}))

    def test_voice_library_mutations_hold_lock_across_load_and_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            library_path = os.path.join(tmp, "voice_library.json")
            Path(library_path).write_text(
                json.dumps({"shared": {}, "casts": {}}), encoding="utf-8")
            barrier = threading.Barrier(8)

            def add_cast(index):
                barrier.wait()
                voice_library_module._mutate_voice_library(
                    lambda lib: lib["casts"].update({f"cast-{index}": {"members": {}}}))

            with patch.object(core_module, "VOICE_LIBRARY_PATH", library_path), \
                 patch.object(voice_library_module, "VOICE_LIBRARY_PATH", library_path):
                threads = [threading.Thread(target=add_cast, args=(index,)) for index in range(8)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()

            saved = json.loads(Path(library_path).read_text(encoding="utf-8"))
            self.assertEqual({f"cast-{index}" for index in range(8)}, set(saved["casts"]))

    def test_voice_config_rejects_empty_ensemble_before_save(self):
        for members in (None, [], ["  "]):
            with self.subTest(members=members), self.assertRaises(ValueError):
                voices_module.VoiceConfigItem(type="ensemble", members=members)

        configured = voices_module.VoiceConfigItem(
            type="ensemble", members=["Petra", "Subaru"])
        self.assertEqual(["Petra", "Subaru"], configured.members)

    def test_voice_suggestion_honors_max_lines(self):
        captured = {}
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content='{"Hero": {"adapter_id": "voice", "reason": "fit"}}'
            ))]
        )

        def create(**kwargs):
            captured["prompt"] = kwargs["messages"][1]["content"]
            return response

        client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)
        ))
        with tempfile.TemporaryDirectory() as tmp:
            script_path = os.path.join(tmp, "script.json")
            with open(script_path, "w", encoding="utf-8") as f:
                json.dump([{"speaker": "Hero", "text": f"distinct line {i}"}
                           for i in range(12)], f)
            with patch.object(voices_module, "SCRIPT_PATH", script_path), \
                 patch.object(voices_module, "VOICE_CONFIG_PATH", os.path.join(tmp, "missing.json")), \
                 patch.object(voices_module, "_build_lora_candidates", return_value=[{
                     "adapter_id": "voice", "name": "Voice", "gender": "unknown",
                     "description": "neutral", "type": "lora",
                 }]), \
                 patch.object(voices_module, "_make_llm_client", return_value=(client, "model")):
                voices_module._suggest_voices_impl(voices_module.SuggestVoicesRequest(max_lines=12))
        for i in range(12):
            self.assertIn(f"distinct line {i}", captured["prompt"])

    def test_generic_cast_keys_are_book_scoped(self):
        self.assertEqual(core_module.get_cast_member_key("Man", "book-01"), "man::book-01")
        self.assertEqual(core_module.get_cast_member_key("Man", "book-05"), "man::book-05")
        self.assertEqual(core_module.get_cast_member_key("Holo", "book-01"), "holo")
        with self.assertRaises(ValueError):
            core_module.get_cast_member_key("Guard", None)

    def test_cast_usage_counts_distinct_members_not_books(self):
        lib = {"shared": {}, "casts": {"series": {"members": {
            "holo": {"name": "Holo", "config": {"adapter_id": "v1"},
                     "assignments": {"b1": {"line_count": 10}, "b2": {"line_count": 20}}},
            "man::b1": {"name": "Man", "config": {"adapter_id": "v1"},
                        "assignments": {"b1": {"line_count": 3}}},
        }}}}
        usage = core_module.get_cast_adapter_usage(lib, "series")
        self.assertEqual(usage["v1"]["character_count"], 2)
        self.assertEqual(usage["v1"]["total_lines"], 33)

    def test_major_characters_get_distinct_voices_before_minor_reuse(self):
        script = ([{"speaker": "Major A", "text": f"a{i}"} for i in range(30)]
                  + [{"speaker": "Major B", "text": f"b{i}"} for i in range(25)]
                  + [{"speaker": "Man", "text": f"m{i}"} for i in range(3)])
        parsed = {"characters": [
            {"name": name, "ranked_adapter_ids": ["v1", "v2"],
             "character_style": f"style {name}", "reason": "book evidence",
             "character_gender": "unknown", "age_group": "unknown",
             "trait_evidence": "none", "trait_confidence": "unknown"}
            for name in ("Major A", "Major B", "Man")
        ]}
        response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=json.dumps(parsed)))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **_kwargs: response)))
        candidates = [
            {"adapter_id": "v1", "name": "V1", "type": "lora", "gender": "unknown", "description": ""},
            {"adapter_id": "v2", "name": "V2", "type": "lora", "gender": "unknown", "description": ""},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            script_path = os.path.join(tmp, "script.json")
            with open(script_path, "w", encoding="utf-8") as f:
                json.dump(script, f)
            with patch.object(voices_module, "SCRIPT_PATH", script_path), \
                 patch.object(voices_module, "VOICE_CONFIG_PATH", os.path.join(tmp, "missing.json")), \
                 patch.object(voices_module, "get_active_book_id", return_value="book-01"), \
                 patch.object(voices_module, "_load_voice_library", return_value={"shared": {}, "casts": {"series": {"members": {}}}}), \
                 patch.object(voices_module, "_build_lora_candidates", return_value=candidates), \
                 patch.object(voices_module, "_make_llm_client", return_value=(client, "model")):
                result = voices_module._suggest_voices_impl(
                    voices_module.SuggestVoicesRequest(cast="series", max_lines=4))
        self.assertEqual(result["suggestions"]["Major A"]["adapter_id"], "v1")
        self.assertEqual(result["suggestions"]["Major B"]["adapter_id"], "v2")
        self.assertTrue(result["suggestions"]["Man"]["reused"])
        self.assertEqual(list(result["suggestions"])[0], "Major A")
        self.assertEqual(result["method"], "llm")
        self.assertEqual(result["suggestions"]["Major A"]["character_style"], "style Major A")

    def test_casting_enforces_gender_and_prefers_closest_age(self):
        script = [{"speaker": "Old Man", "text": f"He spoke wearily line {i}"} for i in range(30)]
        parsed = {"characters": [{
            "name": "Old Man", "ranked_adapter_ids": ["female_old", "male_adult", "male_old"],
            "character_style": "Weathered and deliberate", "reason": "book evidence",
            "character_gender": "female", "age_group": "young_adult", "trait_evidence": "Conflicting LLM evidence",
            "trait_confidence": "high",
        }]}
        response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=json.dumps(parsed)), finish_reason="stop")])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **_kwargs: response)))
        candidates = [
            {"adapter_id": "female_old", "name": "F", "type": "lora", "gender": "female", "age_group": "elderly", "description": ""},
            {"adapter_id": "male_adult", "name": "MA", "type": "lora", "gender": "male", "age_group": "adult", "description": ""},
            {"adapter_id": "male_old", "name": "MO", "type": "lora", "gender": "male", "age_group": "elderly", "description": ""},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            script_path = os.path.join(tmp, "script.json")
            Path(script_path).write_text(json.dumps(script), encoding="utf-8")
            with patch.object(voices_module, "SCRIPT_PATH", script_path), \
                 patch.object(voices_module, "VOICE_CONFIG_PATH", os.path.join(tmp, "missing.json")), \
                 patch.object(voices_module, "get_active_book_id", return_value="b1"), \
                 patch.object(voices_module, "_load_voice_library", return_value={"shared": {}, "casts": {}}), \
                 patch.object(voices_module, "_build_lora_candidates", return_value=candidates), \
                 patch.object(voices_module, "_make_llm_client", return_value=(client, "model")), \
                 patch.object(voices_module, "get_current_status", return_value={"context_length": None}):
                result = voices_module._suggest_voices_impl(voices_module.SuggestVoicesRequest(max_lines=4))
        suggestion = result["suggestions"]["Old Man"]
        self.assertEqual(suggestion["adapter_id"], "male_old")
        self.assertEqual(suggestion["character_gender"], "male")
        self.assertEqual(suggestion["character_age_group"], "elderly")
        self.assertEqual(suggestion["gender_confidence"], "high")
        self.assertEqual(suggestion["age_confidence"], "high")
        self.assertIn("character label: male", suggestion["trait_evidence"])
        self.assertEqual(suggestion["llm_trait_evidence"], "Conflicting LLM evidence")
        self.assertFalse(suggestion["gender_fallback"])

    def test_lora_age_normalization(self):
        self.assertEqual(voices_module._infer_lora_age({"id": "warm_baritone_40s_m"}), "middle_aged")
        self.assertEqual(voices_module._infer_lora_age({"description": "elderly gravelly bass"}), "elderly")
        self.assertEqual(voices_module._infer_lora_age({"age": "40s"}), "middle_aged")
        self.assertEqual(voices_module._infer_lora_age({"age": 67}), "elderly")

    def test_numeric_age_boundaries_and_precedence(self):
        expected = {"aged 12": "child", "13 years old": "teen", "19-year-old": "teen",
                    "20 years old": "young_adult", "aged 39": "adult",
                    "40-year-old": "middle_aged", "aged 60": "elderly"}
        for text, group in expected.items():
            self.assertEqual(voices_module._infer_age_group(text), group, text)
        self.assertEqual(voices_module._infer_age_group("young man, aged 50"), "middle_aged")

    def test_age_parser_handles_invalid_ambiguous_and_decade_values(self):
        expected = {
            "0": "unknown", "1": "child", "67": "elderly", "120": "elderly",
            "121": "unknown", "aged 12 then aged 60": "child", "under 12": "child",
            "20s": "young_adult", "30s": "adult", "40s": "middle_aged",
            "50s": "middle_aged", "60s": "elderly", "70s": "elderly", "80s": "elderly",
        }
        for text, group in expected.items():
            self.assertEqual(voices_module._infer_age_group(text), group, text)

    def test_llm_traits_replace_local_traits_only_with_stronger_authority(self):
        candidates = [
            {"adapter_id": "female", "name": "F", "type": "lora", "gender": "female",
             "age_group": "adult", "description": ""},
            {"adapter_id": "male", "name": "M", "type": "lora", "gender": "male",
             "age_group": "adult", "description": ""},
        ]

        def suggest(confidence):
            parsed = {"characters": [{
                "name": "HERO", "ranked_adapter_ids": ["male", "female"],
                "character_style": "Direct", "reason": "book evidence",
                "character_gender": "male", "age_group": "young_adult",
                "trait_evidence": "The text identifies him as male",
                "trait_confidence": confidence,
            }]}
            response = SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(parsed)), finish_reason="stop")])
            client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
                create=lambda **_kwargs: response)))
            with tempfile.TemporaryDirectory() as tmp:
                script_path = os.path.join(tmp, "script.json")
                # The speaker LABEL carries the trait, not the dialogue. This
                # test used to give "Hero" the line "She was an old woman who
                # entered quietly" and assert Hero was therefore female and
                # elderly - encoding the inverted inference as correct. A
                # character's words describe whoever they are talking about, so
                # dialogue was removed as a trait source; the label is a
                # legitimate one and keeps this test's actual subject, which is
                # the authority ordering between local and LLM traits.
                Path(script_path).write_text(json.dumps([
                    {"speaker": "HERO",
                     "text": "She was an old woman who entered quietly."}
                ]), encoding="utf-8")
                with patch.object(voices_module, "SCRIPT_PATH", script_path), \
                     patch.object(voices_module, "VOICE_CONFIG_PATH", os.path.join(tmp, "missing.json")), \
                     patch.object(voices_module, "get_active_book_id", return_value="b1"), \
                     patch.object(voices_module, "_load_voice_library", return_value={"shared": {}, "casts": {}}), \
                     patch.object(voices_module, "_build_lora_candidates", return_value=candidates), \
                     patch.object(voices_module, "_make_llm_client", return_value=(client, "model")), \
                     patch.object(voices_module, "get_current_status", return_value={"context_length": None}):
                    return voices_module._suggest_voices_impl(
                        voices_module.SuggestVoicesRequest(max_lines=4)
                    )["suggestions"]["HERO"]

        # "HERO" carries no gender and the dialogue is no longer a trait
        # source, so local traits are unknown. A non-authoritative LLM claim
        # must not be adopted on top of that - "unknown" is a real answer, and
        # every consumer treats it as do-not-filter, do-not-penalise.
        for confidence in ("unknown", "low"):
            suggestion = suggest(confidence)
            self.assertEqual(suggestion["character_gender"], "unknown", confidence)
        for confidence in ("medium", "high"):
            suggestion = suggest(confidence)
            self.assertEqual(suggestion["character_gender"], "male", confidence)
            self.assertEqual(suggestion["gender_confidence"], confidence)
            self.assertEqual(suggestion["character_age_group"], "young_adult", confidence)
            self.assertEqual(suggestion["age_confidence"], confidence)

    def test_mixed_llm_trait_acceptance_does_not_merge_conflicting_evidence(self):
        parsed = {"characters": [{
            "name": "King", "ranked_adapter_ids": ["male_old"],
            "character_style": "Measured", "reason": "book evidence",
            "character_gender": "female", "age_group": "elderly",
            "trait_evidence": "An elderly woman speaks", "trait_confidence": "high",
        }]}
        response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=json.dumps(parsed)), finish_reason="stop")])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **_kwargs: response)))
        candidates = [{"adapter_id": "male_old", "name": "MO", "type": "lora",
                       "gender": "male", "age_group": "elderly", "description": ""}]
        with tempfile.TemporaryDirectory() as tmp:
            script_path = os.path.join(tmp, "script.json")
            Path(script_path).write_text(json.dumps([
                {"speaker": "King", "text": "The crown is mine."}
            ]), encoding="utf-8")
            with patch.object(voices_module, "SCRIPT_PATH", script_path), \
                 patch.object(voices_module, "VOICE_CONFIG_PATH", os.path.join(tmp, "missing.json")), \
                 patch.object(voices_module, "get_active_book_id", return_value="b1"), \
                 patch.object(voices_module, "_load_voice_library", return_value={"shared": {}, "casts": {}}), \
                 patch.object(voices_module, "_build_lora_candidates", return_value=candidates), \
                 patch.object(voices_module, "_make_llm_client", return_value=(client, "model")), \
                 patch.object(voices_module, "get_current_status", return_value={"context_length": None}):
                suggestion = voices_module._suggest_voices_impl(
                    voices_module.SuggestVoicesRequest(max_lines=4))["suggestions"]["King"]
        self.assertEqual(suggestion["character_gender"], "male")
        self.assertEqual(suggestion["character_age_group"], "elderly")
        self.assertIn("LM accepted age=elderly", suggestion["trait_evidence"])
        self.assertNotIn("elderly woman", suggestion["trait_evidence"])
        self.assertEqual(suggestion["llm_trait_evidence"], "An elderly woman speaks")

    def test_low_confidence_traits_do_not_hard_filter(self):
        candidates = [
            {"adapter_id": "male", "gender": "male", "age_group": "adult", "description": ""},
            {"adapter_id": "female", "gender": "female", "age_group": "elderly", "description": ""},
        ]
        traits = {"gender": "female", "gender_confidence": "low",
                  "age_group": "elderly", "age_confidence": "low"}
        chosen, _ranked, _new, fallback, _mismatch = voices_module.get_voice_allocation(
            "", candidates, ["male", "female"], traits, None, {}, "minor")
        self.assertEqual(chosen, "male")
        self.assertFalse(fallback)

    def test_gender_fallback_and_recurring_mismatch(self):
        male = {"adapter_id": "male", "gender": "male", "age_group": "adult", "description": ""}
        traits = {"gender": "female", "gender_confidence": "high",
                  "age_group": "young_adult", "age_confidence": "high"}
        chosen, _ranked, _new, fallback, mismatch = voices_module.get_voice_allocation(
            "", [male], ["male"], traits, None, {}, "major")
        self.assertEqual(chosen, "male")
        self.assertTrue(fallback)
        self.assertFalse(mismatch)
        chosen, _ranked, is_new, _fallback, mismatch = voices_module.get_voice_allocation(
            "", [male], ["male"], traits, "male", {}, "major")
        self.assertFalse(is_new)
        self.assertTrue(mismatch)

    def test_authoritative_gender_prefers_exact_then_unknown_then_opposite(self):
        female = {"adapter_id": "female", "gender": "female", "age_group": "adult", "description": ""}
        unknown = {"adapter_id": "unknown", "gender": "unknown", "age_group": "adult", "description": ""}
        male = {"adapter_id": "male", "gender": "male", "age_group": "adult", "description": ""}
        traits = {"gender": "female", "gender_confidence": "high",
                  "age_group": "adult", "age_confidence": "high"}
        cases = [
            ([male, unknown, female], "female", False),
            ([male, unknown], "unknown", True),
            ([male], "male", True),
        ]
        for candidates, expected, expected_fallback in cases:
            chosen, _ranked, _new, fallback, _mismatch = voices_module.get_voice_allocation(
                "", candidates, [c["adapter_id"] for c in candidates],
                traits, None, {}, "major")
            self.assertEqual(chosen, expected)
            self.assertEqual(fallback, expected_fallback)

    def test_recurring_mismatch_requires_authoritative_trait_confidence(self):
        male = {"adapter_id": "male", "gender": "male", "age_group": "child", "description": ""}
        for confidence in ("unknown", "low", "medium", "high"):
            traits = {"gender": "female", "gender_confidence": confidence,
                      "age_group": "elderly", "age_confidence": confidence}
            _chosen, _ranked, _new, _fallback, mismatch = voices_module.get_voice_allocation(
                "", [male], ["male"], traits, "male", {}, "major")
            self.assertEqual(mismatch, confidence in ("medium", "high"), confidence)

    def test_character_trait_evidence_prefers_label_over_dialogue(self):
        traits = voices_module._infer_character_traits(
            "Young Man", "", ["She told her mother that the queen had arrived."])
        self.assertEqual(traits["gender"], "male")
        self.assertEqual(traits["gender_confidence"], "high")
        self.assertEqual(traits["age_group"], "young_adult")

    def test_apply_suggestion_persists_style_and_book_scoped_cast_member(self):
        candidate = {"adapter_id": "v1", "name": "V1", "type": "lora",
                     "gender": "unknown", "description": ""}
        suggestion = {"adapter_id": "v1", "character_style": "Brief wary delivery", "book_id": "book-05",
                      "priority": "minor", "reason": "small suspicious role",
                      "character_gender": "male", "character_age_group": "adult",
                      "voice_gender": "male", "voice_age_group": "middle_aged",
                      "gender_confidence": "high", "age_confidence": "medium",
                      "trait_evidence": "label evidence", "local_trait_evidence": "label evidence",
                      "llm_trait_evidence": "", "gender_fallback": False,
                      "existing_trait_mismatch": True}
        with tempfile.TemporaryDirectory() as tmp:
            voice_path = os.path.join(tmp, "voice_config.json")
            library_path = os.path.join(tmp, "voice_library.json")
            with open(library_path, "w", encoding="utf-8") as f:
                json.dump({"shared": {}, "casts": {"series": {"members": {}}}}, f)
            with patch.object(voices_module, "VOICE_CONFIG_PATH", voice_path), \
                 patch.object(voices_module, "VOICE_LIBRARY_PATH", library_path), \
                 patch.object(core_module, "VOICE_LIBRARY_PATH", library_path), \
                 patch.object(voices_module, "get_active_book_id", return_value="book-05"), \
                 patch.object(voices_module, "_script_line_counts", return_value={"Man": 4}), \
                 patch.object(voices_module, "_build_lora_candidates", return_value=[candidate]):
                result = voices_module._apply_voice_suggestions({"Man": suggestion}, "series")
            voice = json.loads(Path(voice_path).read_text(encoding="utf-8"))
            library = json.loads(Path(library_path).read_text(encoding="utf-8"))
        self.assertEqual(voice["Man"]["character_style"], "Brief wary delivery")
        for field in core_module.get_trait_assignment_metadata(suggestion):
            self.assertEqual(voice["Man"][field], suggestion[field], field)
        member = library["casts"]["series"]["members"]["man::book-05"]
        self.assertEqual(member["config"]["adapter_id"], "v1")
        for field in core_module.get_trait_assignment_metadata(suggestion):
            self.assertEqual(member["config"][field], suggestion[field], field)
            self.assertEqual(member["assignments"]["book-05"][field], suggestion[field], field)
        self.assertEqual(member["assignments"]["book-05"]["character_style"], "Brief wary delivery")
        self.assertEqual(member["assignments"]["book-05"]["character_gender"], "male")
        self.assertEqual(member["assignments"]["book-05"]["age_confidence"], "medium")
        self.assertTrue(member["assignments"]["book-05"]["existing_trait_mismatch"])
        self.assertEqual(result["adapter_usage"]["v1"]["character_count"], 1)

    def test_cast_apply_uses_book_specific_style(self):
        assignment_traits = {
            "character_gender": "female", "character_age_group": "adult",
            "voice_gender": "female", "voice_age_group": "middle_aged",
            "trait_evidence": "book five evidence", "local_trait_evidence": "local evidence",
            "llm_trait_evidence": "LM evidence", "gender_confidence": "high",
            "age_confidence": "medium", "gender_fallback": False,
            "existing_trait_mismatch": True,
        }
        lib = {"shared": {}, "casts": {"series": {"members": {"holo": {
            "name": "Holo", "config": {"type": "lora", "adapter_id": "v1",
                                         "character_style": "default style",
                                         "character_gender": "female",
                                         "character_age_group": "young_adult"},
            "assignments": {"book-05": {"character_style": "book five style",
                                           **assignment_traits}},
        }}}}}
        config, applied = voice_library_module._apply_cast_mapping(
            lib, "series", {"Holo": "holo"}, {}, book_id="book-05")
        self.assertEqual(applied, ["Holo"])
        self.assertEqual(config["Holo"]["character_style"], "book five style")
        self.assertEqual(config["Holo"]["character_age_group"], "adult")
        self.assertEqual(config["Holo"]["age_confidence"], "medium")
        self.assertEqual(config["Holo"]["trait_evidence"], "book five evidence")
        for field, value in assignment_traits.items():
            self.assertEqual(config["Holo"][field], value, field)

    def test_legacy_generic_cast_member_is_not_auto_matched(self):
        lib = {"shared": {}, "casts": {"series": {"members": {
            "man": {"name": "Man", "config": {"type": "lora", "adapter_id": "v1"}}
        }}}}
        self.assertNotIn("man", voice_library_module._cast_match_pool(lib, "series", "book-05"))

    def test_numbered_generic_cast_keys_are_book_scoped(self):
        self.assertEqual(core_module.get_cast_member_key("Man 1", "b5"), "man 1::b5")
        self.assertEqual(core_module.get_cast_member_key("Guard #2", "b5"), "guard #2::b5")

    def test_shared_narrator_is_reused_and_counted(self):
        lib = {"shared": {"narrator": {"name": "Narrator", "config": {"adapter_id": "v1"},
                                         "assignments": {"b1": {"line_count": 100}}}},
               "casts": {"series": {"members": {}}}}
        self.assertEqual(core_module.get_cast_adapter_usage(lib, "series")["v1"]["character_count"], 1)
        self.assertIs(core_module.get_cast_storage_pool(lib, "series", "Narrator"), lib["shared"])

    def test_bulk_generic_mapping_resolves_per_book_member(self):
        lib = {"shared": {}, "casts": {"series": {"members": {
            "man::b1": {"name": "Man", "config": {"adapter_id": "v1"}, "generic": True,
                         "book_id": "b1", "assignments": {"b1": {"character_style": "one"}}},
            "man::b5": {"name": "Man", "config": {"adapter_id": "v2"}, "generic": True,
                         "book_id": "b5", "assignments": {"b5": {"character_style": "five"}}},
        }}}}
        config, _ = voice_library_module._apply_cast_mapping(
            lib, "series", {"Man": "man::b1"}, {}, chars={"Man": 2}, book_id="b5")
        self.assertEqual(config["Man"]["adapter_id"], "v2")
        self.assertEqual(config["Man"]["character_style"], "five")

    def test_stale_suggestion_is_rejected(self):
        with patch.object(voices_module, "_build_lora_candidates", return_value=[]), \
             patch.object(voices_module, "_script_line_counts", return_value={"Narrator": 2}), \
             patch.object(voices_module, "get_active_book_id", return_value="book-b"):
            with self.assertRaisesRegex(Exception, "different book"):
                voices_module._apply_voice_suggestions(
                    {"Narrator": {"adapter_id": "v1", "book_id": "book-a"}}, None)

    def test_apply_narrator_suggestion_uses_shared_pool(self):
        candidate = {"adapter_id": "v1", "name": "V1", "type": "lora",
                     "gender": "unknown", "description": ""}
        suggestion = {"adapter_id": "v1", "book_id": "b1",
                      "character_style": "steady", "priority": "major"}
        with tempfile.TemporaryDirectory() as tmp:
            voice_path = os.path.join(tmp, "voice.json")
            library_path = os.path.join(tmp, "library.json")
            Path(library_path).write_text(
                json.dumps({"shared": {}, "casts": {"series": {"members": {}}}}), encoding="utf-8")
            with patch.object(voices_module, "VOICE_CONFIG_PATH", voice_path), \
                 patch.object(voices_module, "VOICE_LIBRARY_PATH", library_path), \
                 patch.object(core_module, "VOICE_LIBRARY_PATH", library_path), \
                 patch.object(voices_module, "get_active_book_id", return_value="b1"), \
                 patch.object(voices_module, "_script_line_counts", return_value={"Narrator": 100}), \
                 patch.object(voices_module, "_build_lora_candidates", return_value=[candidate]):
                voices_module._apply_voice_suggestions({"Narrator": suggestion}, "series")
            library = json.loads(Path(library_path).read_text(encoding="utf-8"))
        self.assertEqual(library["shared"]["narrator"]["config"]["adapter_id"], "v1")
        self.assertNotIn("narrator", library["casts"]["series"]["members"])

    def test_selective_enrichment_prompt(self):
        fake_llama = SimpleNamespace(Llama=object, llama_supports_gpu_offload=lambda: True)
        path = Path(__file__).resolve().parent.parent / "llm_enricher.py"
        with patch.dict(sys.modules, {"llama_cpp": fake_llama}):
            spec = importlib.util.spec_from_file_location("test_llm_enricher", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        enricher = module.LLMEnricher.__new__(module.LLMEnricher)
        enricher.fields = ["emotional_tone"]
        prompt = enricher._create_prompt({"text": "hello", "start": 0, "end": 1})
        self.assertIn("emotional_tone", prompt)
        self.assertNotIn("speaker_attribution", prompt)
        self.assertNotIn("narration_style", prompt)
        self.assertEqual(
            enricher._parse_llm_output('{"emotional_tone": "calm"}')["emotional_tone"],
            "calm",
        )
