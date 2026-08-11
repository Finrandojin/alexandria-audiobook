import os
import tempfile
import unittest
from unittest.mock import patch

from tts import (
    MINIMAX_DEFAULT_SPEECH_MODEL,
    MINIMAX_SPEECH_ENDPOINTS,
    TTSEngine,
    _instruct_to_minimax_emotion,
)


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class MiniMaxSpeechTests(unittest.TestCase):
    def make_engine(self, **overrides):
        config = {
            "mode": "external",
            "provider": "minimax",
            "api_key": "test",
            "model": MINIMAX_DEFAULT_SPEECH_MODEL,
            "region": "global_en",
            "language": "English",
        }
        config.update(overrides)
        return TTSEngine({"tts": config})

    def test_regional_endpoint_selection(self):
        engine = self.make_engine(region="cn_zh")
        self.assertEqual(engine._minimax_endpoint(), MINIMAX_SPEECH_ENDPOINTS["cn_zh"])

    def test_payload_contains_required_speech_fields(self):
        engine = self.make_engine(model="speech-2.8-turbo")
        payload = engine._build_minimax_payload("Hello", "voice-1")

        self.assertEqual(payload["model"], "speech-2.8-turbo")
        self.assertEqual(payload["text"], "Hello")
        self.assertEqual(payload["voice_setting"], {"voice_id": "voice-1"})
        self.assertEqual(payload["audio_setting"], {"format": "wav"})
        self.assertEqual(payload["output_format"], "hex")
        self.assertFalse(payload["stream"])

    def test_instruct_maps_to_minimax_emotion(self):
        self.assertEqual(_instruct_to_minimax_emotion("shouted furiously"), "angry")

        engine = self.make_engine()
        payload = engine._build_minimax_payload(
            "Stop!", "voice-1", "shouted furiously"
        )
        self.assertEqual(payload["voice_setting"]["emotion"], "angry")

    def test_generate_decodes_hex_audio(self):
        audio = b"RIFFtest-wav"
        response = FakeResponse({
            "data": {"audio": audio.hex(), "status": 2},
            "base_resp": {"status_code": 0},
        })
        engine = self.make_engine()
        voices = {"NARRATOR": {"voice_id": "voice-1"}}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "speech.wav")
            with patch("tts.requests.post", return_value=response) as post_request:
                success = engine.generate_custom_voice(
                    "Hello", "shouted furiously", "NARRATOR", voices, output_path
                )

            self.assertTrue(success)
            with open(output_path, "rb") as output_file:
                self.assertEqual(output_file.read(), audio)

        request_args, request_kwargs = post_request.call_args
        self.assertEqual(request_args[0], MINIMAX_SPEECH_ENDPOINTS["global_en"])
        self.assertEqual(request_kwargs["headers"]["Authorization"], "Bearer test")
        self.assertEqual(
            request_kwargs["json"]["voice_setting"],
            {"voice_id": "voice-1", "emotion": "angry"},
        )

    def test_minimax_does_not_override_clone_or_design_paths(self):
        engine = self.make_engine()
        clone_voices = {"NARRATOR": {"type": "clone"}}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "speech.wav")
            with patch.object(
                engine, "_external_generate_clone", return_value=True
            ) as external_clone, patch.object(engine, "_minimax_generate") as minimax:
                success = engine.generate_clone_voice(
                    "Hello", "NARRATOR", clone_voices, output_path
                )

            self.assertTrue(success)
            external_clone.assert_called_once()
            minimax.assert_not_called()

        design_voices = {
            "NARRATOR": {"type": "design", "description": "Warm narration"}
        }
        with patch.object(
            engine, "generate_design_voice", return_value=True
        ) as design_voice, patch.object(engine, "_minimax_generate") as minimax:
            success = engine.generate_voice(
                "Hello", "calm", "NARRATOR", design_voices, "speech.wav"
            )

        self.assertTrue(success)
        design_voice.assert_called_once()
        minimax.assert_not_called()

    def test_generate_retries_transient_http_failure(self):
        audio = b"RIFFretry-wav"
        responses = [
            FakeResponse({}, status_code=500),
            FakeResponse({
                "data": {"audio": audio.hex(), "status": 2},
                "base_resp": {"status_code": 0},
            }),
        ]
        engine = self.make_engine()
        voices = {"NARRATOR": {"voice": "voice-1"}}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "speech.wav")
            with patch("tts.requests.post", side_effect=responses) as post_request, patch(
                "tts.time.sleep"
            ) as sleep:
                success = engine.generate_custom_voice(
                    "Hello", "", "NARRATOR", voices, output_path
                )

            self.assertTrue(success)
            self.assertEqual(post_request.call_count, 2)
            sleep.assert_called_once_with(0.5)

    def test_generate_rejects_api_error(self):
        response = FakeResponse({
            "data": {"audio": "", "status": 2},
            "base_resp": {"status_code": 1000, "status_msg": "invalid request"},
        })
        engine = self.make_engine()
        voices = {"NARRATOR": {"voice": "voice-1"}}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "speech.wav")
            with patch("tts.requests.post", return_value=response), patch(
                "builtins.print"
            ):
                success = engine.generate_custom_voice(
                    "Hello", "", "NARRATOR", voices, output_path
                )

            self.assertFalse(success)
            self.assertFalse(os.path.exists(output_path))


if __name__ == "__main__":
    unittest.main()
