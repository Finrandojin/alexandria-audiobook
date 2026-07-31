import os
import tempfile
import unittest
from unittest.mock import patch

from tts import (
    MINIMAX_DEFAULT_SPEECH_MODEL,
    MINIMAX_SPEECH_ENDPOINTS,
    TTSEngine,
)


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

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
                success = engine.generate_clone_voice(
                    "Hello", "NARRATOR", voices, output_path
                )

            self.assertTrue(success)
            with open(output_path, "rb") as output_file:
                self.assertEqual(output_file.read(), audio)

        request_args, request_kwargs = post_request.call_args
        self.assertEqual(request_args[0], MINIMAX_SPEECH_ENDPOINTS["global_en"])
        self.assertEqual(request_kwargs["headers"]["Authorization"], "Bearer test")
        self.assertEqual(
            request_kwargs["json"]["voice_setting"], {"voice_id": "voice-1"}
        )

    def test_design_voice_uses_configured_voice_id(self):
        audio = b"RIFFdesign-wav"
        response = FakeResponse({
            "data": {"audio": audio.hex(), "status": 2},
            "base_resp": {"status_code": 0},
        })
        engine = self.make_engine()
        voices = {
            "NARRATOR": {
                "type": "design",
                "voice_id": "design-1",
                "description": "Warm narration",
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "speech.wav")
            with patch("tts.requests.post", return_value=response) as post_request:
                success = engine.generate_voice(
                    "Hello", "calm", "NARRATOR", voices, output_path
                )

            self.assertTrue(success)
            self.assertEqual(
                post_request.call_args.kwargs["json"]["voice_setting"],
                {"voice_id": "design-1"},
            )

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
