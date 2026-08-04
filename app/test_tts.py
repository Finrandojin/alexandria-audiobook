import os
import tempfile
import unittest
from unittest.mock import patch

from tts import (
    MINIMAX_DEFAULT_VOICE_CLONE_MODEL,
    MINIMAX_FILE_UPLOAD_ENDPOINTS,
    MINIMAX_VOICE_CLONE_ENDPOINTS,
    MINIMAX_VOICE_DESIGN_ENDPOINTS,
    TTSEngine,
)


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class MiniMaxVoiceCloneTests(unittest.TestCase):
    def make_engine(self, **overrides):
        config = {
            "mode": "external",
            "provider": "minimax",
            "api_key": "test",
            "region": "global_en",
            "voice_clone_model": MINIMAX_DEFAULT_VOICE_CLONE_MODEL,
        }
        config.update(overrides)
        return TTSEngine({"tts": config})

    def write_audio(self, tmpdir, name="ref.wav"):
        path = os.path.join(tmpdir, name)
        with open(path, "wb") as f:
            f.write(b"RIFF-test-audio")
        return path

    def test_regional_endpoint_selection(self):
        engine = self.make_engine(region="cn_zh")
        self.assertEqual(
            engine._minimax_endpoint_for(MINIMAX_FILE_UPLOAD_ENDPOINTS),
            "https://api.minimaxi.com/v1/files/upload",
        )
        self.assertEqual(
            engine._minimax_endpoint_for(MINIMAX_VOICE_CLONE_ENDPOINTS),
            "https://api.minimaxi.com/v1/voice_clone",
        )
        self.assertEqual(
            engine._minimax_endpoint_for(MINIMAX_VOICE_DESIGN_ENDPOINTS),
            "https://api.minimaxi.com/v1/voice_design",
        )

    def test_upload_audio_returns_file_id(self):
        response = FakeResponse({
            "file": {"file_id": "file-123"},
            "base_resp": {"status_code": 0},
        })
        engine = self.make_engine()

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = self.write_audio(tmpdir)
            with patch("tts.requests.post", return_value=response) as post_request:
                file_id = engine.minimax_upload_audio(audio_path, "voice_clone")

        self.assertEqual(file_id, "file-123")
        request_args, request_kwargs = post_request.call_args
        self.assertEqual(request_args[0], MINIMAX_FILE_UPLOAD_ENDPOINTS["global_en"])
        self.assertEqual(request_kwargs["headers"]["Authorization"], "Bearer test")
        self.assertEqual(request_kwargs["data"]["purpose"], "voice_clone")
        self.assertIn("file", request_kwargs["files"])

    def test_upload_audio_supports_prompt_purpose(self):
        response = FakeResponse({
            "file": {"file_id": "file-456"},
            "base_resp": {"status_code": 0},
        })
        engine = self.make_engine()

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = self.write_audio(tmpdir)
            with patch("tts.requests.post", return_value=response) as post_request:
                file_id = engine.minimax_upload_audio(audio_path, "prompt_audio")

        self.assertEqual(file_id, "file-456")
        self.assertEqual(
            post_request.call_args.kwargs["data"]["purpose"], "prompt_audio"
        )

    def test_upload_audio_rejects_unsupported_purpose(self):
        engine = self.make_engine()
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = self.write_audio(tmpdir)
            with self.assertRaises(ValueError):
                engine.minimax_upload_audio(audio_path, "transcribe")

    def test_upload_audio_rejects_unsupported_extension(self):
        engine = self.make_engine()
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = self.write_audio(tmpdir, name="ref.flac")
            with self.assertRaises(ValueError):
                engine.minimax_upload_audio(audio_path, "voice_clone")

    def test_upload_audio_requires_api_key(self):
        engine = self.make_engine(api_key="")
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = self.write_audio(tmpdir)
            with self.assertRaises(ValueError):
                engine.minimax_upload_audio(audio_path, "voice_clone")

    def test_upload_audio_raises_when_file_missing(self):
        engine = self.make_engine()
        with self.assertRaises(ValueError):
            engine.minimax_upload_audio("/nonexistent/ref.wav", "voice_clone")

    def test_clone_voice_uploads_then_clones(self):
        upload_response = FakeResponse({
            "file": {"file_id": "file-123"},
            "base_resp": {"status_code": 0},
        })
        clone_response = FakeResponse({
            "voice_id": "voice-cloned-1",
            "base_resp": {"status_code": 0},
        })
        engine = self.make_engine()

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = self.write_audio(tmpdir)
            with patch(
                "tts.requests.post", side_effect=[upload_response, clone_response]
            ) as post_request:
                voice_id = engine.minimax_clone_voice(
                    audio_path, "my-voice", "speech-2.6-hd"
                )

        self.assertEqual(voice_id, "voice-cloned-1")
        self.assertEqual(post_request.call_count, 2)
        upload_kwargs = post_request.call_args_list[0].kwargs
        self.assertEqual(upload_kwargs["data"]["purpose"], "voice_clone")
        clone_args, clone_kwargs = post_request.call_args_list[1]
        self.assertEqual(clone_args[0], MINIMAX_VOICE_CLONE_ENDPOINTS["global_en"])
        self.assertEqual(clone_kwargs["json"], {
            "file_id": "file-123",
            "voice_id": "my-voice",
            "model": "speech-2.6-hd",
        })

    def test_clone_voice_uses_configured_default_model(self):
        upload_response = FakeResponse({
            "file": {"file_id": "file-123"},
            "base_resp": {"status_code": 0},
        })
        clone_response = FakeResponse({
            "voice_id": "voice-cloned-1",
            "base_resp": {"status_code": 0},
        })
        engine = self.make_engine(voice_clone_model="speech-01-hd")

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = self.write_audio(tmpdir)
            with patch(
                "tts.requests.post", side_effect=[upload_response, clone_response]
            ) as post_request:
                engine.minimax_clone_voice(audio_path, "my-voice")

        clone_kwargs = post_request.call_args_list[1].kwargs
        self.assertEqual(clone_kwargs["json"]["model"], "speech-01-hd")

    def test_clone_voice_rejects_unsupported_model(self):
        engine = self.make_engine()
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = self.write_audio(tmpdir)
            with patch("tts.requests.post") as post_request:
                with self.assertRaises(ValueError):
                    engine.minimax_clone_voice(audio_path, "my-voice", "speech-99")
            post_request.assert_not_called()

    def test_voice_design_returns_voice_id(self):
        response = FakeResponse({
            "voice_id": "voice-designed-1",
            "base_resp": {"status_code": 0},
        })
        engine = self.make_engine()

        with patch("tts.requests.post", return_value=response) as post_request:
            voice_id = engine.minimax_voice_design(
                "Warm grandmotherly tone", "design-1"
            )

        self.assertEqual(voice_id, "voice-designed-1")
        request_args, request_kwargs = post_request.call_args
        self.assertEqual(request_args[0], MINIMAX_VOICE_DESIGN_ENDPOINTS["global_en"])
        self.assertEqual(request_kwargs["json"], {
            "prompt": "Warm grandmotherly tone",
            "voice_id": "design-1",
        })

    def test_voice_design_rejects_empty_prompt(self):
        engine = self.make_engine()
        with patch("tts.requests.post") as post_request:
            with self.assertRaises(ValueError):
                engine.minimax_voice_design("   ", "design-1")
            post_request.assert_not_called()

    def test_voice_clone_raises_on_api_error(self):
        upload_response = FakeResponse({
            "file": {"file_id": "file-123"},
            "base_resp": {"status_code": 0},
        })
        clone_response = FakeResponse({
            "base_resp": {"status_code": 1004, "status_msg": "invalid model"},
        })
        engine = self.make_engine()

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = self.write_audio(tmpdir)
            with patch(
                "tts.requests.post", side_effect=[upload_response, clone_response]
            ):
                with self.assertRaises(RuntimeError):
                    engine.minimax_clone_voice(audio_path, "my-voice")

    def test_voice_design_raises_on_api_error(self):
        response = FakeResponse({
            "base_resp": {"status_code": 1001, "status_msg": "bad prompt"},
        })
        engine = self.make_engine()
        with patch("tts.requests.post", return_value=response):
            with self.assertRaises(RuntimeError):
                engine.minimax_voice_design("some prompt", "design-1")

    def test_voice_clone_raises_when_voice_id_missing(self):
        upload_response = FakeResponse({
            "file": {"file_id": "file-123"},
            "base_resp": {"status_code": 0},
        })
        clone_response = FakeResponse({
            "base_resp": {"status_code": 0},
        })
        engine = self.make_engine()

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = self.write_audio(tmpdir)
            with patch(
                "tts.requests.post", side_effect=[upload_response, clone_response]
            ):
                with self.assertRaises(RuntimeError):
                    engine.minimax_clone_voice(audio_path, "my-voice")


if __name__ == "__main__":
    unittest.main()
