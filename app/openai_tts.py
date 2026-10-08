"""Client for the OpenAI-compatible /v1/audio/speech API."""

import io
import os
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import requests
import soundfile as sf


VOICE_TYPES = {"custom": "CustomVoice", "clone": "Base", "design": "VoiceDesign"}


class OpenAITTS:
    def __init__(self, config):
        self.base_url = (config.get("base_url") or "").rstrip("/")
        self.api_key = config.get("api_key") or ""
        self.model = (config.get("model") or "").strip()
        # Optional, non-spec extension. Some servers route between a model's variants
        # with it and pick a default when it is absent -- which can make a VoiceDesign
        # deployment hunt for a preset voice it does not have. vLLM-Omni serving
        # Qwen3-TTS is the case this was written against; servers that do not know the
        # field simply ignore it. Empty means "do not send it", leaving the server on
        # its own default.
        self.model_type = (config.get("model_type") or "").strip().lower()
        # Sent as-is; there is no runtime negotiation. A wrong choice is reported
        # straight back to the user rather than being silently retried with another
        # container -- 400 means the request itself is wrong, and changing one field
        # is guesswork, not a retry.
        self.response_format = (config.get("response_format") or "pcm").strip().lower()
        self.concurrency = max(1, min(int(config.get("concurrency", 16)), 32))

    def _url(self, path):
        if not self.base_url or not self.model:
            raise ValueError("Set API URL and Model Name first")
        base = self.base_url if self.base_url.endswith("/v1") else f"{self.base_url}/v1"
        parsed = urlparse(base)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("API URL must be an HTTP(S) address")
        return f"{base}/{path.lstrip('/')}"

    def _headers(self):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def status(self):
        """Cheap reachability check: one GET /v1/models, no generation.

        Used to fail a batch fast when the endpoint is down, rather than reporting
        the same connection error once per chunk. Some services list only a subset
        of their models here (OpenRouter hides TTS ones behind a query parameter),
        so a missing model name is not treated as a failure -- a response at all
        means the URL, key, and model routing are live.
        """
        response = requests.get(self._url("models"), headers=self._headers(), timeout=8)
        response.raise_for_status()
        return {"connected": True, "model_ready": True, "model": self.model}

    @staticmethod
    def _reference_data_url(path):
        if not path:
            raise ValueError("Voice Clone needs a reference recording")
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(os.path.dirname(__file__)), path)
        with open(path, "rb") as source:
            encoded = base64.b64encode(source.read()).decode("ascii")
        mime = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac"}.get(
            os.path.splitext(path)[1].lower(), "audio/wav"
        )
        return f"data:{mime};base64,{encoded}"

    def _item(self, chunk, voices, seed):
        speaker = chunk.get("speaker", "")
        voice = voices.get(speaker, {})
        text = (chunk.get("text") or "").strip()
        if not text:
            raise ValueError(f"Chunk {chunk['index']} has no text")
        # Only OpenAI spec fields, plus optional extensions that some services
        # understand. Which of them matter depends on the model behind the endpoint,
        # not on a client-side mode: a self-hosted vLLM-Omni serving Qwen3-TTS reads
        # task_type / voice / ref_audio, while a gateway such as OpenRouter only looks
        # at input / voice / response_format. The same request shape works for both --
        # a service ignores the fields it does not know.
        # "response_format" is deliberately absent here: _request adds it, so exactly
        # one place decides which container the audio comes back in.
        item = {"input": text, "non_streaming_mode": True, "max_new_tokens": 2048}

        # Non-spec variant selector -- see model_type above. Omitting it leaves the
        # server on its own default, so it is only sent when the config names a mode.
        task_type = VOICE_TYPES.get(self.model_type)
        if task_type:
            item["task_type"] = task_type

        # Preset voice name (lookup-style models such as Qwen3-TTS CustomVoice).
        if voice.get("voice"):
            item["voice"] = voice["voice"]

        # Voice description plus emotion/style instruction (design-style models).
        parts = [voice.get("description", ""), chunk.get("instruct", "")]
        instructions = ", ".join(p.strip() for p in parts if p and p.strip())
        if instructions:
            item["instructions"] = instructions

        # Reference audio (clone-style models such as Qwen3-TTS Base, MOSS-TTS-Nano).
        if voice.get("ref_audio"):
            item["ref_audio"] = self._reference_data_url(voice["ref_audio"])
            item["ref_text"] = voice.get("ref_text") or ""

        # "language" is deliberately never sent. It is not an OpenAI spec field, some
        # backends (BreezeTTS2) reject it outright, and the only source for the value
        # is a selector this mode hides anyway. The model infers it from the text.

        if seed is not None and int(seed) >= 0:
            item["seed"] = int(seed)
        return item

    def _request(self, item):
        try:
            response = requests.post(self._url("audio/speech"),
                                     headers=self._headers(),
                                     json={"model": self.model, **item,
                                           "response_format": self.response_format},
                                     timeout=600)
        except requests.exceptions.Timeout as exc:
            # The raw urllib3 message ("Read timed out. (read timeout=600)") does not
            # tell the user which knob to turn, so say it in their terms instead.
            raise RuntimeError(
                "The service did not answer in time. Check the URL is reachable, and "
                "that the model is loaded -- one still loading its weights will not reply."
            ) from exc
        if not response.ok:
            raise RuntimeError(
                f"Service HTTP {response.status_code}: {response.text[:300]}")
        content = response.content
        if not content:
            raise ValueError("Service returned empty audio")
        return self._normalise_audio(content,
                                     response.headers.get("Content-Type", ""))

    @classmethod
    def _normalise_audio(cls, content, content_type=""):
        """Normalise whatever the service returned into a real RIFF/WAV payload.

        The downstream pipeline reads every chunk back with ``AudioSegment.from_wav``
        (see project.py), which parses the RIFF header rather than sniffing the
        container -- so raw MP3 or headerless PCM fails there even though the bytes
        are perfectly good audio. Three cases:

        * WAV            -- already RIFF, pass through untouched.
        * headerless PCM -- prepend a 44-byte RIFF header. Sample rate and channel
          count differ per backend (Gemini 24 kHz, Fish Audio 44.1 kHz, ...), so
          read what the service reports instead of assuming, and fall back to
          24 kHz mono.
        * MP3            -- decode to PCM and re-wrap. The decode itself is
          lossless: whatever quality was lost, the service lost it when it encoded
          the MP3, and re-wrapping can neither undo nor add to that.
        """
        if content[:4] == b"RIFF":
            return content
        if "pcm" in (content_type or "").lower():
            sample_rate, channels = cls._parse_pcm_params(content_type)
            return cls._wrap_pcm_as_wav(content, sample_rate=sample_rate,
                                        channels=channels)
        if cls._looks_like_mp3(content, content_type):
            return cls._mp3_to_wav(content)
        return content

    @staticmethod
    def _looks_like_mp3(content, content_type=""):
        """Identify MP3 by Content-Type first, then by magic bytes.

        Not every service labels its response honestly, so fall back to sniffing:
        either an ID3v2 tag, or a bare MPEG frame sync (the first 11 bits set).
        """
        if any(token in (content_type or "").lower() for token in ("mpeg", "mp3")):
            return True
        if content[:3] == b"ID3":
            return True
        return len(content) >= 2 and content[0] == 0xFF and (content[1] & 0xE0) == 0xE0

    @staticmethod
    def _mp3_to_wav(mp3):
        """Decode an MP3 payload and re-wrap it as WAV."""
        from pydub import AudioSegment

        try:
            segment = AudioSegment.from_file(io.BytesIO(mp3), format="mp3")
        except Exception as exc:
            raise ValueError(
                f"Service returned MP3 but decoding it failed ({exc}). This path needs "
                f"ffmpeg on PATH; requesting response_format 'pcm' or 'wav' avoids it."
            ) from exc
        if len(segment) == 0:
            raise ValueError("Service returned an MP3 payload with no audio in it")
        # pydub always hands back 16-bit PCM, so reuse the PCM wrapper rather than
        # spawning a second ffmpeg just to prepend a 44-byte header.
        return OpenAITTS._wrap_pcm_as_wav(
            segment.raw_data,
            sample_rate=segment.frame_rate,
            channels=segment.channels,
            bits=segment.sample_width * 8,
        )

    @staticmethod
    def _parse_pcm_params(content_type: str) -> tuple[int, int]:
        """Read sample rate and channel count from a Content-Type header.

        Expects the form ``audio/pcm;rate=24000;channels=1``; anything missing
        falls back to 24 kHz mono, the common case for TTS output.
        """
        sample_rate, channels = 24000, 1
        for part in (content_type or "").split(";"):
            key, _, value = part.strip().partition("=")
            key = key.lower()
            if not value:
                continue
            if key == "rate":
                try:
                    sample_rate = int(value)
                except ValueError:
                    pass
            elif key == "channels":
                try:
                    channels = int(value)
                except ValueError:
                    pass
        return sample_rate, channels

    @staticmethod
    def _wrap_pcm_as_wav(pcm: bytes, sample_rate: int = 24000,
                         channels: int = 1, bits: int = 16) -> bytes:
        """Wrap headerless 16-bit PCM in a minimal WAV container."""
        import struct

        byte_rate = sample_rate * channels * bits // 8
        block_align = channels * bits // 8
        header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"
        header += b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels,
                                        sample_rate, byte_rate, block_align, bits)
        header += b"data" + struct.pack("<I", len(pcm))
        return header + pcm

    def generate_one(self, chunk, voices, seed=None):
        return self._request(self._item(chunk, voices, seed))

    def generate_batch(self, chunks, voices, seed=None,
                       on_progress=None, on_result=None):
        results = {}
        items = {}
        done = 0
        failed = 0
        for chunk in chunks:
            try:
                items[chunk["index"]] = self._item(chunk, voices, seed)
            except Exception as exc:
                results[chunk["index"]] = (None, str(exc))
                failed += 1
                if on_result:
                    on_result(chunk["index"], None, str(exc))

        def run(index):
            try:
                return index, (self._request(items[index]), None)
            except Exception as exc:
                return index, (None, str(exc))

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = [pool.submit(run, index) for index in items]
            for future in as_completed(futures):
                index, result = future.result()
                results[index] = result
                if result[0] is not None:
                    done += 1
                else:
                    failed += 1
                if on_result:
                    on_result(index, result[0], result[1])
                if on_progress:
                    on_progress(done, failed, len(chunks))
        return results
