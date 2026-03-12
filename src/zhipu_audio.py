"""更改的技术细节备注:
1. 更改目的: 补充兼容智谱 AI 音频接口的 STT/TTS 适配层, 用于在 LiveKit FallbackAdapter 中把 glm-asr-2512 与 glm-tts 作为备选模型接入.
2. 涉及文件或模块: src/zhipu_audio.py, tests/test_zhipu_audio.py, tests/agent_FallbackAdapter.py.
3. 技术实现: 新增智谱基础 URL 与音频端点推导, TTS 请求体构造, 非流式 ZhipuSTT/ZhipuTTS 适配器, 并按 LiveKit STT/TTS 基类返回统一事件与音频流.
4. 兼容性影响: 新增模块, 不改变现有百炼适配器; 智谱 STT/TTS 默认以非流式能力暴露, 交由 AgentSession 或 FallbackAdapter 负责包装流式链路.
5. 验证方式: cmd /c "call .venv\\Scripts\\activate.bat && pytest tests/test_zhipu_audio.py".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    stt,
    tts,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

DEFAULT_ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_ZHIPU_TRANSCRIPTIONS_PATH = "/audio/transcriptions"
DEFAULT_ZHIPU_SPEECH_PATH = "/audio/speech"

DEFAULT_ASR_MODEL = "glm-asr-2512"
DEFAULT_TTS_MODEL = "glm-tts"
DEFAULT_TTS_VOICE = "tongtong"
DEFAULT_RESPONSE_FORMAT = "wav"
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_NUM_CHANNELS = 1


def _default_http_timeout(connect_timeout: float | None = None) -> httpx.Timeout:
    return httpx.Timeout(
        connect=connect_timeout or 15.0,
        read=60.0,
        write=30.0,
        pool=15.0,
    )


def _normalize_base_url(base_url: str | None) -> str:
    return (base_url or DEFAULT_ZHIPU_BASE_URL).rstrip("/")


def _resolve_audio_endpoint(
    *,
    base_url: str | None,
    explicit_url: str | None,
    path: str,
) -> str:
    if explicit_url:
        return explicit_url.rstrip("/")

    normalized = _normalize_base_url(base_url)
    if normalized.endswith(path):
        return normalized
    return f"{normalized}{path}"


def _build_tts_payload(
    *,
    model: str,
    text: str,
    voice: str,
    response_format: str,
    speed: float | None = None,
    volume: float | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "input": text,
        "voice": voice,
        "response_format": response_format,
    }
    if speed is not None:
        payload["speed"] = speed
    if volume is not None:
        payload["volume"] = volume
    return payload


def _extract_transcript(payload: dict[str, Any]) -> str:
    text = payload.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    data = payload.get("data")
    if isinstance(data, dict):
        nested_text = data.get("text")
        if isinstance(nested_text, str) and nested_text.strip():
            return nested_text.strip()

    raise ValueError("Zhipu STT response did not contain transcript text")


def _raise_status_error(response: httpx.Response) -> None:
    try:
        body = response.json()
    except Exception:
        body = response.text

    raise APIStatusError(
        f"Zhipu request failed with status {response.status_code}",
        status_code=response.status_code,
        request_id=response.headers.get("x-request-id", ""),
        body=body,
    )


@dataclass
class _ZhipuSTTOptions:
    model: str
    language: str
    prompt: str | None


class ZhipuSTT(stt.STT):
    def __init__(
        self,
        *,
        model: str = DEFAULT_ASR_MODEL,
        api_key: str,
        base_url: str = DEFAULT_ZHIPU_BASE_URL,
        transcriptions_url: str | None = None,
        language: str = "zh",
        prompt: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
                aligned_transcript=False,
            )
        )
        self._opts = _ZhipuSTTOptions(
            model=model,
            language=language,
            prompt=prompt,
        )
        self._api_key = api_key
        self._transcriptions_url = _resolve_audio_endpoint(
            base_url=base_url,
            explicit_url=transcriptions_url,
            path=DEFAULT_ZHIPU_TRANSCRIPTIONS_PATH,
        )
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=_default_http_timeout(),
            follow_redirects=True,
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=20,
                keepalive_expiry=120,
            ),
        )

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "zhipu"

    @property
    def transcriptions_url(self) -> str:
        return self._transcriptions_url

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        transcript_language = language if is_given(language) else self._opts.language
        audio_bytes = rtc.combine_audio_frames(buffer).to_wav_bytes()
        headers = {
            "Authorization": f"Bearer {self._api_key}",
        }
        data: dict[str, str] = {
            "model": self._opts.model,
        }
        if self._opts.prompt:
            data["prompt"] = self._opts.prompt

        files = {
            "file": ("audio.wav", audio_bytes, "audio/wav"),
        }

        try:
            response = await self._client.post(
                self._transcriptions_url,
                headers=headers,
                data=data,
                files=files,
                timeout=_default_http_timeout(conn_options.timeout),
            )
            if response.is_error:
                _raise_status_error(response)
        except httpx.TimeoutException:
            raise APITimeoutError() from None
        except APIStatusError:
            raise
        except Exception as exc:
            raise APIConnectionError() from exc

        payload = response.json()
        transcript = _extract_transcript(payload)
        return stt.SpeechEvent(
            request_id=response.headers.get("x-request-id", ""),
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(
                    text=transcript,
                    language=LanguageCode(transcript_language),
                )
            ],
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


@dataclass
class _ZhipuTTSOptions:
    model: str
    voice: str
    response_format: str
    speed: float | None
    volume: float | None


class ZhipuTTS(tts.TTS):
    def __init__(
        self,
        *,
        model: str = DEFAULT_TTS_MODEL,
        voice: str = DEFAULT_TTS_VOICE,
        response_format: str = DEFAULT_RESPONSE_FORMAT,
        speed: float | None = None,
        volume: float | None = None,
        api_key: str,
        base_url: str = DEFAULT_ZHIPU_BASE_URL,
        speech_url: str | None = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        num_channels: int = DEFAULT_NUM_CHANNELS,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False, aligned_transcript=False),
            sample_rate=sample_rate,
            num_channels=num_channels,
        )
        self._opts = _ZhipuTTSOptions(
            model=model,
            voice=voice,
            response_format=response_format,
            speed=speed,
            volume=volume,
        )
        self._api_key = api_key
        self._speech_url = _resolve_audio_endpoint(
            base_url=base_url,
            explicit_url=speech_url,
            path=DEFAULT_ZHIPU_SPEECH_PATH,
        )
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=_default_http_timeout(),
            follow_redirects=True,
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=20,
                keepalive_expiry=120,
            ),
        )

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "zhipu"

    @property
    def voice(self) -> str:
        return self._opts.voice

    @property
    def speech_url(self) -> str:
        return self._speech_url

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        return _ZhipuChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class _ZhipuChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: ZhipuTTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: ZhipuTTS = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        headers = {
            "Authorization": f"Bearer {self._tts._api_key}",
            "Content-Type": "application/json",
        }
        payload = _build_tts_payload(
            model=self._tts.model,
            text=self.input_text,
            voice=self._tts.voice,
            response_format=self._tts._opts.response_format,
            speed=self._tts._opts.speed,
            volume=self._tts._opts.volume,
        )

        try:
            response = await self._tts._client.post(
                self._tts._speech_url,
                headers=headers,
                json=payload,
                timeout=_default_http_timeout(self._conn_options.timeout),
            )
            if response.is_error:
                _raise_status_error(response)
        except httpx.TimeoutException:
            raise APITimeoutError() from None
        except APIStatusError:
            raise
        except Exception as exc:
            raise APIConnectionError() from exc

        output_emitter.initialize(
            request_id=response.headers.get("x-request-id", ""),
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
            mime_type=response.headers.get("content-type", "")
            or f"audio/{self._tts._opts.response_format}",
        )
        output_emitter.push(response.content)
        output_emitter.flush()
