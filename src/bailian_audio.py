'''更改的技术细节备注:
1. 更改目的: 修正百炼 TTS 请求体结构, 解决自定义音色被静默回退为默认音色的问题, 并保留可选流式接口开关.
2. 涉及文件或模块: src/bailian_audio.py.
3. 技术实现: DashScopeSTT 在 streaming=True 时要求传入 stream_vad, 并委托 LiveKit 的 stt.StreamAdapter 暴露流式识别; DashScopeTTS 修正为按官方 API 将 voice 放入 input, 并支持显式 language_type; 开启 streaming=True 时仍委托 LiveKit 的 tts.StreamAdapter 暴露流式合成.
4. 兼容性影响: 默认构造参数保持非流式行为; 开启流式后属于适配器包装的流式接口, 不等同于直接接入百炼原生实时协议.
5. 验证方式: pytest tests/test_bailian_audio.py, ruff check, py_compile.
'''

from __future__ import annotations

import base64
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import httpx
import openai
from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    stt,
    tts,
    vad,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

DEFAULT_COMPAT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_API_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"
DEFAULT_ASR_MODEL = "qwen3-asr-flash"
DEFAULT_TTS_MODEL = "qwen-tts"
DEFAULT_TTS_VOICE = "Cherry"
DEFAULT_TTS_FORMAT = "wav"
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_NUM_CHANNELS = 1
DEFAULT_TTS_ENDPOINT = "/services/aigc/multimodal-generation/generation"


def _default_http_timeout(connect_timeout: float | None = None) -> httpx.Timeout:
    return httpx.Timeout(
        connect=connect_timeout or 15.0,
        read=60.0,
        write=30.0,
        pool=15.0,
    )


def _normalize_base_url(base_url: str | None, default: str) -> str:
    if not base_url:
        return default
    return base_url.rstrip("/")


def _extract_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()

    if not isinstance(content, Sequence):
        raise ValueError(f"Unsupported completion content type: {type(content)!r}")

    chunks: list[str] = []
    for item in content:
        if isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str):
                chunks.append(text)
            continue

        text = getattr(item, "text", None)
        if isinstance(text, str):
            chunks.append(text)

    transcript = "\n".join(chunk.strip() for chunk in chunks if chunk.strip()).strip()
    if transcript:
        return transcript

    raise ValueError("Completion response did not contain transcript text")


def _build_audio_data_uri(audio_b64: str, *, mime_type: str = "audio/wav") -> str:
    return f"data:{mime_type};base64,{audio_b64}"


def _build_asr_messages(
    *,
    audio_data_uri: str,
) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_data_uri,
                        "format": "wav",
                    },
                },
            ],
        }
    ]


def _build_asr_extra_body(
    *,
    language: str | None,
    prompt: str | None = None,
) -> dict[str, Any]:
    asr_options: dict[str, Any] = {
        "enable_itn": False,
    }
    if language:
        asr_options["language"] = language
    if prompt:
        asr_options["prompt"] = prompt

    return {"asr_options": asr_options}


def _build_tts_payload(
    *,
    model: str,
    text: str,
    voice: str,
    response_format: str,
    sample_rate: int,
    language_type: str | None = None,
) -> dict[str, Any]:
    input_payload: dict[str, Any] = {
        "text": text,
        "voice": voice,
    }
    if language_type:
        input_payload["language_type"] = language_type

    return {
        "model": model,
        "input": input_payload,
        "parameters": {
            "response_format": response_format,
            "sample_rate": sample_rate,
        },
    }


def _extract_tts_result(payload: dict[str, Any]) -> tuple[str, str | None]:
    output = payload.get("output")
    if not isinstance(output, dict):
        raise ValueError("DashScope TTS response is missing output")

    audio = output.get("audio")
    if not isinstance(audio, dict):
        raise ValueError("DashScope TTS response is missing output.audio")

    audio_url = audio.get("url")
    if not isinstance(audio_url, str) or not audio_url:
        raise ValueError("DashScope TTS response is missing output.audio.url")

    audio_format = audio.get("format")
    if isinstance(audio_format, str) and audio_format:
        return audio_url, f"audio/{audio_format.lower()}"

    return audio_url, None


def _raise_status_error(response: httpx.Response) -> None:
    try:
        body = response.json()
    except Exception:
        body = response.text

    raise APIStatusError(
        f"DashScope request failed with status {response.status_code}",
        status_code=response.status_code,
        request_id=response.headers.get("x-request-id", ""),
        body=body,
    )


@dataclass
class _DashScopeSTTOptions:
    model: str
    language: str
    prompt: str | None
    streaming: bool


class DashScopeSTT(stt.STT):
    def __init__(
        self,
        *,
        model: str = DEFAULT_ASR_MODEL,
        api_key: str,
        base_url: str = DEFAULT_COMPAT_BASE_URL,
        language: str = "zh",
        prompt: str | None = None,
        streaming: bool = False,
        stream_vad: vad.VAD | None = None,
        client: openai.AsyncClient | None = None,
    ) -> None:
        if streaming and stream_vad is None:
            raise ValueError("stream_vad is required when streaming=True")

        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=streaming,
                interim_results=False,
                aligned_transcript=False,
            )
        )
        self._opts = _DashScopeSTTOptions(
            model=model,
            language=language,
            prompt=prompt,
            streaming=streaming,
        )
        self._stream_vad = stream_vad
        self._stream_adapter: stt.StreamAdapter | None = None
        self._owns_client = client is None
        self._client = client or openai.AsyncClient(
            api_key=api_key,
            base_url=_normalize_base_url(base_url, DEFAULT_COMPAT_BASE_URL),
            max_retries=0,
            http_client=httpx.AsyncClient(
                timeout=_default_http_timeout(),
                follow_redirects=True,
                limits=httpx.Limits(
                    max_connections=20,
                    max_keepalive_connections=20,
                    keepalive_expiry=120,
                ),
            ),
        )

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "dashscope"

    def _ensure_stream_adapter(self) -> stt.StreamAdapter:
        if self._stream_adapter is None:
            if self._stream_vad is None:
                raise ValueError("stream_vad is required when streaming=True")
            self._stream_adapter = stt.StreamAdapter(stt=self, vad=self._stream_vad)
        return self._stream_adapter

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        lang = language if is_given(language) else self._opts.language
        audio_bytes = rtc.combine_audio_frames(buffer).to_wav_bytes()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        audio_data_uri = _build_audio_data_uri(audio_b64)

        try:
            response = await self._client.chat.completions.create(
                model=self._opts.model,
                messages=_build_asr_messages(
                    audio_data_uri=audio_data_uri,
                ),
                extra_body=_build_asr_extra_body(
                    language=lang,
                    prompt=self._opts.prompt,
                ),
                timeout=_default_http_timeout(conn_options.timeout),
            )
        except openai.APITimeoutError:
            raise APITimeoutError() from None
        except openai.APIStatusError as e:
            raise APIStatusError(
                e.message,
                status_code=e.status_code,
                request_id=e.request_id,
                body=e.body,
            ) from None
        except Exception as e:
            raise APIConnectionError() from e

        if not response.choices:
            raise APIConnectionError("DashScope STT returned no choices")

        transcript = _extract_message_text(response.choices[0].message.content)
        return stt.SpeechEvent(
            request_id=response.id or "",
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(
                    text=transcript,
                    language=LanguageCode(lang or self._opts.language or "zh"),
                )
            ],
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.RecognizeStream:
        if not self.capabilities.streaming:
            return super().stream(language=language, conn_options=conn_options)
        return self._ensure_stream_adapter().stream(
            language=language,
            conn_options=conn_options,
        )

    async def aclose(self) -> None:
        if self._stream_adapter is not None:
            await self._stream_adapter.aclose()
        if self._owns_client:
            await self._client.close()


@dataclass
class _DashScopeTTSOptions:
    model: str
    voice: str
    language_type: str | None
    response_format: str
    endpoint_path: str
    streaming: bool


class DashScopeTTS(tts.TTS):
    def __init__(
        self,
        *,
        model: str = DEFAULT_TTS_MODEL,
        voice: str = DEFAULT_TTS_VOICE,
        language_type: str | None = None,
        response_format: str = DEFAULT_TTS_FORMAT,
        api_key: str,
        base_url: str = DEFAULT_API_BASE_URL,
        endpoint_path: str = DEFAULT_TTS_ENDPOINT,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        num_channels: int = DEFAULT_NUM_CHANNELS,
        streaming: bool = False,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=streaming,
                aligned_transcript=streaming,
            ),
            sample_rate=sample_rate,
            num_channels=num_channels,
        )
        self._opts = _DashScopeTTSOptions(
            model=model,
            voice=voice,
            language_type=language_type,
            response_format=response_format,
            endpoint_path=endpoint_path,
            streaming=streaming,
        )
        self._stream_adapter: tts.StreamAdapter | None = None
        self._base_url = _normalize_base_url(base_url, DEFAULT_API_BASE_URL)
        self._api_key = api_key
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
        return "dashscope"

    @property
    def voice(self) -> str:
        return self._opts.voice

    @property
    def response_format(self) -> str:
        return self._opts.response_format

    def _ensure_stream_adapter(self) -> tts.StreamAdapter:
        if self._stream_adapter is None:
            self._stream_adapter = tts.StreamAdapter(tts=self)
        return self._stream_adapter

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        return _DashScopeChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
        )

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.SynthesizeStream:
        if not self.capabilities.streaming:
            return super().stream(conn_options=conn_options)
        return self._ensure_stream_adapter().stream(conn_options=conn_options)

    async def aclose(self) -> None:
        if self._stream_adapter is not None:
            await self._stream_adapter.aclose()
        if self._owns_client:
            await self._client.aclose()


class _DashScopeChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: DashScopeTTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: DashScopeTTS = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        headers = {
            "Authorization": f"Bearer {self._tts._api_key}",
            "Content-Type": "application/json",
        }
        payload = _build_tts_payload(
            model=self._tts.model,
            text=self.input_text,
            voice=self._tts.voice,
            response_format=self._tts.response_format,
            sample_rate=self._tts.sample_rate,
            language_type=self._tts._opts.language_type,
        )
        endpoint = f"{self._tts._base_url}{self._tts._opts.endpoint_path}"

        try:
            response = await self._tts._client.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=_default_http_timeout(self._conn_options.timeout),
            )
            if response.is_error:
                _raise_status_error(response)

            data = response.json()
            request_id = response.headers.get("x-request-id", "")
            audio_url, mime_type = _extract_tts_result(data)

            audio_response = await self._tts._client.get(
                audio_url,
                timeout=_default_http_timeout(self._conn_options.timeout),
            )
            if audio_response.is_error:
                _raise_status_error(audio_response)
        except httpx.TimeoutException:
            raise APITimeoutError() from None
        except APIStatusError:
            raise
        except Exception as e:
            raise APIConnectionError() from e

        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
            mime_type=audio_response.headers.get("content-type", "")
            or mime_type
            or f"audio/{self._tts.response_format}",
        )
        output_emitter.push(audio_response.content)
        output_emitter.flush()
