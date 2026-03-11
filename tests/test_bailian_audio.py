'''更改的技术细节备注:
1. 更改目的: 为百炼音频适配层补充流式开关的回归测试, 约束适配器包装行为.
2. 涉及文件或模块: tests/test_bailian_audio.py, src/bailian_audio.py, env_utils.py.
3. 技术实现: 继续覆盖 data URI, ASR/TTS 请求结构, 并新增 STT/TTS 在 streaming=True 时委托 LiveKit StreamAdapter 的测试.
4. 兼容性影响: 仅影响测试层; 运行时默认仍保持非流式行为, 流式模式需显式开启.
5. 验证方式: pytest tests/test_bailian_audio.py.
'''

from types import SimpleNamespace
from unittest.mock import AsyncMock

from env_utils import _derive_api_base_url, _derive_compat_base_url
from src.bailian_audio import (
    DashScopeSTT,
    DashScopeTTS,
    _build_asr_extra_body,
    _build_asr_messages,
    _build_audio_data_uri,
    _build_tts_payload,
    _extract_message_text,
    _extract_tts_result,
)


def test_extract_message_text_from_string() -> None:
    assert _extract_message_text("  hello, world  ") == "hello, world"


def test_extract_message_text_from_chunks() -> None:
    content = [
        {"type": "output_text", "text": "你好"},
        SimpleNamespace(text="世界"),
    ]

    assert _extract_message_text(content) == "你好\n世界"


def test_build_asr_messages_includes_audio_and_instruction() -> None:
    messages = _build_asr_messages(
        audio_data_uri="data:audio/wav;base64,abc123",
    )

    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert content[0]["type"] == "input_audio"
    assert content[0]["input_audio"]["data"] == "data:audio/wav;base64,abc123"
    assert len(content) == 1


def test_build_audio_data_uri_adds_required_prefix() -> None:
    assert _build_audio_data_uri("YWJj", mime_type="audio/wav") == "data:audio/wav;base64,YWJj"


def test_build_asr_extra_body_uses_language_hint() -> None:
    assert _build_asr_extra_body(language="zh") == {
        "asr_options": {
            "language": "zh",
            "enable_itn": False,
        }
    }


def test_build_tts_payload_contains_voice_and_format() -> None:
    payload = _build_tts_payload(
        model="qwen-tts",
        text="你好",
        voice="Cherry",
        response_format="wav",
        sample_rate=24000,
    )

    assert payload == {
        "model": "qwen-tts",
        "input": {
            "text": "你好",
            "voice": "Cherry",
        },
        "parameters": {
            "response_format": "wav",
            "sample_rate": 24000,
        },
    }


def test_build_tts_payload_includes_optional_language_type() -> None:
    payload = _build_tts_payload(
        model="qwen-tts",
        text="hello",
        voice="Ethan",
        response_format="wav",
        sample_rate=24000,
        language_type="Chinese",
    )

    assert payload["input"]["language_type"] == "Chinese"


def test_extract_tts_result_reads_nested_output() -> None:
    audio_url, mime_type = _extract_tts_result(
        {
            "output": {
                "audio": {
                    "url": "https://example.com/audio.wav",
                    "format": "wav",
                }
            }
        }
    )

    assert audio_url == "https://example.com/audio.wav"
    assert mime_type == "audio/wav"


def test_env_utils_derives_both_dashscope_base_urls() -> None:
    compat = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api = "https://dashscope.aliyuncs.com/api/v1"

    assert _derive_api_base_url(compat) == api
    assert _derive_compat_base_url(api) == compat


def test_dashscope_adapter_instances_expose_models() -> None:
    dashscope_stt = DashScopeSTT(
        model="qwen3-asr-flash",
        api_key="test-key",
        base_url="https://example.com/compatible-mode/v1",
    )
    dashscope_tts = DashScopeTTS(
        model="qwen-tts",
        voice="Cherry",
        api_key="test-key",
        base_url="https://example.com/api/v1",
    )

    assert dashscope_stt.model == "qwen3-asr-flash"
    assert dashscope_stt.capabilities.streaming is False
    assert dashscope_tts.model == "qwen-tts"
    assert dashscope_tts.voice == "Cherry"
    assert dashscope_tts.capabilities.streaming is False


def test_dashscope_stt_requires_vad_when_streaming_enabled() -> None:
    try:
        DashScopeSTT(
            model="qwen3-asr-flash",
            api_key="test-key",
            base_url="https://example.com/compatible-mode/v1",
            streaming=True,
        )
    except ValueError as exc:
        assert "stream_vad" in str(exc)
    else:
        raise AssertionError("Expected ValueError when stream_vad is missing")


def test_dashscope_stt_stream_uses_livekit_stream_adapter(monkeypatch) -> None:
    stream_vad = object()
    adapter_calls: list[tuple[object, object]] = []
    expected_stream = object()

    class FakeStreamAdapter:
        def __init__(self, *, stt, vad):
            adapter_calls.append((stt, vad))
            self.aclose = AsyncMock()

        def stream(self, *, language=None, conn_options=None):
            return expected_stream

    monkeypatch.setattr("src.bailian_audio.stt.StreamAdapter", FakeStreamAdapter)

    dashscope_stt = DashScopeSTT(
        model="qwen3-asr-flash",
        api_key="test-key",
        base_url="https://example.com/compatible-mode/v1",
        streaming=True,
        stream_vad=stream_vad,
    )

    assert dashscope_stt.capabilities.streaming is True
    assert dashscope_stt.stream() is expected_stream
    assert adapter_calls == [(dashscope_stt, stream_vad)]


def test_dashscope_tts_stream_uses_livekit_stream_adapter(monkeypatch) -> None:
    adapter_calls: list[object] = []
    expected_stream = object()

    class FakeStreamAdapter:
        def __init__(self, *, tts):
            adapter_calls.append(tts)
            self.aclose = AsyncMock()

        def stream(self, *, conn_options=None):
            return expected_stream

    monkeypatch.setattr("src.bailian_audio.tts.StreamAdapter", FakeStreamAdapter)

    dashscope_tts = DashScopeTTS(
        model="qwen-tts",
        voice="Cherry",
        api_key="test-key",
        base_url="https://example.com/api/v1",
        streaming=True,
    )

    assert dashscope_tts.capabilities.streaming is True
    assert dashscope_tts.stream() is expected_stream
    assert adapter_calls == [dashscope_tts]
