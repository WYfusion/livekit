'''更改的技术细节备注:
1. 更改目的: 为百炼音频适配层补充最小回归测试, 防止请求结构回退.
2. 涉及文件或模块: tests/test_bailian_audio.py, src/bailian_audio.py, env_utils.py.
3. 技术实现: 覆盖音频 data URI, ASR extra_body, TTS payload, URL 推导与适配器实例化行为.
4. 兼容性影响: 仅影响测试层, 不改变运行时行为.
5. 验证方式: pytest tests/test_bailian_audio.py.
'''

from types import SimpleNamespace

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
    )

    assert payload == {
        "model": "qwen-tts",
        "input": {"text": "你好"},
        "parameters": {
            "voice": "Cherry",
            "format": "wav",
        },
    }


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
    stt = DashScopeSTT(
        model="qwen3-asr-flash",
        api_key="test-key",
        base_url="https://example.com/compatible-mode/v1",
    )
    tts = DashScopeTTS(
        model="qwen-tts",
        voice="Cherry",
        api_key="test-key",
        base_url="https://example.com/api/v1",
    )

    assert stt.model == "qwen3-asr-flash"
    assert stt.capabilities.streaming is False
    assert tts.model == "qwen-tts"
    assert tts.voice == "Cherry"
