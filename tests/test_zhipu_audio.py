"""更改的技术细节备注:
1. 更改目的: 为智谱 AI 音频适配层补充测试, 先约束 FallbackAdapter 场景下需要的 URL 推导, 请求体结构与适配器默认能力.
2. 涉及文件或模块: tests/test_zhipu_audio.py, src/zhipu_audio.py.
3. 技术实现: 新增对基础 URL 标准化, 语音合成请求体, 显式转写地址覆盖, 以及 ZhipuSTT/ZhipuTTS 默认能力和模型暴露的测试.
4. 兼容性影响: 仅影响测试层, 不改变现有运行时行为.
5. 验证方式: cmd /c "call .venv\\Scripts\\activate.bat && pytest tests/test_zhipu_audio.py".
"""

# ruff: noqa: E402

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.zhipu_audio import (
    DEFAULT_ZHIPU_BASE_URL,
    ZhipuSTT,
    ZhipuTTS,
    _build_tts_payload,
    _resolve_audio_endpoint,
)


def test_resolve_audio_endpoint_from_base_url() -> None:
    assert (
        _resolve_audio_endpoint(
            base_url="https://open.bigmodel.cn/api/paas/v4/",
            explicit_url=None,
            path="/audio/speech",
        )
        == "https://open.bigmodel.cn/api/paas/v4/audio/speech"
    )


def test_resolve_audio_endpoint_prefers_explicit_url() -> None:
    assert (
        _resolve_audio_endpoint(
            base_url=DEFAULT_ZHIPU_BASE_URL,
            explicit_url="https://open.bigmodel.cn/api/paas/v4/audio/transcriptions",
            path="/audio/transcriptions",
        )
        == "https://open.bigmodel.cn/api/paas/v4/audio/transcriptions"
    )


def test_build_tts_payload_matches_zhipu_docs() -> None:
    payload = _build_tts_payload(
        model="glm-tts",
        text="你好,今天天气怎么样.",
        voice="tongtong",
        response_format="wav",
    )

    assert payload == {
        "model": "glm-tts",
        "input": "你好,今天天气怎么样.",
        "voice": "tongtong",
        "response_format": "wav",
    }


def test_build_tts_payload_keeps_optional_speed_and_volume() -> None:
    payload = _build_tts_payload(
        model="glm-tts",
        text="hello",
        voice="female",
        response_format="pcm",
        speed=1.1,
        volume=0.9,
    )

    assert payload["speed"] == 1.1
    assert payload["volume"] == 0.9


def test_zhipu_stt_instance_exposes_model_and_non_streaming_capability() -> None:
    provider = ZhipuSTT(
        model="glm-asr-2512",
        api_key="test-key",
        base_url="https://open.bigmodel.cn/api/paas/v4/",
    )

    assert provider.model == "glm-asr-2512"
    assert provider.capabilities.streaming is False
    assert (
        provider.transcriptions_url
        == "https://open.bigmodel.cn/api/paas/v4/audio/transcriptions"
    )


def test_zhipu_tts_instance_exposes_model_and_non_streaming_capability() -> None:
    provider = ZhipuTTS(
        model="glm-tts",
        voice="tongtong",
        api_key="test-key",
        base_url="https://open.bigmodel.cn/api/paas/v4/",
    )

    assert provider.model == "glm-tts"
    assert provider.voice == "tongtong"
    assert provider.capabilities.streaming is False
    assert provider.speech_url == "https://open.bigmodel.cn/api/paas/v4/audio/speech"
