"""更改的技术细节备注:
1. 更改目的: 将示例脚本改为真实使用 LiveKit FallbackAdapter, 在百炼主链路之外补上兼容智谱 AI 的 GLM-4.7、glm-tts 与 glm-asr-2512 备链路.
2. 涉及文件或模块: tests/agent_FallbackAdapter.py, src/bailian_audio.py, src/zhipu_audio.py.
3. 技术实现: 使用 shared_vad 同时服务 session 与 STT fallback; 百炼配置存在时优先使用 DashScopeSTT/DashScopeTTS 与 Qwen LLM, 智谱始终作为备选 provider 加入 llm/stt/tts.FallbackAdapter.
4. 兼容性影响: 示例脚本现在依赖 ZHIPU_API_KEY; 若未配置百炼环境变量, FallbackAdapter 会退化为仅智谱单链路运行.
5. 验证方式: cmd /c "call .venv\\Scripts\\activate.bat && python -m py_compile tests/agent_FallbackAdapter.py".
"""

# ruff: noqa: E402

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RoomInputOptions,
    WorkerOptions,
    cli,
    llm,
    stt,
    tts,
)
from livekit.plugins import noise_cancellation, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from src.bailian_audio import DashScopeSTT, DashScopeTTS
from src.zhipu_audio import DEFAULT_ZHIPU_BASE_URL, ZhipuSTT, ZhipuTTS

load_dotenv(".env", override=True)


def _first(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip().strip('"')
    return None


def _required(name: str) -> str:
    value = _first(name)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _derive_compat_base_url(base_url: str | None) -> str:
    normalized = (
        base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ).rstrip("/")
    if normalized.endswith("/api/v1"):
        return f"{normalized[:-7]}/compatible-mode/v1"
    if normalized.endswith("/compatible-mode/v1"):
        return normalized
    return normalized


def _derive_api_base_url(base_url: str | None) -> str:
    normalized = (base_url or "https://dashscope.aliyuncs.com/api/v1").rstrip("/")
    if normalized.endswith("/compatible-mode/v1"):
        return f"{normalized[:-19]}/api/v1"
    if normalized.endswith("/api/v1"):
        return normalized
    return normalized


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="你是一个很有用的语音 AI 对话助手, 要求回答简洁.",
        )


async def entrypoint(ctx: JobContext):
    shared_vad = silero.VAD.load()
    alibaba_api_key = _first("ALIBABA_API_KEY", "DASHSCOPE_API_KEY")
    alibaba_base_url = _first("ALIBABA_BASE_URL", "DASHSCOPE_BASE_URL")
    zhipu_api_key = _required("ZHIPU_API_KEY")
    zhipu_base_url = (_first("ZHIPU_BASE_URL") or DEFAULT_ZHIPU_BASE_URL).rstrip("/")
    zhipu_transcriptions_url = _first(
        "ZHIPU_TRANSCRIPTIONS_URL",
        "ZHIPU_TRANSCIPTIONS_URL",
    )

    llm_providers: list[llm.LLM] = []
    stt_providers: list[stt.STT] = []
    tts_providers: list[tts.TTS] = []

    if alibaba_api_key and alibaba_base_url:
        compat_base_url = _derive_compat_base_url(alibaba_base_url)
        api_base_url = _derive_api_base_url(alibaba_base_url)
        llm_providers.append(
            openai.LLM(
                model=os.getenv("ALIBABA_LLM_MODEL", "qwen3.5-plus-2026-02-15"),
                base_url=compat_base_url,
                api_key=alibaba_api_key,
            )
        )
        stt_providers.append(
            DashScopeSTT(
                model=os.getenv("ALIBABA_STT_MODEL", "qwen3-asr-flash"),
                base_url=compat_base_url,
                api_key=alibaba_api_key,
                language="zh",
                streaming=True,
                stream_vad=shared_vad,
            )
        )
        tts_providers.append(
            DashScopeTTS(
                model=os.getenv("ALIBABA_TTS_MODEL", "qwen-tts"),
                voice=os.getenv("ALIBABA_TTS_VOICE", "Ethan"),
                language_type=os.getenv("ALIBABA_TTS_LANGUAGE_TYPE", "Chinese"),
                base_url=api_base_url,
                api_key=alibaba_api_key,
                streaming=True,
            )
        )

    llm_providers.append(
        openai.LLM(
            model=os.getenv("ZHIPU_LLM_MODEL", "glm-4.7"),
            base_url=zhipu_base_url,
            api_key=zhipu_api_key,
        )
    )
    stt_providers.append(
        ZhipuSTT(
            model=os.getenv("ZHIPU_STT_MODEL", "glm-asr-2512"),
            api_key=zhipu_api_key,
            base_url=zhipu_base_url,
            transcriptions_url=zhipu_transcriptions_url,
            language="zh",
        )
    )
    tts_providers.append(
        ZhipuTTS(
            model=os.getenv("ZHIPU_TTS_MODEL", "glm-tts"),
            voice=os.getenv("ZHIPU_TTS_VOICE", "tongtong"),
            response_format=os.getenv("ZHIPU_TTS_FORMAT", "wav"),
            api_key=zhipu_api_key,
            base_url=zhipu_base_url,
        )
    )

    session = AgentSession(
        stt=stt.FallbackAdapter(
            stt_providers,
            vad=shared_vad,
        ),
        llm=llm.FallbackAdapter(
            llm_providers,
        ),
        tts=tts.FallbackAdapter(
            tts_providers,
        ),
        vad=shared_vad,
        turn_detection=MultilingualModel(),
    )

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
