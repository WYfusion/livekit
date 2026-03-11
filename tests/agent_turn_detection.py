'''更改的技术细节备注:
1. 更改目的: 演示百炼 STT/TTS 适配器在 AgentSession 中显式开启流式接口, 并正确指定 TTS 音色与语言类型后的接入方式.
2. 涉及文件或模块: tests/agent_turn_detection.py, src/bailian_audio.py, env_utils.py.
3. 技术实现: 复用同一个 Silero VAD 同时作为 session 级 VAD 与 DashScopeSTT 的 stream_vad, 并为 DashScopeTTS 显式传入 Ethan 音色, Chinese 语言类型与 streaming=True.
4. 兼容性影响: 仅调整示例脚本; 运行时仍依赖现有百炼环境变量与 LiveKit turn detector 配置.
5. 验证方式: py_compile, 手动运行 tests/agent_turn_detection.py.
'''

# ruff: noqa: E402

import sys
from pathlib import Path

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
)
from livekit.plugins import noise_cancellation, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from env_utils import (
    ALIBABA_API_BASE_URL,
    ALIBABA_API_KEY,
    ALIBABA_COMPAT_BASE_URL,
)
from src.bailian_audio import DashScopeSTT, DashScopeTTS


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="你是一个很有用的语音 AI 对话助手, 要求回答简洁.",
        )


async def entrypoint(ctx: JobContext):
    shared_vad = silero.VAD.load()

    session = AgentSession(
        stt=DashScopeSTT(
            model="qwen3-asr-flash",
            base_url=ALIBABA_COMPAT_BASE_URL,
            api_key=ALIBABA_API_KEY,
            language="zh",
            streaming=True,
            stream_vad=shared_vad,
        ),
        llm=openai.LLM(
            model="qwen3.5-plus-2026-02-15",
            base_url=ALIBABA_COMPAT_BASE_URL,
            api_key=ALIBABA_API_KEY,
        ),
        tts=DashScopeTTS(
            model="qwen-tts",
            voice="Ethan",
            language_type="Chinese",
            # Voice list: https://bailian.console.alibabacloud.com/cn-beijing?tab=doc#/doc/?type=model&url=2879134
            base_url=ALIBABA_API_BASE_URL,
            api_key=ALIBABA_API_KEY,
            streaming=True,
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
