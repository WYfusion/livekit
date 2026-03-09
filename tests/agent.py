'''更改的技术细节备注:
1. 更改目的: 将测试入口接入阿里百炼模型, 并支持直接以脚本方式运行.
2. 涉及文件或模块: tests/agent.py, env_utils.py, src/bailian_audio.py.
3. 技术实现: 在文件顶部注入项目根目录到 sys.path; LLM 使用百炼 OpenAI 兼容接口; STT 与 TTS 使用自定义 DashScope 适配器.
4. 兼容性影响: 保留 LiveKit AgentSession 入口形式, 但运行依赖百炼相关环境变量存在.
5. 验证方式: python --help, py_compile, ruff check.
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

from env_utils import (
    ALIBABA_API_BASE_URL,
    ALIBABA_API_KEY,
    ALIBABA_COMPAT_BASE_URL,
)
from src.bailian_audio import DashScopeSTT, DashScopeTTS


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful voice AI assistant.",
        )


async def entrypoint(ctx: JobContext):
    session = AgentSession(
        # Bailian ASR and TTS use their native APIs rather than the LiveKit OpenAI
        # STT/TTS plugin interfaces, so they need custom adapters here.
        stt=DashScopeSTT(
            model="qwen3-asr-flash",
            base_url=ALIBABA_COMPAT_BASE_URL,
            api_key=ALIBABA_API_KEY,
            language="zh",
        ),
        llm=openai.LLM(
            model="qwen3.5-plus-2026-02-15",
            base_url=ALIBABA_COMPAT_BASE_URL,
            api_key=ALIBABA_API_KEY,
        ),
        tts=DashScopeTTS(
            model="qwen-tts",
            voice="Cherry",
            base_url=ALIBABA_API_BASE_URL,
            api_key=ALIBABA_API_KEY,
        ),
        vad=silero.VAD.load(),
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
