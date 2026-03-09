'''更改的技术细节备注:
1. 更改目的: 集中管理百炼与其他模型服务的环境变量读取逻辑.
2. 涉及文件或模块: env_utils.py.
3. 技术实现: 统一读取 API Key 与 Base URL, 并自动推导百炼 compatible-mode/v1 与 api/v1 两类地址.
4. 兼容性影响: 允许 ALIBABA_* 与 DASHSCOPE_* 两套变量别名共存.
5. 验证方式: pytest tests/test_bailian_audio.py, 导入校验.
'''

import os

from dotenv import load_dotenv

load_dotenv(override=True)


def _first(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _derive_compat_base_url(base_url: str | None) -> str:
    if not base_url:
        return "https://dashscope.aliyuncs.com/compatible-mode/v1"

    normalized = base_url.rstrip("/")
    if normalized.endswith("/api/v1"):
        return f"{normalized[:-7]}/compatible-mode/v1"
    if normalized.endswith("/compatible-mode/v1"):
        return normalized
    return normalized


def _derive_api_base_url(base_url: str | None) -> str:
    if not base_url:
        return "https://dashscope.aliyuncs.com/api/v1"

    normalized = base_url.rstrip("/")
    if normalized.endswith("/compatible-mode/v1"):
        return f"{normalized[:-19]}/api/v1"
    if normalized.endswith("/api/v1"):
        return normalized
    return normalized


OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL')

DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')
DASHSCOPE_BASE_URL = os.getenv('DASHSCOPE_BASE_URL')

MINIMAX_API_KEY = os.getenv('MINIMAX_API_KEY')
MINIMAX_BASE_URL = os.getenv('MINIMAX_BASE_URL')

ALIBABA_API_KEY = _first('ALIBABA_API_KEY', 'DASHSCOPE_API_KEY')
ALIBABA_BASE_URL = _first('ALIBABA_BASE_URL', 'DASHSCOPE_BASE_URL')
ALIBABA_COMPAT_BASE_URL = _derive_compat_base_url(ALIBABA_BASE_URL)
ALIBABA_API_BASE_URL = _derive_api_base_url(ALIBABA_BASE_URL)

DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')

K2_API_KEY = os.getenv('K2_API_KEY')
K2_BASE_URL = os.getenv('K2_BASE_URL')

ZHIPU_API_KEY = os.getenv('ZHIPU_API_KEY')
ZHIPU_BASE_URL = os.getenv('ZHIPU_BASE_URL')
ZHIPU_TRANSCIPTIONS_URL = os.getenv('ZHIPU_TRANSCIPTIONS_URL')

LOCAL_BASE_URL = os.getenv('LOCAL_BASE_URL')
