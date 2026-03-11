'''更改的技术细节备注:
1. 更改目的: 集中管理百炼与其他模型服务的环境变量读取逻辑, 并消除 API Key 与 URL 的 str | None 类型告警.
2. 涉及文件或模块: env_utils.py.
3. 技术实现: 统一区分必填变量与可推导变量; 必填 API Key 与 Base URL 使用 _required_first 收敛为 str; 百炼地址继续通过推导函数生成最终 str.
4. 兼容性影响: 允许 ALIBABA_* 与 DASHSCOPE_* 两套变量别名共存, 但被标记为必填的 URL 若缺失会在导入阶段直接报错.
5. 验证方式: py_compile, ruff check, 导入校验.
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


def _required_first(*names: str) -> str:
    value = _first(*names)
    if value is None:
        joined = ", ".join(names)
        raise RuntimeError(f"Missing required environment variable. Set one of: {joined}")
    return value


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


OPENAI_API_KEY: str = _required_first('OPENAI_API_KEY')
OPENAI_BASE_URL: str = _required_first('OPENAI_BASE_URL')

DASHSCOPE_API_KEY: str = _required_first('DASHSCOPE_API_KEY')
DASHSCOPE_BASE_URL: str = _required_first('DASHSCOPE_BASE_URL')

MINIMAX_API_KEY: str = _required_first('MINIMAX_API_KEY')
MINIMAX_BASE_URL: str = _required_first('MINIMAX_BASE_URL')

ALIBABA_API_KEY: str = _required_first('ALIBABA_API_KEY', 'DASHSCOPE_API_KEY')
ALIBABA_BASE_URL = _first('ALIBABA_BASE_URL', 'DASHSCOPE_BASE_URL')
ALIBABA_COMPAT_BASE_URL = _derive_compat_base_url(ALIBABA_BASE_URL)
ALIBABA_API_BASE_URL = _derive_api_base_url(ALIBABA_BASE_URL)

DEEPSEEK_API_KEY: str = _required_first('DEEPSEEK_API_KEY')

K2_API_KEY: str = _required_first('K2_API_KEY')
K2_BASE_URL: str = _required_first('K2_BASE_URL')

ZHIPU_API_KEY: str = _required_first('ZHIPU_API_KEY')
ZHIPU_BASE_URL: str = _required_first('ZHIPU_BASE_URL')
ZHIPU_TRANSCIPTIONS_URL: str = _required_first('ZHIPU_TRANSCIPTIONS_URL')

# LOCAL_BASE_URL: str = _required_first('LOCAL_BASE_URL')
