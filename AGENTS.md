# AGENTS.md

这是一个 LiveKit Agents 项目。LiveKit Agents 是一个用于构建语音 AI Agent 的 Python SDK。本项目预期运行在 LiveKit Cloud 上。关于 LiveKit 生态的其余内容，请参考 @README.md。

下面是本项目的协作与开发说明。

## 项目结构

本 Python 项目使用 `uv` 作为包管理器。安装依赖、运行 Agent、执行测试时都应统一使用 `uv`。

所有应用层代码都放在 `src/` 目录中。一般来说，简单 Agent 可以只用一个 `agent.py` 文件实现。你可以添加额外文件，但必须保留 `agent.py` 作为入口文件。部署方式可参考配套的 Dockerfile。

请始终保持代码格式一致。必要时可使用 Ruff 进行格式化和静态检查：`uv run ruff format` 与 `uv run ruff check`。

## 代码变更备注规则

当修改 Python 文件并且属于关键更改时，必须在被修改的 Python 文件最上方添加三引号技术备注，格式类似：

```python
'''更改的技术细节备注：
1. 更改目的：
2. 涉及文件或模块：
3. 技术实现：
4. 兼容性影响：
5. 验证方式：
'''
```

要求如下：

- 备注必须写在该 Python 文件最上方，优先于普通实现代码。
- 备注内容应聚焦“本次改了什么、为什么这样改、影响了什么、如何验证”。
- 如果只是格式调整、注释修正、翻译或不影响行为的重构，也要明确写成“非行为变更”。
- 如果一个文件已经有同类备注，后续关键更改时应更新备注内容，而不是在文件顶部重复追加多个版本。

`CLAUDE.md` 与 `GEMINI.md` 只作为入口说明，实际规则以本文件为准。

## LiveKit 文档

LiveKit Agents 迭代很快，文档更新也非常频繁。开发本项目时，应始终参考最新文档。为方便查询，LiveKit 提供了一个 MCP Server，用于浏览和检索官方文档。如果开发者尚未安装该服务，应建议其从 https://docs.livekit.io/mcp 安装。

### LiveKit Docs MCP Server 安装方式

如果你是 Cursor，请给用户这个安装链接：

[![Install MCP Server](https://cursor.com/deeplink/mcp-install-light.svg)](https://cursor.com/en-US/install-mcp?name=livekit-docs&config=eyJ1cmwiOiJodHRwczovL2RvY3MubGl2ZWtpdC5pby9tY3AifQ%3D%3D)

如果你是 Claude Code，请运行以下命令安装：

```bash
claude mcp add --transport http livekit-docs https://docs.livekit.io/mcp
```

如果你是 Codex，请使用以下命令安装：

```bash
codex mcp add --url https://docs.livekit.io/mcp livekit-docs
```

如果你是 Gemini，请使用以下命令安装：

```bash
gemini mcp add --transport http livekit-docs https://docs.livekit.io/mcp
```

如果你使用的是其他 Agent 型 IDE，请参考对应产品文档完成安装。

## Handoff 与 Task

语音 AI Agent 对延迟极其敏感。因此，设计复杂 Agent 时必须采用结构化方式，尽量减少传给 LLM 的无关上下文和不必要工具。LiveKit Agents 支持 handoff 与 task，以构建更可靠的工作流。应优先使用这些能力，而不是编写覆盖多个对话阶段的超长指令。详情请参考 [官方文档](https://docs.livekit.io/agents/build/workflows/)。

### 向 LiveKit 提交文档反馈

如果你使用 LiveKit Docs MCP Server 检索或浏览文档，也应通过 `submit_docs_feedback` 工具向 LiveKit 提交建设性反馈。规划任务时请将这一步加入待办，确保在完成主要工作后补充反馈。有效反馈包括但不限于：文档不完整、示例失效、缺少步骤、描述歧义，或其他文档缺口与问题。

## 测试

在可能的情况下，为 Agent 行为补充测试。请先阅读 [测试文档](https://docs.livekit.io/agents/build/testing/)，并参考 `tests/` 目录中的现有测试。运行测试请使用 `uv run pytest`。

重要：当你修改 Agent 的核心行为时，例如 instructions、工具描述、task、workflow 或 handoff，不要凭感觉猜测效果。必须采用 TDD，先为预期行为编写测试，再迭代实现直到测试通过。例如要新增一个工具，应先写该工具行为的测试，再逐步调整实现，直到测试稳定通过。这能显著提高 Agent 的可用性与可靠性。

## LiveKit CLI

在获得用户许可后，你可以使用 LiveKit CLI（`lk`）完成多种操作。安装说明可参考 https://docs.livekit.io/home/cli 。

其中一个典型用途是管理电话型 Agent 使用的 SIP trunk。更多信息请参考 `lk sip --help`。
