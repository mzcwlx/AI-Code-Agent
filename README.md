# AI-Code-Agent

面向软件开发的**智能代码缺陷诊断与自动修复 Agent**。

用户提供一个有 Bug 的 Python 项目和 Bug 描述，Agent 自主完成：
**分析 Bug → Code RAG 检索定位 → 读取真实代码 → 创建回归测试 → 确认 Bug 可复现（baseline FAIL）→ 修改真实源码 → 语法检查 → 运行项目测试 → 最终回归验证 → 输出修复结果**。

全程无人工干预，修复结果只认真实工具返回值，不认 LLM 的自述。

---

## 工作流状态机

Agent Runtime 以 6 阶段状态机运行，每个阶段只允许使用当前阶段的工具：

```
LOCALIZE ──→ VALIDATE ──→ REPAIR ──→ TEST ──→ FINAL_VALIDATE ──→ DONE
定位代码     创建/运行     修改源码    语法+测试   最终回归验证      修复完成
            回归测试
```

| 阶段 | 允许的工具 | 完成条件 |
|---|---|---|
| LOCALIZE | `list_files` `retrieve_code` `read_file` | 定位到可疑代码（evidence ready） |
| VALIDATE | + `create_bug_validation` `run_bug_validation` | 回归测试创建成功且 baseline 复现（FAIL） |
| REPAIR | + `replace_lines`（- `run_test`） | 源码被真实修改（edited） |
| TEST | + `check_syntax` `run_test` | 语法 PASS + 项目测试 PASS |
| FINAL_VALIDATE | `run_bug_validation`（失败可回退迭代修复） | 修改后回归测试 PASS |
| DONE | （无） | 6 个状态位全部为真 |

**DONE 判定条件（缺一不可，全部来自真实执行结果）：**

```
edited ∧ syntax_passed ∧ tests_passed ∧ validation_created
      ∧ baseline_confirmed ∧ validation_passed
```

## 架构

```
浏览器 (HTML/CSS/JS)                    终端
      │                                  │
      ▼                                  ▼
FastAPI (webui/server.py)          python agent.py
      │  子进程调用，不复制逻辑              │
      └────────────┬─────────────────────┘
                   ▼
            Agent Runtime (agent.py)
                   │
     ┌─────────────┼──────────────┐
     ▼             ▼              ▼
Code RAG        工具执行         LLM
(code_rag.py)  (tools.py)   (llm_client.py)
     │             │              │
     └────── Workspace ───────────┘
          (workspace_manager.py)
     克隆/解压项目 · .venv · 依赖安装 · git diff
```

Web UI 只是 Agent Runtime 的外壳：FastAPI 通过子进程调用真实的 `agent.py`，
解析其基于真实工具返回值打印的状态标记，以 SSE 推送给浏览器。
所有成功/失败状态均由 Runtime 设置，前端不自行判定。

## 环境要求

- Python ≥ 3.10
- git（克隆任务仓库、生成 diff）
- 网络访问（GitHub、PyPI、智谱 API）

## 安装

```powershell
cd AI-Code-Agent
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

依赖说明：

| 包 | 用途 |
|---|---|
| `zai-sdk` | 智谱 GLM 客户端（LLM 工具调用） |
| `numpy` `faiss-cpu` | Code RAG 向量检索 |
| `fastapi` `uvicorn` `python-multipart` | Web UI 服务层 |

> 各任务的 `.venv`（含 pytest 及项目自身依赖）由 `workspace_manager.py` 首次运行时自动创建，无需手动安装。

## 使用

### 方式一：Web UI（推荐）

```powershell
cd AI-Code-Agent\webui
python server.py
```

浏览器打开 `http://127.0.0.1:8000`，三种任务入口：

1. **上传项目** — 拖入 ZIP 或单个源文件 + 填写 Bug 描述
2. **GitHub Issue** — 粘贴 issue 链接（自动抓取标题/正文/评论，运行时克隆仓库）
3. **预置任务** — 选择 `tasks.jsonl` 中的 SWE-bench 风格任务（fastapi / requests / rich / httpx 的真实 issue）

点击 Start Repair 后可实时观察：阶段流转、每轮工具调用、RAG 检索结果、
测试计数、真实 `git diff`，完成后下载修复后的项目 ZIP。

### 方式二：命令行

```powershell
cd AI-Code-Agent
python agent.py
```

按提示输入任务 ID（如 `requests_6629`）。若上次运行中断，直接回车即可从断点继续
（会话状态保存于 workspace 的 `.agent_session.json`）。

## 目录结构

```
AI-Code-Agent/
├── agent.py              # Agent Runtime：状态机、工具调用循环、会话持久化
├── tools.py              # 8 个工具：RAG 检索 / 读文件 / 改文件 / 语法检查 / 测试 / 回归验证
├── llm_client.py         # 智谱 GLM 封装（超时 + 重试）
├── code_rag.py           # 代码嵌入 + FAISS 向量检索
├── workspace_manager.py  # 任务仓库克隆/解压、.venv 创建、依赖安装
├── task_loader.py        # 任务加载（tasks.jsonl）
├── tasks.jsonl           # 预置任务清单（repo + base_commit + problem_statement）
├── requirements.txt
├── webui/
│   ├── server.py         # FastAPI：上传/issue/预置任务 + SSE 实时流 + diff + ZIP 下载
│   └── static/           # 前端（index.html / app.js / style.css）
└── workspaces/           # 各任务的工作区（源码 + .venv + git 仓库）
```

## Agent 工具

| 工具 | 说明 |
|---|---|
| `list_files` | 列出 workspace 文件树（防重复刷屏） |
| `retrieve_code` | Code RAG 语义检索相关代码片段 |
| `read_file` | 读取真实源码（带行号；相同范围重复读取会被拦截） |
| `replace_lines` | 按行号区间修改真实源码（成功后清除读取缓存） |
| `check_syntax` | 语法检查（py_compile） |
| `run_test` | 在任务 venv 中运行 pytest |
| `create_bug_validation` | 创建回归测试（创建时预检：collect-only 捕获导入/语法错误；识别探针测试） |
| `run_bug_validation` | 运行回归测试（baseline 复现 / 最终验证共用） |

## 验证案例：requests JSONDecodeError pickle Bug

预置任务 `requests_6629`（psf/requests @ `7a13c041`）：
`requests.exceptions.JSONDecodeError` 无法正确 pickle/unpickle。

Agent 从干净 commit 出发的真实运行记录（上限 30 轮，实际 **11 轮完成**）：

| 轮次 | 动作 | 真实结果 |
|---|---|---|
| 1-2 | `list_files` + `retrieve_code` + `read_file` | 定位 `src/requests/exceptions.py`，MRO 分析正确 |
| 3-4 | 创建回归测试 → 运行 | BROKEN_TEST（Runtime 拦截，Agent 重写测试） |
| 5-6 | 重写测试（纯 assert）→ baseline | **FAIL**（Bug 成功复现） |
| 7 | `replace_lines` | 真实写入源码 |
| 8 | `check_syntax` | **PASS** |
| 9 | `run_test`（-k json） | **24 passed** / returncode 0 |
| 10 | `run_bug_validation` | **PASS** |
| 11 | — | **修复完成（DONE）** |

最终 `git diff` 与独立 pickle 往返验证（`msg/doc/pos` 全保留）均确认修复正确，
修改方向与上游官方修复一致：

```python
class JSONDecodeError(InvalidJSONError, CompatJSONDecodeError):
    ...
    def __reduce__(self):
        return CompatJSONDecodeError.__reduce__(self)
```

## 设计原则

1. **Runtime 是唯一状态来源** — LLM 文字声称"修改成功"不算数，只有真实文件变化、
   真实测试退出码才会翻转状态位。
2. **标准工具调用协议** — 工具结果以 `{role: "tool", tool_call_id, content}` 回传 LLM。
3. **失败必须反馈** — `replace_lines` 失败时把失败结果返回给 LLM 重新尝试，
   不允许假装成功。
4. **防循环** — 相同参数的重复调用（含失败调用）计数拦截；相同范围重复 `read_file` 直接返回提示。
5. **断点恢复** — 每轮保存会话，LLM API 中断后可从断点继续（自动修复悬空 tool_calls）。
6. **不碰测试** — Agent 只修改源码，禁止通过改测试/删测试"通过验收"。

## 已知限制

- 无自带测试的项目：`run_test` 返回 `NO_TESTS` 时 `tests_passed` 无法置真，
  Agent 能完成修复与回归验证但到不了 DONE。
- 全量测试需要外部服务（如 httpbin）时，Agent 选择相关子集（`-k`）运行。
- GitHub Issue 入口使用未认证 API（每小时 60 次请求限制）。

## 安全提示

`llm_client.py` 中的 API key 当前为硬编码，公开分发前请改为环境变量读取，
并在 `.gitignore` 中排除敏感配置。
