import json
import os
import subprocess

from llm_client import chat
from tools import (
    tools_map,
    set_workspace,
    cleanup_bug_validation,
    SESSION_FILE_NAME,
    VALIDATION_DIR_NAME,
)
from task_loader import get_task
from workspace_manager import create_workspace


# ============================================================
# Agent 配置
# ============================================================

MAX_ROUNDS = 30

# ---- 上下文控制（只影响发给 LLM 的消息，不影响工具真实执行结果）----
MAX_TOOL_RESULT_CHARS = 8000   # 单条工具结果进 LLM 上下文的上限
MAX_RECENT_MESSAGES = 8        # 发给 LLM 的最近消息条数下限（按完整轮块保留）
MAX_RECENT_CONTEXT_CHARS = 48000  # 最近上下文总字符上限（超出时从最旧的块开始丢弃）
TASK_SUMMARY_CHARS = 1200      # 任务摘要（problem statement）保留长度


def truncate_tool_result(text, limit=MAX_TOOL_RESULT_CHARS):
    """限制工具结果进入 LLM 上下文的长度。

    只在 result_text 追加进 messages 前调用；
    工具的真实执行结果（stdout 打印 / 返回值）不受影响。
    头尾保留：文件路径与代码开头在头部，traceback / 测试统计在尾部。
    """
    if not isinstance(text, str) or len(text) <= limit:
        return text
    head = int(limit * 0.7)
    tail = int(limit * 0.25)
    omitted = len(text) - head - tail
    return (
        text[:head]
        + f"\n...[工具输出已截断，省略中间 {omitted} 字符（完整输出共 {len(text)} 字符）]...\n"
        + text[-tail:]
    )


def build_llm_context(messages, problem_statement, workspace):
    """构造每轮发给 LLM 的压缩上下文视图。

    messages（完整历史账本）保持不动，用于会话保存与断点恢复；
    这里只生成当轮请求的视图：

        system prompt
        + 当前 Agent State（build_state_context）
        + Task Summary（历史被裁剪时补位）
        + 最近若干完整轮块（assistant + 其 tool 结果成对保留）

    轮块 = 一条 assistant 消息 + 紧随其后的所有 tool 结果。
    以块为单位裁剪，保证 tool_call 与 tool_call_id 配对永远完整。
    """
    system_msgs = [m for m in messages if m.get("role") == "system"]
    body = [m for m in messages if m.get("role") != "system"]

    # 按轮分块：assistant 开新块，其余消息归入当前块
    blocks = []
    for m in body:
        if m.get("role") == "assistant" or not blocks:
            blocks.append([m])
        else:
            blocks[-1].append(m)

    # 从最近的块往前收集，直到条数达标；块不可分割
    recent = []
    count = 0
    for block in reversed(blocks):
        recent.insert(0, block)
        count += len(block)
        if count >= MAX_RECENT_MESSAGES:
            break

    # 字符预算：从最旧的块开始丢弃
    def block_chars(block):
        return sum(len(str(m.get("content") or "")) for m in block)

    while len(recent) > 1 and sum(block_chars(b) for b in recent) > MAX_RECENT_CONTEXT_CHARS:
        recent.pop(0)

    kept = [m for b in recent for m in b]
    dropped_rounds = len(blocks) - len(recent)
    # 初始任务陈述（body 首条 user）是否已被裁掉
    task_still_visible = body and body[0] in kept

    llm_messages = []
    if system_msgs:
        llm_messages.append(system_msgs[0])
    llm_messages.append({
        "role": "system",
        "content": build_state_context(),
    })

    if dropped_rounds > 0 or not task_still_visible:
        summary = (problem_statement or "").strip()
        if len(summary) > TASK_SUMMARY_CHARS:
            summary = summary[:TASK_SUMMARY_CHARS] + f"\n...[问题描述已截断，完整长度 {len(problem_statement)} 字符]"
        notice = (
            "[Context Notice] 为控制上下文长度，更早的 "
            f"{max(dropped_rounds, 0)} 轮对话已省略。\n"
            "关键事实见当前状态；如需查看已省略的代码，"
            "请用不同参数重新 read_file（相同参数会被去重拦截）。\n\n"
            f"[Task Summary]\n{summary}\n\n"
            f"[Workspace]\n{workspace}"
        )
        llm_messages.append({"role": "user", "content": notice})

    llm_messages.extend(kept)
    return llm_messages


edited = False
tests_passed = False
syntax_passed = False
validation_created = False
validation_baseline_confirmed = False
validation_passed = False
validation_failed = False
validation_broken = False
validation_is_probe = False

# baseline 确认后的永久冻结标志：一旦在未修改代码上确认
# 原始 Bug 可复现（AssertionError 型 FAIL），当前验证测试
# 就被永久冻结——禁止 create_bug_validation 重建、禁止
# 修改验证测试文件；此后任何验证 FAIL（包括收集/导入错误）
# 都只能通过修复源代码解决。该标志在会话中永不清除。
validation_frozen = False

# run_test 是否返回过 NO_TESTS（项目没有可运行的现有 pytest 测试）。
# NO_TESTS ≠ 修复失败：一旦为 True，Runtime 禁止再次调用 run_test，
# 测试阶段视为完成，最终验证交给 run_bug_validation。
no_tests_seen = False

# 相关测试是否通过（run_test 结论中与本次修改相关的部分）。
# 与 tests_passed 的区别：现有测试中存在与本次修改无关的失败时
# （unrelated_test_failures 非空），relevant_tests_passed 仍为 True，
# 不阻断修复流程；五元组完成条件与 phase 判定均以本变量为准。
relevant_tests_passed = False

# 与本次修改无关的失败测试 id 列表（run_test UNRELATED 场景收集）
unrelated_test_failures = []

# 当前修改后的文件
last_edited_file = None

# Code RAG / 代码阅读状态
retrieve_count = 0
read_files = set()

# 连续 read_file 次数（调用其他工具即清零）
consecutive_reads = 0

# 连续工具调用失败次数（成功即清零）
consecutive_failures = 0

# 连续路径类失败次数（FileNotFoundError / NotADirectoryError）
consecutive_path_failures = 0

# 已成功执行的工具调用计数（键 = 工具名 + 参数 JSON）
tool_call_counts = {}

# 已成功执行过的 RAG 查询（规范化后），避免同一语义查询重复检索
retrieved_queries = set()

# 最近工具动作摘要，只用于给 LLM 一个紧凑的运行时状态，不无限堆积上下文
recent_actions = []

# 是否已经获得足够证据进入修改阶段
evidence_ready = False


# ============================================================
# Tool Schemas
# ============================================================

tools = [

    # --------------------------------------------------------
    # read_file
    # --------------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "读取当前 Workspace 中指定文件的真实内容。"
                "path 必须是 Workspace 内的相对文件路径。"
                "如果只需要查看部分代码，可以使用 start_line 和 max_lines。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Workspace 内的相对文件路径，"
                            "例如 example.py 或 tests/test_example.py"
                        )
                    },
                    "start_line": {
                        "type": "integer",
                        "description": (
                            "开始读取的行号，从 1 开始。"
                            "不需要指定时默认为 1。"
                        )
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": (
                            "最多读取多少行。"
                            "建议不超过 250。"
                        )
                    }
                },
                "required": ["path"]
            }
        }
    },

    # --------------------------------------------------------
    # replace_lines（首选编辑工具）
    # --------------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "replace_lines",
            "description": (
                "按 read_file 返回的真实 1-based 行号，替换源文件的连续代码。"
                "这是首选的代码修改方式，不需要复述 old 文本。"
                "start_line 和 end_line 均包含在替换范围内。"
                "禁止修改 tests 或 .agent_validation。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Workspace 内的真实相对路径"},
                    "start_line": {"type": "integer", "description": "开始行号，1-based，包含该行"},
                    "end_line": {"type": "integer", "description": "结束行号，1-based，包含该行"},
                    "new_code": {"type": "string", "description": "替换后的完整代码，仅覆盖指定行范围"}
                },
                "required": ["path", "start_line", "end_line", "new_code"]
            }
        }
    },

    # --------------------------------------------------------
    # run_test
    # --------------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "run_test",
            "description": (
                "在当前 Workspace 中运行 pytest。"
                "可以指定测试文件或测试过滤条件。"
                "test_filter 只能填写 pytest -k 表达式，"
                "不要填写 -v、--maxfail 等命令行参数。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "test_path": {
                        "type": "string",
                        "description": (
                            "可选。Workspace 内的测试路径，"
                            "例如 tests/test_requests.py。"
                        )
                    },
                    "test_filter": {
                        "type": "string",
                        "description": (
                            "可选。pytest -k 表达式。"
                            "应根据当前问题和项目测试实际情况填写。"
                            "不要填写 -v 等命令行参数。"
                        )
                    }
                }
            }
        }
    },

    # --------------------------------------------------------
    # list_files
    # --------------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "列出当前 Workspace 中的文件或目录，"
                "用于了解项目结构。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Workspace 内的相对目录路径，"
                            "例如 .、src、tests。"
                        )
                    }
                }
            }
        }
    },

    # --------------------------------------------------------
    # retrieve_code
    # --------------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "retrieve_code",
            "description": (
                "使用 Code RAG 从当前 Workspace 的 Python 代码中"
                "检索与问题最相关的代码片段。"
                "结果只能作为定位线索，不能替代 read_file。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "用于检索代码的问题、错误信息、"
                            "类名、函数名或缺陷机制。"
                        )
                    }
                },
                "required": ["query"]
            }
        }
    },

    # --------------------------------------------------------
    # check_syntax
    # --------------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "check_syntax",
            "description": (
                "检查指定 Python 文件的语法。"
                "修改 Python 文件后必须调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Workspace 内的 Python 文件路径"
                        )
                    }
                },
                "required": ["path"]
            }
        }
    },
    # --------------------------------------------------------
    # create_bug_validation
    # --------------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "create_bug_validation",
            "description": (
                "创建一个临时、通用的 Bug 回归验证测试。"
                "test_code 必须是 pytest 测试函数的完整源码，"
                "用于复现问题描述中的具体缺陷，并验证修复后的期望行为。"
                "该测试存放在 .agent_validation 目录，不得修改项目 tests。"
                "创建后必须先在未修改代码的状态运行一次，以确认 Bug 能复现；"
                "然后修改代码，再次运行同一个验证。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "test_code": {
                        "type": "string",
                        "description": (
                            "完整 pytest 测试文件源码。必须包含至少一个 test_ 开头的测试函数，"
                            "断言应直接对应原始 Bug 的期望修复行为。"
                        )
                    }
                },
                "required": ["test_code"]
            }
        }
    },

    # --------------------------------------------------------
    # run_bug_validation
    # --------------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "run_bug_validation",
            "description": (
                "运行当前临时 Bug 回归验证。"
                "第一次必须在修改代码前运行：若测试失败，说明原始 Bug 成功复现；"
                "修改并通过语法检查、正常测试后，再次运行：若测试通过，说明该 Bug 已修复。"
            ),
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }

]


# ============================================================
# 任务初始化
# ============================================================

instance_id = input("请输入任务 ID：")

task = get_task(instance_id)

workspace = create_workspace(
    repo=task["repo"],
    base_commit=task["base_commit"],
    instance_id=task["instance_id"]
)

set_workspace(workspace)

# ============================================================
# 会话恢复：上次 API 故障退出后，可从断点继续
# 而不是从第 1 轮重新开始（不浪费已完成的轮数）
# ============================================================

session_path = os.path.join(workspace, SESSION_FILE_NAME)
resumed_session = None

if os.path.isfile(session_path):

    try:
        with open(
            session_path,
            "r",
            encoding="utf-8"
        ) as f:
            resumed_session = json.load(f)

        print(
            f"\n检测到未完成的会话"
            f"（已完成 {resumed_session['round_index']} 轮）。"
        )

        answer = input(
            "从断点继续？"
            "（回车=继续 / n=重置 Workspace 重新开始）："
        ).strip().lower()

        if answer in ("n", "no"):
            resumed_session = None

    except (OSError, ValueError, KeyError) as e:
        print(f"\n⚠️ 会话文件无法读取，将重新开始：{e}")
        resumed_session = None

# 用户明确选择重新开始（或会话损坏）：
# 把 Workspace 恢复到 base 状态，避免带着上次的半成品修改开局
if os.path.isfile(session_path) and resumed_session is None:

    subprocess.run(
        ["git", "checkout", "--", "."],
        cwd=workspace,
        capture_output=True
    )

    cleanup_bug_validation()

    os.remove(session_path)

    print("已重置 Workspace（git checkout + 清理验证文件），从第 1 轮重新开始。")

problem_statement = task["problem_statement"]


print("\n========== 当前任务 ==========")
print(f"任务：{task['instance_id']}")
print(f"仓库：{task['repo']}")
print(f"Workspace：{workspace}")
print("\n问题描述：")
print(problem_statement)
print("==============================\n")


# ============================================================
# System Prompt
# ============================================================

SYSTEM_PROMPT = """
你是一个面向真实软件仓库的自主代码缺陷诊断与自动修复 Agent。

你的最终目标不是解释 Bug，而是真正修改当前 Workspace 中的代码，
然后通过真实测试证明修改有效。

============================================================
一、你的工作目标
============================================================

你必须完成：

问题理解
↓
代码定位
↓
真实代码阅读
↓
Bug 原因分析
↓
实际修改
↓
语法检查
↓
测试
↓
根据测试结果继续迭代
↓
确认修复
↓
结束

你不是聊天机器人。

你是代码修复 Agent。

============================================================
二、可用工具
============================================================

你拥有：

- list_files
- retrieve_code
- read_file
- replace_lines
- check_syntax
- run_test
- create_bug_validation
- run_bug_validation

必须通过真实工具完成工作。

所有文件路径必须以 list_files / retrieve_code / read_file 返回的真实路径为准。
不要假设 src/ 等目录布局，不要凭空猜测路径。
工具会在路径写错但 Workspace 内存在唯一同名文件时自动修正路径（如 src/rich/_wrap.py → rich/_wrap.py）；
返回结果中的真实路径必须用于后续所有调用。
同一目录不要重复调用 list_files，内容不会变化。

禁止：

- 伪造工具调用
- 在文本中模拟工具执行
- 声称已经修改代码但没有调用 replace_lines
- 声称测试通过但没有真实 run_test 结果
- 根据自己的推理直接宣布任务成功

============================================================
四、通用 Bug 回归验证机制
============================================================

不要依赖任何针对具体仓库、Issue 编号、类名或 Bug 的硬编码检查。

修复机制优先级：problem statement 明确指出的机制与行为描述
优先于你自己猜测的替代实现。
- problem statement 明确描述了失败机制时，按描述的机制定位和修复；
- 只有 problem statement 没有给出机制线索时，才基于真实代码自行推理；
- 你的猜测与 problem statement 或真实代码冲突时，
  以 problem statement 和真实代码为准，放弃自己的猜测。

如果现有 pytest 已经直接覆盖原始 Bug，可以直接使用相关测试。
如果现有测试不能证明原始 Bug 已被修复，则必须：

1. 在修改代码之前，根据 problem statement 和真实代码创建一个临时 Bug 验证测试；
2. 调用 create_bug_validation；
3. 在任何代码修改之前调用 run_bug_validation；
4. 如果验证测试失败，这是预期的“Bug 已复现”，必须记录为 baseline confirmed；
5. 然后调用 replace_lines 修改真实代码；
6. check_syntax → run_test；
7. 再次调用 run_bug_validation；
8. 只有第二次 PASS 才能把原始 Bug 视为真正修复。

baseline 确认（第 4 步）之后，这个验证测试会被永久冻结：

- 禁止重新调用 create_bug_validation；
- 禁止修改、重建或替换该验证测试（Runtime 会直接拦截）；
- 此后任何 run_bug_validation FAIL 都只有一个含义——
  当前修改没有真正修复原始 Bug；
- 唯一出路是重新分析并修改源代码，绝不是改验证测试。

临时验证测试：
- 不得修改项目 tests；
- 不得修改源代码；
- 必须直接验证 problem statement 明确描述的失败行为；
- 验证测试必须是“最小复现”，只断言 Issue 明确要求保持/恢复的行为；
- 不得凭空增加未被 problem statement、现有测试或真实代码语义支持的断言；
- 不得把错误消息的精确文本、内部属性、对象 repr、参数格式等当成验收条件，除非问题描述或现有测试明确要求；
- 不得为了让测试通过而改变断言；
- 不得在验证测试中引入与 Bug 无关的额外失败点，例如局部类、临时对象无法序列化、网络服务、外部资源等；
- 对序列化/反序列化问题，优先验证“原操作能够成功完成 + 恢复后的对象类型/关键状态符合问题描述”，不要自行假设异常消息等细节；
- 对异常修复问题，验证重点应是问题描述中的异常是否仍然发生，而不是自行重新定义异常的语义；
- 必须使用真实项目代码，而不是模拟一个假的修复结果；
- 创建验证测试后，在修改前运行它；如果出现与目标 Bug 无关的错误，应先修正验证测试本身，再确认 baseline；
- 有效的 baseline 失败必须是 AssertionError（断言失败）。如果失败原因是 AttributeError、TypeError、ImportError 等错误，说明验证测试自身调用了不存在的 API 或写错了代码，必须先修正验证测试再确认 baseline；
- 验证测试只能调用项目中真实存在的 API；不确定 API 是否存在时，先用 read_file 或 retrieve_code 确认；
- baseline 验证测试必须能区分有 Bug 与无 Bug 的行为：直接断言 problem statement 描述的精确期望行为优于宽松不变量（如仅断言不抛异常——有 Bug 的代码也可能满足后者），后者可能导致 baseline 误通过；baseline 误通过后重复运行没有意义，必须重新 create_bug_validation 写更强的断言（调整触发条件、断言粒度）；
- 实验优先于阅读：已定位可疑代码但不确定触发条件时，先写最小复现测试试跑，比继续读文件更接近答案；
- 现有测试通过不代表目标 Bug 不存在：回归测试通常在修复 PR 中才加入。根据 problem statement 描述的场景自己构造触发输入是预期工作，不算凭空断言；
- 验证测试写错不是失败：创建 → BROKEN_TEST → 修正 → 重新创建 是正常迭代，不要为避免 BROKEN_TEST 而无限阅读代码。

============================================================

============================================================
三、严格工具真实性
============================================================

如果没有调用：

replace_lines

就不能说：

“代码已经修改”。

如果没有真实的：

run_test

就不能说：

“测试通过”。

如果工具返回：

FAIL

就不能说：

“测试通过”。

如果工具返回：

ERROR

就不能说：

“工具执行成功”。

只能根据真实工具结果判断。

============================================================
四、Code RAG 使用规则
============================================================

第一次修改代码之前必须成功使用：

retrieve_code

然后必须使用：

read_file

读取真实代码。

retrieve_code 只是定位线索。

不能直接根据 RAG 返回的代码片段进行修改。

基本流程：

problem_statement
↓
retrieve_code
↓
候选文件
↓
read_file
↓
真实代码
↓
分析
↓
replace_lines

============================================================
五、非常重要：禁止无限重复 retrieve_code
============================================================

retrieve_code 是定位工具，不是让你无限搜索的工具。

如果已经：

1. retrieve_code 找到了相关文件
2. read_file 读取了相关真实代码
3. problem_statement 已经明确给出了 Bug 机制

就必须进入修改阶段。

禁止：

retrieve_code
↓
retrieve_code
↓
retrieve_code
↓
retrieve_code
↓
继续搜索相同问题

如果已经阅读了直接相关代码，就不要继续寻找“更多相似代码”。

============================================================
七、修改代码
============================================================

修改代码必须使用：

replace_lines

参数：

- path
- start_line
- end_line
- new_code

start_line / end_line 是 read_file 返回的 1-based 真实行号，
起止行都包含在替换范围内。
new_code 是替换后的完整代码，只覆盖指定行范围。

禁止：

- 修改 tests
- 删除测试
- 修改测试断言
- 跳过测试
- 为了测试通过修改测试

============================================================
八、修改后的强制流程
============================================================

成功修改代码以后：

replace_lines
↓
check_syntax
↓
run_test

不能：

replace_lines
↓
直接宣布成功

也不能：

replace_lines
↓
run_test

必须先：

check_syntax

============================================================
九、测试规则
============================================================

run_test 的 test_filter 是 pytest -k 表达式。

测试选择必须根据当前 Workspace 的实际项目结构和当前 Bug 来决定。
优先运行与修改文件、Bug 行为或相关模块直接对应的现有测试。

如果没有足够直接的现有测试，不得把一个无关测试的 PASS 当作 Bug 修复证据，
应使用 create_bug_validation / run_bug_validation 建立直接的回归验证。

不要把 -v、--verbose、--maxfail 等命令行参数放入 test_filter。

============================================================
十、测试通过不等于 Bug 修复
============================================================

如果：

run_test

返回 PASS，

只能说明：

这个测试通过。

不一定说明：

Bug 已经修复。

最终回归验证应该尽可能直接覆盖 problem statement 描述的失败行为，
而不是只覆盖一个表面相关的函数或测试名称。

如果已有测试与 Bug 没有直接关系：

不要仅凭它 PASS 就宣布成功。

============================================================
十一、测试失败
============================================================

如果测试 FAIL：

不能结束。

必须：

读取 traceback
↓
分析失败原因
↓
read_file / retrieve_code
↓
修改
↓
check_syntax
↓
run_test

继续迭代。

不要为了测试通过修改 tests。

============================================================
十二、任务完成条件
============================================================

只有全部满足：

1. 成功调用 retrieve_code
2. 使用 read_file 阅读真实代码
3. 找到明确缺陷原因
4. 成功调用 replace_lines
5. replace_lines 返回成功
6. 修改后的代码通过 check_syntax
7. 修改后的代码经过测试阶段：
   - 如果项目存在相关 pytest，则与本次修改相关的测试必须全部通过；
     现有测试中与本次修改无关的失败（Runtime 会标记
     failure_relevance = UNRELATED）不阻断完成，
     也不要去修复它们或修改对应测试；
   - 如果项目没有现有测试，则 run_test 返回 NO_TESTS，
     不视为失败，必须由 run_bug_validation 完成最终验证。
8. 最终 run_bug_validation 必须 PASS。
9. 测试或验证必须与问题直接相关。
10. 没有修改 tests

才能宣布：

任务成功完成。

============================================================
十三、最重要的行为
============================================================

如果已经有足够证据：

不要继续搜索。

不要继续解释。

不要继续重复测试。

直接：

replace_lines
↓
check_syntax
↓
run_test

真正完成代码修复。

============================================================
十四、你必须记住
============================================================

你的目标不是：

“给出一个看起来合理的修复方案”。

你的目标是：

“真正修改 Workspace，并用测试证明修复有效”。

============================================================
十五、通用 Bug 回归验证
============================================================

当前 Workspace 使用通用的临时 Bug 回归验证机制。

对于每一个需要修改源代码的 Bug 修复任务：
1. 根据 problem statement 和真实源代码创建临时验证；
2. 修改前运行验证，确认原始 Bug FAIL；
3. 修改真实源代码；
4. check_syntax；
5. run_test（若返回 NO_TESTS，按“十六、NO_TESTS 的行为规则”处理）；
6. 再次运行同一个临时验证，必须 PASS。

临时验证不得绑定任何固定仓库、Issue 编号、类名、函数名或文件路径。
不得修改项目原有 tests。
不得把无关 pytest 的 PASS 当作原始 Bug 已修复的证明。

创建临时验证时必须先做“验收条件审查”：
1. 从 problem statement 提取明确的失败行为；
2. 只为这个失败行为设计最小复现；
3. 每个 assert 都必须能说明它为什么来自 problem statement、现有测试或真实代码语义；
4. 如果某个 assert 只是模型自己的猜测，删除它；
5. 如果测试包含额外 fixture / mock / 临时类，必须确认它不会引入与目标 Bug 无关的失败；
6. baseline FAIL 时必须检查 traceback，确认失败原因就是目标 Bug，而不是验证测试自身写错。

尤其禁止：
- 为了验证 pickle 而额外要求字符串完全相等；
- 为了验证序列化而使用无法被 pickle 的局部类；
- 为了验证异常而自行假设新的异常消息；
这些都属于与目标 Bug 无关的过度断言，除非 problem statement 明确要求。

============================================================
十六、NO_TESTS 的行为规则
============================================================

如果 run_test 返回 NO_TESTS：

1. NO_TESTS 表示当前项目没有可运行的现有 pytest 测试。
2. NO_TESTS 不是修复失败，不等于 FAIL。
3. 禁止再次调用 run_test。
4. 禁止猜测、寻找或虚构 test.py / test_bug.py / tests/ / test_*.py
   等不存在的测试文件。
5. 如果代码已经修改成功并且 check_syntax 通过，
   必须立即调用 run_bug_validation 进行最终 Bug 回归验证。
6. 最终任务是否成功，以 run_bug_validation 的真实返回结果为准，
   不以你自己的文字描述为准。
   你说“已经修复”“代码看起来正确”“应该已经解决”
   都不能作为任务完成的依据。

英文规则（与上述规则同等效力）：

NO_TESTS means that the project has no runnable existing pytest tests.
It is not a repair failure.
After receiving NO_TESTS, do not call run_test again.
Do not guess or invent test files.
If the code has been modified and syntax validation passed, immediately call run_bug_validation.
The final success decision must be based on the actual result of run_bug_validation, not on the model's textual claim.


"""


# ============================================================
# 初始化 messages
# ============================================================

messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    },
    {
        "role": "user",
        "content": f"""
用户的问题：

{problem_statement}

当前任务 Workspace：

{workspace}

当前 Workspace 支持通用的临时 Bug 回归验证。

本 Agent 不针对任何特定仓库、Issue 或 Bug 编写专门逻辑。
对于当前任务，应从 problem statement 和 Workspace 的真实代码中
推导验证场景，并使用 create_bug_validation / run_bug_validation
完成“修改前复现 → 修改后验证”的闭环。临时验证文件将在任务结束时清理。

请自主调查并修复这个问题。

所有文件操作和测试操作必须针对当前 Workspace。
"""
    }
]


# ============================================================
# 恢复会话状态（如有）
# ============================================================

start_round = 0

if resumed_session is not None:

    saved_state = resumed_session["state"]

    for key in (
        "edited",
        "tests_passed",
        "syntax_passed",
        "validation_created",
        "validation_baseline_confirmed",
        "validation_passed",
        "validation_failed",
        "validation_broken",
        "evidence_ready",
        "retrieve_count",
        "consecutive_reads",
        "consecutive_failures",
        "consecutive_path_failures",
        "last_edited_file",
    ):
        globals()[key] = saved_state[key]

    # 旧版本会话文件没有该字段，默认 False
    globals()["validation_is_probe"] = saved_state.get(
        "validation_is_probe",
        False
    )

    # 旧版本会话文件没有 validation_frozen 字段，默认 False。
    # 冻结是永久状态：恢复后依然生效，禁止解冻。
    globals()["validation_frozen"] = saved_state.get(
        "validation_frozen",
        False
    )

    # 旧版本会话文件没有 no_tests_seen 字段，默认 False，
    # 保证恢复后 NO_TESTS → 禁止 run_test 的规则仍然生效
    globals()["no_tests_seen"] = saved_state.get(
        "no_tests_seen",
        False
    )

    # 旧版本会话文件没有这两个字段，默认 False / []。
    # 相关性结论是完成条件五元组的组成部分：
    # "现有测试有无关失败但不阻断完成"的状态断点恢复后必须还原，
    # 否则恢复后会卡在 run_test 阶段（而无关失败可能永远无法消除）。
    globals()["relevant_tests_passed"] = saved_state.get(
        "relevant_tests_passed",
        False
    )

    globals()["unrelated_test_failures"] = list(
        saved_state.get(
            "unrelated_test_failures",
            []
        )
    )

    read_files.clear()
    read_files.update(saved_state.get("read_files", []))

    tool_call_counts.clear()
    for entry in saved_state.get("tool_call_counts", []):
        tool_call_counts[(entry[0], entry[1])] = entry[2]

    retrieved_queries.clear()
    retrieved_queries.update(saved_state.get("retrieved_queries", []))
    recent_actions.clear()
    recent_actions.extend(saved_state.get("recent_actions", [])[-8:])

    messages = resumed_session["messages"]
    start_round = resumed_session["round_index"]

    # 修复被中断的 tool_calls：
    # 如果最后一条 assistant 消息带 tool_calls，但其后没有
    # 对应的 tool 结果消息（进程在工具执行中途被杀死），
    # 必须补上合成结果，否则下一次 LLM 调用会因
    # 消息序列不符合 tool calling 协议而直接报错。
    _dangling = []
    for _msg in reversed(messages):
        if _msg.get("role") == "tool":
            break
        if _msg.get("role") == "assistant" and _msg.get("tool_calls"):
            _dangling = _msg["tool_calls"]
            break

    if _dangling:
        for _tc in _dangling:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": _tc.get("id") or "",
                    "content": (
                        "工具执行被进程中断，结果未知。"
                        "请重新调用该工具获取真实结果。"
                    )
                }
            )
        print(
            f"\n🔧 已为 {len(_dangling)} 个被中断的"
            "工具调用补齐结果消息"
        )

    print(
        f"\n✅ 已恢复会话："
        f"从第 {start_round + 1} 轮继续"
        f"（剩余 {MAX_ROUNDS - start_round} 轮）。"
    )

    if edited:
        print(f"上次已修改文件：{last_edited_file}")


# ============================================================
# 辅助函数
# ============================================================

def arguments_to_dict(arguments):
    """
    Zhipu 的 tool_call arguments 通常是 JSON 字符串。
    这里统一转换成 dict。
    """

    if isinstance(arguments, dict):
        return arguments

    if isinstance(arguments, str):

        try:
            return json.loads(arguments)

        except json.JSONDecodeError as e:

            raise ValueError(
                f"工具参数不是合法 JSON：{arguments}"
            ) from e

    raise TypeError(
        f"未知工具参数类型：{type(arguments).__name__}"
    )


def contains_any(text, keywords):
    """
    判断文本是否包含任意关键词。
    """

    text = text.lower()

    return any(
        keyword.lower() in text
        for keyword in keywords
    )


def should_mark_evidence_ready(path):
    """
    判断是否已经读取到足够的真实源代码证据。
    不绑定任何具体仓库、Issue、类名、函数名或文件路径。
    """
    normalized = path.replace("\\", "/").lower()

    test_like = (
        normalized.startswith("tests/")
        or "/tests/" in normalized
        or normalized.startswith("test_")
        or normalized.endswith("_test.py")
    )

    return not test_like


def repeat_strategy_hint():
    """
    完全重复调用被阻止时，根据当前状态给出
    应该做什么的具体指引（读取全局状态变量）。
    """
    if (
        validation_created
        and not edited
        and not validation_baseline_confirmed
    ):
        if validation_broken:
            return (
                "\n当前验证测试自身有错误（BROKEN_TEST）："
                "修正测试代码后重新 create_bug_validation，"
                "必要时 read_file 确认真实 API。"
            )
        if validation_failed:
            return (
                "\n当前验证测试没有捕捉到 Bug"
                "（未修改的代码上 PASS）："
                "写更强的断言"
                "（直接断言 problem statement 描述的期望行为），"
                "重新 create_bug_validation。"
            )
        return (
            "\n当前应调用 run_bug_validation 确认 baseline FAIL。"
        )

    if evidence_ready and not edited and not validation_created:
        return (
            "\n当前应调用 create_bug_validation 创建复现测试。"
        )

    if edited and not syntax_passed:
        return "\n当前应调用 check_syntax。"

    # 与 next_required_action 的 run_test 分支保持一致：
    # 用 relevant_tests_passed 而不是 tests_passed，
    # 无关失败场景不应再引导重复 run_test。
    if edited and syntax_passed and not relevant_tests_passed:
        return (
            "\n当前应调用 run_test 验证修改，"
            "或继续 replace_lines 修复剩余问题。"
        )

    return ""


def _is_validation_test_path(path):
    """
    判断目标路径是否指向临时验证测试目录（.agent_validation/）。

    用于 Runtime Guard：验证测试只能通过 create_bug_validation
    整体重写（且仅限 baseline 确认之前），绝不允许被
    replace_lines 等编辑工具逐行修改。
    """
    if not isinstance(path, str):
        return False

    normalized = path.replace("\\", "/").strip().strip("/").lower()

    return (
        normalized == VALIDATION_DIR_NAME
        or normalized.startswith(VALIDATION_DIR_NAME + "/")
        or f"/{VALIDATION_DIR_NAME}/" in f"/{normalized}/"
    )


def _frame_matches_edited_file(frame, edited_file):
    """
    判断 traceback 中的一个源码帧路径是否指向被修改的文件。

    frame 形如 "src/foo.py:42" 或 "D:\\proj\\src\\foo.py:42"；
    edited_file 是 replace_lines 成功后的文件路径
    （相对 / 绝对、正斜杠 / 反斜杠均可能出现）。
    双向做路径后缀匹配，并统一盘符、斜杠与大小写差异。
    """
    if (
        not isinstance(frame, str)
        or not isinstance(edited_file, str)
        or not frame
        or not edited_file
    ):
        return False

    frame_norm = frame.replace("\\", "/").strip().lower()

    # 去掉行号后缀（...:42）
    head, _, tail = frame_norm.rpartition(":")
    if tail.isdigit():
        frame_norm = head

    # 去掉 Windows 盘符（d:/... → /...）
    if len(frame_norm) >= 2 and frame_norm[1] == ":":
        frame_norm = frame_norm[2:]

    edited_norm = edited_file.replace("\\", "/").strip().lower()

    if len(edited_norm) >= 2 and edited_norm[1] == ":":
        edited_norm = edited_norm[2:]

    while edited_norm.startswith("./"):
        edited_norm = edited_norm[2:]

    if not frame_norm or not edited_norm:
        return False

    return (
        frame_norm == edited_norm
        or frame_norm.endswith("/" + edited_norm)
        or edited_norm.endswith("/" + frame_norm)
    )


# ============================================================
# Agent Runtime State / Context
# ============================================================

def normalize_query(query):
    """把 RAG 查询规范化，用于识别完全相同/仅空白不同的重复查询。"""
    return " ".join(str(query or "").lower().split())


def current_phase():
    """
    根据 Runtime 真状态计算当前阶段。

    LOCALIZE → VALIDATE → REPAIR → TEST → FINAL_VALIDATE → DONE

    所有输入都来自真实工具结果，LLM 文本不能改变阶段。
    """
    if (
        edited
        and syntax_passed
        and relevant_tests_passed
        and validation_created
        and validation_baseline_confirmed
        and validation_passed
    ):
        return "DONE"
    if edited and syntax_passed and validation_failed:
        # 最终回归验证失败：回到 REPAIR 重新修改代码，
        # 禁止停留在 TEST / FINAL_VALIDATE 原地重复验证。
        return "REPAIR"
    if (
        edited
        and syntax_passed
        and tests_passed
        and validation_created
        and validation_baseline_confirmed
    ):
        return "FINAL_VALIDATE"
    if edited and syntax_passed and no_tests_seen:
        # run_test 已返回 NO_TESTS：项目没有可运行的现有 pytest 测试，
        # 测试阶段视为完成，直接进入最终 Bug 回归验证，
        # 禁止停留在 TEST 阶段反复 run_test。
        return "FINAL_VALIDATE"
    if edited and syntax_passed:
        return "TEST"
    if edited:
        return "REPAIR"
    if evidence_ready and validation_baseline_confirmed:
        return "REPAIR"
    if evidence_ready:
        return "VALIDATE"
    return "LOCALIZE"


def next_required_action():
    """把 Runtime 已知状态转换成唯一的优先动作。"""
    # 完成条件五元组：修改 + 语法 + 相关测试通过
    # + baseline 确认 + 最终验证通过。
    # 注意用的是 relevant_tests_passed 而不是 tests_passed：
    # 与本次修改无关的现有测试失败不阻断完成。
    if (edited and syntax_passed and relevant_tests_passed
                and validation_baseline_confirmed and validation_passed):
        return "STOP: 已满足完成条件，不再调用工具。"
    if edited and syntax_passed and validation_failed:
        # 最终回归验证失败：必须重新修改代码，而不是重复运行验证
        return "replace_lines（回归验证失败，重新修改代码）"
    if edited and syntax_passed and not relevant_tests_passed and no_tests_seen:
        # NO_TESTS：项目没有现有 pytest 测试，禁止 run_test，
        # 直接进行最终 Bug 回归验证
        return "run_bug_validation"
    if (edited and syntax_passed and relevant_tests_passed
                and validation_created and not validation_passed):
        return "run_bug_validation"
    if edited and syntax_passed and not relevant_tests_passed:
        return "run_test"
    if edited and not syntax_passed:
        return "check_syntax"
    if evidence_ready and validation_baseline_confirmed:
        return "replace_lines"
    if evidence_ready and validation_created and not validation_baseline_confirmed:
        if validation_failed or validation_broken:
            return "create_bug_validation（重写验证测试）"
        return "run_bug_validation"
    if evidence_ready and not validation_created:
        return "create_bug_validation"
    return "retrieve_code / read_file（仅在证据不足时）"


def allowed_tools_for_phase():
    """
    根据 Runtime 真状态给出本轮允许的工具，防止 LLM 跨阶段乱跳。

    越靠后的阶段允许的工具越完整：
    后期发现新问题时可以回头读代码、重新修改，
    不会因为进入 FINAL_VALIDATE 而被锁死在单一工具上。
    早期阶段则严格限制：没有确认 baseline 之前不可能修改源码。
    """
    phase = current_phase()
    mapping = {
        "LOCALIZE": {"list_files", "retrieve_code", "read_file"},
        "VALIDATE": {"read_file", "create_bug_validation", "run_bug_validation"},
        "REPAIR": {"read_file", "replace_lines", "check_syntax"},
        "TEST": {
            "read_file", "replace_lines", "check_syntax",
            "run_test", "run_bug_validation",
        },
        # FINAL_VALIDATE 只允许 run_bug_validation：
        # NO_TESTS 之后继续 run_test / 猜测试文件 / 反复改代码都被禁止。
        "FINAL_VALIDATE": {"run_bug_validation"},
        "DONE": set(),
    }
    return mapping.get(phase, set())


def build_state_context():
    """每次 LLM 调用前动态生成一份短状态，而不是永久塞进 messages 历史。"""
    recent = recent_actions[-6:]
    recent_text = "\n".join(
        f"- {item}" for item in recent
    ) or "- 暂无"

    read_text = ", ".join(sorted(read_files)[-10:]) or "无"
    query_text = ", ".join(sorted(retrieved_queries)[-6:]) or "无"

    return f"""
================ CURRENT AGENT STATE ================
这是 Runtime 的真实状态，不是用户的新问题，也不是建议。
你必须优先服从这里的状态，避免重复已经完成的动作。

phase: {current_phase()}
retrieve_count: {retrieve_count}
evidence_ready: {evidence_ready}
validation_created: {validation_created}
validation_baseline_confirmed: {validation_baseline_confirmed}
edited: {edited}
last_edited_file: {last_edited_file or '无'}
syntax_passed: {syntax_passed}
tests_passed: {tests_passed}
relevant_tests_passed: {relevant_tests_passed}
unrelated_test_failures: {", ".join(unrelated_test_failures) or '无'}
validation_failed: {validation_failed}
validation_broken: {validation_broken}
validation_passed: {validation_passed}
no_tests_seen: {no_tests_seen}

已读取文件：
{read_text}

已经成功执行过的 RAG 查询（不要重复这些查询）：
{query_text}

最近工具动作：
{recent_text}

当前唯一优先动作：
{next_required_action()}

当前阶段允许的工具：
{", ".join(sorted(allowed_tools_for_phase())) or "无"}

规则：
1. 如果 evidence_ready=True，禁止继续探索式 retrieve_code/list_files，除非已有证据明确不足。
2. 如果某个 RAG 查询已经执行过，不要再次用同义改写重复检索；应使用已有结果 + read_file。
3. 如果已找到真实源码，下一步优先是验证或修改，而不是继续收集上下文。
4. Runtime Guard 阻止某个动作时，不要反复尝试同一个动作；执行 CURRENT AGENT STATE 中的唯一优先动作。
5. 只有真实工具结果才能改变状态；不要凭文本声称“已完成”。
=======================================================
"""


# ============================================================
# Agent Loop
# ============================================================

llm_failed = False

for round_number in range(start_round, MAX_ROUNDS):

    print(
        f"\n========= 第 {round_number + 1} 轮 ========="
    )

    # --------------------------------------------------------
    # 保存会话进度（放在轮首：所有 continue/break 路径都能覆盖，
    # 保存的是上一轮完全结束后的状态）
    # --------------------------------------------------------

    try:

        session_state = {
            "edited": edited,
            "tests_passed": tests_passed,
            "syntax_passed": syntax_passed,
            "validation_created": validation_created,
            "validation_baseline_confirmed": validation_baseline_confirmed,
            "validation_passed": validation_passed,
            "validation_failed": validation_failed,
            "validation_broken": validation_broken,
            "validation_is_probe": validation_is_probe,
            "validation_frozen": validation_frozen,
            "no_tests_seen": no_tests_seen,
            "relevant_tests_passed": relevant_tests_passed,
            "unrelated_test_failures": list(unrelated_test_failures),
            "evidence_ready": evidence_ready,
            "retrieve_count": retrieve_count,
            "consecutive_reads": consecutive_reads,
            "consecutive_failures": consecutive_failures,
            "consecutive_path_failures": consecutive_path_failures,
            "last_edited_file": last_edited_file,
            "read_files": list(read_files),
            "retrieved_queries": sorted(retrieved_queries),
            "recent_actions": list(recent_actions[-8:]),
            "tool_call_counts": [
                [tool_name, args_json, count]
                for (tool_name, args_json), count
                in tool_call_counts.items()
            ],
        }

        # round_index = 当前即将执行的轮次（0 基），
        # 之前的所有轮次均已完成
        tmp_path = session_path + ".tmp"

        with open(
            tmp_path,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                {
                    "round_index": round_number,
                    "state": session_state,
                    "messages": messages,
                },
                f,
                ensure_ascii=False
            )

        # 原子替换，避免写一半损坏会话文件
        os.replace(tmp_path, session_path)

    except OSError as e:
        print(f"\n⚠️ 会话进度保存失败：{e}")

    # --------------------------------------------------------
    # 如果已经修改 + 相关测试通过 + 最终验证通过（五元组）
    # --------------------------------------------------------

    if (edited and syntax_passed and relevant_tests_passed
                and validation_created
                and validation_baseline_confirmed and validation_passed):

        print("\n========== 修复完成 ==========")
        print(
            "Agent 已实际修改代码，并获得测试通过结果。"
        )
        break

    # --------------------------------------------------------
    # 调用 LLM
    # --------------------------------------------------------

    try:

        # 上下文裁剪：messages 完整历史不动（供会话保存/恢复），
        # 每轮只把 system + 当前状态 + 任务摘要 + 最近完整轮块发给 LLM，
        # 从根本上控制多轮运行的 context growth。
        llm_messages = build_llm_context(
            messages, problem_statement, workspace
        )

        allowed_names = allowed_tools_for_phase()
        active_tools = [
            tool for tool in tools
            if tool["function"]["name"] in allowed_names
        ]

        response = chat(
            llm_messages,
            tools=active_tools
        )

    except Exception as e:

        print(
            "\n========== LLM 调用失败 =========="
        )
        print(
            f"{type(e).__name__}: {e}"
        )
        print(
            "\nLLM API 已连续多次失败，进程退出。\n"
            "会话进度已自动保存："
            "稍后重新运行 python agent.py，"
            "在提示时按回车即可从断点继续，"
            "已完成的轮数不会浪费。"
        )

        llm_failed = True
        break

    assistant_message = response["message"]

    print("\nAgent：")
    print(
        assistant_message.get("content", "")
    )

    # 保存 Agent 消息
    messages.append(assistant_message)

    tool_calls = assistant_message.get(
        "tool_calls"
    )

    # ========================================================
    # 没有工具调用
    # ========================================================

    if not tool_calls:

        # ----------------------------------------------------
        # 真正完成
        # ----------------------------------------------------

        # 完成条件五元组：与本次修改无关的现有测试失败
        # （relevant_tests_passed=True 而 tests_passed=False）
        # 不阻断完成。
        if (edited and syntax_passed and relevant_tests_passed
                and validation_created
                and validation_baseline_confirmed and validation_passed):

            print(
                "\n========== 修复完成 =========="
            )

            print(
                assistant_message.get(
                    "content",
                    ""
                )
            )

            break

        # ----------------------------------------------------
        # Agent 想直接结束，但实际上没有完成
        # ----------------------------------------------------

        print(
            "\nAgent 尚未完成真实代码修复，"
            "不能结束任务。"
        )

        # 根据当前状态给出不同的强制指令
        if evidence_ready and not edited and not validation_created:

            continuation = """
你已经获得了足够的真实代码证据。

在修改任何源代码之前，必须建立针对当前 Bug 的通用临时回归验证。

现在必须调用：
create_bug_validation

验证代码必须来自当前 problem statement、真实源代码和实际 Bug 行为，
不得绑定某个固定仓库、Issue、类名或文件名。

创建成功后立即调用：
run_bug_validation

修改前验证必须 FAIL，才能确认 baseline。
在 baseline 未确认前禁止调用 replace_lines。
"""

        elif evidence_ready and not edited and validation_created and not validation_baseline_confirmed:

            continuation = """
临时 Bug 回归验证已经创建，但修改前 baseline 尚未确认。

现在必须调用：
run_bug_validation

预期结果是 FAIL，表示原始 Bug 可以复现。
只有确认 baseline 后才能调用 replace_lines。
"""

        elif evidence_ready and not edited and validation_created and validation_baseline_confirmed:

            continuation = """
原始 Bug 已经通过临时回归验证确认可以复现。

现在必须调用：
replace_lines

然后严格执行：
check_syntax
→ run_test
→ run_bug_validation

不要只描述修改方案，必须真实修改当前 Workspace 中的源代码。
"""

        elif edited and not syntax_passed:

            continuation = """
代码已经实际修改。

现在不能结束任务。

必须调用：

check_syntax

语法检查通过后才能调用：

run_test
"""

        elif validation_failed and validation_frozen:

            # 冻结状态：验证测试不可重建、不可修改，
            # 与 Runtime Guard 的拦截行为保持一致
            # （create_bug_validation 会被直接拦截）。
            continuation = """
最终 Bug 回归验证失败。

验证测试已在 baseline 确认时永久冻结：
禁止重新调用 create_bug_validation，
禁止修改、重建或替换验证测试（Runtime 会直接拦截）。

最终验证 FAIL 只有一个含义：当前修改没有真正修复原始 Bug。

请根据验证失败的真实输出重新分析：
1. 必要时 read_file 阅读真实代码；
2. 调用 replace_lines 修正真正的缺陷；
3. check_syntax；
4. run_test（若 run_test 曾返回 NO_TESTS，跳过此步并禁止再调用）；
5. 再次 run_bug_validation。

禁止修改 tests。
"""

        elif validation_failed:

            continuation = """
通用 Bug 回归验证没有通过。
如果当前仍未修改源代码，必须重新调用 create_bug_validation 重写验证测试；
不要重复运行同一个无效测试。

这说明当前修改没有真正修复原始 Bug。
不要只重复运行验证而不分析失败原因。

请根据 Bug 验证返回的真实失败输出重新分析：
1. 必要时 read_file 阅读真实代码；
2. 必要时 retrieve_code 定位相关代码；
3. 调用 replace_lines 修正真正的缺陷；
4. check_syntax；
5. run_test（若 run_test 曾返回 NO_TESTS，跳过此步并禁止再调用）；
6. 最后再次 run_bug_validation。

禁止修改 tests。
"""

        elif (
            edited
            and syntax_passed
            and not tests_passed
            and no_tests_seen
        ):

            continuation = """
当前项目没有可运行的现有 pytest 测试（run_test 已返回 NO_TESTS）。

NO_TESTS 不表示修复失败。
禁止再次调用 run_test，禁止猜测或寻找不存在的测试文件。

重新修改并通过语法检查后，现在必须直接调用：

run_bug_validation

最终任务是否成功，以 run_bug_validation 的真实返回结果为准。
不要结束任务。
"""

        elif edited and syntax_passed and not relevant_tests_passed:

            continuation = """
代码已经修改，并且语法检查已经通过。

现在必须调用：

run_test

使用与问题直接相关的测试。
如果 run_test 失败且失败与本次修改相关，先修复源代码再重试。
不要结束任务。
"""

        elif (
            edited
            and syntax_passed
            and relevant_tests_passed
            and validation_created
            and not validation_passed
        ):

            continuation = """
代码已经修改、语法检查通过、pytest 也通过。

当前 Workspace 存在通用 Bug 回归验证。
现在必须调用：

run_bug_validation

必须先创建临时验证测试，然后运行 run_bug_validation。
不要结束任务。
"""

        else:

            continuation = """
当前任务尚未完成。

请继续使用工具调查。

如果已经找到相关代码，
请使用 read_file 阅读真实代码。

如果已经能够确定 Bug 原因，
必须调用 replace_lines。

修改后：

check_syntax
↓
run_test

不要只输出解释。
"""

        messages.append(
            {
                "role": "user",
                "content": continuation
            }
        )

        continue

    # ========================================================
    # 执行工具调用
    # ========================================================

    for call in tool_calls:

        tool_name = call["function"]["name"]

        raw_arguments = call["function"].get(
            "arguments",
            {}
        )

        try:

            arguments = arguments_to_dict(
                raw_arguments
            )

        except Exception as e:

            result = (
                f"工具参数解析失败："
                f"{type(e).__name__}: {e}"
            )

            print(
                f"\n调用工具：{tool_name}"
            )

            print(
                f"参数解析失败：{e}"
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id") or "",
                    "content": result
                }
            )

            continue

        print(
            f"\n调用工具：{tool_name}"
        )

        print(
            f"参数：{arguments}"
        )

        # ====================================================
        # 连续 read_file 计数：调用其他工具即清零
        # ====================================================

        if tool_name != "read_file":
            consecutive_reads = 0

        # ====================================================
        # Runtime Guard 0
        # 完全相同的工具调用禁止无限重复
        # ====================================================

        call_key = (
            tool_name,
            json.dumps(
                arguments,
                sort_keys=True,
                ensure_ascii=False
            )
        )

        if tool_call_counts.get(call_key, 0) >= 3:

            result = {
                "status": "BLOCKED",
                "message": (
                    f"已阻止：这是第 "
                    f"{tool_call_counts[call_key] + 1} 次"
                    f"用完全相同的参数调用 {tool_name}，"
                    "结果不会发生任何变化。\n"
                    "必须改变策略，不要重复相同的调用。"
                    + repeat_strategy_hint()
                )
            }

            print(
                "\n⚠️ Runtime Guard："
                "已阻止完全重复的工具调用"
            )

        # ====================================================
        # Runtime Guard 1
        # 防止无限 retrieve_code
        # ====================================================

        elif tool_name == "retrieve_code":

            query_key = normalize_query(arguments.get("query", ""))

            if query_key and query_key in retrieved_queries:

                result = {
                    "status": "BLOCKED",
                    "message": (
                        "这个 RAG 查询已经成功执行过，禁止重复检索。\n"
                        "请直接使用已有检索结果，并执行 CURRENT AGENT STATE 中的下一步动作："
                        f" {next_required_action()}"
                    )
                }
                print("\n⚠️ Runtime Guard：重复 RAG 查询已阻止")

            elif evidence_ready:

                if not validation_created:
                    next_hint = (
                        "1. create_bug_validation：根据 problem statement "
                        "描述的具体场景写最小复现测试\n"
                        "   （现有测试大概率覆盖不到该 Bug，"
                        "需要自己构造触发输入）；\n"
                        "2. run_bug_validation：修改前 baseline 必须 FAIL。"
                    )
                elif not validation_baseline_confirmed:
                    next_hint = (
                        "1. run_bug_validation：确认修改前 baseline FAIL；\n"
                        "   若返回 BROKEN_TEST，按提示修正测试后"
                        "重新 create_bug_validation。"
                    )
                else:
                    next_hint = (
                        "replace_lines 修改源码"
                        " → check_syntax → run_test。"
                    )

                result = f"""
[RUNTIME BLOCK]

已经通过 read_file / retrieve_code 获得真实代码证据，
继续检索已被禁止（重复调用仍会被拒绝）。

当前必须：
{next_hint}

注意：文件路径必须使用工具返回的真实路径，
不要假设目录布局，以 list_files / read_file 的真实返回为准。
"""

                print(
                    "\n⚠️ Runtime Guard："
                    "已阻止重复 retrieve_code"
                )

            elif retrieve_count >= 3:

                result = """
[RUNTIME BLOCK]

retrieve_code 已达上限，
继续检索已被禁止（重复调用仍会被拒绝）。

请根据已有检索结果直接行动：
1. read_file 阅读检索结果中出现的源码文件
   （必须使用工具返回的真实路径，不要猜目录布局）；
2. 分析 Bug 根因；
3. create_bug_validation 写最小复现测试；
4. run_bug_validation 确认修改前 baseline FAIL。
"""

                print(
                    "\n⚠️ Runtime Guard："
                    "retrieve_code 已达到上限"
                )

            else:

                tool = tools_map.get(
                    tool_name
                )

                if tool is None:

                    result = (
                        f"未找到工具：{tool_name}"
                    )

                else:

                    try:

                        result = tool(
                            **arguments
                        )

                    except Exception as e:

                        result = (
                            "工具执行失败："
                            f"{type(e).__name__}: {e}"
                        )

                # 只有真正执行成功才计数
                if (
                    isinstance(result, str)
                    and not result.startswith(
                        "工具执行失败"
                    )
                ):
                    retrieve_count += 1
                    query_key = normalize_query(arguments.get("query", ""))
                    if query_key:
                        retrieved_queries.add(query_key)

                    # 多次成功检索 = 已获得真实代码证据。
                    # 即使 Agent 一直没成功 read_file，
                    # 也能激活验证引导机制。
                    if (
                        retrieve_count >= 3
                        and not evidence_ready
                    ):
                        evidence_ready = True

                        print(
                            "\n🔎 已通过多次检索获得关键代码证据"
                        )

                        print(
                            "下一步应进入 "
                            "create_bug_validation"
                        )

        # ====================================================
        # 其他工具
        # ====================================================

        else:

            allowed = allowed_tools_for_phase()

            # 冻结专属拦截必须优先于阶段白名单：
            # baseline 确认后 validation_frozen 恒为 True，此时阶段
            # 至少是 REPAIR，create_bug_validation 不在后期阶段白名单
            # 中——若先走阶段拦截，冻结专属文案永远不可达，且通用
            # 阶段文案会让 LLM 误以为只是阶段不对、稍后重试即可。
            # 这里对"冻结后重建验证测试"单独返回冻结专属 BLOCKED
            # 文案（调用本身依然被拒绝，不会放松任何白名单约束）。
            if (
                tool_name == "create_bug_validation"
                and validation_frozen
            ):
                result = {
                    "status": "BLOCKED",
                    "message": (
                        "Bug validation 已完成 baseline 并被冻结，"
                        "禁止重新创建或修改验证测试。"
                        "最终验证失败必须修复源代码。"
                    )
                }
                print(
                    "\n🔒 Runtime Guard："
                    "validation 已冻结，禁止重建验证测试"
                )
                tool = None

            elif tool_name not in allowed:
                result = {
                    "status": "BLOCKED",
                    "message": (
                        f"当前阶段 {current_phase()} 不允许调用 {tool_name}。"
                        f"当前唯一优先动作：{next_required_action()}。"
                    )
                }
                print(
                    f"\n⚠️ Runtime Guard：阶段 {current_phase()} 禁止 {tool_name}"
                )
                tool = None
            else:
                tool = tools_map.get(
                    tool_name
                )

            if tool_name not in allowed:
                pass
            elif tool is None:

                result = (
                    f"未找到工具：{tool_name}"
                )

            else:

                try:

                    # ========================================
                    # Runtime Guard 2
                    # 修改后必须先语法检查
                    # ========================================

                    if (
                        tool_name == "read_file"
                        and evidence_ready
                        and not validation_created
                        and consecutive_reads >= 4
                    ):
                        result = {
                            "status": "BLOCKED",
                            "message": (
                                "已经连续读取多个文件，但尚未创建 Bug 回归验证。"
                                "继续 read_file 已被暂时禁止。\n"
                                "当前状态：已获得代码证据，但未验证 Bug 是否可复现。\n"
                                "必须立即：\n"
                                "1. 根据 problem statement 描述的具体场景"
                                "（而不是现有测试），用 create_bug_validation "
                                "写一个最小复现测试；\n"
                                "2. run_bug_validation 确认 FAIL baseline。\n"
                                "注意：\n"
                                "- problem statement 描述的 Bug 通常没有现成测试覆盖，"
                                "按描述中的场景自己构造触发输入"
                                "是预期工作；\n"
                                "- 如果验证测试写错（返回 BROKEN_TEST），"
                                "根据 traceback 修正后重新 create_bug_validation 即可，"
                                "那时可以继续 read_file 查 API。"
                            )
                        }
                        print(
                            "\n⚠️ Runtime Guard："
                            "连续读取过多文件但仍未创建验证，"
                            "必须先做实验"
                        )

                    elif (
                        tool_name == "replace_lines"
                        and _is_validation_test_path(
                            arguments.get("path")
                        )
                    ):
                        # requests_6629 失败模式：最终验证 FAIL 后
                        # LLM 试图用编辑工具修改验证测试本身。
                        # 必须在 tools 层 PermissionError 之前
                        # 给出结构化 BLOCKED 消息，而不是异常字符串。
                        result = {
                            "status": "BLOCKED",
                            "message": (
                                "验证测试文件（.agent_validation/）只能通过 "
                                "create_bug_validation 整体重写，"
                                "禁止用 replace_lines 等编辑工具修改。\n"
                                "最终验证失败时，必须修复源代码，"
                                "而不是修改验证测试。"
                            )
                        }
                        print(
                            "\n⚠️ Runtime Guard："
                            "禁止用编辑工具修改验证测试文件"
                        )

                    elif (
                        tool_name == "replace_lines"
                        and not validation_created
                    ):
                        result = {
                            "status": "BLOCKED",
                            "message": "尚未创建临时 Bug 回归验证，不能修改源代码。必须先 create_bug_validation。"
                        }
                        print("\n⚠️ Runtime Guard：必须先建立通用 Bug 验证")

                    elif (
                        tool_name == "replace_lines"
                        and not validation_baseline_confirmed
                    ):
                        result = {
                            "status": "BLOCKED",
                            "message": "原始 Bug 尚未完成修改前 baseline 验证，不能修改源代码。必须先 run_bug_validation 并确认 FAIL。"
                        }
                        print("\n⚠️ Runtime Guard：必须先确认 Bug baseline")

                    elif (
                        tool_name == "create_bug_validation"
                        and validation_frozen
                    ):
                        # 永久冻结：baseline 确认后禁止重建验证测试。
                        # 该分支优先于下方的 edited 检查——即使 Agent
                        # 已修改代码，冻结状态也不允许重新创建。
                        result = {
                            "status": "BLOCKED",
                            "message": (
                                "Bug validation 已完成 baseline 并被冻结，"
                                "禁止重新创建或修改验证测试。"
                                "最终验证失败必须修复源代码。"
                            )
                        }
                        print(
                            "\n🔒 Runtime Guard："
                            "validation 已冻结，禁止重建验证测试"
                        )

                    elif (
                        tool_name == "create_bug_validation"
                        and edited
                        and not validation_baseline_confirmed
                    ):
                        result = {
                            "status": "BLOCKED",
                            "message": "代码已经修改，不能首次创建回归验证。必须在修改前建立 baseline。"
                        }
                        print("\n⚠️ Runtime Guard：必须在修改代码前建立 Bug 验证 baseline")

                    elif (
                        tool_name == "run_bug_validation"
                        and not validation_created
                    ):
                        result = {
                            "status": "BLOCKED",
                            "message": "尚未创建临时 Bug 验证测试，不能运行。"
                        }
                        print("\n⚠️ Runtime Guard：必须先 create_bug_validation")

                    elif (
                        tool_name == "run_bug_validation"
                        and not edited
                        and validation_created
                        and not validation_baseline_confirmed
                        and (validation_failed or validation_broken)
                    ):
                        if validation_broken:
                            reason = (
                                "上一次运行显示验证测试自身有错误"
                                "（BROKEN_TEST）"
                            )
                        else:
                            reason = (
                                "上一次运行显示验证测试在未修改的代码上"
                                "已经 PASS，没有捕捉到目标 Bug"
                            )

                        result = {
                            "status": "BLOCKED",
                            "message": (
                                f"已阻止重复运行：{reason}。\n"
                                "重复运行同一个测试文件，"
                                "结果不会发生任何变化。\n\n"
                                "必须重新调用 create_bug_validation，"
                                "创建修正或更强的验证测试：\n"
                                "1. 直接断言 problem statement 描述的"
                                "精确期望行为，"
                                "而不是宽松的不变量"
                                "（有 Bug 的代码也可能满足宽松断言）；\n"
                                "2. 按 problem statement 描述的具体场景"
                                "构造触发输入（边界条件、特定数据形态）；\n"
                                "3. 想观察真实行为：临时写一个必定失败的断言"
                                "（assert False, repr(实际结果)），"
                                "pytest 输出会显示真实值，"
                                "据此调整触发条件。\n\n"
                                "修正测试期间可以继续 read_file "
                                "查看相关 API。"
                            )
                        }
                        print(
                            "\n⚠️ Runtime Guard："
                            "已阻止重复运行无效的验证测试"
                        )

                    elif (
                        tool_name == "run_bug_validation"
                        and not edited
                        and validation_baseline_confirmed
                    ):
                        result = {
                            "status": "BLOCKED",
                            "message": "原始 Bug 已经完成 baseline 验证，修改代码后才能再次运行。"
                        }
                        print("\n⚠️ Runtime Guard：baseline 已确认，必须先修改代码")

                    elif (
                        tool_name == "run_bug_validation"
                        and edited
                        and not syntax_passed
                    ):
                        result = {
                            "status": "BLOCKED",
                            "message": "代码已经修改，但尚未通过 check_syntax。"
                        }
                        print("\n⚠️ Runtime Guard：禁止跳过语法检查")

                    elif (
                        tool_name == "run_bug_validation"
                        and edited
                        and validation_baseline_confirmed
                        and validation_failed
                    ):
                        # 最终验证失败后，如果没有成功的源代码修改
                        # （replace_lines SUCCESS 会重置 validation_failed），
                        # 重复运行验证测试结果不会变化。验证测试已被冻结，
                        # 唯一出路是修复源代码。
                        result = {
                            "status": "BLOCKED",
                            "message": (
                                "上一次最终验证失败，且此后没有修改源代码。"
                                "验证测试已被冻结，重复运行结果不会变化。\n"
                                "必须先用 replace_lines 修复源代码"
                                "（每次成功修改后可再次运行验证），"
                                "或运行 run_test 检查现有测试。"
                            )
                        }
                        print(
                            "\n⚠️ Runtime Guard："
                            "最终验证失败后必须先修复源代码"
                        )

                    elif (
                        tool_name == "run_test"
                        and edited
                        and not syntax_passed
                    ):

                        result = {
                            "status": "BLOCKED",
                            "message": (
                                "代码已经修改，"
                                "但尚未通过 check_syntax。"
                                "必须先调用 check_syntax，"
                                "再运行测试。"
                            )
                        }

                        print(
                            "\n⚠️ Runtime Guard："
                            "禁止修改后跳过语法检查"
                        )

                    elif (
                        tool_name == "run_test"
                        and no_tests_seen
                    ):

                        result = {
                            "status": "BLOCKED",
                            "message": (
                                "run_test 此前已返回 NO_TESTS："
                                "项目没有可运行的现有 pytest 测试。\n"
                                "NO_TESTS 不是修复失败，"
                                "但禁止再次调用 run_test，"
                                "禁止猜测或寻找 test.py / tests/ 等测试文件。\n"
                                "必须立即调用 run_bug_validation，"
                                "验证修改后的真实代码。"
                            )
                        }

                        print(
                            "\n⚠️ Runtime Guard："
                            "NO_TESTS 后禁止再次调用 run_test"
                        )

                    else:

                        result = tool(
                            **arguments
                        )

                except Exception as e:

                    result = (
                        "工具执行失败："
                        f"{type(e).__name__}: {e}"
                    )

        # ====================================================
        # 统计真实执行的工具调用（识别完全重复）
        # ====================================================

        blocked_result = (
            isinstance(result, dict)
            and result.get("status") == "BLOCKED"
        )

        failed_result = (
            isinstance(result, str)
            and (
                result.startswith("工具执行失败")
                or result.startswith("[RUNTIME BLOCK]")
            )
        )

        if blocked_result:
            # 被拦截的调用不计数，不消耗重试预算
            pass
        else:
            # 成功与失败都计数：
            # 相同参数的调用无论成败，重复执行结果都不会变化。
            # 尤其是失败调用（如 PermissionError / 参数错误），
            # LLM 盲目重试相同调用只会浪费轮次。
            tool_call_counts[call_key] = (
                tool_call_counts.get(call_key, 0) + 1
            )

        # ====================================================
        # 处理 create_bug_validation
        # ====================================================

        if tool_name == "create_bug_validation":
            if isinstance(result, dict) and result.get("status") == "CREATED":
                # 注意：这里绝不触碰 validation_frozen。
                # 冻结是永久状态，且上游守卫在冻结后根本不会放行
                # create_bug_validation，此分支只在 baseline 之前可达。
                validation_created = True
                validation_baseline_confirmed = False
                validation_passed = False
                validation_failed = False
                validation_broken = False
                validation_is_probe = bool(
                    result.get("is_probe")
                )
                # 新测试文件使 run_bug_validation 重新有意义
                tool_call_counts.clear()
                print("\n🧪 已创建通用 Bug 回归验证")
            elif isinstance(result, dict) and result.get("status") == "BROKEN_TEST":
                # 预检发现测试无法被 pytest 收集（导入/语法错误）：
                # 文件已写入但运行必然失败，按 validation_broken 流转
                validation_created = True
                validation_baseline_confirmed = False
                validation_passed = False
                validation_failed = False
                validation_broken = True
                validation_is_probe = False
                print(
                    "\n⚠️ 验证测试创建失败：预检发现导入/语法错误"
                    "（BROKEN_TEST）"
                )
            elif isinstance(result, dict) and result.get("status") == "BLOCKED":
                # Runtime Guard（冻结守卫 / 阶段白名单等）拦截产生的
                # BLOCKED 不是一次新的创建结果：绝不能落入下方 else
                # 分支把 validation_created 等状态重置——否则一旦冻结，
                # create_bug_validation 永远被拦截、validation_created
                # 永远被清空、run_bug_validation 又被"尚未创建"守卫
                # 拦截，任务陷入无法完成的死锁（livelock）。
                # 被拦截的调用不改变任何验证状态。
                pass
            else:
                validation_created = False
                validation_baseline_confirmed = False
                validation_passed = False
                validation_failed = False
                validation_broken = False
                validation_is_probe = False

        # ====================================================
        # 处理 run_bug_validation
        # ====================================================

        if tool_name == "run_bug_validation":
            if isinstance(result, dict):
                status = result.get("status")

                if status == "BLOCKED":
                    # Runtime Guard（尚未创建验证 / 阶段白名单等）拦截
                    # 产生的 BLOCKED 不是真实验证结果：保持现有验证
                    # 状态不变，绝不能落入下方分支把它当作
                    # "回归验证未通过"——那会虚假翻转
                    # validation_failed / tests_passed，
                    # 误导 current_phase 回退、破坏完成条件。
                    pass

                # 修改前失败 = 原始 Bug 成功复现，这是有效证据。
                elif not edited and status == "BROKEN_TEST":
                    validation_baseline_confirmed = False
                    validation_passed = False
                    validation_failed = False
                    validation_broken = True
                    print(
                        "\n❌ 验证测试自身有错误（"
                        + str(result.get("error_types"))
                        + "，非 AssertionError），baseline 无效。"
                        "必须先修正验证测试，再确认 baseline。"
                    )

                elif not edited and status == "FAIL":

                    failure_kind = result.get("failure_kind")

                    if failure_kind in ("test_error", "collection_error"):
                        # baseline 之前：验证测试自身抛出 TypeError /
                        # ImportError 等真实异常，或收集/导入失败。
                        # 这不是 Bug 复现证据（修复方案应让测试用
                        # pytest.raises 显式断言预期异常），按
                        # BROKEN_TEST 流转，必须修正测试本身。
                        # 注意：baseline 确认之后（冻结状态）绝不走
                        # 这个分支——下方 edited 分支保证任何 FAIL
                        # 都保持 FAIL，走修复源代码路径。
                        validation_baseline_confirmed = False
                        validation_passed = False
                        validation_failed = False
                        validation_broken = True

                        result["guidance"] = (
                            "验证测试失败原因是测试自身异常"
                            "（" + str(result.get("error_types")) + "），"
                            "不是对 Bug 的断言失败。\n"
                            "如果该异常本身就是目标 Bug 的表现，"
                            "请在测试中用 pytest.raises(异常类型) "
                            "显式断言，让复现结果变成 AssertionError。\n"
                            "必须修正验证测试后重新 create_bug_validation。"
                        )

                        print(
                            "\n❌ 验证测试自身有错误（"
                            + str(result.get("error_types"))
                            + "），baseline 无效。"
                            "必须先修正验证测试，再确认 baseline。"
                        )

                    elif validation_is_probe:
                        # 探针测试（assert False）的失败是设计使然，
                        # 只用于展示真实输出，不是有效的 Bug 复现证据
                        validation_baseline_confirmed = False
                        validation_passed = False
                        validation_failed = True
                        validation_broken = False

                        result["guidance"] = (
                            "探针测试已完成使命：上方输出中的"
                            "断言失败信息展示了真实行为"
                            "（assert False 的消息里带有 repr 值）。\n"
                            "现在请基于这些真实值，"
                            "用精确的断言重写验证测试"
                            "（把 assert False 替换成对期望输出的断言），"
                            "重新调用 create_bug_validation，"
                            "然后 run_bug_validation 确认真正的 baseline。"
                        )

                        print(
                            "\n🔍 探针测试已输出真实行为，"
                            "等待重写为真实断言测试"
                        )

                    else:
                        # AssertionError 型 FAIL = 原始 Bug 成功复现。
                        # 同时永久冻结：此后禁止重建/修改验证测试
                        # （Runtime Guard 依据 validation_frozen 拦截），
                        # 任何后续验证 FAIL 都只能通过修复源代码解决。
                        validation_baseline_confirmed = True
                        validation_frozen = True
                        validation_passed = False
                        validation_failed = False
                        validation_broken = False
                        print("\n✅ 已确认原始 Bug 可以复现")
                        print(
                            "🔒 验证测试已冻结："
                            "禁止重建或修改，"
                            "后续验证失败只能修复源代码"
                        )

                elif not edited and status == "PASS":
                    validation_baseline_confirmed = False
                    validation_passed = False
                    validation_failed = True

                    result["guidance"] = (
                        "验证测试在未修改的代码上就 PASS，"
                        "说明它没有捕捉到目标 Bug，"
                        "重复运行同一个测试不会有任何变化。\n"
                        "必须重新 create_bug_validation 写更强的测试：\n"
                        "1. 直接断言 problem statement 描述的精确期望行为，"
                        "而不是宽松的不变量"
                        "（有 Bug 的代码也可能满足宽松断言）；\n"
                        "2. 按 problem statement 描述的具体场景"
                        "构造触发输入（边界条件、特定数据形态）；\n"
                        "3. 想观察真实行为：临时写必定失败的断言"
                        "（assert False, repr(实际结果)），"
                        "pytest 输出会显示真实值，据此调整。"
                    )

                    print("\n⚠️ 回归验证在修改前就通过，无法证明当前问题确实存在")

                elif edited and not syntax_passed:
                    validation_passed = False
                    validation_failed = True
                    print("\n⚠️ 修改后的代码尚未通过语法检查")

                elif edited and status == "PASS":
                    validation_passed = True
                    validation_failed = False
                    # 最终回归验证 PASS 即视为测试阶段完成：
                    # 重新修改代码后 tests_passed 会被 replace_lines 重置，
                    # 而 NO_TESTS 后 run_test 已被 Runtime 禁止，
                    # 必须在这里恢复 tests_passed，否则 DONE 永远无法达成。
                    tests_passed = True
                    # 同理必须恢复相关测试结论：replace_lines 会把
                    # relevant_tests_passed 重置为 False，而 NO_TESTS
                    # 之后 run_test 被禁止、验证测试又已冻结，
                    # 最终验证 PASS 就是"相关测试通过"的唯一证据源；
                    # 不在这里恢复，完成条件五元组永远无法满足。
                    relevant_tests_passed = True
                    unrelated_test_failures = []
                    print("\n✅ 通用 Bug 回归验证通过")

                else:
                    validation_passed = False
                    validation_failed = True
                    # 回归验证失败：current_phase 会回退到 REPAIR 重新修改，
                    # 同步重置 tests_passed 保持状态一致，
                    # 禁止在验证未通过时声称任务完成。
                    #
                    # 冻结语义（requests_6629 修复核心）：
                    # baseline 确认后，这里接收 edited 状态下的任何 FAIL
                    # ——包括 failure_kind 为 test_error / collection_error
                    # 的情况（验证测试已被冻结，导入错误只能来自源代码
                    # 结构被破坏）。绝不降级为 BROKEN_TEST，绝不引导
                    # Agent 修改验证测试；唯一出路是修复源代码。
                    tests_passed = False
                    print("\n❌ 通用 Bug 回归验证未通过：" + str(status))

        # ====================================================
        # 处理 read_file
        # ====================================================

        if tool_name == "read_file":

            path = arguments.get(
                "path",
                ""
            )

            if isinstance(result, str):

                if not result.startswith(
                    "工具执行失败"
                ):

                    read_files.add(
                        path
                    )

                    consecutive_reads += 1

                    if should_mark_evidence_ready(
                        path
                    ):

                        evidence_ready = True

                        if not validation_created:
                            next_step = (
                                "create_bug_validation"
                            )
                        elif validation_frozen:
                            # 验证测试已冻结：验证失败只能通过修改
                            # 源代码解决，不能引导重写验证测试
                            # （Runtime 会拦截 create_bug_validation）
                            next_step = (
                                "replace_lines（修复源代码）"
                            )
                        elif (
                            validation_failed
                            or validation_broken
                        ):
                            # 测试太宽松（修改前 PASS）或自身有错误：
                            # 下一步是重写更强的测试，不是重复运行
                            next_step = (
                                "create_bug_validation"
                                "（重写更强的验证测试）"
                            )
                        elif not validation_baseline_confirmed:
                            next_step = (
                                "run_bug_validation"
                            )
                        else:
                            next_step = (
                                "replace_lines"
                            )

                        print(
                            "\n🔎 已获得关键代码证据"
                        )

                        print(
                            "下一步应进入 "
                            f"{next_step}"
                        )

        # ====================================================
        # 处理代码修改工具
        # ====================================================

        if tool_name == "replace_lines":

            edit_success = (
                isinstance(result, dict)
                and result.get("status") == "SUCCESS"
            ) or (
                isinstance(result, str)
                and "修改成功" in result
            )

            if edit_success:

                edited = True
                tests_passed = False
                syntax_passed = False
                validation_passed = False
                validation_failed = False

                # 代码已变化：上一轮的测试相关性结论全部失效，
                # 必须重新 run_test 重新判定
                relevant_tests_passed = False
                unrelated_test_failures = []

                # 代码已变化，重新运行测试/验证有了新意义
                tool_call_counts.clear()

                last_edited_file = (
                    result.get("path")
                    if isinstance(result, dict)
                    else arguments.get("path")
                )

                print(
                    "\n✅ 已确认实际修改代码"
                )

                print(
                    f"修改文件："
                    f"{last_edited_file}"
                )

            else:

                print(
                    "\n❌ replace_lines "
                    "没有返回成功结果"
                )

        # ====================================================
        # 处理 check_syntax
        # ====================================================

        if tool_name == "check_syntax":

            if isinstance(result, dict):

                status = result.get(
                    "status"
                )

                if status == "PASS" and edited:

                    syntax_passed = True

                    print(
                        "\n✅ 语法检查通过"
                    )

                elif status == "PASS":

                    # 修改前的语法检查不构成“修改后语法通过”的证据
                    syntax_passed = False

                    print(
                        "\n⚠️ 语法检查通过，"
                        "但代码尚未修改，不作为修改后证据"
                    )

                else:

                    syntax_passed = False

                    print(
                        "\n❌ 语法检查未通过："
                        f"{status}"
                    )

            else:

                syntax_passed = False

        # ====================================================
        # 处理 run_test
        # ====================================================

        if tool_name == "run_test":

            if isinstance(result, dict):

                status = result.get(
                    "status"
                )

                if status == "NO_TESTS":

                    # NO_TESTS ≠ 修复失败：
                    # 它只说明项目没有可运行的现有 pytest 测试。
                    # 修改后代码一旦通过语法检查，测试阶段立即视为完成，
                    # Runtime 直接进入 FINAL_VALIDATE，
                    # 之后由守卫禁止再次调用 run_test。

                    if edited and syntax_passed:
                        no_tests_seen = True
                        tests_passed = True
                        # 没有可运行的现有测试 = 不存在"相关测试失败"，
                        # 相关测试视为通过，完成条件交给最终验证
                        relevant_tests_passed = True
                        unrelated_test_failures = []

                        result["guidance"] = (
                            "当前项目没有可运行的现有 pytest 测试。"
                            "这不表示代码修复失败。"
                            "项目测试阶段视为完成。"
                            "禁止再次调用 run_test，"
                            "禁止猜测或寻找不存在的测试文件。\n"
                            "现在必须立即调用 run_bug_validation，"
                            "验证修改后的真实代码。"
                            "最终成功以 run_bug_validation 的真实结果为准。"
                        )

                        print(
                            "\n⚠️ 项目没有现有测试，"
                            "转入最终 Bug 回归验证"
                        )

                    else:
                        tests_passed = False
                        relevant_tests_passed = False

                elif status == "PASS":

                    # ----------------------------------------
                    # 没有修改代码
                    # ----------------------------------------

                    if not edited:

                        tests_passed = False

                        result["guidance"] = (
                            "注意：现有测试通过 ≠ 目标 Bug 不存在。\n"
                            "回归测试通常在修复 PR 中才加入，"
                            "当前 Workspace 的测试大概率覆盖不到 "
                            "problem statement 描述的缺陷场景。\n"
                            "如果这个 Bug 需要特定的触发条件"
                            "（problem statement 描述的具体场景），"
                            "现有测试通过是正常的。\n"
                            "下一步：根据 problem statement 描述的具体场景，"
                            "用 create_bug_validation 构造最小复现断言，"
                            "然后 run_bug_validation 确认 FAIL baseline。\n"
                            "禁止用继续 read_file 回避实验。"
                        )

                        print(
                            "\n⚠️ 测试通过，"
                            "但 Agent 没有实际修改代码。"
                        )

                    # ----------------------------------------
                    # 修改了但没有语法检查
                    # ----------------------------------------

                    elif not syntax_passed:

                        tests_passed = False
                        relevant_tests_passed = False

                        print(
                            "\n⚠️ 测试通过，"
                            "但修改后的代码没有经过 "
                            "check_syntax。"
                        )

                    # ----------------------------------------
                    # 正常
                    # ----------------------------------------

                    else:

                        tests_passed = True
                        relevant_tests_passed = True
                        unrelated_test_failures = []

                        print(
                            "\n✅ 测试通过"
                        )

                        print(
                            "✅ 已实际修改代码"
                        )

                        print(
                            "✅ 已通过语法检查"
                        )

                else:

                    # FAIL / TIMEOUT 等未通过结果：
                    # 判定失败与本次修改的相关性。
                    # tools 层在 FAIL 时附带 failures（失败测试条目）
                    # 与 frame_paths（FAILURES 段 traceback 中的源码帧）。
                    failures = result.get("failures") or []
                    frame_paths = result.get("frame_paths") or []

                    # 判定材料不足（非 FAIL、没有失败条目、
                    # 会话恢复后丢失被修改文件）时，
                    # 保守按"相关失败"处理，绝不放过该阻断的情况。
                    failure_is_relevant = True

                    if failures and last_edited_file:
                        # 被修改文件出现在任何一个失败的 traceback
                        # 调用链里 → 失败与本次修改相关；
                        # 一个都不出现 → 同进程 pytest 语义下
                        # 本次修改不可能导致这些失败 → 无关失败。
                        failure_is_relevant = any(
                            _frame_matches_edited_file(
                                frame, last_edited_file
                            )
                            for frame in frame_paths
                        )

                    if failure_is_relevant:

                        tests_passed = False
                        relevant_tests_passed = False
                        unrelated_test_failures = []

                        result["failure_relevance"] = "RELEVANT"

                        print(
                            "\n❌ 测试未通过："
                            f"{status}"
                        )

                    else:

                        # 无关失败：只记录，不阻断完成条件。
                        # relevant_tests_passed 置 True 的含义是
                        # "与本次修改相关的测试子集没有失败"。
                        tests_passed = False
                        relevant_tests_passed = True
                        unrelated_test_failures = [
                            entry.get("test_id", "?")
                            for entry in failures
                        ]

                        result["failure_relevance"] = "UNRELATED"

                        result["guidance"] = (
                            f"现有测试有 {len(unrelated_test_failures)} 个失败，"
                            "但失败 traceback 的调用链中没有出现本次修改的文件"
                            f"（{last_edited_file}）——"
                            "这些失败与本次修改无关，不阻断修复流程。\n"
                            "不要修复这些无关失败，更不要修改对应测试。\n"
                            "下一步：继续 Bug 验证流程"
                            "（最终验证 run_bug_validation）。"
                        )

                        print(
                            "\n⚠️ 测试存在失败，"
                            "但均与本次修改无关"
                            f"（{len(unrelated_test_failures)} 个），"
                            "不阻断修复，继续最终验证"
                        )

            else:

                tests_passed = False
                relevant_tests_passed = False

        # ====================================================
        # 连续失败计数：结果以“工具执行失败”开头视为失败
        # ====================================================

        if (
            isinstance(result, str)
            and result.startswith("工具执行失败")
        ):
            consecutive_failures += 1

            if (
                "FileNotFoundError" in result
                or "NotADirectoryError" in result
            ):
                consecutive_path_failures += 1
            else:
                consecutive_path_failures = 0
        else:
            consecutive_failures = 0
            consecutive_path_failures = 0

        # ====================================================
        # 把工具结果交给 LLM
        # ====================================================

        # 保留短的动作轨迹，供下一轮 CURRENT AGENT STATE 使用。
        # dict 结果用 JSON 序列化（LLM 更好读，也和 tool calling 惯例一致）。
        if isinstance(result, dict):
            result_text = json.dumps(
                result,
                ensure_ascii=False,
                default=str
            )
        else:
            result_text = str(result)
        status_text = result.get("status") if isinstance(result, dict) else None
        action_summary = f"{tool_name}({json.dumps(arguments, ensure_ascii=False, sort_keys=True)})"
        if status_text:
            action_summary += f" -> {status_text}"
        elif result_text.startswith("工具执行失败"):
            action_summary += " -> ERROR"
        elif result_text.startswith("[RUNTIME BLOCK]"):
            action_summary += " -> BLOCKED"
        recent_actions.append(action_summary[:500])
        del recent_actions[:-8]

        # 只截断进入 LLM 上下文的内容；
        # 上方 print(result)（stdout / Web UI 解析）仍是完整结果
        result_text = truncate_tool_result(result_text)

        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.get("id") or "",
                "content": result_text
            }
        )

        print(
            "\n工具结果："
        )

        print(result)

    # ========================================================
    # 当前轮结束后，根据状态向 Agent 强化下一步
    # ========================================================

    if consecutive_path_failures >= 3 and not edited:

        messages.append(
            {
                "role": "user",
                "content": f"""
你已经连续 {consecutive_path_failures} 次因为文件路径不存在而失败，
原因几乎都是凭空猜测路径（如假设特定目录布局）。

请立即停止猜测：
1. 以 list_files 返回的真实目录结构为准，
   源码位置因项目而异（可能在根目录，也可能在 src/ 下）；
2. 错误信息中已经给出真实路径或可用测试目标，
   直接使用它们，不要换一个新路径再猜；
3. 上文 retrieve_code 已返回相关源码，
   可以直接基于它行动。
"""
            }
        )

    elif evidence_ready and not edited and not validation_created:

        messages.append(
            {
                "role": "user",
                "content": """
已经获得足够的真实代码证据。
现在不能修改源代码。

必须先：
create_bug_validation
→
run_bug_validation

修改前验证必须 FAIL，确认原始 Bug 可以复现后，
才能调用 replace_lines。

注意：problem statement 描述的 Bug 通常没有现成测试覆盖，
按描述中的场景自己构造输入是预期工作。
不要用继续 read_file 回避实验。
"""
            }
        )

    elif evidence_ready and not edited and validation_created and not validation_baseline_confirmed:

        if validation_broken:

            messages.append(
                {
                    "role": "user",
                    "content": """
当前验证测试自身存在错误（BROKEN_TEST），
调用不存在的 API 或参数错误，无论源代码是否修复都会失败。

必须重新调用 create_bug_validation，
用真实存在的 API 重写验证测试，
然后 run_bug_validation 确认 FAIL baseline。

修正测试时可以继续 read_file 查看真实 API 签名。
"""
                }
            )

        elif validation_failed:

            messages.append(
                {
                    "role": "user",
                    "content": """
当前验证测试没有捕捉到目标 Bug：
在未修改的代码上运行结果是 PASS。
重复调用 run_bug_validation 没有意义，会被直接阻止。

必须重新调用 create_bug_validation，写更强的测试：
1. 直接断言 problem statement 描述的精确期望行为，
   而不是宽松的不变量（有 Bug 的代码也可能满足宽松断言）；
2. 按 problem statement 描述的具体场景构造触发输入
   （边界条件、特定数据形态）；
3. 想看真实行为：临时写必定失败的断言
   （assert False, repr(结果)），
   pytest 输出会显示真实值，据此调整。
"""
                }
            )

        else:

            messages.append(
                {
                    "role": "user",
                    "content": """
临时 Bug 验证已经创建，但 baseline 尚未确认。
现在必须调用 run_bug_validation。
预期结果为 FAIL。
确认 baseline 后才能修改源代码。
"""
                }
            )

    elif evidence_ready and not edited and validation_created and validation_baseline_confirmed:

        messages.append(
            {
                "role": "user",
                "content": """
原始 Bug 已确认可以复现。
现在必须调用 replace_lines 修改真实源代码。
修改后严格执行：
check_syntax
→ run_test
→ run_bug_validation
"""
            }
        )

    elif edited and not syntax_passed:

        messages.append(
            {
                "role": "user",
                "content": """
代码已经实际修改。

下一步必须：

check_syntax

不要运行测试，不要结束任务。
"""
            }
        )

    elif validation_failed:

        messages.append(
            {
                "role": "user",
                "content": """
通用 Bug 回归验证失败。

当前修改没有真正修复原始 Bug。
请根据回归检查的失败输出重新分析，不要重复无脑运行同一个检查。

必要时重新 read_file / retrieve_code，
然后必须调用 replace_lines。

修改后重新执行：
check_syntax
→ run_test（若 run_test 曾返回 NO_TESTS，跳过此步并禁止再调用）
→ run_bug_validation

不要修改 tests。
"""
            }
        )

    elif (
        edited
        and syntax_passed
        and not relevant_tests_passed
        and no_tests_seen
    ):

        messages.append(
    {
        "role": "user",
        "content": """
当前项目没有可运行的现有 pytest 测试（run_test 已返回 NO_TESTS）。

NO_TESTS 不表示修复失败。
禁止再次调用 run_test，禁止猜测或寻找不存在的测试文件。

重新修改并通过语法检查后，现在必须直接调用：

run_bug_validation

最终任务是否成功，以 run_bug_validation 的真实返回结果为准。
不要结束任务。
"""
    }
)

    elif edited and syntax_passed and not tests_passed:

        messages.append(
    {
        "role": "user",
        "content": """
代码已经修改，并且语法检查通过。

现在必须完成测试阶段：

1. 如果 Workspace 中存在与 Bug 直接相关的现有 pytest，
   调用 run_test。
2. 如果 run_test 返回 NO_TESTS，
   不要继续猜测测试文件或重复调用 run_test。
   当前项目没有现有测试，这是允许的情况。
3. 此时应直接调用 run_bug_validation，
   用已经创建的临时 Bug 回归测试验证修改后的真实代码。

不要因为 NO_TESTS 无限寻找不存在的测试。
"""
    }
)

    elif (
        edited
        and syntax_passed
        and relevant_tests_passed
        and validation_created
        and not validation_passed
    ):

        messages.append(
            {
                "role": "user",
                "content": """
代码已经修改、语法检查通过、pytest 已经通过。

当前 Workspace 已经存在通用 Bug 回归验证。
现在必须再次调用：

run_bug_validation

这次验证的是修改后的真实代码。
不要结束任务。
"""
            }
        )


# ============================================================
# 最大轮数结束
# ============================================================

else:

    print(
        "\n========== 达到最大轮数 =========="
    )

    if (edited and syntax_passed and relevant_tests_passed
                and validation_created
                and validation_baseline_confirmed and validation_passed):

        print(
            "Agent 已实际修改代码，"
            "并且测试通过。"
        )

    elif edited:

        print(
            "Agent 已实际修改代码，"
            "但尚未获得有效的最终测试通过结果。"
        )

    else:

        print(
            "Agent 在限定轮数内没有实际修改代码。"
        )

# ============================================================
# 清理临时 Bug 验证文件与会话文件
# （修复完成/达到最大轮数 → 清理；
#   LLM 故障退出 → 全部保留：验证文件是会话状态的一部分，
#   断点恢复后 run_bug_validation 仍要使用它）
# ============================================================

if not llm_failed:

    cleanup_result = cleanup_bug_validation()
    if cleanup_result.get("removed"):
        print("\n🧹 已清理临时 Bug 回归验证文件")

    if os.path.isfile(session_path):
        os.remove(session_path)
        print("🧹 已清理会话文件")


