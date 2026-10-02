import json

from llm_client import chat
from tools import tools_map, set_workspace
from task_loader import get_task
from workspace_manager import create_workspace


# ============================================================
# Agent 配置
# ============================================================

MAX_ROUNDS = 15

edited = False
tests_passed = False
syntax_passed = False

# 当前修改后的文件
last_edited_file = None

# Code RAG / 代码阅读状态
retrieve_count = 0
read_files = set()

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
                            "例如 src/example.py 或 tests/test_example.py"
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
    # replace_in_file
    # --------------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "replace_in_file",
            "description": (
                "对当前 Workspace 中的源代码执行一次精确文本替换。"
                "old 必须来自刚刚读取的真实文件，并且只能出现一次。"
                "禁止修改 tests 目录。"
                "不要重写整个文件，只修改必要的局部代码。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Workspace 内的相对文件路径"
                        )
                    },
                    "old": {
                        "type": "string",
                        "description": (
                            "需要被替换的原始代码。"
                            "必须与真实文件中的代码完全一致。"
                        )
                    },
                    "new": {
                        "type": "string",
                        "description": (
                            "替换后的实际代码。"
                            "只包含必要的局部修改。"
                        )
                    }
                },
                "required": [
                    "path",
                    "old",
                    "new"
                ]
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
                            "可选。pytest -k 表达式，"
                            "例如 pickle 或 json_decode。"
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
- replace_in_file
- check_syntax
- run_test

必须通过真实工具完成工作。

禁止：

- 伪造工具调用
- 在文本中模拟工具执行
- 声称已经修改代码但没有调用 replace_in_file
- 声称测试通过但没有真实 run_test 结果
- 根据自己的推理直接宣布任务成功

============================================================
三、严格工具真实性
============================================================

如果没有调用：

replace_in_file

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
replace_in_file

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

尤其是问题描述已经明确指出：

- pickle
- __reduce__
- MRO
- JSONDecodeError

等机制时，

一旦你阅读：

src/requests/exceptions.py

并看到：

JSONDecodeError

就应该分析并修改。

不能继续无休止搜索。

============================================================
六、当前任务的强制规则
============================================================

如果当前问题同时包含：

pickle
__reduce__
MRO
JSONDecodeError

并且你已经读取：

src/requests/exceptions.py

那么：

下一步必须调用：

replace_in_file

禁止：

- 再次 retrieve_code
- 重复搜索 JSONDecodeError
- 重复搜索 pickle
- 重复运行不能验证 Bug 的测试
- 只描述修改方案

你必须真正修改代码。

============================================================
七、修改代码
============================================================

修改代码必须使用：

replace_in_file

参数只有：

path
old
new

old 必须来自真实 read_file 结果。

old 必须与真实文件中的内容完全一致。

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

replace_in_file
↓
check_syntax
↓
run_test

不能：

replace_in_file
↓
直接宣布成功

也不能：

replace_in_file
↓
run_test

必须先：

check_syntax

============================================================
九、测试规则
============================================================

run_test 的 test_filter 是 pytest -k 表达式。

例如：

pickle

或者：

json_decode

正确：

{
    "test_path": "tests/test_requests.py",
    "test_filter": "pickle"
}

错误：

{
    "test_filter": "pickle -v"
}

不要把：

-v
--verbose
--maxfail
等命令行参数放入 test_filter。

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

如果原问题是：

pickle.dumps
+
pickle.loads
+
JSONDecodeError

那么最终测试应该尽可能直接覆盖这个场景。

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
4. 成功调用 replace_in_file
5. replace_in_file 返回成功
6. 修改后的代码通过 check_syntax
7. 修改后的代码经过真实 run_test
8. run_test 返回 PASS
9. 测试与问题直接相关
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

replace_in_file
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

请自主调查并修复这个问题。

所有文件操作和测试操作必须针对当前 Workspace。
"""
    }
]


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
    根据当前任务和已读取文件判断是否已经拥有
    足够进入修改阶段的证据。

    当前专门强化 JSONDecodeError / pickle / MRO 类问题，
    同时保留通用逻辑。
    """

    normalized = path.replace("\\", "/").lower()

    # 当前 requests_6629 类任务
    if (
        "jsondecodeerror" in problem_statement.lower()
        and "pickle" in problem_statement.lower()
        and "__reduce__" in problem_statement.lower()
        and "mro" in problem_statement.lower()
    ):

        if normalized.endswith(
            "src/requests/exceptions.py"
        ):
            return True

    return False


# ============================================================
# Agent Loop
# ============================================================

for round_number in range(MAX_ROUNDS):

    print(
        f"\n========= 第 {round_number + 1} 轮 ========="
    )

    # --------------------------------------------------------
    # 如果已经修改 + 测试通过
    # --------------------------------------------------------

    if edited and tests_passed:

        print("\n========== 修复完成 ==========")
        print(
            "Agent 已实际修改代码，并获得测试通过结果。"
        )
        break

    # --------------------------------------------------------
    # 调用 LLM
    # --------------------------------------------------------

    try:

        response = chat(
            messages,
            tools=tools
        )

    except Exception as e:

        print(
            "\n========== LLM 调用失败 =========="
        )
        print(
            f"{type(e).__name__}: {e}"
        )
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

        if edited and tests_passed:

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
        if evidence_ready and not edited:

            continuation = """
你已经拥有足够的代码证据。

现在禁止继续 retrieve_code。

必须立即调用：

replace_in_file

然后：

check_syntax
↓
run_test

不要只输出修改方案。
不要解释你准备怎么修改。
必须真实调用 replace_in_file。
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

        elif edited and syntax_passed and not tests_passed:

            continuation = """
代码已经修改，并且语法检查已经通过。

现在必须调用：

run_test

不要结束任务。
"""

        else:

            continuation = """
当前任务尚未完成。

请继续使用工具调查。

如果已经找到相关代码，
请使用 read_file 阅读真实代码。

如果已经能够确定 Bug 原因，
必须调用 replace_in_file。

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
                    "content": result,
                    "tool_name": tool_name
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
        # Runtime Guard 1
        # 防止无限 retrieve_code
        # ====================================================

        if tool_name == "retrieve_code":

            if evidence_ready:

                result = """
[RUNTIME BLOCK]

当前 Agent 已经通过 read_file 获得与问题直接相关的真实代码。

继续 retrieve_code 已被禁止。

必须进入：

replace_in_file
→ check_syntax
→ run_test

不要继续搜索。
"""

                print(
                    "\n⚠️ Runtime Guard："
                    "已阻止重复 retrieve_code"
                )

            elif retrieve_count >= 3:

                result = """
[RUNTIME BLOCK]

retrieve_code 已经连续使用多次，
但 Agent 尚未进行代码修改。

继续检索已被禁止。

请根据已有证据：

1. read_file 阅读真实代码
2. 分析 Bug
3. 调用 replace_in_file
4. check_syntax
5. run_test
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

        # ====================================================
        # 其他工具
        # ====================================================

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

                    # ========================================
                    # Runtime Guard 2
                    # 修改后必须先语法检查
                    # ========================================

                    if (
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

                    if should_mark_evidence_ready(
                        path
                    ):

                        evidence_ready = True

                        print(
                            "\n🔎 已获得关键代码证据"
                        )

                        print(
                            "下一步应进入 replace_in_file"
                        )

        # ====================================================
        # 处理 replace_in_file
        # ====================================================

        if tool_name == "replace_in_file":

            if (
                isinstance(result, str)
                and "修改成功" in result
            ):

                edited = True
                tests_passed = False
                syntax_passed = False

                last_edited_file = (
                    arguments.get("path")
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
                    "\n❌ replace_in_file "
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

                if status == "PASS":

                    syntax_passed = True

                    print(
                        "\n✅ 语法检查通过"
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

                if status == "PASS":

                    # ----------------------------------------
                    # 没有修改代码
                    # ----------------------------------------

                    if not edited:

                        tests_passed = False

                        print(
                            "\n⚠️ 测试通过，"
                            "但 Agent 没有实际修改代码。"
                        )

                    # ----------------------------------------
                    # 修改了但没有语法检查
                    # ----------------------------------------

                    elif not syntax_passed:

                        tests_passed = False

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

                    tests_passed = False

                    print(
                        "\n❌ 测试未通过："
                        f"{status}"
                    )

            else:

                tests_passed = False

        # ====================================================
        # 把工具结果交给 LLM
        # ====================================================

        messages.append(
            {
                "role": "tool",
                "content": str(result),
                "tool_name": tool_name
            }
        )

        print(
            "\n工具结果："
        )

        print(result)

    # ========================================================
    # 当前轮结束后，根据状态向 Agent 强化下一步
    # ========================================================

    if evidence_ready and not edited:

        messages.append(
            {
                "role": "user",
                "content": """
你已经获得了足够的真实代码证据。

现在进入修改阶段。

禁止继续 retrieve_code。

请立即调用：

replace_in_file

成功后：

check_syntax
→
run_test

必须真正修改代码，不要只描述方案。
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

    elif edited and syntax_passed and not tests_passed:

        messages.append(
            {
                "role": "user",
                "content": """
代码已经修改，并且语法检查通过。

现在必须调用：

run_test

使用与问题直接相关的测试。
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

    if edited and tests_passed:

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