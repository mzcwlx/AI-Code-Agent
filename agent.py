import ollama
from tools import tools_map
import os
from code_rag import build_code_rag,retrieve_code,format_results

max_rounds = 100
edited = False
tested_after_edit = False
tests_passed = False


tools = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取文件内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "文件路径"
                    }
                },
                "required": ["path"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "修改文件内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "要修改的文件路径"
                    },
                    "content": {
                        "type": "string",
                        "description": "修改后的完整文件内容"
                    }
                },
                "required": ["path", "content"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "run_test",
            "description": "运行当前项目中的所有pytest测试。该工具不需要任何参数。",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },

    {
    "type": "function",
    "function": {
        "name": "list_files",
        "description": "列出指定目录下的所有文件，用于了解项目结构。读取文件之前，应先通过此工具确认文件路径。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "要查看的目录路径，使用相对路径，例如 . 或 src ，默认为 ."
                }
            }  
        }
    }
},{
    "type": "function",
    "function": {
        "name": "retrieve_code",
        "description": "根据用户的问题，从当前项目代码中检索最相关的代码片段",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "用于检索代码的问题或描述"
                }
            },
            "required": ["query"]
        }
    }
},
]

content = input("请输入您的问题：")
messages = [{
    "role": "system",
    "content": """
你是一个软件缺陷自动修复 Agent。

当前工作目录就是项目根目录。

你的目标不是解释代码，而是：
找到真实问题 → 修改代码 → 运行测试 → 根据测试结果继续修复 → 直到测试通过。

请严格遵守以下规则：

1. 不要猜测项目结构。

2. 不知道某个文件在哪里时，必须先调用 list_files。

3. 只能读取 list_files 返回的实际文件路径。

4. 当用户的问题涉及具体代码时，如果不知道相关代码在哪里，可以使用 retrieve_code。

5. retrieve_code 返回的内容只能作为“检索线索”，不能直接作为最终判断依据。

6. 一旦 retrieve_code 找到了可能相关的文件：
   必须调用 read_file 读取真实文件内容。

7. 如果项目中存在对应的测试文件：
   必须调用 read_file 读取测试文件。

8. 在修改代码之前，必须尽可能获得以下信息：
   - 真实源代码
   - 相关测试代码
   - 当前测试运行结果

9. 不允许仅凭函数名称、RAG 检索结果或自己的猜测直接修改代码。

10. 如果还不能确定问题是什么，继续调用工具获取信息，而不是直接修改代码。

11. 修改代码后，必须调用 run_test。

12. run_test 返回失败时：
   - 仔细阅读测试输出
   - 找到失败的测试
   - 分析失败原因
   - 必要时再次 read_file
   - 修改代码
   - 再次 run_test

13. 如果测试仍然失败，继续修复。
   不要因为已经修改过一次代码就结束。

14. 只有当 run_test 明确显示所有测试通过后，才能认为修复完成。

15. 不要为了让测试通过而随意修改测试代码。
   默认情况下，测试代码是用来验证程序是否正确的。

16. 如果发现测试本身可能存在问题：
   必须先读取测试代码，并结合源代码和测试结果进行判断。
   不要仅凭猜测修改测试。

17. 你可以进行多轮工具调用：

   list_files
       ↓
   retrieve_code
       ↓
   read_file
       ↓
   read_file(test)
       ↓
   run_test
       ↓
   分析
       ↓
   edit_file
       ↓
   run_test
       ↓
   PASS 或继续修复

18. 工具调用顺序不是固定的。
   你需要根据当前已经获得的信息自主决定下一步。

19. 最重要的原则：

   不确定 → 获取信息
   获取信息 → 分析
   分析后 → 修改
   修改后 → 测试
   测试失败 → 根据真实错误继续修复
   测试全部通过 → 结束

重要：
不要为了完成任务而强行寻找或制造问题。

如果你检查了源代码、测试代码并运行测试后，
没有发现能够被证据支持的缺陷，那么可以直接说明：
“当前代码在现有测试下没有发现问题”。

不要凭空假设某个参数可能有其他格式，
也不要把未出现在代码、测试或用户需求中的使用场景，
当成当前任务中的实际缺陷。

特别是：
“潜在问题”“理论上可能”“如果用户传入某种参数”
不能作为修改代码的充分依据。

只有在你有明确证据表明当前代码存在缺陷时，
才应该调用 edit_file。

如果你声称进行了修改、增加测试或验证，
必须实际调用对应工具完成操作，不能只在最终回答中声称已经完成。
"""
}, {
    "role": "user",
    "content": f"""
用户的问题：

{content}

请自主调查并修复这个问题。
如果需要寻找相关代码，请使用 retrieve_code。
"""
}]
for _ in range(max_rounds):
    print("=========第{}轮=========".format(_ + 1))
    response=ollama.chat(model="qwen3:4b",messages=messages,tools=tools)
    print(response["message"])
    print("="*9)
    if response["message"].get("tool_calls"):
        messages.append(response["message"])
        for call in response["message"]["tool_calls"]:
            tool_name = call["function"]["name"]
            arguments = call["function"]["arguments"]
            tool=tools_map.get(tool_name)
            if tool:
                try:
                    result = tool(**arguments)
                except Exception as e:
                    result = f"工具执行失败：{type(e).__name__}: {e}"
                if tool_name == "edit_file":
                    edited = True
                    tested_after_edit = False
                    tests_passed = False

                if tool_name == "run_test":
                    if isinstance(result, dict):
                        if result.get("status") == "PASS":
                            tests_passed = True
                        else:
                            tests_passed = False
                    else:
                        tests_passed = False
                    if edited:
                        edited = False
                        tested_after_edit = True
                messages.append({
                    "role": "tool",
                    "content": str(result),
                    "tool_name": tool_name
                })
            else:
                print(f"未找到工具：{tool_name}")
    else:
        if not tests_passed:
            print("Agent 尚未获得测试通过的验证结果，要求继续修复。")

            messages.append({
            "role": "user",
            "content": """
你不能结束任务。

当前还没有得到“所有测试通过”的验证结果。

请继续调查问题并修改代码。
修改后必须调用 run_test。

只有当 run_test 明确显示所有测试通过后，才能结束任务。
不要仅仅根据自己的分析声称测试已经通过。
"""
        })

            continue

        print(response["message"]["content"])
        break
