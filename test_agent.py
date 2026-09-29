import ollama
from tools import tools_map

max_rounds = 10

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
            "description": "运行指定项目的pytest测试文件，并返回测试结果",
            "parameters": {
                "type": "object",
                "properties": {"path": {
                                        "type": "string",
                                        "description": "要运行测试的文件路径"
                                    }},
                "required": ["path"]
            }
        }
    }
]

content = input("请输入您的问题：")
messages=[{
    "role": "user",
    "content": content
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
                result=tool(**arguments)
                messages.append({
                    "role": "tool",
                    "content": result
                })
            else:
                print(f"未找到工具：{tool_name}")
    else:
        print(response["message"]["content"])
        break
