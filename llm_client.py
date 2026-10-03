# import ollama


# def chat(messages, tools=None):
#     kwargs = {
#         "model": "qwen3:4b",
#         "messages": messages
#     }

#     if tools is not None:
#         kwargs["tools"] = tools

#     return ollama.chat(**kwargs)


# if __name__ == "__main__":
#     messages = [
#         {
#             "role": "user",
#             "content": "请只回答：测试成功"
#         }
#     ]

#     result = chat(messages)

#     print(result["message"]["content"])



from zai import ZhipuAiClient
import time


# 单次请求超时(秒)；超过即抛异常，防止网络挂起导致进程无限卡死
REQUEST_TIMEOUT = 180

# 外层重试次数与基础等待时间
MAX_ATTEMPTS = 5
RETRY_BASE_DELAY = 10


client = ZhipuAiClient(
    api_key="a8be4c613dce4b019f3e4b421b187b47.o9ITsLyqRAece1cN"
)

# 关闭 SDK 内部静默重试（默认 3 次）：
# 请求挂起时内部重试不打印任何信息，最长可静默 12 分钟，
# 表现为进程"卡死"。重试全部交给下方 chat() 的可见重试接管。
client.max_retries = 0

def chat(messages, tools=None):
    kwargs = {
        "model": "glm-5.3-flashx",
        "messages": messages,
        "temperature": 0.6
    }

    if tools is not None:
        # 智谱需要明确的 function tool 类型
        converted_tools = []

        for tool in tools:

            if tool.get("type") == "function":
                converted_tools.append(tool)

            else:
                converted_tools.append({
                    "type": "function",
                    "function": tool["function"]
                })

        kwargs["tools"] = converted_tools

    # 带超时与可见重试的调用。
    # SDK 内部重试是静默的，请求挂起时进程会表现为无限卡死，
    # 所以必须在应用层显式处理超时并打印重试状态。
    response = None
    last_error = None

    for attempt in range(1, MAX_ATTEMPTS + 1):

        # 心跳：让用户知道进程在等 LLM 响应，而不是卡死
        print(
            f"\n📡 调用 LLM 中"
            f"（第 {attempt}/{MAX_ATTEMPTS} 次尝试，"
            f"最长等待 {REQUEST_TIMEOUT} 秒）..."
        )

        try:

            started = time.time()

            response = client.chat.completions.create(
                timeout=REQUEST_TIMEOUT,
                **kwargs
            )

            elapsed = time.time() - started

            print(
                f"✅ LLM 响应完成，用时 {elapsed:.0f} 秒"
            )

            break

        except KeyboardInterrupt:
            raise

        except Exception as e:

            last_error = e

            if attempt < MAX_ATTEMPTS:

                delay = RETRY_BASE_DELAY * attempt

                print(
                    f"\n⏳ LLM 调用失败"
                    f"（第 {attempt}/{MAX_ATTEMPTS} 次）："
                    f"{type(e).__name__}: {e}"
                )
                print(
                    f"   {delay} 秒后重试..."
                )

                time.sleep(delay)

    if response is None:

        print(
            f"\n❌ LLM 调用连续 {MAX_ATTEMPTS} 次失败："
            f"{type(last_error).__name__}: {last_error}"
        )

        raise last_error

    message = response.choices[0].message

    result_message = {
        "role": message.role,
        "content": message.content or ""
    }

    # Tool Calling
    if message.tool_calls:

        result_message["tool_calls"] = []

        for tool_call in message.tool_calls:

            result_message["tool_calls"].append({
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                }
            })

    return {
        "message": result_message
    }