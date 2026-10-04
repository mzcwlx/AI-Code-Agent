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
import os
import json


# 单次请求超时(秒)；超过即抛异常，防止网络挂起导致进程无限卡死
REQUEST_TIMEOUT =180

# 外层重试次数与基础等待时间
MAX_ATTEMPTS = 2
RETRY_BASE_DELAY = 5


API_KEY = os.getenv("ZHIPU_API_KEY")

if not API_KEY:
    raise RuntimeError("未找到环境变量 ZHIPU_API_KEY，请先设置 API Key")

client = ZhipuAiClient(
    api_key=API_KEY
)

# 关闭 SDK 内部静默重试（默认 3 次）：
# 请求挂起时内部重试不打印任何信息，最长可静默 12 分钟，
# 表现为进程"卡死"。重试全部交给下方 chat() 的可见重试接管。
client.max_retries = 0


# ============================================================
# 诊断开关（默认全部关闭，不影响正常运行）
# ============================================================

# True: 不向 API 发送 tools 参数（诊断 Tool Calling 是否为卡死原因）
DEBUG_DISABLE_TOOLS = False

# True: 只发送 system 消息 + 最近一组完整的
#       assistant(tool_calls)/tool 结果对（诊断上下文大小是否为卡死原因）。
#       按完整轮块裁剪，绝不把 tool_call 和 tool result 拆散。
DEBUG_TRIM_MESSAGES = False


def _trim_messages_keep_last_pair(messages):
    """诊断用裁剪：保留开头的全部 system 消息 + 最后一组完整的
    assistant(tool_calls) 及其全部 tool 结果。不修改传入的列表。"""
    msgs = list(messages)
    i = 0
    while i < len(msgs) and msgs[i].get("role") == "system":
        i += 1
    body = msgs[i:]
    if not body:
        return msgs
    j = len(body)
    while j > 0 and body[j - 1].get("role") == "tool":
        j -= 1
    if j > 0 and body[j - 1].get("role") == "assistant" \
            and body[j - 1].get("tool_calls"):
        j -= 1  # 连同发起调用的 assistant 一起保留，保证配对完整
    else:
        j = max(len(body) - 1, 0)  # 尾部没有待配对的 tool 结果：只保留最后一条
    return msgs[:i] + body[j:]


def _debug_message_stats(model, messages, tools_payload):
    """只读统计：只打印请求规模与结构信息。
    不打印消息内容，不打印 API Key，不打印完整请求。"""
    total = 0
    per_message = []
    for idx, m in enumerate(messages):
        c = m.get("content")
        cchars = len(c) if isinstance(c, str) else 0
        total += cchars
        line = (
            f"message[{idx}]: role={m.get('role')} "
            f"content_type={type(c).__name__} content_chars={cchars}"
        )
        tc = m.get("tool_calls") or []
        line += f" has_tool_calls={bool(tc)}"
        if tc:
            line += (
                f" tool_calls_count={len(tc)} "
                f"tool_call_ids={[t.get('id') for t in tc]}"
            )
        if m.get("role") == "tool":
            line += f" tool_call_id={m.get('tool_call_id')}"
        per_message.append(line)

    msgs_json = json.dumps(messages, ensure_ascii=False, default=repr)
    tools_json = (
        json.dumps(tools_payload, ensure_ascii=False, default=repr)
        if tools_payload else ""
    )
    print(
        "\n[LLM DEBUG]\n"
        f"model={model}\n"
        f"message_count={len(messages)}\n"
        f"total_message_chars={total}\n"
        f"tool_count={len(tools_payload) if tools_payload else 0}\n"
        f"serialized_messages_chars={len(msgs_json)}\n"
        f"serialized_tools_chars={len(tools_json)}\n"
        f"serialized_request_chars={len(msgs_json) + len(tools_json)}"
    )
    print("\n".join(per_message))


def _validate_message_structure(messages):
    """只读结构校验：发现异常只打印，不修改消息，不阻断请求。"""
    problems = []
    seen_ids = set()
    answered = set()
    pending = set()
    valid_roles = {"system", "user", "assistant", "tool"}

    for idx, m in enumerate(messages):
        role = m.get("role")
        if role not in valid_roles:
            problems.append(f"message[{idx}] 非法 role: {role!r}")
        c = m.get("content")
        if c is not None and not isinstance(c, str):
            problems.append(
                f"message[{idx}] content 非字符串: {type(c).__name__}"
            )
        try:
            json.dumps(m, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            problems.append(
                f"message[{idx}] 含不可 JSON 序列化对象: {e}"
            )
        if role in ("assistant", "user") and pending:
            problems.append(
                f"message[{idx}] {role} 消息出现时，前一个 assistant 的 "
                f"tool_call 仍未被应答: {sorted(pending)}"
            )
            pending = set()
        if role == "tool":
            tid = m.get("tool_call_id")
            if not tid:
                problems.append(f"message[{idx}] tool 消息缺少 tool_call_id")
            elif tid not in seen_ids:
                problems.append(
                    f"message[{idx}] tool_call_id={tid} 没有对应的 assistant tool_call"
                )
            elif tid in answered:
                problems.append(f"message[{idx}] tool_call_id={tid} 被重复应答")
            else:
                answered.add(tid)
                pending.discard(tid)
        if role == "assistant":
            for t in (m.get("tool_calls") or []):
                cid = t.get("id")
                fn = t.get("function") or {}
                if not cid:
                    problems.append(f"message[{idx}] tool_call 缺少 id")
                elif cid in seen_ids:
                    problems.append(f"message[{idx}] tool_call id 重复: {cid}")
                else:
                    seen_ids.add(cid)
                    pending.add(cid)
                if not fn.get("name"):
                    problems.append(
                        f"message[{idx}] tool_call 缺少 function.name"
                    )
                args = fn.get("arguments")
                if args is not None and not isinstance(args, str):
                    problems.append(
                        f"message[{idx}] tool_call arguments 非字符串: "
                        f"{type(args).__name__}"
                    )

    if pending:
        problems.append(
            f"对话结尾仍有未应答的 tool_call: {sorted(pending)}"
        )

    if problems:
        print("\n[LLM DEBUG] INVALID MESSAGE STRUCTURE")
        for p in problems:
            print(f"  - {p}")
    else:
        print("[LLM DEBUG] message structure OK")


def chat(messages, tools=None):
    # ---- 诊断开关（默认关闭；开启时只影响本次请求的构造）----
    if DEBUG_TRIM_MESSAGES:
        messages = _trim_messages_keep_last_pair(messages)

    kwargs = {
        "model": "glm-5.3-flash",
        "messages": messages,
        "temperature": 0.2,
        "reasoning_effort": "low",
    }

    converted_tools = None

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

        # DEBUG_DISABLE_TOOLS=True 时不发送 tools（诊断 Tool Calling）
        if not DEBUG_DISABLE_TOOLS:
            kwargs["tools"] = converted_tools

    # ---- 只读诊断日志：每次调用打印一次，只含统计信息 ----
    _debug_message_stats(kwargs["model"], messages, kwargs.get("tools"))
    _validate_message_structure(messages)

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