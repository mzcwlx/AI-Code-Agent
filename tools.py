import os
import subprocess
import sys


from code_rag import build_code_rag, retrieve_code, format_results


# ============================================================
# Workspace
# ============================================================

ACTIVE_WORKSPACE = None


# ============================================================
# Code RAG Cache
# ============================================================

# 第一次 retrieve_code 时建立
# 后续 retrieve_code 直接复用
RAG_INDEX = None
RAG_CHUNKS = None
RAG_WORKSPACE = None


# ============================================================
# Workspace 管理
# ============================================================

def set_workspace(workspace):
    """
    设置当前 Agent 工作区。

    切换 Workspace 后，必须清空旧的 Code RAG 缓存，
    防止不同项目之间混用索引。
    """

    global ACTIVE_WORKSPACE
    global RAG_INDEX
    global RAG_CHUNKS
    global RAG_WORKSPACE

    ACTIVE_WORKSPACE = os.path.abspath(workspace)

    if not os.path.isdir(ACTIVE_WORKSPACE):
        raise NotADirectoryError(
            f"Workspace 不存在：{ACTIVE_WORKSPACE}"
        )

    # 切换项目时清空旧 RAG
    RAG_INDEX = None
    RAG_CHUNKS = None
    RAG_WORKSPACE = None


# ============================================================
# Workspace 路径安全
# ============================================================

def get_workspace_path(path):
    """
    将 Agent 提供的相对路径转换成 Workspace 内的绝对路径。

    防止 Agent 通过 ../ 等方式访问 Workspace 外部文件。
    """

    if ACTIVE_WORKSPACE is None:
        raise RuntimeError("尚未设置 Workspace")

    if not path:
        raise ValueError("路径不能为空")

    # 统一路径格式
    path = path.replace("\\", os.sep)

    workspace = os.path.abspath(ACTIVE_WORKSPACE)
    target = os.path.abspath(
        os.path.join(workspace, path)
    )

    # 确保 target 位于 workspace 内
    try:
        common = os.path.commonpath(
            [workspace, target]
        )
    except ValueError:
        raise PermissionError(
            "非法路径：无法访问 Workspace 外部文件"
        )

    if common != workspace:
        raise PermissionError(
            "非法路径：Agent 不允许访问 Workspace 外部文件"
        )

    return target


# ============================================================
# read_file
# ============================================================

def read_file(path, start_line=1, max_lines=200):
    """
    读取 Workspace 中的文件。

    start_line:
        从第几行开始，1-based。

    max_lines:
        最多读取多少行。
    """

    target = get_workspace_path(path)

    if not os.path.isfile(target):
        raise FileNotFoundError(
            f"文件不存在：{path}"
        )

    if start_line < 1:
        raise ValueError(
            "start_line 必须 >= 1"
        )

    if max_lines < 1:
        raise ValueError(
            "max_lines 必须 >= 1"
        )

    with open(
        target,
        "r",
        encoding="utf-8"
    ) as f:
        lines = f.readlines()

    start_index = start_line - 1
    end_index = start_index + max_lines

    selected_lines = lines[
        start_index:end_index
    ]

    if not selected_lines:
        return (
            f"文件 {path} 从第 {start_line} 行开始没有更多内容。"
        )

    result = []

    for index, line in enumerate(
        selected_lines,
        start=start_line
    ):
        result.append(
            f"{index}: {line.rstrip()}"
        )

    return "\n".join(result)


# ============================================================
# replace_in_file
# ============================================================

def replace_in_file(path, old, new):
    """
    对文件执行一次精确替换。

    安全策略：

    1. 不允许修改 tests 目录
    2. old 必须存在
    3. old 必须只出现一次
    4. 只允许替换一次
    """

    target = get_workspace_path(path)

    normalized = path.replace(
        "\\",
        "/"
    ).lower().strip("/")

    # --------------------------------------------------------
    # 禁止修改测试文件
    # --------------------------------------------------------

    if (
        normalized == "tests"
        or normalized.startswith("tests/")
        or "/tests/" in normalized
    ):
        raise PermissionError(
            "Agent 不允许修改 tests 目录中的文件"
        )

    # --------------------------------------------------------
    # 文件存在性检查
    # --------------------------------------------------------

    if not os.path.isfile(target):
        raise FileNotFoundError(
            f"文件不存在：{path}"
        )

    # --------------------------------------------------------
    # 读取文件
    # --------------------------------------------------------

    with open(
        target,
        "r",
        encoding="utf-8"
    ) as f:
        content = f.read()

    # --------------------------------------------------------
    # 检查 old
    # --------------------------------------------------------

    count = content.count(old)

    if count == 0:
        raise ValueError(
            "没有找到要替换的原始代码。"
        )

    if count > 1:
        raise ValueError(
            f"原始代码出现了 {count} 次，"
            "为了避免误修改，拒绝替换。"
        )

    # --------------------------------------------------------
    # 执行替换
    # --------------------------------------------------------

    new_content = content.replace(
        old,
        new,
        1
    )

    # --------------------------------------------------------
    # 写回文件
    # --------------------------------------------------------

    with open(
        target,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(new_content)

    # --------------------------------------------------------
    # 文件发生变化后，旧 RAG 已经失效
    # --------------------------------------------------------

    global RAG_INDEX
    global RAG_CHUNKS
    global RAG_WORKSPACE

    RAG_INDEX = None
    RAG_CHUNKS = None
    RAG_WORKSPACE = None

    return (
        f"文件 {path} 修改成功。\n"
        f"本次只进行了 1 处精确替换。"
    )


# ============================================================
# check_syntax
# ============================================================

def get_python_executable():
    """
    获取当前 Workspace 的 Python。

    优先使用 Workspace 自己的 .venv。
    """

    if ACTIVE_WORKSPACE is None:
        raise RuntimeError(
            "尚未设置 Workspace"
        )

    if os.name == "nt":

        venv_python = os.path.join(
            ACTIVE_WORKSPACE,
            ".venv",
            "Scripts",
            "python.exe"
        )

    else:

        venv_python = os.path.join(
            ACTIVE_WORKSPACE,
            ".venv",
            "bin",
            "python"
        )

    if os.path.isfile(venv_python):
        return venv_python

    # 如果 Workspace 没有虚拟环境
    # 使用当前 Agent 的 Python
    return sys.executable


def check_syntax(path):
    """
    使用 Workspace Python 对 Python 文件执行 py_compile。
    """

    target = get_workspace_path(path)

    if not os.path.isfile(target):
        raise FileNotFoundError(
            f"文件不存在：{path}"
        )

    python = get_python_executable()

    command = [
        python,
        "-m",
        "py_compile",
        target
    ]

    try:

        result = subprocess.run(
            command,
            cwd=ACTIVE_WORKSPACE,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30
        )

    except subprocess.TimeoutExpired:

        return {
            "status": "TIMEOUT",
            "path": path,
            "message": "语法检查超过 30 秒。"
        }

    if result.returncode == 0:

        return {
            "status": "PASS",
            "path": path,
            "message": "语法检查通过。"
        }

    return {
        "status": "FAIL",
        "path": path,
        "stdout": result.stdout,
        "stderr": result.stderr
    }


# ============================================================
# run_test
# ============================================================

def run_test(
    test_path=None,
    test_filter=None
):
    """
    使用 Workspace 自己的 Python 环境执行 pytest。

    test_path:
        指定测试文件，例如：

        tests/test_requests.py

    test_filter:
        pytest -k 表达式，例如：

        pickle

    注意：

    test_filter 只允许传递 pytest -k 表达式，
    不要把命令行参数，例如 -v，塞进这里。
    """

    if ACTIVE_WORKSPACE is None:
        raise RuntimeError(
            "尚未设置 Workspace"
        )

    python = get_python_executable()

    command = [
        python,
        "-m",
        "pytest"
    ]

    if test_path:

        # 测试路径也必须在 Workspace 内
        test_target = get_workspace_path(
            test_path
        )

        if not os.path.isfile(test_target):
            raise FileNotFoundError(
                f"测试文件不存在：{test_path}"
            )

        command.append(test_path)

    if test_filter:

        command.extend([
            "-k",
            test_filter
        ])

    try:

        result = subprocess.run(
            command,
            cwd=ACTIVE_WORKSPACE,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120
        )

    except subprocess.TimeoutExpired as e:

        output = ""

        if e.stdout:
            output += str(e.stdout)

        if e.stderr:
            output += "\n" + str(e.stderr)

        return {
            "status": "TIMEOUT",
            "command": " ".join(command),
            "returncode": None,
            "output": output[-12000:]
        }

    output = ""

    if result.stdout:
        output += result.stdout

    if result.stderr:
        output += "\n" + result.stderr

    # --------------------------------------------------------
    # pytest 返回码
    # --------------------------------------------------------

    if result.returncode == 0:

        status = "PASS"

    elif result.returncode == 5:

        status = "NO_TESTS"

    elif result.returncode == 2:

        status = "INTERRUPTED"

    elif result.returncode == 3:

        status = "INTERNAL_ERROR"

    elif result.returncode == 4:

        status = "USAGE_ERROR"

    else:

        status = "FAIL"

    return {
        "status": status,
        "command": " ".join(command),
        "returncode": result.returncode,
        "output": output[-12000:]
    }


# ============================================================
# list_files
# ============================================================

def list_files(path="."):
    """
    列出 Workspace 中指定目录的文件。
    """

    target = get_workspace_path(path)

    if not os.path.isdir(target):
        raise NotADirectoryError(
            f"目录不存在：{path}"
        )

    entries = []

    for name in sorted(
        os.listdir(target)
    ):

        full_path = os.path.join(
            target,
            name
        )

        if os.path.isdir(full_path):

            entries.append(
                f"[DIR]  {name}"
            )

        else:

            entries.append(
                f"[FILE] {name}"
            )

    if not entries:
        return "(目录为空)"

    return "\n".join(entries)


# ============================================================
# retrieve_code
# ============================================================

def retrieve_code_tool(query):
    """
    Code RAG 检索工具。

    第一次调用：

        构建整个 Workspace 的 Code RAG
        ↓
        embedding
        ↓
        FAISS Index

    后续调用：

        直接复用已经建立的 FAISS Index

    这样可以避免每次 retrieve_code 都重新执行：

        bge-m3 embedding
        ↓
        FAISS 建索引
    """

    global RAG_INDEX
    global RAG_CHUNKS
    global RAG_WORKSPACE

    if ACTIVE_WORKSPACE is None:
        raise RuntimeError(
            "尚未设置 Workspace"
        )

    # ========================================================
    # 第一次查询 / RAG 缓存失效
    # ========================================================

    if (
        RAG_INDEX is None
        or RAG_CHUNKS is None
        or RAG_WORKSPACE != ACTIVE_WORKSPACE
    ):

        print(
            "\n正在首次构建 Code RAG 索引，请稍候..."
        )

        RAG_INDEX, RAG_CHUNKS = build_code_rag(
            ACTIVE_WORKSPACE
        )

        RAG_WORKSPACE = ACTIVE_WORKSPACE

        print(
            f"Code RAG 构建完成，"
            f"共加载 {len(RAG_CHUNKS)} 个代码块。"
        )

    # ========================================================
    # 使用缓存的 FAISS Index
    # ========================================================

    results = retrieve_code(
        query,
        RAG_INDEX,
        RAG_CHUNKS,
        top_k=5
    )

    return format_results(results)


# ============================================================
# Tool Map
# ============================================================

tools_map = {

    "read_file": read_file,

    "replace_in_file": replace_in_file,

    "check_syntax": check_syntax,

    "run_test": run_test,

    "list_files": list_files,

    "retrieve_code": retrieve_code_tool

}