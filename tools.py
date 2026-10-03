import os
import re
import subprocess
import sys
import json
import shutil


# ============================================================
# Code RAG
# ============================================================

try:
    from code_rag import build_code_rag, retrieve_code, format_results
except Exception:
    build_code_rag = None
    retrieve_code = None
    format_results = None


# ============================================================
# Workspace
# ============================================================

ACTIVE_WORKSPACE = None

RAG_INDEX = None
RAG_CHUNKS = None
RAG_WORKSPACE = None

VALIDATION_DIR_NAME = ".agent_validation"
VALIDATION_FILE_NAME = "test_bug_validation.py"
DEPENDENCY_MARKER = ".code_agent_deps_ready.json"


# ============================================================
# Workspace
# ============================================================

def set_workspace(workspace):
    global ACTIVE_WORKSPACE
    global RAG_INDEX, RAG_CHUNKS, RAG_WORKSPACE

    ACTIVE_WORKSPACE = os.path.abspath(workspace)

    if not os.path.isdir(ACTIVE_WORKSPACE):
        raise NotADirectoryError(
            f"Workspace 不存在：{ACTIVE_WORKSPACE}"
        )

    RAG_INDEX = None
    RAG_CHUNKS = None
    RAG_WORKSPACE = None


def get_workspace_path(path):
    if ACTIVE_WORKSPACE is None:
        raise RuntimeError("尚未设置 Workspace")

    if not path:
        raise ValueError("路径不能为空")

    path = path.replace("\\", os.sep)

    workspace = os.path.abspath(ACTIVE_WORKSPACE)
    target = os.path.abspath(os.path.join(workspace, path))

    try:
        common = os.path.commonpath([workspace, target])
    except ValueError:
        raise PermissionError("非法路径：无法访问 Workspace 外部文件")

    if common != workspace:
        raise PermissionError(
            "非法路径：Agent 不允许访问 Workspace 外部文件"
        )

    return target


# ============================================================
# Python / dependency environment
# ============================================================

def get_python_executable():
    if ACTIVE_WORKSPACE is None:
        raise RuntimeError("尚未设置 Workspace")

    if os.name == "nt":
        python = os.path.join(
            ACTIVE_WORKSPACE, ".venv", "Scripts", "python.exe"
        )
    else:
        python = os.path.join(
            ACTIVE_WORKSPACE, ".venv", "bin", "python"
        )

    if os.path.isfile(python):
        return python

    return sys.executable


def _run_pip(python, args, timeout=600):
    command = [python, "-m", "pip"] + args

    result = subprocess.run(
        command,
        cwd=ACTIVE_WORKSPACE,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout
    )

    output = (result.stdout or "") + "\n" + (result.stderr or "")

    return result.returncode, output[-16000:]


def _pytest_available(python):
    result = subprocess.run(
        [python, "-m", "pytest", "--version"],
        cwd=ACTIVE_WORKSPACE,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60
    )
    return result.returncode == 0


def _dependency_files():
    root = ACTIVE_WORKSPACE

    files = []

    # 开发/测试依赖优先
    for name in (
        "requirements-dev.txt",
        "requirements-test.txt",
        "requirements.txt",
    ):
        path = os.path.join(root, name)
        if os.path.isfile(path):
            files.append(path)

    return files


def _has_package_metadata():
    root = ACTIVE_WORKSPACE

    return any(
        os.path.isfile(os.path.join(root, name))
        for name in (
            "pyproject.toml",
            "setup.py",
            "setup.cfg",
        )
    )


def ensure_test_environment():
    """
    自动准备当前 Workspace 的测试环境。

    设计原则：
    1. 优先使用 Workspace 自己的 .venv。
    2. 识别 requirements-dev / requirements-test / requirements。
    3. 识别 pyproject.toml / setup.py / setup.cfg。
    4. 自动确保 pytest 存在。
    5. 成功后写 marker，后续不重复安装。
    6. 如果 marker 存在但 pytest 已损坏，会重新准备环境。
    """

    if ACTIVE_WORKSPACE is None:
        raise RuntimeError("尚未设置 Workspace")

    python = get_python_executable()
    marker_path = os.path.join(
        ACTIVE_WORKSPACE,
        DEPENDENCY_MARKER
    )

    # --------------------------------------------------------
    # 已经准备过：只快速检查 pytest
    # --------------------------------------------------------

    if os.path.isfile(marker_path) and _pytest_available(python):
        return {
            "status": "READY",
            "python": python,
            "message": "Workspace 依赖环境已准备完成。"
        }

    install_log = []

    # --------------------------------------------------------
    # 先升级 pip（失败不直接中断，因为旧 pip 也可能能用）
    # --------------------------------------------------------

    code, output = _run_pip(
        python,
        ["install", "--upgrade", "pip"],
        timeout=300
    )
    install_log.append({
        "step": "upgrade_pip",
        "returncode": code,
        "output": output
    })

    # --------------------------------------------------------
    # 项目依赖
    #
    # requirements 文件 + editable install 都处理。
    # 这样 Rich 这种 pyproject 项目可以正常安装本体依赖，
    # 同时 pytest 单独确保存在。
    # --------------------------------------------------------

    for dependency_file in _dependency_files():
        relative = os.path.relpath(
            dependency_file,
            ACTIVE_WORKSPACE
        )

        code, output = _run_pip(
            python,
            ["install", "-r", relative],
            timeout=900
        )

        install_log.append({
            "step": f"install_{relative}",
            "returncode": code,
            "output": output
        })

        if code != 0:
            raise RuntimeError(
                "项目依赖安装失败："
                f"{relative}\n{output}"
            )

    # --------------------------------------------------------
    # pyproject/setup 项目：安装当前项目本身
    # --------------------------------------------------------

    if _has_package_metadata():
        code, output = _run_pip(
            python,
            ["install", "-e", "."],
            timeout=900
        )

        install_log.append({
            "step": "install_project_editable",
            "returncode": code,
            "output": output
        })

        if code != 0:
            raise RuntimeError(
                "项目本体安装失败。\n"
                f"{output}"
            )

    # --------------------------------------------------------
    # pytest 是 Agent 测试闭环的硬依赖
    # --------------------------------------------------------

    if not _pytest_available(python):
        code, output = _run_pip(
            python,
            ["install", "pytest"],
            timeout=600
        )

        install_log.append({
            "step": "install_pytest",
            "returncode": code,
            "output": output
        })

        if code != 0:
            raise RuntimeError(
                "pytest 自动安装失败。\n"
                f"{output}"
            )

    if not _pytest_available(python):
        raise RuntimeError(
            "依赖安装流程结束，但当前 Workspace 仍然无法执行 pytest。"
        )

    # --------------------------------------------------------
    # 写 marker
    # --------------------------------------------------------

    marker = {
        "python": python,
        "dependencies": [
            os.path.relpath(p, ACTIVE_WORKSPACE)
            for p in _dependency_files()
        ],
        "package_metadata": _has_package_metadata(),
        "pytest": True,
    }

    with open(
        marker_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            marker,
            f,
            ensure_ascii=False,
            indent=2
        )

    return {
        "status": "READY",
        "python": python,
        "message": "Workspace 依赖环境已自动准备完成。",
        "install_log": install_log
    }


# ============================================================
# read_file
# ============================================================

def read_file(path, start_line=1, max_lines=200):
    target = get_workspace_path(path)

    if not os.path.isfile(target):
        raise FileNotFoundError(f"文件不存在：{path}")

    if start_line < 1:
        raise ValueError("start_line 必须 >= 1")

    if max_lines < 1:
        raise ValueError("max_lines 必须 >= 1")

    with open(
        target,
        "r",
        encoding="utf-8"
    ) as f:
        lines = f.readlines()

    start_index = start_line - 1
    end_index = start_index + max_lines

    selected = lines[start_index:end_index]

    if not selected:
        return f"文件 {path} 从第 {start_line} 行开始没有更多内容。"

    return "\n".join(
        f"{index}: {line.rstrip()}"
        for index, line in enumerate(
            selected,
            start=start_line
        )
    )


# ============================================================
# replace_in_file
# ============================================================

def replace_in_file(path, old, new):
    target = get_workspace_path(path)

    normalized = path.replace("\\", "/").lower().strip("/")

    # Agent 永远不能修改项目 tests
    if (
        normalized == "tests"
        or normalized.startswith("tests/")
        or "/tests/" in normalized
    ):
        raise PermissionError(
            "Agent 不允许修改 tests 目录中的文件。"
        )

    # 也不允许直接修改临时验证目录
    if (
        normalized == VALIDATION_DIR_NAME
        or normalized.startswith(VALIDATION_DIR_NAME + "/")
    ):
        raise PermissionError(
            "Agent 不允许通过 replace_in_file 修改临时 Bug 验证文件。"
        )

    if not os.path.isfile(target):
        raise FileNotFoundError(f"文件不存在：{path}")

    with open(
        target,
        "r",
        encoding="utf-8"
    ) as f:
        content = f.read()

    count = content.count(old)

    if count == 0:
        raise ValueError("没有找到要替换的原始代码。")

    if count > 1:
        raise ValueError(
            f"原始代码出现了 {count} 次，为避免误修改，拒绝替换。"
        )

    new_content = content.replace(old, new, 1)

    with open(
        target,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(new_content)

    global RAG_INDEX, RAG_CHUNKS, RAG_WORKSPACE
    RAG_INDEX = None
    RAG_CHUNKS = None
    RAG_WORKSPACE = None

    return {
        "status": "SUCCESS",
        "path": path,
        "message": "文件修改成功，只进行了 1 处精确替换。"
    }


# 兼容部分旧 Agent / prompt 使用的名称
edit_file = replace_in_file


# ============================================================
# check_syntax
# ============================================================

def check_syntax(path):
    target = get_workspace_path(path)

    if not os.path.isfile(target):
        raise FileNotFoundError(f"文件不存在：{path}")

    python = get_python_executable()

    result = subprocess.run(
        [
            python,
            "-m",
            "py_compile",
            target
        ],
        cwd=ACTIVE_WORKSPACE,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30
    )

    if result.returncode == 0:
        return {
            "status": "PASS",
            "path": path,
            "message": "语法检查通过。"
        }

    return {
        "status": "FAIL",
        "path": path,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr
    }


# ============================================================
# run_test
# ============================================================

def run_test(test_path=None, test_filter=None):
    """
    Agent 真正执行 pytest 前，自动准备依赖环境。
    """

    if ACTIVE_WORKSPACE is None:
        raise RuntimeError("尚未设置 Workspace")

    env = ensure_test_environment()
    python = env["python"]

    command = [
        python,
        "-m",
        "pytest"
    ]

    if test_path:
        # 允许 tests/test_x.py::test_name
        file_part = test_path.split("::", 1)[0]
        target = get_workspace_path(file_part)

        if not os.path.isfile(target):
            raise FileNotFoundError(
                f"测试文件不存在：{test_path}"
            )

        command.append(test_path)

    if test_filter:
        command.extend(["-k", test_filter])

    try:
        result = subprocess.run(
            command,
            cwd=ACTIVE_WORKSPACE,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300
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

    output = (result.stdout or "") + "\n" + (result.stderr or "")

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
    target = get_workspace_path(path)

    if not os.path.isdir(target):
        raise NotADirectoryError(f"目录不存在：{path}")

    entries = []

    for name in sorted(os.listdir(target)):
        full_path = os.path.join(target, name)

        if name in {
            ".git",
            "__pycache__",
            ".venv",
            VALIDATION_DIR_NAME,
        }:
            continue

        if os.path.isdir(full_path):
            entries.append(f"[DIR]  {name}")
        else:
            entries.append(f"[FILE] {name}")

    return "\n".join(entries) if entries else "(目录为空)"


# ============================================================
# retrieve_code
# ============================================================

def retrieve_code_tool(query):
    global RAG_INDEX, RAG_CHUNKS, RAG_WORKSPACE

    if ACTIVE_WORKSPACE is None:
        raise RuntimeError("尚未设置 Workspace")

    if retrieve_code is None:
        raise RuntimeError("code_rag.py 无法导入 retrieve_code。")

    # --------------------------------------------------------
    # 新版 Zhipu Code RAG：
    # retrieve_code(query, root=workspace)
    # --------------------------------------------------------

    try:
        results = retrieve_code(
            query,
            root=ACTIVE_WORKSPACE
        )

        if isinstance(results, str):
            return results

        if format_results is not None:
            return format_results(results)

        return str(results)

    except TypeError:
        # ----------------------------------------------------
        # 兼容旧版：
        # build_code_rag(root) -> index, chunks
        # retrieve_code(query, index, chunks)
        # ----------------------------------------------------

        if build_code_rag is None:
            raise

        if (
            RAG_INDEX is None
            or RAG_CHUNKS is None
            or RAG_WORKSPACE != ACTIVE_WORKSPACE
        ):
            print("\n正在首次构建 Code RAG 索引，请稍候...")

            RAG_INDEX, RAG_CHUNKS = build_code_rag(
                ACTIVE_WORKSPACE
            )

            RAG_WORKSPACE = ACTIVE_WORKSPACE

            print(
                f"Code RAG 构建完成，共加载 "
                f"{len(RAG_CHUNKS)} 个代码块。"
            )

        results = retrieve_code(
            query,
            RAG_INDEX,
            RAG_CHUNKS,
            top_k=5
        )

        return (
            format_results(results)
            if format_results is not None
            else str(results)
        )


# ============================================================
# Temporary Bug Validation
# ============================================================

def _validation_file_path():
    validation_dir = get_workspace_path(
        VALIDATION_DIR_NAME
    )

    os.makedirs(
        validation_dir,
        exist_ok=True
    )

    return os.path.join(
        validation_dir,
        VALIDATION_FILE_NAME
    )


def create_bug_validation(test_code):
    """
    创建 Agent 专用临时回归测试。

    不修改项目 tests。
    """

    if ACTIVE_WORKSPACE is None:
        raise RuntimeError("尚未设置 Workspace")

    if not isinstance(test_code, str) or not test_code.strip():
        raise ValueError("test_code 不能为空。")

    if not re.search(
        r"(?m)^\s*(async\s+)?def\s+test_[A-Za-z0-9_]+",
        test_code
    ):
        raise ValueError(
            "test_code 必须至少包含一个 test_ 开头的 pytest 测试函数。"
        )

    # 防止测试代码通过相对路径访问 Workspace 外部
    validation_path = _validation_file_path()

    with open(
        validation_path,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(test_code.rstrip() + "\n")

    return {
        "status": "CREATED",
        "available": True,
        "path": os.path.join(
            VALIDATION_DIR_NAME,
            VALIDATION_FILE_NAME
        ),
        "message": (
            "临时 Bug 回归验证已创建。"
            "必须先在未修改代码的状态运行 run_bug_validation。"
        )
    }


def run_bug_validation():
    """
    运行 .agent_validation/test_bug_validation.py。

    不通过 replace_in_file 修改测试文件。
    """

    if ACTIVE_WORKSPACE is None:
        raise RuntimeError("尚未设置 Workspace")

    validation_path = os.path.join(
        VALIDATION_DIR_NAME,
        VALIDATION_FILE_NAME
    )

    target = get_workspace_path(validation_path)

    if not os.path.isfile(target):
        return {
            "status": "NOT_CREATED",
            "available": False,
            "message": "尚未创建临时 Bug 回归验证。"
        }

    return run_test(
        test_path=validation_path
    )


def cleanup_bug_validation():
    """
    清理临时 Bug 回归测试。
    """

    if ACTIVE_WORKSPACE is None:
        return {
            "removed": False,
            "message": "尚未设置 Workspace。"
        }

    validation_dir = get_workspace_path(
        VALIDATION_DIR_NAME
    )

    if not os.path.exists(validation_dir):
        return {
            "removed": False,
            "message": "没有需要清理的临时验证文件。"
        }

    shutil.rmtree(
        validation_dir,
        ignore_errors=True
    )

    return {
        "removed": True,
        "message": "临时 Bug 回归验证已清理。"
    }


# ============================================================
# Tool Map
# ============================================================

tools_map = {
    "read_file": read_file,
    "replace_in_file": replace_in_file,
    "check_syntax": check_syntax,
    "run_test": run_test,
    "list_files": list_files,
    "retrieve_code": retrieve_code_tool,
    "create_bug_validation": create_bug_validation,
    "run_bug_validation": run_bug_validation,
}


__all__ = [
    "set_workspace",
    "get_workspace_path",
    "get_python_executable",
    "ensure_test_environment",
    "read_file",
    "replace_in_file",
    "edit_file",
    "check_syntax",
    "run_test",
    "list_files",
    "retrieve_code_tool",
    "create_bug_validation",
    "run_bug_validation",
    "cleanup_bug_validation",
    "tools_map",
]