"""
通用 Bug 回归验证。

该模块不包含任何具体仓库、Issue 或 Bug 的知识。
Agent 根据当前问题动态生成一个临时 pytest，用来：
1. 在修改前确认原始 Bug 能复现；
2. 修改后验证同一 Bug 是否消失。
"""

import ast
import os
import subprocess
import sys
import textwrap

VALIDATION_DIR = ".agent_validation"
VALIDATION_FILE = "test_regression.py"
TIMEOUT_SECONDS = 30

# 临时验证脚本禁止直接获得宿主机控制能力。
_BLOCKED_IMPORTS = {
    "subprocess", "socket", "ctypes", "shutil", "multiprocessing",
    "winreg", "pwd", "grp", "resource",
}
_BLOCKED_CALLS = {
    "eval", "exec", "compile", "input", "system", "popen",
}


def _validation_path(workspace):
    return os.path.join(workspace, VALIDATION_DIR, VALIDATION_FILE)


def _validate_source(test_code):
    if not isinstance(test_code, str) or not test_code.strip():
        raise ValueError("test_code 不能为空")

    try:
        tree = ast.parse(test_code)
    except SyntaxError as exc:
        raise ValueError(f"回归测试源码语法错误：{exc}") from exc

    test_functions = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    ]
    if not test_functions:
        raise ValueError("回归测试必须包含至少一个 test_ 开头的 pytest 测试函数")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in _BLOCKED_IMPORTS:
                    raise ValueError(f"回归测试禁止导入高风险模块：{root}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root in _BLOCKED_IMPORTS:
                raise ValueError(f"回归测试禁止导入高风险模块：{root}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _BLOCKED_CALLS:
                raise ValueError(f"回归测试禁止调用高风险函数：{node.func.id}")
            if isinstance(node.func, ast.Attribute) and node.func.attr in _BLOCKED_CALLS:
                raise ValueError(f"回归测试禁止调用高风险函数：{node.func.attr}")


def create_bug_validation(workspace, test_code):
    """创建/覆盖当前任务的临时回归测试，不触碰项目 tests。"""
    _validate_source(test_code)

    directory = os.path.join(workspace, VALIDATION_DIR)
    os.makedirs(directory, exist_ok=True)

    path = _validation_path(workspace)
    with open(path, "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(test_code).lstrip())
        f.write("\n")

    return {
        "status": "CREATED",
        "path": os.path.relpath(path, workspace).replace("\\", "/"),
        "message": "临时 Bug 回归测试已创建。必须先在未修改代码状态运行一次。",
    }


def run_bug_validation(workspace, python_executable=None):
    """运行临时 pytest；返回真实退出码和完整输出。"""
    path = _validation_path(workspace)
    if not os.path.isfile(path):
        return {
            "status": "NO_VALIDATION",
            "returncode": None,
            "output": "尚未创建 Bug 回归测试。",
        }

    python_executable = python_executable or sys.executable
    command = [
        python_executable,
        "-m",
        "pytest",
        path,
        "-q",
        "--disable-warnings",
    ]

    try:
        result = subprocess.run(
            command,
            cwd=workspace,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + (exc.stderr or "")
        return {
            "status": "TIMEOUT",
            "returncode": None,
            "output": output + f"\n回归验证超过 {TIMEOUT_SECONDS} 秒。",
        }

    output = result.stdout + result.stderr
    return {
        "status": "PASS" if result.returncode == 0 else "FAIL",
        "returncode": result.returncode,
        "output": output,
    }


def cleanup_bug_validation(workspace):
    """清理临时回归测试，避免污染仓库。"""
    path = _validation_path(workspace)
    directory = os.path.dirname(path)
    removed = False

    if os.path.isfile(path):
        os.remove(path)
        removed = True

    if os.path.isdir(directory) and not os.listdir(directory):
        os.rmdir(directory)

    return {"status": "CLEANED", "removed": removed}
