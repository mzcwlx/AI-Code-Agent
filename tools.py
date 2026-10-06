import json
import os
import re
import shutil
import subprocess
import sys


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
# RAG 查询缓存：同一 Workspace + 同一查询无需重复做 embedding / FAISS 检索
RAG_QUERY_CACHE = {}

VALIDATION_DIR_NAME = ".agent_validation"
VALIDATION_FILE_NAME = "test_bug_validation.py"
DEPENDENCY_MARKER = ".code_agent_deps_ready.json"
SESSION_FILE_NAME = ".agent_session.json"
PIP_MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"

# 已列出过的目录（防止 Agent 重复列目录刷屏）
_listed_dirs = {}


# ============================================================
# Workspace
# ============================================================

def set_workspace(workspace):
    global ACTIVE_WORKSPACE
    global RAG_INDEX, RAG_CHUNKS, RAG_WORKSPACE, RAG_QUERY_CACHE
    global _listed_dirs, _read_ranges

    ACTIVE_WORKSPACE = os.path.abspath(workspace)

    if not os.path.isdir(ACTIVE_WORKSPACE):
        raise NotADirectoryError(
            f"Workspace 不存在：{ACTIVE_WORKSPACE}"
        )

    RAG_INDEX = None
    RAG_CHUNKS = None
    RAG_WORKSPACE = None
    RAG_QUERY_CACHE.clear()
    _listed_dirs = {}
    # 每次 Agent Run / 会话恢复都重新开始：
    # 恢复后最近上下文可能已被裁剪，必须允许重新读取
    _read_ranges = {}


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


def _pip_network_failed(output):
    return (
        "No matching distribution found" in output
        or "Could not find a version that satisfies" in output
    )


def _pip_install_step(python, args, step, install_log, timeout=900):
    """
    执行一次 pip install。

    如果因为默认 PyPI 源访问失败（国内网络常见），
    自动用清华镜像源重试一次。
    """

    code, output = _run_pip(
        python,
        ["install"] + args,
        timeout=timeout
    )

    if code != 0 and _pip_network_failed(output):
        print(f"pip 默认源安装失败，正在使用镜像源重试：{step}")

        code, output = _run_pip(
            python,
            ["install"] + args + ["-i", PIP_MIRROR],
            timeout=timeout
        )

    install_log.append({
        "step": step,
        "returncode": code,
        "output": output
    })

    return code, output


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

    # 部分项目（如 requests）把测试依赖放在 requirements/ 目录
    for name in (
        os.path.join("requirements", "testing.txt"),
        os.path.join("requirements", "dev.txt"),
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


def _tests_collectable(python, timeout=180):
    """
    用 pytest --collect-only 真实验证测试环境是否可用。

    收集阶段会真实导入被测项目和测试模块，能同时暴露：
    pytest 缺失、项目依赖缺失、项目本体无法 import 等问题。

    返回码 0（收集成功）或 5（没有收集到测试）都视为可用。
    """

    command = [python, "-m", "pytest", "--collect-only", "-q"]

    tests_dir = os.path.join(ACTIVE_WORKSPACE, "tests")
    if os.path.isdir(tests_dir):
        command.append("tests")

    try:
        result = subprocess.run(
            command,
            cwd=ACTIVE_WORKSPACE,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return False

    return result.returncode in (0, 5)


def _collect_error_packages(python, timeout=180):
    """
    收集失败时，从 pytest --collect-only 的错误输出里提取缺失的
    第三方包名，用于自动补装。

    典型场景：旧代码库不带依赖上界（如 fastapi 只写
    starlette>=0.46.0），pip install -e . 会装到最新 starlette，
    而新版 starlette 的 TestClient 依赖新包（如 httpx2），
    导致全部测试模块导入失败。错误输出会明确给出：
        RuntimeError: The starlette.testclient module requires
        the httpx2 package to be installed.
    以及普通的 ModuleNotFoundError: No module named 'xxx'。
    第三类是 pytest 配置文件引用了未安装插件报错：
        ERROR: Unknown config option: timeout
    选项名经 PYTEST_OPTION_PLUGINS 映射到插件包名
    （如 timeout -> pytest-timeout）后一并补装。

    返回去重后的包名列表；收集成功则返回空列表。
    """

    # pytest 配置选项 -> 对应插件包；值为 None 表示忽略
    PYTEST_OPTION_PLUGINS = {
        "timeout": "pytest-timeout",
        "cov": "pytest-cov",
        "asyncio": "pytest-asyncio",
        "mock": "pytest-mock",
        "xdist": "pytest-xdist",
        "randomly": "pytest-randomly",
        "rerunfailures": "pytest-rerunfailures",
        "xprocess": "pytest-xprocess",
        "metadata": "pytest-metadata",
        "split_runs": "pytest-split-runs",
        "basetemp": None,
    }

    command = [python, "-m", "pytest", "--collect-only", "-q"]

    tests_dir = os.path.join(ACTIVE_WORKSPACE, "tests")
    if os.path.isdir(tests_dir):
        command.append("tests")

    try:
        result = subprocess.run(
            command,
            cwd=ACTIVE_WORKSPACE,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return []

    if result.returncode in (0, 5):
        return []

    output = (result.stdout or "") + "\n" + (result.stderr or "")

    names = []

    # RuntimeError: ... requires the httpx2 package to be installed.
    for match in re.finditer(
        r"requires the ([A-Za-z0-9_.-]+) package",
        output
    ):
        names.append(match.group(1))

    # ModuleNotFoundError: No module named 'dirty_equals'
    for match in re.finditer(
        r"No module named '([A-Za-z0-9_.]+)'",
        output
    ):
        names.append(match.group(1).split(".")[0])

    # ERROR: Unknown config option: timeout（strict-config 下直接退出）
    for match in re.finditer(
        r"Unknown config option: ([A-Za-z0-9_]+)",
        output
    ):
        plugin = PYTEST_OPTION_PLUGINS.get(match.group(1))
        if plugin:
            names.append(plugin)

    # 去重且保持顺序
    seen = set()
    unique = []
    for name in names:
        if name not in seen:
            seen.add(name)
            unique.append(name)

    return unique


def _suggest_similar_paths(missing_path):
    """
    在 Workspace 内查找与缺失路径同名的文件，给出修正建议。

    用于纠正 Agent 猜测出的错误路径，
    例如把 rich/containers.py 猜成 src/rich/containers.py。
    """

    basename = os.path.basename(
        missing_path.replace("\\", "/")
    )

    if not basename:
        return []

    skip_dirs = {
        ".git", ".venv", "__pycache__", ".tox",
        ".pytest_cache", "node_modules", "build", "dist",
        VALIDATION_DIR_NAME,
    }

    matches = []

    for dirpath, dirnames, filenames in os.walk(ACTIVE_WORKSPACE):
        dirnames[:] = [
            d for d in dirnames if d not in skip_dirs
        ]

        if basename in filenames:
            matches.append(
                os.path.relpath(
                    os.path.join(dirpath, basename),
                    ACTIVE_WORKSPACE
                ).replace("\\", "/")
            )

            if len(matches) >= 5:
                break

    return matches


def _suggest_test_targets(test_path):
    """
    为不存在的测试路径给出真实可运行的测试目标建议。

    两级匹配：
    1. Workspace 内同名文件（纠正目录猜错）；
    2. 按测试函数名搜索：把 tests/test_wrap.py 的 "wrap"
       拆成关键词，扫描所有测试文件的 def test_* 函数名，
       建议 file::test_func 形式的完整测试目标。
    """

    suggestions = list(
        _suggest_similar_paths(test_path)
    )

    stem = os.path.basename(
        test_path.replace("\\", "/")
    )

    if stem.endswith(".py"):
        stem = stem[:-3]

    keywords = set()

    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", stem):
        token = token.lower()

        if token in {"test", "tests", "py"}:
            continue

        keywords.add(token)

        if token.endswith("s"):
            keywords.add(token[:-1])

    if not keywords:
        return suggestions

    skip_dirs = {
        ".git", ".venv", "__pycache__", ".tox",
        ".pytest_cache", "node_modules", "build", "dist",
        VALIDATION_DIR_NAME, "docs", "examples",
    }

    pattern = re.compile(
        r"(?m)^\s*def\s+(test_[A-Za-z0-9_]+)"
    )

    suggested_files = {
        s.split("::")[0] for s in suggestions
    }

    found = []

    for dirpath, dirnames, filenames in os.walk(ACTIVE_WORKSPACE):
        dirnames[:] = [
            d for d in dirnames if d not in skip_dirs
        ]

        for filename in filenames:
            if not filename.endswith(".py"):
                continue

            full_path = os.path.join(dirpath, filename)

            try:
                with open(
                    full_path,
                    "r",
                    encoding="utf-8"
                ) as f:
                    content = f.read()
            except (OSError, UnicodeDecodeError):
                continue

            for match in pattern.finditer(content):
                func_name = match.group(1).lower()

                if any(k in func_name for k in keywords):
                    relative = os.path.relpath(
                        full_path,
                        ACTIVE_WORKSPACE
                    ).replace("\\", "/")

                    if relative not in suggested_files:
                        found.append(
                            f"{relative}::{match.group(1)}"
                        )

                        if len(found) >= 5:
                            return suggestions + found

    return suggestions + found


def _looks_like_test_path(path):
    normalized = path.replace("\\", "/").lower()

    return (
        normalized.startswith("test")
        or "/test" in normalized
        or normalized.endswith("_test.py")
    )


def _resolve_path_or_raise(path, prefix="文件不存在"):
    """
    解析 Agent 请求的文件路径。

    路径不存在时的策略：
    - Workspace 内恰好存在唯一同名文件 → 自动修正到真实路径。
      （Agent 经常凭训练记忆猜 src/ 布局，给出建议它也不采纳，
      直接自动修正最可靠，一次都不浪费轮数。）
    - 否则抛出带建议的 FileNotFoundError：
      测试类路径给函数级建议（file::test_func），
      其他路径给同名文件建议。

    返回 (target, corrected)：
    corrected 为 None 表示原路径有效，
    否则为自动修正后的 Workspace 相对路径。
    """

    target = get_workspace_path(path)

    if os.path.isfile(target):
        return target, None

    same_name = _suggest_similar_paths(path)

    if len(same_name) == 1:
        corrected = same_name[0]
        return get_workspace_path(corrected), corrected

    message = f"{prefix}：{path}"

    suggestions = (
        _suggest_test_targets(path)
        if _looks_like_test_path(path)
        else same_name
    )

    if suggestions:
        message += (
            "\nWorkspace 中实际可用的目标：\n- "
            + "\n- ".join(suggestions[:8])
        )

    raise FileNotFoundError(message)


def _write_dependency_marker(python):
    marker_path = os.path.join(
        ACTIVE_WORKSPACE,
        DEPENDENCY_MARKER
    )

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
    7. 如果 marker 丢失但环境实际已就绪（例如上次安装成功后
       进程中断，marker 没来得及写入），直接补写 marker，
       绝不在已安装的环境上重复执行 pip install。
    """

    if ACTIVE_WORKSPACE is None:
        raise RuntimeError("尚未设置 Workspace")

    venv_python = os.path.join(
        ACTIVE_WORKSPACE,
        ".venv",
        "Scripts" if os.name == "nt" else "bin",
        "python.exe" if os.name == "nt" else "python"
    )

    if not os.path.isfile(venv_python):
        raise RuntimeError(
            "Workspace 缺少 .venv 虚拟环境，"
            "请先通过 workspace_manager.create_workspace 创建，"
            "不能使用宿主 Python 安装项目依赖。"
        )

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
    # 快速路径：marker 不存在，但环境实际已经就绪。
    #
    # 典型场景：依赖全部安装成功，但 marker 写入前进程被中断。
    # 此时重复 pip install -e . 反而可能因为卸载重装触发
    # Windows 文件占用错误（WinError 5），必须跳过。
    #
    # 用真实 pytest 收集来验证：能收集成功说明
    # pytest、项目依赖、项目本体全部可用。
    # --------------------------------------------------------

    if (
        _pytest_available(python)
        and _tests_collectable(python)
    ):
        _write_dependency_marker(python)

        return {
            "status": "READY",
            "python": python,
            "message": "检测到依赖环境已就绪，已补写 marker，跳过重复安装。"
        }

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

        code, output = _pip_install_step(
            python,
            ["-r", relative],
            f"install_{relative}",
            install_log
        )

        if code != 0:
            raise RuntimeError(
                "项目依赖安装失败："
                f"{relative}\n{output}"
            )

    # --------------------------------------------------------
    # pyproject/setup 项目：安装当前项目本身
    # --------------------------------------------------------

    if _has_package_metadata():
        code, output = _pip_install_step(
            python,
            ["-e", "."],
            "install_project_editable",
            install_log
        )

        if code != 0:
            raise RuntimeError(
                "项目本体安装失败。\n"
                f"{output}"
            )

    # --------------------------------------------------------
    # pytest 是 Agent 测试闭环的硬依赖
    # --------------------------------------------------------

    if not _pytest_available(python):
        code, output = _pip_install_step(
            python,
            ["pytest"],
            "install_pytest",
            install_log,
            timeout=600
        )

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
    # 安装完成后再次做真实测试收集。只有 pytest + 项目依赖都能
    # 被导入，才允许写 READY marker；否则不能把“pytest 存在”误判为环境完整。
    if not _tests_collectable(python):

        # ------------------------------------------------
        # 收集失败 → 解析缺失的第三方包并自动补装。
        #
        # 典型场景：fastapi 等项目对传递依赖不设上界
        # （starlette>=0.46.0），pip 会装到最新 starlette，
        # 而新版 TestClient 依赖新包（httpx2），
        # 项目自身的依赖声明里没有它。
        #
        # 注意：模块导入在第一个缺失包就会中断，装完一批后
        # 同一批文件可能暴露出下一批缺失包，因此用有界循环
        # （最多 5 轮，每个包只尝试安装一次）逐批补装。
        # ------------------------------------------------

        attempted = set()

        for _ in range(5):

            missing = _collect_error_packages(python)

            new = [name for name in missing if name not in attempted]

            if not new:
                break

            attempted.update(new)

            code, output = _pip_install_step(
                python,
                ["install"] + new,
                "install_missing_test_deps",
                install_log,
                timeout=600
            )

            if code != 0:
                break

            if _tests_collectable(python):
                _write_dependency_marker(python)

                return {
                    "status": "READY",
                    "python": python,
                    "message": (
                        "检测到测试依赖缺失，已自动补装："
                        + ", ".join(sorted(attempted))
                    ),
                    "install_log": install_log
                }

        return {
            "status": "NOT_READY",
            "python": python,
            "message": (
                "依赖安装流程完成，但 pytest 收集项目测试仍失败。"
                "不会写入 READY marker，避免把不完整环境误判为可用。"
            ),
            "install_log": install_log,
        }

    # 写 marker
    # --------------------------------------------------------

    _write_dependency_marker(python)

    return {
        "status": "READY",
        "python": python,
        "message": "Workspace 依赖环境已自动准备完成。",
        "install_log": install_log
    }


# ============================================================
# read_file
# ============================================================

# 已读取过的 (文件, 范围)。文件未变化时重复读取相同范围是纯浪费，
# 直接返回短提示；replace_in_file 修改成功后清除对应文件的记录。
_read_ranges = {}


def read_file(path, start_line=1, max_lines=200):
    target, corrected = _resolve_path_or_raise(path)

    real_path = corrected or path

    range_key = (
        os.path.normcase(os.path.abspath(target)),
        start_line,
        max_lines
    )

    if range_key in _read_ranges:
        total = _read_ranges[range_key]
        return (
            f"[READ-BLOCKED] 文件 {real_path} 第 {start_line}-"
            f"{min(start_line + max_lines - 1, total)} 行"
            "已在本会话中读取过，且文件未发生变化，本次读取被拦截。\n"
            "如果当前上下文中仍然看得到该内容，请直接基于已读内容继续工作；\n"
            "如果上下文中已看不到（早期内容可能被裁剪），"
            "请改用更小的 max_lines 分段重新读取——不同参数不会被拦截。"
        )

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

    header = ""

    if corrected:
        header = (
            f"[路径自动修正] 请求的 {path} 不存在，"
            f"已自动读取唯一同名文件 {corrected}。\n"
            "后续所有工具调用请直接使用真实路径。\n\n"
        )

    start_index = start_line - 1
    end_index = start_index + max_lines

    selected = lines[start_index:end_index]

    if not selected:
        return (
            f"{header}文件 {real_path} 共 {len(lines)} 行，"
            f"第 {start_line} 行超出文件范围。"
            f"请使用 1 到 {len(lines)} 之间的 start_line。"
        )

    content = "\n".join(
        f"{index}: {line.rstrip()}"
        for index, line in enumerate(
            selected,
            start=start_line
        )
    )

    last_line = start_line + len(selected) - 1

    _read_ranges[range_key] = len(lines)

    return (
        f"{header}{content}\n"
        f"[文件 {real_path} 共 {len(lines)} 行，"
        f"当前显示 {start_line}-{last_line} 行]"
    )


# ============================================================
# replace_in_file
# ============================================================

def _is_forbidden_edit_path(rel_path):
    rel = rel_path.replace("\\", "/").lower().strip("/")

    return (
        rel == "tests"
        or rel.startswith("tests/")
        or "/tests/" in rel
        or rel == VALIDATION_DIR_NAME
        or rel.startswith(VALIDATION_DIR_NAME + "/")
    )


def replace_in_file(path, old, new):
    if _is_forbidden_edit_path(path):
        raise PermissionError(
            "Agent 不允许修改 tests 目录或临时验证目录中的文件。"
        )

    target, corrected = _resolve_path_or_raise(path)

    # 自动修正可能落到 tests/ 等禁止修改的路径上，必须复查
    if corrected and _is_forbidden_edit_path(corrected):
        raise PermissionError(
            f"路径自动修正后指向 {corrected}，"
            "Agent 不允许修改 tests 目录或临时验证目录中的文件。"
        )

    real_path = corrected or path

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

    # 文件内容已变化，清除该文件的已读范围记录，允许重新读取
    changed_key_prefix = os.path.normcase(
        os.path.abspath(target)
    )
    for key in list(_read_ranges):
        if key[0] == changed_key_prefix:
            del _read_ranges[key]

    global RAG_INDEX, RAG_CHUNKS, RAG_WORKSPACE
    RAG_INDEX = None
    RAG_CHUNKS = None
    RAG_WORKSPACE = None
    RAG_QUERY_CACHE.clear()

    message = "文件修改成功，只进行了 1 处精确替换。"

    if corrected:
        message += (
            f"（路径已自动修正：{path} → {real_path}，"
            "后续请直接使用真实路径。）"
        )

    return {
        "status": "SUCCESS",
        "path": real_path,
        "message": message
    }


def replace_lines(path, start_line, end_line, new_code):
    """
    按真实行号替换 Workspace 内源文件的连续行。

    这是 Agent 的首选编辑接口：LLM 只需要根据 read_file 返回的行号
    指定范围，不需要复述一大段 old 文本，因此比精确字符串替换稳定。
    start_line / end_line 均为 1-based 且包含端点。
    """
    if _is_forbidden_edit_path(path):
        raise PermissionError(
            "Agent 不允许修改 tests 目录或临时验证目录中的文件。"
        )

    target, corrected = _resolve_path_or_raise(path)
    real_path = corrected or path

    if corrected and _is_forbidden_edit_path(corrected):
        raise PermissionError(
            f"路径自动修正后指向 {corrected}，禁止修改。"
        )

    try:
        start_line = int(start_line)
        end_line = int(end_line)
    except (TypeError, ValueError) as exc:
        raise ValueError("start_line 和 end_line 必须是整数。") from exc

    if start_line < 1 or end_line < start_line:
        raise ValueError("行号范围无效：必须满足 1 <= start_line <= end_line。")

    if not isinstance(new_code, str):
        raise TypeError("new_code 必须是字符串。")

    with open(target, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if start_line > len(lines) or end_line > len(lines):
        raise ValueError(
            f"行范围 {start_line}-{end_line} 超出文件范围；"
            f"文件只有 {len(lines)} 行。"
        )

    actual_end = end_line
    replacement = new_code.splitlines(keepends=True)
    if replacement and not replacement[-1].endswith("\n"):
        replacement[-1] += "\n"

    lines[start_line - 1:actual_end] = replacement

    with open(target, "w", encoding="utf-8") as f:
        f.writelines(lines)

    changed_key_prefix = os.path.normcase(os.path.abspath(target))
    for key in list(_read_ranges):
        if key[0] == changed_key_prefix:
            del _read_ranges[key]

    global RAG_INDEX, RAG_CHUNKS, RAG_WORKSPACE
    RAG_INDEX = None
    RAG_CHUNKS = None
    RAG_WORKSPACE = None
    RAG_QUERY_CACHE.clear()

    return {
        "status": "SUCCESS",
        "path": real_path,
        "start_line": start_line,
        "end_line": actual_end,
        "lines_changed": actual_end - start_line + 1,
        "message": "文件按行号修改成功。" + (
            f"（路径已自动修正：{path} → {real_path}）"
            if corrected else ""
        ),
    }


# 兼容部分旧 Agent / prompt 使用的名称
edit_file = replace_in_file


# ============================================================
# check_syntax
# ============================================================

def check_syntax(path):
    target, corrected = _resolve_path_or_raise(path)

    real_path = corrected or path

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
        message = "语法检查通过。"

        if corrected:
            message += (
                f"（路径已自动修正：{path} → {real_path}，"
                "后续请直接使用真实路径。）"
            )

        return {
            "status": "PASS",
            "path": real_path,
            "message": message
        }

    return {
        "status": "FAIL",
        "path": real_path,
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
        target, corrected = _resolve_path_or_raise(
            file_part,
            prefix="测试文件不存在"
        )

        if corrected:
            if "::" in test_path:
                func = test_path.split("::", 1)[1]
                test_path = f"{corrected}::{func}"
            else:
                test_path = corrected

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

    result_payload = {
        "status": status,
        "command": " ".join(command),
        "returncode": result.returncode,
        "output": output[-12000:]
    }

    if status == "FAIL":
        # 附带结构化失败详情（失败测试条目 + FAILURES 段源码帧），
        # agent 层用 frame_paths 与被修改文件比对，
        # 判定失败是否由本次修改引起。
        result_payload.update(_parse_pytest_failures(output))

    return result_payload


# ============================================================
# list_files
# ============================================================

def _normalize_dir_key(path):
    key = path.replace("\\", "/").strip()

    while key.startswith("./"):
        key = key[2:]

    return key.strip("/").lower() or "."


def list_files(path="."):
    global _listed_dirs

    key = _normalize_dir_key(path)

    if key in _listed_dirs:
        _listed_dirs[key] += 1

        return (
            f"目录 {path} 的内容已经列出过，与上文完全相同，"
            f"这是第 {_listed_dirs[key]} 次重复请求。\n"
            "任务过程中目录内容不会变化，不要重复调用 list_files。\n"
            "请基于上文已有的目录列表和检索结果直接行动；"
            "如果要找的文件不存在，错误信息中会给出真实路径建议。"
        )

    target = get_workspace_path(path)

    if not os.path.isdir(target):
        message = f"目录不存在：{path}"

        skip_dirs = {
            ".git", ".venv", "__pycache__", ".tox",
            ".pytest_cache", VALIDATION_DIR_NAME,
        }

        root_dirs = [
            name for name in sorted(os.listdir(ACTIVE_WORKSPACE))
            if os.path.isdir(os.path.join(ACTIVE_WORKSPACE, name))
            and name not in skip_dirs
        ]

        if root_dirs:
            message += (
                "\nWorkspace 根目录下实际存在的目录：\n- "
                + "\n- ".join(root_dirs[:12])
            )

        raise NotADirectoryError(message)

    entries = []

    for name in sorted(os.listdir(target)):
        full_path = os.path.join(target, name)

        if name in {
            ".git",
            "__pycache__",
            ".venv",
            VALIDATION_DIR_NAME,
            DEPENDENCY_MARKER,
            SESSION_FILE_NAME,
        }:
            continue

        if os.path.isdir(full_path):
            entries.append(f"[DIR]  {name}")
        else:
            entries.append(f"[FILE] {name}")

    _listed_dirs[key] = 1

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

    query_key = " ".join(str(query or "").lower().split())
    cache_key = (os.path.abspath(ACTIVE_WORKSPACE), query_key)

    if query_key and cache_key in RAG_QUERY_CACHE:
        return (
            format_results(RAG_QUERY_CACHE[cache_key])
            if format_results is not None
            else str(RAG_QUERY_CACHE[cache_key])
        )

    try:
        results = retrieve_code(
            query,
            root=ACTIVE_WORKSPACE
        )
        if query_key:
            RAG_QUERY_CACHE[cache_key] = results

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

        if query_key:
            RAG_QUERY_CACHE[cache_key] = results

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

    # --------------------------------------------------------
    # 预检：立即收集测试（pytest --collect-only）。
    # 导入错误 / 语法错误在这一步就暴露并返回 BROKEN_TEST，
    # 避免浪费一整轮到 run_bug_validation 才发现测试跑不起来。
    # --------------------------------------------------------

    env = ensure_test_environment()

    validation_rel = os.path.join(
        VALIDATION_DIR_NAME,
        VALIDATION_FILE_NAME
    )

    collect_error = None

    try:
        collect = subprocess.run(
            [
                env["python"],
                "-m",
                "pytest",
                "--collect-only",
                validation_rel
            ],
            cwd=ACTIVE_WORKSPACE,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120
        )

        collect_output = (
            (collect.stdout or "")
            + "\n"
            + (collect.stderr or "")
        )

        # 0 = 收集成功；5 = 没有收集到测试（regex 已保证有 test_ 函数，
        # 走到 5 说明命名/缩进等更隐蔽的问题，同样按损坏处理）
        if collect.returncode not in (0, 5):
            collect_error = collect_output

    except subprocess.TimeoutExpired:
        collect_error = (
            "pytest --collect-only 预检超时（120 秒），"
            "测试文件模块级代码可能存在死循环或极重操作。"
        )

    if collect_error is not None:

        error_types = sorted(set(re.findall(
            r"(ImportError|ModuleNotFoundError|SyntaxError|"
            r"NameError|AttributeError)",
            collect_error
        )))

        return {
            "status": "BROKEN_TEST",
            "error_types": error_types,
            "message": (
                "验证测试创建失败：测试文件无法被 pytest 收集"
                "（导入/语法/命名错误），运行必然失败。\n"
                "请根据下方错误修正测试代码后"
                "重新调用 create_bug_validation。\n"
                "常见原因：从错误的模块导入了不存在的符号"
                "（先用 read_file 确认目标符号的真实所在模块）。"
            ),
            "output": collect_error[-8000:]
        }

    # 探针式测试（assert False 只为查看真实输出）：
    # 它的 FAIL 不是有效的 Bug 复现证据，标记出来，
    # 运行后框架不会把它当作 baseline 确认。
    is_probe = bool(re.search(
        r"assert\s+False",
        test_code
    ))

    message = (
        "临时 Bug 回归验证已创建（预检通过：测试可被收集）。"
        "必须先在未修改代码的状态运行 run_bug_validation。"
    )

    if is_probe:
        message += (
            "\n注意：检测到探针式断言（assert False）。"
            "该测试的失败只用于展示真实输出，"
            "不会被当作有效的 Bug 复现（baseline）。"
            "查看输出后请立即用真实断言重写验证测试，"
            "再次 create_bug_validation。"
        )

    return {
        "status": "CREATED",
        "available": True,
        "is_probe": is_probe,
        "path": os.path.join(
            VALIDATION_DIR_NAME,
            VALIDATION_FILE_NAME
        ),
        "message": message
    }


# ============================================================
# pytest 失败输出解析（run_bug_validation 专用）
# ============================================================

# 断言失败相关类型：这是"有效 baseline FAIL"的证据，
# 说明 Bug 被成功复现，而不是验证测试自身写错。
_ASSERTION_FAILURE_NAMES = {
    "AssertionError",
    "Failed",
}

# 常见"测试自身错误"异常类型：验证测试调用了不存在的 API、
# 传错参数或导入失败时抛出，无论源代码是否修复都会失败。
_TEST_ERROR_EXCEPTION_NAMES = {
    "TypeError",
    "AttributeError",
    "ImportError",
    "ModuleNotFoundError",
    "NameError",
    "UnboundLocalError",
    "SyntaxError",
    "IndentationError",
    "KeyError",
    "IndexError",
    "ValueError",
    "RuntimeError",
    "ZeroDivisionError",
    "FileNotFoundError",
    "FileExistsError",
    "NotADirectoryError",
    "IsADirectoryError",
    "PermissionError",
    "RecursionError",
    "StopIteration",
    "StopAsyncIteration",
    "OSError",
    "IOError",
    "TimeoutError",
    "ArithmeticError",
    "LookupError",
    "UnicodeDecodeError",
    "UnicodeEncodeError",
}

# pytest 输出中异常类型出现的三类位置：
# 1) "E   TypeError: ..." / "E   pkg.mod.CustomError: ..." 错误行；
# 2) 摘要行 "FAILED path::test - TypeError: ..." /
#    "ERROR path::test - ImportError: ..."；
# 3) traceback 尾行 ".../test_x.py:12: AssertionError"。
_E_LINE_PATTERN = re.compile(
    r"^E\s+(?:[\w.]+\.)?([A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE
)

_SUMMARY_LINE_PATTERN = re.compile(
    r"^(?:FAILED|ERROR)\s+\S+\s+-\s+(?:[\w.]+\.)?([A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE
)

_TRACEBACK_TAIL_PATTERN = re.compile(
    r"^(?:[A-Za-z]:)?[\w./\\-]+\.py:\d+:\s*([A-Za-z_][A-Za-z0-9_]*)\s*$",
    re.MULTILINE
)


def _parse_pytest_failure(output):
    """
    解析 pytest FAIL 输出，区分断言失败与测试自身错误。

    不能只依赖 "^E\\s+XxxError" 单一正则：
    - 断言失败：E 行是 "E   assert ..."，traceback 尾行为
      "...: AssertionError"；
    - 测试函数内部异常：E 行直接是异常类型
      （TypeError / AttributeError / ImportError / ...）；
    - fixture setup 失败：出现 "_ ERROR at setup of ..." 段；
    - 收集/导入失败：出现 "ERROR collecting ..." 段。

    返回 dict：
      failure_kind: "assertion" / "test_error" /
                    "collection_error" / "unknown"
      error_types:  排序去重后的异常类型名列表
    """

    error_types = set()

    for pattern in (
        _E_LINE_PATTERN,
        _SUMMARY_LINE_PATTERN,
        _TRACEBACK_TAIL_PATTERN,
    ):
        for match in pattern.finditer(output):
            error_types.add(match.group(1))

    assertion_hits = error_types & _ASSERTION_FAILURE_NAMES

    # 注意：AssertionError / Failed 本身也以 "Error" 结尾，
    # 必须先从"测试自身错误"集合中排除，否则纯断言失败
    # （baseline 的合法复现证据）会被误判成 test_error。
    error_hits = {
        name
        for name in error_types
        if name not in _ASSERTION_FAILURE_NAMES
        and (
            name in _TEST_ERROR_EXCEPTION_NAMES
            or name.endswith("Error")
            or name.endswith("Exception")
        )
    }

    if re.search(
        r"^_+ ERROR collecting |^ERROR collecting ",
        output,
        re.MULTILINE
    ):
        # 收集阶段失败：测试（或其导入的模块）跑不起来，
        # 其 E 行中的 ImportError 等只是失败原因，不单判定。
        failure_kind = "collection_error"
    elif error_hits:
        failure_kind = "test_error"
    elif re.search(
        r"^_+ ERROR at setup of ",
        output,
        re.MULTILINE
    ):
        # fixture 自身抛错（可能是自定义异常类型，
        # E 行识别不出来），同样属于测试自身错误。
        failure_kind = "test_error"
    elif assertion_hits:
        failure_kind = "assertion"
    else:
        # 无法识别时保守处理：默认按断言失败对待，
        # 避免把真实 Bug 复现误判成验证测试损坏。
        failure_kind = "unknown"

    return {
        "failure_kind": failure_kind,
        "error_types": sorted(error_hits | assertion_hits)
    }


# run_test FAIL 输出的失败条目：短摘要行形如
#   FAILED tests/test_x.py::test_case - AssertionError: assert 1 == 2
#   ERROR tests/test_x.py::test_case - ImportError: No module named 'x'
# 错误类型要求大写开头，避免把 "- assert ..." 误认成异常类型。
_FAILED_SUMMARY_ENTRY_PATTERN = re.compile(
    r"^(?:FAILED|ERROR)\s+(?!collecting\s)(\S+)"
    r"(?:\s+-\s+([A-Z][A-Za-z0-9_]*))?",
    re.MULTILINE
)

# traceback 中出现的源码帧路径，形如
#   D:\proj\src\module.py:42        （断言/异常抛出点）
#   src/module.py:42: in buggy_func （调用链中间帧）
_FRAME_PATH_PATTERN = re.compile(
    r"(?:[A-Za-z]:)?[^\s'\"\[\](){}]*\.py:\d+"
)


def _pytest_failure_region(output):
    """
    截取 pytest 输出中的 FAILURES 段。

    帧路径只从该段提取：warnings summary 等其他段也可能提到
    被修改文件的路径（如弃用警告指向导入点），混入后会把
    与本次修改无关的失败误判成相关失败。

    找不到 FAILURES 标记时保守退回整个输出
    （宁可把失败误判成相关，也不放过该阻断的情况）。
    """
    start = output.find("=== FAILURES ===")
    if start == -1:
        return output
    end = len(output)
    for marker in (
        "=== short test summary info ===",
        "=== warnings summary ===",
    ):
        pos = output.find(marker, start + 1)
        if pos != -1:
            end = min(end, pos)
    return output[start:end]


def _parse_pytest_failures(output):
    """
    解析 run_test 的 FAIL 输出，提取：
    - failures:     每个失败测试的条目 {test_id, error_type}
    - frame_paths:  FAILURES 段 traceback 中的所有 .py:N 源码帧路径

    agent 层用 frame_paths 判定失败是否由本次修改引起：
    如果被修改文件没有出现在任何失败的 traceback 调用链里，
    本次修改不可能导致这些失败（同进程 pytest 语义），
    即可记录为"与本次修改无关的失败"，不阻断完成条件。
    """
    failures = []
    for match in _FAILED_SUMMARY_ENTRY_PATTERN.finditer(output):
        failures.append({
            "test_id": match.group(1),
            "error_type": match.group(2)
        })

    failure_region = _pytest_failure_region(output)
    frame_paths = []
    seen = set()
    for match in _FRAME_PATH_PATTERN.finditer(failure_region):
        path = match.group(0)
        if path not in seen:
            seen.add(path)
            frame_paths.append(path)

    return {
        "failures": failures,
        "frame_paths": frame_paths
    }


def run_bug_validation():
    """
    运行 .agent_validation/test_bug_validation.py 并解析失败原因。

    不通过 replace_in_file 修改测试文件。

    状态语义（tools 层只负责解析，不负责终审路由）：
    - PASS / NO_TESTS / TIMEOUT：原样返回；
    - INTERRUPTED / INTERNAL_ERROR / USAGE_ERROR：返回
      BROKEN_TEST（收集/导入阶段失败）；
    - FAIL：解析 traceback 后附带 failure_kind
      （assertion / test_error / collection_error / unknown）
      与 error_types 返回。是否按 BROKEN_TEST 处理由
      agent 层根据 validation_baseline_confirmed 状态决定——
      baseline 确认之后，任何 FAIL 都必须走源码修复路径，
      不允许再降级为 BROKEN_TEST。
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

    result = run_test(
        test_path=validation_path
    )

    if not isinstance(result, dict):
        return result

    status = result.get("status")
    output = result.get("output", "")

    # --------------------------------------------------------
    # 收集阶段的错误（INTERRUPTED = returncode 2 等）：
    # 测试文件本身导入/语法有问题，无论源代码是否修复都跑不起来，
    # 属于 BROKEN_TEST，必须修正后重新创建。
    # --------------------------------------------------------

    if status in ("INTERRUPTED", "INTERNAL_ERROR", "USAGE_ERROR"):
        return {
            "status": "BROKEN_TEST",
            "error_types": ["CollectionError"],
            "failure_kind": "collection_error",
            "message": (
                "验证测试无法运行：pytest 在收集阶段就出错"
                "（导入失败 / 语法错误 / 用法错误），"
                "这不是断言失败，无论源代码是否修复都跑不起来。\n"
                "请根据下方 traceback 修正测试代码"
                "（常见：从不存在的模块导入符号），"
                "然后重新调用 create_bug_validation。"
            ),
            "output": output[-8000:]
        }

    # --------------------------------------------------------
    # FAIL：解析 traceback，区分断言失败与测试自身错误。
    # 注意：这里不再直接把 FAIL 改判为 BROKEN_TEST——
    # baseline 确认之后，任何 FAIL（包括测试自身抛错）都必须
    # 保持 FAIL，走"修复源代码"路径；是否降级为 BROKEN_TEST
    # 由 agent 层根据 validation_baseline_confirmed /
    # validation_frozen 状态在 baseline 之前路由。
    # --------------------------------------------------------

    if status == "FAIL":
        parsed = _parse_pytest_failure(output)
        failure_kind = parsed["failure_kind"]
        error_types = parsed["error_types"]

        if failure_kind == "assertion":
            guidance = (
                "这是 AssertionError 断言失败（有效失败证据）："
                "如果尚未修改源代码，说明 Bug 成功复现。"
                "禁止修改本验证测试。"
            )
        elif failure_kind == "test_error":
            guidance = (
                "这不是断言失败，而是测试函数内部抛出了异常"
                "（常见：调用了不存在的 API、传错参数、导入失败），"
                "无论源代码是否修复该测试都会失败。\n"
                "如果尚未建立 baseline，应根据 traceback 用真实存在的 "
                "API 重写验证测试，并重新调用 create_bug_validation。"
            )
        elif failure_kind == "collection_error":
            guidance = (
                "pytest 在收集阶段失败：验证测试（或其导入的源模块）"
                "存在导入/语法错误。\n"
                "如果尚未建立 baseline，应根据 traceback 重写验证测试"
                "并重新调用 create_bug_validation。"
            )
        else:
            guidance = (
                "无法从输出中识别出明确的异常类型，默认按断言失败处理。"
                "请阅读完整 traceback 判断失败原因。"
            )

        result = dict(result)
        result["failure_kind"] = failure_kind
        result["error_types"] = error_types
        result["message"] = (
            "验证测试失败。"
            f"失败类型：{failure_kind}；"
            f"涉及异常：{ '、'.join(error_types) or '未识别' }。\n"
            + guidance
        )
        result["output"] = output[-8000:]
        return result

    return result


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
    "replace_lines": replace_lines,
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
    "replace_lines",
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