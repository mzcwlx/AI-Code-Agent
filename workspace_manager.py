import os
import shutil
import stat
import subprocess
import time
import sys
import zipfile


WORKSPACE_ROOT = "workspaces"
WORKSPACE_TTL_HOURS = 24  # workspace 保留时长（小时），超过后自动删除

# GitHub 仓库 ZIP 下载超时（秒）。
# 仅用于 GitHub API 下载仓库 ZIP，与 LLM 的 timeout / retry 完全无关。
GITHUB_ZIP_TIMEOUT = 120


def run_command(command, cwd=None):
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"命令执行失败：\n"
            f"命令：{' '.join(command)}\n"
            f"错误：{result.stderr}"
        )

    return result.stdout


def get_python_executable(workspace):
    if os.name == "nt":
        return os.path.join(
            workspace,
            ".venv",
            "Scripts",
            "python.exe"
        )

    return os.path.join(
        workspace,
        ".venv",
        "bin",
        "python"
    )


def prepare_workspace(workspace):
    """
    准备 Workspace 的 Python 虚拟环境和测试依赖。

    依赖安装统一由 tools.ensure_test_environment 负责：
    自动识别 pyproject/setup/requirements，安装项目本体，
    确保 pytest 存在，成功后写 marker 防止重复安装。
    """

    python = get_python_executable(workspace)

    # 1. 创建虚拟环境
    if os.path.isfile(python):
        print(f"虚拟环境已存在：{python}")
    else:
        print("未找到虚拟环境，正在创建 .venv...")

        run_command(
            [sys.executable, "-m", "venv", ".venv"],
            cwd=workspace
        )

        if not os.path.isfile(python):
            raise RuntimeError(
                f"虚拟环境创建失败：{python}"
            )

        print(f"虚拟环境创建完成：{python}")

    # 2. 自动准备依赖环境（失败不阻断，run_test 时会再次尝试）
    try:
        import tools

        tools.set_workspace(workspace)
        result = tools.ensure_test_environment()

        print(f"依赖环境：{result.get('status')}")
        print(result.get("message", ""))
    except Exception as e:
        print(f"⚠️ 依赖环境自动准备失败，Agent 运行测试时会再次尝试：{e}")

    return python


def touch_workspace(workspace):
    """刷新 workspace 的"最后使用时间"（目录 mtime）。

    每次 prepare/create 前调用一次，正在使用的 workspace
    就不会超过 TTL，不会被后台清理线程误删。
    """
    try:
        os.utime(workspace)
    except OSError:
        pass


def _rmtree_force(path):
    """删除目录树；git 对象文件是只读的，Windows 上需先解锁再删。"""

    def onerror(func, p, _exc_info):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except OSError:
            pass

    shutil.rmtree(path, onerror=onerror)


def cleanup_stale_workspaces(ttl_hours=WORKSPACE_TTL_HOURS,
                             workspace_root=None,
                             protected_ids=None):
    """
    删除 workspace_root 下保存超过 ttl_hours 的 workspace 目录。

    "最后使用时间"取目录 mtime；
    create_workspace / reset 时都会刷新 mtime，在用任务不会被误删。

    protected_ids：额外保留的目录名集合（本轮即将使用的 instance_id）。
    返回被删除的目录名列表。
    """
    root = workspace_root or WORKSPACE_ROOT
    if not os.path.isdir(root):
        return []

    protected = set(protected_ids or ())
    ttl_seconds = ttl_hours * 3600
    now = time.time()
    deleted = []

    for name in os.listdir(root):
        path = os.path.join(root, name)
        if not os.path.isdir(path) or name in protected:
            continue
        try:
            st = os.stat(path)
        except OSError:
            continue
        last_active = st.st_mtime
        if now - last_active >= ttl_seconds:
            _rmtree_force(path)
            deleted.append(name)

    return deleted


def _download_repo_zip(repo, base_commit, zip_path):
    """通过 GitHub API 下载指定 base_commit 的仓库 ZIP。

    服务器无法稳定访问 https://github.com（git clone 不可用），
    但 https://api.github.com 已在服务器上验证可访问，
    因此改用 curl 下载 zipball。

    注意：URL 里必须用 base_commit，而不是 main，
    保证拿到的是 benchmark 指定版本的代码。
    """
    zip_url = f"https://api.github.com/repos/{repo}/zipball/{base_commit}"

    run_command([
        "curl",
        "-4",                        # 只走 IPv4
        "-L",                        # 跟随重定向
        "--fail",                    # HTTP 错误（如 404/422）直接非 0 退出
        "--silent", "--show-error",  # 不刷进度条，但保留错误信息
        "--max-time", str(GITHUB_ZIP_TIMEOUT),
        "-H", "Accept: application/vnd.github+json",
        "-o", zip_path,
        zip_url,
    ])

    # curl 成功但内容异常（例如被代理劫持返回 HTML）时兜底拦截
    if not os.path.isfile(zip_path) or os.path.getsize(zip_path) == 0:
        raise RuntimeError(f"GitHub ZIP 下载失败（文件为空）：{zip_url}")

    if not zipfile.is_zipfile(zip_path):
        raise RuntimeError(f"GitHub ZIP 下载失败（内容不是有效 ZIP）：{zip_url}")


def _extract_repo_zip(zip_path, workspace, extract_dir):
    """解压 GitHub ZIP，并把唯一顶层目录的内容复制到 workspace 根目录。

    GitHub zipball 解压后不是直接散开在根目录，而是包在
    "owner-repo-xxxxxxxx/" 这样的唯一顶层目录里。
    必须拆掉这一层，最终结果是：

    workspace/
        file1
        file2
        ...
    """
    # 清理上次异常退出可能残留的解压目录，避免旧文件混入
    if os.path.isdir(extract_dir):
        _rmtree_force(extract_dir)

    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    entries = os.listdir(extract_dir)

    if len(entries) != 1:
        raise RuntimeError(
            f"ZIP 结构异常：期望唯一顶层目录，实际包含：{entries}"
        )

    top_dir = os.path.join(extract_dir, entries[0])

    if not os.path.isdir(top_dir):
        raise RuntimeError(f"ZIP 结构异常：顶层不是目录：{entries[0]}")

    os.makedirs(workspace, exist_ok=False)

    for name in os.listdir(top_dir):
        shutil.move(
            os.path.join(top_dir, name),
            os.path.join(workspace, name)
        )


def _init_local_git_baseline(workspace):
    """把 ZIP 源码变成可用于 git diff / 状态管理的本地 Git 仓库。

    ZIP 不带 .git，但系统后续依赖 git diff / git status /
    修复 diff 等，因此本地重新 git init 并提交基线。

    - 不添加 GitHub remote，之后也不会访问 github.com；
    - 用 git -c 临时配置 user.name / user.email，
      不修改服务器的全局 Git 配置。
    """
    run_command(["git", "init"], cwd=workspace)
    run_command(["git", "add", "-A"], cwd=workspace)
    run_command(
        [
            "git",
            "-c", "user.name=AI-Code-Agent",
            "-c", "user.email=ai-code-agent@localhost",
            "commit",
            "-m", "Initial workspace",
        ],
        cwd=workspace
    )


def create_workspace(repo, base_commit, instance_id):
    os.makedirs(WORKSPACE_ROOT, exist_ok=True)

    workspace = os.path.abspath(
        os.path.join(WORKSPACE_ROOT, instance_id)
    )

    # 刷新"最后使用时间"，避免长任务在用期间被后台清理线程误删
    touch_workspace(workspace)

    if os.path.exists(workspace):
        print(f"Workspace 已存在：{workspace}")
        prepare_workspace(workspace)
        return workspace

    # 临时文件：下载的 ZIP 和解压目录（点前缀，处理完成后立即删除，
    # 不作为永久文件留在 WORKSPACE_ROOT）
    zip_path = os.path.join(WORKSPACE_ROOT, f".{instance_id}.zip")
    extract_dir = os.path.join(WORKSPACE_ROOT, f".{instance_id}_extract")

    print(f"正在下载仓库：{repo}")
    print(f"目标版本：{base_commit}")

    try:
        # 1. GitHub API 下载指定 base_commit 的 ZIP
        #    （服务器无法 git clone github.com，api.github.com 已验证可用）
        _download_repo_zip(repo, base_commit, zip_path)

        print("GitHub 仓库下载完成")

        # 2. 解压，把源码复制到 workspace 根目录（拆掉 zipball 顶层目录）
        print("正在解压...")
        _extract_repo_zip(zip_path, workspace, extract_dir)

        # 3. 重建本地 Git 基线（ZIP 不带 .git，但系统依赖 git diff）
        _init_local_git_baseline(workspace)

        print(f"Workspace 创建完成：{workspace}")
    except Exception as e:
        # 清理所有半成品：
        # 临时 ZIP / 临时解压目录 / 未完成的 workspace 全部删除，
        # 抛出明确异常。绝不能留下半成品 workspace，
        # 否则下次 os.path.exists(workspace) 会误判为已创建成功。
        for path in (extract_dir, workspace):
            if os.path.isdir(path):
                _rmtree_force(path)

        if os.path.isfile(zip_path):
            try:
                os.remove(zip_path)
            except OSError:
                pass

        raise RuntimeError(
            f"创建 Workspace 失败（{repo} @ {base_commit}）：{e}"
        ) from e
    finally:
        # 正常路径也要立即清掉临时文件
        if os.path.isdir(extract_dir):
            _rmtree_force(extract_dir)

        if os.path.isfile(zip_path):
            try:
                os.remove(zip_path)
            except OSError:
                pass

    prepare_workspace(workspace)

    return workspace


if __name__ == "__main__":
    from task_loader import get_task

    task = get_task("fastapi_15661")

    workspace = create_workspace(
        repo=task["repo"],
        base_commit=task["base_commit"],
        instance_id=task["instance_id"]
    )

    print("\n最终 Workspace：")
    print(workspace)