import os
import shutil
import stat
import subprocess
import time


WORKSPACE_ROOT = "workspaces"
WORKSPACE_TTL_HOURS = 24  # workspace 保留时长（小时），超过后自动删除


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
            ["python", "-m", "venv", ".venv"],
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

    repo_url = f"https://github.com/{repo}.git"

    print(f"正在克隆仓库：{repo}")

    run_command([
        "git",
        "clone",
        repo_url,
        workspace
    ])

    print(f"正在切换到版本：{base_commit}")

    run_command(
        ["git", "checkout", base_commit],
        cwd=workspace
    )

    print(f"Workspace 创建完成：{workspace}")

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