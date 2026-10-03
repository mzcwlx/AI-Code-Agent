import os
import subprocess


WORKSPACE_ROOT = "workspaces"


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


def create_workspace(repo, base_commit, instance_id):
    os.makedirs(WORKSPACE_ROOT, exist_ok=True)

    workspace = os.path.abspath(
        os.path.join(WORKSPACE_ROOT, instance_id)
    )

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