import subprocess

def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def edit_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    return f"文件 {path} 修改成功"


def run_test(path):
    result = subprocess.run(
        ["python", "-m", "pytest", path],
        capture_output=True,
        text=True
    )

    return result.stdout + result.stderr


tools_map = {
    "read_file": read_file,
    "edit_file": edit_file,
    "run_test": run_test
}
if __name__ == "__main__":
    print(run_test())