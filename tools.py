import subprocess
import os
from code_rag import build_code_rag,retrieve_code,format_results



def read_file(path):
    path=os.path.abspath(path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def edit_file(path, content):
    path=os.path.abspath(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    return f"文件 {path} 修改成功"


def run_test():
    result = subprocess.run(
        ["python", "-m", "pytest"],
        capture_output=True,
        text=True
    )

    output = result.stdout + result.stderr

    passed = 0
    failed = 0
    errors = 0
    skipped = 0

    import re

    match = re.search(r"(\d+) passed", output)
    if match:
        passed = int(match.group(1))

    match = re.search(r"(\d+) failed", output)
    if match:
        failed = int(match.group(1))

    match = re.search(r"(\d+) error", output)
    if match:
        errors = int(match.group(1))

    match = re.search(r"(\d+) skipped", output)
    if match:
        skipped = int(match.group(1))

    if result.returncode == 0 and passed > 0:
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
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "returncode": result.returncode,
        "output": output
    }

def list_files(path="."):
    files=[]
    for root, dirs, filenames in os.walk(path):
        for filename in filenames:
            if "_pycache__" not in root:
                files.append(os.path.join(root, filename))
    return "\n".join(files)

def retrieve_code_tool(query):
    index, all_chunks = build_code_rag(".")
    results = retrieve_code(
        query,
        index,
        all_chunks,
        top_k=5
    )
    return format_results(results)

tools_map = {
    "read_file": read_file,
    "edit_file": edit_file,
    "run_test": run_test,
    "list_files": list_files,
    "retrieve_code": retrieve_code_tool
}
