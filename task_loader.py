import json


def load_tasks(path="tasks.jsonl"):
    tasks = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            task = json.loads(line)

            tasks.append({
                "instance_id": task["instance_id"],
                "repo": task["repo"],
                "base_commit": task["base_commit"],
                "problem_statement": task["problem_statement"],
            })

    return tasks


def get_task(instance_id, path="tasks.jsonl"):
    tasks = load_tasks(path)

    for task in tasks:
        if task["instance_id"] == instance_id:
            return task

    raise ValueError(f"没有找到任务：{instance_id}")


if __name__ == "__main__":
    tasks = load_tasks()

    print(f"任务数量：{len(tasks)}")

    task = get_task("fastapi_15661")

    print("\n第一个任务：")
    print(f"instance_id: {task['instance_id']}")
    print(f"repo: {task['repo']}")
    print(f"base_commit: {task['base_commit']}")
    print("\nproblem_statement:")
    print(task["problem_statement"])