# -*- coding: utf-8 -*-
"""
CodeDoctor Web UI — FastAPI 服务层。

本文件只是 Agent Runtime 的外壳：
- 不复制 Agent 逻辑，通过子进程调用真实的 agent.py；
- 解析 agent.py 的 stdout 标记，转成 SSE 事件推给浏览器；
- 所有状态（edited / syntax / tests / validation…）都来自
  agent.py 基于真实工具返回值打印的标记，不由 LLM 文本决定。
"""
import atexit
import asyncio
import io
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request as UrlRequest, urlopen

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
TASKS_FILE = ROOT / "tasks.jsonl"
WORKSPACE_ROOT = ROOT / "workspaces"

# 项目根目录加入 sys.path，复用 workspace_manager 的 workspace 清理逻辑
sys.path.insert(0, str(ROOT))
from workspace_manager import (  # noqa: E402
    WORKSPACE_TTL_HOURS,
    cleanup_stale_workspaces,
    touch_workspace,
)

MAX_UPLOAD_BYTES = 200 * 1024 * 1024

# 下载 ZIP 时排除的目录 / 文件
EXCLUDE_DIRS = {
    ".git", ".venv", "__pycache__", ".pytest_cache", ".tox",
    ".agent_validation", ".mypy_cache", ".ruff_cache", ".eggs",
}
EXCLUDE_FILE_PREFIXES = (".agent_session",)
SESSION_FILE_NAME = ".agent_session.json"

# ---------- agent.py stdout 标记（与已验收的 Runtime 输出严格对应） ----------
ROUND_RE = re.compile(r"^=+\s*第\s*(\d+)\s*轮\s*=+$")
TOOL_LINE_RE = re.compile(r"^调用工具：(\S+)\s*$")
ARGS_LINE_RE = re.compile(r"^参数：(.*)$")
PASSED_RE = re.compile(r"(\d+)\s+passed")
FAILED_RE = re.compile(r"(\d+)\s+failed")

FLAG_MARKS = [
    ("✅ 已确认实际修改代码", "edited"),
    ("✅ 语法检查通过", "syntax_passed"),
    ("✅ 测试通过", "tests_passed"),
    ("✅ 已确认原始 Bug 可以复现", "baseline_confirmed"),
    ("✅ 通用 Bug 回归验证通过", "validation_passed"),
    ("🧪 已创建通用 Bug 回归验证", "validation_created"),
]

PHASES = ["LOCALIZE", "VALIDATE", "REPAIR", "TEST", "FINAL_VALIDATE", "DONE"]


def phase_of(flags):
    """与 agent.py current_phase() 相同的推导（仅用于展示）。"""
    if all(flags.values()):
        return "DONE"
    if (flags["edited"] and flags["syntax_passed"] and flags["tests_passed"]
            and flags["validation_created"] and flags["baseline_confirmed"]):
        return "FINAL_VALIDATE"
    if flags["edited"] and flags["syntax_passed"]:
        return "TEST"
    if flags["edited"]:
        return "REPAIR"
    if flags["validation_created"] and flags["baseline_confirmed"]:
        return "REPAIR"
    if flags["validation_created"]:
        return "VALIDATE"
    return "LOCALIZE"


# ============================================================
# RunState / 事件总线
# ============================================================

class RunState:
    def __init__(self, task_id, source, workspace, problem, repo=None, label=None):
        self.task_id = task_id
        self.source = source            # upload | issue | preset
        self.workspace = workspace      # Path（issue 任务在首次运行前可能不存在）
        self.problem = problem
        self.repo = repo
        self.label = label or repo or task_id
        self.status = "created"         # created | running | done | failed
        self.created = datetime.now()
        self.stopped = False
        self.events = []                # SSE 事件回放缓冲
        self.subscribers = []           # [(event_loop, asyncio.Queue)]
        self.proc = None
        self.lock = threading.Lock()
        self.flags = {
            "edited": False,
            "syntax_passed": False,
            "tests_passed": False,
            "validation_created": False,
            "baseline_confirmed": False,
            "validation_passed": False,
        }
        self.round = 0
        self.success_marker = False
        self.max_rounds = False
        self.llm_failed = False
        self.result = None

    def snapshot(self):
        with self.lock:
            return {
                "task_id": self.task_id,
                "source": self.source,
                "repo": self.repo,
                "label": self.label,
                "problem": self.problem[:200],
                "status": self.status,
                "round": self.round,
                "flags": dict(self.flags),
                "phase": phase_of(self.flags),
                "success": self.result["success"] if self.result else None,
                "has_result": self.result is not None,
            }


RUNS = {}


def emit(run, event):
    event.setdefault("ts", datetime.now().strftime("%H:%M:%S"))
    with run.lock:
        run.events.append(event)
        for loop, q in list(run.subscribers):
            try:
                loop.call_soon_threadsafe(q.put_nowait, event)
            except RuntimeError:
                pass


def set_flag(run, key):
    with run.lock:
        if not run.flags[key]:
            run.flags[key] = True
            changed = True
        else:
            changed = False
    if changed:
        emit(run, {
            "type": "status",
            "flags": dict(run.flags),
            "phase": phase_of(run.flags),
        })


# ============================================================
# git 辅助
# ============================================================

def git_run(ws, *args):
    return subprocess.run(
        ["git", "--no-optional-locks", *args],
        cwd=str(ws),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def git_init_commit(ws):
    git_run(ws, "init")
    git_run(ws, "add", "-A")
    git_run(
        ws, "-c", "user.name=CodeDoctor", "-c", "user.email=codedoctor@local",
        "commit", "-m", "initial state (uploaded)",
    )


def reset_workspace(run):
    """把 git 仓库恢复到初始 commit（保留 .venv 与依赖标记）。"""
    ws = run.workspace
    if not (ws and ws.is_dir() and (ws / ".git").exists()):
        return {"reset": False, "message": "workspace 尚未创建，跳过重置"}
    touch_workspace(str(ws))  # 刷新"最后使用时间"，任务运行期间不会被清理
    git_run(ws, "reset", "--hard", "HEAD")
    git_run(ws, "clean", "-fd", "-e", ".venv", "-e", ".code_agent_deps_ready.json")
    # 断点会话一并清除（clean 通常已移除，这里显式兜底）
    session_file = ws / SESSION_FILE_NAME
    if session_file.is_file():
        try:
            session_file.unlink()
        except OSError:
            pass
    return {"reset": True, "message": "已恢复到初始状态（保留虚拟环境）"}


# ============================================================
# agent 子进程：stdout 解析器
# ============================================================

def read_agent_output(run, proc):
    mode = None            # None | text | result | rag
    text_buf = []
    rag_buf = []
    result_buf = []
    last_tool = None
    pending_tool = None

    def flush_text():
        nonlocal text_buf
        text = "\n".join(text_buf).strip()
        text_buf = []
        if text:
            emit(run, {"type": "agent_text", "text": text[:2000]})

    def flush_rag():
        nonlocal rag_buf
        content = "\n".join(rag_buf).strip()
        rag_buf = []
        if content:
            emit(run, {"type": "rag", "content": content[:6000]})

    def flush_result():
        # 工具结果摘要：进入日志与工作流步骤的 ↳ 行
        nonlocal result_buf
        lines = result_buf
        result_buf = []
        if last_tool is None or not lines:
            return
        nonempty = [l for l in lines if l.strip()]
        if not nonempty:
            return
        emit(run, {
            "type": "tool_result",
            "name": last_tool,
            "summary": "\n".join(nonempty[:6])[:400],
            "lines": len(nonempty),
        })

    try:
        for raw in proc.stdout:
            line = raw.rstrip("\r\n")

            round_m = ROUND_RE.match(line.strip())
            if round_m:
                flush_text()
                flush_rag()
                flush_result()
                mode = None
                n = int(round_m.group(1))
                run.round = n
                emit(run, {"type": "round", "n": n})
                continue

            if line.startswith("========== 修复完成 =========="):
                flush_text()
                flush_rag()
                flush_result()
                run.success_marker = True
                continue

            if line.startswith("========== 达到最大轮数 =========="):
                flush_text()
                flush_rag()
                flush_result()
                run.max_rounds = True
                continue

            if line.startswith("========== LLM 调用失败 =========="):
                flush_result()
                run.llm_failed = True
                continue

            if line.startswith("Agent："):
                flush_rag()
                flush_result()
                mode = "text"
                text_buf = []
                continue

            m = TOOL_LINE_RE.match(line)
            if m:
                flush_text()
                flush_rag()
                flush_result()
                pending_tool = m.group(1)
                mode = None
                continue

            if line.startswith("参数：") and pending_tool:
                last_tool = pending_tool
                args_str = line[len("参数："):]
                emit(run, {"type": "tool", "name": last_tool, "args": args_str[:500]})
                pending_tool = None
                mode = None
                continue

            if line.startswith("工具结果"):
                flush_text()
                flush_result()
                if last_tool == "retrieve_code":
                    mode = "rag"
                    rag_buf = []
                else:
                    mode = "result"
                continue

            if line.startswith("修改文件："):
                flush_result()
                emit(run, {"type": "edited_file", "file": line[len("修改文件："):].strip()})
                continue

            # ---- 状态标记（真实 Runtime 输出） ----
            matched = False
            for mark, key in FLAG_MARKS:
                if line.startswith(mark):
                    flush_result()
                    set_flag(run, key)
                    matched = True
                    break
            if matched:
                continue

            # ---- 分模式处理 ----
            if mode == "text":
                text_buf.append(line)
                continue

            if mode == "rag":
                rag_buf.append(line)
                if len(rag_buf) > 120:
                    flush_rag()
                    mode = "result"
                continue

            if mode == "result":
                result_buf.append(line)
                # 统计 pytest 结果（真实测试输出）
                if last_tool in ("run_test", "run_bug_validation"):
                    pm = PASSED_RE.search(line)
                    fm = FAILED_RE.search(line)
                    if pm or fm:
                        emit(run, {
                            "type": "test_counts",
                            "source": last_tool,
                            "passed": int(pm.group(1)) if pm else 0,
                            "failed": int(fm.group(1)) if fm else 0,
                        })
                continue

            # 普通日志行（克隆 / venv / pip 等真实过程）
            if line.strip():
                emit(run, {"type": "log", "line": line[:500]})

        flush_text()
        flush_rag()
        flush_result()
    finally:
        proc.wait()

    # ---- 进程结束，汇总真实结果 ----
    success = run.success_marker or all(run.flags.values())
    if run.llm_failed or (proc.returncode != 0):
        run.status = "failed"
    else:
        run.status = "done"

    run.result = build_result(run, success)
    emit(run, {"type": "done", "success": success, "result": run.result})


def build_result(run, success):
    result = {
        "success": bool(success),
        "flags": dict(run.flags),
        "round": run.round,
        "max_rounds": run.max_rounds,
        "llm_failed": run.llm_failed,
        "stopped": run.stopped,
        "changed_files": [],
        "diff": "",
    }
    ws = run.workspace
    if ws and (ws / ".git").exists():
        diff = git_run(ws, "diff")
        names = git_run(ws, "diff", "--name-only")
        result["diff"] = (diff.stdout or "")[:60000]
        result["changed_files"] = [
            l.strip() for l in (names.stdout or "").splitlines() if l.strip()
        ]
    # 中断后真实的续跑轮次（来自会话文件，而非最后打印的轮次）
    session = read_session(run.workspace)
    if session is not None:
        result["resume_round"] = session["round"]
    return result


# 会话文件 state 字段 → 展示状态位的映射
SESSION_FLAG_MAP = {
    "edited": "edited",
    "syntax_passed": "syntax_passed",
    "tests_passed": "tests_passed",
    "validation_created": "validation_created",
    "validation_baseline_confirmed": "baseline_confirmed",
    "validation_passed": "validation_passed",
}


def read_session(ws):
    """读取断点会话（Runtime 每轮原子保存的真实状态）。"""
    if not ws:
        return None
    session_file = ws / SESSION_FILE_NAME
    if not session_file.is_file():
        return None
    try:
        data = json.loads(session_file.read_text(encoding="utf-8"))
        state = data.get("state", {})
        return {
            "round": int(data.get("round_index", 0)),
            "flags": {rk: bool(state.get(sk)) for sk, rk in SESSION_FLAG_MAP.items()},
        }
    except (OSError, ValueError, TypeError):
        return None


def start_agent(run, reset=False):
    ws = run.workspace
    session = read_session(ws)

    if reset:
        # 用户明确要求从头开始
        reset_workspace(run)
        session = None
    elif session is None and ws and (ws / ".git").exists():
        # 无断点会话：恢复干净的初始状态再开局，
        # 避免带着上一次的半成品修改让 Agent 误判 baseline
        reset_workspace(run)

    # ---- 新一次尝试：清空上次的展示状态 ----
    with run.lock:
        run.events = []
        run.round = 0
        run.success_marker = False
        run.max_rounds = False
        run.llm_failed = False
        run.stopped = False
        run.result = None
        for k in run.flags:
            run.flags[k] = False

    # 断点续跑：用会话文件里的真实状态位初始化展示
    #（agent.py 恢复时不会重打 ✅ 标记，必须在这里播种）
    if session is not None:
        with run.lock:
            run.round = session["round"]
            for k, v in session["flags"].items():
                run.flags[k] = v
        emit(run, {
            "type": "resume",
            "from_round": session["round"],
            "flags": dict(run.flags),
            "phase": phase_of(run.flags),
        })

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "agent.py")],
        cwd=str(ROOT),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        bufsize=1,
    )

    # 第一行 = 任务 ID；若存在未完成会话，第二行空回车 = 从断点继续
    try:
        proc.stdin.write(f"{run.task_id}\n\n")
        proc.stdin.flush()
        proc.stdin.close()
    except OSError:
        pass

    run.proc = proc
    run.status = "running"

    t = threading.Thread(target=read_agent_output, args=(run, proc), daemon=True)
    t.start()


def _kill_all_agents():
    for run in RUNS.values():
        if run.proc and run.proc.poll() is None:
            try:
                run.proc.terminate()
            except OSError:
                pass


atexit.register(_kill_all_agents)


# ============================================================
# 任务登记（tasks.jsonl 与 agent.py 共用）
# ============================================================

def append_task(instance_id, repo, base_commit, problem_statement):
    with open(TASKS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "instance_id": instance_id,
            "repo": repo,
            "base_commit": base_commit,
            "problem_statement": problem_statement,
        }, ensure_ascii=False) + "\n")


def load_task_entries():
    entries = []
    if not TASKS_FILE.exists():
        return entries
    with open(TASKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
                entries.append(t)
            except ValueError:
                continue
    return entries


def task_source(instance_id):
    if instance_id.startswith("upload_"):
        return "upload"
    if instance_id.startswith("issue_"):
        return "issue"
    return "preset"


def new_task_id(prefix):
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(2)}"


# ============================================================
# 工具：目录树 / ZIP
# ============================================================

def build_tree(ws, max_lines=24):
    if not (ws and ws.is_dir()):
        return "（Workspace 尚未创建）"
    try:
        entries = sorted(
            ws.iterdir(),
            key=lambda p: (not p.is_dir(), p.name.lower()),
        )
    except OSError:
        return "（无法读取）"
    visible = [
        p for p in entries
        if p.name not in EXCLUDE_DIRS
        and not p.name.startswith(EXCLUDE_FILE_PREFIXES)
        and p.name != ".code_agent_deps_ready.json"
    ]
    shown = visible[:max_lines]
    lines = []
    for i, p in enumerate(shown):
        is_last = i == len(shown) - 1 and len(shown) == len(visible)
        prefix = "└─" if is_last else "├─"
        lines.append(f"{prefix} {p.name}{'' if p.is_file() else '/'}")
    if len(visible) > max_lines:
        lines.append(f"…（共 {len(visible)} 项）")
    if not lines:
        return "（空）"
    return "\n".join(lines)


def zip_workspace(ws):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for dirpath, dirnames, filenames in os.walk(ws):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
            for fn in filenames:
                if fn.startswith(EXCLUDE_FILE_PREFIXES) or fn == ".code_agent_deps_ready.json":
                    continue
                full = Path(dirpath) / fn
                rel = full.relative_to(ws)
                try:
                    zf.write(full, str(rel))
                except OSError:
                    continue
    buf.seek(0)
    return buf


def safe_extract_zip(data, dest):
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = []
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename
            # 非 UTF-8 标记的中文文件名按 GBK 还原
            if not (info.flag_bits & 0x800):
                try:
                    name = info.filename.encode("cp437").decode("gbk")
                except (UnicodeDecodeError, UnicodeEncodeError):
                    name = info.filename
            names.append((info, name))

        dest_resolved = dest.resolve()
        for info, name in names:
            target = (dest / name).resolve()
            if dest_resolved != target and dest_resolved not in target.parents:
                continue  # zip-slip 防护
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)

    # 整包只有一个顶层目录时上提一层
    entries = list(dest.iterdir())
    if len(entries) == 1 and entries[0].is_dir() and not entries[0].name.startswith("."):
        inner = entries[0]
        for item in list(inner.iterdir()):
            shutil.move(str(item), str(dest / item.name))
        inner.rmdir()


# ============================================================
# GitHub Issue 抓取
# ============================================================

ISSUE_URL_RE = re.compile(r"github\.com/([\w.\-]+)/([\w.\-]+)/issues/(\d+)")


def gh_get(url):
    req = UrlRequest(url, headers={
        "User-Agent": "CodeDoctor-Agent-UI",
        "Accept": "application/vnd.github+json",
    })
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_issue(url):
    m = ISSUE_URL_RE.search(url or "")
    if not m:
        raise HTTPException(400, "无法解析 Issue 链接，需要形如 https://github.com/owner/repo/issues/123")
    owner, repo, number = m.group(1), m.group(2), m.group(3)
    try:
        issue = gh_get(f"https://api.github.com/repos/{owner}/{repo}/issues/{number}")
    except HTTPError as e:
        if e.code == 404:
            raise HTTPException(404, f"Issue 不存在：{owner}/{repo}#{number}")
        if e.code == 403:
            raise HTTPException(403, "GitHub API 限流（未认证每小时 60 次），请稍后再试")
        raise HTTPException(502, f"GitHub API 错误：{e.code}")
    except (URLError, TimeoutError) as e:
        raise HTTPException(502, f"无法访问 GitHub API：{e}")

    parts = [f"[GitHub Issue #{number}] {issue.get('title', '').strip()}"]
    body = (issue.get("body") or "").strip()
    if body:
        parts.append(body)
    try:
        comments = gh_get(f"https://api.github.com/repos/{owner}/{repo}/issues/{number}/comments")
        for c in comments[:10]:
            cb = (c.get("body") or "").strip()
            if cb:
                parts.append(f"---\n评论（{c.get('user', {}).get('login', '?')}）：\n{cb[:1500]}")
    except Exception:
        pass  # 评论抓取失败不阻断
    return owner, repo, number, "\n\n".join(parts), issue.get("title", "")


# ============================================================
# FastAPI 路由
# ============================================================

app = FastAPI(title="CodeDoctor")


@app.get("/api/tasks")
def api_tasks():
    out = []
    seen = set()
    for t in load_task_entries():
        tid = t.get("instance_id", "")
        if not tid or tid in seen:
            continue
        seen.add(tid)
        src = task_source(tid)
        run = RUNS.get(tid)
        out.append({
            "instance_id": tid,
            "repo": t.get("repo"),
            "source": src,
            "status": run.status if run else None,
            "problem": (t.get("problem_statement") or "")[:300],
        })
    return {"tasks": out}


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...), description: str = Form(...)):
    if not description.strip():
        raise HTTPException(400, "Bug 描述不能为空")

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "文件过大（上限 200MB）")
    if not data:
        raise HTTPException(400, "上传文件为空")

    filename = os.path.basename(file.filename or "upload.bin")
    task_id = new_task_id("upload")
    ws = WORKSPACE_ROOT / task_id
    ws.mkdir(parents=True)

    try:
        if filename.lower().endswith(".zip"):
            try:
                safe_extract_zip(data, ws)
            except zipfile.BadZipFile:
                raise HTTPException(400, "不是有效的 ZIP 文件")
            if not any(ws.iterdir()):
                raise HTTPException(400, "ZIP 解压后为空")
        else:
            (ws / filename).write_bytes(data)
        git_init_commit(ws)
    except HTTPException:
        shutil.rmtree(ws, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(ws, ignore_errors=True)
        raise HTTPException(500, f"处理上传失败：{e}")

    append_task(task_id, "local/upload", "HEAD", description.strip())
    run = RunState(
        task_id, "upload", ws, description.strip(),
        repo="local/upload", label=filename,
    )
    RUNS[task_id] = run

    return {
        "task_id": task_id,
        "filename": filename,
        "is_zip": filename.lower().endswith(".zip"),
        "tree": build_tree(ws),
    }


@app.post("/api/issue")
async def api_issue(payload: dict):
    url = (payload or {}).get("url", "")
    note = (payload or {}).get("note", "").strip()
    owner, repo, number, problem, title = fetch_issue(url)
    if note:
        problem += f"\n\n---\n用户补充：\n{note}"

    task_id = new_task_id(f"issue_{owner}_{repo}_{number}")
    append_task(task_id, f"{owner}/{repo}", "HEAD", problem)
    ws = WORKSPACE_ROOT / task_id
    run = RunState(
        task_id, "issue", ws, problem,
        repo=f"{owner}/{repo}", label=f"{owner}/{repo}#{number}",
    )
    RUNS[task_id] = run

    return {
        "task_id": task_id,
        "title": title,
        "problem": problem,
        "note": "仓库将在开始修复时自动克隆（默认分支）",
    }


@app.post("/api/repair")
def api_repair(payload: dict):
    task_id = (payload or {}).get("task_id", "")
    reset = bool((payload or {}).get("reset", False))

    run = RUNS.get(task_id)
    if run is None:
        entry = None
        for t in load_task_entries():
            if t.get("instance_id") == task_id:
                entry = t
                break
        if entry is None:
            raise HTTPException(404, f"任务不存在：{task_id}")
        run = RunState(
            task_id, task_source(task_id),
            WORKSPACE_ROOT / task_id,
            entry.get("problem_statement", ""),
            repo=entry.get("repo"),
            label=entry.get("repo") or task_id,
        )
        RUNS[task_id] = run

    if run.status == "running":
        raise HTTPException(409, "该任务正在运行中")

    # 重置 / 断点续跑由服务端按会话文件自动决定：
    # - 有 .agent_session.json → 断点续跑（LLM 失败、手动终止后不丢轮数）
    # - 无会话（首次 / 已完成 / 达到最大轮数）→ 干净开局
    # - reset=True 为用户明确要求从头开始
    start_agent(run, reset=reset)
    return {"task_id": task_id, "started": True, "reset": reset}


@app.get("/api/status/{task_id}")
def api_status(task_id: str):
    run = RUNS.get(task_id)
    if run is None:
        raise HTTPException(404, f"任务不存在：{task_id}")
    snap = run.snapshot()
    if run.result:
        snap["result"] = run.result
    return snap


@app.get("/api/active")
def api_active():
    """页面刷新 / 重开后恢复现场用：列出有运行记录的任务。"""
    runs = sorted(
        RUNS.values(),
        key=lambda r: (r.status != "running", -r.created.timestamp()),
    )
    out = []
    for run in runs:
        if run.status == "created":
            continue
        snap = run.snapshot()
        if run.result:
            snap["result"] = run.result
        out.append(snap)
    return {"tasks": out}


@app.post("/api/stop")
def api_stop(payload: dict):
    task_id = (payload or {}).get("task_id", "")
    run = RUNS.get(task_id)
    if run is None:
        raise HTTPException(404, f"任务不存在：{task_id}")
    if run.status != "running" or not run.proc or run.proc.poll() is not None:
        raise HTTPException(409, "任务不在运行中")
    run.stopped = True
    try:
        run.proc.terminate()
    except OSError:
        pass
    return {"task_id": task_id, "stopped": True}


@app.get("/api/stream/{task_id}")
async def api_stream(task_id: str):
    run = RUNS.get(task_id)
    if run is None:
        raise HTTPException(404, f"任务不存在：{task_id}")

    def sse(ev):
        return f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

    async def gen():
        q = asyncio.Queue()
        loop = asyncio.get_running_loop()
        with run.lock:
            idx = len(run.events)
            run.subscribers.append((loop, q))
        try:
            for ev in run.events[:idx]:
                yield sse(ev)
            if run.status in ("done", "failed"):
                return
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15)
                except asyncio.TimeoutError:
                    if run.status in ("done", "failed"):
                        return
                    yield ": ping\n\n"
                    continue
                yield sse(ev)
                if ev.get("type") == "done":
                    return
        finally:
            with run.lock:
                run.subscribers = [(l, qq) for (l, qq) in run.subscribers if qq is not q]

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/diff/{task_id}")
def api_diff(task_id: str):
    run = RUNS.get(task_id)
    if run is None:
        raise HTTPException(404, f"任务不存在：{task_id}")
    ws = run.workspace
    if not (ws and (ws / ".git").exists()):
        return {"diff": "", "changed_files": []}
    diff = git_run(ws, "diff")
    names = git_run(ws, "diff", "--name-only")
    return {
        "diff": (diff.stdout or "")[:60000],
        "changed_files": [l.strip() for l in (names.stdout or "").splitlines() if l.strip()],
    }


@app.get("/api/download/{task_id}")
def api_download(task_id: str):
    run = RUNS.get(task_id)
    if run is None:
        raise HTTPException(404, f"任务不存在：{task_id}")
    ws = run.workspace
    if not (ws and ws.is_dir()):
        raise HTTPException(404, "Workspace 不存在")
    if not any(
        p for p in ws.iterdir()
        if p.name not in EXCLUDE_DIRS and not p.name.startswith(EXCLUDE_FILE_PREFIXES)
    ):
        raise HTTPException(404, "Workspace 为空")
    buf = zip_workspace(ws)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{task_id}_fixed.zip"'},
    )


@app.get("/api/workspace/{task_id}")
def api_workspace(task_id: str):
    run = RUNS.get(task_id)
    if run is None:
        raise HTTPException(404, f"任务不存在：{task_id}")
    return {
        "task_id": task_id,
        "workspace": str(run.workspace),
        "exists": run.workspace.is_dir(),
        "tree": build_tree(run.workspace),
    }


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


# ============================================================
# workspace 自动清理（默认保留 WORKSPACE_TTL_HOURS = 24 小时）
# ============================================================

WORKSPACE_CLEAN_INTERVAL = 3600  # 巡检间隔（秒）


def _workspace_cleaner_loop():
    """后台线程：定期删除超过 TTL 未使用的旧 workspace。"""
    while True:
        try:
            # 运行中任务的 workspace 本轮保留（双保险：
            # 正常任务在 reset/创建时已刷新过 mtime）
            protected = {
                r.workspace.name for r in list(RUNS.values())
                if r.status == "running"
            }
            removed = cleanup_stale_workspaces(
                workspace_root=str(WORKSPACE_ROOT),
                protected_ids=protected,
            )
            if removed:
                names = ", ".join(removed[:5])
                if len(removed) > 5:
                    names += " 等"
                print(f"[workspace 清理] 删除 {len(removed)} 个超过 "
                      f"{WORKSPACE_TTL_HOURS} 小时未使用的旧 workspace：{names}")
        except Exception as e:
            print(f"[workspace 清理] 失败（不影响服务）：{e}")
        time.sleep(WORKSPACE_CLEAN_INTERVAL)


def start_workspace_cleaner():
    threading.Thread(
        target=_workspace_cleaner_loop, daemon=True, name="ws-cleaner"
    ).start()


if __name__ == "__main__":
    start_workspace_cleaner()
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
