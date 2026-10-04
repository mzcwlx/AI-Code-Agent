const $ = (id) => document.getElementById(id);

const PHASES = ["LOCALIZE", "VALIDATE", "REPAIR", "TEST", "FINAL_VALIDATE", "DONE"];

const TOOL_ICONS = {
  list_files: "▣",
  retrieve_code: "⌕",
  read_file: "▤",
  replace_lines: "✎",
  check_syntax: "✓",
  run_test: "⚗",
  create_bug_validation: "✚",
  run_bug_validation: "✔",
};

let currentTask = null;
let es = null;
let rawLogLines = [];
let lastToolRow = null;
let presetTasks = [];

function esc(s) {
  const d = document.createElement("div");
  d.textContent = s == null ? "" : String(s);
  return d.innerHTML;
}

function shortArgs(argsStr) {
  if (!argsStr) return "";
  try {
    const o = JSON.parse(argsStr);
    if (typeof o === "object" && o !== null) {
      const keys = ["path", "query", "test_path", "test_filter"];
      for (const k of keys) {
        if (o[k]) return String(o[k]);
      }
    }
    return argsStr;
  } catch (e) {
    return argsStr;
  }
}

function statusText(s) {
  return { running: "运行中", done: "已完成", failed: "已结束" }[s] || s || "";
}

/* ---------------- 结构化日志 ---------------- */

function logLine(ev) {
  const ts = ev.ts || "--:--:--";
  switch (ev.type) {
    case "round": return `[${ts}] ────── 第 ${ev.n} 轮 ──────`;
    case "agent_text":
      return ev.text.split("\n").map((l) => `[${ts}] AGENT │ ${l}`).join("\n");
    case "tool": return `[${ts}] TOOL  ${ev.name} ${ev.args || ""}`;
    case "tool_result": {
      const first = (ev.summary || "").split("\n")[0] || "";
      return `[${ts}] ←     ${ev.name}（${ev.lines} 行） ${first.slice(0, 160)}`;
    }
    case "rag": return `[${ts}] RAG   检索完成（${(ev.content || "").length} 字符）`;
    case "log": return `[${ts}] SYS   ${ev.line}`;
    case "status": return `[${ts}] STATE 阶段 → ${ev.phase}`;
    case "test_counts":
      return `[${ts}] TEST  ${ev.source}: ${ev.passed} passed / ${ev.failed} failed`;
    case "edited_file": return `[${ts}] EDIT  ${ev.file}`;
    case "resume":
      return `[${ts}] RESUME 断点恢复：已保留 ${ev.from_round} 轮进度，从第 ${ev.from_round + 1} 轮继续`;
    case "done":
      return `[${ts}] DONE  ${ev.success ? "修复成功" : "修复未完成"}`;
    default: return null;
  }
}

function appendLog(ev) {
  const line = logLine(ev);
  if (line === null) return;
  rawLogLines.push(line);
  const box = $("rawLog");
  box.textContent = rawLogLines.join("\n");
  box.scrollTop = box.scrollHeight;
}

function downloadLog() {
  if (!currentTask || !rawLogLines.length) return;
  const header = [
    "CodeDoctor 运行日志",
    `任务: ${currentTask.task_id}`,
    `仓库: ${currentTask.repo || "-"}`,
    `导出时间: ${new Date().toLocaleString()}`,
    "────────────────────────────────────────",
    "",
  ].join("\n");
  const blob = new Blob(["\ufeff" + header + rawLogLines.join("\n")], {
    type: "text/plain;charset=utf-8",
  });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${currentTask.task_id}.log`;
  a.click();
  URL.revokeObjectURL(a.href);
}

/* ---------------- 阶段 chips / 状态位 ---------------- */

function renderPhases(current) {
  const box = $("phases");
  box.innerHTML = "";
  const idx = PHASES.indexOf(current);
  PHASES.forEach((p, i) => {
    const chip = document.createElement("span");
    chip.className = "phase" + (i < idx ? " done" : i === idx ? " current" : "");
    chip.textContent = p;
    box.appendChild(chip);
  });
}

function applyFlags(flags, phase) {
  renderPhases(phase);
  if (!flags) return;
  if (flags.baseline_confirmed) {
    $("baseline").textContent = "FAIL";
    $("baseline").style.color = "var(--yellow)";
  }
  if (flags.validation_passed) {
    $("validation").textContent = "PASS";
    $("validation").style.color = "var(--green)";
  }
  if (flags.tests_passed) {
    $("testStatus").textContent = "TESTS PASS";
    $("testStatus").style.color = "var(--green)";
  }
}

/* ---------------- 任务准备 ---------------- */

function activeTab() {
  return document.querySelector(".tab.active").dataset.tab;
}

function setTask(task) {
  currentTask = task;
  localStorage.setItem("codedoctor_task", task.task_id);
  $("taskId").textContent = task.task_id;
  $("heroTitle").textContent = task.title || (task.problem || "").slice(0, 110);
  $("heroSub").textContent =
    `任务 ${task.task_id} · 来源 ${task.source}` +
    (task.filename ? ` · ${task.filename}` : "");
  $("wsProject").textContent = task.repo || task.filename || task.task_id;
  $("startBtn").disabled = false;
  $("prepareMsg").textContent = "";
}

async function refreshTree(taskId) {
  try {
    const res = await fetch(`/api/workspace/${taskId}`);
    if (res.ok) {
      const data = await res.json();
      $("wsTree").textContent = data.tree;
    }
  } catch (e) { /* 忽略 */ }
}

async function prepareTask() {
  const tab = activeTab();
  $("prepareMsg").textContent = "";

  if (tab === "upload") {
    const file = $("fileInput").files[0];
    const desc = $("bugDesc").value.trim();
    if (!file) { $("prepareMsg").textContent = "请先选择 ZIP 或源文件"; return; }
    if (!desc) { $("prepareMsg").textContent = "请填写 Bug 描述"; return; }
    const fd = new FormData();
    fd.append("file", file);
    fd.append("description", desc);
    const res = await fetch("/api/upload", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) { $("prepareMsg").textContent = data.detail || "上传失败"; return; }
    setTask({
      task_id: data.task_id,
      source: "upload",
      problem: desc,
      title: desc.split("\n")[0],
      filename: data.filename,
      repo: data.is_zip ? "uploaded zip" : data.filename,
    });
    $("wsTree").textContent = data.tree;
    $("prepareMsg").textContent = "任务已就绪，可以开始修复";

  } else if (tab === "issue") {
    const url = $("issueUrl").value.trim();
    if (!url) { $("prepareMsg").textContent = "请填写 Issue 链接"; return; }
    const res = await fetch("/api/issue", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, note: $("issueNote").value.trim() }),
    });
    const data = await res.json();
    if (!res.ok) { $("prepareMsg").textContent = data.detail || "创建失败"; return; }
    setTask({
      task_id: data.task_id,
      source: "issue",
      problem: data.problem,
      title: data.title || url,
      repo: "github (首次运行时克隆)",
    });
    $("wsTree").textContent = "（仓库将在开始修复时自动克隆）";
    $("prepareMsg").textContent = "任务已就绪，可以开始修复";

  } else if (tab === "preset") {
    const tid = $("presetSelect").value;
    if (!tid) { $("prepareMsg").textContent = "请选择任务"; return; }
    const info = presetTasks.find((t) => t.instance_id === tid);
    if (!info) return;
    setTask({
      task_id: tid,
      source: "preset",
      problem: info.problem || "",
      title: `${info.repo} · ${tid}`,
      repo: info.repo,
    });
    await refreshTree(tid);
    $("prepareMsg").textContent = "任务已就绪，可以开始修复";
  }
}

async function loadPresets() {
  try {
    const res = await fetch("/api/tasks");
    const data = await res.json();
    presetTasks = (data.tasks || []).filter((t) => t.source === "preset");
    const sel = $("presetSelect");
    sel.innerHTML = "";
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "-- 选择预置任务 --";
    sel.appendChild(opt);
    presetTasks.forEach((t) => {
      const o = document.createElement("option");
      o.value = t.instance_id;
      const badge = t.status
        ? { running: " · 运行中", done: " · 已完成", failed: " · 已结束" }[t.status] || ""
        : "";
      o.textContent = `${t.instance_id}（${t.repo}）${badge}`;
      sel.appendChild(o);
    });
  } catch (e) { /* 忽略 */ }
}

/* ---------------- 运行 ---------------- */

function resetRunUI() {
  $("workflow").innerHTML = "";
  $("round").textContent = "ROUND —";
  $("ragCode").textContent = "尚无检索结果";
  $("ragCount").textContent = "—";
  $("diffFile").textContent = "Waiting...";
  $("diffBox").innerHTML = '<div class="empty">修复完成后显示真实 git diff</div>';
  $("passed").textContent = "0";
  $("failed").textContent = "0";
  $("validation").textContent = "—";
  $("validation").style.color = "";
  $("baseline").textContent = "—";
  $("baseline").style.color = "";
  $("testStatus").textContent = "RUNNING";
  $("testStatus").className = "";
  $("testStatus").style.color = "var(--yellow)";
  $("resultCard").hidden = true;
  $("downloadBtn").hidden = true;
  rawLogLines = [];
  $("rawLog").textContent = "";
  lastToolRow = null;
  renderPhases("LOCALIZE");
}

function setTopStatus(mode, text) {
  const el = $("agentStatus");
  el.className = "status" + (mode === "warn" ? " warn" : mode === "err" ? " err" : "");
  $("statusText").textContent = text;
}

function addStep(icon, message, argsStr, ok) {
  const wf = $("workflow");
  const row = document.createElement("div");
  row.className = "step" + (ok ? " ok" : "");
  row.innerHTML = `<div class="icon">${esc(icon)}</div>
    <div><div class="message">${esc(message)}</div>${
      argsStr ? `<div class="args">${esc(shortArgs(argsStr))}</div>` : ""
    }</div>`;
  wf.appendChild(row);
  wf.scrollTop = wf.scrollHeight;
  return row;
}

function addRoundDivider(n) {
  const wf = $("workflow");
  const div = document.createElement("div");
  div.className = "round-divider";
  div.textContent = `ROUND ${n}`;
  wf.appendChild(div);
}

function addAgentNote(text) {
  const wf = $("workflow");
  const note = document.createElement("div");
  note.className = "agent-note";
  note.textContent = text;
  wf.appendChild(note);
  wf.scrollTop = wf.scrollHeight;
}

function attachToolResult(ev) {
  if (!lastToolRow) return;
  const first = (ev.summary || "").split("\n")[0] || "";
  if (!first) return;
  const div = document.createElement("div");
  div.className = "result";
  div.textContent = `↳ ${first.slice(0, 140)}${first.length > 140 ? "…" : ""}`;
  lastToolRow.lastElementChild.appendChild(div);
  const wf = $("workflow");
  wf.scrollTop = wf.scrollHeight;
}

function renderDiff(diffText, fileLabel) {
  const box = $("diffBox");
  box.innerHTML = "";
  if (!diffText || !diffText.trim()) {
    box.innerHTML = '<div class="empty">没有检测到源码变化</div>';
    $("diffFile").textContent = fileLabel || "—";
    return;
  }
  const lines = diffText.split("\n");
  lines.forEach((l) => {
    const div = document.createElement("div");
    div.className = "dline " + (
      l.startsWith("+") ? "added" :
      l.startsWith("-") ? "removed" :
      l.startsWith("@@") ? "hunk" : "ctx"
    );
    div.textContent = l;
    box.appendChild(div);
  });
  $("diffFile").textContent = fileLabel || "changed";
}

function showResult(ev) {
  const r = ev.result || {};
  const shouldResume = !!(r.llm_failed || r.stopped);
  if (currentTask) currentTask.shouldResume = shouldResume;

  const card = $("resultCard");
  card.hidden = false;

  const badge = $("resultBadge");
  if (ev.success) {
    badge.textContent = "SUCCESS";
    badge.className = "pass";
    $("testStatus").textContent = "PASS";
    $("testStatus").style.color = "var(--green)";
    setTopStatus("", "AGENT ONLINE");
    $("downloadBtn").hidden = false;
    $("downloadBtn").href = `/api/download/${currentTask.task_id}`;
  } else {
    badge.textContent = "FAILED";
    badge.className = "fail";
    $("testStatus").textContent = "FAILED";
    $("testStatus").style.color = "var(--red)";
    setTopStatus("err", "AGENT ERROR");
    $("downloadBtn").hidden = true;
  }

  const flags = r.flags || {};
  const flagList = [
    ["实际修改代码", flags.edited],
    ["语法检查", flags.syntax_passed],
    ["项目测试", flags.tests_passed],
    ["回归测试已创建", flags.validation_created],
    ["Baseline 复现 (FAIL)", flags.baseline_confirmed],
    ["最终验证", flags.validation_passed],
  ]
    .map(([k, v]) => `<li>${esc(k)}：${v ? "PASS" : "—"}</li>`)
    .join("");

  const notes = [];
  const resumeFrom = (r.resume_round != null ? r.resume_round : (r.round || 0)) + 1;
  if (r.stopped) {
    notes.push(`<li style="color:var(--yellow)">已手动终止 — 点击「从断点继续」将从第 ${resumeFrom} 轮恢复，不丢进度</li>`);
  } else if (r.llm_failed) {
    notes.push(`<li style="color:var(--yellow)">LLM 调用失败中断 — 点击「从断点继续」将从第 ${resumeFrom} 轮恢复，不丢进度</li>`);
  } else if (r.max_rounds && !ev.success) {
    notes.push('<li style="color:var(--red)">达到最大轮数仍未完成修复，重跑将从干净状态重新开始</li>');
  }

  const files = (r.changed_files || []).map((f) => esc(f)).join("、") || "无";
  $("resultText").innerHTML = `
    <ul>
      <li>修改文件：${files}</li>
      ${flagList}
      ${notes.join("")}
    </ul>`;

  renderDiff(r.diff, (r.changed_files || []).join(", "));

  $("startBtn").disabled = false;
  $("startBtn").textContent = shouldResume ? "↻ 从断点继续" : "↻ Run Again";
  $("stopBtn").hidden = true;
}

function handleEvent(ev) {
  appendLog(ev);
  switch (ev.type) {
    case "round":
      $("round").textContent = `ROUND ${ev.n}`;
      addRoundDivider(ev.n);
      break;
    case "agent_text":
      addAgentNote(ev.text);
      break;
    case "tool":
      lastToolRow = addStep(TOOL_ICONS[ev.name] || "·", ev.name, ev.args, false);
      break;
    case "tool_result":
      attachToolResult(ev);
      break;
    case "rag":
      $("ragCode").textContent = ev.content;
      $("ragCount").textContent = "RETRIEVED";
      break;
    case "edited_file":
      $("diffFile").textContent = ev.file + "（运行中）";
      addStep("✎", `已修改 ${ev.file}`, "", true);
      break;
    case "test_counts":
      if (ev.source === "run_test") {
        $("passed").textContent = ev.passed;
        $("failed").textContent = ev.failed;
      } else if (ev.source === "run_bug_validation") {
        $("validation").textContent = ev.failed > 0 ? "FAIL" : "PASS";
        $("validation").style.color = ev.failed > 0 ? "var(--red)" : "var(--green)";
      }
      break;
    case "status":
      applyFlags(ev.flags, ev.phase);
      break;
    case "resume":
      addAgentNote(
        `↩ 断点恢复：上次已完成 ${ev.from_round} 轮，本次从第 ${ev.from_round + 1} 轮继续（进度不丢失）`
      );
      applyFlags(ev.flags, ev.phase);
      break;
    case "done":
      showResult(ev);
      if (es) { es.close(); es = null; }
      break;
  }
}

function openStream(taskId) {
  if (es) es.close();
  es = new EventSource(`/api/stream/${taskId}`);
  es.onmessage = (e) => {
    try {
      handleEvent(JSON.parse(e.data));
    } catch (err) { /* 忽略坏帧 */ }
  };
  es.onerror = () => {
    if (es && es.readyState === EventSource.CLOSED) {
      es = null;
      $("startBtn").disabled = false;
    }
  };
}

async function startRepair() {
  if (!currentTask) return;
  const tid = currentTask.task_id;
  // 断点续跑（LLM 失败 / 手动终止）绝不重置；
  // 其余情况由服务端按会话文件自动决定（有会话→续跑，无会话→干净开局）
  const resumeMode = currentTask.shouldResume === true;
  const reset = !resumeMode && activeTab() === "preset" && $("resetCheck").checked;

  $("startBtn").disabled = true;
  $("startBtn").textContent = "● Agent Running";
  $("stopBtn").hidden = false;
  setTopStatus("warn", "AGENT RUNNING");
  resetRunUI();

  try {
    const res = await fetch("/api/repair", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task_id: tid, reset }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      $("heroSub").textContent = data.detail || "启动失败";
      setTopStatus("err", "AGENT ERROR");
      $("startBtn").disabled = false;
      $("startBtn").textContent = "▶ Start Repair";
      $("stopBtn").hidden = true;
      return;
    }
    await refreshTree(tid);
    openStream(tid);
  } catch (err) {
    $("heroSub").textContent = "网络错误：" + err.message;
    setTopStatus("err", "AGENT ERROR");
    $("startBtn").disabled = false;
    $("startBtn").textContent = "▶ Start Repair";
    $("stopBtn").hidden = true;
  }
}

async function stopRepair() {
  if (!currentTask) return;
  if (!window.confirm("确定终止当前 Agent 运行？已完成的轮数会保留，之后可从断点继续。")) return;
  try {
    await fetch("/api/stop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task_id: currentTask.task_id }),
    });
  } catch (e) { /* 忽略 */ }
}

/* ---------------- 刷新 / 重开后恢复现场 ---------------- */

async function restoreSession() {
  let tasks = [];
  try {
    const res = await fetch("/api/active");
    if (res.ok) tasks = (await res.json()).tasks || [];
  } catch (e) { return; }
  if (!tasks.length) return;

  // 优先恢复本设备（localStorage）上次的任务，其次是正在运行的任务
  const saved = localStorage.getItem("codedoctor_task");
  let t = tasks.find((x) => x.task_id === saved);
  if (!t) t = tasks.find((x) => x.status === "running");
  if (!t) return;

  currentTask = {
    task_id: t.task_id,
    source: t.source,
    problem: t.problem || "",
    title: t.label || t.repo || t.task_id,
    filename: t.label,
    repo: t.repo,
    shouldResume: !!(t.result && (t.result.llm_failed || t.result.stopped)),
  };
  localStorage.setItem("codedoctor_task", t.task_id);
  $("taskId").textContent = t.task_id;
  $("heroTitle").textContent = t.label || t.repo || t.task_id;
  $("heroSub").textContent = `任务 ${t.task_id} · 来源 ${t.source} · ${statusText(t.status)}`;
  $("wsProject").textContent = t.repo || t.label || t.task_id;
  $("startBtn").disabled = false;

  await refreshTree(t.task_id);
  resetRunUI();

  if (t.status === "running") {
    $("startBtn").disabled = true;
    $("startBtn").textContent = "● Agent Running";
    $("stopBtn").hidden = false;
    setTopStatus("warn", "AGENT RUNNING");
  } else {
    $("startBtn").textContent = currentTask.shouldResume ? "↻ 从断点继续" : "↻ Run Again";
  }
  // SSE 回放：完整重建历史（轮次 / 工具 / 状态位 / 结果）
  openStream(t.task_id);
}

/* ---------------- 初始化 ---------------- */

document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-pane").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    $(`pane-${btn.dataset.tab}`).classList.add("active");
  });
});

const dz = $("dropzone");
dz.addEventListener("click", () => $("fileInput").click());
dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("drag"); });
dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
dz.addEventListener("drop", (e) => {
  e.preventDefault();
  dz.classList.remove("drag");
  if (e.dataTransfer.files.length) showFile(e.dataTransfer.files[0]);
});
$("fileInput").addEventListener("change", () => {
  if ($("fileInput").files.length) showFile($("fileInput").files[0]);
});

function showFile(f) {
  const chip = $("fileChip");
  chip.hidden = false;
  chip.textContent = `${f.name} · ${(f.size / 1024).toFixed(1)} KB`;
}

$("prepareBtn").addEventListener("click", prepareTask);
$("startBtn").addEventListener("click", startRepair);
$("stopBtn").addEventListener("click", stopRepair);
$("dlLogBtn").addEventListener("click", (e) => {
  e.stopPropagation();
  downloadLog();
});

renderPhases("LOCALIZE");
loadPresets();
restoreSession();
