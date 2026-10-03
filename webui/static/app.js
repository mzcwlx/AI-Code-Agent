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

/* ---------------- 阶段 chips ---------------- */

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

/* ---------------- 任务准备 ---------------- */

function activeTab() {
  return document.querySelector(".tab.active").dataset.tab;
}

function setTask(task) {
  currentTask = task;
  $("taskId").textContent = task.task_id;
  $("heroTitle").textContent = task.title || task.problem.slice(0, 110);
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

let presetTasks = [];

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
      o.textContent = `${t.instance_id}（${t.repo}）`;
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
  $("baseline").textContent = "—";
  $("testStatus").textContent = "RUNNING";
  $("testStatus").className = "";
  $("testStatus").style.color = "var(--yellow)";
  $("resultCard").hidden = true;
  $("downloadBtn").hidden = true;
  rawLogLines = [];
  $("rawLog").textContent = "";
  renderPhases("LOCALIZE");
}

function setTopStatus(mode, text) {
  const el = $("agentStatus");
  el.className = "status" + (mode === "warn" ? " warn" : mode === "err" ? " err" : "");
  $("statusText").textContent = text;
}

function appendRaw(line) {
  rawLogLines.push(line);
  if (rawLogLines.length > 400) rawLogLines.splice(0, rawLogLines.length - 400);
  $("rawLog").textContent = rawLogLines.join("\n");
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

  const files = (r.changed_files || []).map((f) => esc(f)).join("、") || "无";
  $("resultText").innerHTML = `
    <ul>
      <li>修改文件：${files}</li>
      ${flagList}
      ${r.llm_failed ? '<li style="color:var(--red)">LLM 调用失败中断（可直接重试，从断点继续）</li>' : ""}
      ${r.max_rounds && !ev.success ? '<li style="color:var(--red)">达到最大轮数仍未完成修复</li>' : ""}
    </ul>`;

  renderDiff(r.diff, (r.changed_files || []).join(", "));
}

function handleEvent(ev) {
  switch (ev.type) {
    case "round":
      $("round").textContent = `ROUND ${ev.n}`;
      addRoundDivider(ev.n);
      appendRaw(`========= 第 ${ev.n} 轮 =========`);
      break;
    case "agent_text":
      addAgentNote(ev.text);
      appendRaw(ev.text);
      break;
    case "tool":
      addStep(TOOL_ICONS[ev.name] || "·", ev.name, ev.args, false);
      appendRaw(`调用工具：${ev.name} ${ev.args || ""}`);
      break;
    case "log":
      appendRaw(ev.line);
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
    case "status": {
      renderPhases(ev.phase);
      const f = ev.flags;
      if (f.baseline_confirmed) {
        $("baseline").textContent = "FAIL";
        $("baseline").style.color = "var(--yellow)";
      }
      if (f.validation_passed) {
        $("validation").textContent = "PASS";
        $("validation").style.color = "var(--green)";
      }
      if (f.tests_passed) {
        $("testStatus").textContent = "TESTS PASS";
        $("testStatus").style.color = "var(--green)";
      }
      break;
    }
    case "done":
      showResult(ev);
      if (es) { es.close(); es = null; }
      $("startBtn").disabled = false;
      $("startBtn").textContent = "↻ Run Again";
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
  const resetAllowed = currentTask.source !== "upload";
  let reset = resetAllowed && activeTab() === "preset" && $("resetCheck").checked;

  $("startBtn").disabled = true;
  $("startBtn").textContent = "● Agent Running";
  setTopStatus("warn", "AGENT RUNNING");
  resetRunUI();

  try {
    let res = await fetch("/api/repair", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task_id: tid, reset }),
    });

    // 已完成的任务：非上传任务自动带 reset 重试一次
    if (res.status === 409 && resetAllowed && !reset) {
      const d = await res.json();
      if ((d.detail || "").includes("已完成")) {
        reset = true;
        res = await fetch("/api/repair", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ task_id: tid, reset: true }),
        });
      }
    }

    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      $("heroSub").textContent = data.detail || "启动失败";
      setTopStatus("err", "AGENT ERROR");
      $("startBtn").disabled = false;
      $("startBtn").textContent = "▶ Start Repair";
      return;
    }

    await refreshTree(tid);
    openStream(tid);
  } catch (err) {
    $("heroSub").textContent = "网络错误：" + err.message;
    setTopStatus("err", "AGENT ERROR");
    $("startBtn").disabled = false;
    $("startBtn").textContent = "▶ Start Repair";
  }
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

renderPhases("LOCALIZE");
loadPresets();
