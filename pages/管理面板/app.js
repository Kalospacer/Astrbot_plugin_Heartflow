// Heartflow 管理面板：通过 AstrBot 插件页 bridge（window.AstrBotPluginPage）调用 page_api.py 注册的接口。
// iframe 的 sandbox 只有 allow-scripts/forms/downloads：没有 confirm()、没有 localStorage，危险操作用两次点击确认。

const ENDPOINT_PREFIX = "page";
const POLL_MS = 5000;

const state = {
  tab: "overview",
  live: true,
  overview: null,
  log: [],
  logTotal: 0,
  persona: null,
  savedConfig: null,
  draft: null,
  preview: null,
  previewTab: "llm",
  dirty: false,
};

const $ = (id) => document.getElementById(id);

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

// ---------- bridge ----------

async function getBridge() {
  for (let waited = 0; waited < 3000; waited += 80) {
    const bridge = window.AstrBotPluginPage;
    if (bridge && typeof bridge.apiGet === "function") return bridge;
    await new Promise((resolve) => setTimeout(resolve, 80));
  }
  throw new Error("请从 AstrBot 后台的插件页打开此面板");
}

async function api(path, { method = "GET", body, params } = {}) {
  const bridge = await getBridge();
  const endpoint = `${ENDPOINT_PREFIX}/${path}`;
  const payload =
    method === "GET" ? await bridge.apiGet(endpoint, params) : await bridge.apiPost(endpoint, body || {});
  if (payload && typeof payload === "object" && "success" in payload) {
    if (!payload.success) throw new Error(payload.error || "请求失败");
    return payload.data;
  }
  return payload;
}

// ---------- formatting ----------

function fmtClock(ts) {
  const d = new Date(ts * 1000);
  return d.toTimeString().slice(0, 8);
}

function fmtAgo(ts) {
  if (!ts) return "从未";
  const seconds = Math.max(0, Date.now() / 1000 - ts);
  if (seconds < 60) return `${Math.floor(seconds)} 秒前`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)} 分钟前`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} 小时前`;
  return `${Math.floor(seconds / 86400)} 天前`;
}

function fmtSeconds(seconds) {
  if (!seconds) return "—";
  if (seconds < 60) return `${Math.ceil(seconds)} 秒`;
  return `${Math.ceil(seconds / 60)} 分钟`;
}

function splitUmo(umo) {
  const parts = String(umo).split(":");
  if (parts.length < 3) return { id: umo, meta: "" };
  return { id: parts.slice(2).join(":"), meta: `${parts[0]} · ${parts[1]}` };
}

function meter(value, threshold) {
  const pct = Math.max(0, Math.min(1, value ?? 0)) * 100;
  const tick =
    threshold === undefined ? "" : `<span class="meter-tick" style="left:calc(${threshold * 100}% - 1px)"></span>`;
  return `<div class="meter-track"><span class="meter-fill" style="width:${pct}%"></span>${tick}</div>`;
}

function engineLabel(o) {
  return o.judge_mode === "jev" ? `Jev · ${o.jev_decision_mode === "score" ? "Score" : "Noul"}` : "LLM";
}

// ---------- toast & two-step confirm ----------

let toastTimer;
function toast(message, isError = false) {
  const el = $("toast");
  el.textContent = message;
  el.classList.toggle("is-error", isError);
  el.classList.add("is-visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("is-visible"), isError ? 5000 : 2600);
}

function armConfirm(button, confirmText, action) {
  if (button.dataset.armed === "1") {
    button.dataset.armed = "";
    button.querySelector("span").textContent = button.dataset.label;
    action();
    return;
  }
  button.dataset.armed = "1";
  button.dataset.label = button.querySelector("span").textContent;
  button.querySelector("span").textContent = confirmText;
  setTimeout(() => {
    if (button.dataset.armed === "1") {
      button.dataset.armed = "";
      button.querySelector("span").textContent = button.dataset.label;
    }
  }, 3000);
}

// ---------- masthead ----------

function renderStatus() {
  const o = state.overview;
  if (!o) return;
  const rows = [
    ["状态", o.enabled ? "启用" : "已停用", !o.enabled],
    ["引擎", engineLabel(o)],
    ["阈值", Number(o.reply_threshold).toFixed(2)],
    ["人格", o.compress_persona ? "压缩" : "全文"],
    ["防抖", o.debounce_seconds > 0 ? `${o.debounce_seconds}s` : "关闭"],
    ["白名单", o.whitelist_enabled ? `${o.chat_whitelist.length} 个群` : "关闭"],
  ];
  $("statusStrip").innerHTML = rows
    .map(
      ([label, value, off]) =>
        `<div><dt>${label}</dt><dd class="${off ? "is-off" : ""}">${escapeHtml(value)}</dd></div>`,
    )
    .join("");
}

function markUpdated() {
  $("updatedAt").textContent = `更新于 ${new Date().toTimeString().slice(0, 8)}`;
}

function renderLiveButton() {
  const button = $("liveBtn");
  button.setAttribute("aria-pressed", String(state.live));
  button.querySelector("use").setAttribute("href", state.live ? "#i-pause" : "#i-play");
  button.querySelector("span").textContent = state.live ? "暂停实时" : "恢复实时";
}

// ---------- overview ----------

function renderOverview() {
  const o = state.overview;
  if (!o) return;
  const rate = o.judge_count ? Math.round((o.trigger_count / o.judge_count) * 100) : 0;
  const kpis = [
    ["群聊", o.chats.length, "内存中跟踪的会话"],
    ["判断", o.judge_count, "最近 200 条内"],
    ["触发", o.trigger_count, `触发率 ${rate}%`],
    ["阈值", Number(o.reply_threshold).toFixed(2), engineLabel(o)],
  ];
  $("kpis").innerHTML = kpis
    .map(
      ([label, value, sub]) =>
        `<div class="kpi"><div class="kpi-label">${label}</div><div class="kpi-value">${escapeHtml(value)}</div><div class="kpi-sub">${escapeHtml(sub)}</div></div>`,
    )
    .join("");

  if (!o.chats.length) {
    $("chatRows").innerHTML =
      '<tr><td colspan="9" class="empty">还没有群聊消息。开启插件并在允许的群里说话后，这里会出现实时状态。</td></tr>';
    return;
  }
  $("chatRows").innerHTML = o.chats
    .map((c) => {
      const umo = splitUmo(c.umo);
      const signal =
        c.last_signal === null
          ? "—"
          : `${c.last_signal.toFixed(2)} ${c.last_triggered ? '<span class="badge badge-hit">触发</span>' : ""}`;
      return `<tr>
        <td><div class="umo"><span class="umo-id">${escapeHtml(umo.id)}</span><span class="umo-meta">${escapeHtml(umo.meta)}</span></div></td>
        <td><div class="meter">${meter(c.energy)}<span class="meter-value">${Math.round(c.energy * 100)}%</span></div></td>
        <td class="num">${c.total_messages}</td>
        <td class="num">${c.total_replies}</td>
        <td>${fmtAgo(c.last_reply_time)}</td>
        <td>${fmtSeconds(c.cooldown_seconds)}</td>
        <td class="num">${c.pending || "—"}</td>
        <td class="num">${signal}</td>
        <td><button type="button" class="btn btn-ghost btn-danger" data-reset="${escapeHtml(c.umo)}"><svg class="icon" aria-hidden="true"><use href="#i-trash"/></svg><span>重置</span></button></td>
      </tr>`;
    })
    .join("");
}

function fillUmoSelect(select, emptyLabel) {
  const o = state.overview;
  if (!o) return;
  const umos = [...new Set([...o.chats.map((c) => c.umo), ...o.chat_whitelist])];
  const current = select.value;
  select.innerHTML =
    `<option value="">${emptyLabel}</option>` +
    umos.map((u) => `<option value="${escapeHtml(u)}">${escapeHtml(u)}</option>`).join("");
  select.value = umos.includes(current) ? current : "";
}

async function loadOverview() {
  state.overview = await api("overview");
  renderStatus();
  renderOverview();
  fillUmoSelect($("logUmo"), "全部");
  fillUmoSelect($("personaUmo"), "默认配置");
}

// ---------- judgement log ----------

function renderLog() {
  const threshold = state.overview?.reply_threshold;
  $("logCount").textContent = `显示 ${state.log.length} / 共 ${state.logTotal} 条`;
  if (!state.log.length) {
    $("logList").innerHTML = '<li class="empty">暂无判断记录。群里有新消息经过心流判断后会出现在这里。</li>';
    return;
  }
  $("logList").innerHTML = state.log
    .map((item) => {
      const msgs = item.batch.slice(-3);
      const more = item.batch.length - msgs.length;
      const scores = Object.entries(item.scores || {})
        .map(([k, v]) => `<span>${escapeHtml(k)} ${Number(v).toFixed(1)}</span>`)
        .join("");
      return `<li class="log-item">
        <div><div class="log-time">${fmtClock(item.ts)}</div><div class="log-engine">${escapeHtml(item.engine)}</div></div>
        <div>
          <div class="log-umo">${escapeHtml(item.umo)}</div>
          <ul class="log-msgs">${msgs
            .map((m) => `<li><b>${escapeHtml(m.from)}</b>${escapeHtml(m.text)}</li>`)
            .join("")}</ul>
          ${more > 0 ? `<div class="log-more">另有 ${more} 条更早的消息在同一批</div>` : ""}
          ${scores ? `<div class="log-scores">${scores}</div>` : ""}
          ${item.reasoning ? `<details class="log-reason"><summary>判断理由</summary><p>${escapeHtml(item.reasoning)}</p></details>` : ""}
        </div>
        <div class="log-signal">
          <span class="log-signal-value">${Number(item.signal).toFixed(2)}</span>
          <div class="meter">${meter(item.signal, item.threshold ?? threshold)}</div>
          <span class="badge ${item.triggered ? "badge-hit" : "badge-miss"}">${item.triggered ? "触发" : "未触发"}</span>
        </div>
      </li>`;
    })
    .join("");
}

async function loadLog() {
  const params = { limit: "100" };
  if ($("logUmo").value) params.umo = $("logUmo").value;
  if ($("logTriggered").checked) params.triggered = "1";
  const data = await api("judgements", { params });
  state.log = data.items;
  state.logTotal = data.total;
  renderLog();
}

async function toggleLastRequest() {
  const pre = $("lastRequest");
  if (!pre.hidden) {
    pre.hidden = true;
    return;
  }
  const data = await api("last_request");
  pre.textContent = data
    ? `// ${data.engine} · ${fmtClock(data.ts)}\n${JSON.stringify(data.body, null, 2)}`
    : "还没有发出过判断请求。";
  pre.hidden = false;
}

// ---------- persona ----------

function renderPersona() {
  const p = state.persona;
  if (!p) return;
  $("compressToggle").checked = p.compress_persona;
  $("compressedText").value = p.compressed_persona;
  updateCompressedCount();
  $("originalText").textContent = p.original || "（这个会话没有人格设定）";
  $("originalMeta").textContent = `${p.original.length} 字 · 压缩用模型：${p.judge_provider_name || "未配置"}`;
  $("regenPersonaBtn").disabled = !p.judge_provider_name;
}

function updateCompressedCount() {
  $("compressedCount").textContent = `${$("compressedText").value.length} 字`;
}

async function loadPersona() {
  const params = $("personaUmo").value ? { umo: $("personaUmo").value } : undefined;
  state.persona = await api("persona", { params });
  renderPersona();
}

async function savePersona() {
  const data = await api("persona/update", {
    method: "POST",
    body: { compress_persona: $("compressToggle").checked, compressed_persona: $("compressedText").value },
  });
  Object.assign(state.persona, data);
  renderPersona();
  toast("人格设置已保存");
}

async function regeneratePersona() {
  const button = $("regenPersonaBtn");
  button.disabled = true;
  button.querySelector("span").textContent = "压缩中…";
  try {
    const data = await api("persona/regenerate", { method: "POST", body: { umo: $("personaUmo").value } });
    state.persona.compressed_persona = data.compressed_persona;
    renderPersona();
    toast(`已重新压缩：${data.original_length} → ${data.compressed_persona.length} 字`);
  } finally {
    button.disabled = !state.persona?.judge_provider_name;
    button.querySelector("span").textContent = "按右侧原文重新压缩";
  }
}

// ---------- judge config ----------

function dimensionHtml(d, index) {
  const criteria = [0, 1, 2, 3, 4]
    .map(
      (score) =>
        `<li><span class="criteria-score">${score}</span><input type="text" data-dim="${index}" data-field="criteria" data-score="${score}" value="${escapeHtml(d.criteria?.[score] ?? "")}" aria-label="维度 ${index + 1} 的 ${score} 分标准" /></li>`,
    )
    .join("");
  const last = state.draft.score_dimensions.length - 1;
  return `<div class="dim">
    <div class="dim-index">${String(index + 1).padStart(2, "0")}</div>
    <div>
      <div class="dim-head">
        <label class="field"><span class="field-label">key</span><input type="text" data-dim="${index}" data-field="key" value="${escapeHtml(d.key)}" spellcheck="false" /></label>
        <label class="field"><span class="field-label">name</span><input type="text" data-dim="${index}" data-field="name" value="${escapeHtml(d.name)}" /></label>
        <div class="field">
          <span class="field-label">weight <span class="dim-share" data-share="${index}"></span></span>
          <div class="dim-weight">
            <input type="range" min="0" max="1" step="0.05" data-dim="${index}" data-field="weight" value="${d.weight}" aria-label="维度 ${index + 1} 权重" />
            <input type="number" min="0" max="1" step="0.05" data-dim="${index}" data-field="weight" value="${d.weight}" aria-label="维度 ${index + 1} 权重数值" />
          </div>
        </div>
        <div class="dim-tools">
          <button type="button" class="btn btn-ghost btn-icon" data-move="${index}" data-dir="-1" aria-label="上移" ${index === 0 ? "disabled" : ""}><svg class="icon" aria-hidden="true"><use href="#i-up"/></svg></button>
          <button type="button" class="btn btn-ghost btn-icon" data-move="${index}" data-dir="1" aria-label="下移" ${index === last ? "disabled" : ""}><svg class="icon" aria-hidden="true"><use href="#i-down"/></svg></button>
          <button type="button" class="btn btn-ghost btn-icon btn-danger" data-remove="${index}" aria-label="删除维度 ${index + 1}"><svg class="icon" aria-hidden="true"><use href="#i-trash"/></svg><span class="sr-only">删除</span></button>
        </div>
      </div>
      <div class="dim-body">
        <label class="field"><span class="field-label">instructions</span><textarea rows="3" data-dim="${index}" data-field="instructions" spellcheck="false">${escapeHtml(d.instructions)}</textarea></label>
        <span class="field-label">criteria · 0 → 4</span>
        <ol class="criteria">${criteria}</ol>
      </div>
    </div>
  </div>`;
}

function renderDimensions() {
  const dims = state.draft.score_dimensions;
  $("dimensions").innerHTML = dims.length
    ? dims.map(dimensionHtml).join("")
    : '<p class="empty">没有评分维度。llm 模式和 jev·score 模式至少需要一个维度。</p>';
  renderShares();
}

function renderShares() {
  const weights = state.preview?.weights || {};
  state.draft.score_dimensions.forEach((d, index) => {
    const el = document.querySelector(`[data-share="${index}"]`);
    if (el) el.textContent = d.key in weights ? `→ ${Math.round(weights[d.key] * 100)}%` : "";
  });
}

function renderConfigForm() {
  const c = state.draft;
  document.querySelector(`input[name="judge_mode"][value="${c.judge_mode}"]`).checked = true;
  document.querySelector(`input[name="jev_decision_mode"][value="${c.jev_decision_mode}"]`).checked = true;
  $("thresholdRange").value = c.reply_threshold;
  $("thresholdNum").value = c.reply_threshold;
  $("includeReasoning").checked = c.judge_include_reasoning;
  $("preamble").value = c.llm_judge_preamble;
  $("noulInstructions").value = c.jev_noul_question.instructions;
  $("noulTrue").value = c.jev_noul_question.criteria_true;
  $("noulFalse").value = c.jev_noul_question.criteria_false;
  renderDimensions();
}

function renderPreview() {
  const p = state.preview;
  if (!p) return;
  const entries = Object.entries(p.weights);
  $("weightBar").innerHTML = entries.length
    ? entries
        .map(
          ([key, w]) =>
            `<span class="weight-seg" style="flex:${w}" title="${escapeHtml(key)} ${Math.round(w * 100)}%">${escapeHtml(key)} ${Math.round(w * 100)}%</span>`,
        )
        .join("")
    : '<span class="weight-seg" style="flex:1">无维度</span>';
  document.querySelectorAll(".subtab").forEach((tab) => {
    tab.setAttribute("aria-selected", String(tab.dataset.preview === state.previewTab));
  });
  $("previewText").textContent =
    state.previewTab === "llm" ? p.llm_system_prompt : JSON.stringify(p.jev_questions, null, 2);
  renderShares();
}

function setDirty(dirty) {
  state.dirty = dirty;
  $("saveBar").hidden = !dirty;
}

let previewTimer;
function schedulePreview() {
  setDirty(true);
  clearTimeout(previewTimer);
  previewTimer = setTimeout(async () => {
    try {
      state.preview = await api("judge_config/preview", { method: "POST", body: state.draft });
      renderPreview();
    } catch (error) {
      toast(error.message, true);
    }
  }, 350);
}

async function loadConfig() {
  const data = await api("judge_config");
  state.savedConfig = data.config;
  state.draft = clone(data.config);
  state.preview = data.preview;
  renderConfigForm();
  renderPreview();
  setDirty(false);
}

async function saveConfig() {
  const button = $("saveConfigBtn");
  button.disabled = true;
  try {
    const data = await api("judge_config/update", { method: "POST", body: state.draft });
    state.savedConfig = data.config;
    state.draft = clone(data.config);
    state.preview = data.preview;
    renderConfigForm();
    renderPreview();
    setDirty(false);
    toast("判断配置已保存并即时生效");
    loadOverview().catch(() => {});
  } finally {
    button.disabled = false;
  }
}

function onConfigInput(event) {
  const target = event.target;
  const c = state.draft;
  if (target.name === "judge_mode") c.judge_mode = target.value;
  else if (target.name === "jev_decision_mode") c.jev_decision_mode = target.value;
  else if (target.id === "thresholdRange" || target.id === "thresholdNum") {
    const value = Math.max(0, Math.min(1, Number(target.value) || 0));
    c.reply_threshold = value;
    (target.id === "thresholdRange" ? $("thresholdNum") : $("thresholdRange")).value = value;
  } else if (target.id === "includeReasoning") c.judge_include_reasoning = target.checked;
  else if (target.id === "preamble") c.llm_judge_preamble = target.value;
  else if (target.id === "noulInstructions") c.jev_noul_question.instructions = target.value;
  else if (target.id === "noulTrue") c.jev_noul_question.criteria_true = target.value;
  else if (target.id === "noulFalse") c.jev_noul_question.criteria_false = target.value;
  else if (target.dataset.dim !== undefined) {
    const dim = c.score_dimensions[Number(target.dataset.dim)];
    const field = target.dataset.field;
    if (field === "criteria") {
      dim.criteria = dim.criteria || ["", "", "", "", ""];
      dim.criteria[Number(target.dataset.score)] = target.value;
    } else if (field === "weight") {
      dim.weight = Math.max(0, Math.min(1, Number(target.value) || 0));
      document
        .querySelectorAll(`[data-dim="${target.dataset.dim}"][data-field="weight"]`)
        .forEach((el) => el !== target && (el.value = dim.weight));
    } else {
      dim[field] = target.value;
    }
  } else {
    return;
  }
  schedulePreview();
}

function onConfigClick(event) {
  const button = event.target.closest("button");
  if (!button) return;
  const dims = state.draft.score_dimensions;
  if (button.dataset.move !== undefined) {
    const from = Number(button.dataset.move);
    const to = from + Number(button.dataset.dir);
    [dims[from], dims[to]] = [dims[to], dims[from]];
  } else if (button.dataset.remove !== undefined) {
    dims.splice(Number(button.dataset.remove), 1);
  } else if (button.id === "addDimBtn") {
    dims.push({
      __template_key: "dimension",
      key: `dimension_${dims.length + 1}`,
      name: "新维度",
      weight: 0.2,
      instructions: "",
      criteria: ["", "", "", "", ""],
    });
  } else {
    return;
  }
  renderDimensions();
  schedulePreview();
}

// ---------- tabs & polling ----------

const loaders = { overview: loadOverview, log: loadLog, persona: loadPersona, config: loadConfig };

async function refresh(tab = state.tab) {
  try {
    if (tab !== "overview") await loadOverview();
    await loaders[tab]();
    markUpdated();
  } catch (error) {
    toast(error.message, true);
  }
}

function showTab(tab) {
  if (!loaders[tab]) tab = "overview";
  state.tab = tab;
  document.querySelectorAll(".tab").forEach((el) => {
    el.setAttribute("aria-selected", String(el.dataset.tab === tab));
  });
  document.querySelectorAll(".view").forEach((el) => {
    el.hidden = el.id !== `view-${tab}`;
  });
  // 离开配置页时保留草稿；回到配置页且没有未保存修改时才重新拉取
  if (tab === "config" && state.dirty) return;
  refresh(tab);
}

setInterval(() => {
  // 只有概览和日志是实时数据；标签页不可见或手动暂停时不轮询
  if (!state.live || document.hidden) return;
  if (state.tab === "overview" || state.tab === "log") refresh();
}, POLL_MS);

// ---------- wiring ----------

window.addEventListener("hashchange", () => showTab(location.hash.slice(1)));

$("refreshBtn").addEventListener("click", () => refresh());
$("liveBtn").addEventListener("click", () => {
  state.live = !state.live;
  renderLiveButton();
});

$("chatRows").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-reset]");
  if (!button) return;
  armConfirm(button, "确认重置？", async () => {
    try {
      await api("chat/reset", { method: "POST", body: { umo: button.dataset.reset } });
      toast("已重置该群的状态");
      await loadOverview();
    } catch (error) {
      toast(error.message, true);
    }
  });
});

$("logUmo").addEventListener("change", () => loadLog().catch((e) => toast(e.message, true)));
$("logTriggered").addEventListener("change", () => loadLog().catch((e) => toast(e.message, true)));
$("lastRequestBtn").addEventListener("click", () => toggleLastRequest().catch((e) => toast(e.message, true)));

$("personaUmo").addEventListener("change", () => loadPersona().catch((e) => toast(e.message, true)));
$("compressedText").addEventListener("input", updateCompressedCount);
$("savePersonaBtn").addEventListener("click", () => savePersona().catch((e) => toast(e.message, true)));
$("regenPersonaBtn").addEventListener("click", (event) => {
  armConfirm(event.currentTarget, "确认覆盖当前压缩人格？", () =>
    regeneratePersona().catch((e) => toast(e.message, true)),
  );
});

$("configForm").addEventListener("input", onConfigInput);
$("configForm").addEventListener("change", onConfigInput);
$("configForm").addEventListener("click", onConfigClick);
document.querySelectorAll(".subtab").forEach((tab) => {
  tab.addEventListener("click", () => {
    state.previewTab = tab.dataset.preview;
    renderPreview();
  });
});
$("saveConfigBtn").addEventListener("click", () => saveConfig().catch((e) => toast(e.message, true)));
$("discardBtn").addEventListener("click", (event) => {
  armConfirm(event.currentTarget, "确认放弃？", () => {
    state.draft = clone(state.savedConfig);
    renderConfigForm();
    schedulePreview();
    setDirty(false);
  });
});

renderLiveButton();
showTab(location.hash.slice(1) || "overview");
