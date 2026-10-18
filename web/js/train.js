// Train tab: datasets, auto-label, per-track editing, a queue of LoRAs (one per model), live loss chart.
import { $, $$, api, busy, toast, on, emit, showTab, el } from "./core.js";

let presets = {};
let ds = null;
let poll = null;
let models = [];

async function loadList(select) {
  const { datasets } = await api("/api/datasets");
  const s = $("#t-ds");
  s.replaceChildren(el("option", { value: "" }, "Pick a dataset..."),
    ...datasets.map((d) => el("option", { value: d.name }, `${d.name} · ${d.count} files · ${d.labeled} described${d.trigger ? ` · ${d.trigger}` : ""}`)));
  if (select) s.value = select;
}

async function openDs(name) {
  if (!name) { $("#t-body").classList.add("hidden"); ds = null; return; }
  ds = await api(`/api/datasets/${encodeURIComponent(name)}`);
  ds.name = name;
  $("#t-body").classList.remove("hidden");
  $("#t-trigger").value = ds.metadata.custom_tag || "";
  renderSamples();
}

function renderSamples() {
  const n = ds.samples.length, labeled = ds.samples.filter((s) => s.caption).length;
  $("#t-count").textContent = `${n} file(s), ${labeled} described. 8-30 songs that share one sound work best.`;
  const enc = encodeURIComponent;
  $("#t-samples tbody").replaceChildren(...ds.samples.map((s, i) => el("tr", { "data-i": i },
    el("td", { class: "small" }, el("div", {}, s.filename),
      el("audio", { controls: true, preload: "none", src: `/api/datasets/${enc(ds.name)}/audio/${enc(s.filename)}`, style: "width:180px;height:30px" })),
    el("td", {}, el("textarea", { "data-k": "caption", style: "min-height:60px" }, s.caption || "")),
    el("td", {}, el("input", { type: "number", "data-k": "bpm", value: s.bpm ?? "", style: "width:70px" })),
    el("td", {}, el("input", { type: "text", "data-k": "keyscale", value: s.keyscale || "", style: "width:90px" })),
    el("td", {}, el("input", { type: "checkbox", "data-k": "is_instrumental", checked: !!s.is_instrumental })),
    el("td", {}, el("details", {}, el("summary", { class: "small" }, s.is_instrumental ? "instrumental" : "edit"),
      el("textarea", { "data-k": "lyrics", class: "lyrics", style: "min-height:160px;min-width:260px" }, s.lyrics || ""))),
    el("td", {}, el("button", { class: "b small danger", title: "Remove from dataset (moved to data/trash)", onclick: async () => {
      await api(`/api/datasets/${enc(ds.name)}/files/${enc(s.filename)}`, { method: "DELETE" });
      openDs(ds.name);
    } }, "✕")))));
}

function gatherSamples() {
  return $$("#t-samples tbody tr").map((tr) => {
    const s = ds.samples[Number(tr.dataset.i)];
    const out = { filename: s.filename };
    $$("[data-k]", tr).forEach((e) => {
      const k = e.dataset.k;
      out[k] = e.type === "checkbox" ? e.checked : e.type === "number" ? (e.value === "" ? null : Number(e.value)) : e.value;
    });
    return out;
  });
}

async function save() {
  ds = { ...(await api(`/api/datasets/${encodeURIComponent(ds.name)}`, {
    method: "PUT", body: { metadata: { custom_tag: $("#t-trigger").value.trim() }, samples: gatherSamples() },
  })), name: ds.name };
  renderSamples();
  loadList(ds.name);
}

function applyPreset(key) {
  const p = presets[key];
  if (!p) return;
  $("#t-adapter").value = p.adapter_type;
  $("#t-rank").value = p.rank; $("#t-alpha").value = p.alpha; $("#t-lr").value = p.lr;
  $("#t-epochs").value = p.epochs; $("#t-save-every").value = p.save_every; $("#t-accum").value = p.grad_accum;
  $("#t-preset-about").textContent = p.about;
}

function drawChart(points) {
  const c = $("#t-chart");
  const r = c.getBoundingClientRect();
  c.width = r.width * devicePixelRatio; c.height = r.height * devicePixelRatio;
  const g = c.getContext("2d");
  g.clearRect(0, 0, c.width, c.height);
  const css = getComputedStyle(document.documentElement);
  g.fillStyle = css.getPropertyValue("--muted"); g.font = `${11 * devicePixelRatio}px sans-serif`;
  if (!points || points.length < 2) { g.fillText("Loss per epoch appears here", 10 * devicePixelRatio, 20 * devicePixelRatio); return; }
  const ys = points.map((p) => p[1]), xs = points.map((p) => p[0]);
  const minY = Math.min(...ys), maxY = Math.max(...ys), minX = Math.min(...xs), maxX = Math.max(...xs);
  const pad = 28 * devicePixelRatio;
  const X = (x) => pad + ((x - minX) / Math.max(1e-9, maxX - minX)) * (c.width - pad * 1.5);
  const Y = (y) => c.height - pad + -((y - minY) / Math.max(1e-9, maxY - minY)) * (c.height - pad * 1.6);
  g.fillText(maxY.toFixed(3), 2, Y(maxY) + 4); g.fillText(minY.toFixed(3), 2, Y(minY));
  g.strokeStyle = css.getPropertyValue("--accent"); g.lineWidth = 2 * devicePixelRatio;
  g.beginPath();
  points.forEach((p, i) => (i ? g.lineTo(X(p[0]), Y(p[1])) : g.moveTo(X(p[0]), Y(p[1]))));
  g.stroke();
}

function fmtEta(s) {
  if (s == null) return "-";
  const h = Math.floor(s / 3600), m = Math.round((s % 3600) / 60);
  return h ? `${h}h ${m}m` : `${m}m`;
}

async function refreshStatus() {
  const st = await api("/api/train/status");
  $("#t-stage").textContent = st.status === "idle" ? "Nothing training right now." : `${st.run ? `${st.run}: ` : ""}${st.stage || st.status}${st.error ? `: ${st.error}` : ""}`;
  $("#t-epoch").textContent = st.epochs ? `${st.epoch || 0} / ${st.epochs}` : "-";
  $("#t-loss").textContent = st.loss != null ? Number(st.loss).toFixed(4) : "-";
  $("#t-eta").textContent = st.status === "training" ? fmtEta(st.eta_seconds) : "-";
  $("#t-out").textContent = st.output || st.run || "-";
  $("#t-bar").style.width = st.epochs ? `${Math.min(100, ((st.epoch || 0) / st.epochs) * 100)}%` : "0";
  drawChart(st.epoch_losses);
  const log = $("#t-log");
  const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 20;
  log.textContent = (st.log || []).join("\n");
  if (atBottom) log.scrollTop = log.scrollHeight;
}

// ── queue ──
const STATUS = {
  queued: "Queued", waiting: "Waiting for the GPU", importing: "Copying songs", labeling: "Describing songs",
  training: "Training", testing: "Test clip", done: "✓ Done", error: "Failed", cancelled: "Cancelled",
};
const ACTIVE = ["waiting", "importing", "labeling", "training", "testing"];

function renderModels(list) {
  models = list;
  const box = $("#t-models");
  const checked = new Set($$("input", box).filter((i) => i.checked).map((i) => i.value));
  const first = !box.children.length;
  box.replaceChildren(...list.map((m) => el("label", {
    class: "check", "data-tip": m.installed ? `Train a LoRA for ${m.label}.` : `${m.label} isn't downloaded. Download it with ACE-Step first.`,
  }, el("input", { type: "checkbox", value: m.variant, disabled: !m.installed,
    checked: m.installed && (first ? m.variant === (list.find((x) => x.installed && x.variant === "xl_turbo") ? "xl_turbo" : "turbo") : checked.has(m.variant)) }),
  m.installed ? m.label : `${m.label} (not downloaded)`)));
}

function jobRow(j) {
  const act = (what, label, tip, cls = "b small") => el("button", { class: cls, title: tip, onclick: async () => {
    if (what === "cancel" && ACTIVE.includes(j.status) && !confirm(`Cancel ${j.run_name}? Checkpoints saved so far are kept.`)) return;
    await api(`/api/train/queue/${j.id}/${what}`, { method: "POST" });
    refreshQueue();
  } }, label);
  const progress = j.status === "training" && j.epochs
    ? el("div", { class: "progress", style: "margin-top:6px" }, el("i", { style: `width:${Math.min(100, ((j.epoch || 0) / j.epochs) * 100)}%` })) : null;
  const detail = j.status === "training" && j.epochs
    ? `Epoch ${j.epoch || 0}/${j.epochs}${j.loss != null ? ` · loss ${Number(j.loss).toFixed(4)}` : ""}${j.eta_seconds != null ? ` · ${fmtEta(j.eta_seconds)} left` : ""}`
    : j.stage && j.stage !== STATUS[j.status] && j.status !== "done" ? j.stage : "";
  return el("tr", {},
    el("td", {}, el("b", {}, j.dataset), el("div", { class: "muted small" }, j.run_name)),
    el("td", { class: "small" }, j.model_label),
    el("td", { class: "small" },
      el("b", { class: j.status === "error" ? "bad" : j.status === "done" ? "good" : "" }, STATUS[j.status] || j.status),
      detail ? el("div", { class: "muted" }, detail) : null, progress,
      j.error ? el("div", { class: "bad" }, j.error) : null,
      ...(j.notes || []).map((n) => el("div", { class: "muted" }, n))),
    el("td", {}, j.test_song ? el("div", { class: "row", style: "flex-wrap:nowrap" },
      el("audio", { controls: true, preload: "none", src: `/api/songs/${j.test_song}/audio`, style: "width:180px;height:30px" }),
      el("button", { class: "b small", title: "Open the test clip in Studio", onclick: () => { emit("open-studio", j.test_song); showTab("studio"); } }, "🎚️"))
      : el("span", { class: "muted small" }, j.test_clip ? (j.status === "done" ? "-" : "after training") : "off")),
    el("td", {}, el("div", { class: "row", style: "flex-wrap:nowrap" },
      j.status === "queued" ? act("up", "▲", "Move up") : null,
      j.status === "queued" ? act("down", "▼", "Move down") : null,
      ["queued", "waiting"].includes(j.status) && !j.skip_gate ? act("now", "Start now", "Don't wait for an idle GPU for this job") : null,
      j.status === "queued" || ACTIVE.includes(j.status) ? act("cancel", "Cancel", "Cancel this job", "b small danger") : null,
      ["error", "cancelled"].includes(j.status) ? act("retry", "Retry", "Queue it again") : null,
      ["done", "error", "cancelled"].includes(j.status) ? act("remove", "✕", "Remove from the list (the LoRA stays)") : null)));
}

async function refreshQueue() {
  const q = await api("/api/train/queue");
  renderModels(q.models);
  const g = q.gpu;
  $("#t-gpu").textContent = g ? `GPU: ${g.free_gb} of ${g.total_gb} GB free · ${g.util}% busy` : "";
  $("#t-queue tbody").replaceChildren(...(q.jobs.length ? q.jobs.map(jobRow)
    : [el("tr", {}, el("td", { colspan: "5", class: "muted small" }, "Nothing queued. Pick a dataset, tick the models in step 3 and press Add to queue."))]));
  return q.jobs.some((j) => j.status === "queued" || ACTIVE.includes(j.status));
}

function startPolling() {
  if (poll) return;
  poll = setInterval(async () => {
    if (!$("#tab-train").classList.contains("active")) { clearInterval(poll); poll = null; return; }
    try {
      const busyQueue = await refreshQueue();
      await refreshStatus();
      if (!busyQueue) { clearInterval(poll); poll = null; }
    } catch { /* server restarting */ }
  }, 2500);
}

export async function init() {
  presets = (await api("/api/settings")).train_presets;
  $("#t-preset").replaceChildren(...Object.entries(presets).map(([k, p]) => el("option", { value: k }, p.label)));
  $("#t-preset").value = "balanced";
  applyPreset("balanced");
  $("#t-preset").addEventListener("change", (e) => applyPreset(e.target.value));

  $("#t-ds").addEventListener("change", (e) => openDs(e.target.value));
  $("#t-create").addEventListener("click", (e) => busy(e.currentTarget, async () => {
    const name = $("#t-new-name").value.trim(), trigger = $("#t-new-trigger").value.trim();
    if (!name) { toast("Name the dataset"); return; }
    if (!trigger) { toast("Pick a trigger word (a made-up word like zxsynth)"); return; }
    await api("/api/datasets", { method: "POST", body: { name, trigger } });
    $("#t-new-name").value = ""; $("#t-new-trigger").value = "";
    await loadList(name); openDs(name);
  }));
  $("#t-bring").addEventListener("click", (e) => busy(e.currentTarget, async () => {
    const folder = $("#t-existing").value.trim();
    if (!folder) { toast("Paste the folder that holds dataset.json"); return; }
    const r = await api("/api/dataset-import", { method: "POST", body: { folder } });
    toast(`Brought in "${r.name}" with ${r.added} song(s)`);
    $("#t-existing").value = "";
    await loadList(r.name); openDs(r.name);
  }));
  $("#t-files").addEventListener("change", async (e) => {
    const fd = new FormData();
    for (const f of e.target.files) fd.append("files", f);
    toast(`Uploading ${e.target.files.length} file(s)...`);
    const r = await api(`/api/datasets/${encodeURIComponent(ds.name)}/files`, { method: "POST", form: fd });
    toast(`Added ${r.added} file(s)`);
    e.target.value = "";
    openDs(ds.name); loadList(ds.name);
  });
  $("#t-import").addEventListener("click", (e) => busy(e.currentTarget, async () => {
    const r = await api(`/api/datasets/${encodeURIComponent(ds.name)}/import`, { method: "POST", body: { folder: $("#t-folder").value } });
    toast(`Copied ${r.added} file(s)`);
    openDs(ds.name); loadList(ds.name);
  }));
  $("#t-save").addEventListener("click", (e) => busy(e.currentTarget, async () => { await save(); toast("Saved"); }));
  $("#t-autolabel").addEventListener("click", (e) => busy(e.currentTarget, async () => {
    await save();
    await api(`/api/datasets/${encodeURIComponent(ds.name)}/autolabel`, { method: "POST", body: { transcribe: $("#t-transcribe").checked, overwrite: $("#t-overwrite").checked } });
    const t = setInterval(async () => {
      const st = await api("/api/autolabel");
      $("#t-al-status").textContent = st.error ? `Failed: ${st.error}` : st.progress || "";
      if (st.status !== "running") { clearInterval(t); openDs(ds.name); }
    }, 2000);
  }));
  $("#t-start").addEventListener("click", (e) => busy(e.currentTarget, async () => {
    const variants = $$("#t-models input").filter((i) => i.checked).map((i) => i.value);
    if (!variants.length) { toast("Tick at least one model"); return; }
    await save();
    const preset = presets[$("#t-preset").value] || {};
    const opts = {
      adapter_type: $("#t-adapter").value, rank: +$("#t-rank").value, alpha: +$("#t-alpha").value, lr: +$("#t-lr").value,
      epochs: +$("#t-epochs").value, save_every: +$("#t-save-every").value, grad_accum: +$("#t-accum").value,
      batch_size: preset.batch_size || 1, max_duration: +$("#t-maxdur").value || 240, offload_encoder: $("#t-offload").checked,
    };
    const { jobs } = await api("/api/train/queue", { method: "POST", body: {
      dataset: ds.name, variants, preset: $("#t-preset").value, opts, run_prefix: $("#t-run").value.trim(),
      source_folder: $("#t-source").value.trim(), label_missing: $("#t-q-label").checked,
      transcribe: $("#t-q-transcribe").checked, test_clip: $("#t-q-test").checked,
    } });
    toast(`Queued ${jobs.map((j) => j.run_name).join(", ")}`);
    await refreshQueue();
    startPolling();
  }));
  $("#t-q-clear").addEventListener("click", async () => { await api("/api/train/queue/clear", { method: "POST" }); refreshQueue(); });

  on("tab", async (t) => {
    if (t !== "train") return;
    await loadList(ds?.name);
    await refreshStatus();
    if (await refreshQueue()) startPolling();
  });
}
