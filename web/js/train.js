// Train tab: datasets, auto-label, per-track editing, training with live loss chart.
import { $, $$, api, busy, toast, on, el } from "./core.js";

let presets = {};
let ds = null;
let poll = null;

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
  $("#t-stage").textContent = st.status === "idle" ? "" : `${st.stage || st.status}${st.error ? `: ${st.error}` : ""}`;
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
  const running = ["preparing", "preprocessing", "training"].includes(st.status);
  $("#t-start").disabled = running; $("#t-stop").disabled = !running;
  return running;
}

function startPolling() {
  clearInterval(poll);
  poll = setInterval(async () => {
    try {
      const running = await refreshStatus();
      if (!running) { clearInterval(poll); poll = null; }
    } catch { /* server restarting */ }
  }, 2000);
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
    await save();
    const opts = {
      run_name: $("#t-run").value.trim() || undefined, model_variant: $("#t-variant").value || undefined,
      adapter_type: $("#t-adapter").value, rank: +$("#t-rank").value, alpha: +$("#t-alpha").value, lr: +$("#t-lr").value,
      epochs: +$("#t-epochs").value, save_every: +$("#t-save-every").value, grad_accum: +$("#t-accum").value,
      max_duration: +$("#t-maxdur").value || 240, offload_encoder: $("#t-offload").checked,
    };
    await api("/api/train/start", { method: "POST", body: { dataset: ds.name, opts } });
    toast("Training started. ACE-Step pauses until it finishes.");
    startPolling();
  }));
  $("#t-stop").addEventListener("click", async () => {
    if (!confirm("Stop training? Checkpoints saved so far are kept.")) return;
    await api("/api/train/stop", { method: "POST" });
  });

  on("tab", async (t) => {
    if (t !== "train") return;
    await loadList(ds?.name);
    if (await refreshStatus()) startPolling();
  });
}
