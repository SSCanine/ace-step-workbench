// Studio tab: same-seed regenerate, lyric delivery tools, re-sing a selected span (repaint).
import { $, api, busy, toast, emit, on, showTab, el, fmtTime } from "./core.js";
import { followJob } from "./create.js";

let song = null;
let peaks = null;
let sel = null; // {start, end} seconds
const audio = $("#s-audio");
const wave = $("#s-wave");
const canvas = $("canvas", wave);
const region = $(".region", wave);
const playhead = $(".playhead", wave);
const lyr = $("#s-lyrics");

async function loadPicker() {
  const { songs } = await api("/api/songs");
  const pick = $("#s-pick");
  const cur = song?.id || pick.value;
  pick.replaceChildren(el("option", { value: "" }, "Pick a song..."),
    ...songs.map((s) => el("option", { value: s.id }, `${s.title || "Untitled"} · seed ${s.seed} · ${new Date(s.created * 1000).toLocaleString()}`)));
  if (cur) pick.value = cur;
}

async function open(id) {
  if (!id) return;
  song = await api(`/api/songs/${id}`);
  $("#s-pick").value = id;
  $("#s-body").classList.remove("hidden");
  $("#s-title").textContent = song.title || "Untitled";
  $("#s-meta").textContent = `seed ${song.seed} · ${song.bpm ?? "?"} BPM · ${song.keyscale || "?"} · ${fmtTime(Number(song.duration))}${song.lora ? " · LoRA" : ""}`;
  $("#s-lineage").textContent = song.parent ? `made from ${song.parent}` : "";
  lyr.value = song.lyrics || "";
  $("#s-caption").value = song.caption || "";
  $("#s-suggestions").replaceChildren();
  $("#s-results").replaceChildren();
  setSel(null);
  audio.src = `/api/songs/${id}/audio`;
  drawWave(null);
  try {
    const buf = await (await fetch(audio.src)).arrayBuffer();
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const data = await ctx.decodeAudioData(buf);
    ctx.close();
    peaks = computePeaks(data, 1200);
  } catch { peaks = null; }
  drawWave(peaks);
}

function computePeaks(buf, n) {
  const ch = buf.getChannelData(0);
  const step = Math.max(1, Math.floor(ch.length / n));
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let m = 0;
    for (let j = i * step, e = Math.min(ch.length, j + step); j < e; j += 4) m = Math.max(m, Math.abs(ch[j]));
    out[i] = m;
  }
  return out;
}

function drawWave(p) {
  const r = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, r.width * devicePixelRatio);
  canvas.height = Math.max(1, r.height * devicePixelRatio);
  const g = canvas.getContext("2d");
  g.clearRect(0, 0, canvas.width, canvas.height);
  const color = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() || "#7c9cff";
  g.fillStyle = color;
  if (!p) { g.fillRect(0, canvas.height / 2, canvas.width, 1); return; }
  const mid = canvas.height / 2, w = canvas.width / p.length;
  for (let i = 0; i < p.length; i++) {
    const h = Math.max(1, p[i] * mid * 0.95);
    g.fillRect(i * w, mid - h, Math.max(1, w), h * 2);
  }
}

const dur = () => (audio.duration && isFinite(audio.duration) ? audio.duration : Number(song?.duration) || 0);

function setSel(s) {
  sel = s && s.end - s.start >= 0.05 ? s : null;
  if (!sel) { region.classList.add("hidden"); $("#s-sel").textContent = "none"; return; }
  const d = dur() || 1;
  region.classList.remove("hidden");
  region.style.left = `${(sel.start / d) * 100}%`;
  region.style.width = `${((sel.end - sel.start) / d) * 100}%`;
  $("#s-sel").textContent = `${sel.start.toFixed(1)}s - ${sel.end.toFixed(1)}s (${(sel.end - sel.start).toFixed(1)}s)`;
}

function waveEvents() {
  let down = null;
  const t = (e) => {
    const r = wave.getBoundingClientRect();
    return Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)) * dur();
  };
  wave.addEventListener("pointerdown", (e) => { down = t(e); wave.setPointerCapture(e.pointerId); });
  wave.addEventListener("pointermove", (e) => { if (down != null) { const x = t(e); setSel({ start: Math.min(down, x), end: Math.max(down, x) }); } });
  wave.addEventListener("pointerup", (e) => {
    const x = t(e);
    if (down != null && Math.abs(x - down) < 0.15) { audio.currentTime = x; }
    down = null;
  });
  audio.addEventListener("timeupdate", () => {
    playhead.style.left = `${(audio.currentTime / (dur() || 1)) * 100}%`;
    if (playSelEnd != null && audio.currentTime >= playSelEnd) { audio.pause(); playSelEnd = null; }
  });
  window.addEventListener("resize", () => drawWave(peaks));
}
let playSelEnd = null;

// ── lyric editing helpers ──
function lineBounds() {
  const v = lyr.value, pos = lyr.selectionStart;
  const start = v.lastIndexOf("\n", pos - 1) + 1;
  let end = v.indexOf("\n", pos);
  if (end === -1) end = v.length;
  return { start, end };
}

function replaceRange(start, end, text) {
  lyr.setRangeText(text, start, end, "select");
  lyr.focus();
}

// Selection if any, else the last word of the current line.
function target() {
  if (lyr.selectionEnd > lyr.selectionStart) return { start: lyr.selectionStart, end: lyr.selectionEnd };
  const { start, end } = lineBounds();
  const line = lyr.value.slice(start, end);
  const m = line.match(/([A-Za-z'À-ɏ-]+)[^A-Za-z'À-ɏ]*$/);
  if (!m) return null;
  const ws = start + line.lastIndexOf(m[1]);
  return { start: ws, end: ws + m[1].length };
}

const VOW = "aeiouyAEIOUY";
function stretch(word) {
  // repeat the last sung vowel (ignoring a silent final e): alive -> aliiive, night -> niiight
  let i = word.length - 1;
  if (/e$/i.test(word) && i > 1) i--;
  while (i >= 0 && !VOW.includes(word[i])) i--;
  if (i < 0) return word;
  return word.slice(0, i + 1) + word[i].repeat(2) + word.slice(i + 1);
}
function syllables(word) {
  const parts = word.match(/[^aeiouy]*[aeiouy]+(?:[^aeiouy]*$|[^aeiouy](?=[^aeiouy]))?/gi);
  return parts && parts.join("") === word ? parts.join("-") : word;
}

function applyOp(op) {
  if (op === "pause") {
    const { end } = lineBounds();
    const pos = lyr.selectionEnd > lyr.selectionStart ? lyr.selectionEnd : end;
    replaceRange(pos, pos, "...");
    return;
  }
  const t = target();
  if (!t) { toast("Put the cursor on a lyric line or select a word"); return; }
  const text = lyr.value.slice(t.start, t.end);
  const map = {
    caps: (s) => s.toUpperCase(),
    lower: (s) => s.toLowerCase(),
    paren: (s) => (s.startsWith("(") && s.endsWith(")") ? s.slice(1, -1) : `(${s})`),
    stretch: (s) => s.replace(/[A-Za-z']+/g, stretch),
    hyphen: (s) => s.replace(/[A-Za-z]+/g, syllables),
  };
  replaceRange(t.start, t.end, map[op](text));
}

function insertTag(tag) {
  const { start } = lineBounds();
  replaceRange(start, start, `${tag}\n`);
}

// Estimate when a lyric line is sung: spread sung lines over the song weighted by syllables.
function estimateLineTime() {
  const lines = lyr.value.split("\n");
  const caretLine = lyr.value.slice(0, lyr.selectionStart).split("\n").length - 1;
  const syl = (s) => (s.toLowerCase().match(/[aeiouy]+/g) || []).length || 1;
  const items = [];
  lines.forEach((l, i) => {
    const s = l.trim();
    if (!s) return;
    if (/^\[.*\]$/.test(s)) {
      if (/intro|outro|solo|instrumental|break|interlude|drop|build/i.test(s)) items.push({ i, w: 8, sung: false });
      return;
    }
    items.push({ i, w: syl(s), sung: true });
  });
  const it = items.find((x) => x.i === caretLine && x.sung);
  if (!it) { toast("Put the cursor on a sung lyric line (not a [tag])"); return; }
  const total = items.reduce((a, x) => a + x.w, 0);
  const d = dur();
  const lead = d * 0.04, span = d * 0.92;
  let acc = 0;
  for (const x of items) { if (x === it) break; acc += x.w; }
  const start = lead + (acc / total) * span;
  const end = start + (it.w / total) * span;
  setSel({ start: Math.max(0, start - 0.6), end: Math.min(d, end + 0.6) });
  audio.currentTime = sel.start;
  toast("Estimated. Play the selection and drag on the waveform to adjust.");
}

function jobUi() {
  return { box: $("#s-job"), stage: $("#s-stage"), elapsed: $("#s-elapsed"), results: $("#s-results") };
}

export function init() {
  waveEvents();
  $("#s-refresh").addEventListener("click", loadPicker);
  $("#s-pick").addEventListener("change", (e) => open(e.target.value));
  document.querySelectorAll("#tab-studio [data-op]").forEach((b) => b.addEventListener("click", () => applyOp(b.dataset.op)));
  $("#s-tag").addEventListener("change", (e) => { if (e.target.value) insertTag(e.target.value); e.target.value = ""; });
  $("#s-strength").addEventListener("input", (e) => { $("#s-strength-v").textContent = e.target.value; });
  $("#s-play-sel").addEventListener("click", () => {
    if (!sel) { toast("Drag on the waveform to select a part first"); return; }
    audio.currentTime = sel.start; playSelEnd = sel.end; audio.play();
  });
  $("#s-clear-sel").addEventListener("click", () => setSel(null));
  $("#s-line-time").addEventListener("click", estimateLineTime);
  $("#s-reset-lyrics").addEventListener("click", () => { lyr.value = song?.lyrics || ""; });

  $("#s-suggest").addEventListener("click", (e) => busy(e.currentTarget, async () => {
    const { start, end } = lineBounds();
    const line = lyr.value.slice(start, end).trim();
    if (!line) { toast("Put the cursor on the line you want to fix"); return; }
    const { options } = await api("/api/write/phrase", { method: "POST", body: { line, wish: $("#s-wish").value, context: lyr.value } });
    $("#s-suggestions").replaceChildren(...options.map((o) => el("div", { class: "notice" },
      el("div", { class: "mono" }, o.line), el("div", { class: "row" },
        el("span", { class: "muted small" }, o.why), el("span", { class: "spacer" }),
        el("button", { class: "b small", onclick: () => {
          const b = lineBounds();
          replaceRange(b.start, b.end, o.line);
          $("#s-suggestions").replaceChildren();
        } }, "Use this")))));
  }));

  const edits = () => ({ lyrics: lyr.value, caption: $("#s-caption").value });
  $("#s-same-seed").addEventListener("click", (e) => busy(e.currentTarget, async () => {
    const { job } = await api(`/api/songs/${song.id}/remix`, { method: "POST", body: { mode: "same_seed", form: edits() } });
    followJob(job, jobUi());
  }));
  $("#s-variations").addEventListener("click", (e) => busy(e.currentTarget, async () => {
    const { job } = await api(`/api/songs/${song.id}/remix`, { method: "POST", body: { mode: "variation", count: 2, form: edits() } });
    followJob(job, jobUi());
  }));
  $("#s-repaint").addEventListener("click", (e) => busy(e.currentTarget, async () => {
    if (!sel) { toast("Select the part to re-sing: drag on the waveform, or use 'Select line's time'"); return; }
    const form = { ...edits(), repaint_start: +sel.start.toFixed(2), repaint_end: +sel.end.toFixed(2), repaint_strength: Number($("#s-strength").value) };
    const { job } = await api(`/api/songs/${song.id}/repaint`, {
      method: "POST", body: { form, keep_seed: $("#s-keep-seed").checked, count: Number($("#s-takes").value) || 1 },
    });
    followJob(job, jobUi());
  }));
  $("#s-to-create").addEventListener("click", () => {
    const f = song.form || {};
    emit("load-create", { ...f, title: song.title, caption: $("#s-caption").value, lyrics: lyr.value, seed: song.seed, batch_size: 1, lora: song.lora || "", lora_scale: song.lora_scale });
    showTab("create");
  });

  on("open-studio", (id) => { loadPicker().then(() => open(id)); });
  on("tab", (t) => { if (t === "studio") { loadPicker(); if (song) drawWave(peaks); } });
  on("library-changed", loadPicker);
}
