// Write tab: AI songwriter with ACE-Step (unfiltered) and Suno (safe) targets.
import { $, $$, api, busy, toast, copy, readForm, fillForm, emit, showTab, el } from "./core.js";

const root = $("#tab-write");
const res = $("#w-result");
let target = "ace";
let bundle = null;

function setTarget(t) {
  target = t;
  $$("#w-target button").forEach((b) => b.classList.toggle("on", b.dataset.t === t));
  $$(".only-ace", root).forEach((e) => e.classList.toggle("hidden", t !== "ace"));
  $$(".only-suno", root).forEach((e) => e.classList.toggle("hidden", t !== "suno"));
  try { localStorage.setItem("wb.target", t); } catch {}
}

function opts() {
  const f = readForm($("#w-brief"));
  return {
    genre: f.genre, mood: f.mood, vocals: f.vocals, language: f.language, length: f.length,
    structure: f.structure, theme_notes: f.theme_notes, explicit: f.explicit, soften: f.soften,
  };
}

function render(b) {
  bundle = b;
  res.hidden = false;
  setTarget(b.target);
  fillForm(res, {
    title: b.title, caption: b.caption || "", style: b.style || "", exclude: b.exclude || "",
    lyrics: b.lyrics, bpm: b.bpm ?? "", keyscale: b.keyscale || "", timesignature: b.timesignature || "",
    duration: b.duration ?? "", vocal_language: b.vocal_language || "", instrumental: !!b.instrumental,
  });
  $("#w-source").textContent = b.source === "rules" ? "Rule-based draft (no AI writer)" : `Written by ${b.source}`;
  $("#w-notes").textContent = b.notes || "";
  $("#w-notes").hidden = !b.notes;
  const ch = $("#w-changes");
  ch.replaceChildren(...(b.changes || []).map((c) => el("li", {}, c)));
  ch.parentElement.hidden = !(b.changes || []).length;
  counts();
  try { localStorage.setItem("wb.bundle", JSON.stringify(b)); } catch {}
}

function current() {
  const f = readForm(res);
  const b = { ...(bundle || {}), target, title: f.title, lyrics: f.lyrics };
  if (target === "suno") Object.assign(b, { style: f.style, exclude: f.exclude });
  else Object.assign(b, {
    caption: f.caption, bpm: f.bpm, keyscale: f.keyscale, timesignature: f.timesignature,
    duration: f.duration, vocal_language: f.vocal_language, instrumental: f.instrumental,
  });
  return b;
}

function counts() {
  const s = $('[name="style"]', res), l = $('[name="lyrics"]', res);
  $("#w-style-count").textContent = `${s.value.length} / 1000`;
  $("#w-style-count").style.color = s.value.length > 1000 ? "var(--bad)" : "";
  $("#w-lyrics-count").textContent = target === "suno" ? `${l.value.length} / 5000` : `${l.value.length} chars`;
  $("#w-lyrics-count").style.color = target === "suno" && l.value.length > 5000 ? "var(--bad)" : "";
}

export function init() {
  $$("#w-target button").forEach((b) => b.addEventListener("click", () => setTarget(b.dataset.t)));
  let saved = "ace";
  try { saved = localStorage.getItem("wb.target") || "ace"; } catch {}
  setTarget(saved);

  $("#w-write").addEventListener("click", (e) => busy(e.currentTarget, async () => {
    const idea = $('[name="idea"]', root).value;
    if (!idea.trim()) { toast("Type an idea first, or press Surprise me"); return; }
    render(await api("/api/write", { method: "POST", body: { idea, target, opts: opts() } }));
  }));
  $("#w-surprise").addEventListener("click", (e) => busy(e.currentTarget, async () => {
    const { idea } = await api("/api/write/idea", { method: "POST", body: {} });
    $('[name="idea"]', root).value = idea;
  }));
  $("#w-revise").addEventListener("click", (e) => busy(e.currentTarget, async () => {
    const instruction = $("#w-instruction").value.trim();
    if (!instruction) { toast("Say what to change, e.g. 'shorter chorus'"); return; }
    render(await api("/api/write/revise", { method: "POST", body: { bundle: current(), instruction, opts: opts() } }));
    $("#w-instruction").value = "";
  }));
  $("#w-to-suno").addEventListener("click", (e) => busy(e.currentTarget, async () =>
    render(await api("/api/write/convert", { method: "POST", body: { bundle: current(), to: "suno", opts: opts() } }))));
  $("#w-to-ace").addEventListener("click", (e) => busy(e.currentTarget, async () =>
    render(await api("/api/write/convert", { method: "POST", body: { bundle: current(), to: "ace", opts: opts() } }))));
  $("#w-send").addEventListener("click", () => { emit("load-create", current()); showTab("create"); });
  $("#w-copy-style").addEventListener("click", () => copy($('[name="style"]', res).value));
  $("#w-copy-exclude").addEventListener("click", () => copy($('[name="exclude"]', res).value));
  $("#w-copy-lyrics").addEventListener("click", () => copy($('[name="lyrics"]', res).value));
  $("#w-copy-title").addEventListener("click", () => copy($('[name="title"]', res).value));
  $("#w-copy-all").addEventListener("click", () => {
    const b = current();
    copy(`Title: ${b.title}\n\nStyle: ${b.style}\n\nExclude: ${b.exclude}\n\nLyrics:\n${b.lyrics}`);
  });
  res.addEventListener("input", counts);

  try {
    const prev = JSON.parse(localStorage.getItem("wb.bundle") || "null");
    if (prev) render(prev);
  } catch {}
}
