// Create tab: generation form, job progress, result cards.
import { $, $$, api, busy, toast, readForm, fillForm, emit, on, showTab, el, fmtTime } from "./core.js";

const form = $("#c-form");
let loras = [];

export async function loadLoras() {
  try {
    loras = (await api("/api/loras")).loras;
  } catch { loras = []; }
  for (const sel of [$("#c-lora"), $("#s-lora")]) {  // Create and Studio share the list
    const cur = sel.value;
    sel.replaceChildren(el("option", { value: "" }, "None"),
      ...loras.map((l) => el("option", { value: l.path }, l.label + (l.trigger ? `  [${l.trigger}]` : "") + (l.model_label ? `  · ${l.model_label}` : ""))));
    sel.value = cur;
  }
  loraHint();
}

export const hasLora = (path) => loras.some((l) => l.path === path);

// A LoRA trained here has a trigger word; put it at the start of the caption if it's missing.
export function withTrigger(caption, path) {
  const l = loras.find((x) => x.path === path);
  return l && l.trigger && !caption.toLowerCase().includes(l.trigger.toLowerCase()) ? `${l.trigger}, ${caption}` : caption;
}

// A saved song's full settings for the Create form: same seed + same settings = the same song.
export function songForm(song) {
  const f = song.form || {};
  return {
    ...f, title: song.title, caption: f.caption ?? song.caption ?? "", lyrics: f.lyrics ?? song.lyrics ?? "",
    seed: song.seed, batch_size: 1, lora: song.lora || "", lora_scale: song.lora_scale ?? f.lora_scale,
  };
}

let currentModel = "";  // the DiT model ACE-Step runs (from /api/status)

function loraHint() {
  const l = loras.find((x) => x.path === $("#c-lora").value);
  const other = l && l.model_id && currentModel && l.model_id !== currentModel;
  $("#c-lora-hint").textContent = l
    ? (l.trigger ? `Trigger word: "${l.trigger}" (added to the start of your caption automatically).` : "") +
      (l.model_label ? ` Made for ${l.model_label}.` : "") +
      (other ? ` Generating switches ACE-Step to ${l.model_label} first (one restart).` : "")
    : "";
}

function liveValues(root) {
  $$('input[type="range"]', root).forEach((r) => {
    const out = root.querySelector(`[data-v="${r.name}"]`) || (r.name === "lora_scale" ? $("#c-lora-scale-v") : null);
    const upd = () => { if (out) out.textContent = r.value; };
    r.addEventListener("input", upd);
    upd();
  });
}

export function collect() {
  const f = readForm(form);
  f.caption = withTrigger(f.caption, f.lora);
  return f;
}

// Poll a job and render its songs into a results container.
export async function followJob(jobId, ui) {
  ui.box.classList.remove("hidden");
  for (;;) {
    await new Promise((r) => setTimeout(r, 1200));
    let j;
    try { j = await api(`/api/jobs/${jobId}`); } catch (e) { ui.stage.textContent = e.message; continue; }
    ui.stage.textContent = j.stage || j.status;
    if (ui.elapsed) ui.elapsed.textContent = j.elapsed ? `${Math.round(j.elapsed)}s` : "";
    if (j.status === "done") {
      ui.box.classList.add("hidden");
      for (const s of j.songs) ui.results.prepend(await songCard(s.id));
      emit("library-changed");
      return j;
    }
    if (j.status === "error") {
      ui.box.classList.add("hidden");
      toast(j.error || "Generation failed", true, 10000);
      return j;
    }
  }
}

export async function songCard(id) {
  const m = await api(`/api/songs/${id}`);
  const kind = { new: "", remix: "same-seed remix", variation: "variation", repaint: "re-sung section", import: "imported", test: "LoRA test clip" }[m.kind] || m.kind;
  return el("div", { class: "song" },
    el("div", { class: "row" },
      el("b", {}, m.title || "Untitled"),
      kind ? el("span", { class: "muted small" }, `· ${kind}`) : null,
      el("span", { class: "spacer" }),
      el("span", { class: "meta" }, `seed ${m.seed} · ${m.bpm ?? "?"} BPM · ${m.keyscale || "?"} · ${fmtTime(Number(m.duration))}`)),
    el("audio", { controls: true, preload: "none", src: `/api/songs/${id}/audio` }),
    el("div", { class: "row" },
      el("button", { class: "b small", onclick: () => { emit("open-studio", id); showTab("studio"); }, "data-tip": "Regenerate with this seed, change words, switch LoRA or re-sing a part." }, "🎚️ Studio"),
      el("button", { class: "b small", onclick: () => { emit("load-create", songForm(m)); showTab("create"); }, "data-tip": "Put this song's exact settings and seed back in the Create form." }, "↺ Load settings"),
      el("a", { class: "b small", href: `/api/songs/${id}/audio?download=1`, style: "text-decoration:none" }, "⬇ Download")));
}

export function init() {
  liveValues(form);
  $("#c-lora").addEventListener("change", loraHint);
  $("#c-lora").addEventListener("input", loraHint);
  $("#c-dice").addEventListener("click", () => { $('[name="seed"]', form).value = Math.floor(Math.random() * 2147483647); });
  $("#c-clear").addEventListener("click", () => {
    fillForm(form, { title: "", caption: "", lyrics: "", bpm: "", keyscale: "", timesignature: "", duration: "", vocal_language: "", instrumental: false, seed: -1 });
  });
  $("#c-generate").addEventListener("click", (e) => busy(e.currentTarget, async () => {
    const f = collect();
    if (!f.caption.trim()) { toast("Add a caption: describe the sound you want."); return; }
    const { job } = await api("/api/generate", { method: "POST", body: f });
    followJob(job, { box: $("#c-job"), stage: $("#c-stage"), elapsed: $("#c-elapsed"), results: $("#c-results") });
  }));

  // Accepts a Write bundle or a saved song's full form: every key that matches a Create field is filled
  // (steps, sampler, thinking, LM settings, format...), so a seed reproduces exactly.
  on("load-create", async (b) => {
    if (b.lora && !hasLora(b.lora)) await loadLoras();  // a LoRA folder added since the page loaded
    const data = { ...b, caption: b.caption || b.style || "", instrumental: !!b.instrumental };
    for (const k of ["bpm", "keyscale", "timesignature", "duration", "vocal_language"]) data[k] = b[k] ?? "";
    if (b.seed == null) delete data.seed;
    if (b.lora == null) { delete data.lora; delete data.lora_scale; }
    else data.lora_scale = b.lora_scale ?? 0.8;
    fillForm(form, data);
    loraHint();
  });
  on("tab", (t) => { if (t === "create") loadLoras(); });
  on("status", (s) => {
    if (s.ace.model && s.ace.model !== currentModel) { currentModel = s.ace.model; loraHint(); }
    $("#c-ready").textContent = !s.ace.reachable ? "ACE-Step is not running (Settings)." : !s.ace.ready ? "ACE-Step is loading models..." : "";
  });
  loadLoras();
}
