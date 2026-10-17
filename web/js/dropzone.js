// Drop a song file anywhere: its caption, lyrics, LoRA and settings load into Create
// (on the Studio tab it opens in Studio instead). Also works with the drop box's file picker.
import { $, api, toast, emit, showTab, el } from "./core.js";

const zone = $("#c-drop");
const note = $("#c-import-note");
const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes("Files");

async function handle(files) {
  if (!files.length) return;
  const toStudio = $("#tab-studio").classList.contains("active");
  const fd = new FormData();
  for (const f of files) fd.append("files", f, f.name);
  const label = $("b", zone);
  const text = label.textContent;
  zone.classList.add("busy");
  label.textContent = "Reading the song...";
  try {
    const r = await api("/api/import", { method: "POST", form: fd });
    if (r.song_id) emit("library-changed");
    if (toStudio && r.song_id) {
      emit("open-studio", r.song_id);
      toast(`Opened "${r.title}" (${r.source})`);
      return;
    }
    emit("load-create", r.form);
    showTab("create");
    showNote(r);
  } catch (e) {
    toast(e.message, true, 12000);
  } finally {
    zone.classList.remove("busy");
    label.textContent = text;
  }
}

function showNote(r) {
  const res = r.resolved || {};
  const sig = { 2: "2/4", 3: "3/4", 4: "4/4", 6: "6/8" }[res.timesignature];
  const came = [res.bpm && `${res.bpm} BPM`, res.keyscale, sig, res.duration && `${Math.round(res.duration)} s`]
    .filter(Boolean).join(" · ");
  note.replaceChildren(
    el("div", { class: "row" },
      el("b", {}, `Loaded "${r.title}"`),
      el("span", { class: "muted small" }, `from ${r.source}${r.form.seed != null ? ` · seed ${r.form.seed}` : ""}`),
      el("span", { class: "spacer" }),
      r.song_id ? el("button", { class: "b small", onclick: () => { emit("open-studio", r.song_id); showTab("studio"); } }, "🎚️ Open in Studio") : null,
      el("button", { class: "b small", title: "Dismiss", onclick: () => note.classList.add("hidden") }, "✕")),
    came ? el("div", { class: "small muted" }, `It came out at ${came}. Generate as-is to make the same song again.`) : null,
    ...(r.notes || []).map((n) => el("div", { class: "small" }, n)));
  note.classList.remove("hidden");
}

export function init() {
  const input = $("#c-drop-file");
  zone.addEventListener("click", () => input.click());
  zone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  input.addEventListener("change", () => { handle([...input.files]); input.value = ""; });

  // Catch file drops on the whole page: a missed drop would otherwise open the file and leave the app.
  let depth = 0;
  const done = () => { depth = 0; document.body.classList.remove("dragging"); };
  document.addEventListener("dragenter", (e) => { if (hasFiles(e)) { depth++; document.body.classList.add("dragging"); } });
  document.addEventListener("dragleave", (e) => { if (hasFiles(e) && --depth <= 0) done(); });
  document.addEventListener("dragover", (e) => { if (hasFiles(e)) { e.preventDefault(); e.dataTransfer.dropEffect = "copy"; } });
  document.addEventListener("drop", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    done();
    if ($("#tab-train").classList.contains("active")) { toast("To add training audio, use + Add files in step 1."); return; }
    handle([...e.dataTransfer.files]);
  });
}
