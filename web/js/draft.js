// Autosave: what you type in Write, Create and Studio is saved to the Workbench (data/draft.json) as you go and
// restored when the page opens, so a reload, a crash or a restart never loses work. First run: the newest song.
import { $, api, readForm, fillForm, emit, on } from "./core.js";
import * as studio from "./studio.js";
import { loadLoras, songForm } from "./create.js";

let timer = null;
let live = false;  // off while restoring, so a half-restored page is never saved

function snapshot() {
  return {
    v: 1, saved: Date.now(),
    tab: $(".nav button.active")?.dataset.tab || "",
    create: readForm($("#c-form")),
    write: { brief: readForm($("#w-brief")), result: readForm($("#w-result")), shown: !$("#w-result").hidden },
    studio: studio.draftState(),
  };
}

function save(keepalive = false) {
  if (!live) return;
  clearTimeout(timer);
  timer = null;
  fetch("/api/draft", {
    method: "PUT", keepalive, body: JSON.stringify(snapshot()),
    headers: { "X-Workbench": "1", "Content-Type": "application/json" },
  }).catch(() => { /* next change saves again */ });
}

function touch() {
  if (!live) return;
  clearTimeout(timer);
  timer = setTimeout(save, 700);
}

// Restore the last session (or start from the newest song). Returns the tab to show, or "".
export async function restore() {
  let d = null;
  try { d = (await api("/api/draft")).draft; } catch { /* no draft yet */ }
  await loadLoras();  // the LoRA dropdowns need their options before a saved choice can be selected
  if (d) {
    if (d.create) fillForm($("#c-form"), d.create);
    if (d.write) {
      fillForm($("#w-brief"), d.write.brief);
      fillForm($("#w-result"), d.write.result);
      if (d.write.shown) $("#w-result").hidden = false;
    }
    if (d.studio?.song) await studio.restoreDraft(d.studio).catch(() => { /* song was deleted */ });
  } else {
    try {
      const { songs } = await api("/api/songs");
      if (songs.length) emit("load-create", songForm(await api(`/api/songs/${songs[0].id}`)));
    } catch { /* empty library */ }
  }
  return d?.tab || "";
}

export function start() {
  live = true;
  document.addEventListener("input", touch, true);
  document.addEventListener("change", touch, true);
  on("tab", touch);
  on("draft-changed", touch);
  document.addEventListener("visibilitychange", () => { if (document.visibilityState === "hidden") save(true); });
  window.addEventListener("pagehide", () => save(true));
}
