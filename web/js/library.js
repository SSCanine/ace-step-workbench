// Library tab: list, search, play, open in Studio, download, delete (moves to data/trash).
import { $, api, toast, emit, on, showTab, el } from "./core.js";

const player = $("#l-player");
let timer;

async function load() {
  const { songs } = await api(`/api/songs?q=${encodeURIComponent($("#l-q").value)}`);
  const kinds = { new: "new", remix: "same seed", variation: "variation", repaint: "re-sung" };
  const rows = songs.map((s) => el("tr", {},
    el("td", {}, el("b", {}, s.title || "Untitled"), el("div", { class: "muted small" }, (s.caption || "").slice(0, 90))),
    el("td", { class: "small" }, new Date(s.created * 1000).toLocaleString()),
    el("td", { class: "mono small" }, String(s.seed ?? "")),
    el("td", { class: "small" }, kinds[s.kind] || s.kind || ""),
    el("td", { class: "small" }, `${s.bpm ?? "-"} / ${s.keyscale || "-"}`),
    el("td", {}, el("div", { class: "row", style: "flex-wrap:nowrap" },
      el("button", { class: "b small", title: "Play", onclick: () => { player.classList.remove("hidden"); player.src = `/api/songs/${s.id}/audio`; player.play(); } }, "▶"),
      el("button", { class: "b small", onclick: () => { emit("open-studio", s.id); showTab("studio"); } }, "Studio"),
      el("a", { class: "b small", href: `/api/songs/${s.id}/audio?download=1`, style: "text-decoration:none" }, "⬇"),
      el("button", { class: "b small danger", title: "Move to trash (data/trash)", onclick: async () => {
        if (!confirm(`Move "${s.title || "Untitled"}" to the trash folder?`)) return;
        await api(`/api/songs/${s.id}`, { method: "DELETE" });
        toast("Moved to data/trash");
        load();
      } }, "🗑")))));
  $("#l-table tbody").replaceChildren(...(rows.length ? rows : [el("tr", {}, el("td", { colspan: "6", class: "muted" }, "No songs yet. Make one in Create."))]));
}

export function init() {
  $("#l-refresh").addEventListener("click", load);
  $("#l-q").addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(load, 250); });
  on("tab", (t) => { if (t === "library") load(); });
  on("library-changed", () => { if ($("#tab-library").classList.contains("active")) load(); });
}
