// Boot: navigation, status polling, guide, theme.
import { $, $$, api, showTab, emit, on } from "./core.js";
import * as write from "./write.js";
import * as create from "./create.js";
import * as studio from "./studio.js";
import * as library from "./library.js";
import * as train from "./train.js";
import * as settings from "./settings.js";
import * as draft from "./draft.js";
import * as dropzone from "./dropzone.js";

function applyTheme(t) {
  try { localStorage.setItem("wb.theme", t); } catch {}
  const sys = window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  document.documentElement.dataset.theme = t || sys;
}

async function loadGuide() {
  const html = await (await fetch("/static/guide.html")).text();
  $("#guide-body").innerHTML = html;
  $("#guide-body").classList.remove("muted");
  $("#g-create-params").innerHTML = paramTable();
  $$("#guide-body .toc a").forEach((a) => a.addEventListener("click", (e) => {
    e.preventDefault();
    document.getElementById(a.getAttribute("href").slice(1))?.scrollIntoView({ behavior: "smooth" });
  }));
}

// Build the parameter reference from the Create form's own tooltips, so it never drifts.
function paramTable() {
  const rows = $$("#c-form label.f").map((l) => {
    const name = l.querySelector("span")?.childNodes[0]?.textContent.trim();
    const tip = l.querySelector(".tip")?.dataset.tip;
    return name && tip ? `<tr><td><b>${name}</b></td><td>${tip}</td></tr>` : "";
  }).join("");
  const checks = $$("#c-form label.check[data-tip]").map((l) => `<tr><td><b>${l.textContent.trim()}</b></td><td>${l.dataset.tip}</td></tr>`).join("");
  return `<table><tr><th>Setting</th><th>What it does</th></tr>${rows}${checks}</table>
    <p>Speed tip: Thinking off and BPM/key/length filled in = fastest drafts. Turn Thinking on for final takes.</p>`;
}

async function pollStatus() {
  try {
    const s = await api("/api/status");
    const dot = $("#st-ace");
    dot.className = "dot " + (s.ace.ready ? "ok" : s.ace.reachable || s.ace.managed ? "warn" : "");
    $("#st-ace-text").textContent = s.ace.ready ? "ACE-Step ready" : s.ace.reachable || s.ace.managed ? "ACE-Step loading" : s.ace.installed ? "ACE-Step stopped" : "ACE-Step not installed";
    $("#st-writer").textContent = `Writer: ${s.writer.label.replace(/ \(.*\)$/, "")}`;
    $("#w-writer").textContent = s.writer.label + (s.writer.model ? ` · ${s.writer.model}` : "");
    const tr = s.training.status, q = s.queue || {};
    const trainingNow = ["preparing", "preprocessing", "training"].includes(tr);
    $("#st-train-pill").classList.toggle("hidden", !trainingNow && !q.active);
    $("#st-train").textContent = tr === "training" ? `Training ${s.training.epoch || 0}/${s.training.epochs || "?"}`
      : trainingNow ? "Training: preparing"
      : q.status === "waiting" ? "Queue: waiting for the GPU" : `Queue: ${q.stage || q.status}`;
    emit("status", s);
  } catch {
    $("#st-ace").className = "dot bad";
    $("#st-ace-text").textContent = "Workbench offline";
  }
}

async function boot() {
  let theme = "";
  try { theme = localStorage.getItem("wb.theme") || ""; } catch {}
  $("#set-theme").value = theme;
  applyTheme(theme);
  $("#set-theme").addEventListener("change", (e) => applyTheme(e.target.value));

  let picked = false;  // a tab clicked while the page is still loading wins over the restored one
  $$(".nav button").forEach((b) => b.addEventListener("click", () => { picked = true; showTab(b.dataset.tab); }));
  document.addEventListener("click", (e) => {
    const a = e.target.closest("[data-goto]");
    if (a) { e.preventDefault(); showTab(a.dataset.goto); }
  });

  write.init(); create.init(); studio.init(); library.init(); settings.init(); dropzone.init();
  try { await train.init(); } catch (e) { console.error("Train tab failed to load", e); }
  try { await loadGuide(); } catch (e) { $("#guide-body").textContent = "Could not load the guide."; }

  let first = "guide";
  try { first = localStorage.getItem("wb.tab") || (localStorage.getItem("wb.seen") ? "write" : "guide"); localStorage.setItem("wb.seen", "1"); } catch {}
  let restored = "";
  try { restored = await draft.restore(); } catch (e) { console.error("Could not restore the last session", e); }
  if (!picked) showTab(restored || first);
  draft.start();
  pollStatus();
  setInterval(pollStatus, 3000);
}

boot();
