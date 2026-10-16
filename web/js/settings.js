// Settings tab: ACE-Step engine, AI writer provider, folders, interface.
import { $, $$, api, busy, toast, readForm, fillForm, on, el } from "./core.js";

let presets = {};
let logTimer = null;

function showProviderFields(key) {
  const p = presets[key] || {};
  const kind = p.kind;
  $$(".set-url").forEach((e) => e.classList.toggle("hidden", kind !== "openai"));
  $$(".set-model").forEach((e) => e.classList.toggle("hidden", ["none", "ace-lm"].includes(kind)));
  $$(".set-key").forEach((e) => e.classList.toggle("hidden", !(p.needs_key || key === "custom" || key === "openrouter")));
  $$(".set-temp, .set-json").forEach((e) => e.classList.toggle("hidden", kind !== "openai"));
  $$(".set-effort").forEach((e) => e.classList.toggle("hidden", kind !== "anthropic"));
}

async function load() {
  const r = await api("/api/settings");
  presets = r.presets;
  const s = r.settings;
  $("#set-dit").replaceChildren(...r.dit_models.map((m) => el("option", { value: m.id }, m.label)));
  $("#set-lm").replaceChildren(...r.lm_models.map((m) => el("option", { value: m.id }, m.label)));
  fillForm($("#set-ace"), s.ace);
  $("#set-preset").replaceChildren(...Object.entries(presets).map(([k, p]) => el("option", { value: k }, p.label)));
  fillForm($("#set-llm"), s.llm);
  showProviderFields(s.llm.preset);
  fillForm($("#set-train"), { ...s.training, extra_lora_dirs: (s.training.extra_lora_dirs || []).join("\n") });
  const cli = [];
  cli.push(r.cli.claude ? "Claude Code CLI found" : "Claude Code CLI not installed");
  cli.push(r.cli.codex ? "Codex CLI found" : "Codex CLI not installed");
  $("#set-cli").textContent = cli.join(" · ");
  $("#set-tips").checked = s.ui.show_tips !== false;
  document.body.classList.toggle("no-tips", s.ui.show_tips === false);
}

async function checkAceFolder() {
  const msg = $("#set-ace-check-msg");
  try {
    const r = await api(`/api/ace/check?path=${encodeURIComponent($('#set-ace [name="path"]').value.trim())}`);
    msg.textContent = (r.ok ? "✓ " : "✗ ") + r.message;
    msg.style.color = r.ok ? "var(--good)" : "var(--bad)";
    return r.ok;
  } catch (e) { msg.textContent = e.message; return false; }
}

async function saveAce() {
  const f = readForm($("#set-ace"));
  f.path = (f.path || "").trim();
  await api("/api/settings", { method: "POST", body: { ace: f } });
  await checkAceFolder();
}

function tailLog() {
  clearInterval(logTimer);
  logTimer = setInterval(async () => {
    if (!$("#tab-settings").classList.contains("active")) { clearInterval(logTimer); return; }
    try {
      const { lines } = await api("/api/ace/log?n=300");
      const box = $("#set-ace-log");
      const atBottom = box.scrollTop + box.clientHeight >= box.scrollHeight - 20;
      box.textContent = lines.join("\n");
      if (atBottom) box.scrollTop = box.scrollHeight;
    } catch { /* ignore */ }
  }, 2000);
}

export function init() {
  $("#set-ace-check").addEventListener("click", (e) => { e.preventDefault(); checkAceFolder(); });
  $('#set-ace [name="path"]').addEventListener("change", checkAceFolder);
  const temp = $('#set-llm [name="temperature"]');
  temp.addEventListener("input", () => { $('#set-llm [data-v="temperature"]').textContent = temp.value; });
  $("#set-clear-key").addEventListener("click", async (e) => {
    e.preventDefault();
    const preset = $("#set-preset").value;
    await api("/api/settings", { method: "POST", body: { llm: { preset, clear_key: true } } });
    await load();
    toast("Key removed for this provider");
  });
  $("#set-preset").addEventListener("change", (e) => {
    const p = presets[e.target.value] || {};
    fillForm($("#set-llm"), { base_url: p.base_url || "", model: p.model || "", api_key: "" });
    showProviderFields(e.target.value);
  });
  $("#set-llm-save").addEventListener("click", (e) => busy(e.currentTarget, async () => {
    const f = readForm($("#set-llm"));
    f.kind = (presets[f.preset] || {}).kind || "openai";
    await api("/api/settings", { method: "POST", body: { llm: f } });
    await load();
    toast("AI writer saved");
  }));
  $("#set-llm-test").addEventListener("click", (e) => busy(e.currentTarget, async () => {
    const f = readForm($("#set-llm"));
    f.kind = (presets[f.preset] || {}).kind || "openai";
    await api("/api/settings", { method: "POST", body: { llm: f } });
    $("#set-llm-result").textContent = "";
    try {
      const r = await api("/api/llm/test", { method: "POST" });
      $("#set-llm-result").textContent = `✓ Works. Reply: ${r.reply}`;
      $("#set-llm-result").style.color = "var(--good)";
    } catch (err) {
      $("#set-llm-result").textContent = `✗ ${err.message}`;
      $("#set-llm-result").style.color = "var(--bad)";
    }
  }));
  $("#set-list-models").addEventListener("click", async (e) => {
    e.preventDefault();
    const f = readForm($("#set-llm"));
    await api("/api/settings", { method: "POST", body: { llm: { ...f, kind: (presets[f.preset] || {}).kind } } });
    try {
      const { models } = await api("/api/llm/models");
      $("#set-models").replaceChildren(...models.map((m) => el("option", { value: m })));
      toast(models.length ? `${models.length} models found. Click the Model box to pick one.` : "No models listed");
    } catch (err) { toast(err.message, true); }
  });

  $("#set-ace-save").addEventListener("click", (e) => busy(e.currentTarget, async () => { await saveAce(); toast("Saved. Restart ACE-Step to apply model changes."); }));
  $("#set-ace-start").addEventListener("click", (e) => busy(e.currentTarget, async () => { await saveAce(); await api("/api/ace/start", { method: "POST" }); toast("Starting ACE-Step (first start can take minutes)"); }));
  $("#set-ace-restart").addEventListener("click", (e) => busy(e.currentTarget, async () => { await saveAce(); await api("/api/ace/restart", { method: "POST" }); toast("Restarting ACE-Step"); }));
  $("#set-ace-stop").addEventListener("click", (e) => busy(e.currentTarget, async () => { await api("/api/ace/stop", { method: "POST" }); toast("ACE-Step stopped"); }));

  $("#set-train-save").addEventListener("click", (e) => busy(e.currentTarget, async () => {
    const f = readForm($("#set-train"));
    f.extra_lora_dirs = f.extra_lora_dirs.split("\n").map((x) => x.trim()).filter(Boolean);
    await api("/api/settings", { method: "POST", body: { training: f } });
    toast("Saved");
  }));
  $("#set-tips").addEventListener("change", async (e) => {
    document.body.classList.toggle("no-tips", !e.target.checked);
    await api("/api/settings", { method: "POST", body: { ui: { show_tips: e.target.checked } } });
  });

  on("status", (s) => {
    const a = s.ace;
    const msg = !a.installed ? `Not installed at ${a.path}. Run install.bat, or set the folder of an existing ACE-Step 1.5 install below.`
      : a.ready ? `Running · ${a.model || ""}${a.llm ? ` + ${a.lm || "LM"}` : " (LM off)"}${a.managed ? "" : " · started outside the Workbench"}`
      : a.reachable ? "Loading models..." : a.managed ? "Starting..." : `Stopped${a.error ? `: ${a.error}` : ""}`;
    $("#set-ace-status").textContent = msg;
  });
  on("tab", (t) => { if (t === "settings") { load(); tailLog(); } });
  load();
}
