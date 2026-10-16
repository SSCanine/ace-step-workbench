// Shared helpers: API calls, toasts, tabs, small DOM utils.
export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

export async function api(path, { method = "GET", body, form } = {}) {
  const opts = { method, headers: { "X-Workbench": "1" } };
  if (form) opts.body = form;
  else if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const r = await fetch(path, opts);
  let data = null;
  try { data = await r.json(); } catch { /* empty body */ }
  if (!r.ok) throw new Error((data && (data.detail || data.error)) || `HTTP ${r.status}`);
  return data;
}

let toastTimer;
export function toast(msg, bad = false, ms = 4500) {
  let el = $(".toast");
  if (!el) { el = document.createElement("div"); el.className = "toast"; el.setAttribute("role", "status"); document.body.append(el); }
  el.textContent = msg;
  el.classList.toggle("bad", bad);
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, ms);
}

export async function busy(btn, fn) {
  const label = btn.textContent;
  btn.disabled = true;
  btn.textContent = btn.dataset.busy || "Working...";
  try { return await fn(); }
  catch (e) { toast(e.message, true, 8000); throw e; }
  finally { btn.disabled = false; btn.textContent = label; }
}

export function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v === true) e.setAttribute(k, "");
    else if (v !== false && v != null) e.setAttribute(k, v);
  }
  for (const k of kids.flat()) if (k != null) e.append(k.nodeType ? k : document.createTextNode(k));
  return e;
}

export const bus = new EventTarget();
export const emit = (name, detail) => bus.dispatchEvent(new CustomEvent(name, { detail }));
export const on = (name, fn) => bus.addEventListener(name, (e) => fn(e.detail));

export function showTab(name) {
  $$(".nav button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  $$(".page").forEach((p) => p.classList.toggle("active", p.id === `tab-${name}`));
  try { localStorage.setItem("wb.tab", name); } catch { /* storage blocked */ }
  emit("tab", name);
  window.scrollTo(0, 0);
}

export function fmtTime(sec) {
  if (sec == null || isNaN(sec)) return "-";
  const m = Math.floor(sec / 60), s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

export async function copy(text) {
  try { await navigator.clipboard.writeText(text); toast("Copied"); }
  catch { toast("Copy failed: select the text and press Ctrl+C", true); }
}

// Form <-> object for elements with name= inside a root.
export function readForm(root) {
  const out = {};
  for (const e of $$("[name]", root)) {
    if (e.type === "checkbox") out[e.name] = e.checked;
    else if (e.type === "number" || e.type === "range") out[e.name] = e.value === "" ? null : Number(e.value);
    else out[e.name] = e.value;
  }
  return out;
}
export function fillForm(root, obj) {
  for (const [k, v] of Object.entries(obj || {})) {
    const e = $(`[name="${k}"]`, root);
    if (!e) continue;
    if (e.type === "checkbox") e.checked = !!v;
    else e.value = v ?? "";
    e.dispatchEvent(new Event("input", { bubbles: true }));
  }
}
