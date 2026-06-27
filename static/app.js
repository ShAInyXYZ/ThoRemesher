import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

// --------------------------------------------------------------------------- //
//  State
// --------------------------------------------------------------------------- //
const state = {
  sessionId: null,
  name: "",
  curvature: false,
  wireframe: false,
  sync: true,
  syncing: false,
};

const loader = new GLTFLoader();

// --------------------------------------------------------------------------- //
//  Viewer (one per side)
// --------------------------------------------------------------------------- //
function createViewer(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0d12);

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1e5);
  camera.position.set(3, 2, 5);

  const grid = new THREE.GridHelper(20, 40, 0x223040, 0x182230);
  grid.material.transparent = true;
  grid.material.opacity = 0.5;
  scene.add(grid);

  scene.add(new THREE.HemisphereLight(0xbfd8ff, 0x202830, 1.0));
  const key = new THREE.DirectionalLight(0xffffff, 1.6);
  key.position.set(5, 8, 6);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0x88aaff, 0.6);
  fill.position.set(-6, 3, -4);
  scene.add(fill);

  const holder = new THREE.Group();
  scene.add(holder);

  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;

  return { canvas, renderer, scene, camera, controls, holder, meshes: [] };
}

const left = createViewer(document.getElementById("canvas-left"));
const right = createViewer(document.getElementById("canvas-right"));
const viewers = { left, right };
window.__viewers = viewers; // debug hook

function resizeViewer(v) {
  const r = v.canvas.getBoundingClientRect();
  const w = Math.max(2, r.width), h = Math.max(2, r.height);
  v.renderer.setSize(w, h, false);
  v.camera.aspect = w / h;
  v.camera.updateProjectionMatrix();
}
function resizeAll() {
  resizeViewer(left);
  resizeViewer(right);
}
window.addEventListener("resize", resizeAll);

// --------------------------------------------------------------------------- //
//  Linked cameras
// --------------------------------------------------------------------------- //
function copyView(src, dst) {
  dst.camera.position.copy(src.camera.position);
  dst.controls.target.copy(src.controls.target);
  dst.camera.zoom = src.camera.zoom;
  dst.camera.updateProjectionMatrix();
}
left.controls.addEventListener("change", () => {
  if (!state.sync || state.syncing) return;
  state.syncing = true;
  copyView(left, right);
  right.controls.update();
  state.syncing = false;
});
right.controls.addEventListener("change", () => {
  if (!state.sync || state.syncing) return;
  state.syncing = true;
  copyView(right, left);
  left.controls.update();
  state.syncing = false;
});

// --------------------------------------------------------------------------- //
//  Render loop
// --------------------------------------------------------------------------- //
function animate() {
  requestAnimationFrame(animate);
  left.controls.update();
  right.controls.update();
  left.renderer.render(left.scene, left.camera);
  right.renderer.render(right.scene, right.camera);
}
resizeAll();
animate();

// --------------------------------------------------------------------------- //
//  Mesh loading into a viewer
// --------------------------------------------------------------------------- //
function clearHolder(v) {
  for (const o of [...v.holder.children]) {
    v.holder.remove(o);
    o.traverse?.((c) => {
      c.geometry?.dispose?.();
      if (c.material) (Array.isArray(c.material) ? c.material : [c.material]).forEach((m) => m.dispose());
    });
  }
  v.meshes = [];
}

function applyMaterials(v) {
  for (const m of v.meshes) {
    const hasColor = !!m.geometry.getAttribute("color");
    if (!m.geometry.getAttribute("normal")) m.geometry.computeVertexNormals();
    const mat = new THREE.MeshStandardMaterial({
      color: 0xbfc8d2,
      metalness: 0.05,
      roughness: 0.78,
      flatShading: false,
      vertexColors: state.curvature && hasColor,
      wireframe: state.wireframe,
      side: THREE.DoubleSide,
    });
    m.material = mat;
  }
}

async function loadModel(v, url) {
  const buf = await (await fetch(url)).arrayBuffer();
  const gltf = await loader.parseAsync(buf, "");
  clearHolder(v);
  const root = gltf.scene;
  root.traverse((c) => {
    if (c.isMesh) v.meshes.push(c);
  });
  v.holder.add(root);
  applyMaterials(v);
  fitView(v);
}

function fitView(v) {
  if (!v.meshes.length) return;
  const box = new THREE.Box3().setFromObject(v.holder);
  if (box.isEmpty()) return;
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  const center = sphere.center;
  const radius = Math.max(sphere.radius, 1e-4);
  const fov = (v.camera.fov * Math.PI) / 180;
  const dist = (radius / Math.sin(fov / 2)) * 1.15;
  v.camera.near = dist / 200;
  v.camera.far = dist * 200;
  v.camera.position.copy(center).add(new THREE.Vector3(dist * 0.5, dist * 0.42, dist));
  v.camera.lookAt(center);
  v.controls.target.copy(center);
  v.controls.maxDistance = dist * 12;
  v.controls.minDistance = dist / 40;
  v.controls.update();

  const gSize = radius * 4;
  left.scene.children.forEach((c) => { if (c.isGridHelper) { c.scale.setScalar(gSize / 20); c.position.set(center.x, box.min.y, center.z); } });
  right.scene.children.forEach((c) => { if (c.isGridHelper) { c.scale.setScalar(gSize / 20); c.position.set(center.x, box.min.y, center.z); } });

  if (state.sync) {
    state.syncing = true;
    copyView(v, v === left ? right : left);
    (v === left ? right : left).controls.update();
    state.syncing = false;
  }
}

// --------------------------------------------------------------------------- //
//  Upload
// --------------------------------------------------------------------------- //
async function uploadFile(file) {
  setStatus(`Uploading ${file.name} …`);
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch("/api/upload", { method: "POST", body: fd });
  if (!res.ok) throw new Error((await res.text()) || "upload failed");
  const data = await res.json();
  state.sessionId = data.session_id;
  state.name = data.name;
  document.getElementById("st-orig").textContent = fmtTri(data.orig.faces);
  document.getElementById("st-proc").textContent = "—";
  document.getElementById("st-red").textContent = "—";
  document.getElementById("st-time").textContent = "—";
  document.getElementById("empty-right").style.display = "";
  setStatus(`Loaded ${data.name} — ${fmtTri(data.orig.faces)} tris`);
  await loadModel(left, `/api/model/${state.sessionId}?which=orig&color=1`);
  applyMaterials(left);
  document.getElementById("btn-remesh").disabled = false;
}

// --------------------------------------------------------------------------- //
//  Remesh
// --------------------------------------------------------------------------- //
async function doRemesh() {
  if (!state.sessionId) return;
  const btn = document.getElementById("btn-remesh");
  btn.disabled = true;
  document.body.classList.add("busy");
  setStatus("Remeshing …", true);
  try {
    const body = {
      session_id: state.sessionId,
      flat_factor: +val("flat_factor"),
      detail_factor: +val("detail_factor"),
      contrast: +val("contrast"),
      feature_angle: +val("feature_angle"),
      iterations: +val("iterations"),
      pre_simplify: document.getElementById("pre_simplify").checked,
      colorize: document.getElementById("colorize").checked,
      preserve_boundary: document.getElementById("preserve_boundary").checked,
      max_work_faces: +val("max_work_faces"),
    };
    const res = await fetch("/api/remesh", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error((await res.text()) || "remesh failed");
    const s = await res.json();
    document.getElementById("st-orig").textContent = fmtTri(s.orig.faces);
    document.getElementById("st-proc").textContent = fmtTri(s.proc.faces);
    document.getElementById("st-red").textContent = s.face_reduction_pct + "%";
    document.getElementById("st-time").textContent = (s.elapsed_ms / 1000).toFixed(2) + "s";
    document.getElementById("empty-right").style.display = "none";
    await loadModel(right, `/api/model/${state.sessionId}?which=proc&color=1`);
    document.getElementById("btn-export").disabled = false;
    setStatus(`Done — ${fmtTri(s.proc.faces)} tris (-${s.face_reduction_pct}%) in ${(s.elapsed_ms / 1000).toFixed(2)}s`);
  } catch (e) {
    setStatus("Error: " + e.message);
    alert("Remesh failed:\n" + e.message);
  } finally {
    btn.disabled = false;
    document.body.classList.remove("busy");
  }
}

// --------------------------------------------------------------------------- //
//  Toggles
// --------------------------------------------------------------------------- //
function refreshAllMaterials() {
  applyMaterials(left);
  applyMaterials(right);
}
document.getElementById("btn-curv").addEventListener("click", (e) => {
  state.curvature = !state.curvature;
  e.currentTarget.classList.toggle("on", state.curvature);
  refreshAllMaterials();
});
document.getElementById("btn-wire").addEventListener("click", (e) => {
  state.wireframe = !state.wireframe;
  e.currentTarget.classList.toggle("on", state.wireframe);
  refreshAllMaterials();
});
document.getElementById("btn-sync").addEventListener("click", (e) => {
  state.sync = !state.sync;
  e.currentTarget.classList.toggle("on", state.sync);
  if (state.sync) { state.syncing = true; copyView(left, right); right.controls.update(); state.syncing = false; }
});

// --------------------------------------------------------------------------- //
//  Slider readouts
// --------------------------------------------------------------------------- //
const bind = (id, fmt) => {
  const el = document.getElementById(id);
  const out = document.getElementById("v-" + id.replace("_factor", "").replace("feature_angle", "feat").replace("iterations", "iter"));
  const upd = () => { if (out) out.textContent = fmt(+el.value); };
  el.addEventListener("input", upd); upd();
};
bind("flat_factor", (v) => v.toFixed(1) + "×");
bind("detail_factor", (v) => v.toFixed(1) + "×");
bind("contrast", (v) => v.toFixed(1));
bind("feature_angle", (v) => v.toFixed(0) + "°");
bind("iterations", (v) => v.toFixed(0));
{
  const el = document.getElementById("max_work_faces");
  const out = document.getElementById("v-ret");
  const upd = () => { if (out) out.textContent = (+el.value / 1000) + "k"; };
  el.addEventListener("input", upd); upd();
}

document.getElementById("btn-remesh").addEventListener("click", doRemesh);
document.getElementById("btn-open").addEventListener("click", () => document.getElementById("file-input").click());
document.getElementById("file-input").addEventListener("change", (e) => { if (e.target.files[0]) uploadFile(e.target.files[0]); });

// --------------------------------------------------------------------------- //
//  Presets (localStorage)
// --------------------------------------------------------------------------- //
const PRESET_KEY = "remesher_presets_v1";
const BUILTIN_PRESETS = {
  "Balanced (recommended)": { flat_factor: 3, detail_factor: 1, contrast: 2.5, feature_angle: 30, iterations: 4, max_work_faces: 60000, pre_simplify: false, colorize: true, preserve_boundary: true },
  "High detail (characters)": { flat_factor: 2.5, detail_factor: 1, contrast: 3, feature_angle: 25, iterations: 4, max_work_faces: 100000, pre_simplify: false, colorize: true, preserve_boundary: true },
  "Max reduction (game-ready)": { flat_factor: 6, detail_factor: 1.2, contrast: 3, feature_angle: 35, iterations: 3, max_work_faces: 50000, pre_simplify: false, colorize: true, preserve_boundary: true },
  "Aggressive simplify": { flat_factor: 8, detail_factor: 1.5, contrast: 4, feature_angle: 40, iterations: 2, max_work_faces: 40000, pre_simplify: true, colorize: true, preserve_boundary: true },
  "Gentle cleanup": { flat_factor: 2, detail_factor: 0.9, contrast: 2, feature_angle: 20, iterations: 5, max_work_faces: 150000, pre_simplify: false, colorize: true, preserve_boundary: true },
};

function loadCustomPresets() {
  try { return JSON.parse(localStorage.getItem(PRESET_KEY) || "{}"); } catch { return {}; }
}
function refreshPresetSelect() {
  const sel = document.getElementById("preset-select");
  const custom = loadCustomPresets();
  const customNames = Object.keys(custom).sort();
  const cur = sel.value;
  let html = "";
  for (const n of Object.keys(BUILTIN_PRESETS)) {
    html += `<option value="${escapeAttr(n)}">${escapeText(n)}</option>`;
  }
  if (customNames.length) {
    html += '<option disabled>— custom —</option>';
    for (const n of customNames) {
      html += `<option value="${escapeAttr(n)}">${escapeText(n)}</option>`;
    }
  }
  sel.innerHTML = html;
  if (cur) sel.value = cur;
}
function applyPreset(name) {
  const presets = { ...BUILTIN_PRESETS, ...loadCustomPresets() };
  const p = presets[name];
  if (!p) return;
  for (const [k, v] of Object.entries(p)) {
    const el = document.getElementById(k);
    if (!el) continue;
    if (el.type === "checkbox") el.checked = !!v;
    else el.value = v;
  }
  bindRefresh();
  setStatus(`Preset: ${name}`);
}
function collectParams() {
  const ids = ["flat_factor", "detail_factor", "contrast", "feature_angle", "iterations", "max_work_faces"];
  const out = {};
  for (const id of ids) out[id] = +document.getElementById(id).value;
  out.pre_simplify = document.getElementById("pre_simplify").checked;
  out.colorize = document.getElementById("colorize").checked;
  out.preserve_boundary = document.getElementById("preserve_boundary").checked;
  return out;
}

// auto-apply on preset change (no Apply button)
document.getElementById("preset-select").addEventListener("change", (e) => {
  if (e.target.value) applyPreset(e.target.value);
});
// save with icon: auto-name "Custom preset N"
document.getElementById("btn-save-preset").addEventListener("click", () => {
  const custom = loadCustomPresets();
  let n = 1;
  while (custom[`Custom preset ${n}`]) n++;
  const name = `Custom preset ${n}`;
  custom[name] = collectParams();
  localStorage.setItem(PRESET_KEY, JSON.stringify(custom));
  refreshPresetSelect();
  document.getElementById("preset-select").value = name;
  setStatus(`Saved ${name}`);
});
refreshPresetSelect();
// load default preset on startup
applyPreset("Balanced (recommended)");
document.getElementById("preset-select").value = "Balanced (recommended)";

// re-bind readouts helper (called after preset apply)
function bindRefresh() {
  const map = [
    ["flat_factor", (v) => v.toFixed(1) + "×"],
    ["detail_factor", (v) => v.toFixed(1) + "×"],
    ["contrast", (v) => v.toFixed(1)],
    ["feature_angle", (v) => v.toFixed(0) + "°"],
    ["iterations", (v) => v.toFixed(0)],
  ];
  for (const [id, fmt] of map) {
    const el = document.getElementById(id);
    const vid = id.replace("_factor", "").replace("feature_angle", "feat").replace("iterations", "iter");
    const out = document.getElementById("v-" + vid);
    if (out) out.textContent = fmt(+el.value);
  }
  const mwf = document.getElementById("max_work_faces");
  const vret = document.getElementById("v-ret");
  if (vret) vret.textContent = (+mwf.value / 1000) + "k";
}

// --------------------------------------------------------------------------- //
//  Export
// --------------------------------------------------------------------------- //
const btnExport = document.getElementById("btn-export");
const exportMenu = document.getElementById("export-menu");
btnExport.addEventListener("click", (e) => {
  e.stopPropagation();
  exportMenu.classList.toggle("hidden");
});
document.addEventListener("click", (e) => {
  if (!exportMenu.contains(e.target) && e.target !== btnExport) exportMenu.classList.add("hidden");
});
exportMenu.querySelectorAll("button").forEach((b) => {
  b.addEventListener("click", () => {
    if (!state.sessionId) return;
    const fmt = b.dataset.fmt;
    const url = `/api/export/${state.sessionId}?which=proc&fmt=${fmt}`;
    const a = document.createElement("a");
    a.href = url;
    a.download = "";
    document.body.appendChild(a);
    a.click();
    a.remove();
    exportMenu.classList.add("hidden");
    setStatus(`Exported remeshed model as ${fmt.toUpperCase()}`);
  });
});

// --------------------------------------------------------------------------- //
//  Drag & drop
// --------------------------------------------------------------------------- //
const dz = document.getElementById("dropzone");
let dragDepth = 0;
window.addEventListener("dragenter", (e) => { if (hasFiles(e)) { dragDepth++; dz.classList.remove("hidden"); } });
window.addEventListener("dragover", (e) => { if (hasFiles(e)) e.preventDefault(); });
window.addEventListener("dragleave", () => { dragDepth = Math.max(0, dragDepth - 1); if (dragDepth === 0) dz.classList.add("hidden"); });
window.addEventListener("drop", async (e) => {
  if (!hasFiles(e)) return;
  e.preventDefault(); dragDepth = 0; dz.classList.add("hidden");
  const f = [...e.dataTransfer.files].find((x) => /\.(fbx|glb|gltf|obj|ply|stl)$/i.test(x.name));
  if (f) { try { await uploadFile(f); } catch (err) { setStatus("Error: " + err.message); alert(err.message); } }
});
function hasFiles(e) { return e.dataTransfer && [...(e.dataTransfer.types || [])].includes("Files"); }

// --------------------------------------------------------------------------- //
//  Divider
// --------------------------------------------------------------------------- //
(function initDivider() {
  const split = document.getElementById("split");
  const divider = document.getElementById("divider");
  let dragging = false;
  const start = () => { dragging = true; divider.classList.add("active"); document.body.style.cursor = "col-resize"; };
  const move = (x) => {
    if (!dragging) return;
    const r = split.getBoundingClientRect();
    const p = Math.min(0.85, Math.max(0.15, (x - r.left) / r.width));
    document.getElementById("panel-left").style.flex = `0 0 ${p * 100}%`;
    resizeAll();
  };
  divider.addEventListener("mousedown", start);
  window.addEventListener("mousemove", (e) => move(e.clientX));
  window.addEventListener("mouseup", () => { if (dragging) { dragging = false; divider.classList.remove("active"); document.body.style.cursor = ""; } });
})();

// --------------------------------------------------------------------------- //
//  Helpers
// --------------------------------------------------------------------------- //
function val(id) { return document.getElementById(id).value; }
function fmtTri(n) {
  if (n == null) return "—";
  if (n >= 1000) return (n / 1000).toFixed(n >= 100000 ? 0 : 1) + "k";
  return String(n);
}
function setStatus(t, busy) {
  const el = document.getElementById("status");
  el.textContent = t;
  el.classList.toggle("spin", !!busy);
}
function escapeText(s) { const d = document.createElement("div"); d.textContent = s; return d.innerHTML; }
function escapeAttr(s) { return String(s).replace(/"/g, "&quot;").replace(/</g, "&lt;"); }
