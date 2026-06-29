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
  features: false,
  wireframe: false,
  sync: true,
  syncing: false,
  mode: "tris",
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
  v.ghost = null;  // ghost (shrinkwrap preview) lives in holder; cleared with it
}

function applyMaterials(v) {
  // ONLY the quad overlay (proc pane) replaces the tri-wireframe — its quad
  // borders are the wires, so the surface must stay solid. Everywhere else the
  // Wireframe toggle shows a grey all-edges overlay (built lazily) ON TOP of the
  // solid surface, so you get solid + wireframe together (never a hidden fill).
  const triWire = false;  // never make the surface material itself wireframe
  // ONE logic for both panes: the edge overlay (quadOverlay on proc, wireOverlay
  // on input — both fetched from a server endpoint) follows the wireframe toggle.
  if (v.quadOverlay) v.quadOverlay.visible = state.wireframe;
  if (v.wireOverlay) v.wireOverlay.visible = state.wireframe;
  if (v.featureOverlay) v.featureOverlay.visible = state.wireframe;
  for (const m of v.meshes) {
    // FEATURES view: paint the surface by region label (overrides curvature).
    const useFeat = state.features && v.featureColors &&
      m.geometry.getAttribute("position") &&
      v.featureColors.length === m.geometry.getAttribute("position").count;
    if (useFeat) {
      const fc = new Float32Array(v.featureColors.flat());
      m.geometry.setAttribute("color", new THREE.BufferAttribute(fc, 3));
    }
    if (!m.geometry.getAttribute("normal")) m.geometry.computeVertexNormals();
    // only use vertex colours in curvature / features mode; otherwise a flat
    // material colour (so a GLB-baked default colour can't keep it white).
    const useVColor = (state.curvature || useFeat) && !!m.geometry.getAttribute("color");
    // dark-grey surface is the DEFAULT (so wires always pop); the lighter grey is
    // only used while a curvature/feature colour view is active (it isn't).
    const surfColor = useVColor ? 0xbfc8d2 : 0x808890;
    const mat = new THREE.MeshStandardMaterial({
      color: surfColor,
      metalness: 0.05,
      roughness: 0.9,
      flatShading: false,
      vertexColors: useVColor,
      wireframe: triWire,
      side: THREE.DoubleSide,
    });
    m.material = mat;
  }
}

async function loadModel(v, url, keepView = false) {
  const buf = await (await fetch(url)).arrayBuffer();
  const gltf = await loader.parseAsync(buf, "");
  clearHolder(v);
  v.quadOverlay = null;
  v.featureOverlay = null;
  v.wireOverlay = null;
  const root = gltf.scene;
  root.traverse((c) => {
    if (c.isMesh) v.meshes.push(c);
  });
  v.holder.add(root);
  applyMaterials(v);
  if (!keepView) fitView(v);   // keepView: curvature/feature swaps must NOT reset the camera
}

// draw true quad borders (proc): REPLACES the tri-wireframe (slot=quadOverlay)
async function loadQuadEdges(v, sid) {
  return loadEdgeOverlay(v, `/api/quadedges/${sid}`, "quadOverlay", 0x6fd3ff);
}
// input full wireframe — SAME machinery as loadQuadEdges (server endpoint -> line
// overlay), so both panes work identically.
async function loadOrigWire(v, sid) {
  return loadEdgeOverlay(v, `/api/origedges/${sid}`, "wireOverlay", 0x6fd3ff);
}

// generic: fetch {vertices, edges} and draw as a LineSegments overlay
async function loadEdgeOverlay(v, url, slot = "quadOverlay", color = 0x6fd3ff) {
  try {
    const res = await fetch(url);
    if (!res.ok) return;
    const { vertices, edges } = await res.json();
    // Overlay coords share the GLB's coordinate space (verified identical bbox);
    // no rotation, or the lines would float off the surface.
    const pos = new Float32Array(edges.length * 6);
    for (let i = 0; i < edges.length; i++) {
      const a = vertices[edges[i][0]], b = vertices[edges[i][1]];
      pos.set([a[0], a[1], a[2], b[0], b[1], b[2]], i * 6);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    const line = new THREE.LineSegments(
      g, new THREE.LineBasicMaterial({ color }));
    v[slot] = line;
    line.visible = state.wireframe;
    v.holder.add(line);
    applyMaterials(v);
  } catch (_) { /* no overlay */ }
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
async function onSessionLoaded(data) {
  state.sessionId = data.session_id;
  state.name = data.name;
  document.getElementById("st-orig").textContent = fmtTri(data.orig.faces);
  document.getElementById("st-proc").textContent = "—";
  document.getElementById("st-red").textContent = "—";
  document.getElementById("st-time").textContent = "—";
  document.getElementById("empty-right").style.display = "";
  setStatus(`Loaded ${data.name} — ${fmtTri(data.orig.faces)} tris`);
  // a NEW object -> reset the colour-view toggles so stale curvature/feature state
  // can't corrupt the fresh load (wireframe is a harmless independent toggle).
  if (state.curvature) { state.curvature = false; document.getElementById("btn-curv").classList.remove("on"); }
  if (state.features) { state.features = false; document.getElementById("btn-features").classList.remove("on"); clearFeatures(left); }
  // clear any previous remesh from the right pane
  clearHolder(right); right.quadOverlay = right.featureOverlay = right.wireOverlay = null;
  // load the REAL input geometry and REFIT the camera (new object -> reset view).
  await loadModel(left, `/api/model/${state.sessionId}?which=orig`);  // keepView=false -> refits
  await loadOrigWire(left, state.sessionId);   // input wireframe (server endpoint, like proc)
  applyMaterials(left);
  document.getElementById("btn-remesh").disabled = false;
}

// reload the left/input pane with the right geometry for the current view:
// densified colour mesh ONLY for the curvature heatmap, else the real input.
async function refreshInputModel() {
  const useColor = state.curvature && !state.features;
  const url = `/api/model/${state.sessionId}?which=orig` + (useColor ? "&color=1" : "");
  await loadModel(left, url, true);   // keepView -> no camera reset on curv/feature toggle
  // the input wireframe always comes from /api/origedges (the REAL input edges),
  // independent of whether the displayed surface is the densified curvature mesh.
  await loadOrigWire(left, state.sessionId);
  if (state.features) {   // sharp-edge highlights belong to the Features view only
    await loadFeatures(left, state.sessionId);
  }
  applyMaterials(left);
}

async function uploadFile(file) {
  setStatus(`Uploading ${file.name} …`);
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch("/api/upload", { method: "POST", body: fd });
  if (!res.ok) throw new Error((await res.text()) || "upload failed");
  await onSessionLoaded(await res.json());
}

async function loadDemo(name) {
  setStatus(`Loading demo: ${name} …`);
  const res = await fetch(`/api/demo/${name}`);
  if (!res.ok) throw new Error((await res.text()) || "demo load failed");
  await onSessionLoaded(await res.json());
}

// populate the demo dropdown + wire it
(async () => {
  try {
    const sel = document.getElementById("demo-select");
    const { demos } = await (await fetch("/api/demos")).json();
    for (const d of demos) {
      const o = document.createElement("option");
      o.value = d;
      o.textContent = d.charAt(0).toUpperCase() + d.slice(1);
      sel.appendChild(o);
    }
    sel.addEventListener("change", async (e) => {
      if (!e.target.value) return;
      try { await loadDemo(e.target.value); }
      catch (err) { setStatus("Error: " + err.message); alert(err.message); }
      e.target.value = "";  // reset so the same demo can be re-picked
    });
  } catch (_) { /* demos optional */ }
})();

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
    const isQuad = state.mode === "quad";
    const body = isQuad ? {
      session_id: state.sessionId,
      quad: true,
      quad_engine: document.getElementById("quad_engine").value,
      quad_target: Math.max(1, Math.round(+val("quad_target_num") || +val("quad_target"))),
      quad_sharp_mode: document.getElementById("quad_sharp_mode").value,
      quad_sharp_angle: +val("quad_sharp_angle"),
      colorize: document.getElementById("quad_colorize").checked,
    } : {
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
    const procLabel = s.quads != null ? `${fmtTri(s.quads)}q` : fmtTri(s.proc.faces);
    document.getElementById("st-proc").textContent = procLabel;
    document.getElementById("st-red").textContent = s.face_reduction_pct + "%";
    document.getElementById("st-time").textContent = (s.elapsed_ms / 1000).toFixed(2) + "s";
    document.getElementById("empty-right").style.display = "none";
    await loadModel(right, `/api/model/${state.sessionId}?which=proc&color=1`);
    // wireframe overlay for the remeshed pane: quad borders in QUAD mode, the
    // full triangle wireframe (proc edges) in TRIS mode — works in any mode.
    if (s.quads != null) await loadQuadEdges(right, state.sessionId);
    else await loadEdgeOverlay(right, `/api/procedges/${state.sessionId}`, "wireOverlay", 0x6fd3ff);
    document.getElementById("btn-export").disabled = false;
    document.getElementById("postprocess").hidden = false;  // enable shrinkwrap
    resetShrinkwrapUI();
    const doneMsg = s.quads != null
      ? `Done — ${fmtTri(s.quads)} quads (${s.quad_ratio}% quad) [${s.quad_engine}] in ${(s.elapsed_ms / 1000).toFixed(2)}s`
      : `Done — ${fmtTri(s.proc.faces)} tris (-${s.face_reduction_pct}%) in ${(s.elapsed_ms / 1000).toFixed(2)}s`;
    setStatus(doneMsg);
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
  // quad overlays follow the wireframe toggle
  for (const v of [left, right]) {
    if (v.quadOverlay) v.quadOverlay.visible = state.wireframe;
    if (v.featureOverlay) v.featureOverlay.visible = state.wireframe;
  }
}
document.getElementById("btn-curv").addEventListener("click", async (e) => {
  state.curvature = !state.curvature;
  e.currentTarget.classList.toggle("on", state.curvature);
  if (state.curvature && state.features) {  // curvature and features are exclusive colour views
    state.features = false;
    document.getElementById("btn-features").classList.remove("on");
    clearFeatures(left);
  }
  // swap the input geometry: densified colour mesh for curvature, real input else
  if (state.sessionId) await refreshInputModel();
  refreshAllMaterials();
});
document.getElementById("btn-wire").addEventListener("click", (e) => {
  state.wireframe = !state.wireframe;
  e.currentTarget.classList.toggle("on", state.wireframe);
  refreshAllMaterials();
});
document.getElementById("btn-features").addEventListener("click", async (e) => {
  state.features = !state.features;
  e.currentTarget.classList.toggle("on", state.features);
  if (state.features && state.curvature) {  // features owns the colour view; turn curvature off
    state.curvature = false;
    document.getElementById("btn-curv").classList.remove("on");
  }
  if (!state.features) clearFeatures(left);
  // refreshInputModel reloads the real input (features uses it 1:1) + overlays
  if (state.sessionId) await refreshInputModel();
  refreshAllMaterials();
});

// Build the "Features" debug overlay for the INPUT: region colours on the
// surface + bright crease loops + flow-direction arrows = what the algorithm
// perceives before remeshing.
async function loadFeatures(v, sid) {
  if (!sid) return;
  clearFeatures(v);
  let data;
  try {
    const res = await fetch(`/api/features/${sid}`);
    if (!res.ok) return;
    data = await res.json();
  } catch (_) { return; }
  const grp = new THREE.Group();
  const verts = data.vertices;
  // region colours -> paint the surface meshes
  if (data.vertex_colors) {
    v.featureColors = data.vertex_colors;
  }
  // crease loops: circle=cyan, other=yellow
  for (const cl of data.crease_lines || []) {
    const pts = cl.pts.map((p) => new THREE.Vector3(p[0], p[1], p[2]));
    if (cl.closed && pts.length) pts.push(pts[0].clone());
    const g = new THREE.BufferGeometry().setFromPoints(pts);
    const col = cl.circle ? 0x35e0ff : 0xffd23a;
    grp.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color: col, linewidth: 2 })));
  }
  // flow arrows (short segments)
  if (data.flow && data.flow.length) {
    const pos = new Float32Array(data.flow.length * 6);
    data.flow.forEach((f, i) => pos.set([...f.a, ...f.b], i * 6));
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    grp.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0xff5fa2 })));
  }
  v.featureGroup = grp;
  v.holder.add(grp);
}
function clearFeatures(v) {
  if (v.featureGroup) { v.holder.remove(v.featureGroup); v.featureGroup = null; }
  v.featureColors = null;
}
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

// quad readouts (explicit id -> label, since ids don't match the bind pattern)
[["quad_sharp_angle", "qsharp", (v) => v.toFixed(0) + "°"],
].forEach(([id, vid, fmt]) => {
  const el = document.getElementById(id), out = document.getElementById("v-" + vid);
  if (el && out) { const upd = () => out.textContent = fmt(+el.value); el.addEventListener("input", upd); upd(); }
});

// Target quads: editable number box + slider, two-way synced. The NUMBER box is
// the source of truth (lets you type any value, even beyond the slider range).
(() => {
  const slider = document.getElementById("quad_target");
  const num = document.getElementById("quad_target_num");
  if (!slider || !num) return;
  // slider drag -> number box
  slider.addEventListener("input", () => { num.value = slider.value; });
  // typed number -> clamp the slider to its range (number keeps the exact value)
  num.addEventListener("input", () => {
    const v = Math.max(1, Math.round(+num.value || 0));
    slider.value = Math.min(+slider.max, Math.max(+slider.min, v));
  });
})();

// TRIS / QUAD mode toggle
function setMode(m) {
  state.mode = m;
  document.getElementById("mode-tris").classList.toggle("active", m === "tris");
  document.getElementById("mode-quad").classList.toggle("active", m === "quad");
  document.getElementById("tris-settings").classList.toggle("hidden", m !== "tris");
  document.getElementById("quad-settings").classList.toggle("hidden", m !== "quad");
  // swap the preset list to match the mode
  if (typeof refreshPresetSelect === "function") {
    refreshPresetSelect();
    const sel = document.getElementById("preset-select");
    if (sel && sel.options.length) sel.selectedIndex = 0;
  }
}
document.getElementById("mode-tris").addEventListener("click", () => setMode("tris"));
document.getElementById("mode-quad").addEventListener("click", () => setMode("quad"));

document.getElementById("btn-remesh").addEventListener("click", doRemesh);
document.getElementById("btn-open").addEventListener("click", () => document.getElementById("file-input").click());
document.getElementById("file-input").addEventListener("change", (e) => { if (e.target.files[0]) uploadFile(e.target.files[0]); });

// --------------------------------------------------------------------------- //
//  Presets (localStorage)
// --------------------------------------------------------------------------- //
const PRESET_KEY = "remesher_presets_v1";
const BUILTIN_TRIS = {
  "Balanced (recommended)": { flat_factor: 3, detail_factor: 1, contrast: 2.5, feature_angle: 30, iterations: 4, max_work_faces: 60000, pre_simplify: false, colorize: true, preserve_boundary: true },
  "High detail (characters)": { flat_factor: 2.5, detail_factor: 1, contrast: 3, feature_angle: 25, iterations: 4, max_work_faces: 100000, pre_simplify: false, colorize: true, preserve_boundary: true },
  "Max reduction (game-ready)": { flat_factor: 6, detail_factor: 1.2, contrast: 3, feature_angle: 35, iterations: 3, max_work_faces: 50000, pre_simplify: false, colorize: true, preserve_boundary: true },
  "Aggressive simplify": { flat_factor: 8, detail_factor: 1.5, contrast: 4, feature_angle: 40, iterations: 2, max_work_faces: 40000, pre_simplify: true, colorize: true, preserve_boundary: true },
  "Gentle cleanup": { flat_factor: 2, detail_factor: 0.9, contrast: 2, feature_angle: 20, iterations: 5, max_work_faces: 150000, pre_simplify: false, colorize: true, preserve_boundary: true },
};
const BUILTIN_QUADS = {
  "Balanced (recommended)": { quad_target: 2500, quad_sharp_mode: "auto", quad_sharp_angle: 35 },
  "Hard-surface (props)": { quad_target: 2500, quad_sharp_mode: "hard", quad_sharp_angle: 30 },
  "Dense detail": { quad_target: 5000, quad_sharp_mode: "auto", quad_sharp_angle: 30 },
  "Low poly (game)": { quad_target: 2500, quad_sharp_mode: "hard", quad_sharp_angle: 40 },
  "Organic / smooth": { quad_target: 2500, quad_sharp_mode: "smooth", quad_sharp_angle: 35 },
};
function builtinsFor(mode) { return mode === "quad" ? BUILTIN_QUADS : BUILTIN_TRIS; }

function loadCustomPresets() {
  try { return JSON.parse(localStorage.getItem(PRESET_KEY) || "{}"); } catch { return {}; }
}
function refreshPresetSelect() {
  const sel = document.getElementById("preset-select");
  const custom = loadCustomPresets();
  const customNames = Object.keys(custom).sort();
  const cur = sel.value;
  let html = "";
  for (const n of Object.keys(builtinsFor(state.mode))) {
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
  const presets = { ...builtinsFor(state.mode), ...loadCustomPresets() };
  const p = presets[name];
  if (!p) return;
  for (const [k, v] of Object.entries(p)) {
    const el = document.getElementById(k);
    if (!el) continue;
    if (el.type === "checkbox") el.checked = !!v;
    else el.value = v;
    // fire the element's own listener so its value readout updates (quad sliders)
    el.dispatchEvent(new Event("input", { bubbles: true }));
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
// initialise mode (sets panel visibility + the matching preset list)
setMode("tris");
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

// --------------------------------------------------------------------------- //
//  Postprocess — Shrinkwrap
//  Preview: a translucent ghost of the ORIGINAL is shown in the remeshed pane so
//  you can see the alignment. "Wrap" runs the accurate projection on the backend
//  and reloads the result; the offset (%) is converted to model units client-side.
// --------------------------------------------------------------------------- //
let _swDiag = 1;  // bbox diagonal of the current model, for %-to-units on offset/distance

function resetShrinkwrapUI() {
  const en = document.getElementById("sw_enable");
  if (en) en.checked = false;
  document.getElementById("sw_controls").hidden = true;
  removeGhost();
}

function removeGhost() {
  if (right.ghost) { right.holder.remove(right.ghost); right.ghost.geometry?.dispose?.(); right.ghost.material?.dispose?.(); right.ghost = null; }
}

async function showGhost() {
  // load the original mesh as a translucent overlay in the remeshed pane
  removeGhost();
  const buf = await (await fetch(`/api/model/${state.sessionId}?which=orig`)).arrayBuffer();
  const gltf = await loader.parseAsync(buf, "");
  const grp = new THREE.Group();
  const box = new THREE.Box3();
  gltf.scene.traverse((c) => {
    if (c.isMesh) {
      const g = c.geometry.clone();
      box.expandByObject(c);
      const mesh = new THREE.Mesh(g, new THREE.MeshStandardMaterial({
        color: 0x4cc4ff, transparent: true, opacity: 0.22, depthWrite: false, side: THREE.DoubleSide,
      }));
      grp.add(mesh);
    }
  });
  _swDiag = box.getSize(new THREE.Vector3()).length() || 1;
  right.ghost = grp;
  right.holder.add(grp);
}

async function reloadProc() {
  // reload the (wrapped/reset) proc mesh + its wireframe overlay, keep the view
  const wasWire = state.wireframe;
  await loadModel(right, `/api/model/${state.sessionId}?which=proc&color=1`, true);
  if (state.mode === "tris") await loadEdgeOverlay(right, `/api/procedges/${state.sessionId}`, "wireOverlay", 0x6fd3ff);
  else await loadQuadEdges(right, state.sessionId);
  state.wireframe = wasWire; applyMaterials(right);
  // keep the ghost visible while shrinkwrap stays enabled (so you can re-wrap)
  if (document.getElementById("sw_enable")?.checked) await showGhost();
}

(function setupShrinkwrap() {
  const enable = document.getElementById("sw_enable");
  const controls = document.getElementById("sw_controls");
  if (!enable) return;

  enable.addEventListener("change", async () => {
    if (enable.checked) { controls.hidden = false; await showGhost(); }
    else { controls.hidden = true; removeGhost(); }
  });

  // readouts
  const dist = document.getElementById("sw_distance"), off = document.getElementById("sw_offset");
  dist.addEventListener("input", () => {
    document.getElementById("v-swdist").textContent = +dist.value === 0 ? "∞" : (+dist.value) + "%";
  });
  off.addEventListener("input", () => {
    document.getElementById("v-swoff").textContent = (+off.value).toFixed(1) + "%";
  });

  document.getElementById("sw_wrap").addEventListener("click", async () => {
    setStatus("Shrinkwrapping…", true);
    document.body.classList.add("busy");
    try {
      const res = await fetch("/api/shrinkwrap", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: state.sessionId,
          mode: document.getElementById("sw_mode").value,
          distance: (+dist.value / 100) * _swDiag,   // % of bbox diagonal -> units
          offset: (+off.value / 100) * _swDiag,
          reset: false,
        }),
      });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      await reloadProc();
      setStatus("Shrinkwrapped onto original.");
    } catch (e) {
      setStatus("Error: " + e.message); alert("Shrinkwrap failed:\n" + e.message);
    } finally { document.body.classList.remove("busy"); }
  });

  document.getElementById("sw_reset").addEventListener("click", async () => {
    setStatus("Resetting…", true);
    try {
      await fetch("/api/shrinkwrap", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: state.sessionId, reset: true }),
      });
      await reloadProc();
      setStatus("Reset to pre-shrinkwrap.");
    } catch (e) { setStatus("Error: " + e.message); }
  });
})();
