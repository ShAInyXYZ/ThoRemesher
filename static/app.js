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
  procIsQuad: false,  // whether the current proc mesh is a quad result (set at remesh)
};

const loader = new GLTFLoader();

// --------------------------------------------------------------------------- //
//  Viewer (one per side)
// --------------------------------------------------------------------------- //
function createViewer(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0a0b);

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1e5);
  camera.position.set(3, 2, 5);

  const grid = new THREE.GridHelper(20, 40, 0x2a2a2e, 0x1c1c1f);
  grid.material.transparent = true;
  grid.material.opacity = 0.5;
  scene.add(grid);

  scene.add(new THREE.HemisphereLight(0xe8e8e4, 0x232322, 1.0));
  const key = new THREE.DirectionalLight(0xffffff, 1.6);
  key.position.set(5, 8, 6);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0x9a9e90, 0.6);
  fill.position.set(-6, 3, -4);
  scene.add(fill);

  const holder = new THREE.Group();
  scene.add(holder);

  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;

  return { canvas, renderer, scene, camera, controls, holder, meshes: [], gen: 0 };
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
  // ONE place that owns pane appearance: surface material + overlay visibility.
  // The wireframe is a LineSegments overlay ON TOP of the solid surface; the
  // surface material uses polygonOffset to sit a hair BEHIND in the depth
  // buffer, so coplanar lines always win cleanly (no z-fighting flicker).
  if (v.quadOverlay) v.quadOverlay.visible = state.wireframe;
  if (v.wireOverlay) v.wireOverlay.visible = state.wireframe;
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
    // vertex colours only in curvature / features mode; else flat grey.
    const useVColor = (state.curvature || useFeat) && !!m.geometry.getAttribute("color");
    m.material?.dispose?.();          // don't leak the previous material
    m.material = new THREE.MeshStandardMaterial({
      color: useVColor ? 0xc6c6c2 : 0x8b8b87,
      metalness: 0.05,
      roughness: 0.9,
      vertexColors: useVColor,
      side: THREE.DoubleSide,
      polygonOffset: true,            // push surface back so wire overlays win
      polygonOffsetFactor: 1,
      polygonOffsetUnits: 1,
    });
  }
}

async function loadModel(v, url, keepView = false) {
  // generation guard: rapid toggles / loads can overlap — only the LATEST call
  // for this viewer may touch the scene, or a slow response wins over a new one
  const gen = ++v.gen;
  const res = await fetch(url);
  if (!res.ok) throw new Error("model load failed: " + res.statusText);
  const buf = await res.arrayBuffer();
  const gltf = await loader.parseAsync(buf, "");
  if (gen !== v.gen) return;             // superseded while downloading
  clearHolder(v);
  v.quadOverlay = null;
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
  return loadEdgeOverlay(v, `/api/quadedges/${sid}`, "quadOverlay", 0xb8d94b);
}
// input full wireframe — SAME machinery as loadQuadEdges (server endpoint -> line
// overlay), so both panes work identically.
async function loadOrigWire(v, sid) {
  return loadEdgeOverlay(v, `/api/origedges/${sid}`, "wireOverlay", 0xb8d94b);
}

// generic: fetch the BINARY edge blob and draw as an indexed LineSegments overlay.
// Blob layout (see app.py _edge_blob): uint32 nV, uint32 nE, float32 verts
// (nV*3), uint32 edge indices (nE*2). Binary + indexed geometry is what makes
// dense-mesh wireframes appear instantly instead of after seconds of JSON.
async function loadEdgeOverlay(v, url, slot = "quadOverlay", color = 0xb8d94b) {
  try {
    const gen = v.gen;                   // belongs to the current model load
    const res = await fetch(url);
    if (!res.ok) return;
    const buf = await res.arrayBuffer();
    if (gen !== v.gen) return;           // a newer model replaced this one mid-fetch
    const [nV, nE] = new Uint32Array(buf, 0, 2);
    const verts = new Float32Array(buf, 8, nV * 3);
    const index = new Uint32Array(buf, 8 + nV * 12, nE * 2);
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(verts, 3));
    g.setIndex(new THREE.BufferAttribute(index, 1));
    const line = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color }));
    line.renderOrder = 1;             // draw after the surface (which is offset back)
    v[slot] = line;
    line.visible = state.wireframe;
    v.holder.add(line);
    applyMaterials(v);
  } catch (err) {
    console.warn("edge overlay failed:", url, err);
  }
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
  // only THIS pane's grid — sizing the other pane's grid from this bbox is wrong
  v.scene.children.forEach((c) => { if (c.isGridHelper) { c.scale.setScalar(gSize / 20); c.position.set(center.x, box.min.y, center.z); } });

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
  if (state.curvature) { state.curvature = false; setToggle("btn-curv", false); }
  if (state.features) { state.features = false; setToggle("btn-features", false); clearFeatures(left); }
  // clear any previous remesh from the right pane
  clearHolder(right); right.quadOverlay = right.wireOverlay = null;
  // a new session has no proc mesh: export/postprocess from the OLD session must
  // not stay live (stale-session API calls, wrap with the old model's units)
  document.getElementById("btn-export").disabled = true;
  document.getElementById("postprocess").hidden = true;
  resetShrinkwrapUI();
  state.procIsQuad = false;
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
  const sid = state.sessionId;   // pin the session: a demo/file load mid-remesh must not cross-wire
  const btn = document.getElementById("btn-remesh");
  btn.disabled = true;
  // loading another model during a long remesh (NeurCross = minutes) would mix sessions
  document.getElementById("btn-open").disabled = true;
  document.getElementById("demo-select").disabled = true;
  document.body.classList.add("busy");
  btn.classList.add("busy");
  const btnLabel = btn.textContent;
  btn.textContent = "Working…";

  // Live progress: the remesh can take seconds (QuadWild) to minutes (NeurCross)
  // and AutoRemesher runs silently in a subprocess, so without a ticking elapsed
  // counter the app looks frozen. Show the engine + target + elapsed, updated
  // every 200ms, so it's obvious the job is alive.
  const isQuad = state.mode === "quad";
  const engineLabel = isQuad
    ? document.querySelector(`#quad_engine option[value="${document.getElementById("quad_engine").value}"]`)?.textContent || "quad"
    : "tris";
  const targetLabel = isQuad
    ? `${Math.max(1, Math.round(+val("quad_target_num") || +val("quad_target")))}q`
    : `${+val("iterations")}i`;
  const t0 = performance.now();
  const tick = () => {
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    setStatus(`Remeshing — ${engineLabel} · ${targetLabel} · ${secs}s …`, true);
  };
  tick();
  const progressTimer = setInterval(tick, 200);
  const stopTimer = () => clearInterval(progressTimer);
  try {
    const isQuad = state.mode === "quad";
    const body = isQuad ? {
      session_id: sid,
      quad: true,
      quad_engine: document.getElementById("quad_engine").value,
      quad_target: Math.max(1, Math.round(+val("quad_target_num") || +val("quad_target"))),
      quad_sharp_mode: document.getElementById("quad_sharp_mode").value,
      quad_sharp_angle: +val("quad_sharp_angle"),
      quad_adaptivity: +val("quad_adaptivity"),
      colorize: document.getElementById("quad_colorize").checked,
    } : {
      session_id: sid,
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
    stopTimer();
    if (state.sessionId !== sid) return;   // another model was loaded meanwhile — drop this result
    document.getElementById("st-orig").textContent = fmtTri(s.orig.faces);
    const procLabel = s.quads != null ? `${fmtTri(s.quads)}q` : fmtTri(s.proc.faces);
    document.getElementById("st-proc").textContent = procLabel;
    document.getElementById("st-red").textContent = s.face_reduction_pct + "%";
    document.getElementById("st-time").textContent = (s.elapsed_ms / 1000).toFixed(2) + "s";
    document.getElementById("empty-right").style.display = "none";
    await loadModel(right, `/api/model/${sid}?which=proc&color=1`);
    // wireframe overlay for the remeshed pane: quad borders in QUAD mode, the
    // full triangle wireframe (proc edges) in TRIS mode — works in any mode.
    state.procIsQuad = s.quads != null;   // what the proc mesh IS (reloadProc relies on it)
    if (state.procIsQuad) await loadQuadEdges(right, sid);
    else await loadEdgeOverlay(right, `/api/procedges/${sid}`, "wireOverlay", 0xb8d94b);
    document.getElementById("btn-export").disabled = false;
    document.getElementById("postprocess").hidden = false;  // enable shrinkwrap
    resetShrinkwrapUI();
    const doneMsg = s.quads != null
      ? `Done — ${fmtTri(s.quads)} quads (${s.quad_ratio}% quad) [${s.quad_engine}] in ${(s.elapsed_ms / 1000).toFixed(2)}s`
      : `Done — ${fmtTri(s.proc.faces)} tris (-${s.face_reduction_pct}%) in ${(s.elapsed_ms / 1000).toFixed(2)}s`;
    setStatus(doneMsg);
  } catch (e) {
    stopTimer();
    setStatus("Error: " + e.message);
    alert("Remesh failed:\n" + e.message);
  } finally {
    btn.disabled = false;
    btn.classList.remove("busy");
    btn.textContent = btnLabel;
    document.getElementById("btn-open").disabled = false;
    document.getElementById("demo-select").disabled = false;
    document.body.classList.remove("busy");
  }
}

// --------------------------------------------------------------------------- //
//  Toggles
// --------------------------------------------------------------------------- //
function refreshAllMaterials() {
  // applyMaterials is the single owner of surface + overlay state — no extra logic here
  applyMaterials(left);
  applyMaterials(right);
}
// one place sets a toggle's visual + accessibility state
function setToggle(id, on) {
  const b = document.getElementById(id);
  b.classList.toggle("on", on);
  b.setAttribute("aria-pressed", String(on));
}
document.getElementById("btn-curv").addEventListener("click", async () => {
  state.curvature = !state.curvature;
  setToggle("btn-curv", state.curvature);
  if (state.curvature && state.features) {  // curvature and features are exclusive colour views
    state.features = false;
    setToggle("btn-features", false);
    clearFeatures(left);
  }
  // swap the input geometry: densified colour mesh for curvature, real input else
  try { if (state.sessionId) await refreshInputModel(); }
  catch (e) { setStatus("View failed: " + e.message); }
  refreshAllMaterials();
});
document.getElementById("btn-wire").addEventListener("click", () => {
  state.wireframe = !state.wireframe;
  setToggle("btn-wire", state.wireframe);
  refreshAllMaterials();
});
document.getElementById("btn-features").addEventListener("click", async () => {
  state.features = !state.features;
  setToggle("btn-features", state.features);
  if (state.features && state.curvature) {  // features owns the colour view; turn curvature off
    state.curvature = false;
    setToggle("btn-curv", false);
  }
  if (!state.features) clearFeatures(left);
  // refreshInputModel reloads the real input (features uses it 1:1) + overlays
  try { if (state.sessionId) await refreshInputModel(); }
  catch (e) { setStatus("View failed: " + e.message); }
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
  setToggle("btn-sync", state.sync);
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
 ["quad_adaptivity", "qadapt", (v) => v.toFixed(1)],
].forEach(([id, vid, fmt]) => {
  const el = document.getElementById(id), out = document.getElementById("v-" + vid);
  if (el && out) { const upd = () => out.textContent = fmt(+el.value); el.addEventListener("input", upd); upd(); }
});

// AutoRemesher's adaptivity slider is only meaningful for that engine — show it
// only when "autoremesher" is selected, hidden otherwise.
(() => {
  const engine = document.getElementById("quad_engine");
  const field = document.getElementById("quad-adaptivity-field");
  if (!engine || !field) return;
  const sync = () => { field.hidden = engine.value !== "autoremesher"; };
  engine.addEventListener("change", sync);
  sync();
})();

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
  // swap the preset list to match the mode — and APPLY what the dropdown shows,
  // so the controls always match the selected preset (it used to just display one)
  if (typeof refreshPresetSelect === "function") {
    refreshPresetSelect();
    const sel = document.getElementById("preset-select");
    if (sel && sel.options.length) {
      sel.selectedIndex = 0;
      applyPreset(sel.value);
    }
  }
}
document.getElementById("mode-tris").addEventListener("click", () => setMode("tris"));
document.getElementById("mode-quad").addEventListener("click", () => setMode("quad"));

document.getElementById("btn-remesh").addEventListener("click", doRemesh);
document.getElementById("btn-open").addEventListener("click", () => document.getElementById("file-input").click());
document.getElementById("file-input").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  e.target.value = "";   // else picking the SAME file again never re-fires change
  if (!f) return;
  try { await uploadFile(f); }
  catch (err) { setStatus("Error: " + err.message); alert("Upload failed:\n" + err.message); }
});

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
  "Balanced (recommended)": { quad_target: 2500, quad_sharp_mode: "auto", quad_sharp_angle: 35, quad_adaptivity: 0.7 },
  "Hard-surface (props)": { quad_target: 2500, quad_sharp_mode: "hard", quad_sharp_angle: 30, quad_adaptivity: 0.4 },
  "Dense detail": { quad_target: 5000, quad_sharp_mode: "auto", quad_sharp_angle: 30, quad_adaptivity: 0.9 },
  "Low poly (game)": { quad_target: 2500, quad_sharp_mode: "hard", quad_sharp_angle: 40, quad_adaptivity: 0.6 },
  "Organic / smooth": { quad_target: 2500, quad_sharp_mode: "smooth", quad_sharp_angle: 35, quad_adaptivity: 0.8 },
};
function builtinsFor(mode) { return mode === "quad" ? BUILTIN_QUADS : BUILTIN_TRIS; }

function loadCustomPresets() {
  // stored shape: { name: { mode: "tris"|"quad", params: {...} } }
  // (older entries were bare param objects — migrate them as tris presets)
  try {
    const raw = JSON.parse(localStorage.getItem(PRESET_KEY) || "{}");
    for (const [k, v] of Object.entries(raw)) {
      if (v && !v.params) raw[k] = { mode: "tris", params: v };
    }
    return raw;
  } catch { return {}; }
}
function refreshPresetSelect() {
  const sel = document.getElementById("preset-select");
  const custom = loadCustomPresets();
  // customs are PER MODE — a quad preset in the tris list would apply nothing
  const customNames = Object.keys(custom).filter((n) => custom[n].mode === state.mode).sort();
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
  const custom = loadCustomPresets();
  const p = custom[name]?.mode === state.mode ? custom[name].params
          : builtinsFor(state.mode)[name];
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
  // collect the CURRENT mode's controls — saving tris values as a "quad preset"
  // was the old bug (the preset then applied nothing visible)
  if (state.mode === "quad") {
    return {
      quad_target_num: +val("quad_target_num") || +val("quad_target"),
      quad_sharp_mode: document.getElementById("quad_sharp_mode").value,
      quad_sharp_angle: +val("quad_sharp_angle"),
      quad_adaptivity: +val("quad_adaptivity"),
      quad_colorize: document.getElementById("quad_colorize").checked,
    };
  }
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
// save with icon: auto-name "Custom preset N", tagged with the current mode
document.getElementById("btn-save-preset").addEventListener("click", () => {
  const custom = loadCustomPresets();
  let n = 1;
  while (custom[`Custom preset ${n}`]) n++;
  const name = `Custom preset ${n}`;
  custom[name] = { mode: state.mode, params: collectParams() };
  localStorage.setItem(PRESET_KEY, JSON.stringify(custom));
  refreshPresetSelect();
  document.getElementById("preset-select").value = name;
  setStatus(`Saved ${name} (${state.mode.toUpperCase()})`);
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
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") exportMenu.classList.add("hidden");
});
exportMenu.querySelectorAll("button").forEach((b) => {
  b.addEventListener("click", async () => {
    if (!state.sessionId) { exportMenu.classList.add("hidden"); return; }
    const fmt = b.dataset.fmt;
    exportMenu.classList.add("hidden");
    setStatus(`Exporting ${fmt.toUpperCase()}…`, true);
    try {
      // fetch -> blob (a bare anchor-click can't see errors and reports
      // "Exported" even when the server 404s)
      const res = await fetch(`/api/export/${state.sessionId}?which=proc&fmt=${fmt}`);
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
      const blob = await res.blob();
      const name = (res.headers.get("Content-Disposition") || "").match(/filename="(.+?)"/)?.[1]
        || `remeshed.${fmt}`;
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = name;
      a.click();
      URL.revokeObjectURL(a.href);
      setStatus(`Exported ${name}`);
    } catch (err) {
      setStatus("Export failed: " + err.message);
      alert("Export failed:\n" + err.message);
    }
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
  if (document.getElementById("btn-open").disabled) {   // remesh in flight
    setStatus("Busy remeshing — drop the file again when it finishes");
    return;
  }
  const files = [...e.dataTransfer.files];
  const f = files.find((x) => /\.(fbx|glb|gltf|obj|ply|stl)$/i.test(x.name));
  if (!f) {
    // say WHY nothing happened instead of silently closing the overlay
    setStatus(files.length ? `Can't load "${files[0].name}" — use FBX, GLB, GLTF, OBJ, PLY or STL` : "Nothing dropped");
    return;
  }
  try { await uploadFile(f); }
  catch (err) { setStatus("Error: " + err.message); alert(err.message); }
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
function resetShrinkwrapUI() {
  const en = document.getElementById("sw_enable");
  if (en) en.checked = false;
  document.getElementById("sw_controls").hidden = true;
  removeGhost();
}

function removeGhost() {
  if (right.ghost) {
    right.holder.remove(right.ghost);
    // the ghost is a Group: dispose its CHILDREN or the full original mesh
    // (potentially millions of tris) stays in GPU memory on every cycle
    right.ghost.traverse((c) => {
      c.geometry?.dispose?.();
      if (c.material) (Array.isArray(c.material) ? c.material : [c.material]).forEach((m) => m.dispose());
    });
    right.ghost = null;
  }
}

async function showGhost() {
  // load the original mesh as a translucent overlay in the remeshed pane
  removeGhost();
  const res = await fetch(`/api/model/${state.sessionId}?which=orig`);
  if (!res.ok) throw new Error("ghost load failed: " + res.statusText);
  const gltf = await loader.parseAsync(await res.arrayBuffer(), "");
  const grp = new THREE.Group();
  gltf.scene.traverse((c) => {
    if (c.isMesh) {
      grp.add(new THREE.Mesh(c.geometry, new THREE.MeshStandardMaterial({
        color: 0xb8d94b, transparent: true, opacity: 0.22, depthWrite: false, side: THREE.DoubleSide,
      })));
    }
  });
  right.ghost = grp;
  right.holder.add(grp);
}

function procDiag() {
  // model size for %->units conversion, from the ALWAYS-PRESENT proc mesh (the
  // ghost may still be downloading — using it made the units racy/stale)
  const box = new THREE.Box3();
  for (const m of right.meshes) box.expandByObject(m);
  return box.isEmpty() ? 1 : box.getSize(new THREE.Vector3()).length() || 1;
}

async function reloadProc() {
  // reload the (wrapped/reset) proc mesh + its wireframe overlay, keep the view
  await loadModel(right, `/api/model/${state.sessionId}?which=proc&color=1`, true);
  // overlay matches what the mesh IS (recorded at remesh time), not the UI tab
  if (state.procIsQuad) await loadQuadEdges(right, state.sessionId);
  else await loadEdgeOverlay(right, `/api/procedges/${state.sessionId}`, "wireOverlay", 0xb8d94b);
  applyMaterials(right);
  // keep the ghost visible while shrinkwrap stays enabled (so you can re-wrap)
  if (document.getElementById("sw_enable")?.checked) await showGhost();
}

(function setupShrinkwrap() {
  const enable = document.getElementById("sw_enable");
  const controls = document.getElementById("sw_controls");
  if (!enable) return;

  enable.addEventListener("change", async () => {
    if (enable.checked) {
      controls.hidden = false;
      try { await showGhost(); }
      catch (e) { setStatus("Ghost preview failed: " + e.message); }
    } else { controls.hidden = true; removeGhost(); }
  });

  // readouts
  const dist = document.getElementById("sw_distance"), off = document.getElementById("sw_offset");
  dist.addEventListener("input", () => {
    document.getElementById("v-swdist").textContent = +dist.value === 0 ? "∞" : (+dist.value) + "%";
  });
  off.addEventListener("input", () => {
    document.getElementById("v-swoff").textContent = (+off.value).toFixed(1) + "%";
  });

  const wrapBtn = document.getElementById("sw_wrap");
  const resetBtn = document.getElementById("sw_reset");

  wrapBtn.addEventListener("click", async () => {
    setStatus("Shrinkwrapping…", true);
    document.body.classList.add("busy");
    wrapBtn.disabled = resetBtn.disabled = true;   // no double-fire while in flight
    try {
      const diag = procDiag();
      const res = await fetch("/api/shrinkwrap", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: state.sessionId,
          mode: document.getElementById("sw_mode").value,
          distance: (+dist.value / 100) * diag,   // % of proc bbox diagonal -> units
          offset: (+off.value / 100) * diag,
          reset: false,
        }),
      });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      await reloadProc();
      setStatus("Shrinkwrapped onto original.");
    } catch (e) {
      setStatus("Error: " + e.message); alert("Shrinkwrap failed:\n" + e.message);
    } finally {
      document.body.classList.remove("busy");
      wrapBtn.disabled = resetBtn.disabled = false;
    }
  });

  resetBtn.addEventListener("click", async () => {
    setStatus("Resetting…", true);
    wrapBtn.disabled = resetBtn.disabled = true;
    try {
      const res = await fetch("/api/shrinkwrap", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: state.sessionId, reset: true }),
      });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      await reloadProc();
      setStatus("Reset to pre-shrinkwrap.");
    } catch (e) { setStatus("Error: " + e.message); }
    finally { wrapBtn.disabled = resetBtn.disabled = false; }
  });
})();

// --------------------------------------------------------------------------- //
//  Filled slider tracks — the CSS paints the accent up to --p (the value)
// --------------------------------------------------------------------------- //
function paintRange(el) {
  const min = +el.min || 0, max = +el.max || 100;
  const p = ((+el.value - min) / (max - min)) * 100;
  el.style.setProperty("--p", p + "%");
}
document.querySelectorAll('input[type="range"]').forEach((el) => {
  paintRange(el);
  el.addEventListener("input", () => paintRange(el));   // covers preset apply too (dispatched input)
});
