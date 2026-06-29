"""
FastAPI server for the curvature-aware adaptive remesher.

Endpoints
---------
POST /api/upload        multipart file upload -> stores a session, returns stats
GET  /api/model/{sid}   ?which=orig|proc&color=0|1  -> GLB bytes for a viewer
POST /api/remesh        {session_id, params...}     -> runs pipeline, returns stats
GET  /api/health        liveness probe
"""
from __future__ import annotations

import io
import os
import tempfile
import time
import uuid
from collections import OrderedDict
from typing import Any, Dict

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import remesh_engine as eng


HERE = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.join(HERE, "_tmp")
os.makedirs(TMP, exist_ok=True)

app = FastAPI(title="Curvature-Aware Remesher")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def no_cache_html_js(request: Request, call_next):
    """Prevent the browser from caching stale app.js / index.html."""
    response = await call_next(request)
    path = request.url.path.lower()
    if any(path.endswith(ext) for ext in (".html", ".js", ".css")) or path == "/":
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


# --------------------------------------------------------------------------- #
#  Sessions (in-memory)
# --------------------------------------------------------------------------- #
class Session:
    __slots__ = ("path", "orig", "proc", "orig_color", "proc_color", "name",
                 "error", "quads", "proc_prewrap")

    def __init__(self, path, name):
        self.path = path
        self.name = name
        self.orig = None        # (V, F)
        self.proc = None        # (V, F)
        self.orig_color = None  # (V,4)
        self.proc_color = None  # (V,4)
        self.quads = None       # (Q,4) HumanLogic quad faces
        self.proc_prewrap = None  # cached proc verts before shrinkwrap (re-run/reset)
        self.error = None


SESSIONS: "OrderedDict[str, Session]" = OrderedDict()
_MAX_SESSIONS = 24  # cap memory: each session holds full V/F + color copies


def _put_session(sid: str, sess: Session) -> None:
    """Store a session, evicting the oldest once over the cap (FIFO)."""
    SESSIONS[sid] = sess
    SESSIONS.move_to_end(sid)
    while len(SESSIONS) > _MAX_SESSIONS:
        SESSIONS.popitem(last=False)


# --------------------------------------------------------------------------- #
#  Schemas
# --------------------------------------------------------------------------- #
class RemeshRequest(BaseModel):
    session_id: str
    flat_factor: float = 3.0
    detail_factor: float = 1.0
    contrast: float = 2.0
    feature_angle: float = 30.0
    iterations: int = 6
    pre_simplify: bool = False
    pre_simplify_target: float = 0.25
    preserve_boundary: bool = True
    colorize: bool = True
    max_work_faces: int = 50000
    # Quad mode
    quad: bool = False
    quad_target: int = 2000         # approx quad count (density)
    quad_sharp_mode: str = "auto"   # "auto" | "hard" (keep sharp edges) | "smooth"
    quad_sharp_angle: float = 35.0  # dihedral angle that counts as a sharp edge
    quad_work_faces: int = 15000
    quad_engine: str = "quadwild"   # "quadwild" (default) | "visibility" | "shrinkwrap" | "humanlogic"
    # legacy HumanLogic-only knobs (ignored by quadwild; kept for that engine)
    quad_feature_angle: float = 35.0
    quad_feature_weight: float = 8.0
    quad_ridge_weight: float = 3.0


class ShrinkwrapRequest(BaseModel):
    session_id: str
    mode: str = "nearest"     # "nearest" | "project"
    distance: float = 0.0     # max search distance, model units (0 = unlimited)
    offset: float = 0.0       # inflation above the surface, model units
    reset: bool = False       # restore the pre-shrinkwrap proc and stop


def _compute_colors(V, F):
    try:
        return eng.curvature_colors(eng.detail_score(V, F))
    except Exception:  # noqa: BLE001
        return None


def _densify_for_color(V, F, min_verts=4000):
    """A per-vertex heatmap needs enough vertices to carry a gradient. A coarse
    primitive (e.g. a 98-vertex cylinder with NO wall-interior verts) can only
    show two colors. Midpoint-subdivide until there are interior samples, so the
    curvature view reads as a real gradient (flat wall blue, sharp rim red).
    Returns a densified (V, F) — used ONLY for the colored curvature GLB."""
    import trimesh as _tm

    m = _tm.Trimesh(np.asarray(V, np.float64), np.asarray(F, np.int64), process=False)
    guard = 0
    while len(m.vertices) < min_verts and guard < 5:
        m = m.subdivide()
        guard += 1
    return np.asarray(m.vertices), np.asarray(m.faces)


def _stats_dict(V, F):
    return {"vertices": int(len(V)), "faces": int(len(F))}


# --------------------------------------------------------------------------- #
#  Routes
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health():
    return {"ok": True, "pymeshlab": eng.pymeshlab is not None}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "")[1].lower() or ".bin"
    sid = uuid.uuid4().hex[:12]
    path = os.path.join(TMP, f"{sid}{suffix}")
    with open(path, "wb") as fh:
        fh.write(await file.read())

    sess = Session(path, file.filename or "model")
    try:
        V, F = eng.load_mesh(path)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=415, detail=f"Could not load mesh: {e}")
    finally:
        # the upload file is only read by load_mesh — don't leave it in _tmp forever
        try:
            os.remove(path)
        except OSError:
            pass
    sess.orig = (V, F)
    sess.orig_color = None  # computed lazily on first /api/model?color=1 request
    _put_session(sid, sess)
    bb = V.max(axis=0) - V.min(axis=0) if len(V) else np.zeros(3)
    return {
        "session_id": sid,
        "name": sess.name,
        "orig": _stats_dict(V, F),
        "bbox_diag": float(np.linalg.norm(bb)) if len(V) else 0.0,
    }


DEMOS = {
    "sphere": lambda t: t.creation.icosphere(subdivisions=4),
    "torus": lambda t: t.creation.torus(2.0, 0.7, major_sections=64, minor_sections=32),
    "cube": lambda t: t.creation.box(extents=(2, 2, 2)).subdivide().subdivide(),
    "cylinder": lambda t: t.creation.cylinder(radius=1, height=3, sections=48),
    "cone": lambda t: t.creation.cone(radius=1, height=2.5, sections=48),
    "capsule": lambda t: t.creation.capsule(height=2, radius=0.7, count=[32, 32]),
    "bumpy": lambda t: _bumpy_sphere(t),
}


def _bumpy_sphere(t):
    """A sphere with high-frequency bumps — exercises the curvature heatmap."""
    m = t.creation.icosphere(subdivisions=5)
    v = m.vertices.copy()
    disp = 0.07 * np.sin(v[:, 0] * 25) * np.sin(v[:, 1] * 25) * np.sin(v[:, 2] * 25)
    v += m.vertex_normals * disp[:, None]
    return t.Trimesh(vertices=v, faces=m.faces, process=False)


@app.get("/api/demos")
def list_demos():
    return {"demos": list(DEMOS.keys())}


@app.get("/api/demo/{name}")
def load_demo(name: str):
    """Create a built-in demo mesh and start a session (like /api/upload)."""
    import trimesh

    maker = DEMOS.get(name)
    if maker is None:
        raise HTTPException(status_code=404, detail=f"unknown demo: {name}")
    m = maker(trimesh)
    V = np.asarray(m.vertices, dtype=np.float64)
    F = np.asarray(m.faces, dtype=np.int64)
    sid = uuid.uuid4().hex[:12]
    sess = Session(os.path.join(TMP, f"{sid}.demo"), f"{name} (demo)")
    sess.orig = (V, F)
    _put_session(sid, sess)
    bb = V.max(axis=0) - V.min(axis=0) if len(V) else np.zeros(3)
    return {
        "session_id": sid,
        "name": sess.name,
        "orig": _stats_dict(V, F),
        "bbox_diag": float(np.linalg.norm(bb)) if len(V) else 0.0,
    }


@app.get("/api/model/{sid}")
def model(sid: str, which: str = "orig", color: int = 0):
    sess = SESSIONS.get(sid)
    if sess is None:
        raise HTTPException(status_code=404, detail="unknown session")
    if which == "proc" and sess.proc is None:
        raise HTTPException(status_code=409, detail="not remeshed yet")
    V, F = sess.proc if which == "proc" else sess.orig
    colors = None
    if color:
        if which == "proc":
            colors = sess.proc_color
            if colors is None and sess.proc is not None:
                colors = _compute_colors(*sess.proc)
                sess.proc_color = colors
        else:
            # densify the INPUT for the curvature heatmap so a coarse primitive
            # (no wall-interior verts) still shows a real gradient, then color it.
            cached = sess.orig_color
            if cached is None:
                Vd, Fd = _densify_for_color(V, F)
                cd = _compute_colors(Vd, Fd)
                sess.orig_color = (Vd, Fd, cd)
                cached = sess.orig_color
            Vd, Fd, cd = cached
            if cd is not None:
                return Response(content=eng.to_glb_bytes(Vd, Fd, cd),
                                media_type="model/gltf-binary")
    data = eng.to_glb_bytes(V, F, colors)
    return Response(content=data, media_type="model/gltf-binary")


@app.get("/api/features/{sid}")
def features(sid: str):
    """Stage 1-2 ANALYSIS overlay: what the algorithm perceives BEFORE remeshing —
    sharp crease loops, classified regions (flat / developable / doubly-curved),
    and the detected flow direction. Lets you see how the engine reads the shape.
    Returns JSON: {crease_lines:[{pts, circle}], region_colors:[(v_id->rgb)],
    flow:[{a,b}], vertices}."""
    import trimesh as _tm

    sess = SESSIONS.get(sid)
    if sess is None or sess.orig is None:
        raise HTTPException(status_code=404, detail="unknown session")
    V, F = sess.orig
    mesh = _tm.Trimesh(np.asarray(V, np.float64), np.asarray(F, np.int64),
                       process=True)
    out = {"vertices": mesh.vertices.tolist(), "crease_lines": [],
           "vertex_colors": None, "flow": []}
    try:
        import features as ftmod

        fs = ftmod.analyze(mesh, feature_angle=35.0)
        # crease loops -> ordered point lists, flagged circle vs other
        for cl in fs.crease_loops:
            out["crease_lines"].append(
                {"pts": np.asarray(cl.pts).tolist(), "circle": bool(cl.is_circle),
                 "closed": bool(cl.closed)})
        # per-vertex region color: map each face's label to a colour, then to its
        # verts (flat=blue, developable=green, doubly-curved=orange, transition=red)
        lab_rgb = {"flat": [60, 110, 230], "curved_dev": [60, 200, 120],
                   "curved_double": [240, 150, 50], "transition": [230, 70, 70]}
        vcol = np.full((len(mesh.vertices), 3), 120, np.float64)
        cnt = np.zeros(len(mesh.vertices))
        for fi, lab in enumerate(fs.face_label):
            c = lab_rgb.get(str(lab), [120, 120, 120])
            for vid in mesh.faces[fi]:
                vcol[vid] += c
                cnt[vid] += 1
        cnt[cnt == 0] = 1
        vcol = (vcol / cnt[:, None])
        out["vertex_colors"] = (vcol / 255.0).tolist()
    except Exception as e:  # noqa: BLE001
        out["error"] = str(e)[:160]
    # flow direction: principal direction pd1 per vertex, sampled as short segments
    try:
        import igl
        r = igl.principal_curvature(
            np.ascontiguousarray(mesh.vertices), np.ascontiguousarray(mesh.faces))
        pd1 = np.asarray(r[0])
        L = float(mesh.extents.max())
        step = max(1, len(mesh.vertices) // 600)   # cap arrows
        s = 0.03 * L
        for vi in range(0, len(mesh.vertices), step):
            p = mesh.vertices[vi]
            d = pd1[vi]
            nd = np.linalg.norm(d)
            if nd < 1e-9:
                continue
            d = d / nd * s
            out["flow"].append({"a": (p - d).tolist(), "b": (p + d).tolist()})
    except Exception:  # noqa: BLE001
        pass
    return JSONResponse(out)


@app.get("/api/origedges/{sid}")
def orig_edges(sid: str):
    """ALL edges of the INPUT mesh, in the SAME {vertices, edges} format as
    /api/quadedges — so the input wireframe uses the identical overlay machinery
    as the remeshed side. The wireframe of the real input topology, full grid."""
    sess = SESSIONS.get(sid)
    if sess is None or sess.orig is None:
        raise HTTPException(status_code=404, detail="unknown session")
    V, F = sess.orig
    F = np.asarray(F, np.int64).reshape(-1, 3)
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    return JSONResponse({"vertices": np.asarray(V).tolist(), "edges": e.tolist()})


@app.get("/api/procedges/{sid}")
def proc_edges(sid: str):
    """ALL edges of the REMESHED (proc) mesh — used for the right-pane wireframe in
    TRIS mode (where there are no quads). Same {vertices, edges} format."""
    sess = SESSIONS.get(sid)
    if sess is None or sess.proc is None:
        raise HTTPException(status_code=409, detail="not remeshed yet")
    V, F = sess.proc
    F = np.asarray(F, np.int64).reshape(-1, 3)
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    return JSONResponse({"vertices": np.asarray(V).tolist(), "edges": e.tolist()})


@app.get("/api/quadedges/{sid}")
def quad_edges(sid: str):
    """Return the quad border edges (no triangle diagonals) for a wireframe."""
    sess = SESSIONS.get(sid)
    if sess is None or sess.proc is None or sess.quads is None:
        raise HTTPException(status_code=409, detail="no quad mesh")
    V = sess.proc[0]
    q = np.asarray(sess.quads, np.int64).reshape(-1, 4)
    e = np.vstack([q[:, [0, 1]], q[:, [1, 2]], q[:, [2, 3]], q[:, [3, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    return JSONResponse({"vertices": V.tolist(), "edges": e.tolist()})


@app.get("/api/export/{sid}")
def export(sid: str, which: str = "proc", fmt: str = "glb"):
    """Download the original or remeshed mesh as glb/obj/ply/stl."""
    sess = SESSIONS.get(sid)
    if sess is None:
        raise HTTPException(status_code=404, detail="unknown session")
    if which == "proc" and sess.proc is None:
        raise HTTPException(status_code=409, detail="not remeshed yet")
    V, F = sess.proc if which == "proc" else sess.orig
    fmt = fmt.lower()
    if fmt not in ("glb", "obj", "ply", "stl"):
        raise HTTPException(status_code=400, detail="format must be glb/obj/ply/stl")
    # pass quads so OBJ keeps real 4-sided faces (proc only; GLB/PLY/STL stay tri)
    quads = sess.quads if which == "proc" else None
    data, media, suffix = eng.export_mesh(V, F, fmt, quads=quads)
    base = os.path.splitext(sess.name or "model")[0]
    tag = "remeshed" if which == "proc" else "original"
    fname = f"{base}_{tag}.{suffix}"
    return Response(
        content=data,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@app.post("/api/shrinkwrap")
def shrinkwrap(req: ShrinkwrapRequest):
    """Project the remeshed verts onto the original surface (Postprocess panel).

    Re-runnable: the pre-shrinkwrap verts are cached, so each call wraps the
    ORIGINAL remesh (not an already-wrapped one) and `reset` restores it. The
    `offset` (model units) is the inflation; the frontend passes it as a % of the
    bbox diagonal already converted to units."""
    import postprocess as pp

    sess = SESSIONS.get(req.session_id)
    if sess is None or sess.proc is None:
        raise HTTPException(status_code=409, detail="not remeshed yet")
    # remember the un-wrapped verts the first time so re-runs/reset are clean
    if sess.proc_prewrap is None:
        sess.proc_prewrap = sess.proc[0].copy()

    pv, pf = sess.proc_prewrap, sess.proc[1]   # always wrap from the pristine verts
    if req.reset:
        sess.proc = (sess.proc_prewrap, pf)
        sess.proc_prewrap = None
    else:
        ov, of = sess.orig
        try:
            new_v = pp.shrinkwrap(pv, pf, ov, of, mode=req.mode,
                                  distance=req.distance, offset=req.offset)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"shrinkwrap failed: {e}")
        sess.proc = (new_v, pf)
    sess.proc_color = _compute_colors(*sess.proc) if sess.proc_color is not None else None
    return {"session_id": req.session_id, "wrapped": not req.reset}


@app.post("/api/remesh")
def remesh(req: RemeshRequest):
    sess = SESSIONS.get(req.session_id)
    if sess is None or sess.orig is None:
        raise HTTPException(status_code=404, detail="unknown session")
    V, F = sess.orig
    if len(F) == 0:
        raise HTTPException(status_code=400, detail="empty mesh")

    # ---- HumanLogic quad mode ----
    if req.quad:
        _t0 = time.time()
        engine_name = req.quad_engine
        try:
            if req.quad_engine == "neurcross":
                qv, quads, qtris = eng.neurcross_quad(
                    V, F, target=req.quad_target, work_faces=req.quad_work_faces,
                )
            elif req.quad_engine == "quadwild":
                qv, quads, qtris = eng.quadwild_quad(
                    V, F, target=req.quad_target, work_faces=req.quad_work_faces,
                    sharp_mode=req.quad_sharp_mode, sharp_angle=req.quad_sharp_angle,
                )
            elif req.quad_engine == "humanlogic":
                qv, quads, qtris = eng.humanlogic_quad(
                    V, F, target=req.quad_target,
                    feature_angle=req.quad_feature_angle,
                    feature_weight=req.quad_feature_weight,
                    ridge_weight=req.quad_ridge_weight,
                    work_faces=req.quad_work_faces,
                )
            elif req.quad_engine == "shrinkwrap":
                qv, quads, qtris = eng.shrinkwrap_quad(
                    V, F, target=req.quad_target, work_faces=req.quad_work_faces,
                )
            else:  # "visibility" — full visibility-shell shrinkwrap (handles torus)
                qv, quads, qtris = eng.visibility_shell_quad(
                    V, F, target=req.quad_target, work_faces=req.quad_work_faces,
                )
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"{engine_name} failed: {e}")
        tri = eng.triangulate_quads(quads, qtris)
        sess.proc = (qv, tri)
        sess.proc_color = _compute_colors(qv, tri) if req.colorize else None
        sess.quads = quads
        sess.proc_prewrap = None   # a fresh remesh invalidates any prior shrinkwrap
        n_quads, n_tris = len(quads), len(qtris)
        proc_faces = n_quads + n_tris
        return {
            "session_id": req.session_id,
            "orig": {"vertices": len(V), "faces": len(F)},
            "proc": {"vertices": len(qv), "faces": proc_faces},
            "quads": n_quads,
            "tris_left": n_tris,
            "quad_ratio": round(100.0 * n_quads / max(proc_faces, 1), 1),
            "quad_engine": engine_name,
            "face_reduction_pct": round(100.0 * (1.0 - proc_faces / max(len(F), 1)), 2),
            "elapsed_ms": int((time.time() - _t0) * 1000),
        }

    params = eng.PipelineParams(
        flat_factor=float(req.flat_factor),
        detail_factor=float(req.detail_factor),
        contrast=float(req.contrast),
        feature_angle=float(req.feature_angle),
        iterations=int(req.iterations),
        pre_simplify=bool(req.pre_simplify),
        pre_simplify_target=float(req.pre_simplify_target),
        preserve_boundary=bool(req.preserve_boundary),
        max_work_faces=int(req.max_work_faces),
    )
    try:
        result = eng.run_pipeline(V, F, params)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"remesh failed: {e}")

    sess.proc = (result.vertices, result.faces)
    sess.proc_color = result.colors if req.colorize else None
    s = result.stats
    return {
        "session_id": req.session_id,
        "orig": {"vertices": s["orig_vertices"], "faces": s["orig_faces"]},
        "proc": {"vertices": s["proc_vertices"], "faces": s["proc_faces"]},
        "face_reduction_pct": s["face_reduction_pct"],
        "elapsed_ms": s["elapsed_ms"],
    }


# Static frontend (served at /)
app.mount("/", StaticFiles(directory=os.path.join(HERE, "static"), html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
