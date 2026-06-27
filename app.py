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
    __slots__ = ("path", "orig", "proc", "orig_color", "proc_color", "name", "error")

    def __init__(self, path, name):
        self.path = path
        self.name = name
        self.orig = None        # (V, F)
        self.proc = None        # (V, F)
        self.orig_color = None  # (V,4)
        self.proc_color = None  # (V,4)
        self.error = None


SESSIONS: Dict[str, Session] = {}


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


def _compute_colors(V, F):
    try:
        return eng.curvature_colors(eng.detail_score(V, F))
    except Exception:  # noqa: BLE001
        return None


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
    sess.orig = (V, F)
    sess.orig_color = None  # computed lazily on first /api/model?color=1 request
    SESSIONS[sid] = sess
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
            colors = sess.orig_color
            if colors is None:
                colors = _compute_colors(V, F)
                sess.orig_color = colors
    data = eng.to_glb_bytes(V, F, colors)
    return Response(content=data, media_type="model/gltf-binary")


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
    data, media, suffix = eng.export_mesh(V, F, fmt)
    base = os.path.splitext(sess.name or "model")[0]
    tag = "remeshed" if which == "proc" else "original"
    fname = f"{base}_{tag}.{suffix}"
    return Response(
        content=data,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@app.post("/api/remesh")
def remesh(req: RemeshRequest):
    sess = SESSIONS.get(req.session_id)
    if sess is None or sess.orig is None:
        raise HTTPException(status_code=404, detail="unknown session")
    V, F = sess.orig
    if len(F) == 0:
        raise HTTPException(status_code=400, detail="empty mesh")

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
