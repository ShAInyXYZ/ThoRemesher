"""
NeurCross-steered quad remeshing (hybrid).

NeurCross (SIGGRAPH 2025, github.com/QiujieDong/NeurCross) computes a high-quality
NEURAL cross field via a per-shape self-supervised SDF network. It does NOT output
a mesh — the paper extracts quads with libigl+libQEx (integer-grid param, which
modern libigl dropped).

This module bypasses that gap: it runs NeurCross to get the neural cross field,
then injects it into QuadWild's pipeline (which externalises its own field as a
`.rosy` file). QuadWild's fast, robust Bi-MDF extraction then produces a watertight
pure-quad mesh GUIDED BY the neural field. Best of both, no CoMISo build.

Cost: NeurCross overfits a network per mesh -> minutes/mesh on a GPU. Falls back
to vanilla QuadWild if NeurCross is unavailable or fails.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile

import numpy as np
import trimesh
from scipy.spatial import cKDTree

import quadwild as qw

log = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
_NC_DIR = os.path.join(_HERE, "_neurcross")
_NC_TRAIN = os.path.join(_NC_DIR, "quad_mesh", "train_quad_mesh.py")


def available():
    """NeurCross runnable iff its training script + a CUDA torch are present."""
    if not os.path.exists(_NC_TRAIN):
        return False
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def _run_neurcross(input_ply, out_dir, n_samples, timeout):
    """Run NeurCross training; returns the path to the saved cross-field .txt."""
    shape = os.path.splitext(os.path.basename(input_ply))[0]
    cmd = ["python3", _NC_TRAIN,
           "--data_path", input_ply,
           "--logdir", out_dir,
           "--n_samples", str(n_samples)]
    try:
        subprocess.run(cmd, cwd=os.path.join(_NC_DIR, "quad_mesh"),
                       timeout=timeout, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        pass    # NeurCross checkpoints the field every 500 iters — even on a
                # timeout we use the latest saved field (progressive, not all-or-nothing)
    cf_dir = os.path.join(out_dir, shape, "save_crossField")
    if not os.path.isdir(cf_dir):
        return None
    # the highest-iteration saved field (best); robust numeric sort by iter index
    def _iter(f):
        try:
            return int(f.rsplit("_", 1)[-1].split(".")[0])
        except Exception:  # noqa: BLE001
            return -1
    fields = sorted((f for f in os.listdir(cf_dir) if f.endswith(".txt")), key=_iter)
    return os.path.join(cf_dir, fields[-1]) if fields else None


def _transfer_field(nc_field_path, nc_input_mesh, rem_obj_path, rosy_path):
    """Transfer NeurCross's per-face cross field onto QuadWild's remeshed faces and
    write a QuadWild-format `.rosy` (line1=nfaces, line2=4, then a 3D dir/face)."""
    cf = np.loadtxt(nc_field_path)            # (N, 6): alpha_xyz + beta_xyz
    if cf.ndim != 2 or cf.shape[1] < 3:
        return False
    alpha = cf[:, :3]
    src = trimesh.load(nc_input_mesh, process=False)
    # NeurCross samples per input face -> use face centers; else fall back to verts
    src_pts = src.triangles_center if len(src.faces) == len(alpha) else src.vertices
    if len(src_pts) != len(alpha):
        m = min(len(src_pts), len(alpha))
        src_pts, alpha = src_pts[:m], alpha[:m]
    rem = trimesh.load(rem_obj_path, process=False)
    fc = rem.triangles_center
    fn = rem.face_normals
    # nearest neural point per QuadWild face, project to face tangent, normalise
    _, idx = cKDTree(src_pts).query(fc)
    field = alpha[idx]
    field = field - np.sum(field * fn, axis=1, keepdims=True) * fn
    norm = np.linalg.norm(field, axis=1, keepdims=True).ravel()
    # where the neural dir was ~normal to the face (projection ~0), substitute a
    # valid in-plane direction so the .rosy has no zero-length (degenerate) entries
    bad = norm < 1e-9
    if bad.any():
        ref = np.tile([1.0, 0.0, 0.0], (bad.sum(), 1))
        ref[np.abs(fn[bad, 0]) > 0.9] = [0.0, 1.0, 0.0]   # avoid parallel-to-normal
        field[bad] = ref - np.sum(ref * fn[bad], axis=1, keepdims=True) * fn[bad]
        norm[bad] = np.linalg.norm(field[bad], axis=1)
    field = field / np.maximum(norm, 1e-12)[:, None]
    # QuadWild .rosy: line1 = #faces, line2 = 4 (RoSy symmetry), one dir per face
    with open(rosy_path, "w") as f:
        f.write(f"{len(rem.faces)}\n4\n")
        f.write("\n".join(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}" for v in field) + "\n")
    return True


def remesh(vertices, faces, scale_fact=1.0, n_samples=10000, sharp=False,
           sharp_thr=35.0, timeout=240):
    """NeurCross-steered quad remesh. Returns (V, quads, tris) or None.

    Runs NeurCross to get a neural cross field, then drives QuadWild's extraction
    with it. `sharp=False` (Organic) by default — NeurCross's strength is smooth/
    organic shapes. ~minutes/mesh on GPU.
    """
    if not available() or not qw.available():
        return None
    work = tempfile.mkdtemp(prefix="nc_")
    try:
        # NeurCross optimizes a network PER vertex-sample; its cost scales with
        # mesh size, so feed it a COARSE proxy (the neural field is smooth and
        # gets resampled onto QuadWild's own faces anyway). Decimate to a light
        # mesh so a fit takes ~1 min, not many.
        nc_V, nc_F = vertices, faces
        try:
            import remesh_engine as eng
            if len(faces) > 3000:
                nc_V, nc_F = eng.preprocess_for_quad(
                    np.asarray(vertices, np.float64), np.asarray(faces, np.int64),
                    work_faces=2500, smooth=0)
        except Exception:  # noqa: BLE001
            pass
        ply = os.path.join(work, "shape.ply")
        trimesh.Trimesh(np.asarray(nc_V, np.float64),
                        np.asarray(nc_F, np.int64), process=False).export(ply)
        cf = _run_neurcross(ply, work, n_samples, timeout)
        if cf is None:
            return None
        # field injector: overwrite QuadWild's .rosy with the neural field
        def field_fn(rem_obj, rosy):
            _transfer_field(cf, ply, rem_obj, rosy)
        return qw.remesh(vertices, faces, sharp=sharp, sharp_thr=sharp_thr,
                         scale_fact=scale_fact, timeout=180, field_fn=field_fn)
    except Exception:  # noqa: BLE001
        log.exception("neurcross.remesh failed")
        return None
    finally:
        shutil.rmtree(work, ignore_errors=True)
