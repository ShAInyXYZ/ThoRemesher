"""
QuadWild-BiMDF wrapper — modern feature-line-driven PURE-quad remeshing.

QuadWild-BiMDF (CGG Bern, TOG 2023; github.com/cgg-bern/quadwild-bimdf) is the
maintained successor to QuadWild (Pietroni 2021). Its free Bi-MDF / libSatsuma
min-deviation-flow solver does the edge-count reconciliation we used to hand-roll,
producing 100% pure-quad watertight meshes with sharp edges preserved.

It ships as two Linux CLI binaries (no Python API):
  step 1  `quadwild  mesh.obj 2 <prep_config>`        -> mesh_rem_p0.obj
  step 2  `quad_from_patches  mesh_rem_p0.obj 1 <flow_config>` -> *_quadrangulation.obj

CRITICAL: the prebuilt binary SEGFAULTS (rc=139) on cleanup AFTER writing its
output. So success is decided by the OUTPUT FILE EXISTING, never by the exit code.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile

import numpy as np

log = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
_QW_DIR = os.path.join(_HERE, "_quadwild")
_QUADWILD = os.path.join(_QW_DIR, "quadwild")
_QFP = os.path.join(_QW_DIR, "quad_from_patches")
_PREP_MECH = os.path.join(_QW_DIR, "config", "prep_config", "basic_setup_Mechanical.txt")
_PREP_ORG = os.path.join(_QW_DIR, "config", "prep_config", "basic_setup_Organic.txt")
_FLOW = os.path.join(_QW_DIR, "config", "main_config", "flow.txt")


def available():
    """True if the QuadWild binaries + configs are present and executable."""
    return all(os.path.exists(p) for p in (_QUADWILD, _QFP, _PREP_MECH, _FLOW))


def _write_obj(path, V, F):
    with open(path, "w") as fh:
        for v in V:
            fh.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for f in F:
            fh.write("f " + " ".join(str(int(i) + 1) for i in f) + "\n")


def _read_obj(path):
    """Read an OBJ; returns (V, faces) where faces is a list of vertex-id lists
    (quads, tris, n-gons all preserved)."""
    V, Faces = [], []
    with open(path) as fh:
        for ln in fh:
            if ln.startswith("v "):
                V.append([float(x) for x in ln.split()[1:4]])
            elif ln.startswith("f "):
                Faces.append([int(p.split("/")[0]) - 1 for p in ln.split()[1:]])
    return np.asarray(V, np.float64), Faces


def _run(cmd, cwd, timeout):
    """Run a QuadWild binary. Returns the CompletedProcess; we IGNORE the return
    code (the binary segfaults on cleanup) and judge success by output files."""
    try:
        return subprocess.run(cmd, cwd=cwd, timeout=timeout,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return None


def _prep_config(work, sharp, sharp_thr, do_remesh=True, name="prep.txt"):
    """Write a prep config with the chosen sharp-edge threshold.

    do_remesh=False is REQUIRED for the field-injection path: it tells QuadWild
    to keep the (already-remeshed) triangulation so an externally-supplied .rosy
    field stays face-index-aligned. Re-remeshing would invalidate the field."""
    path = os.path.join(work, name)
    thr = sharp_thr if sharp else -1
    alpha = 0.01 if sharp else 0.02
    with open(path, "w") as fh:
        fh.write(f"do_remesh {1 if do_remesh else 0}\nsharp_feature_thr {thr}\n"
                 f"alpha {alpha}\nscaleFact 1\n")
    return path


def _flow_config(work, scale_fact):
    """Copy the default flow config but override scaleFact (the density knob:
    higher scaleFact = COARSER / fewer quads)."""
    path = os.path.join(work, "flow.txt")
    with open(_FLOW) as fh:
        lines = fh.readlines()
    out = []
    for ln in lines:
        if ln.startswith("scaleFact "):
            out.append(f"scaleFact {scale_fact}\n")
        else:
            out.append(ln)
    with open(path, "w") as fh:
        fh.writelines(out)
    return path


def remesh(vertices, faces, sharp=True, sharp_thr=35.0, scale_fact=1.0,
           timeout=120, field_fn=None):
    """Quad-remesh with QuadWild-BiMDF.

    sharp       : True -> preserve crease edges (Mechanical), False -> smooth (Organic)
    sharp_thr   : dihedral angle (deg) above which an edge is a sharp feature
    scale_fact  : density knob — HIGHER = coarser/fewer quads, LOWER = denser
                  (scaleFact in QuadWild's flow config; ~0.5..2 useful range)
    field_fn    : optional callback field_fn(rem_obj_path, rosy_path) that WRITES a
                  cross field (e.g. a NeurCross neural field) to rosy_path, sampled
                  on rem_obj_path's faces. When given, QuadWild IMPORTS that field
                  for the trace+quadrangulation instead of computing its own — so the
                  quads follow the supplied field. (See _remesh_with_field for why
                  the naive "overwrite mesh_rem.rosy" approach does NOT work.)

    Returns (V, quads Nx4, tris Nx3) or None on failure (caller falls back).
    """
    if not available():
        return None
    work = tempfile.mkdtemp(prefix="qw_")
    try:
        inp = os.path.join(work, "mesh.obj")
        _write_obj(inp, np.asarray(vertices, np.float64), np.asarray(faces, np.int64))
        if field_fn is not None:
            p0 = _remesh_with_field(work, inp, sharp, sharp_thr, field_fn, timeout)
        else:
            # plain path: mode 2 remeshes + computes field + traces -> mesh_rem_p0.*
            prep = _prep_config(work, sharp, sharp_thr)
            _run([_QUADWILD, inp, "2", prep], cwd=_QW_DIR, timeout=timeout)
            p0 = os.path.join(work, "mesh_rem_p0.obj")
        if p0 is None or not os.path.exists(p0):
            return None
        # final step: quad generation via Bi-MDF flow (segfaults AFTER writing output)
        flow = _flow_config(work, scale_fact)
        base = os.path.splitext(os.path.basename(p0))[0]
        _run([_QFP, p0, "1", flow], cwd=_QW_DIR, timeout=timeout)
        out = os.path.join(work, f"{base}_1_quadrangulation.obj")
        if not os.path.exists(out):               # genuinely failed
            return None
        return _parse_quad_obj(out)
    except Exception:  # noqa: BLE001
        log.exception("quadwild.remesh failed")  # don't silently fall back without a trace
        return None
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _remesh_with_field(work, inp, sharp, sharp_thr, field_fn, timeout):
    """Inject an external cross field and return the traced `_p0.obj` path (or None).

    Why this dance is necessary (verified against QuadWild source): a single
    `quadwild ... 2` call ALWAYS re-runs remesh+field internally, overwriting any
    .rosy you placed. To make it IMPORT an external field you must (a) pass the
    .rosy as a CLI arg so it loads instead of computing, and (b) set do_remesh 0
    on an already-remeshed mesh so face indices stay aligned to the field.
    """
    # 1) mode 1: remesh + compute field -> mesh_rem.{obj,rosy,sharp}
    prep = _prep_config(work, sharp, sharp_thr)
    _run([_QUADWILD, inp, "1", prep], cwd=_QW_DIR, timeout=timeout)
    rem = os.path.join(work, "mesh_rem.obj")
    sharp_f = os.path.join(work, "mesh_rem.sharp")
    if not os.path.exists(rem):
        return None
    # 2) build the external field on mesh_rem's faces (per-face .rosy)
    my_rosy = os.path.join(work, "my.rosy")
    field_fn(rem, my_rosy)
    if not os.path.exists(my_rosy):
        return None
    # 3) mode 2 on the REMESHED mesh, do_remesh 0, importing our field
    prep_nr = _prep_config(work, sharp, sharp_thr, do_remesh=False, name="prep_nr.txt")
    _run([_QUADWILD, rem, "2", prep_nr, sharp_f, my_rosy], cwd=_QW_DIR, timeout=timeout)
    p0 = os.path.join(work, "mesh_rem_rem_p0.obj")   # note: input was *_rem -> *_rem_rem
    return p0 if os.path.exists(p0) else None


def _parse_quad_obj(out):
    """Read a quadrangulation OBJ -> (V, quads Nx4, tris Nx3) or None."""
    V, Faces = _read_obj(out)
    if not len(V) or not Faces:
        return None
    quads = np.asarray([f for f in Faces if len(f) == 4], np.int64).reshape(-1, 4)
    tris = [f for f in Faces if len(f) == 3]
    # triangulate any rare n-gons (fan) into the tri bucket so nothing is dropped
    for ng in (f for f in Faces if len(f) > 4):
        tris.extend([ng[0], ng[k], ng[k + 1]] for k in range(1, len(ng) - 1))
    tris = np.asarray(tris, np.int64).reshape(-1, 3)
    if not len(quads) and not len(tris):
        return None
    return V, quads, tris
