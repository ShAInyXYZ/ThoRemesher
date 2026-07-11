"""
AutoRemesher — curvature-ADAPTIVE pure-quad remeshing via Geogram QuadCover.

AutoRemesher (github.com/huxingyi/autoremesher, MIT; by Jeremy HU / Dust3D) is a
field-guided quad remesher whose engine is Geogram's FrameField + GlobalParam2d
::quad_cover — the Bompadre-Ray-Cross 4-symmetric frame field + integer-grid
global parameterization (the MIQ / QuadCover lineage). Unlike QuadWild's
field-TRACING approach (trace separatrices into patches, parameterize per
patch, reconcile edge counts), this is PARAMETERIZATION-FIRST: one global
integer-grid UV, then lift quads off its seams.

WHY THIS ENGINE (the payoff): curvature-adaptive quad DENSITY. AutoRemesher
computes a per-face scaling field from vertex curvature (max normal-turn per
unit edge length, normalized to the mesh average, pow(curv, -adaptivity),
clamped 0.3×–3×) and feeds it to quad_cover as the `adaptive_scaling` facet
attribute. The global UV grid is then DENSER where the surface curves and
RELAXED on flats — so small/detailed regions get more quads and large/flat
regions get fewer, instead of QuadWild's single global scaleFact that leaves
small areas detail-starved. Adaptivity 0 = uniform; 1 = fully adaptive.

The compiled extension (autoremesher_ext) binds AutoRemesher's orchestrator
class, which internally chains: curvature-adaptive isotropic pre-remesh →
FrameField → quad_cover (with the adaptive_scaling density field) →
QuadExtractor (UV→quad lifting: edge collapse, hole-fix, non-manifold cleanup).
Binding the orchestrator gets the ~600-line QuadExtractor surgery for free.

Field injection (future): AutoRemesher's Parameterizer takes an external
triangleFieldVectors, so a neural cross field (NeurCross) could feed quad_cover
directly — no QuadWild .rosy/do_remesh-0 dance. Not wired yet; documented here
as the natural follow-up seam.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

try:
    import autoremesher_ext as _ext
    _HAVE_EXT = True
except Exception:  # noqa: BLE001  (extension not built — QUAD mode falls back)
    _ext = None
    _HAVE_EXT = False


def available() -> bool:
    """True if the compiled autoremesher_ext extension is importable.

    Unlike _quadwild/ (a vendored binary) this is an in-process pybind11
    module, so availability is just 'did it build + import.'"""
    return _HAVE_EXT


def remesh(vertices, faces, target_quad=2500, adaptivity=0.7,
           sharp=True, sharp_thr=35.0, scaling=1.0):
    """Curvature-adaptive pure-quad remesh via Geogram QuadCover (AutoRemesher).

    target_quad : approximate OUTPUT quad count. AutoRemesher sizes its voxel
                  grid off a TARGET TRIANGLE COUNT (initializeVoxelSize derives
                  voxel = sqrt(area / (count · 0.433)) ), so we pass
                  target_triangle_count = 2 * target_quad  (1 quad ≈ 2 tris).
                  Empirically the mapping is loose — treat target_quad as a
                  density hint, not an exact count (same as QuadWild's path).
    adaptivity  : 0.0 = uniform density (QuadWild-like), 1.0 = fully
                  curvature-adaptive (dense where it curves, relaxed on flats).
                  THE feature this engine adds. Range [0,1].
    sharp       : True -> preserve crease edges (hard-surface); False -> smooth
                  (organic). Maps to sharp_edge_degrees: sharp uses sharp_thr,
                  not-sharp uses 180° (no edge is "sharp").
    sharp_thr   : dihedral angle (deg) above which an edge counts as sharp.
    scaling     : edge-scaling factor (1.0–4.0). Higher = coarser global grid.
                  Leave at 1.0 and control density via target_quad instead.

    Returns (V float64 (n,3), quads (k,4) int64, tris (l,3) int64) or None on
    failure (caller falls back). AutoRemesher occasionally leaves a few n-gons
    (pentagons/hexagons); these are tri-fanned into the tris bucket so nothing
    is dropped — same policy as quadwild._parse_quad_obj.

    CRASH ISOLATION: Geogram's quad_cover can SIGSEGV on certain island
    topologies (same class of fragility QuadWild's binary has). Since this is
    an in-process pybind11 module, a segfault would kill the whole server — so
    the remesh runs in a CHILD PROCESS. A crash there returns None (triggering
    the QuadWild fallback) instead of taking down uvicorn. QuadWild gets the
    same guarantee for free because it's already a subprocess.
    """
    if not available():
        return None
    V = np.ascontiguousarray(vertices, np.float64)
    F = np.ascontiguousarray(faces, np.int64)
    if not len(V) or not len(F):
        return None

    try:
        return _remesh_in_subprocess(V, F, target_quad, adaptivity,
                                     sharp, sharp_thr, scaling)
    except Exception:  # noqa: BLE001
        log.exception("autoremesher.remesh failed")
        return None


def _remesh_inproc(V, F, target_quad, adaptivity, sharp, sharp_thr, scaling):
    """The actual in-process call. Returns (V, quads, tris) or None. Can raise."""
    ar = _ext.AutoRemesher(V, F)
    ar.set_target_triangle_count(int(2 * max(target_quad, 50)))
    ar.set_gradient_adaptivity(float(np.clip(adaptivity, 0.0, 1.0)))
    ar.set_sharp_edge_degrees(float(sharp_thr) if sharp else 180.0)
    ar.set_smooth_normal_degrees(0.0)
    ar.set_scaling(float(scaling))
    if not ar.remesh():
        return None
    qv = np.asarray(ar.remeshed_vertices(), np.float64)
    polys = ar.remeshed_quads()
    if qv is None or not len(qv) or not len(polys):
        return None

    quads = [p for p in polys if len(p) == 4]
    tris = [p for p in polys if len(p) == 3]
    # tri-fan any n-gons (>4) into the tris bucket, matching quadwild's policy
    for ng in (p for p in polys if len(p) > 4):
        for k in range(1, len(ng) - 1):
            tris.append([ng[0], ng[k], ng[k + 1]])

    quads = np.asarray(quads, np.int64).reshape(-1, 4) if quads else np.empty((0, 4), np.int64)
    tris = np.asarray(tris, np.int64).reshape(-1, 3) if tris else np.empty((0, 3), np.int64)
    if not len(quads) and not len(tris):
        return None
    return qv, quads, tris


def _child_remesh(inp, out, target_quad, adaptivity, sharp, sharp_thr, scaling):
    """Module-level child-process target (must be top-level for spawn pickling).
    Reads V/F from inp (.npz), runs the remesh, writes result to out (.npz).
    Any failure -> empty out file (caller treats as None).

    Geogram/OpenNL prints solver convergence to stdout; we redirect the child's
    stdout to devnull so it doesn't flood the server console."""
    import numpy as np
    try:
        import os, sys
        # Geogram/OpenNL prints solver convergence via C printf to fd 1, which
        # bypasses Python's sys.stdout. Redirect at the OS fd level so it can't
        # leak to the parent's console.
        _devnull = os.open(os.devnull, os.O_WRONLY)
        _saved_stdout = os.dup(1)
        os.dup2(_devnull, 1)
        try:
            d = np.load(inp)
            res = _remesh_inproc(d["V"], d["F"], target_quad, adaptivity,
                                 sharp, sharp_thr, scaling)
        finally:
            os.dup2(_saved_stdout, 1)
            os.close(_devnull)
            os.close(_saved_stdout)
        if res is None:
            open(out, "wb").close()    # empty file = None signal
        else:
            qv, quads, tris = res
            np.savez(out, V=qv, quads=quads, tris=tris)
    except Exception:  # noqa: BLE001  (any error -> treat as failure)
        try:
            open(out, "wb").close()
        except Exception:
            pass


def _remesh_in_subprocess(V, F, target_quad, adaptivity, sharp, sharp_thr, scaling):
    """Run _remesh_inproc in a child process so a SIGSEGV returns None instead
    of crashing the server. V/F + result are passed via a temp .npz (arrays are
    too large for a pipe). Times out after 10 min."""
    import os, tempfile, shutil, multiprocessing as mp
    # fork (not spawn): the child inherits the parent's already-imported
    # autoremesher_ext + initialized Geogram state. spawn would re-import fresh
    # and can't find the local .so; fork is safe here because Geogram's globals
    # are per-process (copy-on-write) and the child runs one remesh then exits.
    mp_context = mp.get_context("fork")

    work = tempfile.mkdtemp(prefix="ar_sub_")
    inp = os.path.join(work, "in.npz")
    out = os.path.join(work, "out.npz")
    np.savez(inp, V=V, F=F)

    p = mp_context.Process(target=_child_remesh,
                           args=(inp, out, target_quad, adaptivity,
                                 sharp, sharp_thr, scaling))
    p.start()
    p.join(timeout=600)    # 10 min cap (matches QuadWild's default timeout)
    ok = p.exitcode == 0 and os.path.exists(out) and os.path.getsize(out) > 0
    res = None
    if ok:
        try:
            d = np.load(out)
            res = (d["V"], d["quads"], d["tris"])
        except Exception:  # noqa: BLE001
            res = None
    if p.is_alive():
        p.terminate()
        p.join(timeout=5)
    shutil.rmtree(work, ignore_errors=True)
    if p.exitcode not in (0, None) and p.exitcode is not None:
        log.warning("autoremesher child exited %d (likely segfault) — falling back",
                    p.exitcode)
    return res
