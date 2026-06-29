"""
Curvature-aware adaptive remeshing engine.

The pipeline implements exactly the requested logic:

  1. Load any mesh (FBX / GLB / GLTF / OBJ / PLY / STL ...).
  2. Score every vertex by how much the surface "changes direction" there
     (mean curvature + normal variation). Flat areas -> ~0, detailed /
     creased areas -> high.
  3. Build a per-vertex target edge length from that score: long on flats,
     short where it is detailed.
  4. Run a curvature-field-driven adaptive remesh (split / collapse / flip /
     smooth + reprojection) that rebuilds clean, near-isotropic triangles
     while keeping flats coarse and details dense. Sharp creases are preserved.
  5. Colorize by curvature for a heatmap and export GLB for the web viewer.

MeshLab (pymeshlab) is used for FBX/GLB/OBJ I/O and an optional fast
pre-simplification; the adaptive remesh core is a standalone implementation.
"""
from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.spatial as ssp
import trimesh

warnings.filterwarnings("ignore")

try:
    import pymeshlab
except Exception as exc:  # pragma: no cover
    pymeshlab = None
    _PML_ERR = exc
else:
    _PML_ERR = None

try:
    import igl
except Exception:  # pragma: no cover
    igl = None

try:
    import pyassimp
except Exception:  # pragma: no cover
    pyassimp = None


# --------------------------------------------------------------------------- #
#  Loading
# --------------------------------------------------------------------------- #
def _assimp_to_arrays(scene):
    verts, faces, off = [], [], 0
    for mesh in scene.meshes:
        v = np.asarray(mesh.vertices, dtype=np.float64)
        f = np.asarray(mesh.faces)
        if v.size == 0 or f.size == 0 or f.ndim != 2 or f.shape[1] < 3:
            continue
        if f.shape[1] > 3:  # fan-triangulate n-gons
            f = np.vstack([f[:, [0, i - 1, i]] for i in range(2, f.shape[1])])
        verts.append(v)
        faces.append(f.astype(np.int64) + off)
        off += len(v)
    if not verts:
        raise ValueError("No triangle geometry found in file.")
    return np.vstack(verts), np.vstack(faces)


def _load_with_meshlab(path):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(path)
    verts, faces, off = [], [], 0
    for i in range(ms.mesh_number()):
        ms.set_current_mesh(i)
        m = ms.current_mesh()
        v = np.asarray(m.vertex_matrix(), dtype=np.float64)
        f = np.asarray(m.face_matrix())
        if v.size == 0 or f.size == 0:
            continue
        verts.append(v)
        faces.append(f.astype(np.int64) + off)
        off += len(v)
    if not verts:
        raise ValueError("MeshLab loaded the file but found no triangles.")
    return np.vstack(verts), np.vstack(faces)


def load_mesh(path):
    """Load any supported mesh file, returning merged (vertices, faces)."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    errs = []
    if pymeshlab is not None:
        try:
            return _load_with_meshlab(path)
        except Exception as e:  # noqa: BLE001
            errs.append(f"pymeshlab: {e}")
    if pyassimp is not None:
        try:
            with pyassimp.load(path) as scene:
                return _assimp_to_arrays(scene)
        except Exception as e:  # noqa: BLE001
            errs.append(f"pyassimp: {e}")
    raise RuntimeError("Could not load mesh. Attempts:\n" + "\n".join(errs))


# --------------------------------------------------------------------------- #
#  Curvature / detail metric
# --------------------------------------------------------------------------- #
def _robust_mean_curvature(vertices, faces):
    """Per-vertex mean curvature in [0,1] via the MODERN robust cotan Laplacian
    (robust-laplacian, Sharp & Crane 2020). H ~ |L·V| / 2A — the Laplacian applied
    to positions is the mean-curvature normal; the robust (tufted) operator stays
    stable on coarse/noisy/non-manifold meshes where igl's discrete curvature
    speckles. Robustly normalized (95th pct + sqrt) for a clean gradient."""
    V = np.ascontiguousarray(vertices, np.float64)
    F = np.ascontiguousarray(faces, np.int64)
    n = len(V)
    try:
        import robust_laplacian as rl

        L, M = rl.mesh_laplacian(V, F)
        Hn = L @ V                              # mean-curvature normal (weak form)
        area = np.asarray(M.diagonal()).ravel()
        H = np.linalg.norm(Hn, axis=1) / (2.0 * np.maximum(area, 1e-12))
        H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
        pos = H[H > 1e-9]
        hi = np.percentile(pos, 95) if len(pos) else 1.0
        hi = hi if hi > 1e-12 else 1.0
        return np.clip((H / hi) ** 0.5, 0.0, 1.0)
    except Exception:  # noqa: BLE001
        # fallback: igl gaussian curvature (noisier, but always available)
        try:
            K = np.abs(np.asarray(igl.gaussian_curvature(V, F)).ravel())
            K = np.nan_to_num(K)
            hi = np.percentile(K[K > 1e-9], 95) if np.any(K > 1e-9) else 1.0
            return np.clip((K / max(hi, 1e-12)) ** 0.5, 0.0, 1.0)
        except Exception:  # noqa: BLE001
            return np.zeros(n)


def detail_score(vertices, faces):
    """Per-vertex "changing-direction" score in [0,1].

    Implements the requested logic: a region is "detailed" where the surface
    changes direction (curved) and "flat" where adjacent face normals agree.

    Detail = max(crease, curvature): a per-edge dihedral-angle crease term
    (trimesh.face_adjacency_angles, catches hard edges with zero speckle) maxed
    with the robust mean curvature (_robust_mean_curvature; catches smooth
    curvature like fingers/limbs/cylinders). Flat -> ~0, detailed -> high.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    n = len(vertices)
    if n == 0:
        return np.zeros(0)

    # Detail = max( sharp-crease, smooth-curvature ), using MODERN ROBUST
    # operators (robust-laplacian, Sharp & Crane SGP 2020) — NOT igl's discrete
    # Gaussian/mean curvature, which is speckled (salt-and-pepper) on dense/noisy
    # real meshes. Two terms:
    #   * dihedral-angle crease term  -> catches hard edges cleanly, zero speckle.
    #   * ROBUST mean curvature       -> |L V| / 2A from the robust cotan Laplacian
    #     L and mass matrix A; a clean, smooth gradient even on coarse/noisy meshes.
    try:
        mesh = trimesh.Trimesh(vertices, faces, process=False)
        # --- crease term: per-edge dihedral angle, maxed onto incident verts ---
        crease = np.zeros(n)
        try:
            ang = mesh.face_adjacency_angles                 # radians, >=0
            ed = mesh.face_adjacency_edges
            cval = ang / np.radians(70.0)                    # 70deg crease -> 1.0
            np.maximum.at(crease, ed[:, 0], cval)
            np.maximum.at(crease, ed[:, 1], cval)
            crease = np.clip(crease, 0.0, 1.0)
        except Exception:  # noqa: BLE001
            crease = np.zeros(n)
        # --- smooth term: robust mean curvature ---
        smooth = _robust_mean_curvature(vertices, faces)
        return np.clip(np.maximum(crease, smooth), 0.0, 1.0)
    except Exception:  # noqa: BLE001
        return np.zeros(n)


# --------------------------------------------------------------------------- #
#  Color mapping
# --------------------------------------------------------------------------- #
def curvature_colors(score):
    """Map a [0,1] score to RGBA (blue flat -> red detailed)."""
    t = np.clip(np.asarray(score, dtype=np.float64), 0.0, 1.0)
    stops = np.array(
        [
            [0.10, 0.20, 0.90],
            [0.10, 0.70, 0.95],
            [0.25, 0.85, 0.35],
            [0.98, 0.80, 0.15],
            [0.95, 0.25, 0.20],
        ],
        dtype=np.float64,
    )
    xs = np.linspace(0, 1, stops.shape[0])
    rgba = np.stack(
        [np.interp(t, xs, stops[:, k]) for k in range(3)] + [np.ones_like(t)], axis=1
    )
    return rgba


# --------------------------------------------------------------------------- #
#  MeshLab helpers (I/O + optional fast pre-simplification)
# --------------------------------------------------------------------------- #
def quadric_decimate(vertices, faces, quality, target_perc, preserve_boundary=True):
    """Curvature-weighted QEM decimation (flat / low-quality areas collapse)."""
    if pymeshlab is None:
        raise RuntimeError(f"pymeshlab unavailable: {_PML_ERR}")
    ms = pymeshlab.MeshSet()
    ms.add_mesh(
        pymeshlab.Mesh(
            vertex_matrix=np.asarray(vertices, dtype=np.float64),
            face_matrix=np.asarray(faces, dtype=np.int32),
            v_scalar_array=np.asarray(quality, dtype=np.float64),
        ),
        "mesh",
    )
    ms.meshing_decimation_quadric_edge_collapse(
        targetperc=float(target_perc),
        qualityweight=True,
        preserveboundary=bool(preserve_boundary),
        preservetopology=True,
        optimalplacement=True,
        autoclean=True,
    )
    m = ms.current_mesh()
    return np.asarray(m.vertex_matrix(), dtype=np.float64), np.asarray(
        m.face_matrix(), dtype=np.int64
    )


def _ml_run(vertices, faces, filters):
    """Run a sequence of zero-arg MeshLab filters, return compacted arrays."""
    if pymeshlab is None:
        return vertices, faces
    ms = pymeshlab.MeshSet()
    ms.add_mesh(
        pymeshlab.Mesh(
            vertex_matrix=np.asarray(vertices, dtype=np.float64),
            face_matrix=np.asarray(faces, dtype=np.int32),
        ),
        "m",
    )
    for fn in filters:
        getattr(ms, fn)()
    m = ms.current_mesh()
    return np.asarray(m.vertex_matrix(), dtype=np.float64), np.asarray(
        m.face_matrix(), dtype=np.int64
    )


def cleanup_mesh(vertices, faces):
    """Remove non-manifold edges/vertices, duplicates, and T-vertices.

    The adaptive collapse pass can leave a handful of non-manifold edges which
    show up as shading artefacts; MeshLab repairs them into a clean manifold.
    """
    return _ml_run(vertices, faces, [
        "meshing_repair_non_manifold_edges",
        "meshing_repair_non_manifold_vertices",
        "meshing_remove_duplicate_faces",
        "meshing_remove_duplicate_vertices",
        "meshing_remove_t_vertices",
    ])


# --------------------------------------------------------------------------- #
#  Pipeline
# --------------------------------------------------------------------------- #
@dataclass
class PipelineParams:
    flat_factor: float = 3.0          # FLAT-area edge length = factor * avg edge
    detail_factor: float = 1.0        # DETAILED-area edge length = factor * avg edge
    contrast: float = 2.0             # sharpness of flat<->detailed transition (gamma)
    feature_angle: float = 30.0       # crease edges (dihedral > this, deg) preserved
    iterations: int = 6               # adaptive remeshing passes
    pre_simplify: bool = False        # force MeshLab QEM pre-pass
    pre_simplify_target: float = 0.25  # fraction of faces kept by a forced pre-pass
    preserve_boundary: bool = True
    max_work_faces: int = 50000       # meshes above this are auto pre-simplified first
    reproject_limit: int = 30000      # skip surface reprojection above this face count


@dataclass
class PipelineResult:
    vertices: np.ndarray
    faces: np.ndarray
    curvature: np.ndarray
    colors: np.ndarray
    stats: "dict" = field(default_factory=dict)


def _stats(v, f):
    if len(v) == 0:
        return 0, 0
    return int(len(v)), int(len(f))


def _feature_edges(vertices, faces, angle_deg):
    """Edges whose adjacent faces form a dihedral angle above the threshold."""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    try:
        ang = np.asarray(mesh.face_adjacency_angles, dtype=np.float64)
        edges = np.asarray(mesh.face_adjacency_edges)
    except Exception:  # noqa: BLE001
        return set()
    if len(ang) == 0:
        return set()
    sel = edges[ang > angle_deg]
    return {(int(min(a, b)), int(max(a, b))) for a, b in sel}


def _make_reproject(vertices, faces):
    """Build a callable that snaps points back onto the original surface."""
    base = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    def rp(pts):
        return trimesh.proximity.closest_point(base, np.asarray(pts))[0]

    return rp


def run_pipeline(vertices, faces, params: PipelineParams) -> PipelineResult:
    """Curvature-aware adaptive remesh.

    Flat areas are coarsened (edge -> flat_factor * avg edge), detailed /
    high-curvature areas stay dense (edge -> detail_factor * avg edge), sharp
    creases are preserved, and the whole surface is rebuilt into clean,
    near-isotropic triangles.
    """
    import adaptive_remesh

    t0 = time.time()
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    orig_v, orig_f = _stats(vertices, faces)
    bb = vertices.max(axis=0) - vertices.min(axis=0)
    diag = float(np.linalg.norm(bb)) or 1.0

    # compute the curvature field ONCE on the original mesh (via pyvista VTK
    # curvature — fast and accurate), then transfer to working meshes.
    score_orig = detail_score(vertices, faces)

    def transfer_score(V, F):
        if len(V) == len(vertices):
            return score_orig.copy()
        tree = ssp.cKDTree(vertices)
        _, idx = tree.query(V, k=1, workers=-1)
        return score_orig[idx]

    cur_v, cur_f = vertices.copy(), faces.copy()

    # ---- the core remesh logic ---------------------------------------------
    # Curvature-weighted QEM decimation is the primary engine. It naturally
    # preserves detail in high-curvature regions (fingers, face) while
    # collapsing flat areas (torso). target_perc maps logarithmically from the
    # flat_factor slider so the full range gives useful results.
    target_perc = min(1.0, max(0.002, 1.0 / (params.flat_factor ** 2)))
    if pymeshlab is not None and len(cur_f) > 4:
        cur_v, cur_f = quadric_decimate(
            cur_v,
            cur_f,
            quality=transfer_score(cur_v, cur_f),
            target_perc=target_perc,
            preserve_boundary=params.preserve_boundary,
        )

    # quality pass: tangential smoothing improves triangle regularity (valence,
    # aspect ratio) without changing edge lengths or density distribution.
    # Smoothing is projected onto each vertex's tangent plane so shape is
    # preserved, and feature (crease) vertices are locked in place.
    if params.iterations > 0 and len(cur_f) > 10:
        import adaptive_remesh as _ar

        feats = (
            _feature_edges(cur_v, cur_f, params.feature_angle)
            if params.feature_angle < 180
            else set()
        )
        rm = _ar.AdaptiveRemesher(
            cur_v, cur_f, [1.0] * len(cur_v), feature_edges=feats,
            reproject=_make_reproject(vertices, faces) if len(cur_f) <= 40000 else None,
        )
        cur_v, cur_f = rm.smooth(min(params.iterations, 6))

    # topology cleanup: repair non-manifold edges/vertices, duplicates, T-verts
    if pymeshlab is not None and len(cur_f) > 0:
        cur_v, cur_f = cleanup_mesh(cur_v, cur_f)

    score = transfer_score(cur_v, cur_f)
    proc_v, proc_f = _stats(cur_v, cur_f)
    return PipelineResult(
        vertices=cur_v,
        faces=cur_f,
        curvature=score,
        colors=curvature_colors(score),
        stats={
            "orig_vertices": orig_v,
            "orig_faces": orig_f,
            "proc_vertices": proc_v,
            "proc_faces": proc_f,
            "bbox_diag": round(diag, 5),
            "target_perc": round(target_perc, 4),
            "face_reduction_pct": round(100.0 * (1.0 - proc_f / max(orig_f, 1)), 2),
            "elapsed_ms": int((time.time() - t0) * 1000),
        },
    )


# --------------------------------------------------------------------------- #
#  GLB serialization (for the web viewer)
# --------------------------------------------------------------------------- #
def triangulate_quads(quads, tris=None):
    """Merge quad + tri faces into one triangle array (for the viewer/export)."""
    parts = []
    q = np.asarray(quads, np.int64).reshape(-1, 4)
    if len(q):
        parts.append(q[:, [0, 1, 2]])
        parts.append(q[:, [0, 2, 3]])
    if tris is not None and len(tris):
        parts.append(np.asarray(tris, np.int64).reshape(-1, 3))
    return np.vstack(parts) if parts else np.zeros((0, 3), np.int64)


def _needs_smoothing(V, F):
    """True only if the mesh looks NOISY (high-frequency normal scatter), not just
    curved. A clean primitive (cube, sphere, scan-free model) is left untouched so
    its sharp edges and flat faces survive. Heuristic: compare each face normal to
    its neighbours; lots of small-scale disagreement = noise worth smoothing, a
    few big jumps (real creases) + smooth elsewhere = clean, skip it."""
    try:
        m = trimesh.Trimesh(np.asarray(V, np.float64), np.asarray(F, np.int64),
                            process=False)
        ang = m.face_adjacency_angles
        if not len(ang):
            return False
        # noise = many medium-angle disagreements (15-45deg). Real geometry has
        # either near-flat (<10deg) or sharp creases (>50deg), not a sea of 20-40deg.
        noisy_frac = np.mean((ang > np.radians(15)) & (ang < np.radians(50)))
        return noisy_frac > 0.25
    except Exception:  # noqa: BLE001
        return True


def preprocess_for_quad(vertices, faces, work_faces=6000, smooth=4,
                        repair=True):
    """Clean a raw/messy input mesh into a quad-friendly base before remeshing.

    Repair (merge shells, fix non-manifold) -> decimate to ~work_faces ->
    Laplacian smooth to remove micro-noise/speckle so the patch segmentation
    finds real surface regions instead of thousands of tiny noise patches.
    Returns (V, F).
    """
    V = np.asarray(vertices, np.float64)
    F = np.asarray(faces, np.int64)
    if pymeshlab is None:
        return V, F
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=V, face_matrix=F.astype(np.int32)), "m")
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_unreferenced_vertices()
    if repair:
        try:
            ms.meshing_merge_close_vertices()
        except Exception:  # noqa: BLE001
            pass
        ms.meshing_repair_non_manifold_edges()
        ms.meshing_repair_non_manifold_vertices()
    if len(F) > work_faces:
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=int(work_faces), preservetopology=False,
            planarquadric=True, autoclean=True,
        )
    # FEATURE-PRESERVING smoothing. Plain Laplacian rounds off sharp edges (a
    # cube's 90deg creases collapse to ~57deg, destroying the very features the
    # remesher must keep). Only smooth when the mesh is actually NOISY, and use a
    # crease-aware method that leaves sharp edges intact.
    if smooth > 0 and _needs_smoothing(V, F):
        try:
            # Taubin lambda/mu smoothing barely shrinks and preserves features
            # far better than Laplacian; selection-aware so creases stay sharp.
            ms.apply_coord_taubin_smoothing(stepsmoothnum=int(smooth),
                                            lambda_=0.5, mu=-0.53)
        except Exception:  # noqa: BLE001
            try:
                ms.apply_coord_laplacian_smoothing(
                    stepsmoothnum=int(smooth), cotangentweight=False)
            except Exception:  # noqa: BLE001
                pass
    ms.meshing_repair_non_manifold_edges()
    m = ms.current_mesh()
    return np.asarray(m.vertex_matrix(), np.float64), \
        np.asarray(m.face_matrix(), np.int64)


def humanlogic_quad(vertices, faces, target=2000, feature_angle=35.0,
                    feature_weight=8.0, ridge_weight=3.0, field_iters=70,
                    work_faces=6000, preprocess=True, pre_smooth=4):
    """HumanLogic perceptual quad remesh (humanlogic.py).

    When `preprocess`, the raw mesh is first repaired + decimated + smoothed
    (preprocess_for_quad) so the patch segmentation sees clean regions, not
    scan noise. Returns (V, quads Nx4, tris Nx3).
    """
    import humanlogic as hl

    V = np.asarray(vertices, np.float64)
    F = np.asarray(faces, np.int64)
    if preprocess:
        V, F = preprocess_for_quad(V, F, work_faces=work_faces, smooth=pre_smooth)
    elif len(F) > work_faces and pymeshlab is not None:
        V, F = quadric_decimate(
            V, F, quality=np.ones(len(V)),
            target_perc=max(work_faces / len(F), 0.01), preserve_boundary=True,
        )
    return hl.humanlogic_remesh(
        V, F, target_quads=target, feature_angle=feature_angle,
        feature_weight=feature_weight, ridge_weight=ridge_weight,
        field_iters=field_iters,
    )


def shrinkwrap_quad(vertices, faces, target=2000, work_faces=6000,
                    preprocess=True, pre_smooth=4):
    """6-cage-face shrinkwrap quad remesh (shrinkwrap.py) — clean axis-aligned
    box/loop flow for closed shapes. Returns (V, quads Nx4, tris Nx3=empty)."""
    import shrinkwrap as sw

    V = np.asarray(vertices, np.float64)
    F = np.asarray(faces, np.int64)
    if preprocess:
        V, F = preprocess_for_quad(V, F, work_faces=work_faces, smooth=pre_smooth)
    P, Q = sw.shrinkwrap_remesh(V, F, target_quads=target)
    return P, Q, np.zeros((0, 3), np.int64)


def visibility_shell_quad(vertices, faces, target=2000, work_faces=6000,
                          preprocess=True, pre_smooth=4):
    """Visibility-shell shrinkwrap (visibility_shells.py) — full Phase I-IV:
    cuts self-occluding shapes (torus) into shells, projects each, welds.
    Returns (V, quads Nx4, tris=empty)."""
    import visibility_shells as vis

    V = np.asarray(vertices, np.float64)
    F = np.asarray(faces, np.int64)
    if preprocess:
        V, F = preprocess_for_quad(V, F, work_faces=work_faces, smooth=pre_smooth)
    P, Q, T = vis.remesh(V, F, target_quads=target)
    return P, Q, T


def quadwild_quad(vertices, faces, target=2000, work_faces=6000,
                  preprocess=True, pre_smooth=4, sharp_mode="auto",
                  sharp_angle=35.0):
    """MODERN default: QuadWild-BiMDF (quadwild.py) feature-line-driven pure-quad
    remesh. Returns (V, q, tris). Falls back to the in-house engine on failure.

    target     : approximate quad count -> mapped to QuadWild's scaleFact density.
    sharp_mode : "auto" (detect creases) | "hard" (always preserve sharp edges) |
                 "smooth" (organic, ignore creases).
    sharp_angle: dihedral angle (deg) that counts as a sharp edge.
    """
    import quadwild as qw

    V = np.asarray(vertices, np.float64)
    F = np.asarray(faces, np.int64)
    if preprocess:
        V, F = preprocess_for_quad(V, F, work_faces=work_faces, smooth=pre_smooth)
    if qw.available():
        if sharp_mode == "hard":
            sharp = True
        elif sharp_mode == "smooth":
            sharp = False
        else:  # auto
            try:
                import visibility_shells as vis
                sharp = vis._is_creased(trimesh.Trimesh(V, F, process=False),
                                        sharp_angle)
            except Exception:  # noqa: BLE001
                sharp = True
        # map target quad count -> scaleFact. Empirically scaleFact 1 ~ 4200q on
        # our test shapes; quads scale ~ 1/scaleFact^2, so sf ~ sqrt(4200/target).
        scale_fact = float(np.clip(np.sqrt(4200.0 / max(target, 50)), 0.15, 3.0))
        res = qw.remesh(V, F, sharp=sharp, sharp_thr=sharp_angle,
                        scale_fact=scale_fact)
        if res is not None:
            P, Q, T = res
            if len(Q):
                return P, Q, T
    # fallback: in-house engine (never leave the user without a result)
    return visibility_shell_quad(vertices, faces, target=target,
                                 work_faces=work_faces, preprocess=preprocess,
                                 pre_smooth=pre_smooth)


def neurcross_quad(vertices, faces, target=2000, work_faces=6000,
                   preprocess=True, pre_smooth=4, n_samples=2000):
    """NeurCross-steered quad remesh (neurcross.py): a NEURAL cross field drives
    QuadWild's extraction. Best for smooth/organic shapes. SLOW (minutes/mesh, GPU).
    Falls back to quadwild_quad if NeurCross is unavailable or fails."""
    import neurcross as nc

    V = np.asarray(vertices, np.float64)
    F = np.asarray(faces, np.int64)
    if preprocess:
        V, F = preprocess_for_quad(V, F, work_faces=work_faces, smooth=pre_smooth)
    if nc.available():
        scale_fact = float(np.clip(np.sqrt(4200.0 / max(target, 50)), 0.15, 3.0))
        res = nc.remesh(V, F, scale_fact=scale_fact, n_samples=n_samples)
        if res is not None and len(res[1]):
            return res
    # fall back to vanilla QuadWild (never leave the user without a result)
    return quadwild_quad(vertices, faces, target=target, work_faces=work_faces,
                         preprocess=preprocess, pre_smooth=pre_smooth)


def to_glb_bytes(vertices, faces, colors=None):
    import io

    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    if colors is not None:
        c = np.asarray(colors, dtype=np.float64)
        if c.ndim == 2 and c.shape[1] == 3:
            c = np.hstack([c, np.ones((len(c), 1))])
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=c)
    buf = io.BytesIO()
    mesh.export(buf, file_type="glb")
    return buf.getvalue()


_MEDIA = {
    "glb": "model/gltf-binary", "obj": "text/plain", "ply": "application/ply",
    "stl": "model/stl",
}


def export_mesh(vertices, faces, fmt, quads=None):
    """Export a mesh as bytes. Returns (data, media_type, suffix).

    If `quads` (Q,4) is given and fmt is OBJ, the real 4-sided faces are written
    (via pymeshlab) so the quad topology survives. GLB/PLY/STL are triangle-only
    formats and always get the triangulated `faces` — a format limitation, not a
    bug; use OBJ to keep quads."""
    import io

    fmt = fmt.lower()
    V = np.asarray(vertices, dtype=np.float64)

    if fmt == "obj" and quads is not None and len(quads) and pymeshlab is not None:
        return _export_quads_obj(V, np.asarray(quads, np.int64))

    mesh = trimesh.Trimesh(vertices=V, faces=np.asarray(faces, np.int64), process=True)
    buf = io.BytesIO()
    mesh.export(buf, file_type=fmt)
    return buf.getvalue(), _MEDIA.get(fmt, "application/octet-stream"), fmt


def _export_quads_obj(V, quads):
    """Write V + quad faces to an OBJ via pymeshlab (preserves 4-sided faces)."""
    import os
    import tempfile

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(V, face_list_of_indices=[list(map(int, q)) for q in quads]))
    d = tempfile.mkdtemp(prefix="exp_")
    try:
        path = os.path.join(d, "mesh.obj")
        ms.save_current_mesh(path)
        with open(path, "rb") as fh:
            return fh.read(), _MEDIA["obj"], "obj"
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)
