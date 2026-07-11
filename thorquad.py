"""
ThorQuad — ThoRemesher's own field-guided quad engine.

The stack (ZRemesher-style, each stage swappable):
  1. ANALYZE   curvature sizing + creases + principal directions   (reused)
  2. GUIDE     smooth 4-RoSy cross field (this module) or NeurCross
  3. PARTITION separatrix tracing from singularities -> quad patches
  4. GRID      per-patch parameterization -> integer grid -> stitch
  5. RELAX     tangential smoothing + projection                    (reused)

This module implements stage 2 + the singularity analysis stage 3 builds on.

Math notes (the parts that bite):
- The cross field is stored per-face as a COMPLEX number c = e^{i4θ}, θ the
  angle of one cross direction in the face's local basis. The ×4 makes the
  4-fold rotational symmetry invisible to smoothing (all 4 arms are the same c).
- Parallel transport between adjacent faces f→g is a rotation by
  r = α_g − α_f, where α is the angle of their SHARED edge in each face's
  basis (unfolding the dihedral). In ×4 representation: multiply by e^{i4r}.
- A vertex's singularity index is  I = (angle_defect + Σ ring rotations)/(π/2),
  an integer. Σ I over all vertices = 4·χ(mesh) — the verification invariant
  (sphere: 8, torus: 0). Index +1 ≙ a valence-3 pole, −1 ≙ valence-5.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import trimesh

try:
    import igl
except Exception:  # pragma: no cover
    igl = None


# --------------------------------------------------------------------------- #
#  Face bases + parallel transport
# --------------------------------------------------------------------------- #
def face_bases(V, F):
    """Orthonormal (e1, e2, n) per face; e1 along the first edge."""
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    e1 = p1 - p0
    e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-12)
    n = np.cross(p1 - p0, p2 - p0)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    e2 = np.cross(n, e1)
    return e1, e2, n


def _angle_in_basis(vec, e1, e2):
    """Angle of 3D vectors in each face's (e1,e2) tangent basis."""
    return np.arctan2(np.einsum("ij,ij->i", vec, e2),
                      np.einsum("ij,ij->i", vec, e1))


def transport(V, F, mesh, e1, e2):
    """Per face-adjacency pair (f,g): rotation r = α_g − α_f taking angles
    measured in f's basis to g's basis across their shared edge.
    Returns (pairs (A,2), r (A,))."""
    pairs = mesh.face_adjacency                       # (A, 2) face indices
    edges = mesh.face_adjacency_edges                 # (A, 2) vertex indices
    ev = V[edges[:, 1]] - V[edges[:, 0]]              # same 3D vector for both
    ev = ev / np.maximum(np.linalg.norm(ev, axis=1, keepdims=True), 1e-12)
    a_f = _angle_in_basis(ev, e1[pairs[:, 0]], e2[pairs[:, 0]])
    a_g = _angle_in_basis(ev, e1[pairs[:, 1]], e2[pairs[:, 1]])
    return pairs, (a_g - a_f)


# --------------------------------------------------------------------------- #
#  Field initialization (principal directions + crease constraints)
# --------------------------------------------------------------------------- #
def init_field(V, F, e1, e2, mesh, crease_angle=40.0):
    """Initial per-face ×4 field + weights + locks.

    - Anisotropic regions: aligned to principal curvature directions,
      weighted by |k1|−|k2| (umbilic/flat faces contribute ~nothing).
    - Faces touching a crease edge: LOCKED to the crease direction.
    Returns (c0 complex (F,), locked bool (F,))."""
    nF = len(F)
    c0 = np.zeros(nF, np.complex128)

    if igl is not None:
        try:
            out = igl.principal_curvature(
                np.ascontiguousarray(V), np.ascontiguousarray(F))
            pd1, _pd2, k1, k2 = out[:4]   # newer igl appends a bad-vertex list
            # RELATIVE anisotropy in [0,1]: ~0 on umbilic/flat (sphere/plane,
            # where PD1 is numeric noise — must contribute NOTHING), ~1 where
            # one curvature dominates (cylinder walls, bends). Thresholded so
            # noise can't masquerade as signal after normalization.
            aniso = np.abs(np.abs(k1) - np.abs(k2)) / (np.abs(k1) + np.abs(k2) + 1e-12)
            w = np.where(aniso > 0.25, aniso, 0.0)
            for k in range(3):                        # accumulate the 3 corners
                vi = F[:, k]
                d = pd1[vi]
                # project into the face plane
                n = np.cross(e1, e2)
                d = d - np.einsum("ij,ij->i", d, n)[:, None] * n
                phi = _angle_in_basis(d, e1, e2)
                c0 += w[vi] * np.exp(4j * phi)
            c0 /= 3.0                                 # keep |c0| in ~[0,1]
        except Exception:  # noqa: BLE001
            import logging
            logging.getLogger(__name__).exception("principal-curvature init failed")
            c0[:] = 0

    # crease constraint: lock faces adjacent to a sharp edge to the edge dir
    locked = np.zeros(nF, bool)
    ang = mesh.face_adjacency_angles
    sharp = ang > np.radians(crease_angle)
    if sharp.any():
        pairs = mesh.face_adjacency[sharp]
        edges = mesh.face_adjacency_edges[sharp]
        ev = V[edges[:, 1]] - V[edges[:, 0]]
        ev /= np.maximum(np.linalg.norm(ev, axis=1, keepdims=True), 1e-12)
        for col in (0, 1):
            f = pairs[:, col]
            phi = _angle_in_basis(ev, e1[f], e2[f])
            c0[f] = np.exp(4j * phi)                  # hard overwrite
            locked[f] = True
    return c0, locked


# --------------------------------------------------------------------------- #
#  Field smoothing — Globally Optimal Direction Fields (Knöppel et al. 2013)
# --------------------------------------------------------------------------- #
def smooth_field(c0, locked, pairs, r, n_faces, fidelity=1.0, **_legacy):
    """Globally smoothest ×4 field via the connection Laplacian.

    Unconstrained: the eigenvector of L with the smallest eigenvalue IS the
    smoothest field (no iteration, no local minima — this is what kills the
    spurious ±1 singularity pairs a Jacobi relaxation leaves behind).
    With alignment (creases hard, anisotropy soft): one sparse linear solve
      (L + W) c = W c_init,   W = diag(alignment weights).
    Returns unit complex field (F,)."""
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    vals = np.concatenate([np.exp(-4j * r), np.exp(4j * r)])  # Hermitian pair
    A = sp.coo_matrix((vals, (rows, cols)), shape=(n_faces, n_faces)).tocsr()
    deg = np.asarray(np.abs(A).sum(axis=1)).ravel()
    L = sp.diags(deg).tocsr() - A                      # connection Laplacian

    w = np.abs(c0) * fidelity                          # soft alignment weight
    w[locked] = 1e4                                    # creases: effectively hard
    c_init = c0 / np.maximum(np.abs(c0), 1e-12)

    if w.max() < 1e-8:
        # nothing to align to (e.g. a sphere): globally optimal = smallest
        # eigenvector of L (shift-invert; L is PSD so shift slightly negative)
        from scipy.sparse.linalg import eigsh
        _vals, vecs = eigsh(L, k=1, sigma=-0.01, which="LM")
        c = vecs[:, 0].astype(np.complex128)
    else:
        from scipy.sparse.linalg import spsolve
        c = spsolve((L + sp.diags(w)).tocsc(), w * c_init)

    mag = np.abs(c)
    bad = mag < 1e-12
    if bad.any():
        c[bad] = 1.0
        mag = np.abs(c)
    return c / mag


# --------------------------------------------------------------------------- #
#  Singularities (libigl: integer mismatch mod 4 — immune to 2π ambiguities)
# --------------------------------------------------------------------------- #
def field_vectors(c, e1, e2):
    """Complex ×4 field -> two orthogonal per-face 3D cross directions."""
    theta = np.angle(c) / 4.0
    d1 = np.cos(theta)[:, None] * e1 + np.sin(theta)[:, None] * e2
    n = np.cross(e1, e2)
    d2 = np.cross(n, d1)
    return np.ascontiguousarray(d1), np.ascontiguousarray(d2)


def singularities(V, F, c, e1, e2):
    """Per-vertex singularity index via igl's cross-field mismatch.
    igl reports (Σ mismatch around vertex) mod 4 in {1,2,3}; 3 ≙ index −1.
    Returns (dict vertex -> index in {−1,+1,+2}, checksum = Σ index)."""
    d1, d2 = field_vectors(c, e1, e2)
    mm = igl.cross_field_mismatch(np.ascontiguousarray(V, np.float64),
                                  np.ascontiguousarray(F, np.int64),
                                  d1, d2, True)
    is_sing, s_idx = igl.find_cross_field_singularities(
        np.ascontiguousarray(V, np.float64), np.ascontiguousarray(F, np.int64), mm)
    idx = {}
    total = 0
    for v in np.nonzero(is_sing)[0]:
        I = int(s_idx[v])
        I = I - 4 if I == 3 else I     # mod-4 rep -> signed index
        idx[int(v)] = I
        total += I
    return idx, total, mm


# --------------------------------------------------------------------------- #
#  Driver
# --------------------------------------------------------------------------- #
def compute_field(V, F, crease_angle=40.0, iters=200):
    """Full stage-2 pass: smooth crease-aware 4-RoSy field + singularities.
    Returns dict with everything downstream stages need."""
    V = np.ascontiguousarray(V, np.float64)
    F = np.ascontiguousarray(F, np.int64)
    mesh = trimesh.Trimesh(V, F, process=False)
    e1, e2, n = face_bases(V, F)
    pairs, r = transport(V, F, mesh, e1, e2)
    c0, locked = init_field(V, F, e1, e2, mesh, crease_angle)
    c = smooth_field(c0, locked, pairs, r, len(F), iters=iters)
    sing, checksum, mismatch = singularities(V, F, c, e1, e2)
    return {
        "field": c, "e1": e1, "e2": e2, "normal": n,
        "pairs": pairs, "transport": r, "locked": locked,
        "singularities": sing, "index_sum": checksum, "mismatch": mismatch,
        "mesh": mesh,
    }


if __name__ == "__main__":
    # Invariant suite: Σ index must equal 4χ, and the singularity COUNT on the
    # canonical shapes must hit the known optimum (8/0/8 — cube corners etc).
    # KNOWN LIMITATION (expect_sum=None): rotationally-symmetric shapes put
    # index-±4 POLES at cap centers/apexes (full 2π circulation). The mod-4
    # mismatch detector cannot see them (4 ≡ 0 mod 4) — they read as
    # "0 singularities". Pole detection lands with the stage-3 tracer, which
    # needs ordered vertex rings anyway.
    suite = [
        # name, mesh, expected index sum, expected singularity count
        ("sphere", trimesh.creation.icosphere(subdivisions=3), 8, 8),
        ("torus", trimesh.creation.torus(2.0, 0.7, major_sections=48, minor_sections=24), 0, 0),
        ("cube", trimesh.creation.box().subdivide().subdivide().subdivide(), 8, 8),
        ("capsule", trimesh.creation.capsule(height=2, radius=0.7, count=[24, 24]), None, 0),
        ("cylinder", trimesh.creation.cylinder(radius=1, height=3, sections=48), None, 0),
    ]
    fails = 0
    for name, m, expect_sum, expect_n in suite:
        r = compute_field(np.asarray(m.vertices), np.asarray(m.faces))
        n_sing = len(r["singularities"])
        if expect_sum is None:
            status = "POLE" if n_sing == expect_n else "FAIL"  # ±4 poles: undetectable for now
        else:
            status = "OK  " if (r["index_sum"] == expect_sum and n_sing == expect_n) else "FAIL"
        fails += status == "FAIL"
        print(f"[{status}] {name:8s} index_sum={r['index_sum']:+d} singularities={n_sing}")
    raise SystemExit(1 if fails else 0)
