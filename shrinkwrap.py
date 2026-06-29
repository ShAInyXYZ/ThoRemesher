"""
Volumetric shrinkwrap quad remeshing (6-cage-face projection).

The flow problem: parameterizing a closed/rounded shape onto a disk (harmonic /
LSCM) makes the grid spiral around the seam — a cube's obvious box flow comes out
swirled. Shrinkwrap fixes this by casting ray grids inward from the 6 bounding-box
faces: each cage face sees "its" side of the shape head-on, so the projected grid
is clean and axis-aligned by construction. Each surface point is owned by the one
cage direction that sees it best (normal most opposed to the ray), so the 6 grids
tile without overlap, then weld at their silhouette seams.

This is the core of VolumetricSpectralShrinkwrap_Remesh.md (Phases II-IV). The
Phase-I visibility-shell decomposition (visibility graph, oriented-cycle cuts) is
skipped: it only matters for self-occluding shapes (torus knots, deep tunnels),
and genus-0 props/bodies don't self-occlude along all 6 axes.
# ponytail: no visibility-graph decomposition until a shape actually self-occludes.
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree


_DIRS = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0],
                  [0, -1, 0], [0, 0, 1], [0, 0, -1.0]])


def _face_grid(mesh, d, ctr, L, margin, N):
    """Cast an N×N ray grid from the cage face on the -d side, along +d. Returns
    (hit_points, ray_index, hit_tri) for first hits."""
    up = np.array([0, 0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0, 0])
    a1 = np.cross(d, up); a1 /= np.linalg.norm(a1)
    a2 = np.cross(d, a1)
    org = ctr - d * (L / 2 + margin)
    g = np.linspace(-L / 2, L / 2, N)
    UU, VV = np.meshgrid(g, g, indexing="ij")
    pts = org + UU.ravel()[:, None] * a1 + VV.ravel()[:, None] * a2
    rd = np.tile(d, (len(pts), 1))
    loc, iray, itri = mesh.ray.intersects_location(pts, rd, multiple_hits=False)
    return loc, iray, itri


def shrinkwrap_remesh(vertices, faces, target_quads=2000, weld_factor=0.5):
    """Quad-remesh a closed mesh by 6-cage-face shrinkwrap projection.

    Returns (V, quads Nx4). Quads only — the projected lattice is pure quad by
    construction. Flow follows the projection axes (clean box/loop flow).
    """
    mesh = trimesh.Trimesh(np.asarray(vertices, np.float64),
                           np.asarray(faces, np.int64), process=False)
    V = mesh.vertices
    bb_min, bb_max = V.min(0), V.max(0)
    ext = bb_max - bb_min
    ctr = (bb_min + bb_max) / 2
    L = float(ext.max())
    if L < 1e-9:
        return V, np.zeros((0, 4), np.int64)
    margin = 0.1 * L
    # grid resolution: total quads ~ 6 faces * N^2 visible fraction (~0.5) -> N
    N = max(6, int(round(np.sqrt(target_quads / 3.0))))

    all_pos, all_quads, voff = [], [], 0
    for d in _DIRS:
        loc, iray, itri = _face_grid(mesh, d, ctr, L, margin, N)
        if not len(loc):
            continue
        # ownership: this point belongs to the cage face whose direction the
        # surface normal most opposes (sees most head-on). Ties -> first wins,
        # so each silhouette point lands on exactly one face.
        fn = mesh.face_normals[itri]
        align = -(fn @ d)
        best = np.max(-(fn @ _DIRS.T), axis=1)
        owned = align >= best - 1e-6
        keep = np.where(owned)[0]
        if len(keep) < 4:
            continue
        gid = -np.ones(N * N, np.int64)
        gid[iray[keep]] = np.arange(len(keep))
        pos = loc[keep]

        def idx(i, j, _gid=gid):
            return _gid[i * N + j]

        for i in range(N - 1):
            for j in range(N - 1):
                a, b, c, e = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
                if min(a, b, c, e) >= 0:
                    all_quads.append([a + voff, b + voff, c + voff, e + voff])
        all_pos.append(pos)
        voff += len(pos)

    if not all_pos:
        return V, np.zeros((0, 4), np.int64)
    P = np.vstack(all_pos)
    Q = np.asarray(all_quads, np.int64).reshape(-1, 4)
    P, Q = _weld(P, Q, weld_factor * (L / N))
    return P, Q


def _weld(P, Q, radius):
    """Union-find weld of vertices within `radius` (welds the silhouette seams
    where adjacent face-grids meet). Drops degenerate quads."""
    if radius <= 0 or not len(P):
        return P, Q
    tree = cKDTree(P)
    parent = np.arange(len(P))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in tree.query_pairs(radius):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    roots = np.array([find(i) for i in range(len(P))])
    uniq, inv = np.unique(roots, return_inverse=True)
    # average welded positions
    newP = np.zeros((len(uniq), 3))
    cnt = np.zeros(len(uniq))
    np.add.at(newP, inv, P)
    np.add.at(cnt, inv, 1)
    newP /= cnt[:, None]
    Q2 = inv[Q]
    # drop quads that collapsed (duplicate corner)
    good = np.array([len(set(q)) == 4 for q in Q2], bool)
    return newP, Q2[good]
