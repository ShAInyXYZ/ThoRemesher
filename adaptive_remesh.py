"""
Curvature-field-driven adaptive remeshing.

MeshLab's isotropic remesh takes a single target edge length, so it equalises
triangle size everywhere and erases the flat-vs-detailed density contrast.
This module implements a classic adaptive remeshing pass
(split long edges / collapse short ones / flip for valence / tangential
smoothing + reprojection) driven by a *per-vertex target edge length* so that
flat areas end up coarse and detailed areas end up dense, while still
producing clean, near-isotropic triangles.

Reference algorithm: Botsch et al., "Polygon Mesh Processing", ch. 6.
"""
from __future__ import annotations

import numpy as np
import trimesh


# --------------------------------------------------------------------------- #
#  Low level helpers
# --------------------------------------------------------------------------- #
def _edges_of_faces(F):
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    return np.sort(e, axis=1)


def _build_adjacency(F, n_verts):
    """Return (neighbours[v] -> set, edge_faces[(a,b)] -> list[face])."""
    nbr = [set() for _ in range(n_verts)]
    edge_faces = {}
    for fi, (a, b, c) in enumerate(F):
        for u, v in ((a, b), (b, c), (c, a)):
            nbr[u].add(v)
            nbr[v].add(u)
            key = (u, v) if u < v else (v, u)
            edge_faces.setdefault(key, []).append(fi)
    return nbr, edge_faces


def _triangle_normals(V, F):
    p0 = V[F[:, 0]]
    p1 = V[F[:, 1]]
    p2 = V[F[:, 2]]
    n = np.cross(p1 - p0, p2 - p0)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    return n / ln


# --------------------------------------------------------------------------- #
#  One adaptive remeshing iteration
# --------------------------------------------------------------------------- #
class AdaptiveRemesher:
    def __init__(self, V, F, target, feature_edges=None, reproject=None):
        self.V = [np.asarray(p, dtype=np.float64) for p in V]
        self.target = list(target)
        self.feat = set(feature_edges or [])  # set of (min,max) crease edges
        self.reproject = reproject  # callable: pts(N,3)->pts(N,3) on orig surface
        self.F = None

    # ---- split edges longer than 4/3 * target ------------------------------
    def _split(self):
        F = self.F
        V, tgt = self.V, self.target
        to_split = set()
        for (a, b) in _edges_of_faces(F):
            mid_t = 0.5 * (tgt[a] + tgt[b])
            if mid_t <= 0:
                continue
            if np.linalg.norm(V[a] - V[b]) > (4.0 / 3.0) * mid_t:
                to_split.add((int(min(a, b)), int(max(a, b))))
        if not to_split:
            return
        new_mid = {}

        def mid(au, bv):
            key = (au, bv) if au < bv else (bv, au)
            nid = new_mid.get(key)
            if nid is None:
                nid = len(V)
                V.append(0.5 * (V[au] + V[bv]))
                tgt.append(0.5 * (tgt[au] + tgt[bv]))
                if key in self.feat:
                    self.feat.add((key[0], nid))
                    self.feat.add((nid, key[1]))
                    self.feat.discard(key)
                new_mid[key] = nid
            return nid

        out = []
        for (a, b, c) in F:
            sab = (min(a, b), max(a, b)) in to_split
            sbc = (min(b, c), max(b, c)) in to_split
            sca = (min(c, a), max(a, c)) in to_split
            cnt = sab + sbc + sca
            if cnt == 0:
                out.append((a, b, c))
            elif cnt == 1:
                if sab:
                    m = mid(a, b); out += [(a, m, c), (m, b, c)]
                elif sbc:
                    m = mid(b, c); out += [(a, b, m), (a, m, c)]
                else:
                    m = mid(c, a); out += [(a, b, m), (m, b, c)]
            elif cnt == 2:
                if not sab:
                    m1, m2 = mid(b, c), mid(c, a)
                    out += [(a, b, m1), (a, m1, m2), (m2, m1, c)]
                elif not sbc:
                    m1, m2 = mid(a, b), mid(c, a)
                    out += [(m1, b, c), (a, m1, c), (m1, m2, c)]  # see note
                    out[-3:] = [(a, m1, m2), (m1, b, c), (m1, c, m2)]
                else:  # sab and sbc
                    m1, m2 = mid(a, b), mid(b, c)
                    out += [(a, m1, m2), (m1, b, m2), (a, m2, c)]
            else:  # 3 edges
                mab, mbc, mca = mid(a, b), mid(b, c), mid(c, a)
                out += [(a, mab, mca), (mab, b, mbc), (mca, mbc, c), (mab, mbc, mca)]
        self.F = np.asarray(out, dtype=np.int64)

    # ---- collapse edges shorter than 4/5 * target --------------------------
    def _collapse(self):
        F = self.F
        V, tgt = self.V, self.target
        n = len(V)
        Va = np.asarray(V)
        # process unique edges in order of increasing length
        edges = np.unique(_edges_of_faces(F), axis=0)
        lengths = np.linalg.norm(Va[edges[:, 0]] - Va[edges[:, 1]], axis=1)
        order = np.argsort(lengths)

        nbr, edge_faces = _build_adjacency(F, n)
        removed = np.zeros(len(F), dtype=bool)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for ei in order:
            a0, b0 = int(edges[ei, 0]), int(edges[ei, 1])
            # keep the denser (smaller-target) vertex, collapse the other into it
            if tgt[a0] <= tgt[b0]:
                k0, d0 = a0, b0
            else:
                k0, d0 = b0, a0
            k, d = find(k0), find(d0)
            if k == d:
                continue
            la = np.linalg.norm(V[k] - V[d])
            mid_t = 0.5 * (tgt[k] + tgt[d])
            if mid_t <= 0 or la >= (4.0 / 5.0) * mid_t:
                continue
            # link condition: only the shared opposite vert(s) may be common
            common = nbr[k] & nbr[d]
            common.discard(k)
            common.discard(d)
            key = (k, d) if k < d else (d, k)
            fcs = edge_faces.get(key, [])
            is_boundary = len(fcs) <= 1
            if is_boundary:
                if len(common) != 1:
                    continue
            else:
                if len(common) != 2:
                    continue
            # collapse d -> k: remove faces on edge (k,d), repoint d's neighbours to k
            for fi in fcs:
                removed[fi] = True
            for w in list(nbr[d]):
                if w == d or w == k:
                    continue
                nbr[w].discard(d)
                nbr[w].add(k)
                nbr[k].add(w)
                ek = (d, w) if d < w else (w, d)
                nk = (k, w) if k < w else (w, k)
                if ek in edge_faces:
                    lst = edge_faces.pop(ek)
                    edge_faces.setdefault(nk, []).extend(
                        ff for ff in lst if not removed[ff]
                    )
            nbr[k].discard(d)
            nbr[d] = set()
            parent[d] = k
            V[k] = 0.5 * (V[k] + V[d])
            tgt[k] = min(tgt[k], tgt[d])

        # apply remap + drop removed/degenerate/duplicate
        Fm = np.array([[find(int(x)) for x in f] for f in F], dtype=np.int64)
        mask = (
            (~removed)
            & (Fm[:, 0] != Fm[:, 1])
            & (Fm[:, 1] != Fm[:, 2])
            & (Fm[:, 2] != Fm[:, 0])
        )
        Fm = Fm[mask]
        # dedupe by sorted vertex triple
        key = np.sort(Fm, axis=1)
        _, idx = np.unique(key, axis=0, return_index=True)
        self.F = Fm[np.sort(idx)]

    # ---- flip edges to improve valence -------------------------------------
    def _flip(self):
        F = self.F
        V = self.V
        n = len(V)
        nbr, edge_faces = _build_adjacency(F, n)
        # valence target
        for (a, b), fcs in list(edge_faces.items()):
            if len(fcs) != 2:
                continue
            f1, f2 = fcs
            t1, t2 = F[f1], F[f2]
            # opposite vertices
            c = [v for v in t1 if v != a and v != b]
            d = [v for v in t2 if v != a and v != b]
            if not c or not d:
                continue
            c, d = int(c[0]), int(d[0])
            if c == d:
                continue
            # valence deviation before vs after (target 6, 4 on boundary)
            def dev(v):
                return abs(len(nbr[v]) - 6)

            before = dev(a) + dev(b) + dev(c) + dev(d)
            after = dev(c) + dev(d) + (abs(len(nbr[a]) - 1 - 6)) + (
                abs(len(nbr[b]) - 1 - 6)
            )
            if after >= before:
                continue
            # avoid creating a flipped/degenerate triangle
            na = np.cross(V[b] - V[a], V[c] - V[a])
            if np.linalg.norm(np.cross(V[b] - V[a], V[d] - V[a])) < 1e-12:
                continue
            # new faces (a,d,c) and (b,c,d); ensure non-zero area & consistent normal
            for newf in ((a, d, c), (b, c, d)):
                if len({newf[0], newf[1], newf[2]}) < 3:
                    break
            else:
                # normal orientation check vs original triangles
                o1 = np.cross(V[b] - V[a], V[c] - V[a])
                o2 = np.cross(V[b] - V[a], V[d] - V[a])
                n1 = np.cross(V[d] - V[c], V[a] - V[c])
                if (np.dot(o1, n1) <= 0) or (np.dot(o2, np.cross(V[c] - V[d], V[b] - V[d])) <= 0):
                    continue
                F[f1] = (a, d, c)
                F[f2] = (b, c, d)
                # refresh adjacency for these verts
                nbr[a].discard(b); nbr[b].discard(a)
                nbr[c].add(d); nbr[d].add(c)
        self.F = F

    # ---- tangential smoothing (feature-aware) ------------------------------
    def _smooth(self):
        F = self.F
        V = self.V
        n = len(V)
        if n == 0:
            return
        nbr, _ = _build_adjacency(F, n)
        featv = set()
        for (a, b) in self.feat:
            featv.add(a); featv.add(b)
        newV = [None] * n
        # area-weighted vertex normal
        fnorm = _triangle_normals(np.asarray(V), F)
        areas = np.linalg.norm(
            np.cross(
                np.asarray(V)[F[:, 1]] - np.asarray(V)[F[:, 0]],
                np.asarray(V)[F[:, 2]] - np.asarray(V)[F[:, 0]],
            ),
            axis=1,
        )
        vnorm = np.zeros((n, 3))
        for fi, (a, b, c) in enumerate(F):
            w = areas[fi]
            vnorm[a] += fnorm[fi] * w
            vnorm[b] += fnorm[fi] * w
            vnorm[c] += fnorm[fi] * w
        vnlen = np.linalg.norm(vnorm, axis=1, keepdims=True)
        vnlen[vnlen == 0] = 1.0
        vnorm /= vnlen
        Varr = np.asarray(V)
        for v in range(n):
            if v in featv or not nbr[v]:
                newV[v] = Varr[v]
                continue
            # area-weighted centroid of neighbours
            c = np.zeros(3)
            for w in nbr[v]:
                c += Varr[w]
            c /= len(nbr[v])
            # project displacement onto tangent plane
            disp = c - Varr[v]
            disp = disp - vnorm[v] * np.dot(disp, vnorm[v])
            newV[v] = Varr[v] + 0.5 * disp  # damped (lambda=0.5)
        Varr = np.asarray(newV)
        if self.reproject is not None:
            mask = np.array([v not in featv for v in range(n)])
            if mask.any():
                Varr[mask] = self.reproject(Varr[mask])
        self.V = [np.asarray(p) for p in Varr]

    # ---- driver ------------------------------------------------------------
    def run(self, iters=6):
        for _ in range(int(iters)):
            self._split()
            self._collapse()
            self._flip()
            self._smooth()
        V = np.asarray(self.V)
        F = np.asarray(self.F, dtype=np.int64)
        return _compact(V, F)


def _compact(V, F):
    """Drop vertices that no face references and remap to a dense index range."""
    if len(F) == 0:
        return V[:0], F
    used = np.unique(F)
    remap = -np.ones(len(V), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return V[used], remap[F]


def adaptive_remesh(
    vertices, faces, target, feature_edges=None, iters=6, reproject_mesh=None
):
    """Curvature-field-driven adaptive remesh.

    target       : (V,) per-vertex target edge length.
    feature_edges: iterable of (a,b) crease edges to preserve.
    reproject_mesh: (V,F) original surface to reproject onto (shape fidelity).
    """
    reproject = None
    if reproject_mesh is not None:
        ro, fo = reproject_mesh
        try:
            base = trimesh.Trimesh(ro, fo, process=False)

            def reproject(pts):
                return trimesh.proximity.closest_point(base, np.asarray(pts))[0]
        except Exception:  # noqa: BLE001
            reproject = None
    rm = AdaptiveRemesher(
        np.asarray(vertices), np.asarray(faces), target, feature_edges, reproject
    )
    rm.F = np.asarray(faces, dtype=np.int64)
    return rm.run(iters=iters)
