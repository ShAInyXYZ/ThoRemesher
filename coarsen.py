"""
Curvature-adaptive quad coarsening — a POST-STITCH simplify pass.

Runs on the finished watertight quad mesh from seam_stitch.stitch. On flat,
grid-structured regions (cube faces, flat caps) it merges aligned BxB blocks of
quads (B = 2^level) into a few large quads, where the level is driven by a
Dunyach (2013) curvature sizing field sampled from the ORIGINAL surface. Curved
regions (sphere, torus barrel) get level 0 -> stay dense. The pass is
self-validating: if it cannot keep the mesh watertight + all-quad it returns the
input unchanged, so the worst case is "no adaptivity", never "broken".

Watertight mechanism: every emitted face references vertices that ALREADY exist
on the fine mesh (region grid nodes). A coarse block is tiled by a fan-to-centre
template whose perimeter follows the block's outer nodes; any side of the block
that abuts FINE cells (the frontier) is subdivided into B sub-edges so the fan
matches the fine mesh node-for-node. No new vertices, no T-junctions.
"""
from __future__ import annotations

import collections

import numpy as np
import networkx as nx
from scipy.spatial import cKDTree

try:
    import igl
except Exception:  # pragma: no cover
    igl = None


# --------------------------------------------------------------------------- #
#  1. Dunyach (2013) curvature sizing field on the INPUT mesh
# --------------------------------------------------------------------------- #
def sizing_field(surf_mesh, eps_frac=0.0008, Lmin_frac=0.01, Lmax_frac=0.5):
    """Per-INPUT-vertex target edge length L(v)=sqrt(8*eps/k) (Dunyach 2013).
    k=max(|k1|,|k2|); flat -> k~0 -> L huge -> clamped to Lmax. Scale-free
    (fractions of bbox diag). Returns (V, L)."""
    V = np.asarray(surf_mesh.vertices, np.float64)
    F = np.asarray(surf_mesh.faces, np.int64)
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    if diag < 1e-12 or igl is None:
        return V, np.full(len(V), Lmax_frac * max(diag, 1.0))
    eps = eps_frac * diag
    Lmin = Lmin_frac * diag
    Lmax = Lmax_frac * diag
    r = igl.principal_curvature(V, F)
    k1, k2 = np.asarray(r[2]), np.asarray(r[3])
    k = np.maximum(np.abs(k1), np.abs(k2))
    with np.errstate(divide="ignore", invalid="ignore"):
        L = np.sqrt(8.0 * eps / np.maximum(k, 1e-12))
    L = np.clip(np.nan_to_num(L, nan=Lmax, posinf=Lmax), Lmin, Lmax)
    return V, L


# --------------------------------------------------------------------------- #
#  2. Coplanar grid regions
# --------------------------------------------------------------------------- #
def _quad_normals(P, Q):
    n = np.cross(P[Q[:, 1]] - P[Q[:, 0]], P[Q[:, 3]] - P[Q[:, 0]])
    nn = np.linalg.norm(n, axis=1, keepdims=True)
    nn[nn < 1e-12] = 1.0
    return n / nn


def coplanar_grid_regions(P, Q, ang_tol_deg=6.0):
    """Group quads into regions coplanar within ang_tol AND connected by shared
    full edges. Cube faces / flat caps -> one big region each; curved surfaces ->
    tiny regions. Returns region[qi]."""
    qn = _quad_normals(P, Q)
    e2q = collections.defaultdict(list)
    for qi, q in enumerate(Q):
        for a, b in ((q[0], q[1]), (q[1], q[2]), (q[2], q[3]), (q[3], q[0])):
            e2q[tuple(sorted((int(a), int(b))))].append(qi)
    g = nx.Graph()
    g.add_nodes_from(range(len(Q)))
    cos_tol = np.cos(np.radians(ang_tol_deg))
    for qs in e2q.values():
        if len(qs) == 2 and qn[qs[0]] @ qn[qs[1]] >= cos_tol:
            g.add_edge(qs[0], qs[1])
    region = np.full(len(Q), -1, np.int64)
    for r, comp in enumerate(nx.connected_components(g)):
        for q in comp:
            region[q] = r
    return region


# --------------------------------------------------------------------------- #
#  3. Recover integer (i,j) grid coordinates for a coplanar region
# --------------------------------------------------------------------------- #
def grid_coords(P, Q, region_quads):
    """Recover integer (i,j) grid node coords for every corner vertex of a
    coplanar region TOPOLOGICALLY (robust to smoothing that makes the lattice
    geometrically irregular). Each quad has 4 corners; we propagate a consistent
    (i,j) labelling across shared edges by BFS, using a per-region in-plane basis
    only to fix the two axis labels. Returns (node_ij, ij_to_vid, ni, nj) or None
    if the region is not a clean quad grid."""
    verts = np.unique(Q[region_quads].ravel())
    if len(verts) < 4:
        return None
    pts = P[verts]
    c = pts.mean(0)
    u, s, vt = np.linalg.svd(pts - c, full_matrices=False)
    if len(s) < 2 or s[1] < 1e-9:
        return None
    e1, e2 = vt[0], vt[1]
    # dominant in-plane edge direction -> grid axis (period pi/2)
    edirs = []
    for qi in region_quads:
        q = Q[qi]
        for x, y in ((q[0], q[1]), (q[1], q[2]), (q[2], q[3]), (q[3], q[0])):
            d = P[y] - P[x]
            edirs.append([d @ e1, d @ e2])
    edirs = np.asarray(edirs)
    nrm = np.linalg.norm(edirs, axis=1)
    edirs = edirs[nrm > 1e-9]
    if not len(edirs):
        return None
    ang = np.arctan2(edirs[:, 1], edirs[:, 0]) % (np.pi / 2)
    th = np.angle(np.mean(np.exp(1j * 4 * ang))) / 4.0
    ca, sa = np.cos(th), np.sin(th)
    g1 = ca * e1 + sa * e2          # grid axis u
    g2 = -sa * e1 + ca * e2         # grid axis v

    # ---- topological (i,j) propagation by BFS over quads ----
    # For each quad, order its 4 corners CCW and classify each edge as +u/-u/+v/-v
    # by its projection onto (g1,g2). Assign the first quad (0,0)/(1,0)/(1,1)/(0,1)
    # then flood-fill neighbours so shared corners agree.
    rqset = list(region_quads)
    # edge -> quads (within region)
    e2q = collections.defaultdict(list)
    for qi in rqset:
        q = Q[qi]
        for a, b in ((q[0], q[1]), (q[1], q[2]), (q[2], q[3]), (q[3], q[0])):
            e2q[tuple(sorted((int(a), int(b))))].append(qi)

    def quad_corner_dirs(qi):
        """Return list of (vid, (i_off, j_off)) for the quad's 4 corners on a
        local unit cell, oriented so edge directions match (g1, g2). None if the
        quad is not a clean axis cell."""
        q = [int(x) for x in Q[qi]]
        # local 2D coords
        loc = np.array([[ (P[v]-c)@g1, (P[v]-c)@g2 ] for v in q])
        # which corner is min-u,min-v etc. — find ordering by angle around centroid
        cc = loc.mean(0)
        rel = loc - cc
        # the 4 corners should be in 4 quadrants; map to offsets
        offs = []
        for r in rel:
            iu = 1 if r[0] > 0 else 0
            iv = 1 if r[1] > 0 else 0
            offs.append((iu, iv))
        if len(set(offs)) != 4:
            return None
        return list(zip(q, offs))

    node_ij = {}
    # seed
    seed = rqset[0]
    cd = quad_corner_dirs(seed)
    if cd is None:
        return None
    for v, off in cd:
        node_ij[v] = off
    visited = {seed}
    stack = [seed]
    while stack:
        qi = stack.pop()
        cd = quad_corner_dirs(qi)
        if cd is None:
            return None
        # this quad already has at least one labelled vertex; compute the
        # translation between its local offsets and the global labels
        local = dict(cd)
        known = [(v, node_ij[v]) for v, _ in cd if v in node_ij]
        if not known:
            return None
        # offset = global - local must be consistent across known verts
        dv = None
        for v, gij in known:
            lij = local[v]
            d = (gij[0] - lij[0], gij[1] - lij[1])
            if dv is None:
                dv = d
            elif d != dv:
                return None
        for v, lij in cd:
            gij = (lij[0] + dv[0], lij[1] + dv[1])
            if v in node_ij and node_ij[v] != gij:
                return None
            node_ij[v] = gij
        # walk to neighbours
        q = Q[qi]
        for a, b in ((q[0], q[1]), (q[1], q[2]), (q[2], q[3]), (q[3], q[0])):
            for nb in e2q[tuple(sorted((int(a), int(b))))]:
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)

    if len(visited) != len(rqset):
        return None  # region not connected as one grid
    # normalise to non-negative
    imin = min(v[0] for v in node_ij.values())
    jmin = min(v[1] for v in node_ij.values())
    node_ij = {v: (ij[0] - imin, ij[1] - jmin) for v, ij in node_ij.items()}
    ij_to_vid = {}
    for v, ij in node_ij.items():
        if ij in ij_to_vid:
            return None
        ij_to_vid[ij] = v
    # validate every quad is a unit axis cell
    for qi in rqset:
        coords = [node_ij[int(v)] for v in Q[qi]]
        iis = sorted(set(cc[0] for cc in coords))
        jjs = sorted(set(cc[1] for cc in coords))
        if len(iis) != 2 or len(jjs) != 2:
            return None
        if iis[1] - iis[0] != 1 or jjs[1] - jjs[0] != 1:
            return None
    ni = max(v[0] for v in node_ij.values()) + 1
    nj = max(v[1] for v in node_ij.values()) + 1
    return node_ij, ij_to_vid, ni, nj


# --------------------------------------------------------------------------- #
#  4. Per-cell desired coarsening level from the sizing field
# --------------------------------------------------------------------------- #
def _cell_levels(P, Q, region_quads, node_ij, surf_tree, surf_L,
                 base_spacing, max_level):
    """Desired level per fine cell (keyed by lower-left node (i,j))."""
    lvl = {}
    cens = P[Q[region_quads]].mean(axis=1)
    Lq = surf_L[surf_tree.query(cens)[1]]
    for n, qi in enumerate(region_quads):
        coords = [node_ij[int(v)] for v in Q[qi]]
        i0 = min(cc[0] for cc in coords)
        j0 = min(cc[1] for cc in coords)
        lv = int(np.floor(np.log2(max(Lq[n] / base_spacing, 1.0))))
        lvl[(i0, j0)] = max(0, min(max_level, lv))
    return lvl


# --------------------------------------------------------------------------- #
#  5. Tile one region into coarse blocks + transition fans (all existing vids)
# --------------------------------------------------------------------------- #
def _tile_region(node_ij, ij_to_vid, ni, nj, cell_lvl, max_level):
    """Re-tile a coplanar grid region as a balanced quadtree of square leaves,
    all-quad and non-degenerate, reusing only existing grid nodes. Returns a quad
    list or None to leave the region fine.

    Each leaf of level L (size B=2^L) is emitted with a per-side resolution that
    matches its neighbour on that side: a side is split into 2^(L - nL) equal
    sub-edges where nL is the (<=L, by 2:1 balance) level of the neighbour across
    it. The leaf is then tiled by a fan to its CENTRE node, where each fan quad
    spans ONE sub-edge plus the centre and the adjacent perimeter node, arranged
    so no quad has three collinear corners (a sub-edge endpoint, the next
    perimeter node and the centre are not collinear). 2:1 balance guarantees a
    neighbour's nodes land exactly on this leaf's sub-edge endpoints, so the
    frontier matches node-for-node -> watertight."""
    cells = set()
    for key in ij_to_vid:
        i, j = key
        if all((i + di, j + dj) in ij_to_vid for di in (0, 1) for dj in (0, 1)):
            cells.add((i, j))
    if not cells:
        return None

    interior = set()
    for (i, j) in cells:
        if all((i + di, j + dj) in cells
               for di in (-1, 0, 1) for dj in (-1, 0, 1)):
            interior.add((i, j))

    i0 = min(c[0] for c in cells)
    j0 = min(c[1] for c in cells)
    imax = max(c[0] for c in cells)
    jmax = max(c[1] for c in cells)

    # ---- greedy leaf cover (largest blocks first), curvature-gated ----
    covered = set()
    leaves = []                  # (oi, oj, L)
    for L in range(max_level, 0, -1):
        B = 1 << L
        oi = i0
        while oi + B <= imax + 1:
            oj = j0
            while oj + B <= jmax + 1:
                cc = [(oi + a, oj + b) for a in range(B) for b in range(B)]
                ok = all(fc in interior and fc not in covered and
                         cell_lvl.get(fc, 0) >= L for fc in cc)
                if ok and all((oi + di, oj + dj) in ij_to_vid
                              for di in range(B + 1) for dj in range(B + 1)):
                    covered.update(cc)
                    leaves.append((oi, oj, L))
                oj += B
            oi += B
    for c in cells:
        if c not in covered:
            leaves.append((c[0], c[1], 0))
    if not any(L > 0 for _, _, L in leaves):
        return None

    def build_cell_leaf(leaf_list):
        cl = {}
        for k, (oi, oj, L) in enumerate(leaf_list):
            B = 1 << L
            for a in range(B):
                for b in range(B):
                    cl[(oi + a, oj + b)] = k
        return cl

    # ---- 2:1 balance: split leaves until edge-adjacent levels differ by <=1 ----
    changed, guard = True, 0
    while changed and guard < 40:
        changed, guard = False, guard + 1
        cell_leaf = build_cell_leaf(leaves)
        new_leaves = []
        for (oi, oj, L) in leaves:
            if L == 0:
                new_leaves.append((oi, oj, L))
                continue
            B = 1 << L
            border = ([(oi + t, oj - 1) for t in range(B)] +
                      [(oi + t, oj + B) for t in range(B)] +
                      [(oi - 1, oj + t) for t in range(B)] +
                      [(oi + B, oj + t) for t in range(B)])
            too_fine = False
            for fc in border:
                nl = leaves[cell_leaf[fc]][2] if fc in cell_leaf else (
                    0 if fc in cells else L)
                if L - nl >= 2:
                    too_fine = True
                    break
            if too_fine:
                h = B // 2
                for da in (0, h):
                    for db in (0, h):
                        new_leaves.append((oi + da, oj + db, L - 1))
                changed = True
            else:
                new_leaves.append((oi, oj, L))
        leaves = new_leaves

    # ---- emit each leaf as 4 non-degenerate QUADRANT quads ----
    # Under 2:1 balance the 4-quadrant template (corners + 4 side-midpoints +
    # centre) matches every neighbour node-for-node: a same-level neighbour
    # shares the side-midpoint; a one-level-finer neighbour presents two leaves
    # whose shared corner coincides with this leaf's side-midpoint; a coarser
    # neighbour presents a single edge that equals this leaf's full side (this
    # leaf is the finer one and its side-midpoint then sits on the coarser
    # neighbour's sub-edge, which the coarser leaf's OWN midpoint already creates).
    new_q = []
    for (oi, oj, L) in leaves:
        if L == 0:
            new_q.append([ij_to_vid[(oi, oj)], ij_to_vid[(oi + 1, oj)],
                          ij_to_vid[(oi + 1, oj + 1)], ij_to_vid[(oi, oj + 1)]])
            continue
        B = 1 << L
        h = B // 2
        try:
            c00 = ij_to_vid[(oi, oj)]
            c10 = ij_to_vid[(oi + B, oj)]
            c11 = ij_to_vid[(oi + B, oj + B)]
            c01 = ij_to_vid[(oi, oj + B)]
            mb = ij_to_vid[(oi + h, oj)]
            mr = ij_to_vid[(oi + B, oj + h)]
            mt = ij_to_vid[(oi + h, oj + B)]
            ml = ij_to_vid[(oi, oj + h)]
            ce = ij_to_vid[(oi + h, oj + h)]
        except KeyError:
            return None
        new_q.append([c00, mb, ce, ml])
        new_q.append([mb, c10, mr, ce])
        new_q.append([ce, mr, c11, mt])
        new_q.append([ml, ce, mt, c01])
    return new_q


# --------------------------------------------------------------------------- #
#  6. Watertight / all-quad validation
# --------------------------------------------------------------------------- #
def _boundary_set(quads):
    ec = collections.Counter()
    for q in quads:
        q = [int(x) for x in q]
        for a, b in ((q[0], q[1]), (q[1], q[2]), (q[2], q[3]), (q[3], q[0])):
            ec[tuple(sorted((a, b)))] += 1
    return {e for e, c in ec.items() if c == 1}


def _same_boundary(orig_quads, new_quads):
    """True iff the re-tiled region exposes exactly the same open-boundary edges
    as the original region. Guarantees the rest of the mesh (welded to that
    boundary) stays watertight."""
    return _boundary_set(orig_quads) == _boundary_set(new_quads)


def is_watertight_allquad(P, Q):
    if Q is None or not len(Q):
        return False
    Q = np.asarray(Q, np.int64)
    if Q.shape[1] != 4:
        return False
    if any(len(set(map(int, q))) != 4 for q in Q):
        return False
    e = np.vstack([Q[:, [0, 1]], Q[:, [1, 2]], Q[:, [2, 3]], Q[:, [3, 0]]])
    cnt = collections.Counter(tuple(sorted(map(int, x))) for x in e)
    return all(c == 2 for c in cnt.values())


# --------------------------------------------------------------------------- #
#  7. Driver
# --------------------------------------------------------------------------- #
def coarsen(P, Q, surf_mesh, base_spacing, max_level=4):
    """Curvature-adaptive coarsening of flat grid regions. Self-validating:
    reverts to (P, Q) on any failure. Returns (P, Q)."""
    P0, Q0 = np.asarray(P, float), np.asarray(Q, np.int64)
    try:
        if surf_mesh is None or not len(Q0):
            return P0, Q0
        surf_V, surf_L = sizing_field(surf_mesh)
        surf_tree = cKDTree(surf_V)
        region = coplanar_grid_regions(P0, Q0)
        new_Q = []
        for r in np.unique(region):
            rq = np.where(region == r)[0]
            gc = grid_coords(P0, Q0, rq) if len(rq) >= 4 else None
            if gc is None:
                new_Q.extend(Q0[rq].tolist())
                continue
            node_ij, ij_to_vid, ni, nj = gc
            cell_lvl = _cell_levels(P0, Q0, rq, node_ij, surf_tree, surf_L,
                                    base_spacing, max_level)
            tiled = _tile_region(node_ij, ij_to_vid, ni, nj, cell_lvl,
                                 max_level)
            # accept the re-tiling ONLY if it presents the SAME boundary edges as
            # the original region (so the rest of the untouched mesh, which welds
            # to that boundary, stays watertight). Otherwise leave the region
            # fine. This makes every region independently safe before assembly.
            if tiled is not None and _same_boundary(Q0[rq], tiled):
                new_Q.extend(tiled)
            else:
                new_Q.extend(Q0[rq].tolist())
        Qn = np.asarray(new_Q, np.int64).reshape(-1, 4)
        # Keep ALL vertices (compact() later drops the orphans + tri-only verts
        # safely). Q on its own is NOT closed — a few junction holes are filled
        # by the stitch's tris. The correct invariant: the re-tiled quad mesh
        # must expose EXACTLY the same open-boundary edges as the input quad mesh
        # so the unchanged tris still close them. (Per-region _same_boundary
        # already enforces this locally; re-check globally as a safety net.)
        if any(len(set(map(int, q))) != 4 for q in Qn):
            return P0, Q0
        if _boundary_set(Qn) != _boundary_set(Q0):
            return P0, Q0
        return P0, Qn
    except Exception:
        return P0, Q0
