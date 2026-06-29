"""
Visibility-Shell Shrinkwrap remeshing — full Phase I-IV.

Faithful implementation of VolumetricSpectralShrinkwrap_Remesh.md. The naive
6-face shrinkwrap (shrinkwrap.py) shadow-collapses self-occluding shapes (the
torus inner ring is invisible to every axis from outside). This module resolves
that with the paper's visibility-shell decomposition:

  Phase I  — decompose M into visibility shells (regions each fully visible from
             one cage direction) by cutting along geodesic-curvature ridges where
             the visibility signature conflicts (§3).
  Phase II — shrinkwrap-project each shell from its own direction (§4).
  Phase III— extract the quad lattice per shell (§5).
  Phase IV — weld shells along cut boundaries by arc-length correspondence and
             Laplacian-smooth the seams (§6).

Geometry backends: igl.exact_geodesic (MMP exact polyhedral geodesics),
igl.principal_curvature (mean curvature H), trimesh.ray (visibility tests).
"""
from __future__ import annotations

import collections

import numpy as np
import trimesh
import networkx as nx
from scipy.spatial import cKDTree

try:
    import igl
except Exception:  # pragma: no cover
    igl = None

DIRS = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0],
                 [0, -1, 0], [0, 0, 1], [0, 0, -1.0]])

# how far below the best-seeing axis a direction may still claim a hit (band
# overlap, so adjacent ownership grids share weldable border samples)
_OVERLAP_TOL = 0.0   # STRICT ownership: grids PARTITION the surface (seam is shared)


# --------------------------------------------------------------------------- #
#  Phase I.1 — visibility predicate V_d(p)  (§2.2)
# --------------------------------------------------------------------------- #
def visibility_signatures(mesh, grid_n=96):
    """Per-face 6-bit visibility signature, defined by the ACTUAL projection
    (§2.2, §4.1). Bit k is set iff the face is the FIRST hit of some inward ray
    from cage face k — i.e. it is directly reachable by shrinkwrap from k, not
    hidden behind nearer geometry. A face with an all-zero signature is shadowed
    from every cage (a torus inner-ring face): that is the true self-occlusion
    that forces a shell cut.
    """
    V = mesh.vertices
    bb_min, bb_max = V.min(0), V.max(0)
    ctr = (bb_min + bb_max) / 2
    L = float((bb_max - bb_min).max())
    nf = len(mesh.faces)
    sig = np.zeros((nf, 6), bool)
    for k, d in enumerate(DIRS):
        up = np.array([0, 0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0, 0])
        a1 = np.cross(d, up); a1 /= np.linalg.norm(a1)
        a2 = np.cross(d, a1)
        g = np.linspace(-L / 2, L / 2, grid_n)
        UU, VV = np.meshgrid(g, g, indexing="ij")
        org = (ctr - d * (0.6 * L) +
               UU.ravel()[:, None] * a1 + VV.ravel()[:, None] * a2)
        _, _, itri = mesh.ray.intersects_location(
            org, np.tile(d, (len(org), 1)), multiple_hits=False)
        sig[np.unique(itri), k] = True
    return sig


def shadowed_faces(sig):
    """Faces with an all-zero visibility signature — never the first hit from any
    cage face, i.e. self-occluded (§3.1). These force a visibility-shell cut."""
    return np.where(~sig.any(axis=1))[0]


# --------------------------------------------------------------------------- #
#  Phase I.4 — geodesic-curvature seam cut  (§3.2)
# --------------------------------------------------------------------------- #
def mean_curvature(V, F):
    """Per-vertex mean curvature H = (k1+k2)/2 via libigl quadric fit."""
    r = igl.principal_curvature(V.astype(np.float64), F.astype(np.int64))
    return 0.5 * (np.asarray(r[2]) + np.asarray(r[3]))


def phi_field(V, F, shadow_faces, lam):
    """Φ(p) = d_geodesic(p, shadowed-set) + λ·κ_H(p)  (§3.2).

    Exact geodesic distance from the shadowed (self-occluded) faces, plus a
    curvature term so the seam ridge prefers low-curvature, unobtrusive lines.
    Returns per-vertex Φ.
    """
    Fd = np.ascontiguousarray(F, np.int64)
    src = np.unique(Fd[shadow_faces]).astype(np.int64)
    if not len(src):
        return None
    Vd = np.ascontiguousarray(V, np.float64)
    tgt = np.arange(len(V), dtype=np.int64)
    empty = np.array([], np.int64)
    gd = np.asarray(igl.exact_geodesic(Vd, Fd, src, empty, tgt, empty))
    H = np.abs(mean_curvature(V, Fd))
    Hn = (H - H.min()) / (np.ptp(H) + 1e-9)
    return gd + lam * Hn * (np.ptp(gd) + 1e-9)


def cut_at_phi_ridge(mesh, shadow_faces, lam=0.5):
    """Split M into two sides along the ridge of Φ (§3.2). The shadowed region
    and its visible complement end up on opposite sides; the seam falls on the
    geodesic-equidistant, low-curvature ridge. Returns a per-face boolean side.
    """
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    phi = phi_field(V, F, shadow_faces, lam)
    if phi is None:
        return None
    fphi = phi[F].mean(axis=1)
    # the shadowed faces sit at Φ≈0; the ridge is the watershed between them and
    # the far side. Cut at the Φ value that best separates shadowed from visible:
    # use the midpoint between the shadowed faces' max Φ and the global median.
    shadow_phi_max = fphi[shadow_faces].max() if len(shadow_faces) else 0.0
    thr = 0.5 * (shadow_phi_max + np.median(fphi))
    return fphi >= thr


# --------------------------------------------------------------------------- #
#  Phase I (driver) — decompose into visibility shells
# --------------------------------------------------------------------------- #
def decompose_shells(mesh, lam=0.5, max_cuts=4):
    """Decompose M into visibility shells. Returns a per-face shell id.

    Iteratively: build visibility regions + occlusion graph, find oriented-cycle
    conflicts, cut along the Φ-ridge to break them. Repeat until the graph is
    acyclic (no shell self-occludes) or max_cuts reached.
    """
    nf = len(mesh.faces)
    shell = np.zeros(nf, np.int64)          # start: everything in shell 0
    fa = mesh.face_adjacency
    for _ in range(max_cuts):
        sig = visibility_signatures(mesh)
        shadow = shadowed_faces(sig)
        # Drop isolated shadow noise: convex shapes (capsule, sphere) produce a
        # handful of scattered grazing-angle ray MISSES that look "shadowed" but
        # are not real self-occlusion. Only a substantial connected shadow region
        # (a torus inner ring) is a genuine occlusion that forces a shell cut.
        if len(shadow):
            sset = set(int(x) for x in shadow)
            g = nx.Graph(); g.add_nodes_from(sset)
            g.add_edges_from([(int(a), int(b)) for a, b in fa
                              if int(a) in sset and int(b) in sset])
            big = set()
            for comp in nx.connected_components(g):
                if len(comp) >= max(4, int(0.005 * nf)):
                    big |= comp
            shadow = np.array(sorted(big), np.int64)
        if not len(shadow):
            break                            # nothing self-occluded -> done
        side = cut_at_phi_ridge(mesh, shadow, lam)
        if side is None:
            break
        # split every existing shell by the ridge side -> new shell ids
        shell = shell * 2 + side.astype(np.int64)
        _, shell = np.unique(shell, return_inverse=True)
    return shell


# --------------------------------------------------------------------------- #
#  Phase II-III — per-shell shrinkwrap projection + lattice  (§4-5)
# --------------------------------------------------------------------------- #
def _project_one_dir(shell_mesh, d, N, ctr, L, margin):
    """Cast an N×N ray grid from cage face -d along +d onto the shell; return a
    grid-shaped (N*N) vertex map and the hit positions, plus the quads formed by
    fully-hit 2×2 cells. One clean axis-aligned lattice."""
    up = np.array([0, 0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0, 0])
    a1 = np.cross(d, up); a1 /= np.linalg.norm(a1)
    a2 = np.cross(d, a1)
    org = ctr - d * (L / 2 + margin)
    g = np.linspace(-L / 2, L / 2, N)
    UU, VV = np.meshgrid(g, g, indexing="ij")
    pts = org + UU.ravel()[:, None] * a1 + VV.ravel()[:, None] * a2
    loc, iray, itri = shell_mesh.ray.intersects_location(
        pts, np.tile(d, (len(pts), 1)), multiple_hits=False)
    if not len(loc):
        return None, None
    # OWNERSHIP: keep a hit only if THIS direction sees its face most head-on
    # (|n·d| maximal over the 6 cage axes). Without this, every direction also
    # grazes faces owned by other axes -> massive double-coverage and a vertex
    # pile-up the weld can't reconcile (sphere euler blows up). With it, the 6
    # lattices tile the surface, each owning the band it views most squarely.
    fn = shell_mesh.face_normals[itri]
    own_score = np.abs(fn @ d)
    best = np.max(np.abs(fn @ DIRS.T), axis=1)
    # Overlap bands by `overlap_tol`: a hit is kept if this direction is within
    # tol of the best-seeing axis, not strictly the single best. Adjacent grids
    # then share a row of border samples that weld together, closing the
    # silhouette seam gaps (strict ownership leaves a 1-cell tear at each band
    # border). Larger tol = more overlap = fewer gaps but more weld merging.
    owned = own_score >= best - _OVERLAP_TOL
    gid = -np.ones(N * N, np.int64)
    pos = []
    for ridx, p, keep in zip(iray, loc, owned):
        if keep and gid[ridx] < 0:
            gid[ridx] = len(pos); pos.append(p)
    if not pos:
        return None, None
    pos = np.asarray(pos)
    quads = []
    for i in range(N - 1):
        for j in range(N - 1):
            a, b, c, e = (gid[i * N + j], gid[(i + 1) * N + j],
                          gid[(i + 1) * N + j + 1], gid[i * N + j + 1])
            if min(a, b, c, e) >= 0:
                quads.append([a, b, c, e])
    if not quads:
        return None, None
    return pos, np.asarray(quads, np.int64).reshape(-1, 4)


def project_shell(shell_mesh, N, ctr, L, margin, band0=0):
    """Shrinkwrap a shell from EACH of the 6 cage faces; a surface point is kept
    from the cage that sees it most head-on (ownership), so a shell spanning
    several faces (e.g. a torus half wrapping top+side+inner) is fully covered
    with clean per-direction lattices. Returns (pos, quads, band) where band[i]
    is a global id (shell·6 + dir) used by the seam snap to detect which border
    vertices come from different projection grids."""
    all_pos, all_q, bands, voff = [], [], [], 0
    for k, d in enumerate(DIRS):
        pos, quads = _project_one_dir(shell_mesh, d, N, ctr, L, margin)
        if pos is None:
            continue
        all_pos.append(pos)
        all_q.append(quads + voff)
        bands.append(np.full(len(pos), band0 * 6 + k, np.int64))
        voff += len(pos)
    if not all_pos:
        return None
    return np.vstack(all_pos), np.vstack(all_q), np.concatenate(bands)


# --------------------------------------------------------------------------- #
#  Phase IV — seam snap + weld shells  (§6)
# --------------------------------------------------------------------------- #
def _boundary_verts(P, Q):
    """Vertices on an open boundary edge (edge used by a single quad)."""
    e = np.vstack([Q[:, [0, 1]], Q[:, [1, 2]], Q[:, [2, 3]], Q[:, [3, 0]]])
    e = np.sort(e, axis=1)
    uniq, cnt = np.unique(e, axis=0, return_counts=True)
    bedges = uniq[cnt == 1]
    return np.unique(bedges) if len(bedges) else np.array([], np.int64)


def weld(P, Q, radius):
    """Union-find weld within radius — arc-length seam matching collapses to a
    geometric weld when both shells sampled the shared cut at the same spacing
    (§6.1). Averages welded positions (§6.2 Laplacian seam relaxation, 1 step)."""
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
    newP = np.zeros((len(uniq), 3)); cnt = np.zeros(len(uniq))
    np.add.at(newP, inv, P); np.add.at(cnt, inv, 1)
    newP /= cnt[:, None]
    Q2 = inv[Q]
    good = np.array([len(set(map(int, q))) == 4 for q in Q2], bool)
    return newP, Q2[good]


# --------------------------------------------------------------------------- #
#  Driver
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
#  Feature-aligned router (creased -> humanlogic patches, smooth -> shrinkwrap)
# --------------------------------------------------------------------------- #
def _is_creased(mesh, feature_angle=35.0):
    """Decide whether `mesh` is a hard-surface (creased) solid.

    Reads the `face_adjacency_angles` distribution: a creased solid is bimodal
    (flat-face interiors near 0 + a population of hard edges above the feature
    angle that form connected crease loops). A smooth shape has no such hard-edge
    population. A connectivity guard (mirroring decompose_shells, lines 184-193)
    rejects scattered scan noise so only genuine crease chains trip the path.
    """
    fa = mesh.face_adjacency
    ang = np.degrees(mesh.face_adjacency_angles)
    if not len(ang):
        return False
    crease_mask = ang > feature_angle
    strong_frac = float(np.mean(ang > 60.0))
    crease_frac = float(np.mean(crease_mask))
    if strong_frac <= 0.002 or crease_frac <= 0.0:
        return False
    # connectivity guard: crease edges must form a connected chain spanning a real
    # fraction of the bbox, not scattered grazing-angle scan triangles.
    sharp_edges = mesh.face_adjacency_edges[crease_mask]
    if not len(sharp_edges):
        return False
    g = nx.Graph()
    g.add_edges_from((int(a), int(b)) for a, b in sharp_edges)
    L = float(mesh.extents.max())
    for comp in nx.connected_components(g):
        if len(comp) < 4:
            continue
        pts = mesh.vertices[list(comp)]
        span = float((pts.max(0) - pts.min(0)).max())
        if span >= 0.25 * L:
            return True
    return False


def remesh(vertices, faces, target_quads=2000, lam=0.5, weld_factor=0.6,
           feature_angle=40.0):
    """Feature-aligned quad remesh dispatcher (app entry point).

    Routes by crease content: creased hard-surface inputs go to the
    feature-aligned patch engine (`humanlogic`), which lays edge LOOPS along every
    crease (crisp 90° edges) and coarse flat-face quads; smooth inputs keep the
    visibility-shell shrinkwrap. A watertight gate guards the creased path: if it
    ever fails to produce a watertight all-quad mesh, fall back to the shrinkwrap
    engine (known watertight on the smooth shapes). Returns (V, quads, tris).
    """
    mesh = trimesh.Trimesh(np.asarray(vertices, np.float64),
                           np.asarray(faces, np.int64), process=True)
    if _is_creased(mesh, feature_angle=35.0):
        try:
            import humanlogic as hl
            import coarsen as cz
            P, Q, T = hl.humanlogic_remesh(
                mesh.vertices, mesh.faces, target_quads=target_quads,
                feature_angle=35.0)
            # coarsen flat-face grid regions (self-validating; reverts on failure)
            sp = float(np.sqrt(float(mesh.area) / max(target_quads, 1)))
            P, Q = cz.coarsen(P, Q, mesh, sp)
            # CONFORM the quad mesh exactly onto the original input (the user's
            # "shrinkwrap it on the original to conform exactly"): pull every
            # vertex to the closest point on the input surface, and snap verts
            # near a crease ONTO the crease lines so the rim is a clean circle,
            # not the jagged resampled polygon.
            P = _conform_to_surface(P, Q, mesh, feature_angle=35.0)
            # watertight gate: a creased mesh has NO poles, so it must be a clean
            # all-quad closed surface. Accept only if so; else fall back to smooth.
            if (T is None or not len(T)) and cz.is_watertight_allquad(P, Q):
                # unify winding so every face points outward (humanlogic can emit
                # a flipped face -> would render dark). BFS-consistent + signed-vol.
                import seam_stitch as _st
                P, Q, _T = _st._unify_winding(P, Q, np.zeros((0, 3), np.int64))
                return P, Q, np.zeros((0, 3), np.int64)
        except Exception:  # noqa: BLE001
            pass
        # creased path failed watertight -> smooth fallback (never regress)
    return _shrinkwrap_remesh(vertices, faces, target_quads=target_quads,
                              lam=lam, weld_factor=weld_factor,
                              feature_angle=feature_angle)


def _conform_to_surface(P, Q, mesh, feature_angle=35.0):
    """Conform the quad mesh onto the original input — and make the CREASE EDGE
    LOOPS conform as clean curves (the user's ask: "the edges must conform").

    1. Project every interior vertex onto the input surface (kills bulge).
    2. Extract the input's crease EDGE LOOPS (ordered rings of sharp edges) and
       the quad mesh's crease edge loops. For each quad crease loop, RE-PROJECT
       its vertices onto the matching INPUT crease loop AS AN ORDERED CURVE and
       redistribute them evenly by arc length — so a cylinder rim becomes a clean
       smooth circle, not a jagged 48-gon with bunched/gapped verts.
    """
    P = np.asarray(P, float).copy()
    Q = np.asarray(Q, np.int64).reshape(-1, 4)
    if not len(P):
        return P
    # 1. project everything onto the surface first
    try:
        cp, _, _ = trimesh.proximity.closest_point(mesh, P)
        P = cp
    except Exception:  # noqa: BLE001
        pass
    # 2. conform crease edge LOOPS as curves
    try:
        in_loops = _crease_loops_of_mesh(mesh, feature_angle)        # input creases
        q_loops = _quad_crease_loops(P, Q, feature_angle)            # output crease loops
        if in_loops and q_loops:
            # densely sample each input crease loop as a smooth polyline
            in_curves = [mesh.vertices[lp] for lp in in_loops]
            in_trees = [cKDTree(c) for c in in_curves]
            for qloop in q_loops:
                qpts = P[qloop]
                # match this output loop to the nearest input crease loop
                ctr = qpts.mean(0)
                mi = int(np.argmin([np.linalg.norm(c.mean(0) - ctr) for c in in_curves]))
                target = in_curves[mi]
                # project each output loop vert onto the input crease curve, then
                # redistribute EVENLY along that curve's arc length
                P = _conform_loop_to_curve(P, qloop, target)
    except Exception:  # noqa: BLE001
        pass
    return P


def _crease_loops_of_mesh(mesh, feature_angle):
    """Ordered crease edge loops of the input (rings of sharp edges)."""
    ang = mesh.face_adjacency_angles
    cre = mesh.face_adjacency_edges[ang > np.radians(feature_angle)]
    if not len(cre):
        return []
    g = nx.Graph()
    g.add_edges_from(map(tuple, cre))
    loops = []
    for comp in nx.connected_components(g):
        sub = g.subgraph(comp)
        if any(d != 2 for _, d in sub.degree()):
            continue
        loops.append(_order_cycle(sub))
    return [lp for lp in loops if lp]


def _quad_crease_loops(P, Q, feature_angle):
    """Ordered crease edge loops of the QUAD mesh (edges whose 2 quads meet
    sharply)."""
    p = P[Q]
    n = np.cross(p[:, 1] - p[:, 0], p[:, 3] - p[:, 0])
    nl = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.where(nl < 1e-12, 1.0, nl)
    edge2f = collections.defaultdict(list)
    for fi, q in enumerate(Q):
        for a, b in ((q[0], q[1]), (q[1], q[2]), (q[2], q[3]), (q[3], q[0])):
            edge2f[tuple(sorted((int(a), int(b))))].append(fi)
    cth = np.cos(np.radians(feature_angle))
    g = nx.Graph()
    for (a, b), fs in edge2f.items():
        if len(fs) == 2 and float(n[fs[0]] @ n[fs[1]]) < cth:
            g.add_edge(a, b)
    loops = []
    for comp in nx.connected_components(g):
        sub = g.subgraph(comp)
        if any(d != 2 for _, d in sub.degree()):
            continue
        loops.append(_order_cycle(sub))
    return [lp for lp in loops if lp]


def _order_cycle(sub):
    """Order a simple-cycle graph into a vertex ring."""
    start = min(sub.nodes())
    loop, prev, cur = [start], None, start
    while True:
        nb = [w for w in sub.neighbors(cur) if w != prev]
        if not nb or nb[0] == start:
            break
        loop.append(nb[0]); prev, cur = cur, nb[0]
    return loop if len(loop) == sub.number_of_nodes() else []


def _conform_loop_to_curve(P, qloop, target_pts):
    """Project each output loop vertex PERPENDICULARLY onto the input crease curve
    (closest point on the smooth polyline), WITHOUT moving it tangentially. This
    pulls the rim onto the true crease circle (radius becomes exact, jaggedness
    gone) while keeping each vert connected to its interior grid neighbour, so the
    mesh doesn't tear. A light smoothing pass along the loop removes residual
    polygon kinks from the input's own faceting."""
    P = P.copy()
    tp = np.vstack([target_pts, target_pts[:1]])            # closed polyline
    a = tp[:-1]
    b = tp[1:]
    for v in qloop:
        P[v] = _closest_on_segs(P[v], a, b)
    # tangential smoothing along the loop (Laplacian on the ring), then re-project
    # so the rim is a smooth curve, not a kinked polygon, but stays on the crease.
    m = len(qloop)
    for _ in range(3):
        cur = P[qloop]
        sm = 0.5 * cur + 0.25 * np.roll(cur, 1, 0) + 0.25 * np.roll(cur, -1, 0)
        for i, v in enumerate(qloop):
            P[v] = _closest_on_segs(sm[i], a, b)
    return P


def _closest_on_segs(pt, a, b):
    """Closest point on segments a->b to pt."""
    ab = b - a
    denom = np.einsum("ij,ij->i", ab, ab) + 1e-12
    t = np.clip(np.einsum("ij,ij->i", pt - a, ab) / denom, 0, 1)
    proj = a + t[:, None] * ab
    k = int(np.argmin(np.linalg.norm(proj - pt, axis=1)))
    return proj[k]


def _shrinkwrap_remesh(vertices, faces, target_quads=2000, lam=0.5,
                       weld_factor=0.6, feature_angle=40.0):
    """Full visibility-shell shrinkwrap quad remesh. Returns (V, quads Nx4)."""
    mesh = trimesh.Trimesh(np.asarray(vertices, np.float64),
                           np.asarray(faces, np.int64), process=False)
    bb_min, bb_max = mesh.vertices.min(0), mesh.vertices.max(0)
    ctr = (bb_min + bb_max) / 2
    L = float((bb_max - bb_min).max())
    if L < 1e-9:
        return mesh.vertices, np.zeros((0, 4), np.int64)
    margin = 0.1 * L

    shell = decompose_shells(mesh, lam=lam)
    nshell = int(shell.max() + 1)
    # total quads ≈ nshell · 6 dirs · N² · (~0.3 visible fraction per dir)
    N = max(6, int(round(np.sqrt(target_quads / max(nshell * 6 * 0.3, 1)))))

    all_pos, all_quads, all_band, voff = [], [], [], 0
    for s in range(nshell):
        fids = np.where(shell == s)[0]
        if len(fids) < 2:
            continue
        sub = mesh.submesh([fids], append=True)
        res = project_shell(sub, N, ctr, L, margin, band0=s)
        if res is None:
            continue
        pos, quads, band = res
        all_pos.append(pos)
        all_quads.append(quads + voff)
        all_band.append(band)
        voff += len(pos)

    if not all_pos:
        return mesh.vertices, np.zeros((0, 4), np.int64), np.zeros((0, 3), np.int64)
    P = np.vstack(all_pos)
    Q = np.vstack(all_quads)
    band = np.concatenate(all_band)
    spacing = L / N
    # Phases II-VII: geometric seam segmentation -> per-seam arc-length reconcile +
    # zipper -> N-grid junction poles -> compaction. (seam_stitching_shrinkwrap_paper.md)
    import seam_stitch as st
    P, Q, T = st.stitch(P, Q, band, spacing, surf_mesh=mesh,
                        feature_angle=feature_angle)
    return P, Q, T
