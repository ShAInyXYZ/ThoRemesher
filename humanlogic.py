"""
HumanLogic quad remesher — patch-based perceptual retopology (HumanLogic.md).

The artist reads SHARPNESS (creases) and FLATNESS, which cut the surface into
coherent regions, and grids each region cleanly. This engine does exactly that:

  1. Detect creases (sharp dihedral edges)  -> §3 feature lines.
  2. Cut the surface into PATCHES along creases (flat-ish regions bounded by
     sharp edges; a cube -> 6 faces, a cylinder -> tube + 2 caps) -> §2 + flatness.
  3. Flatten each patch to 2D (planar projection, or LSCM if curved).
  4. Lay a clean quad GRID over each patch in 2D and lift it back to the surface.
  5. WELD grid vertices shared along crease edges so patches join into one mesh.

This avoids the global-field seam problem: each patch is gridded in its own
parameterization, and the creases are exactly the patch boundaries the artist
would put edge loops on.

Backbone: trimesh (segmentation, projection) + libigl (LSCM for curved patches).
"""
from __future__ import annotations

import collections

import numpy as np
import scipy.spatial as ssp
import trimesh

try:
    import igl
except Exception:  # pragma: no cover
    igl = None


# --------------------------------------------------------------------------- #
#  Global unwrap for smooth closed / wrap-around patches (sphere, rounded box)
# --------------------------------------------------------------------------- #
def _seam_cut_to_disk(Vp, Fp):
    """Cut a closed/near-closed patch open to a topological disk along a seam
    (shortest edge path between the two extremal vertices). Returns (Vc, Fc) of
    the opened mesh, or None if it cannot be cut/already open."""
    import networkx as nx
    if igl is None:
        return None
    # if it already has a usable boundary, no cut needed
    try:
        bl = np.atleast_1d(igl.boundary_loop(Fp.astype(np.int64)))
    except Exception:  # noqa: BLE001
        bl = []
    if len(bl) >= 8:
        return Vp, Fp
    # seam between farthest-apart-along-principal-axis vertices
    c = Vp.mean(0)
    axis = np.linalg.svd(Vp - c)[2][0]
    proj = (Vp - c) @ axis
    a0, a1 = int(np.argmin(proj)), int(np.argmax(proj))
    g = nx.Graph()
    for f in Fp:
        for u, v in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            g.add_edge(int(u), int(v), weight=np.linalg.norm(Vp[u] - Vp[v]))
    try:
        path = nx.shortest_path(g, a0, a1, weight="weight")
    except Exception:  # noqa: BLE001
        return None
    ps = {(min(path[i], path[i + 1]), max(path[i], path[i + 1]))
          for i in range(len(path) - 1)}
    C = np.zeros((len(Fp), 3), dtype=bool)
    for fi, f in enumerate(Fp):
        for k, (u, v) in enumerate(((f[0], f[1]), (f[1], f[2]), (f[2], f[0]))):
            if (min(int(u), int(v)), max(int(u), int(v))) in ps:
                C[fi, k] = True
    try:
        Vc, Fc, _ = igl.cut_mesh(Vp.astype(np.float64), Fp.astype(np.int64), C)
        return np.asarray(Vc, np.float64), np.asarray(Fc, np.int64)
    except Exception:  # noqa: BLE001
        return None


def _unwrap_uv(Vp, Fp):
    """Harmonic parameterization of a disk patch to the unit circle. Robust and
    bijective for disks. Returns uv (N,2) or None."""
    if igl is None:
        return None
    try:
        bl = np.atleast_1d(igl.boundary_loop(Fp.astype(np.int64)))
        if len(bl) < 3:
            return None
        bnd_uv = igl.map_vertices_to_circle(Vp.astype(np.float64), bl.astype(np.int64))
        uv = np.asarray(igl.harmonic(Vp.astype(np.float64), Fp.astype(np.int64),
                                     bl.astype(np.int64), np.asarray(bnd_uv), 1))
        if uv.shape == (len(Vp), 2) and np.isfinite(uv).all():
            return uv
    except Exception:  # noqa: BLE001
        pass
    return None


def _grid_global_unwrap(mesh, fidx, spacing):
    """Grid a smooth closed/wrap-around patch as ONE connected chart: cut a seam
    -> harmonic-unwrap the whole surface to a disk -> grid the UV -> lift to 3D.

    Returns (pos, quads) or (None, None). This replaces the crude 6-axis-box
    charting that folds and shatters on smooth blobs (sphere, rounded cube).
    """
    Vp, Fp, vids = _patch_submesh(mesh, fidx)
    cut = _seam_cut_to_disk(Vp, Fp)
    if cut is None:
        return None, None
    Vc, Fc = cut
    uv = _unwrap_uv(Vc, Fc)
    if uv is None:
        return None, None
    # grid the UV bbox, keep cells inside the param, lift via barycentric on Vc
    umin, umax = uv[:, 0].min(), uv[:, 0].max()
    vmin, vmax = uv[:, 1].min(), uv[:, 1].max()
    uv_area = (umax - umin) * (vmax - vmin)
    surf_area = float(trimesh.Trimesh(Vc, Fc, process=False).area)
    if surf_area < 1e-12:
        return None, None
    dens = spacing * np.sqrt(uv_area / surf_area)   # uv-space grid spacing
    nu = max(2, int(round((umax - umin) / dens)))
    nv = max(2, int(round((vmax - vmin) / dens)))
    gu = np.linspace(umin, umax, nu)
    gv = np.linspace(vmin, vmax, nv)
    GU, GV = np.meshgrid(gu, gv, indexing="ij")
    guv = np.column_stack([GU.ravel(), GV.ravel()])
    inside = _point_in_patch(guv, uv, Fc)
    gid = -np.ones(nu * nv, np.int64)
    gid[inside] = np.arange(inside.sum())
    pos = _uv_to_world(guv[inside], uv, Fc, Vc)
    quads = []
    def idx(i, j):
        return gid[i * nv + j]
    for i in range(nu - 1):
        for j in range(nv - 1):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            if a >= 0 and b >= 0 and c >= 0 and d >= 0:
                quads.append([a, b, c, d])
    if not quads:
        return None, None
    return pos, np.asarray(quads, np.int64).reshape(-1, 4)


# --------------------------------------------------------------------------- #
#  1-2. Crease detection + patch segmentation
# --------------------------------------------------------------------------- #
def segment_patches(mesh, sharp_deg=35.0):
    """Cut the mesh into patches along sharp dihedral creases.

    Returns per-face patch id (int array). Faces are in the same patch iff
    connected through edges whose dihedral angle is below the crease threshold.
    """
    F = mesh.faces
    adj = mesh.face_adjacency
    ang = np.degrees(mesh.face_adjacency_angles)
    fadj = collections.defaultdict(list)
    for k in range(len(adj)):
        if ang[k] <= sharp_deg:
            a, b = int(adj[k, 0]), int(adj[k, 1])
            fadj[a].append(b)
            fadj[b].append(a)
    patch = -np.ones(len(F), np.int64)
    pid = 0
    for f0 in range(len(F)):
        if patch[f0] >= 0:
            continue
        dq = collections.deque([f0])
        patch[f0] = pid
        while dq:
            f = dq.popleft()
            for g in fadj[f]:
                if patch[g] < 0:
                    patch[g] = pid
                    dq.append(g)
        pid += 1
    return patch, pid


# --------------------------------------------------------------------------- #
#  3. Flatten a patch to 2D
# --------------------------------------------------------------------------- #
def _subpatch_ids(mesh, subpatches):
    """Per-face sub-patch id from a list of face-index arrays."""
    sid = -np.ones(len(mesh.faces), np.int64)
    for i, fidx in enumerate(subpatches):
        sid[fidx] = i
    return sid


def _seam_network(mesh, sid):
    """Edges separating different sub-patches (+ open boundary edges).

    Returns adjacency dict vert->set(verts) over seam edges, and the set of
    junction vertices (degree != 2, where 3+ patches meet or a seam ends).
    """
    adj = mesh.face_adjacency
    adje = mesh.face_adjacency_edges
    seam = set()
    for k in range(len(adj)):
        a, b = int(adj[k, 0]), int(adj[k, 1])
        if sid[a] != sid[b]:
            u, v = int(adje[k, 0]), int(adje[k, 1])
            seam.add((min(u, v), max(u, v)))
    g = collections.defaultdict(set)
    for u, v in seam:
        g[u].add(v); g[v].add(u)
    junctions = {v for v, ns in g.items() if len(ns) != 2}
    return g, junctions


def _seam_segments(g, junctions):
    """Split the seam network into segments (junction-to-junction vertex chains)."""
    visited_edges = set()
    segments = []

    def walk(start, nxt):
        chain = [start, nxt]
        prev, cur = start, nxt
        visited_edges.add((min(start, nxt), max(start, nxt)))
        # stop at a junction OR when we loop back to start (closed seam loop)
        while cur not in junctions and cur != start:
            nbrs = [w for w in g[cur] if w != prev]
            nbrs = [w for w in nbrs
                    if (min(cur, w), max(cur, w)) not in visited_edges]
            if not nbrs:
                break
            w = nbrs[0]
            visited_edges.add((min(cur, w), max(cur, w)))
            chain.append(w)
            prev, cur = cur, w
        return chain

    # junction-to-junction segments first
    for j in junctions:
        for nb in g[j]:
            if (min(j, nb), max(j, nb)) in visited_edges:
                continue
            segments.append(walk(j, nb))
    # closed seam loops (no junctions): walk any remaining edge until it closes
    for v in list(g.keys()):
        for nb in g[v]:
            e = (min(v, nb), max(v, nb))
            if e not in visited_edges:
                segments.append(walk(v, nb))
    return segments


def _resample_chain(V, chain, spacing):
    """Resample a polyline (vertex-id chain) to ~spacing -> 3D points incl ends."""
    pts = V[chain]
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = seg.sum()
    if total < 1e-9:
        return pts[[0]]
    n = max(1, int(round(total / spacing)))
    cum = np.concatenate([[0], np.cumsum(seg)])
    ts = np.linspace(0, total, n + 1)
    out = np.empty((n + 1, 3))
    for i, t in enumerate(ts):
        k = np.searchsorted(cum, t, side="right") - 1
        k = min(max(k, 0), len(seg) - 1)
        local = (t - cum[k]) / max(seg[k], 1e-12)
        out[i] = pts[k] * (1 - local) + pts[k + 1] * local
    return out


def _open_subpatches(mesh, fidx):
    """Classify a patch and return [(face_idx, kind)] for the driver.

    kind = "disk"   -> a single flattenable open disk (normals in a cone, real
                       boundary) -> handled by tube/fan/Coons paths.
    kind = "global" -> a smooth closed / wrap-around blob (sphere, rounded box,
                       organic part) -> gridded as ONE chart by global unwrap
                       (seam-cut + harmonic), avoiding the crude axis-box charts
                       that fold and shatter.
    """
    _, Fp, _ = _patch_submesh(mesh, fidx)
    try:
        bnd = np.atleast_1d(igl.boundary_loop(Fp.astype(np.int64))) if igl else []
    except Exception:  # noqa: BLE001
        bnd = []
    fn = mesh.face_normals[fidx]
    spread_ok = np.linalg.norm(fn.mean(0)) > 0.6       # normals within a cone
    boundary_ok = len(bnd) >= 0.15 * np.sqrt(len(fidx)) and len(bnd) >= 3
    if spread_ok and boundary_ok:
        return [(fidx, "disk")]
    return [(fidx, "global")]


def _patch_submesh(mesh, faces_idx):
    """Extract a patch as its own (V, F, orig_vert_ids)."""
    F = mesh.faces[faces_idx]
    vids = np.unique(F)
    remap = -np.ones(len(mesh.vertices), np.int64)
    remap[vids] = np.arange(len(vids))
    return mesh.vertices[vids], remap[F], vids


def _flatten(Vp, Fp, normal):
    """Flatten a patch to 2D. Planar projection if nearly flat, else LSCM.

    Returns uv (Nx2). Falls back to planar projection if LSCM unavailable/fails.
    """
    c = Vp.mean(0)
    # planarity: how much the patch deviates from its average plane
    dev = np.abs((Vp - c) @ normal)
    extent = np.linalg.norm(Vp.max(0) - Vp.min(0)) + 1e-9
    planar = dev.max() / extent < 0.05

    if planar or igl is None:
        e0 = np.array([1.0, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1.0, 0])
        e0 = e0 - normal * np.dot(e0, normal)
        e0 /= np.linalg.norm(e0) + 1e-12
        e1 = np.cross(normal, e0)
        return np.column_stack([(Vp - c) @ e0, (Vp - c) @ e1])

    # curved -> LSCM (conformal flatten), pin 2 far-apart boundary verts
    try:
        bnd = np.atleast_1d(igl.boundary_loop(Fp.astype(np.int64)))
        if len(bnd) >= 2:
            b = np.array([bnd[0], bnd[len(bnd) // 2]], np.int64)
            bc = np.array([[0.0, 0.0], [1.0, 0.0]])
            ok, uv = igl.lscm(Vp.astype(np.float64), Fp.astype(np.int64), b, bc)
            uv = np.asarray(uv)
            if uv.shape == (len(Vp), 2) and np.isfinite(uv).all():
                return uv
    except Exception:  # noqa: BLE001
        pass
    # fallback: planar projection
    e0 = np.array([1.0, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1.0, 0])
    e0 = e0 - normal * np.dot(e0, normal); e0 /= np.linalg.norm(e0) + 1e-12
    e1 = np.cross(normal, e0)
    return np.column_stack([(Vp - c) @ e0, (Vp - c) @ e1])


# --------------------------------------------------------------------------- #
#  4. Grid a patch in 2D, lift to surface
# --------------------------------------------------------------------------- #
def _point_in_patch(uv_pts, patch_uv, patch_F):
    """Bool mask: which uv_pts lie inside the patch's 2D triangulation."""
    import matplotlib.path  # noqa: F401  (only for hull fallback)
    # use barycentric test against the patch triangles
    inside = np.zeros(len(uv_pts), bool)
    tris = patch_uv[patch_F]  # (T,3,2)
    for t in range(len(tris)):
        a, b, c = tris[t]
        v0 = b - a; v1 = c - a
        d00 = v0 @ v0; d01 = v0 @ v1; d11 = v1 @ v1
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-18:
            continue
        rel = uv_pts - a
        d20 = rel @ v0; d21 = rel @ v1
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1 - v - w
        inside |= (u >= -1e-6) & (v >= -1e-6) & (w >= -1e-6)
    return inside


def _uv_to_world(query_uv, uv, Fp, Vp_world):
    """Map 2D points back to 3D via barycentric coords on the patch triangles."""
    out = np.zeros((len(query_uv), 3))
    tris = uv[Fp]
    from scipy.spatial import cKDTree
    centroids = tris.mean(1)
    tree = cKDTree(centroids)
    for qi, q in enumerate(query_uv):
        # try the nearest few triangles
        _, cand = tree.query(q, k=min(8, len(tris)))
        cand = np.atleast_1d(cand)
        placed = False
        for t in cand:
            a, b, c = tris[t]
            v0 = b - a; v1 = c - a; rel = q - a
            d00 = v0 @ v0; d01 = v0 @ v1; d11 = v1 @ v1
            denom = d00 * d11 - d01 * d01
            if abs(denom) < 1e-18:
                continue
            d20 = rel @ v0; d21 = rel @ v1
            v = (d11 * d20 - d01 * d21) / denom
            w = (d00 * d21 - d01 * d20) / denom
            u = 1 - v - w
            if u >= -1e-4 and v >= -1e-4 and w >= -1e-4:
                W = Vp_world[Fp[t]]
                out[qi] = u * W[0] + v * W[1] + w * W[2]
                placed = True
                break
        if not placed:
            # nearest triangle centroid's vertices avg (robustness)
            out[qi] = Vp_world[Fp[cand[0]]].mean(0)
    return out


# --------------------------------------------------------------------------- #
#  Driver
# --------------------------------------------------------------------------- #
def _all_boundary_loops(mesh, fidx):
    """All ordered boundary loops of a patch (original vertex ids).

    A disk has 1 loop, a tube (cylinder side) has 2, etc.
    """
    Vp, Fp, vids = _patch_submesh(mesh, fidx)
    try:
        bf = igl.boundary_facets(Fp.astype(np.int64))
        bf = bf[0] if isinstance(bf, tuple) else bf
    except Exception:  # noqa: BLE001
        return []
    g = collections.defaultdict(list)
    for a, b in bf:
        g[int(a)].append(int(b)); g[int(b)].append(int(a))
    seen = set()
    loops = []
    for s in list(g.keys()):
        if s in seen:
            continue
        loop = [s]; seen.add(s); cur, prev = s, None
        while True:
            nxts = [w for w in g[cur] if w != prev and (w not in seen or w == s)]
            if not nxts:
                break
            w = nxts[0]
            if w == s:
                break
            loop.append(w); seen.add(w); prev, cur = cur, w
        if len(loop) >= 3:
            loops.append([int(vids[i]) for i in loop])
    return loops


def _apex_vertex(mesh, fidx, loop):
    """The convergence vertex of a tapering patch (cone tip): the NON-rim patch
    vertex shared by the most patch faces. Returns (vertex_id, valence)."""
    rim_set = set(int(v) for v in loop)
    faces = mesh.faces[fidx]
    counts = collections.Counter()
    for f in faces:
        for v in f:
            if int(v) not in rim_set:
                counts[int(v)] += 1
    if not counts:
        return None, 0
    v, c = counts.most_common(1)[0]
    return v, c


def _is_tapering(mesh, fidx, loop):
    """True only for a cone-like patch that collapses to an apex POINT OFF the
    rim plane. Requires both: (a) a high-convergence apex vertex (many faces
    share it), and (b) that apex sits well off the rim plane (rules out flat
    fan-triangulated caps, whose center is ON the plane). Domes (valence-6) and
    flat caps are excluded -> handled by the Coons path."""
    # A single high-convergence center vertex (cone tip OR flat fan cap).
    v, val = _apex_vertex(mesh, fidx, loop)
    if v is None or val < max(8, 0.5 * len(loop)):
        return False
    # ...but a FLAT cap's center is coplanar with its rim (it's just a fan center
    # on a disk) — that must be GRIDDED, not fanned to a pole (else the cap gets a
    # radial starburst). Only a real cone APEX, which sits well OFF the rim plane,
    # is tapering. Measure the apex's distance from the rim's best-fit plane,
    # relative to the rim radius.
    rim = mesh.vertices[loop]
    ctr = rim.mean(0)
    # rim plane normal = principal axis of least variance
    u, s, vt = np.linalg.svd(rim - ctr)
    n = vt[2]
    rim_radius = np.linalg.norm(rim - ctr, axis=1).mean()
    apex_off_plane = abs((mesh.vertices[v] - ctr) @ n)
    return apex_off_plane > 0.25 * rim_radius      # real cone apex, not a flat cap


def _grid_disk_fan(mesh, fidx, loop, spacing, rim_pool=None):
    """Grid a cone-like disk patch as concentric quad rings from the rim that
    collapse to the cone's ORIGINAL apex vertex.

    The innermost ring connects directly to the apex point (one shared center
    pole) — so there is no separate cap patch to weld and the tip is always
    connected. The apex is a single valence-N pole at the true tip. Returns
    (pos, quads, tris) — the apex closes with triangles (degenerate quads).
    """
    if rim_pool is not None and frozenset(loop) in rim_pool:
        rim = rim_pool[frozenset(loop)]
    else:
        rp = mesh.vertices[loop]
        circ = np.linalg.norm(np.diff(np.vstack([rp, rp[:1]]), axis=0), axis=1).sum()
        rim = _resample_loop(rp, max(4, int(round(circ / spacing))))
    N = len(rim)
    rim_c = rim.mean(0)
    av, _val = _apex_vertex(mesh, fidx, loop)
    apex = mesh.vertices[av] if av is not None else \
        trimesh.proximity.closest_point(mesh, rim_c[None])[0][0]
    rim_rad = np.linalg.norm(rim - rim_c, axis=1).mean()

    # Concentric quad rings from the rim toward the apex, but STOP while the ring
    # is still large enough that its N points don't weld together (radius must
    # stay above ~the ring's own point spacing = circumference/N). Then KEEP the
    # original apex vertex and triangle-fan the remaining gap to it. This is the
    # robust rule for extreme/sharp peaks: keep the tip, fill the gap.
    min_ring_rad = max(2.0 * spacing, 1.5 * (2 * np.pi * rim_rad / N))
    mean_slant = max(np.linalg.norm(rim - apex, axis=1).mean(), 1e-9)
    rings = [rim]
    t = 0.0
    for _ in range(200):
        next_t = t + spacing / mean_slant     # advance ~one spacing along slant
        if next_t >= 1.0:
            break
        ring = (1 - next_t) * rim + next_t * apex
        ring_rad = np.linalg.norm(ring - ring.mean(0), axis=1).mean()
        if ring_rad < min_ring_rad:
            break
        rings.append(trimesh.proximity.closest_point(mesh, ring)[0])
        t = next_t
    P = np.vstack(rings + [apex[None]])
    M_rings = len(rings) - 1
    apex_idx = len(rings) * N
    quads = []
    for i in range(M_rings):
        a0, b0 = i * N, (i + 1) * N
        for j in range(N):
            quads.append([a0 + j, a0 + (j + 1) % N, b0 + (j + 1) % N, b0 + j])
    # fan the last (still-sizable) ring to the kept apex vertex (real triangles)
    last = M_rings * N
    tris = [[last + j, last + (j + 1) % N, apex_idx] for j in range(N)]
    return P, np.asarray(quads, np.int64).reshape(-1, 4), \
        np.asarray(tris, np.int64).reshape(-1, 3)


def _grid_flat_disk(mesh, fidx, loop, spacing, rim_pool=None):
    """Grid a FLAT circular cap (e.g. a cylinder end) as concentric quad rings
    whose OUTER RING IS THE EXACT RIM LOOP (shared with the tube -> clean circular
    rim, watertight weld), shrinking toward a single centre pole in the rim plane.

    This replaces the square 4-sided Coons patch that forced a circle into a
    square (corners collapsing to radius 0.707, jagged rim). Returns (pos, quads,
    tris); tris only at the small centre fan if the inner ring can't be a clean
    quad ring.
    """
    if rim_pool is not None and frozenset(loop) in rim_pool:
        rim = rim_pool[frozenset(loop)]
    else:
        rp = mesh.vertices[loop]
        circ = np.linalg.norm(np.diff(np.vstack([rp, rp[:1]]), axis=0), axis=1).sum()
        rim = _resample_loop(rp, max(4, int(round(circ / spacing))))
    N = len(rim)
    centre = rim.mean(0)
    rim_rad = np.linalg.norm(rim - centre, axis=1).mean()
    # concentric rings from the rim shrinking toward the centre, stopping while
    # the ring is still large enough not to self-weld, then fan to one centre pole.
    min_ring_rad = max(1.2 * spacing, 1.3 * (2 * np.pi * rim_rad / N))
    rings = [rim]
    rad = rim_rad
    while rad - spacing > min_ring_rad:
        rad -= spacing
        f = rad / rim_rad
        ring = centre + f * (rim - centre)       # shrink toward centre, in plane
        ring = trimesh.proximity.closest_point(mesh, ring)[0]   # conform to cap
        rings.append(ring)
    P = np.vstack(rings + [trimesh.proximity.closest_point(mesh, centre[None])[0]])
    centre_idx = len(rings) * N
    M = len(rings) - 1
    quads = []
    for i in range(M):
        a0, b0 = i * N, (i + 1) * N
        for j in range(N):
            quads.append([a0 + j, a0 + (j + 1) % N, b0 + (j + 1) % N, b0 + j])
    last = M * N
    tris = [[last + j, last + (j + 1) % N, centre_idx] for j in range(N)]
    return P, np.asarray(quads, np.int64).reshape(-1, 4), \
        np.asarray(tris, np.int64).reshape(-1, 3)


def _is_flat_disk(mesh, fidx, loop):
    """True ONLY for a flat CIRCULAR cap: one closed planar loop that is round
    (no sharp polygon corners). A flat SQUARE face (cube) is excluded — it has 4
    sharp corners and must be Coons-gridded into clean quads, not ring-fanned.
    Concentric-ring fan is for round caps (cylinder/cone ends) only.
    """
    rp = mesh.vertices[loop]
    c = rp.mean(0)
    u, s, vt = np.linalg.svd(rp - c)
    n = vt[2]
    rad = np.linalg.norm(rp - c, axis=1)
    mrad = rad.mean()
    if mrad < 1e-9:
        return False
    # planar boundary
    if np.abs((rp - c) @ n).max() >= 0.2 * mrad:
        return False
    # flat interior (a cap, not a dome)
    if np.linalg.norm(mesh.face_normals[fidx].mean(0)) <= 0.9:
        return False
    # ROUND: roughly constant radius (a circle) AND no sharp corners. A square's
    # corners are at sqrt(2)*edge from centre vs edge-mid at edge -> radius varies
    # ~41%. A polygon-circle's radius is near-constant (<~5%).
    if rad.std() / mrad > 0.12:
        return False
    # corner test: count boundary verts with a sharp turn (interior angle far
    # from straight). A circle has ~none; a square has 4.
    P2 = rp
    m = len(P2)
    sharp = 0
    for i in range(m):
        a = P2[(i - 1) % m] - P2[i]
        b = P2[(i + 1) % m] - P2[i]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            continue
        cosang = a @ b / (na * nb)
        if cosang > -0.3:                 # turn sharper than ~107deg = a corner
            sharp += 1
    return sharp <= 2                     # circle: ~0 corners; square: 4 -> excluded


def _plane_normal(pts):
    c = pts.mean(0)
    u, s, vt = np.linalg.svd(pts - c)
    return vt[2]


def _snap_to_rim(pos, rim_tree, rim_points, spacing):
    """Snap grid points near a shared rim onto the canonical rim point so that
    the tube and its caps use identical rim vertices (-> weld)."""
    if rim_tree is None or not len(pos):
        return pos
    d, idx = rim_tree.query(pos)
    near = d < 0.5 * spacing
    pos = pos.copy()
    pos[near] = rim_points[idx[near]]
    return pos


def _grid_tube(mesh, loops, spacing, rim_pool=None):
    """Grid a tube (2 boundary rims) as a swept ring-grid between the rims.

    Both rims use the SHARED rim-pool resampling (so they weld to adjacent caps);
    the two rims are paired by nearest-rotation; M rows interpolate between them.
    """
    def pooled(loop):
        if rim_pool is not None:
            r = rim_pool.get(frozenset(loop))
            if r is not None:
                return r
        rp = mesh.vertices[loop]
        c = np.linalg.norm(np.diff(np.vstack([rp, rp[:1]]), axis=0), axis=1).sum()
        return _resample_loop(rp, max(4, int(round(c / spacing))))

    R0 = pooled(loops[0])
    R1 = pooled(loops[1])
    # both rims must have the same ring count to sweep; resample the finer one
    N = min(len(R0), len(R1))
    if len(R0) != N:
        R0 = _resample_loop(R0, N)
    if len(R1) != N:
        R1 = _resample_loop(R1, N)
    # align R1's start + direction to R0 (nearest start, consistent winding)
    R1 = _align_ring(R0, R1)
    # rows along the height
    h = np.linalg.norm(R0.mean(0) - R1.mean(0))
    M = max(1, int(round(h / spacing)))
    pos = []
    for i in range(M + 1):
        t = i / M
        ring = (1 - t) * R0 + t * R1
        pos.append(ring)
    pos = np.vstack(pos)
    # project interior rows onto the surface (rims already on it)
    pos = trimesh.proximity.closest_point(mesh, pos)[0]
    quads = []
    for i in range(M):
        for j in range(N):
            a = i * N + j
            b = i * N + (j + 1) % N
            c = (i + 1) * N + (j + 1) % N
            d = (i + 1) * N + j
            quads.append([a, b, c, d])
    return pos, np.asarray(quads, np.int64).reshape(-1, 4)


def _resample_loop(pts, n):
    """Resample a CLOSED loop to exactly n points (no duplicate endpoint)."""
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    total = seg.sum()
    cum = np.concatenate([[0], np.cumsum(seg)])
    ts = np.linspace(0, total, n, endpoint=False)
    out = np.empty((n, 3))
    for i, t in enumerate(ts):
        k = min(max(np.searchsorted(cum, t, "right") - 1, 0), len(seg) - 1)
        local = (t - cum[k]) / max(seg[k], 1e-12)
        out[i] = closed[k] * (1 - local) + closed[k + 1] * local
    return out


def _align_ring(R0, R1):
    """Rotate/flip R1 so its points pair with R0 by nearest position + winding."""
    n = len(R1)
    # match winding: if reversing R1 puts its centroid-relative order closer, flip
    # find the offset that minimizes sum of distances
    best, best_d, best_flip = 0, 1e18, False
    for flip in (False, True):
        Rt = R1[::-1] if flip else R1
        for off in range(n):
            cand = np.roll(Rt, -off, axis=0)
            d = np.linalg.norm(cand - R0, axis=1).sum()
            if d < best_d:
                best_d, best, best_flip = d, off, flip
    Rt = R1[::-1] if best_flip else R1
    return np.roll(Rt, -best, axis=0)


def _patch_boundary_sides(mesh, fidx, junction_set):
    """Ordered boundary loop of a patch, split into sides at junction corners.

    Returns a list of sides; each side is an ordered list of ORIGINAL vertex ids
    from one corner to the next (corners included at both ends, shared). Returns
    None if the patch has no usable boundary.
    """
    Vp, Fp, vids = _patch_submesh(mesh, fidx)
    try:
        bnd = np.atleast_1d(igl.boundary_loop(Fp.astype(np.int64))) if igl else []
    except Exception:  # noqa: BLE001
        bnd = []
    if len(bnd) < 4:
        return None
    loop = [int(vids[i]) for i in bnd]            # original vertex ids, ordered
    corners = [k for k, v in enumerate(loop) if v in junction_set]
    if len(corners) < 2:
        # no junctions on this loop (e.g. an isolated chart) -> 4 even corners
        n = len(loop)
        corners = [0, n // 4, n // 2, 3 * n // 4]
    sides = []
    for i in range(len(corners)):
        a = corners[i]
        b = corners[(i + 1) % len(corners)]
        if b > a:
            side = loop[a:b + 1]
        else:
            side = loop[a:] + loop[:b + 1]
        if len(side) >= 2:
            sides.append(side)
    return sides


def _coons(bottom, top, left, right):
    """Bilinearly-blended Coons patch from 4 ordered boundary point arrays.

    bottom/top have n+1 points, left/right have m+1. Corners are shared:
    bottom[0]==left[0], bottom[-1]==right[0], top[0]==left[-1], top[-1]==right[-1].
    Returns S[(m+1),(n+1),3] grid; boundary equals the inputs exactly.
    """
    n = len(bottom) - 1
    m = len(left) - 1
    u = np.linspace(0, 1, n + 1)
    v = np.linspace(0, 1, m + 1)
    S = np.zeros((m + 1, n + 1, 3))
    c00, c10 = bottom[0], bottom[n]
    c01, c11 = top[0], top[n]
    for i in range(m + 1):
        vi = v[i]
        for j in range(n + 1):
            uj = u[j]
            Lc = (1 - uj) * left[i] + uj * right[i]
            Ld = (1 - vi) * bottom[j] + vi * top[j]
            B = ((1 - uj) * (1 - vi) * c00 + uj * (1 - vi) * c10
                 + (1 - uj) * vi * c01 + uj * vi * c11)
            S[i, j] = Lc + Ld - B
    return S


def _patch_planarity(mesh, fidx):
    """Planar deviation of a patch (0 = flat). Mirrors _flatten's planarity test
    (lines 304-307): max distance from the average plane / patch extent."""
    Vp = mesh.vertices[np.unique(mesh.faces[fidx].ravel())]
    if len(Vp) < 3:
        return 0.0
    c = Vp.mean(0)
    n = _plane_normal(Vp)
    dev = np.abs((Vp - c) @ n)
    extent = np.linalg.norm(Vp.max(0) - Vp.min(0)) + 1e-9
    return float(dev.max() / extent)


def humanlogic_remesh(vertices, faces, target_quads=2000, feature_angle=35.0,
                      weld_factor=0.25, flat_factor=3.0, **_ignored):
    """Patch-based HumanLogic remesh via transfinite (Coons) gridding.

    Creases at `feature_angle` cut the surface into patches. Shared seam curves
    are resampled ONCE into a fixed number of points; each patch is gridded by
    Coons interpolation between its 4 (resampled) sides, so patches sharing a
    seam use identical boundary points and weld exactly. Returns (V, quads, tris).
    """
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices, np.float64),
                           faces=np.asarray(faces, np.int64), process=True)
    patch, npatch = segment_patches(mesh, feature_angle)
    spacing3d = np.sqrt(float(mesh.area) / max(target_quads, 1))
    # Flat-face coarsening: a hard-surface solid whose patches are predominantly
    # planar (a box) wants few BIG flat quads, not a fine uniform grid. Scale the
    # SINGLE global spacing up when the (area-weighted) majority of patch area is
    # flat. Keeping ONE spacing preserves the shared-seam resampling invariant, so
    # every weld stays 1:1 and the mesh stays watertight.
    flat_area = total_area = 0.0
    for p in range(npatch):
        fp = np.where(patch == p)[0]
        if not len(fp):
            continue
        a = float(mesh.area_faces[fp].sum())
        total_area += a
        if _patch_planarity(mesh, fp) < 0.02:
            flat_area += a
    if total_area > 0 and flat_area / total_area > 0.7:
        spacing3d *= flat_factor

    subpatches = []           # bare face-idx arrays for the disk/tube/fan paths
    global_patches = []       # smooth closed/wrap-around blobs -> global unwrap
    for p in range(npatch):
        fidx = np.where(patch == p)[0]
        if not len(fidx):
            continue
        for sub, kind in _open_subpatches(mesh, fidx):
            if kind == "global":
                global_patches.append(sub)
            else:
                subpatches.append(sub)

    all_pos_global, all_quads_global, gvoff = [], [], 0
    for fidx in global_patches:
        pos, quads = _grid_global_unwrap(mesh, fidx, spacing3d)
        if pos is None or not len(quads):
            continue
        all_pos_global.append(pos)
        all_quads_global.append(quads + gvoff)
        gvoff += len(pos)

    sid = _subpatch_ids(mesh, subpatches)
    _, junctions = _seam_network(mesh, sid)

    # --- shared RIM pool: a "rim" is a closed boundary loop SHARED by two
    # patches (a tube and its cap). Resample each shared rim ONCE so both snap to
    # the same points and weld. Loops belonging to only one patch (e.g. a sphere
    # chart border) are NOT rims and are excluded, so they aren't disturbed. ---
    loop_owners = collections.defaultdict(int)
    loop_verts = {}
    for fidx in subpatches:
        for loop in _all_boundary_loops(mesh, fidx):
            k = frozenset(loop)
            loop_owners[k] += 1
            loop_verts[k] = loop
    rim_pool = {}
    for k, owners in loop_owners.items():
        if owners < 2:
            continue  # not shared -> not a weld rim
        rpts = mesh.vertices[loop_verts[k]]
        circ = np.linalg.norm(
            np.diff(np.vstack([rpts, rpts[:1]]), axis=0), axis=1).sum()
        N = max(4, int(round(circ / spacing3d)))
        rim_pool[k] = _resample_loop(rpts, N)
    rim_points = np.vstack(list(rim_pool.values())) if rim_pool else np.zeros((0, 3))
    rim_tree = ssp.cKDTree(rim_points) if len(rim_points) else None

    # --- shared side resampling: a side is keyed by its endpoint vertex pair so
    # two patches sharing it resample to the SAME count and SAME points. ---
    side_pts_cache = {}   # frozenset(end verts) + length signature -> 3D points

    def side_key(side):
        return (min(side[0], side[-1]), max(side[0], side[-1]), len(side))

    def resampled_side(side):
        # length-based subdivision count, shared across patches by geometry
        pts = mesh.vertices[side]
        L = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
        n = max(1, int(round(L / spacing3d)))
        key = (min(side[0], side[-1]), max(side[0], side[-1]), n)
        if key in side_pts_cache:
            cached, ckey = side_pts_cache[key]
            # orient cached chain to match this side's start
            if np.linalg.norm(cached[0] - mesh.vertices[side[0]]) > \
               np.linalg.norm(cached[-1] - mesh.vertices[side[0]]):
                return cached[::-1]
            return cached
        rs = _resample_chain(mesh.vertices, side, L / n)
        side_pts_cache[key] = (rs, key)
        return rs

    all_pos, all_quads, all_tris, voff = [], [], [], 0
    for fidx in subpatches:
        loops = _all_boundary_loops(mesh, fidx)
        # TUBE patches (a smooth side wall, e.g. cylinder) have TWO boundary loops
        # (top + bottom rim). Grid as a swept ring-grid between the two rims.
        if len(loops) == 2:
            try:
                pos, quads = _grid_tube(mesh, loops, spacing3d, rim_pool)
            except Exception:  # noqa: BLE001
                pos, quads = None, None
            if pos is not None and len(quads):
                pos = _snap_to_rim(pos, rim_tree, rim_points, spacing3d)
                all_pos.append(pos)
                all_quads.append(quads + voff)
                voff += len(pos)
            continue

        # TAPERING DISK (cone side / pointed tip): one rim that collapses to an
        # apex. Grid as concentric rings -> 1 center pole (fan).
        if len(loops) == 1 and _is_tapering(mesh, fidx, loops[0]):
            try:
                pos, quads, tris = _grid_disk_fan(mesh, fidx, loops[0],
                                                  spacing3d, rim_pool)
            except Exception:  # noqa: BLE001
                pos, quads, tris = None, None, None
            if pos is not None and (len(quads) or len(tris)):
                pos = _snap_to_rim(pos, rim_tree, rim_points, spacing3d)
                all_pos.append(pos)
                if len(quads):
                    all_quads.append(quads + voff)
                if len(tris):
                    all_tris.append(tris + voff)
                voff += len(pos)
            continue

        # FLAT CIRCULAR CAP (one closed planar rim, e.g. a cylinder end): grid as
        # concentric rings whose outer ring IS the rim loop (clean circle, welds
        # to the tube), not a square Coons patch (which collapses the circle to a
        # 0.707-radius square and jaggs the rim).
        if len(loops) == 1 and _is_flat_disk(mesh, fidx, loops[0]):
            try:
                pos, quads, tris = _grid_flat_disk(mesh, fidx, loops[0],
                                                   spacing3d, rim_pool)
            except Exception:  # noqa: BLE001
                pos, quads, tris = None, None, None
            if pos is not None and (len(quads) or len(tris)):
                pos = _snap_to_rim(pos, rim_tree, rim_points, spacing3d)
                all_pos.append(pos)
                if len(quads):
                    all_quads.append(quads + voff)
                if len(tris):
                    all_tris.append(tris + voff)
                voff += len(pos)
            continue

        try:
            sides = _patch_boundary_sides(mesh, fidx, junctions)
        except Exception:  # noqa: BLE001
            sides = None
        if not sides or len(sides) < 3:
            continue
        if len(sides) != 4:
            sides = _merge_to_four(sides) if len(sides) > 4 else None
        if not sides or len(sides) != 4:
            continue
        try:
            # resample the 4 sides; enforce opposite-side equal counts
            rs = [resampled_side(s) for s in sides]
            bottom, right, top, left = rs[0], rs[1], rs[2][::-1], rs[3][::-1]
            nb = max(len(bottom), len(top)); nl = max(len(left), len(right))
            bottom = _resample_n(bottom, nb); top = _resample_n(top, nb)
            left = _resample_n(left, nl); right = _resample_n(right, nl)
            bottom, top, left, right = _align_corners(bottom, top, left, right)
            S = _coons(bottom, top, left, right)
            m1, n1, _ = S.shape
            if m1 < 2 or n1 < 2:
                continue
            pos = trimesh.proximity.closest_point(mesh, S.reshape(-1, 3))[0]
            quads = []
            for i in range(m1 - 1):
                for j in range(n1 - 1):
                    a = i * n1 + j; b = i * n1 + j + 1
                    c = (i + 1) * n1 + j + 1; d = (i + 1) * n1 + j
                    quads.append([a, b, c, d])
            quads = np.asarray(quads, np.int64).reshape(-1, 4)
        except Exception:  # noqa: BLE001
            continue
        pos = _snap_to_rim(pos, rim_tree, rim_points, spacing3d)
        all_pos.append(pos)
        all_quads.append(quads + voff)
        voff += len(pos)

    # merge global-unwrap patches (offset their indices past the local ones)
    base = sum(len(p) for p in all_pos)
    for p, qd in zip(all_pos_global, all_quads_global):
        all_pos.append(p)
        all_quads.append(qd + base)
        base += len(p)

    if not all_pos:
        return mesh.vertices, np.zeros((0, 4), np.int64), np.zeros((0, 3), np.int64)
    V = np.vstack(all_pos)
    Q = np.vstack(all_quads) if all_quads else np.zeros((0, 4), np.int64)
    T = np.vstack(all_tris) if all_tris else np.zeros((0, 3), np.int64)
    V, Q, T = _weld(V, Q, max(weld_factor * spacing3d, 1e-6), T)
    Q = _unify_winding(V, Q, mesh)
    return V, Q, T


def _resample_n(pts, n):
    """Resample a polyline to exactly n points (incl. ends)."""
    pts = np.asarray(pts)
    if len(pts) == n:
        return pts
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = seg.sum()
    if total < 1e-12:
        return np.repeat(pts[[0]], n, axis=0)
    cum = np.concatenate([[0], np.cumsum(seg)])
    ts = np.linspace(0, total, n)
    out = np.empty((n, 3))
    for i, t in enumerate(ts):
        k = min(max(np.searchsorted(cum, t, "right") - 1, 0), len(seg) - 1)
        local = (t - cum[k]) / max(seg[k], 1e-12)
        out[i] = pts[k] * (1 - local) + pts[k + 1] * local
    return out


def _align_corners(bottom, top, left, right):
    """Flip sides so the 4 shared corners coincide for the Coons blend."""
    # left should start at bottom[0]
    if np.linalg.norm(left[0] - bottom[0]) > np.linalg.norm(left[-1] - bottom[0]):
        left = left[::-1]
    if np.linalg.norm(right[0] - bottom[-1]) > np.linalg.norm(right[-1] - bottom[-1]):
        right = right[::-1]
    if np.linalg.norm(top[0] - left[-1]) > np.linalg.norm(top[-1] - left[-1]):
        top = top[::-1]
    return bottom, top, left, right


def _merge_to_four(sides):
    """Greedily merge an N-sided boundary into 4 sides (by vertex count)."""
    sides = [list(s) for s in sides]
    while len(sides) > 4:
        lens = [len(s) for s in sides]
        i = int(np.argmin(lens))
        j = (i + 1) % len(sides)
        sides[i] = sides[i] + sides[j][1:]
        del sides[j]
    return sides


def _weld(V, quads, radius, tris=None):
    """Weld vertices within `radius` (joins patches along shared seams)."""
    tree = ssp.cKDTree(V)
    parent = np.arange(len(V))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in tree.query_pairs(radius, output_type="ndarray"):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = np.array([find(i) for i in range(len(V))])
    uniq, inv = np.unique(roots, return_inverse=True)
    newV = np.zeros((len(uniq), 3)); cnt = np.zeros(len(uniq))
    np.add.at(newV, inv, V); np.add.at(cnt, inv, 1.0)
    newV /= cnt[:, None]
    q = inv[quads] if len(quads) else quads
    qgood = np.array([len(set(r)) == 4 for r in q]) if len(q) else np.zeros(0, bool)
    q = q[qgood].reshape(-1, 4)
    if tris is not None and len(tris):
        t = inv[tris]
        tgood = np.array([len(set(r)) == 3 for r in t])
        t = t[tgood].reshape(-1, 3)
    else:
        t = np.zeros((0, 3), np.int64)
    return newV, q, t


def _unify_winding(V, quads, mesh):
    if not len(quads):
        return quads
    p = V[quads]
    qn = np.cross(p[:, 1] - p[:, 0], p[:, 3] - p[:, 0])
    qn /= np.clip(np.linalg.norm(qn, axis=1, keepdims=True), 1e-12, None)
    c = p.mean(1)
    _, _, fid = trimesh.proximity.closest_point(mesh, c)
    flip = np.einsum("ij,ij->i", qn, mesh.face_normals[fid]) < 0
    quads[flip] = quads[flip][:, ::-1]
    return quads
