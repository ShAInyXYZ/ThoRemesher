"""
STAGE 1-2 of the feature-driven quad remesh: FEATURE ANALYSIS + MARK STORE.

Pure analysis on the INPUT trimesh. No remeshing happens here. This module
classifies the surface (sharp creases, flat-vs-curved, developable-vs-doubly-
curved), grows label-aware regions, traces ordered crease curves (closed rings
and open junction-to-junction chains), flags which rings are circles (the cap-rim
gate that fixes the cylinder-cap failure), and packs everything into a
`FeatureStore` of POSITIONS + IDENTITIES that stages 3-4 consume.

Key predicates (re-implemented locally so this module stays import-light and
free of circular deps on the remesh engine):
  * crease predicate: mesh.face_adjacency_angles > feature_angle
    (same as humanlogic.segment_patches / visibility_shells._crease_loops_of_mesh)
  * developable-vs-doubly-curved test: principal-curvature ratio |k_min|/|k_max|
    via igl.principal_curvature (_principal_ratio). ~0 on developable surfaces,
    ~1 on doubly-curved ones. Replaces the old integrated-Gaussian-curvature test.
  * flood-fill region growing (same shape as humanlogic.segment_patches) but
    BLOCKED by crease edges AND by label changes (flat<->curved).
  * ordered cycle tracing (same shape as visibility_shells._order_cycle)
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field

import numpy as np
import networkx as nx
import trimesh

try:
    import igl
except Exception:  # pragma: no cover
    igl = None


# --------------------------------------------------------------------------- #
#  Data structures
# --------------------------------------------------------------------------- #
@dataclass
class CreaseLoop:
    id: int
    vids: list                 # ORDERED ring/chain of input vertex ids
    pts: np.ndarray            # (n,3) positions = mesh.vertices[vids]
    closed: bool               # True = ring (cap rim), False = open chain
    is_circle: bool            # True iff planar + circular (THE cap-rim flag)
    plane_n: np.ndarray        # (3,) best-fit plane normal
    center: np.ndarray         # (3,) loop centroid
    radius: float              # mean |pt-center|
    radius_std: float          # std of |pt-center| (the failure metric)


@dataclass
class Region:
    id: int
    fids: np.ndarray           # input face ids in this region
    kind: str                  # "flat" | "curved_dev" | "curved_double" | "transition"
    bounding_loops: list       # CreaseLoop ids that bound this region
    normal: np.ndarray         # area-weighted face normal (flat regions only)
    planarity: float           # max angular deviation of face normals (deg)


@dataclass
class FeatureStore:
    crease_loops: list         # list[CreaseLoop]
    regions: list              # list[Region]
    transitions: list          # list[Region] flagged kind=="transition"
    face_region: np.ndarray    # (nF,) region id per input face (-1 if none)
    feature_angle: float
    face_label: np.ndarray     # (nF,) "flat"/"curved_dev"/"curved_double" per face


# --------------------------------------------------------------------------- #
#  STAGE 1 — per-face classification
# --------------------------------------------------------------------------- #
def _crease_edges(mesh, feature_angle):
    """Sharp dihedral edges (the exact humanlogic / visibility_shells predicate).

    Returns:
      crease_edge_set : set of frozenset({u,v}) input edges above feature_angle
      crease_mask     : (n_adj,) bool over mesh.face_adjacency, edge is a crease
    """
    ang = mesh.face_adjacency_angles
    crease_mask = ang > np.radians(feature_angle)
    cre_edges = mesh.face_adjacency_edges[crease_mask]
    crease_edge_set = {frozenset((int(u), int(v))) for u, v in cre_edges}
    return crease_edge_set, crease_mask


# --------------------------------------------------------------------------- #
#  STAGE 2a — label-aware region growing (flood-fill)
# --------------------------------------------------------------------------- #
def _grow_regions(mesh, crease_edge_set, label_break=None):
    """Flood-fill faces into regions. Flooding is BLOCKED by crease edges (and,
    if `label_break` is given, by a per-face label change too, so a chamfered
    flat cap and a curved wall split even across a sub-crease edge). Same flood
    shape as humanlogic.segment_patches with the extra blockers.

    Geometry (normal, planarity) per region is filled in; `kind` is assigned
    later by `_classify_regions`. Returns (face_region (nF,), regions)."""
    F = np.asarray(mesh.faces, np.int64)
    adj = mesh.face_adjacency
    adje = mesh.face_adjacency_edges

    nbr = collections.defaultdict(list)
    for k in range(len(adj)):
        a, b = int(adj[k, 0]), int(adj[k, 1])
        u, v = int(adje[k, 0]), int(adje[k, 1])
        if frozenset((u, v)) in crease_edge_set:
            continue                       # blocked by crease
        if label_break is not None and label_break[a] != label_break[b]:
            continue                       # blocked by label change
        nbr[a].append(b)
        nbr[b].append(a)

    face_region = -np.ones(len(F), np.int64)
    rid = 0
    regions = []
    fa = mesh.area_faces
    fn = mesh.face_normals
    for f0 in range(len(F)):
        if face_region[f0] >= 0:
            continue
        comp = []
        dq = collections.deque([f0])
        face_region[f0] = rid
        while dq:
            f = dq.popleft()
            comp.append(f)
            for g in nbr[f]:
                if face_region[g] < 0:
                    face_region[g] = rid
                    dq.append(g)
        comp = np.asarray(comp, np.int64)
        w = fa[comp]
        n = (fn[comp] * w[:, None]).sum(0)
        nn = np.linalg.norm(n)
        n = n / nn if nn > 1e-12 else np.array([0.0, 0.0, 1.0])
        # planarity = max angular spread of face normals (deg), 0 for true flat
        cosv = np.clip(fn[comp] @ n, -1.0, 1.0)
        planarity = float(np.degrees(np.arccos(cosv)).max())
        regions.append(Region(
            id=rid, fids=comp, kind="", bounding_loops=[],
            normal=n, planarity=planarity,
        ))
        rid += 1
    return face_region, regions


# --------------------------------------------------------------------------- #
#  STAGE 1/2 — classify each grown region (planarity + Gaussian curvature)
# --------------------------------------------------------------------------- #
def _classify_regions(mesh, face_region, regions, flat_planarity_deg=2.0,
                      dev_ratio=0.45):
    """Assign each region a kind and write a per-face label array.

    A region is `flat` iff its face normals are coplanar within
    `flat_planarity_deg`. Non-flat regions split developable vs doubly-curved by
    the PRINCIPAL-CURVATURE RATIO |k_min|/|k_max| per vertex (median over the
    region): a developable surface (cylinder wall, cone) has one principal
    curvature ~0 so the ratio ~0; a doubly-curved one (sphere, blend) has both
    principals comparable so the ratio ~1.

    This replaces the old integrated-Gaussian-curvature test, which could NOT
    separate the two on discrete meshes — a cylinder's Gaussian K is ~0 in theory
    but noisy in practice and indistinguishable from a sphere's. The principal
    ratio is the correct, robust signal (verified: cylinder 0.005, cone 0.095,
    sphere 0.997). `dev_ratio` is the developable threshold.

    Returns face_label (nF,) str. Mutates region.kind in place."""
    V = np.asarray(mesh.vertices, np.float64)
    F = np.asarray(mesh.faces, np.int64)
    ratio = _principal_ratio(mesh)          # per-vertex |k_min|/|k_max| in [0,1]
    face_label = np.empty(len(F), dtype=object)
    for r in regions:
        if r.planarity <= flat_planarity_deg:
            r.kind = "flat"
        else:
            vids = np.unique(F[r.fids])
            rr = float(np.median(ratio[vids])) if len(vids) else 1.0
            r.kind = "curved_dev" if rr < dev_ratio else "curved_double"
        face_label[r.fids] = r.kind
    return face_label


def _principal_ratio(mesh):
    """Per-vertex |k_min|/|k_max| via igl principal curvature. ~0 on a developable
    surface (one principal curvature ~0), ~1 on a doubly-curved one (sphere)."""
    V = np.ascontiguousarray(mesh.vertices, np.float64)
    F = np.ascontiguousarray(mesh.faces, np.int64)
    try:
        r = igl.principal_curvature(V, F)
        k1, k2 = np.abs(np.asarray(r[2])), np.abs(np.asarray(r[3]))
        kmin = np.minimum(k1, k2)
        kmax = np.maximum(k1, k2)
        return kmin / (kmax + 1e-9)
    except Exception:  # noqa: BLE001
        return np.zeros(len(V))


# --------------------------------------------------------------------------- #
#  STAGE 2b — trace ordered crease curves (closed rings + open chains)
# --------------------------------------------------------------------------- #
def _order_cycle(sub):
    """Order a simple-cycle graph into a vertex ring (visibility_shells._order_cycle)."""
    start = min(sub.nodes())
    loop, prev, cur = [start], None, start
    while True:
        nb = [w for w in sub.neighbors(cur) if w != prev]
        if not nb or nb[0] == start:
            break
        loop.append(nb[0])
        prev, cur = cur, nb[0]
    return loop if len(loop) == sub.number_of_nodes() else []


def _walk_chain(g, junctions, start, nxt, visited):
    """Walk an open seam segment junction->junction (humanlogic._seam_segments)."""
    chain = [start, nxt]
    prev, cur = start, nxt
    visited.add((min(start, nxt), max(start, nxt)))
    while cur not in junctions and cur != start:
        nbrs = [w for w in g[cur]
                if w != prev and (min(cur, w), max(cur, w)) not in visited]
        if not nbrs:
            break
        w = nbrs[0]
        visited.add((min(cur, w), max(cur, w)))
        chain.append(w)
        prev, cur = cur, w
    return chain


def _circle_stats(pts, circle_tol):
    """Best-fit plane + circle stats for an ordered ring of points.

    Returns (plane_n, center, radius, radius_std, is_circle)."""
    c = pts.mean(0)
    d = pts - c
    # least-variance axis = plane normal
    n = np.linalg.svd(d)[2][2]
    r = np.linalg.norm(d, axis=1)
    rmean = float(r.mean()) or 1.0
    planar = float(np.abs(d @ n).max()) / rmean < circle_tol
    rstd = float(r.std())
    is_circle = bool(planar and (rstd / rmean < circle_tol))
    return n, c, float(r.mean()), rstd, is_circle


def _trace_crease_loops(mesh, crease_edge_set, circle_tol):
    """Trace ordered crease curves from the sharp-edge set.

    Closed components with all-degree-2 -> rings (reject if any junction);
    components with junctions -> split into open junction-to-junction chains.
    For each ring, compute is_circle (the cap-rim gate). Returns list[CreaseLoop]."""
    if not crease_edge_set:
        return []
    g = collections.defaultdict(set)
    for e in crease_edge_set:
        u, v = tuple(e)
        g[u].add(v)
        g[v].add(u)
    junctions = {v for v, ns in g.items() if len(ns) != 2}

    V = np.asarray(mesh.vertices, np.float64)
    loops = []
    lid = 0

    # build nx graph for connected-component analysis
    G = nx.Graph()
    for e in crease_edge_set:
        u, v = tuple(e)
        G.add_edge(u, v)

    visited = set()
    for comp in nx.connected_components(G):
        sub = G.subgraph(comp)
        degs = dict(sub.degree())
        if all(d == 2 for d in degs.values()):
            # closed ring
            ring = _order_cycle(sub)
            if not ring:
                continue
            pts = V[ring]
            n, c, r, rstd, isc = _circle_stats(pts, circle_tol)
            loops.append(CreaseLoop(
                id=lid, vids=ring, pts=pts, closed=True, is_circle=isc,
                plane_n=n, center=c, radius=r, radius_std=rstd,
            ))
            lid += 1
        else:
            # has junctions -> open chains
            for j in [v for v in comp if v in junctions]:
                for nb in list(g[j]):
                    e = (min(j, nb), max(j, nb))
                    if e in visited:
                        continue
                    chain = _walk_chain(g, junctions, j, nb, visited)
                    if len(chain) < 2:
                        continue
                    pts = V[chain]
                    n, c, r, rstd, isc = _circle_stats(pts, circle_tol)
                    loops.append(CreaseLoop(
                        id=lid, vids=chain, pts=pts, closed=False,
                        is_circle=False,        # open chains are never cap rims
                        plane_n=n, center=c, radius=r, radius_std=rstd,
                    ))
                    lid += 1
    return loops


# --------------------------------------------------------------------------- #
#  STAGE 2c — attach bounding loops to regions
# --------------------------------------------------------------------------- #
def _attach_bounding_loops(mesh, face_region, regions, crease_loops):
    """Record which crease loops bound each region (loop vert is on a region face)."""
    F = np.asarray(mesh.faces, np.int64)
    vert_regions = collections.defaultdict(set)
    for fi, r in enumerate(face_region):
        if r < 0:
            continue
        for v in F[fi]:
            vert_regions[int(v)].add(int(r))
    for lp in crease_loops:
        touched = set()
        for v in lp.vids:
            touched |= vert_regions.get(int(v), set())
        for r in touched:
            if lp.id not in regions[r].bounding_loops:
                regions[r].bounding_loops.append(lp.id)


# --------------------------------------------------------------------------- #
#  Top-level
# --------------------------------------------------------------------------- #
def analyze(mesh, feature_angle=35.0, circle_tol=0.05):
    """Run STAGE 1-2: classify, grow regions, trace crease loops -> FeatureStore."""
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.Trimesh(np.asarray(mesh.vertices, np.float64),
                               np.asarray(mesh.faces, np.int64), process=False)

    crease_edge_set, _ = _crease_edges(mesh, feature_angle)
    face_region, regions = _grow_regions(mesh, crease_edge_set)
    face_label = _classify_regions(mesh, face_region, regions)
    crease_loops = _trace_crease_loops(mesh, crease_edge_set, circle_tol)
    _attach_bounding_loops(mesh, face_region, regions, crease_loops)

    transitions = [r for r in regions if r.kind == "transition"]
    return FeatureStore(
        crease_loops=crease_loops,
        regions=regions,
        transitions=transitions,
        face_region=face_region,
        feature_angle=feature_angle,
        face_label=face_label,
    )


# --------------------------------------------------------------------------- #
#  Self-test (numeric/topology only) — run: python3 features.py
# --------------------------------------------------------------------------- #
def _report(name, store):
    closed = [lp for lp in store.crease_loops if lp.closed]
    open_ = [lp for lp in store.crease_loops if not lp.closed]
    circles = [lp for lp in closed if lp.is_circle]
    flat = [r for r in store.regions if r.kind == "flat"]
    cdev = [r for r in store.regions if r.kind == "curved_dev"]
    cdbl = [r for r in store.regions if r.kind == "curved_double"]
    print(f"\n=== {name} ===")
    print(f"  total regions      : {len(store.regions)}")
    print(f"    flat             : {len(flat)}  (sizes={[len(r.fids) for r in flat]})")
    print(f"    curved_dev       : {len(cdev)} (sizes={[len(r.fids) for r in cdev]})")
    print(f"    curved_double    : {len(cdbl)} (sizes={[len(r.fids) for r in cdbl]})")
    print(f"  crease loops total : {len(store.crease_loops)}")
    print(f"    closed rings     : {len(closed)} (lengths={[len(lp.vids) for lp in closed]})")
    print(f"    open chains      : {len(open_)} (lengths={[len(lp.vids) for lp in open_]})")
    print(f"    is_circle rings  : {len(circles)}")
    for lp in circles:
        print(f"      circle id={lp.id}: n={len(lp.vids)}  R={lp.radius:.4f}  "
              f"R_std/R={lp.radius_std / (lp.radius or 1):.5f}")
    return dict(regions=len(store.regions), flat=len(flat), cdev=len(cdev),
                cdbl=len(cdbl), loops=len(store.crease_loops),
                closed=len(closed), open=len(open_), circles=len(circles))


if __name__ == "__main__":
    # --- cylinder: expect 2 flat caps + 1 developable wall, 2 circle rims ---
    cyl = trimesh.creation.cylinder(radius=1.0, height=2.0, sections=48)
    cyl_store = analyze(cyl, feature_angle=35.0)
    cyl_r = _report("CYLINDER (r=1, h=2, sections=48)", cyl_store)

    # --- cube: expect 6 flat faces, 12 crease edges meeting at 8 corners,
    #     0 circles. The 12 edges form open chains between the 8 corner junctions.
    cube = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    cube_store = analyze(cube, feature_angle=35.0)
    cube_r = _report("CUBE (2x2x2)", cube_store)

    # junction / corner count for the cube crease graph (sanity on topology)
    ces = _crease_edges(cube, 35.0)[0]
    cg = collections.defaultdict(set)
    for e in ces:
        u, v = tuple(e)
        cg[u].add(v); cg[v].add(u)
    corners = sorted(cg)
    deg_hist = collections.Counter(len(cg[v]) for v in corners)
    print(f"\n  cube crease graph: {len(corners)} crease verts, "
          f"degree histogram {dict(deg_hist)} (expect 8 verts all degree 3)")

    # --- numeric assertions (FAIL LOUD; this is a topology test, not eyeballing) ---
    print("\n=== ASSERTIONS ===")
    ok = True
    def check(cond, msg):
        global ok
        status = "PASS" if cond else "FAIL"
        if not cond:
            ok = False
        print(f"  [{status}] {msg}")

    # cylinder
    check(cyl_r["closed"] == 2, f"cylinder has 2 closed crease rings (got {cyl_r['closed']})")
    check(cyl_r["circles"] == 2, f"cylinder both rims flagged is_circle (got {cyl_r['circles']})")
    check(cyl_r["flat"] == 2, f"cylinder has 2 flat caps (got {cyl_r['flat']})")
    check(cyl_r["cdev"] == 1, f"cylinder wall = 1 developable region (got {cyl_r['cdev']})")
    check(cyl_r["cdbl"] == 0, f"cylinder has 0 doubly-curved regions (got {cyl_r['cdbl']})")
    for lp in cyl_store.crease_loops:
        if lp.closed:
            check(len(lp.vids) == 48, f"cylinder rim ring has 48 verts (got {len(lp.vids)})")

    # cube
    check(cube_r["flat"] == 6, f"cube has 6 flat faces (got {cube_r['flat']})")
    check(cube_r["cdev"] == 0 and cube_r["cdbl"] == 0,
          f"cube has 0 curved regions (got dev={cube_r['cdev']} dbl={cube_r['cdbl']})")
    check(cube_r["circles"] == 0, f"cube has 0 circle rims (got {cube_r['circles']})")
    check(len(corners) == 8, f"cube crease graph has 8 corner verts (got {len(corners)})")
    check(all(len(cg[v]) == 3 for v in corners),
          "all 8 cube corners have crease degree 3")

    print(f"\nRESULT: {'ALL PASS' if ok else 'SOME FAILED'}")
    import sys
    sys.exit(0 if ok else 1)
