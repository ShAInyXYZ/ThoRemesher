"""
Seam-stitching for volumetric shrinkwrap quad remeshing.

Implements seam_stitching_shrinkwrap_paper.md (Phases II-VII), with the
union-curve + explicit-junction architecture that makes CURVED-surface seams
co-extensive (the original per-side reconciliation only closed on the cube,
where the two sides happened to coincide):

  - EXPLICIT JUNCTIONS first. Each band boundary loop splits (by partner-band
    labeling with a generous radius) into runs, one per adjacent band. The run
    ENDPOINTS of all bands cluster into corner "junctions" where >=3 bands meet.
    Each junction collapses to ONE shared vertex. This kills the corner gaps
    (no more `-1` runs and no per-corner tears) and gives every seam two shared,
    pinned endpoints.
  - UNION consensus seam curve. A seam (i,j) is the pair of runs that share the
    same two junctions. Its single shared curve is built from the UNION of BOTH
    sides' vertices, ordered along the seam and pinned junction->junction, then
    resampled. Because it uses the union it covers the full silhouette span
    either jagged staircase reaches, and lies between the two offset sides.
  - RE-POINT BOTH SIDES onto that one curve (+ pin run endpoints to the shared
    junction vertices). Both sides then reference identical seam vertices ->
    manifold shared edges, watertight, regardless of how each side was sampled.
  - N-grid JUNCTIONS: leftover small rings -> poles/fans (resolve_junctions).

The input is the per-band vertex/quad soup from visibility_shells.project (V, Q,
band) where band[v] tags which projection grid vertex v came from. Output is a
watertight (V, quads, tris) — tris only for unavoidably-odd junction holes.
"""
from __future__ import annotations

import collections

import numpy as np
import networkx as nx
from scipy.spatial import cKDTree


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #
def _quad_edges(Q):
    return np.vstack([Q[:, [0, 1]], Q[:, [1, 2]], Q[:, [2, 3]], Q[:, [3, 0]]])


def boundary_edges_global(Q):
    """Undirected edges used by exactly one quad across the WHOLE mesh."""
    e = _quad_edges(Q)
    keyed = collections.Counter(tuple(sorted(map(int, x))) for x in e)
    return [x for x, c in keyed.items() if c == 1]


def resample_polyline(pts, m, closed=False):
    """Resample an ordered polyline to exactly m points by arc length.
    Open: endpoints preserved. Closed: m points around the loop (no repeat)."""
    pts = np.asarray(pts, float)
    if closed:
        pts = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total < 1e-12:
        return np.repeat(pts[:1], m, axis=0)
    if closed:
        t = np.linspace(0.0, total, m + 1)[:-1]
    else:
        t = np.linspace(0.0, total, m)
    out = np.column_stack([np.interp(t, s, pts[:, k]) for k in range(3)])
    return out


# --------------------------------------------------------------------------- #
#  Phase II — geometric seam segmentation (§3, hardened)
# --------------------------------------------------------------------------- #
def _band_boundary_loops(Q, band):
    """Ordered boundary loop(s) of each band, computed per-band (so junction
    corners don't merge bands). Returns {band: [loop, ...]} of global vert ids."""
    out = collections.defaultdict(list)
    by_band = collections.defaultdict(list)
    for q in Q:
        by_band[int(band[q[0]])].append(q)
    for b, qs in by_band.items():
        Qb = np.asarray(qs, np.int64)
        e = _quad_edges(Qb)
        # directed boundary half-edge = reverse twin absent (keeps orientation)
        he = set(map(tuple, e.tolist()))
        bhe = [(a, b2) for (a, b2) in e.tolist() if (b2, a) not in he]
        succ = collections.defaultdict(list)
        for a, b2 in bhe:
            succ[a].append(b2)
        remaining = set(bhe)
        while remaining:
            a0, b0 = next(iter(remaining))
            loop = [a0]
            a, b2 = a0, b0
            ok = True
            while True:
                remaining.discard((a, b2))
                if b2 == a0:
                    break
                loop.append(b2)
                nb = [w for w in succ[b2] if (b2, w) in remaining]
                if not nb:
                    ok = False
                    break
                a, b2 = b2, nb[0]
            if ok and len(loop) >= 3:
                out[b].append(loop)
    return out


def build_segments(P, Q, band, spacing, r_match=3.0, r_junc=3.0):
    """Union-curve seam segmentation with explicit junctions.

    1. Partner-label every boundary vertex by the nearest DIFFERENT-band boundary
       vertex within r_match·spacing (generous radius: on a curved surface the two
       sides of a seam are offset jagged staircases up to ~2 cells apart, so a
       tight radius spuriously labels corners -1 and tears them). Walk each band's
       boundary loop; split where the partner band flips -> one run per adjacent
       band, with the corner vertices at both ends.
    2. Cluster all run ENDPOINTS within r_junc·spacing. A cluster touching >=3
       bands is a JUNCTION (a cube-corner where >=3 grids meet); collapse it to
       one shared vertex (centroid).
    3. Pair runs into seams: run (a, partner=b) with run (b, partner=a) that share
       the same two junctions (closest endpoints). The two paired runs are the two
       offset sides of one seam.

    Returns (segments, junctions, jpos) where
      segments = [{a, b, va, vb, ja, jb, closed}]  (ja/jb = junction ids or -1)
      junctions = {junction_id: [member endpoint global vert ids]}
      jpos = {junction_id: 3-vector centroid position}
    Endpoints of each run are tagged with their junction id (run_junc) so the
    zipper can pin them to the shared junction vertex.
    """
    loops_by_band = _band_boundary_loops(Q, band)
    allb = sorted(set(int(v) for lps in loops_by_band.values() for lp in lps for v in lp))
    if not allb:
        return [], {}, {}
    allb_arr = np.array(allb)
    pos = P[allb_arr]
    tree = cKDTree(pos)
    bandof = {int(v): int(band[v]) for v in allb}

    def partner(v):
        d, idx = tree.query(P[v], k=min(30, len(allb)))
        for dd, jj in zip(np.atleast_1d(d), np.atleast_1d(idx)):
            w = int(allb_arr[jj])
            if bandof[w] != bandof[v] and dd <= r_match * spacing:
                return bandof[w]
        return -1

    # split each loop into constant-partner runs
    runs = []  # dict(a, b, verts, closed)
    for b, lps in loops_by_band.items():
        for lp in lps:
            m = len(lp)
            lab = [partner(v) for v in lp]
            flips = sorted(i for i in range(m) if lab[i] != lab[(i - 1) % m])
            if not flips:
                runs.append(dict(a=b, b=lab[0], verts=list(lp), closed=True))
                continue
            for s0, s1 in zip(flips, flips[1:] + [flips[0] + m]):
                idxs = [lp[k % m] for k in range(s0, s1 + 1)]  # both corners incl.
                runs.append(dict(a=b, b=lab[s0 % m], verts=idxs, closed=False))

    # ----- junctions: cluster run endpoints -----
    ep_list = []          # (run_index, end {0|1}, global vert id, band)
    for ri, r in enumerate(runs):
        if r["closed"]:
            continue
        ep_list.append((ri, 0, r["verts"][0], r["a"]))
        ep_list.append((ri, 1, r["verts"][-1], r["a"]))
    junctions, jpos, run_junc = {}, {}, {}  # run_junc[ri] = (j_end0, j_end1)
    if ep_list:
        epos = np.array([P[gv] for (_, _, gv, _) in ep_list])
        etree = cKDTree(epos)
        g = nx.Graph(); g.add_nodes_from(range(len(ep_list)))
        g.add_edges_from(etree.query_pairs(r_junc * spacing))
        jid = 0
        for comp in nx.connected_components(g):
            bands_here = set(ep_list[i][3] for i in comp)
            members = [ep_list[i][2] for i in comp]
            if len(bands_here) >= 3:
                junctions[jid] = members
                jpos[jid] = np.mean([P[v] for v in members], axis=0)
                for i in comp:
                    ri, end, _, _ = ep_list[i]
                    e0, e1 = run_junc.get(ri, (-1, -1))
                    run_junc[ri] = (jid, e1) if end == 0 else (e0, jid)
                jid += 1

    # ----- pair runs into seams sharing two junctions -----
    # Best (junction-sharing) matches first, so degenerate stubs don't grab a long
    # run away from its true (equal-junction) partner. Candidate score prefers:
    #   (1) identical junction SET, then (2) small midpoint distance + length
    # mismatch. Length mismatch is penalised because a |va|=2 stub paired with a
    # |vb|=31 seam crushes the long side when rank-mapped (-> tears at high res).
    segments = []
    used = [False] * len(runs)
    cand = []  # (priority, dist+lenpen, i, j)
    for i, ri in enumerate(runs):
        if ri["closed"]:
            continue
        ja_i, jb_i = run_junc.get(i, (-1, -1))
        ci_mid = P[ri["verts"]].mean(0)
        for j, rj in enumerate(runs):
            if j <= i or rj["closed"]:
                continue
            if rj["a"] != ri["b"] or rj["b"] != ri["a"]:
                continue
            ja_j, jb_j = run_junc.get(j, (-1, -1))
            same_set = ({ja_i, jb_i} == {ja_j, jb_j}) and -1 not in (ja_i, jb_i)
            d = np.linalg.norm(P[rj["verts"]].mean(0) - ci_mid)
            la, lb = len(ri["verts"]), len(rj["verts"])
            lenpen = spacing * abs(la - lb) / max(la, lb)
            cand.append((0 if same_set else 1, d + lenpen, i, j))
    cand.sort()
    for prio, score, i, j in cand:
        if used[i] or used[j]:
            continue
        used[i] = used[j] = True
        ja_i, jb_i = run_junc.get(i, (-1, -1))
        segments.append(dict(a=runs[i]["a"], b=runs[i]["b"],
                             va=runs[i]["verts"], vb=runs[j]["verts"],
                             ja=ja_i, jb=jb_i, closed=False))

    # ----- pair leftover CLOSED loops (no partner within r_match): the two
    # halves of a self-occluded ring whose surface dips beyond projection reach
    # (the torus inner equator). Each such full loop is matched to its nearest
    # opposite full loop and zipped as a closed seam. Only triggers on genus/
    # self-occluding shapes; convex shapes have no leftover closed loops. -----
    closed_runs = [i for i, r in enumerate(runs) if r["closed"] and not used[i]]
    for ci, i in enumerate(closed_runs):
        if used[i]:
            continue
        mi = P[runs[i]["verts"]].mean(0)
        best, bestd = -1, np.inf
        for j in closed_runs:
            if j == i or used[j] or runs[j]["a"] == runs[i]["a"]:
                continue
            d = np.linalg.norm(P[runs[j]["verts"]].mean(0) - mi)
            if d < bestd:
                best, bestd = j, d
        if best >= 0:
            used[i] = used[best] = True
            segments.append(dict(a=runs[i]["a"], b=runs[best]["a"],
                                 va=runs[i]["verts"], vb=runs[best]["verts"],
                                 ja=-1, jb=-1, closed=True))
    return segments, junctions, jpos


# --------------------------------------------------------------------------- #
#  Phase III-V — reconcile edge counts + zipper each 2-owner seam (§4-5)
# --------------------------------------------------------------------------- #
def zipper_segments(P, Q, segments, spacing, junctions=None, jpos=None,
                    surf_tree=None, surf_V=None):
    """Build ONE shared consensus curve per seam from the UNION of both sides,
    pinned junction->junction, and re-point BOTH sides' boundary vertices onto
    it. Run endpoints are pinned to the shared junction vertices so adjacent
    seams meet exactly. Returns (P, Q).

    This is co-extensive by construction: the curve spans the same two junction
    endpoints for both sides (no corner gaps) and is built from the union (covers
    the full span either offset staircase reaches). Both sides snap to identical
    seam vertices -> manifold shared edges -> watertight."""
    P = list(np.asarray(P, float))
    junctions = junctions or {}
    jpos = jpos or {}
    remap = {}            # old vert id -> shared (seam or junction) vert id

    def project(pt):
        if surf_tree is None:
            return pt
        return surf_V[surf_tree.query(pt)[1]]

    # one shared vertex per junction (collapse all members onto it)
    junc_vid = {}
    for jid, members in junctions.items():
        c = project(jpos[jid]) if surf_tree is not None else jpos[jid]
        vid = len(P); P.append(np.asarray(c, float)); junc_vid[jid] = vid
        for m in members:
            remap[int(m)] = vid

    new_quads = []
    for seg in segments:
        va, vb = list(seg["va"]), list(seg["vb"])
        ja, jb = seg.get("ja", -1), seg.get("jb", -1)
        if len(va) < 2 or len(vb) < 2:
            continue
        pa = P_arr(P, va)
        pb = P_arr(P, vb)
        # ----- CLOSED seam: two disjoint full loops (e.g. the torus inner-equator
        # rings) bridged by a quad STRIP (they must NOT collapse onto one curve —
        # that would pinch the tube). Resample both to a shared count, align, and
        # connect with nA quads. -----
        if seg.get("closed"):
            nA = max(4, min(len(va), len(vb)))
            ra = resample_polyline(pa, nA, closed=True)
            rb = resample_polyline(pb, nA, closed=True)
            # align rb's phase + winding to ra
            k = int(np.argmin(np.linalg.norm(rb - ra[0], axis=1)))
            rb = np.roll(rb, -k, axis=0)
            if nA > 2 and np.linalg.norm(rb[-1] - ra[1]) < np.linalg.norm(rb[1] - ra[1]):
                rb = np.vstack([rb[0], rb[1:][::-1]])
            ida = list(range(len(P), len(P) + nA)); P.extend(list(ra))
            idb = list(range(len(P), len(P) + nA)); P.extend(list(rb))
            for t in range(nA):
                u = (t + 1) % nA
                new_quads.append([ida[t], ida[u], idb[u], idb[t]])
            # re-point both original loops onto their resampled ring
            ta = cKDTree(ra); tb = cKDTree(rb)
            for v in va:
                remap[int(v)] = ida[int(ta.query(P[v])[1])]
            for v in vb:
                remap[int(v)] = idb[int(tb.query(P[v])[1])]
            continue
        # ----- orient both sides to run junction-ja -> junction-jb -----
        pin_s = jpos[ja] if ja in jpos else None
        pin_e = jpos[jb] if jb in jpos else None
        if pin_s is not None:
            if np.linalg.norm(pa[-1] - pin_s) < np.linalg.norm(pa[0] - pin_s):
                va, pa = va[::-1], pa[::-1]
            if np.linalg.norm(pb[-1] - pin_s) < np.linalg.norm(pb[0] - pin_s):
                vb, pb = vb[::-1], pb[::-1]
        else:
            # no junction: orient B to match A's direction
            d_keep = np.linalg.norm(pa[0] - pb[0]) + np.linalg.norm(pa[-1] - pb[-1])
            d_rev = np.linalg.norm(pa[0] - pb[-1]) + np.linalg.norm(pa[-1] - pb[0])
            if d_rev < d_keep:
                vb, pb = vb[::-1], pb[::-1]
        # ----- ONE consensus curve from the UNION, pinned junction->junction.
        # Built from both sides' interiors so it covers the full span and lies
        # between the two offset staircases. Resampled to nA points; nA is the
        # SMALLER side count so neither side has to skip a consensus vertex when
        # rank-mapped (skipping is what tears the seam). -----
        chain_pts = [pin_s] if pin_s is not None else [0.5 * (pa[0] + pb[0])]
        # interleave both sides' interior points by fractional arc position
        ta = np.linspace(0, 1, len(pa))
        tb = np.linspace(0, 1, len(pb))
        inter = sorted([(ta[i], pa[i]) for i in range(1, len(pa) - 1)] +
                       [(tb[i], pb[i]) for i in range(1, len(pb) - 1)],
                       key=lambda x: x[0])
        chain_pts += [p for _, p in inter]
        chain_pts.append(pin_e if pin_e is not None else 0.5 * (pa[-1] + pb[-1]))
        chain = np.asarray(chain_pts, float)
        nA = max(2, min(len(va), len(vb)))
        consensus = resample_polyline(chain, nA, closed=False)
        if surf_tree is not None:
            consensus = np.array([project(p) for p in consensus])
        # allocate consensus vertex ids; reuse junction vertices at the ends
        seam_ids = [None] * nA
        if ja in junc_vid:
            seam_ids[0] = junc_vid[ja]
        else:
            seam_ids[0] = len(P); P.append(consensus[0])
        for t in range(1, nA - 1):
            seam_ids[t] = len(P); P.append(consensus[t])
        if jb in junc_vid:
            seam_ids[nA - 1] = junc_vid[jb]
        else:
            seam_ids[nA - 1] = len(P); P.append(consensus[-1])
        # ----- RANK-map each side's vertices onto the consensus in order.
        # vertex k of a side (0..len-1) -> consensus round(k/(len-1)*(nA-1)).
        # Monotonic & non-skipping for both sides -> shared edges align 1:1. -----
        for vs_side, ps_side in ((va, pa), (vb, pb)):
            L = len(vs_side)
            for k, v in enumerate(vs_side):
                t = int(round(k / (L - 1) * (nA - 1)))
                remap[int(v)] = seam_ids[t]

    # apply remap (chase) to Q once
    def root(x):
        seen = 0
        while x in remap and remap[x] != x and seen < len(P) + 5:
            x = remap[x]; seen += 1
        return x
    Q = np.asarray(Q, np.int64)
    if new_quads:
        Q = np.vstack([Q, np.asarray(new_quads, np.int64)])
    full = np.arange(len(P))
    for a in remap:
        full[a] = root(a)
    Q = full[Q]
    return np.asarray(P, float), Q


def P_arr(P, ids):
    return np.array([P[i] for i in ids])


# --------------------------------------------------------------------------- #
#  Phase VI — N-grid junction resolution + pole insertion (§6, verified)
# --------------------------------------------------------------------------- #
def _walk_ring(sub):
    comp = list(sub.nodes())
    if any(deg != 2 for _, deg in sub.degree()):
        return []
    start = min(comp)
    loop, prev, cur = [start], None, start
    while True:
        nb = [w for w in sub.neighbors(cur) if w != prev]
        if not nb or nb[0] == start:
            break
        loop.append(nb[0]); prev, cur = cur, nb[0]
    return loop if len(loop) == len(comp) else []


def resolve_junctions(P, Q, project_fn=None, max_loop=24):
    """Close N-grid junction holes (§6) by CAPPING (never collapsing — collapsing
    a ring's verts onto one point degenerates the surrounding quads and tears
    them). n==3 -> one triangle; n==4 -> one quad; even -> centre-pole quad fan;
    odd -> quad fan + one closing triangle. The few tris sit only at the handful
    of >=3-grid corner poles."""
    Q = np.asarray(Q, np.int64)
    if not len(Q):
        return np.asarray(P, float), Q, np.zeros((0, 3), np.int64)
    P = list(np.asarray(P, float))
    bnd = boundary_edges_global(Q)
    if not bnd:
        return np.asarray(P, float), Q, np.zeros((0, 3), np.int64)
    g = nx.Graph(); g.add_edges_from(bnd)

    def centre(loop):
        c = np.mean([P[i] for i in loop], axis=0)
        return project_fn(c) if project_fn is not None else c

    newquads, newtris, vmerge = [], [], {}
    for comp in nx.connected_components(g):
        if not (3 <= len(comp) <= max_loop):
            continue
        loop = _walk_ring(g.subgraph(comp))
        if not loop:
            continue
        n = len(loop)
        if n == 3:
            newtris.append([loop[0], loop[1], loop[2]])
        elif n == 4:
            newquads.append([loop[0], loop[1], loop[2], loop[3]])
        elif n % 2 == 0:
            m = n // 2; c = len(P); P.append(centre(loop))
            for i in range(m):
                newquads.append([loop[(2*i) % n], loop[(2*i+1) % n], loop[(2*i+2) % n], c])
        else:
            m = (n - 1) // 2; c = len(P); P.append(centre(loop))
            for i in range(m):
                newquads.append([loop[2*i], loop[2*i+1], loop[2*i+2], c])
            newtris.append([loop[n-1], loop[0], c])

    def root(x):
        while x in vmerge and vmerge[x] != x:
            x = vmerge[x]
        return x
    P = np.asarray(P, float)
    remap = np.arange(len(P))
    for a in vmerge:
        remap[a] = root(a)
    Q = remap[Q]
    if newquads:
        Q = np.vstack([Q, remap[np.asarray(newquads, np.int64)]])
    Q = Q[[len(set(map(int, q))) == 4 for q in Q]]
    T = remap[np.asarray(newtris, np.int64)] if newtris else np.zeros((0, 3), np.int64)
    return P, Q, T


# --------------------------------------------------------------------------- #
#  Phase VII — compaction + validation
# --------------------------------------------------------------------------- #
def compact(P, Q, T):
    """Drop unused vertices, reindex. Drop degenerate and duplicate faces."""
    Q = np.asarray(Q, np.int64).reshape(-1, 4)
    T = np.asarray(T, np.int64).reshape(-1, 3) if T is not None and len(T) else np.zeros((0, 3), np.int64)
    Q = Q[[len(set(map(int, q))) == 4 for q in Q]]
    if len(T):
        T = T[[len(set(map(int, t))) == 3 for t in T]]
    # drop duplicate faces (same vertex set) — a junction ring can be capped twice
    if len(Q):
        seenq, keepq = set(), []
        for q in Q:
            k = tuple(sorted(map(int, q)))
            if k not in seenq:
                seenq.add(k); keepq.append(q)
        Q = np.asarray(keepq, np.int64).reshape(-1, 4)
    else:
        seenq = set()
    if len(T):
        seent, keept = set(), []
        for t in T:
            k = tuple(sorted(map(int, t)))
            # drop dup tris and tris coincident with a quad's vertex subset
            if k in seent:
                continue
            seent.add(k); keept.append(t)
        T = np.asarray(keept, np.int64).reshape(-1, 3) if keept else np.zeros((0, 3), np.int64)
    used = np.unique(np.concatenate([Q.ravel(), T.ravel()])) if len(T) else np.unique(Q)
    if not len(used):
        return np.asarray(P, float), Q, T
    old2new = -np.ones(len(P), np.int64)
    old2new[used] = np.arange(len(used))
    return np.asarray(P)[used], old2new[Q], (old2new[T] if len(T) else T)


def _resolve_folds(P, Q):
    """Repair quad FOLDS: two quads that share a triangle (3 of 4 verts) overlap
    on that triangle -> a non-manifold double-cover (the bench's quad-splitting
    turns the shared triangle into a duplicate face). This happens where two seam
    strips both reach into a >=3-band junction. Replace each such pair with ONE
    quad through the 4 outer vertices (the two odd verts + the two shared verts
    each odd vert is adjacent to); the redundant third shared vert is dropped."""
    Q = np.asarray(Q, np.int64).reshape(-1, 4)
    if not len(Q):
        return Q
    import itertools
    tri2q = collections.defaultdict(list)
    for qi, q in enumerate(Q):
        if len(set(map(int, q))) != 4:
            continue
        for c in itertools.combinations(sorted(map(int, q)), 3):
            tri2q[c].append(qi)
    dead = set()
    newq = []
    for tri, qs in tri2q.items():
        if len(qs) != 2:
            continue
        qa, qb = qs
        if qa in dead or qb in dead:
            continue
        A, B = [int(x) for x in Q[qa]], [int(x) for x in Q[qb]]
        S = set(tri)
        oa = [v for v in A if v not in S]
        ob = [v for v in B if v not in S]
        if len(oa) != 1 or len(ob) != 1:
            continue
        oa, ob = oa[0], ob[0]
        # the two shared verts adjacent (cyclically) to the odd vert in each quad
        def neigh(quad, odd):
            i = quad.index(odd)
            return {quad[(i - 1) % 4], quad[(i + 1) % 4]}
        keep = neigh(A, oa) & neigh(B, ob)   # shared verts on the union boundary
        if len(keep) != 2:
            continue
        x, y = list(keep)
        dead.add(qa); dead.add(qb)
        newq.append([oa, x, ob, y])
    if not dead:
        return Q
    Q2 = [q for qi, q in enumerate(Q) if qi not in dead]
    Q2 = (np.asarray(Q2, np.int64).reshape(-1, 4) if Q2 else np.zeros((0, 4), np.int64))
    if newq:
        Q2 = np.vstack([Q2, np.asarray(newq, np.int64)])
    Q2 = Q2[[len(set(map(int, q))) == 4 for q in Q2]]
    return Q2


def _repair_nonmanifold(P, Q, T):
    """Final safety net (paper §7.3): an edge shared by >2 faces is a fold/double
    cover. Greedily drop the offending face with the smallest area until every
    edge has <=2 incident faces. Removes the few residual non-manifold edges left
    on noisy real scans without touching the clean interior."""
    P = np.asarray(P, float)
    Q = np.asarray(Q, np.int64).reshape(-1, 4)
    T = np.asarray(T, np.int64).reshape(-1, 3) if T is not None and len(T) else np.zeros((0, 3), np.int64)

    def face_area(idx):
        if idx < len(Q):
            q = Q[idx]
            return 0.5 * (np.linalg.norm(np.cross(P[q[1]] - P[q[0]], P[q[2]] - P[q[0]]))
                          + np.linalg.norm(np.cross(P[q[2]] - P[q[0]], P[q[3]] - P[q[0]])))
        t = T[idx - len(Q)]
        return 0.5 * np.linalg.norm(np.cross(P[t[1]] - P[t[0]], P[t[2]] - P[t[0]]))

    for _ in range(8):
        edge2f = collections.defaultdict(list)
        for fi, q in enumerate(Q):
            for a, b in ((q[0], q[1]), (q[1], q[2]), (q[2], q[3]), (q[3], q[0])):
                edge2f[tuple(sorted((int(a), int(b))))].append(fi)
        for ti, t in enumerate(T):
            for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                edge2f[tuple(sorted((int(a), int(b))))].append(len(Q) + ti)
        bad = {f for fs in edge2f.values() if len(fs) > 2 for f in fs}
        if not bad:
            break
        # drop the smallest-area face among those on over-used edges
        drop = min(bad, key=face_area)
        if drop < len(Q):
            Q = np.delete(Q, drop, axis=0)
        else:
            T = np.delete(T, drop - len(Q), axis=0)
    return P, Q, T


# --------------------------------------------------------------------------- #
#  Full seam-stitch driver
# --------------------------------------------------------------------------- #
def stitch(P, Q, band, spacing, surf_mesh=None, feature_angle=40.0):
    """Take the per-band projection soup (P, Q, band) and produce a watertight
    (V, quads, tris) by paper Phases II-VII."""
    surf_tree = surf_V = None
    if surf_mesh is not None:
        surf_V = np.asarray(surf_mesh.vertices)
        surf_tree = cKDTree(surf_V)
    segments, junctions, jpos = build_segments(P, Q, band, spacing)
    # NOTE: do NOT vertex-snap the consensus during zippering — snapping to the
    # nearest ORIGINAL vertex collapses distinct seam samples onto one coarse
    # input vertex and tears the seam. The union consensus already rides the
    # surface closely; surface projection (if wanted) is a later smoothing pass.
    P, Q = zipper_segments(P, Q, segments, spacing, junctions=junctions,
                           jpos=jpos, surf_tree=None, surf_V=None)
    # drop quads that collapsed during re-pointing
    Q = Q[[len(set(map(int, q))) == 4 for q in Q]]
    # weld coincidences created by the seam re-pointing (a small fraction of a
    # cell — the two re-pointed sides land on the same shared seam vert exactly,
    # but float roundoff can split them).
    P, Q = _weld_exact(P, Q, 1e-4 * spacing)
    # repair quad folds (two quads sharing a triangle) at junctions before capping
    Q = _resolve_folds(P, Q)
    # GRID-FILL balanced even holes with a clean quad loft (Blender Grid Fill):
    # density-matched, reuses the loop's own border verts so it welds 1:1. Holes
    # that can't be cleanly gridded (odd / unbalanced / tiny) fall to junctions.
    P, Q = fill_holes(P, Q, spacing)
    P, Q, T = resolve_junctions(P, Q, project_fn=None)
    P, Q, T = _repair_nonmanifold(P, Q, T)
    P, Q, T = _unify_winding(P, Q, T)
    # Curvature-adaptive coarsening is DISABLED: it removes the edge-loop verts the
    # feature-lock needs (collapsing flat blocks across sharp edges), producing
    # folds and wobbly creases. Feature-aware coarsening (keep crease edge loops,
    # coarsen only flat interiors) is the proper version — a separate task.
    # FEATURE LOCK: pin (don't move) the grid verts already lying ON a sharp
    # crease, so smoothing doesn't round them — but do NOT yank distant verts onto
    # creases (that folds the grid where flow crosses the edge diagonally). Crisp
    # edges that require an actual edge LOOP along the crease need feature-aligned
    # projection, not post-snap (a separate task).
    pinned = set()
    if surf_mesh is not None:
        pinned = _pin_existing_crease_verts(P, surf_mesh, feature_angle, spacing)
    P, Q, T = smooth_quads(P, Q, T, surf_mesh=surf_mesh, iters=4, w=0.4,
                           pinned=pinned)
    P, Q, T = compact(P, Q, T)
    return P, Q, T


def _pin_existing_crease_verts(P, surf_mesh, feature_angle, spacing):
    """Pin only verts that ALREADY sit essentially on a sharp crease (within a
    small fraction of a cell). Pinning keeps them from being smoothed off the
    edge; it does NOT move anything (so it can't fold the grid)."""
    P = np.asarray(P, float)
    cr = _crease_segments(surf_mesh, feature_angle)
    if cr is None or not len(cr):
        return set()
    a, b = cr[:, 0], cr[:, 1]
    pinned = set()
    tol = 0.2 * (spacing or 1.0)
    for vi in range(len(P)):
        _, dist = _closest_on_segments(P[vi], a, b)
        if dist < tol:
            pinned.add(vi)
    return pinned


def _crease_segments(mesh, feature_angle):
    """Sharp crease edges of the mesh as (M,2,3) endpoint pairs."""
    try:
        ang = mesh.face_adjacency_angles
        edges = mesh.face_adjacency_edges[ang > np.radians(feature_angle)]
        if not len(edges):
            return None
        return mesh.vertices[edges]            # (M,2,3)
    except Exception:  # noqa: BLE001
        return None


def _closest_on_segments(p, a, b):
    """Closest point on a set of segments a->b to point p. Returns (point, dist)."""
    ab = b - a
    denom = np.einsum("ij,ij->i", ab, ab) + 1e-12
    t = np.clip(np.einsum("ij,ij->i", p - a, ab) / denom, 0, 1)
    proj = a + t[:, None] * ab
    d = np.linalg.norm(proj - p, axis=1)
    k = int(np.argmin(d))
    return proj[k], d[k]


def smooth_quads(P, Q, T, surf_mesh=None, iters=5, w=0.5, pinned=None):
    """Quad-aware Laplacian smoothing: pull each interior vertex toward the
    average of its quad-edge neighbours so rows space evenly (hand-made look),
    then re-project onto the original surface so the shape is preserved. Boundary
    vertices (open edges) are pinned. This is the relaxation that turns a valid-
    but-uneven grid into clean, uniform flow."""
    P = np.asarray(P, float).copy()
    Q = np.asarray(Q, np.int64).reshape(-1, 4)
    T = np.asarray(T, np.int64).reshape(-1, 3) if T is not None and len(T) else np.zeros((0, 3), np.int64)
    if not len(Q):
        return P, Q, T
    nbr = collections.defaultdict(set)
    for q in Q:
        for k in range(4):
            nbr[int(q[k])].add(int(q[(k + 1) % 4]))
            nbr[int(q[k])].add(int(q[(k - 1) % 4]))
    for t in T:
        for k in range(3):
            nbr[int(t[k])].add(int(t[(k + 1) % 3]))
            nbr[int(t[k])].add(int(t[(k - 1) % 3]))
    # pin boundary verts AND feature-snapped crease verts (so sharp edges that
    # were locked onto the crease lines stay crisp instead of relaxing back into
    # a wobble).
    bnd = set()
    for a, b in boundary_edges_global(Q):
        bnd.add(a); bnd.add(b)
    if pinned:
        bnd |= set(int(v) for v in pinned)
    moveable = np.array([v for v in nbr if v not in bnd])
    if not len(moveable):
        return P, Q, T
    import trimesh as _tm
    for _ in range(iters):
        newP = P.copy()
        for v in moveable:
            ns = list(nbr[v])
            if ns:
                newP[v] = P[v] * (1 - w) + P[ns].mean(0) * w
        if surf_mesh is not None:
            # re-project moved verts onto the true surface (accurate closest point)
            cp, _, _ = _tm.proximity.closest_point(surf_mesh, newP[moveable])
            newP[moveable] = cp
        P = newP
    return P, Q, T


def fill_holes(P, Q, spacing, max_loop=40):
    """Grid-fill every clean even/balanced boundary loop with a quad loft. Leaves
    odd/unbalanced/large loops for the junction-pole pass. Returns (P, Q)."""
    P = np.asarray(P, float)
    Q = np.asarray(Q, np.int64).reshape(-1, 4)
    for _ in range(6):  # holes may reveal new ones after a fill
        bnd = boundary_edges_global(Q)
        if not bnd:
            break
        g = nx.Graph()
        g.add_edges_from(bnd)
        filled_any = False
        for comp in list(nx.connected_components(g)):
            if not (4 <= len(comp) <= max_loop):
                continue
            loop = _walk_ring(g.subgraph(comp))
            if not loop or len(loop) % 2 != 0:
                continue  # odd loop can't be pure-quad gridded
            newpts, nq = grid_fill(P, loop, spacing)
            if nq is None:
                continue
            if len(newpts):
                P = np.vstack([P, newpts])
            Q = np.vstack([Q, nq])
            filled_any = True
        if not filled_any:
            break
    return P, Q


def _four_corners(pts):
    """Pick 4 corner indices on a closed loop = the 4 sharpest turns. Returns
    sorted indices into pts."""
    n = len(pts)
    ang = np.zeros(n)
    for i in range(n):
        a = pts[(i - 1) % n] - pts[i]
        b = pts[(i + 1) % n] - pts[i]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            ang[i] = np.pi
            continue
        ang[i] = np.arccos(np.clip(a @ b / (na * nb), -1, 1))  # small angle = sharp
    # sharpest = smallest interior angle; spread them out (non-adjacent)
    order = np.argsort(ang)
    corners = []
    for idx in order:
        if all(min((idx - c) % n, (c - idx) % n) >= 2 for c in corners):
            corners.append(int(idx))
        if len(corners) == 4:
            break
    while len(corners) < 4:  # fallback: evenly spaced
        corners = [int(round(k * n / 4)) % n for k in range(4)]
        break
    return sorted(corners)


def _coons(b, t, l, r):
    """Coons bilinear blend from 4 boundary curves. b,t length nu+1; l,r length
    nv+1. Returns (nu+1, nv+1, 3) with boundary == the curves exactly."""
    nu = len(b) - 1
    nv = len(l) - 1
    S = np.zeros((nu + 1, nv + 1, 3))
    for i in range(nu + 1):
        u = i / nu
        for j in range(nv + 1):
            v = j / nv
            Lc = (1 - u) * l[j] + u * r[j]
            Ld = (1 - v) * b[i] + v * t[i]
            B = ((1 - u) * (1 - v) * b[0] + (1 - u) * v * t[0] +
                 u * (1 - v) * b[nu] + u * v * t[nu])
            S[i, j] = Lc + Ld - B
    return S


def grid_fill(P, loop, spacing):
    """Fill a closed boundary loop with a clean quad grid (Blender 'Grid Fill' /
    transfinite Coons). Density is matched to `spacing` so the new quads are the
    same size as the surrounding mesh; the loop's OWN vertices are reused on the
    boundary so the patch welds 1:1 with no new seam.

    Returns (new_points, quads) where quads index into P extended by new_points
    (interior only). Boundary quads reference existing loop vertex ids directly.
    Returns (None, None) if the loop can't be 4-sided gridded.
    """
    loop = list(loop)
    n = len(loop)
    if n < 4:
        return None, None
    pts = P[loop]
    c = _four_corners(pts)
    # 4 sides between consecutive corners (as loop-index runs, inclusive ends)
    sides = []
    for k in range(4):
        a, b = c[k], c[(k + 1) % 4]
        idx = [loop[(a + s) % n] for s in range(((b - a) % n) + 1)]
        sides.append(idx)
    s0, s1, s2, s3 = sides
    # opposite sides must have EQUAL vertex counts to grid cleanly and reuse the
    # loop's own border verts (so the patch welds 1:1, no new seam). Only fill
    # when the loop already balances (len(s0)==len(s2) and len(s1)==len(s3));
    # otherwise the seam reconciliation should have evened it — skip (caller caps).
    if len(s0) != len(s2) or len(s1) != len(s3):
        return None, None
    nu = len(s0) - 1
    nv = len(s1) - 1
    if nu < 1 or nv < 1:
        return None, None
    # border vertex-id grid, reusing the EXISTING loop verts on all 4 sides
    gid = -np.ones((nu + 1, nv + 1), np.int64)
    for i in range(nu + 1):
        gid[i, 0] = s0[i]            # bottom
        gid[i, nv] = s2[::-1][i]     # top (reversed so it runs same way as bottom)
    for j in range(nv + 1):
        gid[0, j] = s3[::-1][j]      # left
        gid[nu, j] = s1[j]           # right
    # Coons interior from the actual border positions
    bcur = P[[gid[i, 0] for i in range(nu + 1)]]
    tcur = P[[gid[i, nv] for i in range(nu + 1)]]
    lcur = P[[gid[0, j] for j in range(nv + 1)]]
    rcur = P[[gid[nu, j] for j in range(nv + 1)]]
    S = _coons(bcur, tcur, lcur, rcur)
    newpts = []
    base = len(P)
    for i in range(1, nu):
        for j in range(1, nv):
            gid[i, j] = base + len(newpts)
            newpts.append(S[i, j])
    quads = []
    for i in range(nu):
        for j in range(nv):
            quads.append([gid[i, j], gid[i + 1, j], gid[i + 1, j + 1], gid[i, j + 1]])
    npts = np.asarray(newpts) if newpts else np.zeros((0, 3))
    return npts, np.asarray(quads, np.int64)


def _unify_winding(P, Q, T):
    """Make every face wind consistently and point OUTWARD. The 6 cage-direction
    grids are wound per their own camera, so neighbouring patches disagree; the
    viewer then renders inward-facing backfaces (dark 'holes'). BFS-propagate a
    consistent orientation across shared edges, then flip the whole mesh if the
    majority points inward (signed-volume test)."""
    P = np.asarray(P, float)
    Q = np.asarray(Q, np.int64).reshape(-1, 4)
    T = np.asarray(T, np.int64).reshape(-1, 3) if T is not None and len(T) else np.zeros((0, 3), np.int64)
    faces = [list(map(int, q)) for q in Q] + [list(map(int, t)) for t in T]
    nq = len(Q)
    if not faces:
        return P, Q, T
    # directed-edge -> face map for adjacency + orientation agreement
    de = collections.defaultdict(list)   # (a,b) -> list of face idx using a->b
    for fi, f in enumerate(faces):
        for k in range(len(f)):
            de[(f[k], f[(k + 1) % len(f)])].append(fi)
    adj = collections.defaultdict(list)  # face -> [(nbr, same_dir?)]
    for fi, f in enumerate(faces):
        for k in range(len(f)):
            a, b = f[k], f[(k + 1) % len(f)]
            for fj in de.get((b, a), []):      # opposite dir = consistent
                if fj != fi:
                    adj[fi].append((fj, True))
            for fj in de.get((a, b), []):      # same dir = inconsistent
                if fj != fi:
                    adj[fi].append((fj, False))
    flip = [None] * len(faces)
    for seed in range(len(faces)):
        if flip[seed] is not None:
            continue
        flip[seed] = False
        stack = [seed]
        while stack:
            u = stack.pop()
            for v, same in adj[u]:
                want = flip[u] if same else (not flip[u])
                if flip[v] is None:
                    flip[v] = want
                    stack.append(v)
    def do_flip(f, fl):
        return f[::-1] if fl else f
    faces2 = [do_flip(f, flip[i]) for i, f in enumerate(faces)]
    Q2 = np.array([faces2[i] for i in range(nq)], np.int64).reshape(-1, 4) if nq else Q
    T2 = np.array([faces2[i] for i in range(nq, len(faces2))], np.int64).reshape(-1, 3) if len(T) else T
    # Final orientation authority: trimesh.fix_normals repairs winding correctly
    # even for genus-1 (torus), where a global signed-volume flip can't fix one
    # internally-consistent component wound opposite the other. Build a triangle
    # proxy, let trimesh reorient it, and map the per-face flip back to the quads.
    import trimesh as _tm
    qtri = np.vstack([Q2[:, [0, 1, 2]], Q2[:, [0, 2, 3]]]) if len(Q2) else np.zeros((0, 3), np.int64)
    tri = np.vstack([qtri, T2]) if len(T2) else qtri
    if len(tri):
        try:
            tm = _tm.Trimesh(P, tri, process=False)
            before = tm.faces.copy()
            tm.fix_normals()                     # consistent + outward, per component
            after = tm.faces
            flipped = ~np.all(before == after, axis=1)
            nq2 = len(Q2)
            # a quad is flipped if BOTH its triangles flipped (consistent decision)
            if len(Q2):
                qflip = flipped[:nq2] & flipped[nq2:2 * nq2]
                Q2[qflip] = Q2[qflip][:, ::-1]
            if len(T2):
                tflip = flipped[2 * nq2:]
                Q2  # noqa
                T2[tflip] = T2[tflip][:, ::-1]
        except Exception:  # noqa: BLE001
            v0, v1, v2 = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
            if np.sum(np.cross(v0, v1) * v2) / 6.0 < 0:
                Q2 = Q2[:, ::-1] if len(Q2) else Q2
                T2 = T2[:, ::-1] if len(T2) else T2
    return P, Q2, T2


def _weld_exact(P, Q, eps):
    """Merge vertices within eps (cleans up exact duplicates from re-pointing)."""
    P = np.asarray(P, float)
    if not len(P):
        return P, Q
    tree = cKDTree(P)
    parent = np.arange(len(P))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, j in tree.query_pairs(eps):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    roots = np.array([find(i) for i in range(len(P))])
    uniq, inv = np.unique(roots, return_inverse=True)
    newP = np.zeros((len(uniq), 3)); cnt = np.zeros(len(uniq))
    np.add.at(newP, inv, P); np.add.at(cnt, inv, 1)
    newP /= cnt[:, None]
    Q2 = inv[np.asarray(Q, np.int64)]
    Q2 = Q2[[len(set(map(int, q))) == 4 for q in Q2]]
    return newP, Q2
