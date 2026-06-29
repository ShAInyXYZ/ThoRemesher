"""
Post-process operations on a remeshed result. Currently: shrinkwrap.

These run AFTER remeshing, on demand — the user enables them in the Postprocess
panel. They mutate vertex positions only (topology/quads are preserved).
"""
from __future__ import annotations

import numpy as np
import trimesh


def shrinkwrap(rem_v, rem_f, tgt_v, tgt_f, mode="nearest",
               distance=0.0, offset=0.0):
    """Project the remeshed vertices onto the original (target) surface.

    mode      : "nearest"  -> closest point anywhere on the target surface
                "project"  -> move along each vertex's normal onto the target
    distance  : max search distance; verts farther than this from the target
                stay put (0 = unlimited). In target-bbox-diagonal fractions is
                NOT used — this is absolute model units.
    offset    : keep the result this far ABOVE the surface along the target
                normal (model units). Used for the inflation knob.

    Topology is untouched — only rem_v moves. Returns new (N,3) vertices.
    """
    rem_v = np.asarray(rem_v, np.float64)
    tgt = trimesh.Trimesh(np.asarray(tgt_v, np.float64),
                          np.asarray(tgt_f, np.int64), process=False)

    if mode == "project":
        src = trimesh.Trimesh(rem_v, np.asarray(rem_f, np.int64), process=False)
        pts, hit_dist, normals = _project_along_normals(src, tgt, distance)
    else:  # nearest
        pts, _fid, normals = _nearest_on_surface(tgt, rem_v)
        hit_dist = np.linalg.norm(pts - rem_v, axis=1)

    out = rem_v.copy()
    moved = np.ones(len(rem_v), bool) if distance <= 0 else (hit_dist <= distance)
    out[moved] = pts[moved]
    if offset:
        out[moved] += offset * normals[moved]
    return out


def _nearest_on_surface(tgt, pts):
    """Closest point on the target surface for each pt + the target's face normal there."""
    closest, _dist, fid = trimesh.proximity.closest_point(tgt, pts)
    normals = tgt.face_normals[fid]
    return closest, fid, normals


def _project_along_normals(src, tgt, distance):
    """Cast each src vertex along its (outward) vertex normal onto the target.
    Falls back to nearest-surface for rays that miss."""
    origins = src.vertices
    dirs = src.vertex_normals
    # ray both ways (the target may be inside or outside) — pick the nearer hit
    locs, idx_ray, idx_tri = tgt.ray.intersects_location(
        np.vstack([origins, origins]),
        np.vstack([dirs, -dirs]), multiple_hits=False)
    pts = origins.copy()
    normals = np.tile([0.0, 0.0, 1.0], (len(origins), 1))
    best = np.full(len(origins), np.inf)
    for loc, ri, ti in zip(locs, idx_ray, idx_tri):
        vi = ri % len(origins)
        d = np.linalg.norm(loc - origins[vi])
        if d < best[vi]:
            best[vi] = d
            pts[vi] = loc
            normals[vi] = tgt.face_normals[ti]
    # rays that missed entirely -> nearest-surface fallback
    miss = ~np.isfinite(best)
    if miss.any():
        cp, _f, nrm = _nearest_on_surface(tgt, origins[miss])
        pts[miss] = cp
        normals[miss] = nrm
        best[miss] = np.linalg.norm(cp - origins[miss], axis=1)
    return pts, best, normals
