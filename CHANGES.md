# Changes — quad engines, NeurCross fix, shrinkwrap, audit cleanup

Summary of what was added/changed since the last commit. Delete this file before
committing if you don't want it in history — it's a handoff note, not docs.

## New features

- **Pure-quad remeshing (QUAD mode)** via **QuadWild-BiMDF** — now the default quad
  engine. 100% quad, watertight, feature-aligned. Segfault-tolerant wrapper (judges
  success by output file, not exit code).
- **NeurCross neural engine** (optional, GPU) — neural cross-field steering for organic
  shapes. Selectable in the Engine dropdown; falls back to QuadWild if unavailable.
- **Shrinkwrap postprocess** — project the remesh onto the original surface
  (nearest / project-along-normals), with max-distance and inflation-offset controls,
  a translucent ghost preview, and re-run/reset.
- **Quad-preserving OBJ export** + Export moved to its own section at the panel bottom.
- **Editable target-quads field** (type any count) and slider up to 100k.
- **Features view** and **curvature heatmap** using modern operators.

## Important fix

- **NeurCross was silently broken and is now fixed.** The original `.rosy` "bridge"
  never actually steered QuadWild — every NeurCross run was vanilla QuadWild. Root
  cause: `quadwild ... 2` recomputes its field internally and overwrites any `.rosy`.
  Fixed by using QuadWild's real (undocumented) field-import path: remesh (mode 1),
  write the neural field per-face, re-run with `do_remesh 0` and the `.rosy` passed as
  a CLI arg so it imports the field. Verified the field now changes the output.

## Audit cleanup

- Repo **614M → 53M**: removed dead `_libqex` / `_openmesh*` C++ builds (libQEx unused),
  cleared the `_tmp` upload cache, removed bytecode.
- `.gitignore` now excludes the vendored binary dirs (`_quadwild`, `_neurcross`, etc.).
- Fixed real bugs: upload temp-file leak + unbounded `SESSIONS` (now FIFO-capped);
  `AdaptiveRemesher` discarded its `F` (now keeps it + public `smooth()`); added logging
  to swallow-all excepts (those hid the NeurCross bug).
- Deleted ~390 lines of verified zero-caller dead code across the in-house engines.
- Fixed stale docstrings that described removed code (pyvista/VTK, Gaussian-curvature).
- `requirements.txt` corrected (dropped pyvista; added libigl, robust-laplacian, networkx).
- README rewritten to match the actual app (was describing the old VTK/QEM-only pipeline).

## New files

`quadwild.py`, `neurcross.py`, `postprocess.py`, `features.py`, `coarsen.py`,
`humanlogic.py`, `seam_stitch.py`, `shrinkwrap.py`, `visibility_shells.py`

## Not committed (gitignored)

`_quadwild/` (GPL-3 binary), `_neurcross/` (AGPL-3 repo), `_tmp/`, `__pycache__/`.

A fresh clone won't have these. **`setup.sh`** installs deps and checks for the
engines, printing where to download each if missing (also documented in the README
Quick Start). QUAD mode needs `_quadwild/`; TRIS mode works without it.
