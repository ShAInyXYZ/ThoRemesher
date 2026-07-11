<p align="center">
  <img src="images/logo.svg" alt="ThoRemesher" width="100" />
</p>

<h1 align="center">ThoRemesher</h1>

<p align="center">
  <em>Curvature-aware tri & pure-quad remeshing for 3D models</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/ThoRemesher-curvature--aware-4cc2ff?style=for-the-badge" alt="ThoRemesher" />
  <img src="https://img.shields.io/badge/python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/three.js-viewer-000000?style=for-the-badge&logo=three.js&logoColor=white" alt="Three.js" />
</p>

---

A local web app that takes any 3D model (FBX / GLB / GLTF / OBJ / PLY / STL) and remeshes it — **triangles** (curvature-adaptive, dense where it matters, coarse on flats) or **pure quads** (feature-aligned, watertight) — in a live, camera-linked **before / after** split view.

---

## What It Does

- **TRIS mode** — curvature-adaptive triangle remesh: dense topology where the surface curves, coarse on flats, sharp creases preserved.
- **QUAD mode** — 100% pure-quad output via **QuadWild-BiMDF**, with an optional **NeurCross** neural cross-field engine for organic shapes.
- **Curvature heatmap** — blue (flat) → red (detailed), so you can see what the algorithm perceives.
- **Features view** — regions classified developable / doubly-curved, crease lines, flow direction.
- **Shrinkwrap postprocess** — project the remesh back onto the original surface.
- **Dual camera-linked viewers**, presets, drag-and-drop, and **quad-preserving OBJ export**.

## Quad Engines

| Engine | What it is | Speed | Best for |
|---|---|---|---|
| **QuadWild-BiMDF** (default) | Feature-line-driven pure-quad remesher (CGG Bern, TOG 2023). Bi-MDF / libSatsuma flow solver. | ~2–3 s | Everything; hard-surface |
| **AutoRemesher** (optional) | Parameterization-first pure-quad remesher (Geogram `FrameField` + `quad_cover`, the QuadCover lineage). Curvature-**adaptive** quad density — dense where it curves, relaxed on flats. MIT. | ~1–5 s | Shapes with a detail gradient (small features that QuadWild's uniform density starves) |
| **NeurCross** (optional) | Neural cross-field steering (SIGGRAPH 2025). A per-shape network computes the field; QuadWild extracts quads guided by it. | ~1–2 min, needs a CUDA GPU | Organic shapes |

> **Why AutoRemesher:** QuadWild uses a single global `scaleFact`, so small/detailed regions get the same quad density as large flat ones — detail can be lost. AutoRemesher computes a per-face curvature scaling field and bakes it into a global integer-grid UV, so quads are *denser where the surface curves and relaxed on flats*. The **Density adaptivity** slider (0 = uniform, 1 = fully adaptive) controls the gradient. It's a different *school* of remeshing (parameterization-first vs field-tracing), complementary to QuadWild — try both.

> **NeurCross integration note:** QuadWild computes its own cross field internally and does not document an external-field input. ThoRemesher drives it through the undocumented `.rosy` import path — run mode 1 to remesh, write the neural field per-face, then re-run with `do_remesh 0` and the `.rosy` passed as a CLI argument so QuadWild *imports* the field instead of recomputing it. See `quadwild.py` / `neurcross.py`.

## Quick Start

```bash
./setup.sh   # installs deps + checks the vendored quad engines
./run.sh
# then open http://127.0.0.1:8000
```

Drop a `.fbx` / `.glb` / `.obj` … into the window, pick **TRIS** or **QUAD**, press **Remesh**.

> ⚠️ **The quad engines are NOT in the repo.** They are large third-party
> binaries/repos, gitignored, so a fresh clone **will not have them** — QUAD mode
> needs them fetched separately (TRIS mode works without). `./setup.sh` tells you
> exactly what's missing and where to get it:
>
> - **`_quadwild/`** — QuadWild-BiMDF prebuilt Linux binary + `config/` —
>   [cgg-bern/quadwild-bimdf](https://github.com/cgg-bern/quadwild-bimdf) (GPL-3).
>   Required for QUAD mode. Must contain `_quadwild/quadwild`,
>   `_quadwild/quad_from_patches`, and `_quadwild/config/…`.
> - **`_neurcross/`** — NeurCross repo, only for the neural engine —
>   [QiujieDong/NeurCross](https://github.com/QiujieDong/NeurCross) (AGPL-3).
>   Needs a CUDA GPU + `torch`. Optional; the engine falls back to QuadWild without it.
> - **`_autoremesher/`** — AutoRemesher source + pybind11 build, only for the
>   adaptive engine — [huxingyi/autoremesher](https://github.com/huxingyi/autoremesher) (MIT).
>   Build deps: `cmake`, `libtbb-dev`, `pybind11`. Build with
>   `./_autoremesher/clone_source.sh && ./_autoremesher/build.sh`. Optional; the
>   engine falls back to QuadWild without it.

## Requirements

- Python 3.10+
- A WebGL-capable browser
- (QUAD mode) the `_quadwild/` binaries on Linux
- (AutoRemesher) `cmake` + `libtbb-dev` + `pybind11` to build the extension
- (NeurCross) a CUDA GPU + `torch`

<details>
<summary><strong>Dependencies</strong></summary>

```bash
pip install -r requirements.txt
```

| Library | Role |
|---|---|
| **pymeshlab** | FBX/GLB/OBJ/PLY/STL I/O, decimation, quad-OBJ export, cleanup |
| **pyassimp** (libassimp) | robust FBX reading |
| **libigl** | principal-curvature region classification |
| **robust-laplacian** | robust mean curvature for the heatmap (Sharp & Crane 2020) |
| **trimesh / scipy / numpy / networkx** | mesh utilities, KD-tree transfer, GLB serialization |
| **fastapi / uvicorn** | backend server |

> `pyassimp` needs `libassimp` system-wide. Debian/Ubuntu: `sudo apt install libassimp-dev`.

</details>

## How It Works

**Curvature** uses two modern operators (no VTK):

- **Heatmap** (`detail_score`): robust mean curvature `H = |L·V| / 2A` from the robust cotan Laplacian (Sharp & Crane 2020), maxed with a dihedral-angle crease term. Stable on coarse/noisy meshes where discrete Gaussian curvature speckles.
- **Region classification** (`features.py`): principal-curvature ratio `|k_min| / |k_max|` to separate developable (≈0) from doubly-curved (≈1) surfaces.

**QUAD** runs QuadWild-BiMDF (two CLI steps: remesh+field, then Bi-MDF flow quadrangulation). The prebuilt binary segfaults on cleanup *after* writing its output, so success is judged by the output file existing, never the exit code.

**Shrinkwrap** (postprocess) projects remeshed vertices onto the original surface — nearest-surface or normal-projection, with a max search distance and an inflation offset. Topology is preserved; it's re-runnable and resettable.

## Controls

**QUAD mode**

| Control | Meaning |
|---|---|
| Engine | QuadWild-BiMDF (fast) · AutoRemesher (adaptive density) · NeurCross (neural, slow, GPU) |
| Target quads | approximate output quad count (type any value, 2.5k–100k slider) |
| Density adaptivity | 0 uniform → 1 fully curvature-adaptive (AutoRemesher only) |
| Sharp edges | auto-detect / hard (keep creases) / smooth |
| Sharp angle | dihedral angle above which an edge counts as sharp |

**Postprocess → Shrinkwrap**

| Control | Meaning |
|---|---|
| Type | nearest surface · project along normals |
| Max distance | verts farther than this from the target stay put (0 = unlimited) |
| Offset | inflation above the surface (0% = on surface, 5% max) |

## Export

| Format | Topology |
|---|---|
| **OBJ** | **keeps quads** — use this for quad meshes |
| GLB / PLY / STL | triangulated (these formats are triangle-only by spec) |

## Files

| File | Purpose |
|---|---|
| `app.py` | FastAPI server — upload, remesh, shrinkwrap, model/edges, features, export |
| `remesh_engine.py` | loader, curvature/detail score, engine dispatch, preprocessing, export |
| `quadwild.py` | QuadWild-BiMDF wrapper (incl. external-field injection) |
| `autoremesher.py` | AutoRemesher (Geogram QuadCover) adaptive-quad wrapper |
| `neurcross.py` | NeurCross neural-field hybrid (steers QuadWild) |
| `features.py` | feature analysis — crease loops, region classification, flow |
| `postprocess.py` | shrinkwrap (project remesh onto original) |
| `adaptive_remesh.py` | split / collapse / flip / tangential-smooth triangle remesher |
| `visibility_shells.py`, `seam_stitch.py`, `humanlogic.py`, `coarsen.py`, `shrinkwrap.py` | in-house quad engines (fallbacks / alternates) |
| `static/` | three.js viewers, UI, styles |

---

<p align="center">
  See the vendored engines for their own (GPL-3 / AGPL-3) licenses.
</p>
</content>
