# PRISM — Platform for Research In Spectroscopy & Materials

A desktop application for importing, processing, and analyzing scientific
spectra: Raman (with mineral identification), XRD (phase identification and
high-temperature series), XAS/XANES/EXAFS, DTA/DSC/TGA thermal analysis,
SAXS/WAXS, glass property prediction, and publication figure building.

The app is PySide6/Qt-based: one main window with a left navigation rail of
technique workspaces, color-coded by module. Each module (Raman, Fitting,
XRD, XAS, Thermal, Processing, Figures, SAXS/WAXS, Glass) can be switched
on/off in the **Modules** menu — a fresh install starts with only Raman
enabled, so a single-technique user sees a simple app.

## Workspaces

| Workspace | What it does |
|---|---|
| **Library** | Import data files (auto-detected parser, or Custom Import with parser/column override), browse, preview, rename/duplicate/reorder, delete with Undo, Combine/scale (sum, average, weighted subtraction), export as text. Feeds the other workspaces. |
| **Raman** | Simple Plot: multi-spectrum plotting (separate or stacked), smoothing, color schemes, axis controls, CIF Bragg-peak overlays with a per-CIF manager, difference mode, click-to-annotate, PNG/SVG/PDF export. |
| **Raman ID** | RRUFF database match-assist: ranks mineral candidates by Raman peak overlap, with database filters (laser wavelength, oriented/unoriented, high-res vs broad-scan, quality) applied before ranking; shows each candidate's laser excitation wavelength and overlays the reference spectrum — identification is always the user's explicit decision, never automatic. Requires a local RRUFF cache (see below). |
| **Peak Fitting** | Single-spectrum peak fitting (Gaussian, pseudo-Voigt, true Voigt, EMG via lmfit): classic one-shot LM or Origin-style stepwise LM, auto peak finding with an adjustable detection limit, click-to-pick peaks, parameter linking, residual subplot, F-test confidence intervals, per-component CSV export, fit reports with R², ±1σ errors and centroids, save/load parameter models. |
| **Multi-Fit** | Batch fitting: apply a saved parameter model ("recipe" — the same JSON files Peak Fitting saves) to many spectra at once; results table + CSV export. |
| **Baseline** | Baseline subtraction (arPLS, ALS, polynomial, spline, rubberband) with live preview, drag-to-pick fit regions, per-spectrum settings memory, and batch apply producing `_bl` spectra. |
| **XRD ID** | QualX-style phase identification over **your own registered card databases** (PRISM ships none — see below): search-match with figure-of-merit ranking, chemistry/source filters, stick-pattern previews, iterative Accept for mixtures, an element-aware card browser, and the Raman↔XRD cross-check. |
| **HT-XRD** | High-temperature XRD series: import a folder of patterns (temperature from `.rasx` metadata or a Jana-style `???` filename template), temperature-colored waterfall, a Maps tab (heatmap with linear/log/sqrt/power scales, difference maps, 3D surface), and multi-window peak tracking with absence detection and transition flags. |
| **XAS** | Full XAS/XANES/EXAFS pipeline: EasyXAFS ZIP / CSV / Athena `.prj` import, μ(E) builder with deglitching, Larch normalization and EXAFS/FT, merge/average, difference spectra, linear-combination fitting, PCA species count, edge definer, sample-mass calculator (Hephaestus-style), Athena `.dat`/`.prj` export. Requires `xraylarch` (see `requirements-xas.txt`). |
| **DTA / Thermal** | Tg determination by three methods (double tangent, parallel tangent, \|dY\| max) with agreement scoring, integration/extrema "Calculs", batch processing with CSV export. |
| **SAXS/WAXS** | Curve loading/reduction (background subtraction, corrections), Guinier/Porod/correlation-peak analysis, WAXS crystallinity fitting. |
| **Glass** | Composition-based property calculation: optical basicity Λ (recommended per-oxide values, oxygen-weighted Duffy mixing) and GlassNet machine-learning predictions (~80 properties) from pasted or CSV composition tables. |
| **Calculations** | 30+ spectrum operations: arithmetic, normalization, interpolation, derivatives, smoothing, despiking (with click-picked spike positions), area/moments, correlation, clustering (KMeans/hierarchical with PCA), and more. |
| **Figures** | Publication figure building: multi-layer XY builder with per-layer plot types and dual axes, difference plots, 2D/series views (heatmap, contours, 3D waterfall), table plots (histogram, box, violin, correlation matrix), point fitting with a model library, ternary diagrams, Raman+XRD combination figures; Publication/Presentation/Poster style presets. |

Cross-cutting: `.prism` project files (everything survives closing the app;
legacy project files still load), a Python console (View menu) with the live
app objects in scope, dark mode by default, background threading for batch
operations, per-module guides in the Help menu, and keyboard shortcuts
(Ctrl+O import, Ctrl+S save project, Ctrl+E export, Ctrl+Z undo delete).

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt      # core science stack
pip install -r requirements-qt.txt   # PySide6, for the Qt app
pip install -r requirements-xas.txt  # xraylarch, optional — XAS workspace
pip install -r requirements-dev.txt  # pytest + pytest-qt, for running the tests
```

## Running

```bash
python qt_main.py
```

Or double-click `PRISM.bat`. A standalone `PRISM.exe` (no Python needed) can
be built with `build_exe.bat`; the built `dist/PRISM/` folder is portable
(~350 MB; the Larch-dependent XAS steps require the Python route).

Note on startup time: the very first launch after installing or pulling
compiles bytecode and lets the antivirus scan the scientific stack — allow
it a minute. Subsequent launches take a few seconds.

## XRD card databases (XRD ID workspace)

PRISM ships **no** XRD reference data. Instead, download whichever card
database you have the rights to use — any QualX-format `.sq` file works —
and register it in the XRD ID workspace with **Add database…** (or **Add
folder…** to register several at once):

- A **QualX-format** `.sq` is converted once into PRISM's indexed format
  (stored under `~/.raman_cache/xrd_id/imported/`; minutes for hundreds of
  thousands of cards, then searches take ~1 s).
- A **PRISM-format** `.sq` (e.g. an indexed file shared by a colleague) is
  registered in place, no copy.
- Any number of registered databases can be enabled at once; one search
  probes them all, and every hit reports its database, source tag, and
  original card code.

Respect the license of every database you register: only pass a converted
`.sq` to people covered by the same rights you downloaded it under, and
never post licensed database content publicly.

## Building the RRUFF database cache (Raman ID workspace)

The Raman ID workspace needs a one-time local ingest of the RRUFF Raman
database (https://rruff.net — please cite: Lafuente, Downs, Yang & Stone (2015),
"The power of databases: the RRUFF project"). Three ways to build it —
**no Python install is needed for any of them**:

1. **In the app** (simplest): open the Raman ID workspace and click
   **Download RRUFF database…** (and, for the XRD-overlay button,
   **Download AMCSD structures…**). Downloads run in the background and can
   be re-run if interrupted.
2. **Portable exe, without opening the GUI**: double-click
   `Download-RRUFF-database.bat` (or the `.ps1`) next to `PRISM.exe` —
   these just run `PRISM.exe --build-rruff-cache` headlessly. Progress is
   logged to `rruff_download.log` in the same folder. `Download-AMCSD-structures.bat`
   is the CIF-overlay counterpart.
3. **From source**: `python qt_main.py --build-rruff-cache` (add
   `--categories excellent_oriented fair_oriented ...` to fetch only some
   quality tiers), or in Python directly:
   ```python
   import rruff_science as rs
   rs.download_and_build_rruff_cache()   # downloads + indexes in one call
   ```

Any of these lands the cache in `~/.raman_cache/rruff/` (~1.2 GB for the
full database, ~28,000 spectra / ~2,500 minerals) and it's loaded
automatically by the Raman ID workspace. The downloaded ZIPs are kept
under `~/.raman_cache/rruff/downloads/` so a re-run resumes instead of
re-downloading everything.

## Running the tests

```bash
pytest
```

## Repository layout

Science layer (framework-agnostic, fully tested — no GUI imports):
- `io_universal.py` — pluggable parser framework (XY text, TA SDT, SAXS EDF, Rigaku `.rasx`, JCAMP-DX, …)
- `cif_tools.py` — CIF parsing + Bragg peak generation (disk-cached)
- `dta_science.py` — Tg/derivative/integration math
- `fitting_science.py` — lmfit peak models, fitting entry point, peak finding
- `xas_science.py` / `xas_mass.py` — XAS/XANES/EXAFS engine + sample-mass calculator
- `rruff_science.py` — RRUFF database ingest + match ranking + pack/unpack
- `xrd_id_science.py` — XRD search-match engine + database registry
- `htxrd_science.py` — HTXRD series loading + peak tracking + transition flagging
- `calc_science.py` / `cluster_science.py` / `spectrum_math.py` / `baseline_science.py` — the Calculations toolbox
- `glass_science.py` — optical basicity + GlassNet wrapper
- `figures_science.py` — point-fit models, ternary geometry, style presets
- `saxs_core/` — SAXS/WAXS curve model, reduction, analysis
- `project_io.py` — `.prism` project files

Qt layer:
- `qt_main.py` — entry point; `qt_shell.py` — main window/navigation/modules
- `qt_widgets.py` (shared plot widget with debounced redraws), `qt_theme.py`,
  `qt_models.py` (Spectrum/SpectrumLibrary), `qt_settings_store.py`,
  `qt_exception_hook.py`, `qt_worker.py`, `qt_help.py`
- One `qt_*.py` per workspace

## Troubleshooting

If a file fails to import in the Library, the parser registry may have
misdetected the format — check `io_universal.py`'s parser list; every parser
records its decision in the returned metadata (`selected_parser`).

## Sharing the local databases with colleagues

Reference databases are single files once built/registered:

| Database | File | Share it? |
|---|---|---|
| XRD ID | the registered `.sq` files (see `~/.raman_cache/xrd_id/`) | **Only within the license of each database** — a converted `.sq` carries the original database's content, so hand it only to people covered by the same rights (USB/network drive), and never post licensed content publicly. |
| RRUFF Raman | run `rruff_science.pack_rruff_database()` → one `rruff_pack.sq` (~1 GB) | Yes, with attribution (Lafuente et al. 2015). Import on the other machine with `rruff_science.unpack_rruff_database(path)`. |

These files cannot live in this git repository: GitHub hard-rejects files
over 100 MB (and LFS quotas don't fit multi-GB scientific databases).
Rebuild from your own downloads instead, or copy the single files directly.

## License

MIT — see [LICENSE](LICENSE). Developed in the NOME group at Washington
State University, with support from the U.S. Department of Energy;
federally funded software can carry government/data rights beyond the
license text, so check with your institution's tech transfer office before
distributing outside the group. See [NOTICE](NOTICE.md) for the full
institutional disclaimer.
