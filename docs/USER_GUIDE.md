<!-- Generated from qt_help.py by make_user_guide.py. Edit qt_help.py, then regenerate. -->

# PRISM — quick-start guide

**The flow:** bring data in through the **Library**, then switch to the
workspace for your technique in the left rail. Everything you derive
(baseline-subtracted, combined, fitted) lands back in the Library as a new
spectrum. **File › Save project** keeps the whole session in one
`.prism` file (older project files still load).

## Library

**Import files…** auto-detects the format (plain XY, TA SDT thermal exports,
SAXS EDF, Rigaku `.rasx`, JCAMP-DX). If it guesses wrong, use
**Custom import…** (Ctrl+I) to pick the parser and the X/Y columns yourself, with a
preview. Right-click spectra to rename, duplicate, reorder, export as text
(Ctrl+E), combine (sum / average / weighted subtraction), or delete.
**Undo (Ctrl+Z)** steps back through library changes: deletes, renames,
duplicates, combine results, applied baselines, and accepted mineral IDs.

## Baseline

Pick a method — **arPLS** is the automatic default; polynomial / spline /
rubberband fit *through regions you choose*. Type regions like
`100-400; 1800-2600` or toggle **Add region by dragging** and sweep them
on the plot. **Preview** shows raw + baseline + subtracted; **Apply** creates a
`_bl` spectrum.

**Tip for glasses:** broad bands need a stiff arPLS — use λ around
**1e7**. At the usual 1e5 the baseline eats the band itself. (Verified against
the ISG pressure series: λ=1e7 beat a hand-drawn baseline.)

## Raman (Simple Plot)

Multi-select spectra to overlay them: separate panels or stacked with an
offset. Load CIF files to overlay predicted Bragg positions (managed per-CIF:
color, label, visibility). Difference mode shows A−B for exactly two selected
spectra. Toggle **Annotate on click** to pin peak-position labels.

## Peak Fitting

Set up components three ways: in the **Fit param.** table, with
**Auto-find peaks** (its *detection limit* spinner sets how strong a
candidate must be, in units of the noise — lower it to catch weak peaks), or
with **Pick peaks on plot** — toggle it and click each peak apex.
Then **Fit !**. Shapes: G (Gaussian), GL (pseudo-Voigt), V (true Voigt),
EMG (asymmetric, signed skew). The `FWHM=#` column links a component's
width to another's, and the **Name** column labels a component (e.g.
"ν1 PO4") — names follow into the legend, reports, and CSV exports.
Reports include R², ±1σ errors, and peak centroids;
**Conf. intervals** runs rigorous F-test profiling. Save a configuration as a
*model* to reuse it — models are also the recipes **Multi-Fit** applies
across many spectra at once.

**Origin-like mode** is a faithful stepwise Levenberg-Marquardt:
*1 iteration* performs exactly one damped parameter update (watch the
curve move), and *Fit until converged* repeats until the relative χ²
change drops below the tolerance — the same behavior as Origin's NLFit
dialog.

**Width convention:** the "FWHM" fields have always been *half*-widths
(HWHM) — a historical convention kept so old saved models stay valid. Double
them when quoting true FWHM.

## XAS

Import EasyXAFS ZIPs, CSVs, or Athena `.prj` from the workspace's own
buttons (CSV import recognizes an already-normalized `flat`/`norm` column,
so a previously-exported or externally-processed normalized spectrum loads
straight in). Tabs follow the workflow: Pre-processing (smoothing, Bragg
angle→energy correction, click-based tie-point alignment) → μ(E) Builder
(deglitch: noise-calibrated by default, or the older rolling z-score) →
Normalization / EXAFS (Larch; the FT window (kmin/kmax) and autobk's own
background-fit range (bkg kmax) are independent — leave bkg kmax blank to
use the old shared-range behavior, or widen it past the FT window to fit
the background spline over more data than the FT trusts, standard Athena
convention) → Analysis (average or sum repeat scans, difference, PCA
species count) → LCF (below) → Export (Athena formats). Importing an
Athena `.prj` needs no Larch.

### LCF

Combinatorial linear-combination fitting. Select your imported samples as
**targets** and your imported standards as **references**, set a component
count range (e.g. 2–3) and weight bounds, optionally require specific
references to always be included, then **Run batch LCF**: every target gets
fit against *every combination* of references in that size range, so which
combination actually explains each spectrum best doesn't have to be
guessed in advance. The summary table shows each target's best combination;
click a row to preview it — for a single fit against one fixed reference
set (no combination search), just select exactly the references you want
and set both component-count spinners to that same count. **Generate report
(PDF + MD)** writes one multi-page PDF and one Markdown file for the whole
batch — two pages per sample (the best-fit overlay with residual, then a
stats page: best-fit weight bar chart, a rank-vs-metric plot showing how
close the runner-up combinations came, and the full ranking table sorted by
R², RMS, or reduced χ²).

## DTA / Thermal

Pick X/Y/dY channels, compute Tg three ways (double tangent, parallel
tangent, |dY| max) — the result panel says whether the methods agree.
"Calculs" integrates or finds extrema over a range; Batch processes a folder.

## Raman ID (RRUFF)

Auto-find (or type) your spectrum's peak positions, then **Find matches**:
candidates are ranked by peak overlap, each showing its **laser wavelength** —
relative intensities vary between wavelengths, so judge the overlay yourself.
The **Database filters** restrict candidates before ranking: laser λ
(matched ±2 nm, so 532 also covers 532.6), oriented/unoriented, high-res vs
broad-scan, and quality category. Shift/Ctrl-click several result rows to
overlay multiple references at once.
Nothing is labeled until you click **Accept** — and Accept works
iteratively for mixtures: the accepted phase is recorded, the peaks it
explains are removed from the query, and the table immediately shows matches
for the REMAINING peaks (already-accepted references are excluded). Repeat
until every peak is explained; each accept is one Ctrl+Z away. One more click
overlays the matched mineral's predicted XRD pattern (AMCSD structure) in
the Raman workspace (needs the AMCSD download below).
**Download RRUFF database…** / **Download AMCSD structures…** fetch
and index those databases directly from rruff.net — no Python needed, runs
in the background, safe to re-run if interrupted.

## HT-XRD

Import a whole folder of patterns; temperatures come from `.rasx`
metadata or a filename template like `scan_???.xy`. The waterfall is
colored by temperature. The **Maps** tab adds a 2D heatmap
(linear/log/sqrt/power color scales), a difference map or difference waterfall
vs a reference pattern ("first", an index, or a temperature), a 3D surface,
an optional time axis from the heating rate, and dashed peak-guide lines
(`{slice:2θ; slice:2θ}` anchors, interpolated).

**Tracking**, three ways. (1) **Pick track guide** on the waterfall:
click the same peak at two (or more) temperatures — bottom and top, say — then
**Track picked guide** follows the peak along that line; one click from an
intermediate temperature also works for a peak that appears or dies.
(2) **Auto-track all peaks** finds every peak of the first pattern and
tracks them all (the generated windows appear in the field for editing).
(3) Typed windows — `28.5-29.5 @ 28.98; 31-32` — where the
`@` anchor picks WHICH peak to track when a window holds more than
one. In every mode the fit is seeded from the previous pattern (so it follows
a drifting peak instead of jumping to a stronger neighbor), and a peak weaker
than the *Absence σ* noise threshold is reported as absent instead of a
garbage fit — vanished/appeared peaks are flagged as transition signatures
alongside fit-quality anomalies.

## Clustering

Select a series (e.g. a multi-point map), choose KMeans or hierarchical and
a cluster count: PCA scatter colored by cluster, per-cluster mean spectra, and
an assignment table.

## Calculations

Every spectrum operation in one place. Between spectra: add, subtract,
multiply, divide, average, weighted sum, and **modulated addition**
(A + w(x)·B with a constant / linear-ramp / Gaussian envelope). Per
spectrum: scale/offset, normalizations (max, area, min-max), log/ln/exp/√/
power/reciprocal/|y|, x-shift and x-scale calibration, crop, resample,
smoothing (Savitzky-Golay, moving average, median), cosmic-ray despiking,
Savitzky-Golay derivatives, cumulative integral. Analysis:
**linear-combination fitting** (target ≈ Σ cᵢ·refᵢ, free or non-negative
coefficients, with R² and percentages) and summary statistics.
*Selection order matters*: the first-selected spectrum is A (or the LCF
target). Preview shows inputs faint + result bold; Apply adds derived
spectra to the Library (undoable).

## XRD ID

QualX-style phase identification against YOUR OWN card databases —
PRISM ships none. Download any database you have the rights to use (any
QualX-format .sq works) and register it with **Add database…** (or
**Add folder…** for several at once): a QualX-format file is indexed
once locally; a PRISM-format .sq (e.g. shared by a colleague) is used in
place. Check any number of registered databases to probe them all in one
search — every card keeps its source and code, and results say which
database each hit came from.
Auto-find (or type) the pattern's 2θ peaks, set λ and the match tolerance,
optionally restrict by chemistry (must-contain / must-exclude elements) or
source, then **Search match**: candidates ranked by figure of merit
(intensity-weighted coverage both ways), previewed as stick patterns under
the query (shift/ctrl-click to compare several). **Accept** is iterative
for mixtures, exactly like Raman ID: the phase is recorded, its peaks
leave the query, the rest is re-searched; Ctrl+Z undoes.
**Check Raman-identified phases here** overlays the XRD reference lines
of every phase accepted in Raman ID — the Raman→XRD cross-check in one
click. The Card browser tab looks up any card by name/mineral/formula.

## Figures

Publication-figure building (the Origin-inspired module). **XY builder**:
stack library spectra as layers — per-layer plot type (line, scatter,
line+symbols, sticks, filled area, bars, steps), color, vertical offset, and
multi-panel assignment — with Publication/Presentation/Poster style presets,
log axes, and export at exact centimeter size and dpi (PNG/SVG/PDF/TIFF).
**Point fitting**: Origin's classic models (linear, polynomials,
exponential decay/growth, power law, logarithmic, Boltzmann sigmoid,
Gaussian, Lorentzian, Arrhenius) with parameters ±1σ and R².
**Ternary**: barycentric composition plots from a CSV table, optional
color-mapped value column. **Raman ↔ XRD**: the cross-technique figure —
your Raman spectrum with its accepted Mineral-ID phases above the XRD
pattern with each accepted phase's reference stick pattern.

## SAXS/WAXS

Small/wide-angle scattering (ported from the author's own POMME suite).
Import Xenocs-style 1D ASCII curves; **Reduction** subtracts the empty
capillary with manual, auto, transmission, or physics-based (xraydb)
scaling; **Analysis** fits Guinier (Rg), generalized Porod (slope +
background), and pseudo-Bragg peaks (d-spacing, correlation length) with
auto region detection; **WAXS** auto-finds and fits pseudo-Voigt peaks
(d-spacings, crystallinity index). Curves can be sent to the Library.
The XAS workspace also gained a **Sample mass** tab — the Hephaestus
calculator on oxide compositions (mol%/wt%) or formulas: masses for total
μt = 1 / 2.5 and edge step Δμt = 1 in a pellet of chosen diameter.

## Glass

Composition-based glass property calculation. Paste (or load as CSV) a
composition table — header = oxides, one row per sample, mol% or wt% —
then: **Optical basicity &Lambda;** (oxygen-weighted Duffy mixing with the
recommended per-oxide values of Rodriguez & McCloy, PNNL-20184 Table B.1),
and **GlassNet predictions** (Tg, viscosity, density, refractive index and
~80 more; Cassar 2023, trained on SciGlass — estimates to validate, not
measurements). Compare against SciGlass, INTERGLAD, glassproperties.com.

## Shortcuts

`Ctrl+O` import · `Ctrl+I` custom import ·
`Ctrl+E` export · `Ctrl+S` save project ·
`Ctrl+Shift+O` open project · `Ctrl+Z` undo ·
`Ctrl+Q` quit · `F1` this guide.
View menu: dark mode, Python console (the live app objects are in scope).


---

## PRISM 2.6.0

*Platform for Research In Spectroscopy & Materials*

Import, process, and analyze scientific spectra: Raman, XAS/XANES/EXAFS,
DTA/DSC/TGA, XRD (including phase identification and high-temperature
series), SAXS.

Source: `github.com/sams808/PRISM`

### Please cite the reference databases that contribute to your work

**RRUFF** (Raman reference spectra): Lafuente B, Downs R T, Yang H,
Stone N (2015) "The power of databases: the RRUFF project." In: Highlights
in Mineralogical Crystallography, pp 1–30. Also acknowledge each matched
sample's owner/source shown in the Raman ID workspace.

**AMCSD** (crystal structures used for XRD overlays): Downs R T,
Hall-Wallace M (2003) "The American Mineralogist Crystal Structure
Database." American Mineralogist 88, 247–250.

© 2026 NOME Group, Washington State University.
Licensed under the MIT License — see LICENSE.

### Built on

numpy · scipy · pandas · matplotlib · lmfit · rampy · scikit-learn ·
PySide6 · xraylarch (optional, XAS)
