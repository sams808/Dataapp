"""
qt_help.py — in-app Help: the F1 quick-start guide (per-workspace, written
for a group member opening the app for the first time) and the About
dialog (versions + the citations the reference databases ask for).

docs/USER_GUIDE.md is generated from this file — run tools/make_user_guide.py
after editing, never edit the markdown directly.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import QDialog, QTextBrowser, QVBoxLayout, QWidget

# Single source of the app version: shown in the window title and About,
# and matched by the git tag at release time.
APP_VERSION = "2.5.0"
APP_NAME = "PRISM"
APP_TAGLINE = "Platform for Research In Spectroscopy & Materials"


def asset_path(name: str) -> str:
    """Path to a bundled asset (repo-root assets/ in dev, or the PyInstaller
    bundle dir when frozen)."""
    import os
    import sys
    if hasattr(sys, "_MEIPASS"):
        base = sys._MEIPASS
    else:
        # this file lives in core/, so assets/ is one level up at repo root
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "assets", name)


# Credits — shown from the Modules toolbar and the Help menu.
# <!-- PRISM was imagined, specified, and battle-tested by Sam Souda. -->
CREDITS_HTML = """
<div style='text-align:center'><h2>PRISM</h2>
<p><i>Platform for Research In Spectroscopy &amp; Materials</i></p></div>
<p>Developed in the NOME group, Washington State University.<br>
Supported by the U.S. Department of Energy.<br>
Thanks to Prof. John S. McCloy for the trust.</p>
<p>Reference databases: RRUFF (Lafuente et al. 2015), AMCSD (Downs &amp;
Hall-Wallace 2003), plus whatever XRD card databases you register yourself
&mdash; full citations in About.</p>
<!-- conceived and directed by Sam Souda -->
<p style='font-size:8pt; color:#888'>made with ChatGPT, improved with Claude&nbsp;&nbsp;&middot;&nbsp;&nbsp;s.s.</p>
"""
HELP_HTML = """
<h1>PRISM — quick-start guide</h1>

<p><b>The flow:</b> bring data in through the <b>Library</b>, then switch to the
workspace for your technique in the left rail. Everything you derive
(baseline-subtracted, combined, fitted) lands back in the Library as a new
spectrum. <b>File &rsaquo; Save project</b> keeps the whole session in one
<code>.prism</code> file (older project files still load).</p>

<h2>Library</h2>
<p><b>Import files…</b> auto-detects the format (plain XY, TA SDT thermal exports,
SAXS EDF, Rigaku <code>.rasx</code>, JCAMP-DX). If it guesses wrong, use
<b>Custom import…</b> (Ctrl+I) to pick the parser and the X/Y columns yourself, with a
preview. Right-click spectra to rename, duplicate, reorder, export as text
(Ctrl+E), combine (sum / average / weighted subtraction), or delete.
<b>Undo (Ctrl+Z)</b> steps back through library changes: deletes, renames,
duplicates, combine results, applied baselines, and accepted mineral IDs.</p>

<h2>Baseline</h2>
<p>Pick a method — <b>arPLS</b> is the automatic default; polynomial / spline /
rubberband fit <i>through regions you choose</i>. Type regions like
<code>100-400; 1800-2600</code> or toggle <b>Add region by dragging</b> and sweep them
on the plot. <b>Preview</b> shows raw + baseline + subtracted; <b>Apply</b> creates a
<code>_bl</code> spectrum.</p>
<p><b>Tip for glasses:</b> broad bands need a stiff arPLS — use λ around
<b>1e7</b>. At the usual 1e5 the baseline eats the band itself. (Verified against
the ISG pressure series: λ=1e7 beat a hand-drawn baseline.)</p>

<h2>Raman (Simple Plot)</h2>
<p>Multi-select spectra to overlay them: separate panels or stacked with an
offset. Load CIF files to overlay predicted Bragg positions (managed per-CIF:
color, label, visibility). Difference mode shows A−B for exactly two selected
spectra. Toggle <b>Annotate on click</b> to pin peak-position labels.</p>

<h2>Peak Fitting</h2>
<p>Set up components three ways: in the <b>Fit param.</b> table, with
<b>Auto-find peaks</b> (its <i>detection limit</i> spinner sets how strong a
candidate must be, in units of the noise — lower it to catch weak peaks), or
with <b>Pick peaks on plot</b> — toggle it and click each peak apex.
Then <b>Fit&nbsp;!</b>. Shapes: G (Gaussian), GL (pseudo-Voigt), V (true Voigt),
EMG (asymmetric, signed skew). The <code>FWHM=#</code> column links a component's
width to another's, and the <b>Name</b> column labels a component (e.g.
"ν1 PO4") — names follow into the legend, reports, and CSV exports.
Reports include R², ±1σ errors, and peak centroids;
<b>Conf. intervals</b> runs rigorous F-test profiling. Save a configuration as a
<i>model</i> to reuse it — models are also the recipes <b>Multi-Fit</b> applies
across many spectra at once.</p>
<p><b>Origin-like mode</b> is a faithful stepwise Levenberg-Marquardt:
<i>1 iteration</i> performs exactly one damped parameter update (watch the
curve move), and <i>Fit until converged</i> repeats until the relative χ²
change drops below the tolerance — the same behavior as Origin's NLFit
dialog.</p>
<p><b>Width convention:</b> the "FWHM" fields have always been <i>half</i>-widths
(HWHM) — a historical convention kept so old saved models stay valid. Double
them when quoting true FWHM.</p>

<h2>XAS</h2>
<p>Import EasyXAFS ZIPs, CSVs, or Athena <code>.prj</code> from the workspace's own
buttons. Tabs follow the workflow: Pre-processing (smoothing, Bragg
angle→energy correction, click-based tie-point alignment) → μ(E) Builder
(with deglitching) → Normalization / EXAFS (Larch) → Analysis (average or sum
repeat scans, difference, linear-combination fitting, PCA species count) →
Export (Athena formats). Importing an Athena <code>.prj</code> needs no Larch.</p>

<h2>DTA / Thermal</h2>
<p>Pick X/Y/dY channels, compute Tg three ways (double tangent, parallel
tangent, |dY| max) — the result panel says whether the methods agree.
"Calculs" integrates or finds extrema over a range; Batch processes a folder.</p>

<h2>Raman ID (RRUFF)</h2>
<p>Auto-find (or type) your spectrum's peak positions, then <b>Find matches</b>:
candidates are ranked by peak overlap, each showing its <b>laser wavelength</b> —
relative intensities vary between wavelengths, so judge the overlay yourself.
The <b>Database filters</b> restrict candidates before ranking: laser λ
(matched ±2 nm, so 532 also covers 532.6), oriented/unoriented, high-res vs
broad-scan, and quality category. Shift/Ctrl-click several result rows to
overlay multiple references at once.
Nothing is labeled until you click <b>Accept</b> — and Accept works
iteratively for mixtures: the accepted phase is recorded, the peaks it
explains are removed from the query, and the table immediately shows matches
for the REMAINING peaks (already-accepted references are excluded). Repeat
until every peak is explained; each accept is one Ctrl+Z away. One more click
overlays the matched mineral's predicted XRD pattern (AMCSD structure) in
the Raman workspace (needs the AMCSD download below).
<b>Download RRUFF database…</b> / <b>Download AMCSD structures…</b> fetch
and index those databases directly from rruff.net — no Python needed, runs
in the background, safe to re-run if interrupted.</p>

<h2>HT-XRD</h2>
<p>Import a whole folder of patterns; temperatures come from <code>.rasx</code>
metadata or a filename template like <code>scan_???.xy</code>. The waterfall is
colored by temperature. The <b>Maps</b> tab adds a 2D heatmap
(linear/log/sqrt/power color scales), a difference map or difference waterfall
vs a reference pattern ("first", an index, or a temperature), a 3D surface,
an optional time axis from the heating rate, and dashed peak-guide lines
(<code>{slice:2θ; slice:2θ}</code> anchors, interpolated).</p>
<p><b>Tracking</b>, three ways. (1) <b>Pick track guide</b> on the waterfall:
click the same peak at two (or more) temperatures — bottom and top, say — then
<b>Track picked guide</b> follows the peak along that line; one click from an
intermediate temperature also works for a peak that appears or dies.
(2) <b>Auto-track all peaks</b> finds every peak of the first pattern and
tracks them all (the generated windows appear in the field for editing).
(3) Typed windows — <code>28.5-29.5 @ 28.98; 31-32</code> — where the
<code>@</code> anchor picks WHICH peak to track when a window holds more than
one. In every mode the fit is seeded from the previous pattern (so it follows
a drifting peak instead of jumping to a stronger neighbor), and a peak weaker
than the <i>Absence σ</i> noise threshold is reported as absent instead of a
garbage fit — vanished/appeared peaks are flagged as transition signatures
alongside fit-quality anomalies.</p>

<h2>Clustering</h2>
<p>Select a series (e.g. a multi-point map), choose KMeans or hierarchical and
a cluster count: PCA scatter colored by cluster, per-cluster mean spectra, and
an assignment table.</p>

<h2>Calculations</h2>
<p>Every spectrum operation in one place. Between spectra: add, subtract,
multiply, divide, average, weighted sum, and <b>modulated addition</b>
(A + w(x)·B with a constant / linear-ramp / Gaussian envelope). Per
spectrum: scale/offset, normalizations (max, area, min-max), log/ln/exp/√/
power/reciprocal/|y|, x-shift and x-scale calibration, crop, resample,
smoothing (Savitzky-Golay, moving average, median), cosmic-ray despiking,
Savitzky-Golay derivatives, cumulative integral. Analysis:
<b>linear-combination fitting</b> (target ≈ Σ cᵢ·refᵢ, free or non-negative
coefficients, with R² and percentages) and summary statistics.
<i>Selection order matters</i>: the first-selected spectrum is A (or the LCF
target). Preview shows inputs faint + result bold; Apply adds derived
spectra to the Library (undoable).</p>

<h2>XRD ID</h2>
<p>QualX-style phase identification against YOUR OWN card databases —
PRISM ships none. Download any database you have the rights to use (any
QualX-format .sq works) and register it with <b>Add database…</b> (or
<b>Add folder…</b> for several at once): a QualX-format file is indexed
once locally; a PRISM-format .sq (e.g. shared by a colleague) is used in
place. Check any number of registered databases to probe them all in one
search — every card keeps its source and code, and results say which
database each hit came from.
Auto-find (or type) the pattern's 2θ peaks, set λ and the match tolerance,
optionally restrict by chemistry (must-contain / must-exclude elements) or
source, then <b>Search match</b>: candidates ranked by figure of merit
(intensity-weighted coverage both ways), previewed as stick patterns under
the query (shift/ctrl-click to compare several). <b>Accept</b> is iterative
for mixtures, exactly like Raman ID: the phase is recorded, its peaks
leave the query, the rest is re-searched; Ctrl+Z undoes.
<b>Check Raman-identified phases here</b> overlays the XRD reference lines
of every phase accepted in Raman ID — the Raman→XRD cross-check in one
click. The Card browser tab looks up any card by name/mineral/formula.</p>

<h2>Figures</h2>
<p>Publication-figure building (the Origin-inspired module). <b>XY builder</b>:
stack library spectra as layers — per-layer plot type (line, scatter,
line+symbols, sticks, filled area, bars, steps), color, vertical offset, and
multi-panel assignment — with Publication/Presentation/Poster style presets,
log axes, and export at exact centimeter size and dpi (PNG/SVG/PDF/TIFF).
<b>Point fitting</b>: Origin's classic models (linear, polynomials,
exponential decay/growth, power law, logarithmic, Boltzmann sigmoid,
Gaussian, Lorentzian, Arrhenius) with parameters ±1σ and R².
<b>Ternary</b>: barycentric composition plots from a CSV table, optional
color-mapped value column. <b>Raman ↔ XRD</b>: the cross-technique figure —
your Raman spectrum with its accepted Mineral-ID phases above the XRD
pattern with each accepted phase's reference stick pattern.</p>

<h2>SAXS/WAXS</h2>
<p>Small/wide-angle scattering (ported from the author's own POMME suite).
Import Xenocs-style 1D ASCII curves; <b>Reduction</b> subtracts the empty
capillary with manual, auto, transmission, or physics-based (xraydb)
scaling; <b>Analysis</b> fits Guinier (Rg), generalized Porod (slope +
background), and pseudo-Bragg peaks (d-spacing, correlation length) with
auto region detection; <b>WAXS</b> auto-finds and fits pseudo-Voigt peaks
(d-spacings, crystallinity index). Curves can be sent to the Library.
The XAS workspace also gained a <b>Sample mass</b> tab — the Hephaestus
calculator on oxide compositions (mol%/wt%) or formulas: masses for total
μt = 1 / 2.5 and edge step Δμt = 1 in a pellet of chosen diameter.</p>

<h2>Glass</h2>
<p>Composition-based glass property calculation. Paste (or load as CSV) a
composition table &mdash; header = oxides, one row per sample, mol% or wt% &mdash;
then: <b>Optical basicity &Lambda;</b> (oxygen-weighted Duffy mixing with the
recommended per-oxide values of Rodriguez &amp; McCloy, PNNL-20184 Table B.1),
and <b>GlassNet predictions</b> (Tg, viscosity, density, refractive index and
~80 more; Cassar 2023, trained on SciGlass &mdash; estimates to validate, not
measurements). Compare against SciGlass, INTERGLAD, glassproperties.com.</p>

<h2>Shortcuts</h2>
<p><code>Ctrl+O</code> import · <code>Ctrl+I</code> custom import ·
<code>Ctrl+E</code> export · <code>Ctrl+S</code> save project ·
<code>Ctrl+Shift+O</code> open project · <code>Ctrl+Z</code> undo ·
<code>Ctrl+Q</code> quit · <code>F1</code> this guide.
View menu: dark mode, Python console (the live app objects are in scope).</p>
"""

ABOUT_HTML = f"""
<h2>{APP_NAME} {APP_VERSION}</h2>
<p><i>{APP_TAGLINE}</i></p>
<p>Import, process, and analyze scientific spectra: Raman, XAS/XANES/EXAFS,
DTA/DSC/TGA, XRD (including phase identification and high-temperature
series), SAXS.</p>
<p>Source: <code>github.com/sams808/PRISM</code></p>
<h3>Please cite the reference databases that contribute to your work</h3>
<p><b>RRUFF</b> (Raman reference spectra): Lafuente B, Downs R T, Yang H,
Stone N (2015) "The power of databases: the RRUFF project." In: Highlights
in Mineralogical Crystallography, pp 1–30. Also acknowledge each matched
sample's owner/source shown in the Raman ID workspace.</p>
<p><b>AMCSD</b> (crystal structures used for XRD overlays): Downs R T,
Hall-Wallace M (2003) "The American Mineralogist Crystal Structure
Database." American Mineralogist 88, 247–250.</p>
<p style='font-size:8pt; color:#888'>© 2026 NOME Group, Washington State University.
Licensed under the MIT License — see LICENSE.</p>
<h3>Built on</h3>
<p>numpy · scipy · pandas · matplotlib · lmfit · rampy · scikit-learn ·
PySide6 · xraylarch (optional, XAS)</p>
"""


class HelpDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None, *, html: str = HELP_HTML, title: str = "Quick-start guide"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(760, 620)
        layout = QVBoxLayout(self)
        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        self.browser.setHtml(html)
        layout.addWidget(self.browser)


# Per-module in-depth guides (Help > Module guides). Deliberately verbose --
# these are the training documents for new group members.
def _guide(title, body):
    return "<h1>" + title + "</h1>" + body


MODULE_GUIDES = {
    "Raman": _guide("Raman module", """
<p>The Raman block covers plotting (<b>Raman</b>), identification
(<b>Raman ID</b>), and fitting (<b>Peak Fitting</b>/<b>Multi-Fit</b>).</p>
<h2>Workflow</h2><ol>
<li>Import spectra in the Library (auto-detected; Custom import if not).</li>
<li>Baseline-subtract in <b>Baseline</b>: arPLS &lambda;&asymp;1e7 for broad glass
bands &mdash; at 1e5 the baseline eats the band. Verify with Preview before Apply.</li>
<li>Identify in <b>Raman ID</b>: auto-find peaks (lower the &times;&sigma; limit for
weak bands), filter the database by your laser &lambda; and sample type, then judge
the overlay yourself &mdash; the ranking assists, you decide. Accept iteratively
for mixtures; every accept is Ctrl+Z-undoable and excluded from re-searches.</li>
<li>Fit in <b>Peak Fitting</b>: pick components by clicking apexes, auto-find,
or the table. G/GL/V/EMG shapes; widths are HWHM (double for true FWHM).
Classic = one-shot Levenberg-Marquardt; Origin-like = visible step-by-step
iterations. Save models to reuse in Multi-Fit batches.</li></ol>
<h2>Pitfalls</h2><ul><li>Compare intensities only within one laser wavelength
&mdash; resonance changes relative heights between 532 and 785 nm.</li>
<li>Cosmic-ray spikes bias fits: despike first (Calculations).</li>
<li>Fit R&sup2; &gt; 0.99 with absurd widths means overlapping components:
link widths (FWHM=#) or fix &eta;.</li></ul>"""),
    "XRD": _guide("XRD module", """
<p><b>XRD ID</b> is a QualX-style search-match on card databases YOU
register — PRISM ships none. Any QualX-format .sq you have the rights to
use works; enable several at once and one search probes them all.</p>
<h2>Databases</h2><ol>
<li><b>Add database…</b> registers one .sq (<b>Add folder…</b> scans a
folder). A QualX-format file is indexed once locally — minutes for
hundreds of thousands of cards, then instant forever; a PRISM-format .sq
is used in place.</li>
<li>The checkboxes choose which registered databases each search probes;
every hit shows its source and card code.</li>
<li>Respect each database's license: only share converted files with
people covered by the same rights you downloaded it under.</li></ol>
<h2>Search-match, step by step</h2><ol>
<li>Auto-find 2&theta; peaks (or type them). Check &lambda; (default Cu K&alpha; 1.5406 &Aring;).</li>
<li>Restrict chemistry: contains-all with elements you KNOW are present;
excludes with those that cannot be. This is the biggest ranking help.</li>
<li>Read the FoM as intensity-weighted coverage both ways: a phase whose
strong lines are absent from your pattern is penalized even if every one of
your peaks matches it.</li>
<li>Narrow with the card filters when you know the crystallography:
symmetry (crystal system) and card quality are check-droplists, the
space-group field matches by substring. Identical cards carried by
several databases are shown once.</li>
<li>Accept per phase; the accepted phase's reference bars STAY on the
plot (muted color) and the query peaks it explains turn gray while the
remainder is re-searched &mdash; repeat until everything is explained.
<b>Clear</b> starts the session over (recorded identifications are
untouched; undo those with Ctrl+Z in the Library).</li></ol>
<h2>HT-XRD</h2><p>Temperature series: waterfall + Maps (a heatmap with
log/sqrt scaling reveals weak reflections; the difference map vs a reference
slice localizes transitions). Track peaks by clicking the same reflection at
two temperatures on the waterfall &mdash; absence detection reports where a peak
genuinely vanishes instead of fitting noise; vanished/appeared flags are the
transition candidates.</p>
<h2>Cross-check</h2><p>Check Raman-identified phases here overlays the XRD
reference lines of everything accepted in Raman ID &mdash; agreement between the
two techniques is strong evidence; disagreement usually means an amorphous
phase (Raman-visible, XRD-silent) or trace crystallites (the reverse).</p>"""),
    "XAS": _guide("XAS module", """
<p>The pipeline follows the tabs left to right.</p>
<ol><li><b>Import</b>: EasyXAFS ZIP (I0/It channels), CSV, Athena .prj
(no Larch needed for import).</li>
<li><b>Pre-processing</b>: smooth only for e0-finding, never the data you
quantify; Bragg-glitch energy correction; tie-point alignment for scan
drift (Mode C: click matching features BEFORE and AFTER).</li>
<li><b>&mu;(E) Builder</b>: ln(I0/It); deglitch with z&asymp;6 removes monochromator
glitches without touching EXAFS oscillations.</li>
<li><b>Normalization / EXAFS</b> (needs Larch &rarr; run via the Python install,
not the portable exe): pre/post-edge ranges are passed EXPLICITLY (a silent
Larch pitfall this app fixes); autobk rbkg &asymp; 1.0; k-weight 2 is the usual
glass choice.</li>
<li><b>Analysis</b>: average repeat scans (or sum partial acquisitions),
difference spectra, linear-combination fitting against reference standards
(coefficients = phase fractions), PCA species count.</li>
<li><b>Sample mass</b>: paste your oxide composition (mol% or wt%), pick
element+edge and pellet diameter &rarr; masses for total &mu;t = 1 / 2.5 and step
&Delta;&mu;t = 1 (Hephaestus rules of thumb). The <b>target &mu;t</b> field is your
own target absorption length (thickness in units of 1/&mu;) &mdash; change it from
the 2.5 default to get the mass for whatever thickness you actually want,
shown alongside the two fixed reference values. If &Delta;&mu;t = 1 needs several
times the &mu;t = 2.5 mass, the absorber is dilute: measure in fluorescence.</li></ol>"""),
    "Thermal": _guide("Thermal module", """
<p><b>DTA / Thermal</b> computes Tg three independent ways &mdash; double tangent,
parallel tangent, |dY| max &mdash; and says whether they agree. Trust the value
only when the spread is small; a disagreement flag usually means a poorly
chosen baseline window (adjust LOW/HIGH) or an overlapping event. Calculs
integrates peaks and finds extrema; Batch processes a whole folder.</p>
<p>Convention: onset temperatures (tangent methods) are the reportable Tg;
|dY| max is the inflection &mdash; systematically higher. State which you quote.</p>"""),
    "Processing": _guide("Processing module", """
<p><b>Baseline</b>: arPLS (automatic; &lambda; stiffness &mdash; 1e7 for broad glass
bands), ALS, polynomial/spline/rubberband through picked regions. Settings
are remembered per spectrum; batch Apply creates _bl spectra.</p>
<p><b>Calculations</b> is the everything-else toolbox: arithmetic between
spectra (selection ORDER defines A), modulated addition (blend a reference
in over a chosen range via ramp/gaussian envelopes), normalizations,
log/exp/power transforms, x-calibration (shift/scale), crop, resample,
smoothing (SG keeps peak shapes; median kills spikes), despiking,
derivatives, integrals, linear-combination fitting (non-negative for
physical mixtures), statistics, and cluster analysis of spectral series
(KMeans/hierarchical over a common grid, PCA scores, per-cluster means &mdash;
for multi-point maps, find which spectra group together before averaging).</p>"""),
    "SAXS/WAXS": _guide("SAXS/WAXS module", """
<p>Ported from POMME. <b>Reduction</b>: subtract the empty capillary &mdash;
auto matches the high-q tail, transmission uses measured T, physics computes
both attenuations from compositions via xraydb; inspect the corrected curve
before analyzing.</p>
<p><b>Analysis</b>: Guinier is only valid for qRg &#8818; 1.3 &mdash; the report
prints qRg max, take it seriously; Rg &rarr; sphere-equivalent diameter shown.
The generalized Porod slope m reads surface character (4 = smooth interface,
3&ndash;4 rough, &lt;3 mass fractal). Pseudo-Bragg peaks give d = 2&pi;/q0 and an
apparent correlation length from the width.</p>
<p><b>WAXS</b>: pseudo-Voigt multi-peak fit with amorphous-hump detection &mdash;
the crystallinity index is crystalline area / total area, comparable only
between samples measured identically.</p>
<p><b>Composite fit</b>: a physically-meaningful component library (flat
background, power law, Guinier, Guinier-Porod, Beaucage unified, DAB,
Teubner-Strey, broad peak) summed into a fittable composite &mdash; the
general-purpose successor to the single pseudo-Bragg peak fit above, built
for structure-factor peaks (a genuine correlation length &xi;, not just an
apparent one) riding on top of other scattering contributions. <b>Auto (BIC
ladder)</b> runs the full staged pipeline (hygiene &rarr; background &rarr;
+peak &rarr; +low-q component &rarr; global refit &rarr; diagnostics &rarr;
BIC-based model selection) and reports whichever composite the data actually
supports; a specific preset fits that exact composite directly instead.
Windows (W_peak/W_hiq/W_loq) auto-propose from the curve's own shape and stay
editable. <b>Batch</b> fits every curve selected in the list below
independently &mdash; each gets its own auto-proposed windows, never one
curve's windows blanket-applied across the batch &mdash; with a results table
and CSV export (including the d<sub>TS</sub>/d<sub>gauss</sub> cross-check
against the pseudo-Bragg fit). A ladder-demoted preset (e.g. a peak-free
curve landing on plain background) is the model-selection logic correctly
saying the data doesn't support a peak &mdash; check the Flags line before
overriding with a forced preset.</p>"""),
    "Figures": _guide("Figures module", """
<p>Build the figure ONCE, export at the journal exact size.
<b>XY builder</b>: add spectra as layers; each layer has type, color,
offset, panel; presets set publication-grade fonts/ticks. Export at cm
size + dpi (600 for line art). <b>Point fitting</b>: model library with
&plusmn;1&sigma; and R&sup2; &mdash; fit Tg vs composition, Arrhenius, sigmoids.
<b>Ternary</b>: composition tables from CSV, color-mapped property.
<b>Raman &harr; XRD</b>: the two-panel identification figure with accepted
phases annotated.</p>"""),
    "Fitting": _guide("Fitting module", """
<p><b>Peak Fitting</b> for one spectrum, <b>Multi-Fit</b> for batches &mdash; the
same JSON models serve both, so refine interactively then apply to a series.
Widths are HWHM everywhere (historic convention; double for true FWHM). Use
confidence intervals (F-test) when a parameter matters for the conclusion,
not just the covariance &plusmn;1&sigma;. In Origin-like mode each iteration is one
damped LM update: watch parameters walk, stop when &chi;&sup2; stalls.</p>"""),
}
