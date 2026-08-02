PRISM 2.6.0 — Composite SAXS model fitting, raw EDF ingestion, and a full internal reorganization.

## SAXS/WAXS — Composite model fitting (new)
- **New "Composite fit" tab** in the SAXS/WAXS workspace: physically meaningful component models (background, power-law, Guinier, Guinier-Porod, Beaucage unified, DAB, Ornstein-Zernike, Teubner-Strey peak) summed into composite models and fit in a staged, reproducible sequence — hygiene → background → peak → low-q shape → global refinement → model-selection ladder.
- **Model-selection ladder** (BG → BG_DAB → BG_TS → BG_TS_OZ → BG_TS_PL2 → BG_TS_GP) picks the simplest model that actually fits, with a significance guardrail so noise never gets mistaken for a real peak, and a per-window visual-equivalence gate so a statistically-preferred model that's visibly wrong in some region is rejected rather than silently reported.
- **Uncertainty that's honest about what the data can constrain**: profile-likelihood confidence intervals for the peak spacing and correlation length, with an explicit "not identified within this data's range" flag (plus a reported lower bound) rather than a falsely precise number when the fit genuinely can't pin a parameter down.
- **Batch runner** across a whole sample series with parameter continuation between neighboring samples, CSV export, and comparison against the legacy Gaussian peak fit.
- Runs on a background thread (`qt_worker.py`) so a staged or batch fit never freezes the UI.
- Validated against a 20-curve synthetic battery (recovers known peak spacing/correlation length within tolerance, never mistakes a peak-free curve for one with a peak) plus a frozen real-profile regression test.

## SAXS — raw EDF frame ingestion
- General-purpose reduction of raw 2D detector frames from a beamstopless instrument (Xeuss/Zeus Pro) via pyFAI, feeding the existing empty-subtraction pipeline unchanged: exposure-rate normalization, a direct-beam exclusion mask, multi-empty averaging with uncertainty propagation, and geometry validated against a real calibration frame.

## XAS — sample mass calculator
- **User-adjustable target absorption length** (target μt field, defaults to 2.5) instead of only the two fixed Hephaestus reference points.
- **User-adjustable eV-above-edge offset** for the μ/ρ evaluation point (defaults to 3, matching Hephaestus' own convention) — fixes a case where the previous hardcoded +50 eV point produced a ~6x mass discrepancy against a Hephaestus reference calculation.

## Figures
- Independent "ticks" and "tick labels" checkboxes in the XY builder (publication figures often want tick marks without labels on a shared axis, or vice versa); fixed font scaling on export.

## Internal reorganization (invisible day-to-day, but the app is now much easier to extend)
- **The ~50 flat Python files at the repo root are now organized into per-technique packages** (`core/`, `raman/`, `xrd/`, `xas/`, `dta/`, `saxs/`, `glass/`, `processing/`, `figures/`, `tools/`) matching the app's own nav-rail groupings, with tests mirrored the same way.
- **Split the two largest files**: `composite_staged.py` (2845 lines) into 6 focused internal modules, and `qt_xas.py` (1474 lines, 8 tabs) into one module per tab — both purely structural, no behavior change.
- Deduplicated a `_to_float` helper that had drifted into 11 slightly-different copies, and a ~110-line duplicated pair of low-q model-fitting helpers.
- Added a GitHub Actions CI workflow (pytest + lint on every push/PR); pinned all dependency versions; added `glasspy` and `pyFAI`/`fabio` to the requirements files (previously used by real features but silently absent, leaving their tests permanently skipped on a fresh install).
- Fixed a portable-build size regression (349 MB → 684 MB → back to 349 MB): a transitive PyInstaller hook was silently bundling PyTorch despite an explicit exclude rule; excluding the actual root-cause package fixes it for good.
- `LICENSE` now contains the exact, unmodified MIT license text (for correct license detection); the institutional funding disclaimer moved to `NOTICE.md`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
