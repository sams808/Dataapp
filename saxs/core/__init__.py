"""
saxs.core — the SAXS/WAXS science engine, ported from the author's own
'pomme' project (github.com/sams808/pomme) into PRISM as the SAXS/WAXS
module. Framework-agnostic: Curve model with provenance, Xenocs-style 1D
ASCII loader, physics-based empty/background reduction (xraydb),
Guinier/Porod/peak/unified analysis, WAXS multi-peak fitting, and the
composite-model staged-fitting pipeline (component library, presets,
staged fit, batch runner).

Re-exports the names qt_saxs.py (and most other external callers) need,
so `from saxs.core import Curve, load_curve, fit_staged, ...` works
without reaching into individual submodules.
"""
from saxs.core.analysis import (
    auto_detect_guinier_region,
    auto_detect_peak_window,
    auto_detect_porod_region,
    fit_guinier,
    fit_porod_general,
    fit_pseudo_bragg_peak,
)
from saxs.core.chemistry import CapillaryConfig, SamplePhysicsConfig
from saxs.core.composite_batch import BatchItem, batch_to_csv_rows, run_batch, write_batch_csv
from saxs.core.composite_fit import PRESETS, CompositeModel, build_composite, build_preset
from saxs.core.composite_models import COMPONENTS
from saxs.core.composite_staged import FitResult, fit_staged, propose_windows
from saxs.core.curve import Curve
from saxs.core.loader import load_curve
from saxs.core.reduction import CorrectionSettings, correct_sample
from saxs.core.waxs import auto_find_peaks, fit_waxs_peaks

__version__ = "PRISM"

__all__ = [
    "auto_detect_guinier_region", "auto_detect_peak_window", "auto_detect_porod_region",
    "fit_guinier", "fit_porod_general", "fit_pseudo_bragg_peak",
    "CapillaryConfig", "SamplePhysicsConfig",
    "BatchItem", "batch_to_csv_rows", "run_batch", "write_batch_csv",
    "PRESETS", "CompositeModel", "build_composite", "build_preset",
    "COMPONENTS",
    "FitResult", "fit_staged", "propose_windows",
    "Curve",
    "load_curve",
    "CorrectionSettings", "correct_sample",
    "auto_find_peaks", "fit_waxs_peaks",
]
