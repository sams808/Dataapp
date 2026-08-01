"""
baseline_science.py — baseline estimation/subtraction (framework-agnostic),
wrapping rampy.baseline. The Qt successor to the old Tk BaselineParamWindow
workflow (whose per-spectrum dual-dict state bug was one of the audit's
original findings — per-spectrum settings now live in a PerItemSettingsStore
keyed by spectrum id, at the UI layer).

rampy.baseline(x, y, bir, method, **kwargs) notes (checked against the
installed version's signature/docstring):
  - bir ("baseline interpolation regions") is an (n, 2) array of x-ranges
    the baseline is fitted THROUGH; required positionally even by the
    als/arPLS methods that ignore it.
  - methods used here: "poly" (polynomial_order), "unispline" (s),
    "rubberband", "als" (lam, p), "arPLS" (lam, ratio).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

BASELINE_METHODS = ["arPLS", "als", "poly", "unispline", "rubberband"]

# method -> [(param_name, label, default)] — drives the generic UI fields.
BASELINE_PARAM_DEFS: Dict[str, List[Tuple[str, str, str]]] = {
    "arPLS": [("lam", "λ", "1e5"), ("ratio", "ratio", "0.01")],
    "als": [("lam", "λ", "1e5"), ("p", "p", "0.01")],
    "poly": [("polynomial_order", "order", "3")],
    "unispline": [("s", "s", "1.0")],
    "rubberband": [],
}


def parse_roi_text(text: str) -> Optional[np.ndarray]:
    """Parse ROI ranges from a user string like "100-200; 500-600" (also
    accepts commas between the pair, e.g. "100,200; 500,600"). Returns an
    (n, 2) array, or None for empty input. Raises ValueError on malformed
    or inverted ranges."""
    text = (text or "").strip()
    if not text:
        return None
    rois = []
    for chunk in re.split(r"[;\n]+", text):
        chunk = chunk.strip()
        if not chunk:
            continue
        # A dash that FOLLOWS a digit is the range separator ("100-200"),
        # not a minus sign — rewrite it to a comma before extracting
        # numbers, so signed values ("-100 - -50") still parse.
        chunk_norm = re.sub(r"(?<=[\d.])\s*-\s*(?=[\d.+-])", ",", chunk)
        nums = re.findall(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", chunk_norm)
        if len(nums) != 2:
            raise ValueError(f"Bad ROI segment: {chunk!r} (expected two numbers, e.g. '100-200')")
        lo, hi = float(nums[0]), float(nums[1])
        if hi <= lo:
            raise ValueError(f"Bad ROI range: {lo}..{hi} (max must exceed min)")
        rois.append([lo, hi])
    if not rois:
        return None
    return np.asarray(rois, dtype=float)


def compute_baseline(
    x: np.ndarray, y: np.ndarray, *, method: str = "arPLS",
    roi: Optional[np.ndarray] = None, params: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_clean, y_subtracted, baseline) for the spectrum — x is
    returned too because non-finite points are dropped and the data is
    sorted, so the output grid may differ from the input's.

    roi: (n, 2) x-ranges the baseline is fitted through. Required by the
    fit-through methods (poly/unispline/rubberband); optional for als/arPLS
    (which ignore it — a full-range placeholder is passed to satisfy
    rampy's positional signature).
    """
    import rampy as rp

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    order = np.argsort(x, kind="mergesort")
    x, y = x[order], y[order]
    if len(x) < 10:
        raise ValueError("Need at least 10 points for baseline estimation.")

    if method not in BASELINE_METHODS:
        raise ValueError(f"Unknown baseline method: {method!r} (expected one of {BASELINE_METHODS}).")

    needs_roi = method in ("poly", "unispline", "rubberband")
    if roi is None:
        if needs_roi:
            raise ValueError(f"Method {method!r} needs at least one baseline region (ROI).")
        roi = np.array([[float(x.min()), float(x.max())]])
    else:
        roi = np.asarray(roi, dtype=float)
        lo, hi = float(x.min()), float(x.max())
        clipped = np.clip(roi, lo, hi)
        clipped = clipped[clipped[:, 1] > clipped[:, 0]]
        if needs_roi and len(clipped) == 0:
            raise ValueError("All baseline regions fall outside the spectrum's x-range.")
        roi = clipped if len(clipped) else np.array([[lo, hi]])

    kwargs = dict(params or {})
    y_sub, base = rp.baseline(x, y, roi, method, **kwargs)
    return x, np.asarray(y_sub, float).ravel(), np.asarray(base, float).ravel()
