"""
spectrum_math.py — spectrum arithmetic (framework-agnostic): sum, average,
weighted subtraction, scaling — the replacement (and superset) of the old
Tk app's SpectralSumWindow, which was never ported in the first Qt pass.

All multi-spectrum operations interpolate onto a common grid spanning the
OVERLAP of the inputs (values outside a spectrum's measured range would be
extrapolated fiction), with optional per-spectrum area normalization first.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


def _common_grid(spectra_xy: Sequence[Tuple[np.ndarray, np.ndarray]], n_points: Optional[int] = None) -> np.ndarray:
    lo = max(float(np.nanmin(x)) for x, _ in spectra_xy)
    hi = min(float(np.nanmax(x)) for x, _ in spectra_xy)
    if hi <= lo:
        raise ValueError(f"Spectra have no common x-range (overlap [{lo}, {hi}]).")
    if n_points is None:
        n_points = max(len(x) for x, _ in spectra_xy)
    return np.linspace(lo, hi, int(n_points))


def _on_grid(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    order = np.argsort(x[m], kind="mergesort")
    return np.interp(grid, x[m][order], y[m][order])


def _area_normalize(grid: np.ndarray, y: np.ndarray) -> np.ndarray:
    area = np.trapz(y, grid)
    if abs(area) < 1e-30:
        return y
    return y / area * 100.0


def combine_spectra(
    spectra_xy: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    op: str = "sum",
    weights: Optional[Sequence[float]] = None,
    normalize_first: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Combine 2+ spectra on a common grid.

    op: "sum" | "average" | "subtract" (first minus the weighted sum of
    the rest — the classic reference-subtraction case). weights: optional
    per-spectrum multipliers applied after interpolation (default 1.0
    each). normalize_first: area-normalize each spectrum (area -> 100)
    before combining, the old SpectralSumWindow behavior.
    """
    if len(spectra_xy) < 2:
        raise ValueError("Need at least 2 spectra to combine.")
    op = op.lower().strip()
    if op not in ("sum", "average", "subtract"):
        raise ValueError(f"Unknown operation: {op!r} (expected 'sum', 'average', or 'subtract').")
    if weights is not None and len(weights) != len(spectra_xy):
        raise ValueError(f"weights length ({len(weights)}) must match the number of spectra ({len(spectra_xy)}).")

    grid = _common_grid(spectra_xy)
    ys = []
    for i, (x, y) in enumerate(spectra_xy):
        yi = _on_grid(x, y, grid)
        if normalize_first:
            yi = _area_normalize(grid, yi)
        if weights is not None:
            yi = yi * float(weights[i])
        ys.append(yi)

    stacked = np.vstack(ys)
    if op == "sum":
        return grid, stacked.sum(axis=0)
    if op == "average":
        return grid, stacked.mean(axis=0)
    # subtract: first minus the sum of the rest
    return grid, stacked[0] - stacked[1:].sum(axis=0)


def scale_spectrum(x: np.ndarray, y: np.ndarray, *, factor: float = 1.0, offset: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """y' = factor * y + offset (x unchanged)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    return x.copy(), float(factor) * y + float(offset)
