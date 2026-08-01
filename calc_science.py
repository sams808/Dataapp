"""
calc_science.py — the Calculations engine (framework-agnostic): every
spectrum operation the app offers in one registry, so the Calculations
workspace can present them generically (the same param-defs pattern as
baseline_science).

Groups:
  multi-spectrum  — add / subtract / multiply / divide / average / weighted
                    sum / modulated addition (A + w(x)·B) — all on the
                    common overlap grid (spectrum_math conventions)
  single-spectrum — scale/offset, normalize (max/area/minmax), log10 / ln /
                    exp / sqrt / power / reciprocal / absolute, x-shift and
                    x-scale (calibration), crop, resample, despike,
                    derivative (Savitzky-Golay), cumulative integral,
                    smoothing (Savitzky-Golay / moving average / median)
  analysis        — linear-combination fitting (target ≈ Σ cᵢ·refᵢ, free or
                    non-negative coefficients), summary statistics
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.spectrum_math import _common_grid, _on_grid


XY = Tuple[np.ndarray, np.ndarray]


def _clean(x: np.ndarray, y: np.ndarray) -> XY:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    order = np.argsort(x[m], kind="mergesort")
    return x[m][order], y[m][order]


# =============================================================================
# Multi-spectrum operations (common overlap grid; first spectrum = A)
# =============================================================================

def multi_op(spectra_xy: Sequence[XY], op: str, *, weights: Optional[Sequence[float]] = None,
             eps: float = 1e-12) -> XY:
    """add | subtract (A − Σ rest) | multiply (∏) | divide (A / ∏ rest) |
    average | weighted_sum (needs weights). Division guards near-zero
    denominators with NaN rather than exploding."""
    if len(spectra_xy) < 2:
        raise ValueError("Need at least 2 spectra.")
    if weights is not None and len(weights) != len(spectra_xy):
        raise ValueError("weights length must match the number of spectra.")
    grid = _common_grid(spectra_xy)
    ys = [_on_grid(x, y, grid) for x, y in spectra_xy]
    if weights is not None:
        ys = [yi * float(w) for yi, w in zip(ys, weights)]
    stacked = np.vstack(ys)

    if op == "add":
        return grid, stacked.sum(axis=0)
    if op == "average":
        return grid, stacked.mean(axis=0)
    if op == "weighted_sum":
        if weights is None:
            raise ValueError("weighted_sum needs weights.")
        return grid, stacked.sum(axis=0)
    if op == "subtract":
        return grid, stacked[0] - stacked[1:].sum(axis=0)
    if op == "multiply":
        return grid, np.prod(stacked, axis=0)
    if op == "divide":
        denom = np.prod(stacked[1:], axis=0)
        out = np.where(np.abs(denom) > eps, stacked[0] / np.where(np.abs(denom) > eps, denom, 1.0), np.nan)
        return grid, out
    raise ValueError(f"Unknown multi-spectrum op: {op!r}")


def modulated_addition(a_xy: XY, b_xy: XY, *, envelope: str = "constant",
                       k: float = 1.0, x1: Optional[float] = None, x2: Optional[float] = None,
                       center: Optional[float] = None, width: Optional[float] = None) -> XY:
    """A + w(x)·B with a chosen modulation envelope:
      constant — w = k everywhere
      ramp     — w rises linearly 0→k between x1 and x2 (0 before, k after)
      gaussian — w = k·exp(−(x−center)²/2σ²) with σ = width
    The tool for e.g. blending a reference in only over part of the range."""
    grid = _common_grid([a_xy, b_xy])
    a = _on_grid(*a_xy, grid)
    b = _on_grid(*b_xy, grid)
    if envelope == "constant":
        w = np.full_like(grid, float(k))
    elif envelope == "ramp":
        lo = float(x1) if x1 is not None else float(grid[0])
        hi = float(x2) if x2 is not None else float(grid[-1])
        if hi <= lo:
            raise ValueError("ramp needs x2 > x1")
        w = np.clip((grid - lo) / (hi - lo), 0.0, 1.0) * float(k)
    elif envelope == "gaussian":
        c = float(center) if center is not None else float(grid.mean())
        s = float(width) if width else (float(grid[-1] - grid[0]) / 10.0)
        w = float(k) * np.exp(-0.5 * ((grid - c) / s) ** 2)
    else:
        raise ValueError(f"Unknown envelope: {envelope!r}")
    return grid, a + w * b


# =============================================================================
# Single-spectrum operations
# =============================================================================

def transform(x: np.ndarray, y: np.ndarray, op: str, *, factor: float = 1.0,
              offset: float = 0.0, power: float = 2.0, eps: float = 1e-12) -> XY:
    """scale_offset (factor·y + offset) | normalize_max | normalize_area
    (→100) | normalize_minmax (0..1) | log10 | ln | exp | sqrt | power |
    reciprocal | absolute | x_shift (x + offset) | x_scale (factor·x).
    log/sqrt of non-positive values become NaN, not errors."""
    x, y = _clean(x, y)
    if op == "scale_offset":
        return x, float(factor) * y + float(offset)
    if op == "normalize_max":
        m = float(np.nanmax(np.abs(y)))
        return x, y / m if m > 0 else y
    if op == "normalize_area":
        a = float(np.trapz(y, x))
        return x, y / a * 100.0 if abs(a) > eps else y
    if op == "normalize_minmax":
        lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
        return x, (y - lo) / (hi - lo) if hi > lo else y * 0.0
    if op == "log10":
        return x, np.where(y > 0, np.log10(np.where(y > 0, y, 1.0)), np.nan)
    if op == "ln":
        return x, np.where(y > 0, np.log(np.where(y > 0, y, 1.0)), np.nan)
    if op == "exp":
        return x, np.exp(np.clip(y, -700, 700))
    if op == "sqrt":
        return x, np.where(y >= 0, np.sqrt(np.abs(y)), np.nan)
    if op == "power":
        return x, np.sign(y) * np.abs(y) ** float(power)
    if op == "reciprocal":
        return x, np.where(np.abs(y) > eps, 1.0 / np.where(np.abs(y) > eps, y, 1.0), np.nan)
    if op == "absolute":
        return x, np.abs(y)
    if op == "x_shift":
        return x + float(offset), y.copy()
    if op == "x_scale":
        if abs(float(factor)) < eps:
            raise ValueError("x_scale factor must be non-zero")
        return x * float(factor), y.copy()
    raise ValueError(f"Unknown transform: {op!r}")


def crop(x: np.ndarray, y: np.ndarray, *, xmin: float, xmax: float) -> XY:
    x, y = _clean(x, y)
    if xmax <= xmin:
        raise ValueError("crop needs xmax > xmin")
    m = (x >= xmin) & (x <= xmax)
    if not np.any(m):
        raise ValueError(f"No data in [{xmin:g}, {xmax:g}]")
    return x[m], y[m]


def resample(x: np.ndarray, y: np.ndarray, *, n_points: int = 1000,
             xmin: Optional[float] = None, xmax: Optional[float] = None) -> XY:
    x, y = _clean(x, y)
    lo = float(xmin) if xmin is not None else float(x[0])
    hi = float(xmax) if xmax is not None else float(x[-1])
    if hi <= lo or int(n_points) < 2:
        raise ValueError("resample needs xmax > xmin and n_points >= 2")
    grid = np.linspace(lo, hi, int(n_points))
    return grid, np.interp(grid, x, y)


def smooth(x: np.ndarray, y: np.ndarray, *, method: str = "savgol",
           window: int = 11, polyorder: int = 3) -> XY:
    """savgol | moving_average | median"""
    from scipy.signal import savgol_filter, medfilt
    x, y = _clean(x, y)
    win = max(3, int(window) | 1)
    win = min(win, len(y) - (1 - len(y) % 2))
    if win < 3:
        return x, y
    if method == "savgol":
        return x, savgol_filter(y, window_length=win, polyorder=min(int(polyorder), win - 1), mode="interp")
    if method == "moving_average":
        kernel = np.ones(win) / win
        return x, np.convolve(y, kernel, mode="same")
    if method == "median":
        return x, medfilt(y, kernel_size=win)
    raise ValueError(f"Unknown smoothing method: {method!r}")


def despike(x: np.ndarray, y: np.ndarray, *, z: float = 6.0, window: int = 7,
            positions=None) -> XY:
    """Remove cosmic-ray spikes: points more than z robust-sigmas from the
    rolling median are replaced by it. `positions` (list of x values, e.g.
    clicked on the plot) additionally forces removal AT those spots
    regardless of the statistical test — each is replaced by a linear
    interpolation across the surrounding window."""
    from scipy.signal import medfilt
    x, y = _clean(x, y)
    win = max(3, int(window) | 1)
    med = medfilt(y, kernel_size=min(win, len(y) - (1 - len(y) % 2)))
    resid = y - med
    mad = float(np.median(np.abs(resid - np.median(resid))))
    sigma = 1.4826 * mad if mad > 1e-30 else float(np.std(resid)) or 1.0
    out = np.where(np.abs(resid) > float(z) * sigma, med, y)
    if positions:
        half = max(1, int(window) // 2)
        for pos in positions:
            i = int(np.argmin(np.abs(x - float(pos))))
            lo, hi = max(0, i - half), min(len(out) - 1, i + half)
            out[lo:hi + 1] = np.interp(x[lo:hi + 1],
                                       [x[max(0, lo - 1)], x[min(len(x) - 1, hi + 1)]],
                                       [out[max(0, lo - 1)], out[min(len(x) - 1, hi + 1)]])
    return x, out


def derivative(x: np.ndarray, y: np.ndarray, *, order: int = 1,
               window: int = 11, polyorder: int = 3) -> XY:
    """Savitzky-Golay derivative (noise-tolerant d/dx or d²/dx²)."""
    from scipy.signal import savgol_filter
    x, y = _clean(x, y)
    win = max(5, int(window) | 1)
    win = min(win, len(y) - (1 - len(y) % 2))
    dx = float(np.mean(np.diff(x))) or 1.0
    d = savgol_filter(y, window_length=win, polyorder=min(int(polyorder), win - 1),
                      deriv=int(order), delta=dx, mode="interp")
    return x, d


def cumulative_integral(x: np.ndarray, y: np.ndarray) -> XY:
    from scipy.integrate import cumulative_trapezoid
    x, y = _clean(x, y)
    return x, cumulative_trapezoid(y, x, initial=0.0)


def statistics_report(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    x, y = _clean(x, y)
    apex = int(np.argmax(y))
    return {
        "n_points": len(x), "x_min": float(x[0]), "x_max": float(x[-1]),
        "y_min": float(np.min(y)), "y_max": float(np.max(y)),
        "y_mean": float(np.mean(y)), "y_std": float(np.std(y)),
        "area": float(np.trapz(y, x)),
        "centroid_x": float(np.trapz(x * y, x) / np.trapz(y, x)) if abs(np.trapz(y, x)) > 1e-30 else float("nan"),
        "apex_x": float(x[apex]), "apex_y": float(y[apex]),
    }


# =============================================================================
# Linear-combination fitting
# =============================================================================

@dataclass
class LcfResult:
    coefficients: List[float]
    names: List[str]
    r_squared: float
    grid: np.ndarray
    y_target: np.ndarray
    y_fit: np.ndarray
    residual: np.ndarray


def linear_combination_fit(target_xy: XY, refs_xy: Sequence[XY], *,
                           ref_names: Optional[Sequence[str]] = None,
                           non_negative: bool = True) -> LcfResult:
    """Fit target ≈ Σ cᵢ·refᵢ on the common grid — the generic version of
    the XAS workspace's LCF, available for any spectra. non_negative uses
    scipy NNLS (physical mixtures); otherwise plain least squares."""
    if not refs_xy:
        raise ValueError("Need at least one reference.")
    grid = _common_grid([target_xy, *refs_xy])
    t = _on_grid(*target_xy, grid)
    R = np.vstack([_on_grid(x, y, grid) for x, y in refs_xy]).T  # (n_pts, n_refs)

    if non_negative:
        from scipy.optimize import nnls
        coefs, _ = nnls(R, t)
    else:
        coefs, *_ = np.linalg.lstsq(R, t, rcond=None)

    y_fit = R @ coefs
    resid = t - y_fit
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    names = list(ref_names) if ref_names else [f"ref{i + 1}" for i in range(len(refs_xy))]
    return LcfResult(coefficients=[float(c) for c in coefs], names=names,
                     r_squared=r2, grid=grid, y_target=t, y_fit=y_fit, residual=resid)


# =============================================================================
# Operation registry for the generic UI (name → group, n-inputs, param defs)
# param def: (key, label, default) — all free-text floats/ints in the UI
# =============================================================================

CALC_OPERATIONS: Dict[str, Dict] = {
    # multi
    "Add (Σ)": {"group": "multi", "op": "add", "params": []},
    "Subtract (A − rest)": {"group": "multi", "op": "subtract", "params": []},
    "Multiply (∏)": {"group": "multi", "op": "multiply", "params": []},
    "Divide (A ÷ rest)": {"group": "multi", "op": "divide", "params": []},
    "Average": {"group": "multi", "op": "average", "params": []},
    "Weighted sum": {"group": "multi", "op": "weighted_sum", "params": [("weights", "weights (comma-sep)", "1, 1")]},
    "Modulated addition (A + w(x)·B)": {"group": "modulated", "op": "modulated", "params": [
        ("envelope", "envelope (constant/ramp/gaussian)", "constant"), ("k", "k", "1.0"),
        ("x1", "x1 (ramp)", ""), ("x2", "x2 (ramp)", ""),
        ("center", "center (gaussian)", ""), ("width", "width (gaussian)", "")]},
    # single transforms
    "Scale / offset (a·y + b)": {"group": "transform", "op": "scale_offset", "params": [("factor", "a", "1.0"), ("offset", "b", "0.0")]},
    "Normalize to max": {"group": "transform", "op": "normalize_max", "params": []},
    "Normalize to area (→100)": {"group": "transform", "op": "normalize_area", "params": []},
    "Normalize min-max (0..1)": {"group": "transform", "op": "normalize_minmax", "params": []},
    "log10(y)": {"group": "transform", "op": "log10", "params": []},
    "ln(y)": {"group": "transform", "op": "ln", "params": []},
    "exp(y)": {"group": "transform", "op": "exp", "params": []},
    "√y": {"group": "transform", "op": "sqrt", "params": []},
    "y^n": {"group": "transform", "op": "power", "params": [("power", "n", "2.0")]},
    "1/y": {"group": "transform", "op": "reciprocal", "params": []},
    "|y|": {"group": "transform", "op": "absolute", "params": []},
    "Shift x (x + b)": {"group": "transform", "op": "x_shift", "params": [("offset", "b", "0.0")]},
    "Scale x (a·x)": {"group": "transform", "op": "x_scale", "params": [("factor", "a", "1.0")]},
    # single tools
    "Crop x-range": {"group": "crop", "op": "crop", "params": [("xmin", "x min", "0"), ("xmax", "x max", "100")]},
    "Resample": {"group": "resample", "op": "resample", "params": [("n_points", "points", "1000"), ("xmin", "x min (opt)", ""), ("xmax", "x max (opt)", "")]},
    "Smooth (Savitzky-Golay)": {"group": "smooth", "op": "savgol", "params": [("window", "window", "11"), ("polyorder", "poly order", "3")]},
    "Smooth (moving average)": {"group": "smooth", "op": "moving_average", "params": [("window", "window", "11")]},
    "Smooth (median)": {"group": "smooth", "op": "median", "params": [("window", "window", "11")]},
    "Despike (cosmic rays)": {"group": "despike", "op": "despike", "params": [("z", "z threshold", "6.0"), ("window", "window", "7"), ("positions", "spike x (click or type, comma-sep)", "")]},
    "Derivative dy/dx": {"group": "derivative", "op": "1", "params": [("window", "window", "11"), ("polyorder", "poly order", "3")]},
    "Derivative d²y/dx²": {"group": "derivative", "op": "2", "params": [("window", "window", "11"), ("polyorder", "poly order", "3")]},
    "Cumulative integral ∫y dx": {"group": "integral", "op": "cumulative", "params": []},
    # analysis
    "Linear-combination fit (target ≈ Σ cᵢ·refᵢ)": {"group": "lcf", "op": "lcf", "params": [("non_negative", "non-negative (1/0)", "1")]},
    "Statistics": {"group": "stats", "op": "stats", "params": []},
    "Cluster (KMeans)": {"group": "cluster", "op": "kmeans", "params": [("n_clusters", "clusters", "3")]},
    "Cluster (hierarchical)": {"group": "cluster", "op": "hierarchical", "params": [("n_clusters", "clusters", "3")]},
}
