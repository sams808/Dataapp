"""
dta_science.py — framework-agnostic DTA/DSC/TGA math: derivatives, Tg detection
(double tangent, parallel tangent, |dY| max), and the shared manual-baseline
parameter resolver that keeps interactive Compute and batch processing from
diverging (they must call resolve_baseline_params() with the same inputs and
get the same output — that's what makes "batch ignores the Manual toggle" and
similar bugs structurally impossible instead of just fixed-for-now).

No tkinter/matplotlib dependency — pure numpy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# =============================================================================
# 1) Smoothing & derivatives
# =============================================================================

def moving_average_10(y: np.ndarray) -> np.ndarray:
    """
    Minimal smoothing (fixed MA10) with reflected padding.
    IMPORTANT: In this project, smoothing is applied ONLY to derivative arrays.
    """
    y = np.asarray(y, dtype=float)
    if y.size < 3:
        return y.copy()

    w = 10
    if y.size < w:
        w = max(3, (y.size // 2) * 2 + 1)

    pad = w // 2
    ypad = np.pad(y, (pad, pad), mode="reflect")
    kernel = np.ones(w, dtype=float) / float(w)
    ys = np.convolve(ypad, kernel, mode="valid")
    return ys[: y.size]


def compute_derivative(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """dy/dx with NaN safety and preserving output length."""
    mask = np.isfinite(x) & np.isfinite(y)
    out = np.full_like(y, np.nan, dtype=float)
    if mask.sum() < 3:
        return out
    out[mask] = np.gradient(y[mask], x[mask])
    return out


def _fit_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Least-squares line fit y = m x + b."""
    m, b = np.polyfit(x, y, 1)
    return float(m), float(b)


def _line_y(m: float, b: float, x: np.ndarray) -> np.ndarray:
    return m * x + b


def _intersect_lines(m1: float, b1: float, m2: float, b2: float) -> float:
    denom = (m1 - m2)
    if abs(denom) < 1e-12:
        return float("nan")
    return float((b2 - b1) / denom)


def _root_on_grid(x: np.ndarray, f: np.ndarray, x_ref: Optional[float] = None) -> float:
    """
    Find root(s) of f(x)=0 on a sampled grid using sign changes + linear interpolation.
    If multiple roots exist, return the one closest to x_ref.
    Fallback: nearest-to-zero sample if no sign change exists.
    """
    mask = np.isfinite(x) & np.isfinite(f)
    x = x[mask]
    f = f[mask]
    if len(x) < 3:
        return float("nan")

    s = np.sign(f)
    idx = np.where(s[:-1] * s[1:] < 0)[0]

    roots: List[float] = []
    for i in idx:
        x1, x2 = x[i], x[i + 1]
        f1, f2 = f[i], f[i + 1]
        if abs(f2 - f1) < 1e-12:
            xr = float(x1)
        else:
            xr = float(x1 - f1 * (x2 - x1) / (f2 - f1))
        roots.append(xr)

    if roots:
        if x_ref is None or not np.isfinite(x_ref):
            return float(roots[0])
        r = np.array(roots, dtype=float)
        return float(r[np.nanargmin(np.abs(r - x_ref))])

    j = int(np.nanargmin(np.abs(f)))
    return float(x[j])


# =============================================================================
# 2) Tg detection core (derivative window + bounds)
# =============================================================================

@dataclass
class TransitionInfo:
    xw: np.ndarray
    yw: np.ndarray
    dy: np.ndarray
    i0: int
    x0: float
    y0: float
    m0: float
    b0: float
    i_left: int
    i_right: int
    x_left: float
    x_right: float


def _transition_from_derivative(
    x: np.ndarray,
    y: np.ndarray,
    x_min: float,
    x_max: float,
    threshold: float = 0.20,
    smooth_derivative: bool = False,
) -> TransitionInfo:
    """
    Inside [x_min, x_max]:
    - compute dy/dx from y
    - optional MA10 smoothing on dy
    - i0 = argmax |dy|
    - x_left / x_right from where |dy| drops below threshold * peak(|dy|)

    Returns TransitionInfo used to define automatic baseline regions.
    """
    if x_min > x_max:
        x_min, x_max = x_max, x_min

    w = (x >= x_min) & (x <= x_max)
    if w.sum() < 20:
        raise ValueError("Tg window too small (need ~20+ points).")

    xw = x[w]
    yw = y[w]

    dy = compute_derivative(yw, xw)
    if smooth_derivative:
        dy = moving_average_10(dy)

    i0 = int(np.nanargmax(np.abs(dy)))
    x0 = float(xw[i0])
    y0 = float(yw[i0])
    m0 = float(dy[i0])
    b0 = float(y0 - m0 * x0)

    peak = float(np.nanmax(np.abs(dy)))
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError("Derivative peak not meaningful (flat or NaNs). Adjust Tg window.")

    thr = float(max(0.01, min(0.95, threshold)) * peak)

    left_candidates = np.where(np.abs(dy[:i0]) < thr)[0]
    i_left = int(left_candidates[-1]) if len(left_candidates) else max(0, i0 - max(5, int(0.1 * len(xw))))

    right_candidates = np.where(np.abs(dy[i0:]) < thr)[0]
    i_right = int(i0 + right_candidates[0]) if len(right_candidates) else min(len(xw) - 1, i0 + max(5, int(0.1 * len(xw))))

    i_left = max(0, min(i_left, i0 - 2))
    i_right = min(len(xw) - 1, max(i_right, i0 + 2))

    return TransitionInfo(
        xw=xw, yw=yw, dy=dy, i0=i0, x0=x0, y0=y0, m0=m0, b0=b0,
        i_left=i_left, i_right=i_right,
        x_left=float(xw[i_left]), x_right=float(xw[i_right]),
    )


def _peak_index_in_range(xw: np.ndarray, dy: np.ndarray, x1: float, x2: float) -> int:
    """Index of max|dy| within [x1, x2] on the provided arrays."""
    a, b = sorted((float(x1), float(x2)))
    m = (xw >= a) & (xw <= b) & np.isfinite(dy)
    if m.sum() < 5:
        return int(np.nanargmax(np.abs(dy)))
    idx = np.where(m)[0]
    j_local = int(np.nanargmax(np.abs(dy[m])))
    return int(idx[j_local])


# =============================================================================
# 3) Tg methods
# =============================================================================

@dataclass
class TgDoubleTangentResult:
    tg: float
    m_low: float
    b_low: float
    m_slope: float
    b_slope: float
    x_ref: float
    x_left: float
    x_right: float
    low_used: Tuple[float, float]
    slope_used: Optional[Tuple[float, float]]


def compute_tg_double_tangent(
    x: np.ndarray,
    y: np.ndarray,
    x_min: float,
    x_max: float,
    threshold: float = 0.20,
    guard_frac: float = 0.00,
    smooth_derivative: bool = False,
    manual_low: Optional[Tuple[float, float]] = None,
    manual_slope: Optional[Tuple[float, float]] = None,
) -> TgDoubleTangentResult:
    """
    Double tangent Tg:
    - LOW baseline tangent: line fit on low side (AUTO) or manual range.
    - Slope tangent:
        * AUTO: tangent at max|dY/dX| from y(x)
        * MANUAL: line fit on y(x) in manual_slope range
    Tg = intersection(LOW tangent, slope tangent)
    """
    info = _transition_from_derivative(x, y, x_min, x_max, threshold, smooth_derivative)

    span = float(x_max - x_min)
    guard = float(max(0.0, guard_frac) * span)

    # LOW baseline mask
    if manual_low is not None:
        a, b = sorted((float(manual_low[0]), float(manual_low[1])))
        lmask = (info.xw >= a) & (info.xw <= b)
        if lmask.sum() < 5:
            raise ValueError("Manual LOW range has too few points.")
        low_used = (a, b)
    else:
        lmask = (info.xw >= x_min) & (info.xw <= (info.x_left - guard))
        if lmask.sum() < 5:
            lmask = (info.xw >= x_min) & (info.xw <= info.x_left)
        if lmask.sum() < 5:
            raise ValueError("Not enough points for AUTO LOW baseline. Widen window or reduce Guard.")
        low_used = (x_min, float(info.x_left))

    m_low, b_low = _fit_line(info.xw[lmask], info.yw[lmask])

    # Slope tangent
    if manual_slope is not None:
        s1, s2 = sorted((float(manual_slope[0]), float(manual_slope[1])))
        smask = (info.xw >= s1) & (info.xw <= s2)
        if smask.sum() < 5:
            raise ValueError("Manual slope range has too few points.")
        m_slope, b_slope = _fit_line(info.xw[smask], info.yw[smask])
        x_ref = float(0.5 * (s1 + s2))
        slope_used = (s1, s2)
    else:
        m_slope, b_slope = info.m0, info.b0
        x_ref = float(info.x0)
        slope_used = None

    tg = _intersect_lines(m_low, b_low, m_slope, b_slope)

    return TgDoubleTangentResult(
        tg=float(tg), m_low=m_low, b_low=b_low, m_slope=m_slope, b_slope=b_slope,
        x_ref=float(x_ref), x_left=float(info.x_left), x_right=float(info.x_right),
        low_used=low_used, slope_used=slope_used,
    )


@dataclass
class TgParallelTangentResult:
    tg: float
    m_par: float
    b_low: float
    b_high: float
    b_mid: float
    x_ref: float
    x_left: float
    x_right: float
    low_used: Tuple[float, float]
    high_used: Tuple[float, float]
    slope_used: Optional[Tuple[float, float]]
    # Actual mode used for each baseline ("range" or "point") — NOT simply an
    # echo of what was requested. When both LOW and HIGH are requested as
    # "point" (under-defined) this silently falls back to AUTO ranges
    # internally; these fields report that fallback so callers building a
    # results panel or CSV export can't misreport "point" mode alongside a
    # Tg value that was actually computed from AUTO ranges.
    low_mode: str = "range"
    high_mode: str = "range"


def compute_tg_parallel_improved(
    x: np.ndarray,
    y: np.ndarray,
    x_min: float,
    x_max: float,
    smooth_derivative: bool = False,
    manual_low_range: Optional[Tuple[float, float]] = None,
    manual_high_range: Optional[Tuple[float, float]] = None,
    manual_low_point: Optional[float] = None,
    manual_high_point: Optional[float] = None,
) -> TgParallelTangentResult:
    """
    Parallel tangents (improved manual mode):

    Each baseline (LOW / HIGH) can be defined either by:
      - a range [x1, x2]  -> linear fit on y(x) in that region
      - a single point x0 -> baseline forced PARALLEL to the other baseline,
                             passing through (x0, y(x0))

    Automatic behavior:
      - If both are ranges: fit two baselines, take the mean slope, then recompute
        intercepts with that common slope (so the two final baselines are parallel).
      - If one is range and the other is point: fit the range baseline, then create
        the parallel line through the point.
      - If neither manual spec is valid, or both are "point" (under-defined —
        two points alone can't determine two independent baselines plus a
        common slope): use AUTO regions (based on derivative peak) and proceed
        like the "both ranges" case (fit both, mean slope, parallelize). The
        returned low_mode/high_mode reflect this fallback as "range"/"range".

    Tg is defined as the intersection between the midline (average of the two parallel
    baselines) and the signal y(x). Root is chosen near the derivative peak within the window.
    """
    # Sanity
    if x_min > x_max:
        x_min, x_max = x_max, x_min

    # Build window arrays
    w = (x >= x_min) & (x <= x_max) & np.isfinite(x) & np.isfinite(y)
    if w.sum() < 20:
        raise ValueError("Tg window too small (need ~20+ points).")
    xw = x[w]
    yw = y[w]

    # Derivative peak for reference root choice
    dy = compute_derivative(yw, xw)
    if smooth_derivative:
        dy = moving_average_10(dy)
    i0 = int(np.nanargmax(np.abs(dy)))
    x_ref = float(xw[i0])

    # AUTO regions (fallback) from derivative bounds (fixed internal threshold)
    # Note: threshold/guard were removed from the UI on purpose.
    info = _transition_from_derivative(x, y, x_min, x_max, threshold=0.20, smooth_derivative=smooth_derivative)
    auto_low = (x_min, float(info.x_left))
    auto_high = (float(info.x_right), x_max)

    # Resolve manual specs (range has priority over point if both are provided)
    low_mode = None
    high_mode = None
    low_used = None
    high_used = None

    if manual_low_range is not None:
        low_mode = "range"
        low_used = tuple(sorted((float(manual_low_range[0]), float(manual_low_range[1]))))
    elif manual_low_point is not None and np.isfinite(manual_low_point):
        low_mode = "point"
        low_used = float(manual_low_point)

    if manual_high_range is not None:
        high_mode = "range"
        high_used = tuple(sorted((float(manual_high_range[0]), float(manual_high_range[1]))))
    elif manual_high_point is not None and np.isfinite(manual_high_point):
        high_mode = "point"
        high_used = float(manual_high_point)

    # If nothing usable, fall back to AUTO ranges
    if low_mode is None:
        low_mode = "range"
        low_used = auto_low
    if high_mode is None:
        high_mode = "range"
        high_used = auto_high

    def _fit_on_range(rng: Tuple[float, float]) -> Tuple[float, float, Tuple[float, float]]:
        a, b = sorted((float(rng[0]), float(rng[1])))
        m = (xw >= a) & (xw <= b)
        if m.sum() < 5:
            raise ValueError("Baseline range has too few points.")
        mfit, bfit = _fit_line(xw[m], yw[m])
        return mfit, bfit, (a, b)

    def _through_point_parallel(m_par: float, x0: float) -> float:
        # Point is defined by its X; Y is read from the signal (interpolation).
        x0 = float(x0)
        if x0 < float(xw[0]) or x0 > float(xw[-1]):
            raise ValueError("Point is outside the current Tg window.")
        y0 = float(np.interp(x0, xw, yw))
        return float(y0 - m_par * x0)

    # Compute baselines depending on cases
    if low_mode == "range" and high_mode == "range":
        mL, bL_fit, low_rng = _fit_on_range(low_used)
        mH, bH_fit, high_rng = _fit_on_range(high_used)
        m_par = float(0.5 * (mL + mH))
        b_low = float(np.nanmean(yw[(xw >= low_rng[0]) & (xw <= low_rng[1])] - m_par * xw[(xw >= low_rng[0]) & (xw <= low_rng[1])]))
        b_high = float(np.nanmean(yw[(xw >= high_rng[0]) & (xw <= high_rng[1])] - m_par * xw[(xw >= high_rng[0]) & (xw <= high_rng[1])]))
        low_used_out = low_rng
        high_used_out = high_rng
        actual_low_mode, actual_high_mode = "range", "range"

    elif low_mode == "range" and high_mode == "point":
        m_par, b_low_fit, low_rng = _fit_on_range(low_used)
        b_low = float(np.nanmean(yw[(xw >= low_rng[0]) & (xw <= low_rng[1])] - m_par * xw[(xw >= low_rng[0]) & (xw <= low_rng[1])]))
        b_high = _through_point_parallel(m_par, high_used)
        low_used_out = low_rng
        high_used_out = (float(high_used), float(high_used))
        actual_low_mode, actual_high_mode = "range", "point"

    elif low_mode == "point" and high_mode == "range":
        m_par, b_high_fit, high_rng = _fit_on_range(high_used)
        b_high = float(np.nanmean(yw[(xw >= high_rng[0]) & (xw <= high_rng[1])] - m_par * xw[(xw >= high_rng[0]) & (xw <= high_rng[1])]))
        b_low = _through_point_parallel(m_par, low_used)
        low_used_out = (float(low_used), float(low_used))
        high_used_out = high_rng
        actual_low_mode, actual_high_mode = "point", "range"

    else:
        # point + point is under-defined -> fall back to AUTO ranges.
        # This is the exact fallback the mode-misreporting bug was about:
        # actual_low_mode/actual_high_mode MUST be reset to "range" here,
        # not left as "point", since that's what's really being computed.
        mL, bL_fit, low_rng = _fit_on_range(auto_low)
        mH, bH_fit, high_rng = _fit_on_range(auto_high)
        m_par = float(0.5 * (mL + mH))
        b_low = float(np.nanmean(yw[(xw >= low_rng[0]) & (xw <= low_rng[1])] - m_par * xw[(xw >= low_rng[0]) & (xw <= low_rng[1])]))
        b_high = float(np.nanmean(yw[(xw >= high_rng[0]) & (xw <= high_rng[1])] - m_par * xw[(xw >= high_rng[0]) & (xw <= high_rng[1])]))
        low_used_out = low_rng
        high_used_out = high_rng
        actual_low_mode, actual_high_mode = "range", "range"

    b_mid = float(0.5 * (b_low + b_high))
    f = yw - (m_par * xw + b_mid)
    tg = _root_on_grid(xw, f, x_ref=x_ref)

    return TgParallelTangentResult(
        tg=float(tg),
        m_par=float(m_par),
        b_low=float(b_low), b_high=float(b_high), b_mid=float(b_mid),
        x_ref=float(x_ref),
        x_left=float(info.x_left),
        x_right=float(info.x_right),
        low_used=tuple(low_used_out),
        high_used=tuple(high_used_out),
        slope_used=None,
        low_mode=actual_low_mode,
        high_mode=actual_high_mode,
    )


def compute_tg_derivative(
    x: np.ndarray,
    y_for_derivative: np.ndarray,
    x_min: float,
    x_max: float,
    smooth_derivative: bool = False,
    restrict_range: Optional[Tuple[float, float]] = None,
) -> float:
    """
    Derivative Tg:
      Tg = argmax_x |dY/dX| within [x_min, x_max]
    If restrict_range is provided, the peak search is limited to that subrange.
    """
    if x_min > x_max:
        x_min, x_max = x_max, x_min

    w = (x >= x_min) & (x <= x_max)
    if w.sum() < 20:
        return float("nan")

    xw = x[w]
    yw = y_for_derivative[w]

    dy = compute_derivative(yw, xw)
    if smooth_derivative:
        dy = moving_average_10(dy)

    if restrict_range is not None:
        i0 = _peak_index_in_range(xw, dy, restrict_range[0], restrict_range[1])
    else:
        i0 = int(np.nanargmax(np.abs(dy)))

    return float(xw[i0])


# =============================================================================
# 4) Shared manual-baseline parameter resolver
# =============================================================================

@dataclass
class BaselineParams:
    low_range: Optional[Tuple[float, float]]
    low_point: Optional[float]
    high_range: Optional[Tuple[float, float]]
    high_point: Optional[float]
    manual_slope: Optional[Tuple[float, float]]


def resolve_baseline_params(
    *,
    manual_enabled: bool,
    low_use_point: bool,
    low_point_x: Optional[float],
    low_min: Optional[float],
    low_max: Optional[float],
    high_use_point: bool,
    high_point_x: Optional[float],
    high_min: Optional[float],
    high_max: Optional[float],
    slope_min: Optional[float] = None,
    slope_max: Optional[float] = None,
) -> BaselineParams:
    """Resolve manual baseline/slope overrides from raw widget-style inputs.

    This is the ONLY function either interactive Compute or batch processing
    should use to turn "Manual" checkbox state + typed fields into actual
    low_range/low_point/high_range/high_point/manual_slope values. Calling it
    identically from both call sites — same inputs in, same outputs out — is
    what makes "batch silently applies leftover manual values when Manual is
    unchecked" structurally impossible rather than just fixed for today: if
    manual_enabled is False, EVERYTHING resolves to None regardless of what's
    still typed in the fields.
    """
    if not manual_enabled:
        return BaselineParams(None, None, None, None, None)

    low_range = None
    low_point = None
    if low_use_point:
        if low_point_x is not None and np.isfinite(low_point_x):
            low_point = float(low_point_x)
    elif low_min is not None and low_max is not None and np.isfinite(low_min) and np.isfinite(low_max):
        low_range = tuple(sorted((float(low_min), float(low_max))))

    high_range = None
    high_point = None
    if high_use_point:
        if high_point_x is not None and np.isfinite(high_point_x):
            high_point = float(high_point_x)
    elif high_min is not None and high_max is not None and np.isfinite(high_min) and np.isfinite(high_max):
        high_range = tuple(sorted((float(high_min), float(high_max))))

    manual_slope = None
    if slope_min is not None and slope_max is not None and np.isfinite(slope_min) and np.isfinite(slope_max):
        manual_slope = tuple(sorted((float(slope_min), float(slope_max))))

    return BaselineParams(low_range, low_point, high_range, high_point, manual_slope)


# =============================================================================
# Multi-method Tg agreement (cross-technique backlog item): the tool computes
# Tg three independent ways (double tangent, parallel tangent, |dY| max);
# instead of leaving the user to eyeball-compare, quantify the spread and
# flag when the methods disagree beyond a threshold.
# =============================================================================

def tg_agreement(values: "dict[str, float | None]", threshold: float = 5.0) -> "dict":
    """Compare Tg values from multiple methods.

    values: {method_name: tg_or_None}; non-finite/None entries are ignored.
    threshold: max acceptable spread (same units as the Tg values, typically
    deg C). Returns {"n": usable-method count, "spread": max-min (nan if
    n < 2), "agree": bool or None (None when fewer than 2 methods produced
    a value — agreement is then undefined, not vacuously true), "extremes":
    (low_method, high_method) or None}.
    """
    finite = {name: float(v) for name, v in values.items()
              if v is not None and np.isfinite(v)}
    n = len(finite)
    if n < 2:
        return {"n": n, "spread": float("nan"), "agree": None, "extremes": None}
    lo_name = min(finite, key=finite.get)
    hi_name = max(finite, key=finite.get)
    spread = finite[hi_name] - finite[lo_name]
    return {
        "n": n,
        "spread": float(spread),
        "agree": bool(spread <= float(threshold)),
        "extremes": (lo_name, hi_name),
    }
