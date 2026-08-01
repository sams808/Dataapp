"""Model-free SAXS analysis engines.

Ported from the validated saxs_region_workbench v25: Guinier, generalized
Porod (+ partial invariant), pseudo-Bragg peak on a power-law baseline, and
the Beaucage-style unified power-Rg fit, plus the automatic region detectors.
All functions operate on bare numpy arrays (q in 1/Angstrom).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class GuinierResult:
    qmin: float
    qmax: float
    I0: float
    Rg: float
    slope: float
    intercept: float
    r2: float
    npts: int


@dataclass
class PorodResult:
    qmin: float
    qmax: float
    A: float
    m: float
    B: float
    r2_log: float
    npts: int
    used_background: bool
    mode: str = "General"
    Kp: float = float("nan")
    qinv_min: float = float("nan")
    qinv_max: float = float("nan")
    Qp_partial: float = float("nan")
    Vp_from_I0: float = float("nan")
    Rp_sphere_from_Vp: float = float("nan")


@dataclass
class PeakResult:
    qmin: float
    qmax: float
    q0: float
    sigma: float
    fwhm: float
    amp: float
    baseline_c0: float
    baseline_c1: float
    area: float
    d_spacing: float
    xi_app: float
    baseline_at_q0: float
    rel_height: float
    window_area: float
    area_fraction: float
    r2: float
    npts: int


@dataclass
class UnifiedPowerRgResult:
    qmin: float
    qmax: float
    level: int
    scale: float
    background: float
    rg: List[float]
    power: List[float]
    B: List[float]
    G: List[float]
    r2_log: float
    chi2_red: float
    npts: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(y, kernel, mode="same")


def nearest_index(x: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(x - value)))


def slice_region(x: np.ndarray, y: np.ndarray, xmin: float, xmax: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo, hi = sorted((float(xmin), float(xmax)))
    mask = (x >= lo) & (x <= hi)
    idx = np.where(mask)[0]
    if idx.size < 5:
        raise ValueError("Selected region contains fewer than 5 points.")
    return x[mask], y[mask], mask


def robust_r2(y_true: np.ndarray, y_fit: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_fit) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0:
        return 1.0
    return 1.0 - ss_res / ss_tot


def tail_background(y: np.ndarray) -> float:
    tail = y[int(0.9 * len(y)):]
    if tail.size == 0:
        tail = y[-5:]
    return float(np.median(tail))


def clean_for_analysis(q: np.ndarray, I: np.ndarray, err: Optional[np.ndarray] = None):
    """Keep finite, q>0, I>0 points (log-based analyses need both positive)."""
    q = np.asarray(q, dtype=float)
    I = np.asarray(I, dtype=float)
    mask = np.isfinite(q) & np.isfinite(I) & (q > 0) & (I > 0)
    if err is not None:
        err = np.asarray(err, dtype=float)
        mask &= np.isfinite(err) & (err > 0)
    order = np.argsort(q[mask])
    qc = q[mask][order]
    Ic = I[mask][order]
    ec = err[mask][order] if err is not None else None
    return qc, Ic, ec


# ---------------------------------------------------------------------------
# Guinier shape conversions
# ---------------------------------------------------------------------------

GUINIER_SHAPES = [
    "Sphere",
    "Spherical shell",
    "Ellipse (2D)",
    "Ellipsoid",
    "Prism",
    "Cylinder",
]


def compute_guinier_shape_summary(Rg: float, shape: str, p1: float = 1.0, p2: float = 1.0) -> List[str]:
    """Convert Rg into shape parameters; p1/p2 are aspect ratios where needed."""
    Rg2 = float(Rg) ** 2
    p1 = max(float(p1), 1e-9)
    p2 = max(float(p2), 1e-9)
    lines = [f"Selected shape: {shape}"]
    if shape == "Sphere":
        R = math.sqrt(5.0 / 3.0) * Rg
        lines += ["Formula: Rg² = 3R² / 5", f"Equivalent radius R = {R:.6g}"]
    elif shape == "Spherical shell":
        eta = min(p1, 0.999999)
        denom = 1.0 - eta ** 5
        numer = 1.0 - eta ** 3
        factor = (3.0 / 5.0) * denom / numer if numer > 0 else float("nan")
        if np.isfinite(factor) and factor > 0:
            R1 = math.sqrt(Rg2 / factor)
            R2 = eta * R1
            lines += [
                "Formula: Rg² = (3/5) (R1⁵ - R2⁵) / (R1³ - R2³)",
                f"Assumption: η = R2/R1 = {eta:.6g}",
                f"Outer radius R1 = {R1:.6g}",
                f"Inner radius R2 = {R2:.6g}",
            ]
        else:
            lines += ["Formula: Rg² = (3/5) (R1⁵ - R2⁵) / (R1³ - R2³)", "Choose 0 < R2/R1 < 1."]
    elif shape == "Ellipse (2D)":
        ratio = p1
        a = math.sqrt(4.0 * Rg2 / (1.0 + ratio ** 2))
        b = ratio * a
        lines += [
            "Formula: Rg² = (a² + b²) / 4",
            f"Assumption: b/a = {ratio:.6g}",
            f"Semi-axis a = {a:.6g}",
            f"Semi-axis b = {b:.6g}",
        ]
    elif shape == "Ellipsoid":
        ba, ca = p1, p2
        a = math.sqrt(5.0 * Rg2 / (1.0 + ba ** 2 + ca ** 2))
        lines += [
            "Formula: Rg² = (a² + b² + c²) / 5",
            f"Assumptions: b/a = {ba:.6g}, c/a = {ca:.6g}",
            f"Semi-axis a = {a:.6g}",
            f"Semi-axis b = {ba * a:.6g}",
            f"Semi-axis c = {ca * a:.6g}",
        ]
    elif shape == "Prism":
        BA, CA = p1, p2
        A = math.sqrt(12.0 * Rg2 / (1.0 + BA ** 2 + CA ** 2))
        lines += [
            "Formula: Rg² = (A² + B² + C²) / 12",
            f"Assumptions: B/A = {BA:.6g}, C/A = {CA:.6g}",
            f"Edge A = {A:.6g}",
            f"Edge B = {BA * A:.6g}",
            f"Edge C = {CA * A:.6g}",
        ]
    elif shape == "Cylinder":
        lr = p1
        R = math.sqrt(Rg2 / (0.5 + (lr ** 2) / 12.0))
        lines += [
            "Formula: Rg² = R²/2 + l²/12",
            f"Assumption: l/R = {lr:.6g}",
            f"Radius R = {R:.6g}",
            f"Length l = {lr * R:.6g}",
        ]
    return lines


# ---------------------------------------------------------------------------
# Automatic region detection
# ---------------------------------------------------------------------------


def auto_detect_guinier_region(q: np.ndarray, I: np.ndarray) -> Tuple[float, float]:
    """Search low-q windows where ln(I) vs q^2 is most linear."""
    n = len(q)
    max_end = max(8, int(0.28 * n))
    best = None
    logI = np.log(np.clip(I, 1e-300, None))
    q2 = q ** 2

    for i0 in range(0, min(8, n - 8)):
        for i1 in range(i0 + 7, max_end):
            x = q2[i0:i1]
            y = logI[i0:i1]
            if len(x) < 7:
                continue
            m, b = np.polyfit(x, y, 1)
            if m >= 0:
                continue
            yfit = m * x + b
            r2 = robust_r2(y, yfit)
            Rg = math.sqrt(max(-3.0 * m, 0.0))
            qRg_max = float(q[i1 - 1] * Rg)
            penalty = 0.0
            if qRg_max > 1.35:
                penalty += (qRg_max - 1.35) * 0.25
            score = r2 - penalty + 0.003 * len(x)
            if best is None or score > best[0]:
                best = (score, i0, i1)

    if best is None:
        return float(q[0]), float(q[min(max_end, n - 1)])
    _, i0, i1 = best
    return float(q[i0]), float(q[i1 - 1])


def auto_detect_porod_region(q: np.ndarray, I: np.ndarray) -> Tuple[float, float]:
    """Search high-q windows where log(I) vs log(q) is most linear."""
    n = len(q)
    start_min = int(0.55 * n)
    logq = np.log(np.clip(q, 1e-300, None))
    logI = np.log(np.clip(I, 1e-300, None))
    best = None

    for i0 in range(start_min, max(start_min + 1, n - 8)):
        for i1 in range(i0 + 7, n + 1):
            x = logq[i0:i1]
            y = logI[i0:i1]
            if len(x) < 7:
                continue
            m, b = np.polyfit(x, y, 1)
            yfit = m * x + b
            r2 = robust_r2(y, yfit)
            slope_abs = abs(m)
            penalty = 0.0
            if not (1.0 <= slope_abs <= 5.2):
                penalty += 0.15 * min(abs(slope_abs - 3.5), 4.0)
            score = r2 - penalty + 0.002 * len(x)
            if best is None or score > best[0]:
                best = (score, i0, i1)

    if best is None:
        return float(q[start_min]), float(q[-1])
    _, i0, i1 = best
    return float(q[i0]), float(q[i1 - 1])


def auto_detect_peak_window(q: np.ndarray, I: np.ndarray) -> Tuple[float, float]:
    """Detect a broad pseudo-Bragg feature using a smoothed q^2 I(q) signal."""
    s = moving_average((q ** 2) * I, max(9, len(q) // 35))
    i_peak = int(np.argmax(s))
    peak_val = float(s[i_peak])
    half = 0.60 * peak_val

    left = i_peak
    while left > 0 and s[left] > half:
        left -= 1
    right = i_peak
    while right < len(q) - 1 and s[right] > half:
        right += 1

    left = max(0, left - max(3, len(q) // 80))
    right = min(len(q) - 1, right + max(3, len(q) // 80))
    if right - left < 8:
        left = max(0, i_peak - 6)
        right = min(len(q) - 1, i_peak + 6)
    return float(q[left]), float(q[right])


# ---------------------------------------------------------------------------
# Local fits
# ---------------------------------------------------------------------------


def fit_guinier(q: np.ndarray, I: np.ndarray, qmin: float, qmax: float) -> GuinierResult:
    qf, If, _ = slice_region(q, I, qmin, qmax)
    x = qf ** 2
    y = np.log(np.clip(If, 1e-300, None))
    slope, intercept = np.polyfit(x, y, 1)
    if slope >= 0:
        raise ValueError("Guinier slope is non-negative in the selected range; this is not a valid Guinier region.")
    yfit = slope * x + intercept
    I0 = math.exp(intercept)
    Rg = math.sqrt(max(-3.0 * slope, 0.0))
    return GuinierResult(
        qmin=float(qf[0]),
        qmax=float(qf[-1]),
        I0=float(I0),
        Rg=float(Rg),
        slope=float(slope),
        intercept=float(intercept),
        r2=float(robust_r2(y, yfit)),
        npts=int(len(qf)),
    )


def porod_model_with_bg(q: np.ndarray, A: float, m: float, B: float) -> np.ndarray:
    return A * np.power(q, -m) + B


def fit_porod_general(q: np.ndarray, I: np.ndarray, qmin: float, qmax: float) -> PorodResult:
    qf, If, _ = slice_region(q, I, qmin, qmax)
    x = np.log(np.clip(qf, 1e-300, None))
    y = np.log(np.clip(If, 1e-300, None))
    slope, intercept = np.polyfit(x, y, 1)
    A = math.exp(intercept)
    m = -slope
    yfit = intercept + slope * x
    r2_log = robust_r2(y, yfit)
    return PorodResult(float(qf[0]), float(qf[-1]), float(A), float(m), 0.0, float(r2_log), int(len(qf)), False, mode="General")


def fit_porod_advanced(q: np.ndarray, I: np.ndarray, qmin: float, qmax: float) -> PorodResult:
    """Strict Porod law I = Kp q^-4 (least squares through the origin)."""
    qf, If, _ = slice_region(q, I, qmin, qmax)
    x = np.power(qf, -4.0)
    denom = float(np.dot(x, x))
    if denom <= 0:
        raise ValueError("Invalid advanced Porod window.")
    Kp = float(np.dot(x, If) / denom)
    fit = Kp * x
    r2_log = robust_r2(np.log(np.clip(If, 1e-300, None)), np.log(np.clip(fit, 1e-300, None)))
    return PorodResult(float(qf[0]), float(qf[-1]), Kp, 4.0, 0.0, float(r2_log), int(len(qf)), False, mode="Advanced", Kp=Kp)


def compute_partial_porod_invariant(q: np.ndarray, I: np.ndarray, qmin: float, qmax: float) -> Tuple[float, float, float]:
    qf, If, _ = slice_region(q, I, qmin, qmax)
    integrand = qf ** 2 * If
    Qp = float(np.trapz(integrand, qf))
    return float(qf[0]), float(qf[-1]), Qp


def enrich_porod_with_invariant(result: PorodResult, Qp_tuple: Optional[Tuple[float, float, float]], I0: Optional[float]) -> PorodResult:
    if Qp_tuple is None:
        return result
    qinv_min, qinv_max, Qp = Qp_tuple
    result.qinv_min = qinv_min
    result.qinv_max = qinv_max
    result.Qp_partial = Qp
    if I0 is not None and np.isfinite(I0) and I0 > 0 and np.isfinite(Qp) and Qp > 0:
        Vp = (2.0 * math.pi / Qp) * I0
        result.Vp_from_I0 = Vp
        if Vp > 0:
            result.Rp_sphere_from_Vp = ((3.0 * Vp) / (4.0 * math.pi)) ** (1.0 / 3.0)
    return result


def fit_pseudo_bragg_peak(q: np.ndarray, I: np.ndarray, qmin: float, qmax: float) -> PeakResult:
    """Gaussian peak on a fixed power-law baseline anchored at the window ends."""
    qf, If, _ = slice_region(q, I, qmin, qmax)
    if len(qf) < 5:
        raise ValueError("Peak window is too small for a stable Gaussian fit.")

    q_left, q_right = float(qf[0]), float(qf[-1])
    I_left, I_right = float(If[0]), float(If[-1])
    x_left = math.log10(max(q_left, 1e-300))
    x_right = math.log10(max(q_right, 1e-300))
    y_left = math.log10(max(I_left, 1e-300))
    y_right = math.log10(max(I_right, 1e-300))
    dx = max(x_right - x_left, 1e-30)
    c1 = (y_right - y_left) / dx
    c0 = y_left - c1 * x_left

    def baseline_line(qv: np.ndarray) -> np.ndarray:
        q_safe = np.maximum(np.asarray(qv, dtype=float), 1e-300)
        return 10.0 ** (c0 + c1 * np.log10(q_safe))

    resid = If - baseline_line(qf)
    i_max = int(np.argmax(resid))
    q0_guess = float(qf[i_max])
    amp0 = float(max(resid[i_max], np.max(resid), 1e-12))
    sigma0 = float(max((qf[-1] - qf[0]) / 8.0, np.min(np.diff(qf)) * 2.0))

    def peak_only(qv: np.ndarray, amp: float, q0: float, sigma: float) -> np.ndarray:
        return baseline_line(qv) + amp * np.exp(-0.5 * ((np.asarray(qv, dtype=float) - q0) / sigma) ** 2)

    lower = [0.0, qf[0], max(np.min(np.diff(qf)) * 0.5, 1e-12)]
    upper = [1e30, qf[-1], max(qf[-1] - qf[0], 1e-12)]
    popt, _ = curve_fit(peak_only, qf, If, p0=[amp0, q0_guess, sigma0], bounds=(lower, upper), maxfev=40000)
    amp, q0, sigma = [float(v) for v in popt]
    fit = peak_only(qf, amp, q0, sigma)
    fwhm = 2.0 * math.sqrt(2.0 * math.log(2.0)) * sigma
    area = amp * sigma * math.sqrt(2.0 * math.pi)
    d = 2.0 * math.pi / q0 if q0 > 0 else float("nan")
    xi_app = 2.0 * math.pi / fwhm if fwhm > 0 else float("nan")
    baseline_at_q0 = float(baseline_line(np.array([q0]))[0])
    rel_height = amp / baseline_at_q0 if baseline_at_q0 > 0 else float("nan")
    window_area = float(np.trapz(If, qf))
    area_fraction = area / window_area if window_area > 0 else float("nan")
    return PeakResult(
        qmin=float(qf[0]),
        qmax=float(qf[-1]),
        q0=q0,
        sigma=sigma,
        fwhm=float(fwhm),
        amp=amp,
        baseline_c0=float(c0),
        baseline_c1=float(c1),
        area=float(area),
        d_spacing=float(d),
        xi_app=float(xi_app),
        baseline_at_q0=float(baseline_at_q0),
        rel_height=float(rel_height),
        window_area=float(window_area),
        area_fraction=float(area_fraction),
        r2=float(robust_r2(If, fit)),
        npts=int(len(qf)),
    )


# ---------------------------------------------------------------------------
# Unified power-Rg (Beaucage-style, SasView-like)
# ---------------------------------------------------------------------------


def unified_power_rg_intensity(q: np.ndarray, scale: float, background: float, level: int,
                               rg: Sequence[float], power: Sequence[float],
                               B: Sequence[float], G: Sequence[float]) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    qq = np.clip(q, 1e-300, None)
    if level <= 0:
        return scale / qq + background

    total = np.zeros_like(qq)
    root6 = math.sqrt(6.0)
    max_level = min(level, len(rg), len(power), len(B), len(G))
    for i in range(max_level):
        rgi = max(float(rg[i]), 1e-12)
        pi = float(power[i])
        Bi = float(B[i])
        Gi = float(G[i])

        guinier_term = Gi * np.exp(-(qq * rgi) ** 2 / 3.0)
        erf_term = np.clip(erf(qq * rgi / root6), 1e-15, None)
        qstar = qq / np.power(erf_term, 3.0)

        cutoff = 1.0
        if i + 1 < max_level and float(rg[i + 1]) > 0:
            cutoff = np.exp(-(qq * float(rg[i + 1])) ** 2 / 3.0)

        porod_term = Bi * cutoff * np.power(1.0 / np.clip(qstar, 1e-300, None), pi)
        total += guinier_term + porod_term

    return background + scale * total


def fit_unified_power_rg(q: np.ndarray, I: np.ndarray, err: Optional[np.ndarray], qmin: float, qmax: float,
                         level: int, initial: Dict[str, float], active: Sequence[str]) -> UnifiedPowerRgResult:
    qf, If, mask = slice_region(q, I, qmin, qmax)
    errf = err[mask] if err is not None else None
    level = int(level)

    names_all = ["scale", "background"]
    for i in range(1, level + 1):
        names_all.extend([f"rg{i}", f"power{i}", f"B{i}", f"G{i}"])

    active = [name for name in active if name in names_all]
    if not active:
        raise ValueError("No active parameters selected for unified_power_Rg fit.")

    current = {name: float(initial.get(name, 0.0)) for name in names_all}

    def unpack_full(packed: Sequence[float]) -> Dict[str, float]:
        full = current.copy()
        for name, value in zip(active, packed):
            full[name] = float(value)
        return full

    def model_from_packed(qv: np.ndarray, *packed: float) -> np.ndarray:
        full = unpack_full(packed)
        rg = [full.get(f"rg{i}", 0.0) for i in range(1, level + 1)]
        pw = [full.get(f"power{i}", 0.0) for i in range(1, level + 1)]
        bb = [full.get(f"B{i}", 0.0) for i in range(1, level + 1)]
        gg = [full.get(f"G{i}", 0.0) for i in range(1, level + 1)]
        return unified_power_rg_intensity(qv, full["scale"], full["background"], level, rg, pw, bb, gg)

    p0, lower, upper = [], [], []
    for name in active:
        val = float(current[name])
        p0.append(val)
        if name in ("scale", "background"):
            lower.append(0.0); upper.append(np.inf)
        elif name.startswith("rg"):
            lower.append(1e-8); upper.append(np.inf)
        elif name.startswith("power"):
            lower.append(0.01); upper.append(8.0)
        elif name.startswith("B") or name.startswith("G"):
            lower.append(0.0); upper.append(np.inf)
        else:
            lower.append(-np.inf); upper.append(np.inf)

    sigma = None
    absolute_sigma = False
    if errf is not None and np.all(np.isfinite(errf)) and np.all(errf > 0):
        sigma = errf
        absolute_sigma = True

    popt, _ = curve_fit(
        model_from_packed, qf, If, p0=p0, bounds=(lower, upper),
        sigma=sigma, absolute_sigma=absolute_sigma, maxfev=60000,
    )
    full = unpack_full(popt)
    rg = [full.get(f"rg{i}", 0.0) for i in range(1, level + 1)]
    pw = [full.get(f"power{i}", 0.0) for i in range(1, level + 1)]
    bb = [full.get(f"B{i}", 0.0) for i in range(1, level + 1)]
    gg = [full.get(f"G{i}", 0.0) for i in range(1, level + 1)]
    fit = unified_power_rg_intensity(qf, full["scale"], full["background"], level, rg, pw, bb, gg)
    y = np.log(np.clip(If, 1e-300, None))
    yfit = np.log(np.clip(fit, 1e-300, None))
    r2_log = robust_r2(y, yfit)
    dof = max(len(qf) - len(active), 1)
    if sigma is not None:
        chi2 = float(np.sum(((If - fit) / sigma) ** 2))
    else:
        log_resid = np.log(np.clip(If, 1e-300, None)) - np.log(np.clip(fit, 1e-300, None))
        chi2 = float(np.sum(log_resid ** 2))
    chi2_red = chi2 / dof
    return UnifiedPowerRgResult(
        qmin=float(qf[0]), qmax=float(qf[-1]), level=level,
        scale=float(full["scale"]), background=float(full["background"]),
        rg=[float(v) for v in rg], power=[float(v) for v in pw],
        B=[float(v) for v in bb], G=[float(v) for v in gg],
        r2_log=float(r2_log), chi2_red=float(chi2_red), npts=int(len(qf)),
    )
