"""WAXS analysis: axis conversions, pseudo-Voigt multi-peak fitting,
d-spacings and crystallinity index.

Conventions:
- widths are TRUE FWHM (label them as such in any UI);
- auto peak detection excludes a 2% x-range edge margin and derives its
  prominence threshold from the interior, so edge artifacts (beamstop tail,
  detector edge) never seed peaks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

CU_KALPHA_ANG = 1.5406


# ---------------------------------------------------------------------------
# Axis conversions (q in 1/Angstrom, two-theta in degrees)
# ---------------------------------------------------------------------------


def q_to_two_theta(q: np.ndarray, wavelength_ang: float) -> np.ndarray:
    arg = np.clip(np.asarray(q, dtype=float) * wavelength_ang / (4.0 * math.pi), -1.0, 1.0)
    return np.degrees(2.0 * np.arcsin(arg))


def two_theta_to_q(two_theta_deg: np.ndarray, wavelength_ang: float) -> np.ndarray:
    theta = np.radians(np.asarray(two_theta_deg, dtype=float) / 2.0)
    return 4.0 * math.pi * np.sin(theta) / wavelength_ang


def d_spacing_from_q(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return np.where(q > 0, 2.0 * math.pi / np.clip(q, 1e-300, None), np.nan)


# ---------------------------------------------------------------------------
# Peak model
# ---------------------------------------------------------------------------


def pseudo_voigt(x: np.ndarray, amp: float, center: float, fwhm: float, eta: float) -> np.ndarray:
    """Pseudo-Voigt with amplitude = peak height, true FWHM, eta = Lorentzian fraction."""
    x = np.asarray(x, dtype=float)
    fwhm = max(float(fwhm), 1e-12)
    eta = min(max(float(eta), 0.0), 1.0)
    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    gauss = np.exp(-0.5 * ((x - center) / sigma) ** 2)
    lorentz = 1.0 / (1.0 + ((x - center) / (fwhm / 2.0)) ** 2)
    return amp * ((1.0 - eta) * gauss + eta * lorentz)


def pseudo_voigt_area(amp: float, fwhm: float, eta: float) -> float:
    """Analytic area of the pseudo-Voigt above."""
    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    gauss_area = amp * sigma * math.sqrt(2.0 * math.pi)
    lorentz_area = amp * math.pi * fwhm / 2.0
    eta = min(max(float(eta), 0.0), 1.0)
    return (1.0 - eta) * gauss_area + eta * lorentz_area


@dataclass
class PeakSpec:
    """Initial guess / configuration for one peak."""
    center: float
    amp: float
    fwhm: float
    eta: float = 0.5
    is_amorphous: bool = False
    vary_eta: bool = True


@dataclass
class FittedPeak:
    center: float
    amp: float
    fwhm: float
    eta: float
    area: float
    d_spacing: float
    is_amorphous: bool


@dataclass
class WaxsFitResult:
    peaks: List[FittedPeak]
    baseline_coeffs: List[float]
    baseline: np.ndarray
    total_fit: np.ndarray
    components: List[np.ndarray]
    r2: float
    crystallinity_index: Optional[float]
    xmin: float
    xmax: float
    npts: int
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Auto peak detection
# ---------------------------------------------------------------------------


def auto_find_peaks(
    x: np.ndarray,
    y: np.ndarray,
    max_peaks: int = 12,
    min_rel_prominence: float = 0.02,
    default_eta: float = 0.5,
) -> List[PeakSpec]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n < 10:
        return []

    # Exclude a 2% edge margin; thresholds come from the interior only.
    margin = max(3, int(0.02 * n))
    interior = slice(margin, n - margin)
    yi = y[interior]
    if yi.size < 5:
        return []
    span = float(np.nanmax(yi) - np.nanmin(yi))
    if span <= 0:
        return []
    prominence = max(min_rel_prominence * span, 1e-12)

    idx, props = find_peaks(yi, prominence=prominence)
    if idx.size == 0:
        return []

    order = np.argsort(props["prominences"])[::-1][:max_peaks]
    idx = idx[order] + margin

    specs: List[PeakSpec] = []
    dx = float(np.median(np.diff(x))) if n > 1 else 1.0
    for i in sorted(idx):
        center = float(x[i])
        # Half-height walk to estimate FWHM.
        base = float(np.nanmin(yi))
        half = base + (float(y[i]) - base) / 2.0
        li = i
        while li > 0 and y[li] > half:
            li -= 1
        ri = i
        while ri < n - 1 and y[ri] > half:
            ri += 1
        fwhm = max(float(x[ri] - x[li]), 2.0 * dx)
        specs.append(PeakSpec(center=center, amp=float(y[i] - base), fwhm=fwhm, eta=default_eta))
    return specs


# ---------------------------------------------------------------------------
# Multi-peak fit
# ---------------------------------------------------------------------------


def fit_waxs_peaks(
    x: np.ndarray,
    y: np.ndarray,
    peaks: List[PeakSpec],
    baseline_degree: int = 1,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
) -> WaxsFitResult:
    """Fit sum(pseudo-Voigt) + polynomial baseline of given degree (0-3)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lo = float(np.min(x)) if xmin is None else float(xmin)
    hi = float(np.max(x)) if xmax is None else float(xmax)
    lo, hi = sorted((lo, hi))
    mask = (x >= lo) & (x <= hi) & np.isfinite(x) & np.isfinite(y)
    xf, yf = x[mask], y[mask]
    if xf.size < 5 * max(1, len(peaks)):
        raise ValueError("Not enough points in the fit window for the requested number of peaks.")
    if not peaks:
        raise ValueError("No peaks specified.")
    deg = int(min(max(baseline_degree, 0), 3))
    nb = deg + 1

    warnings: List[str] = []
    xspan = hi - lo

    def model(xv: np.ndarray, *params: float) -> np.ndarray:
        out = np.polyval(params[:nb], xv)
        j = nb
        for spec in peaks:
            if spec.vary_eta:
                amp, center, fwhm, eta = params[j:j + 4]
                j += 4
            else:
                amp, center, fwhm = params[j:j + 3]
                eta = spec.eta
                j += 3
            out = out + pseudo_voigt(xv, amp, center, fwhm, eta)
        return out

    # Initial baseline: linear through low percentile ends.
    ymin = float(np.min(yf))
    p0: List[float] = [0.0] * deg + [ymin]
    lower: List[float] = [-np.inf] * nb
    upper: List[float] = [np.inf] * nb
    for spec in peaks:
        p0 += [max(spec.amp, 1e-12), spec.center, max(spec.fwhm, 1e-6)]
        lower += [0.0, lo, 1e-8]
        upper += [np.inf, hi, xspan]
        if spec.vary_eta:
            p0.append(min(max(spec.eta, 0.0), 1.0))
            lower.append(0.0)
            upper.append(1.0)

    popt, _ = curve_fit(model, xf, yf, p0=p0, bounds=(lower, upper), maxfev=100000)

    baseline_coeffs = [float(v) for v in popt[:nb]]
    baseline = np.polyval(baseline_coeffs, xf)
    components: List[np.ndarray] = []
    fitted: List[FittedPeak] = []
    j = nb
    for spec in peaks:
        if spec.vary_eta:
            amp, center, fwhm, eta = [float(v) for v in popt[j:j + 4]]
            j += 4
        else:
            amp, center, fwhm = [float(v) for v in popt[j:j + 3]]
            eta = float(spec.eta)
            j += 3
        comp = pseudo_voigt(xf, amp, center, fwhm, eta)
        components.append(comp)
        area = pseudo_voigt_area(amp, fwhm, eta)
        d_sp = float(2.0 * math.pi / center) if center > 0 else float("nan")
        fitted.append(FittedPeak(center=center, amp=amp, fwhm=fwhm, eta=eta, area=area,
                                 d_spacing=d_sp, is_amorphous=spec.is_amorphous))
        if fwhm > 0.9 * xspan:
            warnings.append(f"Peak at {center:.4g} has FWHM comparable to the fit window; result is unreliable.")

    total = baseline + np.sum(components, axis=0)
    ss_res = float(np.sum((yf - total) ** 2))
    ss_tot = float(np.sum((yf - np.mean(yf)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    ci = crystallinity_index(fitted)

    return WaxsFitResult(
        peaks=fitted,
        baseline_coeffs=baseline_coeffs,
        baseline=baseline,
        total_fit=total,
        components=components,
        r2=float(r2),
        crystallinity_index=ci,
        xmin=lo,
        xmax=hi,
        npts=int(xf.size),
        warnings=warnings,
    )


def crystallinity_index(peaks: List[FittedPeak]) -> Optional[float]:
    """CI = crystalline peak area / total fitted peak area.

    Requires at least one peak flagged amorphous AND one crystalline,
    otherwise the ratio is meaningless and None is returned.
    """
    if not peaks:
        return None
    cryst = sum(p.area for p in peaks if not p.is_amorphous)
    amorph = sum(p.area for p in peaks if p.is_amorphous)
    total = cryst + amorph
    if total <= 0 or amorph == 0 or cryst == 0:
        return None
    return float(cryst / total)
