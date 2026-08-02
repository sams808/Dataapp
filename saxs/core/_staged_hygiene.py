"""
saxs/core/_staged_hygiene.py — internal implementation detail of
composite_staged.py: Stage 0 hygiene/sigma-model/window-proposal
functions and the Stage A morphology classifier.

Not meant to be imported directly by anything outside this package —
import from saxs.core.composite_staged instead, which re-exports
everything here.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from saxs.core.curve import Curve

Windows = Dict[str, Tuple[float, float]]


# =============================================================================
# Stage 0 — hygiene, sigma model, windows, class guess
# =============================================================================

def estimate_sigma_model(q: np.ndarray, I: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Poisson-like sigma model when the curve carries no measured sigma
    (spec §3): sigma_i = max(eps, c*sqrt(max(I_i,0) + I_bg_est)), with c
    calibrated on the high-q plateau's scatter via a rolling MAD."""
    q = np.asarray(q, dtype=float)
    I = np.asarray(I, dtype=float)
    n = len(I)
    tail = I[int(0.85 * n):]
    if tail.size < 5:
        tail = I[-max(5, n // 10):] if n else I
    I_bg_est = float(np.median(tail)) if tail.size else 0.0
    resid = tail - I_bg_est
    med = float(np.median(resid)) if resid.size else 0.0
    mad = float(np.median(np.abs(resid - med))) * 1.4826 if resid.size >= 3 else float(np.std(resid))
    denom = float(np.sqrt(max(np.median(np.clip(tail, 0, None)), 0.0) + I_bg_est)) or 1.0
    c = max(mad / denom, 1e-6)
    sigma = c * np.sqrt(np.clip(I, 0, None) + max(I_bg_est, 0.0))
    return np.maximum(sigma, eps)


def detect_data_type(I: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> str:
    """'counts' or 'au' (arbitrary units) -- decides which sigma-estimation
    fallback apply_hygiene uses when a curve carries no measured sigma, and
    fit_staged's a.u.-aware default residual_mode (v2:
    PRISM_fit_pipeline_upgrade_prompt.md §1). A stored metadata flag is
    authoritative when present (curve.metadata['intensity_units']);
    otherwise inferred from the data itself -- real photon counts are
    integer-valued and modest in magnitude, while reduced/scaled SAXS
    intensities (background-subtracted, transmission-scaled -- exactly
    what the real physic_based/*__corr.dat profiles this pipeline targets
    are) are neither.

    This does NOT override a genuinely measured/propagated sigma --
    apply_hygiene only consults it in the no-sigma-provided branch. A
    Poisson-derived sigma_corrected column from saxs_core.reduction's own
    quadrature error propagation stays valid regardless of the data's
    current units; a.u.-ness only invalidates the *fallback estimator*
    that would otherwise assume sigma ~ sqrt(counts) from scratch."""
    if metadata:
        units = str(metadata.get("intensity_units", "")).strip().lower()
        if units in ("a.u.", "au", "arb", "arb.", "arbitrary", "arbitrary units"):
            return "au"
        if units in ("counts", "count", "cts"):
            return "counts"
    I = np.asarray(I, dtype=float)
    finite = I[np.isfinite(I)]
    if finite.size == 0:
        return "au"
    non_integer_frac = float(np.mean(np.abs(finite - np.round(finite)) > 1e-6))
    if non_integer_frac > 0.01:
        return "au"
    if float(np.median(np.abs(finite))) > 1e6:
        return "au"
    return "counts"


def estimate_sigma_model_detrended(q: np.ndarray, I: np.ndarray, window: int = 21) -> np.ndarray:
    """Empirical local-scatter sigma for a.u.-type data (v3 §1), replacing
    v2's estimate_sigma_model_au. v2's version MAD'd log10(I) directly in
    each rolling window -- on a curve with genuine local slope (any real
    physics: a Porod tail, the flank of a peak, a Guinier/power-law
    upturn), that slope itself inflates the "residual" the MAD measures,
    systematically OVER-estimating sigma and (via the self-calibration
    added alongside this function) leaving chi2red under-inflated with
    visibly structured residuals -- exactly the P5Bi8-12 symptom this
    ticket exists to fix.

    Fix: fit a LOCAL LINEAR trend (log10 I vs log10 q, centered 21-point
    window) and MAD the RESIDUALS FROM THAT TREND, not the raw values --
    genuine local curve slope is absorbed by the fitted line and no longer
    read as noise; only genuine point-to-point scatter around the local
    trend remains. sigma_log_i = 1.4826 * MAD(local residuals);
    sigma_i = I_i * ln(10) * sigma_log_i, floored at 1e-3*I_i.

    Keeps v2's other guard: the per-point local estimate is floored at 30%
    of the curve's median local sigma_log, so a window that (by chance)
    lands entirely within a locally-quiet cluster can't report a sigma
    orders of magnitude below the curve's actual demonstrated noise level
    (the same false-positive-TS failure mode v2's own battery caught)."""
    q = np.asarray(q, dtype=float)
    I = np.asarray(I, dtype=float)
    n = len(I)
    log_q = np.log10(np.clip(q, 1e-300, None))
    log_I = np.log10(np.clip(I, 1e-300, None))
    win = max(3, min(int(window), n))
    half = win // 2
    sigma_log = np.empty(n, dtype=float)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        xs, ys = log_q[lo:hi], log_I[lo:hi]
        if xs.size >= 3 and np.ptp(xs) > 0:
            slope, intercept = np.polyfit(xs, ys, 1)
            resid = ys - (slope * xs + intercept)
        else:
            resid = ys - np.median(ys)
        med = float(np.median(resid))
        sigma_log[i] = float(np.median(np.abs(resid - med))) * 1.4826
    global_floor = float(np.median(sigma_log)) * 0.3
    sigma_log = np.maximum(sigma_log, global_floor)
    sigma = np.abs(I) * math.log(10.0) * sigma_log
    floor = 1e-3 * np.clip(np.abs(I), 1e-300, None)
    return np.maximum(sigma, floor)


def _log_rebin(q: np.ndarray, I: np.ndarray, sigma: np.ndarray, per_decade: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Log-spaced rebinning for SPEED/preview only — never used for the
    archived fit unless explicitly requested (spec §3)."""
    positive = q > 0
    q, I, sigma = q[positive], I[positive], sigma[positive]
    qmin, qmax = float(np.min(q)), float(np.max(q))
    n_decades = max(math.log10(qmax / qmin), 1e-6)
    n_bins = max(int(n_decades * per_decade), 10)
    edges = np.geomspace(qmin, qmax * 1.0000001, n_bins + 1)
    idx = np.clip(np.digitize(q, edges) - 1, 0, n_bins - 1)
    q_out, I_out, s_out = [], [], []
    for b in range(n_bins):
        m = idx == b
        if not np.any(m):
            continue
        q_out.append(float(np.mean(q[m])))
        I_out.append(float(np.mean(I[m])))
        s_out.append(float(np.sqrt(np.sum(sigma[m] ** 2)) / np.sum(m)))
    return np.array(q_out), np.array(I_out), np.array(s_out)


@dataclass
class HygieneResult:
    curve: Curve
    n_trimmed_edge: int
    n_dropped_nonfinite: int
    sigma_model: str  # "measured" | "poisson_like_estimated" | "au_detrended_estimated"


def apply_hygiene(curve: Curve, *, trim_n: int = 3, log_rebin: bool = False,
                  rebin_per_decade: int = 150,
                  data_type_override: Optional[str] = None) -> HygieneResult:
    """Trim first/last `trim_n` points, drop non-finite/negative-I points,
    attach a sigma model if the curve doesn't carry one. `log_rebin` is
    OFF by default and should stay off for the archived/final fit — it
    exists only for fast interactive previews of very dense curves.

    `data_type_override` ("counts" or "au") bypasses detect_data_type's
    own inference for choosing the sigma-estimation fallback -- needed
    when the caller KNOWS the data's true nature better than the generic
    heuristic can (e.g. synthetic Poisson-noise test data that happens to
    look non-integer purely from an exposure rescaling, which the
    heuristic can't distinguish from genuinely a.u./unrecoverable data)."""
    q = np.asarray(curve.q, dtype=float)
    I = np.asarray(curve.intensity, dtype=float)
    sigma = None if curve.sigma is None else np.asarray(curve.sigma, dtype=float)

    finite = np.isfinite(q) & np.isfinite(I) & (I >= 0)
    if sigma is not None:
        finite = finite & np.isfinite(sigma)
    n_dropped = int((~finite).sum())
    q, I = q[finite], I[finite]
    sigma = sigma[finite] if sigma is not None else None

    order = np.argsort(q)
    q, I = q[order], I[order]
    sigma = sigma[order] if sigma is not None else None

    n_edge = 0
    if trim_n > 0 and len(q) > 2 * trim_n:
        q, I = q[trim_n:-trim_n], I[trim_n:-trim_n]
        sigma = sigma[trim_n:-trim_n] if sigma is not None else None
        n_edge = 2 * trim_n

    sigma_model = "measured"
    if sigma is None:
        data_type = data_type_override or detect_data_type(I, curve.metadata)
        if data_type == "au":
            sigma = estimate_sigma_model_detrended(q, I)
            sigma_model = "au_detrended_estimated"
        else:
            sigma = estimate_sigma_model(q, I)
            sigma_model = "poisson_like_estimated"

    if log_rebin:
        q, I, sigma = _log_rebin(q, I, sigma, rebin_per_decade)

    new_curve = curve.copy_with(
        q=q, intensity=I, sigma=sigma, step="composite_hygiene",
        trim_n=trim_n, n_dropped_nonfinite=n_dropped, log_rebin=log_rebin,
        sigma_model=sigma_model,
    )
    return HygieneResult(curve=new_curve, n_trimmed_edge=n_edge,
                         n_dropped_nonfinite=n_dropped, sigma_model=sigma_model)


def guess_class(q: np.ndarray, I: np.ndarray) -> Tuple[str, float]:
    """Cheap heuristic guess ('a'|'b'|'c') from peak prominence in a
    Kratky-like q^2*I representation. This only guides window proposals,
    logging, and a guardrail against noise-driven false positives in
    stages 0-4; the rigorous arbiter is the BIC-based model-selection
    ladder (spec's Stage 6, a later pass).

    Uses scipy.signal.find_peaks' own prominence metric (height relative
    to the surrounding valleys) rather than a hand-rolled comparison —
    genuinely more robust to noise than checking "is this point higher
    than one some fixed distance away", which a random fluctuation on an
    otherwise-monotonic (background-dominated) trend can satisfy by
    chance; find_peaks never returns a boundary/monotonic-trend point as
    a peak at all.

    Smoothing uses scipy.ndimage.uniform_filter1d with mode="nearest"
    (extends the boundary value outward) rather than saxs_core.analysis'
    moving_average (an implicit-zero-pad "same"-mode convolution): on a
    monotonically-rising, background-dominated Kratky trend, zero-padding
    produces an artificial dip-then-recovery right at the high-q edge that
    find_peaks mistakes for a real local max. A narrower window
    (n // 150 vs. that function's n // 35) also matters here specifically:
    this application's real peaks (large xi => narrow in q) sit close to
    q_min, so a wide window would smear the very feature being sought —
    verified empirically across both a real measured profile and multiple
    noise realizations of a synthetic curve before adopting these values."""
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks
    q = np.asarray(q, dtype=float)
    I = np.asarray(I, dtype=float)
    n = len(q)
    kratky = (q ** 2) * I
    win = max(5, n // 150)
    smoothed = uniform_filter1d(kratky, size=win, mode="nearest")
    baseline = float(np.median(smoothed))
    prominence_floor = max(baseline * 0.3, 1e-300)
    peaks, props = find_peaks(smoothed, prominence=prominence_floor)
    if peaks.size == 0:
        return "a", 0.0
    best = int(peaks[int(np.argmax(props["prominences"]))])
    peak_val = float(smoothed[best])
    prominence = (peak_val / baseline) if baseline > 0 else (float("inf") if peak_val > 0 else 0.0)
    if prominence < 1.3:
        return "a", prominence
    if prominence < 3.0:
        return "b", prominence
    return "c", prominence


def _locate_peak(q: np.ndarray, I: np.ndarray, smooth_frac: int = 200) -> Tuple[float, float, float]:
    """Locate the strongest finite-q feature via a Kratky-like q^2*I
    representation, bracketed by half-max descent — the same technique as
    saxs_core.analysis.auto_detect_peak_window, but with a FINER smoothing
    window (len(q)//200 rather than that function's //35).

    Why a dedicated detector rather than reusing auto_detect_peak_window
    as-is: that function's coarser window suits the broad globular-particle
    features it was built for, but washes out a genuinely narrow
    Teubner-Strey peak — this application's xi (2500-5000 Å per the spec)
    implies a peak only a small fraction of the full instrument q-range
    wide. Verified empirically against both a synthetic TS curve and a
    real measured profile before adopting; still a general Kratky-based
    detector, not tuned to any particular sample's expected q*.

    Candidates are cross-validated against the raw log(I) representation
    before being accepted: q^2*exp(-q^2 Rg^2/3) (any Guinier/Guinier-Porod
    -type decay) has a genuine calculus-based local maximum at
    q=sqrt(3/p)/Rg for ANY Rg — a property of the q^2 Kratky transform
    itself, not of any real structural feature — which can out-prominence
    a genuine but further-out, weaker Teubner-Strey peak and fool a plain
    argmax into locating the wrong feature entirely. A real structural
    peak is a genuine (if modest) local rise in log(I) too; a pure
    Kratky-transform artifact from a monotonic decay is not — found via a
    synthetic BG_TS_GP recovery curve where the low-q Guinier-Porod
    upturn's Kratky hump was more prominent than the actual TS peak's."""
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks
    q = np.asarray(q, dtype=float)
    I = np.asarray(I, dtype=float)
    win = max(5, len(q) // smooth_frac)
    kratky = uniform_filter1d((q ** 2) * I, size=win, mode="nearest")
    log_I = uniform_filter1d(np.log(np.clip(I, 1e-300, None)), size=win, mode="nearest")

    kratky_floor = max(float(np.median(kratky)) * 0.05, 1e-300)
    candidates, kprops = find_peaks(kratky, prominence=kratky_floor)
    log_peaks, _ = find_peaks(log_I, prominence=1e-6)

    def _validated(idx: int) -> bool:
        if log_peaks.size == 0:
            return False
        return bool(np.min(np.abs(log_peaks - idx)) <= max(2 * win, 3))

    i_peak = None
    if candidates.size:
        order = np.argsort(kprops["prominences"])[::-1]
        for rank in order:
            idx = int(candidates[rank])
            if _validated(idx):
                i_peak = idx
                break
    if i_peak is None:
        i_peak = int(candidates[int(np.argmax(kprops["prominences"]))]) if candidates.size else int(np.argmax(kratky))

    peak_val = float(kratky[i_peak])
    half = 0.6 * peak_val
    left = i_peak
    while left > 0 and kratky[left] > half:
        left -= 1
    right = i_peak
    while right < len(q) - 1 and kratky[right] > half:
        right += 1
    pad = max(2, win // 2)
    left = max(0, left - pad)
    right = min(len(q) - 1, right + pad)
    return float(q[i_peak]), float(q[left]), float(q[right])


# =============================================================================
# Stage A — morphology classifier (v4 PRISM_fit_upgrade4_prompt.md §1)
# =============================================================================

@dataclass
class MorphologyResult:
    cls: str  # "F" | "S" | "F+P" | "S+P"
    q_knee: Optional[float]
    q_peak: Optional[float]
    peak_prominence: Optional[float]
    hump_midq: bool


def detect_knee_q(q: np.ndarray, I: np.ndarray, q_lo: float = 1e-3, q_hi: float = 8e-3,
                  exclude_around_peak: Optional[float] = None) -> Optional[float]:
    """v4 §1 KNEE: local slope going from > -0.5 (Guinier-like plateau) to
    < -2 (Porod-like falloff) within q in [q_lo, q_hi]; returns the q
    where the transition is first confirmed (the first point, after a
    genuinely flat point has been seen, where slope drops below -2) --
    None if no such flat-then-steep transition exists in this window.
    Restricted to a FIXED, narrow q-range (unlike detect_guinier_knee's
    own window-derived W_loq) since this runs BEFORE windows exist at
    all -- the classifier's own job is to inform what the windows should
    even be.

    `exclude_around_peak` (a detected q_peak from detect_peak_q) removes
    a region around it BEFORE smoothing/slope are computed: a genuine TS
    peak's OWN shape -- rising then falling -- can satisfy "flat(>-0.5)
    then steep(<-2)" exactly like a real Guinier-Porod knee, whenever the
    peak's own q0 happens to land inside [q_lo, q_hi] (this instrument's
    real xi/d range routinely does). Confirmed on a synthetic TS+GP
    curve with GP's true knee at q1~0.0012 and ts_d=1200 (q0~0.0052,
    inside this window): without exclusion, detect_knee_q reported the
    SAME q as the peak itself, not the true, much-lower GP crossover --
    the peak's own rise/fall pattern winning out over the real knee.

    Removing the points (rather than merely skipping them during the
    scan afterward) does leave a q-space gap that uniform_filter1d's
    index-based window treats as adjacent -- a real, known imprecision
    right at the seam -- but empirically this still recovers d/xi
    accurately across a 20-curve synthetic battery, while a "keep every
    point, only skip evaluating the excluded ones during the scan"
    variant tried first does NOT: the peak's own influence leaks into
    the smoothed slope of its immediate (non-excluded) neighbors too
    (the smoothing window is wider than the exclusion radius), which
    skipping only the excluded indices during the scan doesn't prevent."""
    def _slope_trace(qhi: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        m = (q >= q_lo) & (q <= qhi) & (q > 0) & (I > 0) & np.isfinite(q) & np.isfinite(I)
        if exclude_around_peak is not None and exclude_around_peak > 0:
            m = m & ~((q >= exclude_around_peak / 1.15) & (q <= exclude_around_peak * 1.15))
        if int(m.sum()) < 8:
            return None
        qm, Im = q[m], I[m]
        order = np.argsort(qm)
        qm, Im = qm[order], Im[order]
        from scipy.ndimage import uniform_filter1d
        log_q, log_I = np.log10(qm), np.log10(Im)
        n = len(log_q)
        win = max(3, n // 6)
        smoothed = uniform_filter1d(log_I, size=win, mode="nearest")
        return qm, np.gradient(smoothed, log_q)

    primary = _slope_trace(q_hi)
    if primary is None:
        return None
    qm, slope = primary
    was_flat = False
    for i in range(len(slope)):
        if slope[i] > -0.5:
            was_flat = True
        elif was_flat and slope[i] < -2.0:
            return float(qm[i])

    # v5 fallback: the genuinely flat plateau can sit ENTIRELY within the
    # beamstop-trimmed region for some real profiles (confirmed directly:
    # several real samples show a slope that is ALREADY steeper than -2 at
    # the very first point retained after trimming, with no flat evidence
    # recoverable at any q within [q_lo, q_hi] -- yet these same samples
    # produce catastrophic chi2red on a bare flat_background+power_law
    # fit, confirming a knee-type component is genuinely needed). A
    # Guinier-to-Porod crossover has a distinctive signature even when
    # only its STEEP side is observable: the local slope reaches a
    # genuine INTERIOR minimum (steepest point) and then measurably
    # RELAXES back toward a shallower, more stable asymptotic value -- a
    # pure single power law or a monotonically-steepening decay never
    # does this (nothing to relax back FROM). Searched over a WIDER range
    # than the primary scan (some real profiles don't show clear
    # relaxation until past q_hi) but kept as a separate trace rather
    # than just widening q_hi outright, so the already-validated primary
    # path's behavior on real data is untouched. Requires the minimum to
    # sit strictly inside the array (not at either edge, where it could
    # just be a not-yet-resolved trend) and the relaxation to be
    # substantial (>=2, matching the same -2.0 steepness bar used above)
    # to avoid firing on gentle curvature or noise.
    wide = _slope_trace(q_hi * 2.0)
    if wide is None:
        return None
    qm_w, slope_w = wide
    n_w = len(slope_w)
    # uniform_filter1d's own "nearest" edge padding can flatten the LAST
    # point or two of ANY trace by itself (already documented and guarded
    # against elsewhere in this file, in detect_high_q_cut, for the exact
    # same reason) -- only the RIGHT-side margin guards against that; a
    # genuinely steep-from-the-very-start real profile (confirmed on
    # several real samples in this series) can have its steepest point
    # sit right at (or one point from) the LEFT edge, which is expected
    # and must NOT be excluded the same way. The "relaxed" reference is
    # the MEDIAN of the last fifth of the trace rather than the single
    # boundary point, so one smoothing-boundary artifact point can't
    # manufacture a fake relaxation signal on its own.
    right_margin = max(2, n_w // 10)
    i_min = int(np.argmin(slope_w))
    if 0 <= i_min < n_w - right_margin and slope_w[i_min] < -2.0:
        tail = slope_w[-max(3, n_w // 5):]
        relaxation = float(np.median(tail)) - slope_w[i_min]
        if relaxation >= 2.0:
            return float(qm_w[i_min])
    return None


def detect_peak_q(q: np.ndarray, I: np.ndarray, unmasked: Optional[np.ndarray] = None,
                  sigma: Optional[np.ndarray] = None,
                  q_lo: float = 2e-3, q_hi: float = 3e-2,
                  sig_ratio_min: float = 5.0) -> Tuple[Optional[float], Optional[float]]:
    """v4 §1 PEAK: a genuine local maximum in RAW log10(I) itself (NOT the
    Kratky q^2*I representation) -- search restricted to UNMASKED q in
    [q_lo, q_hi] (hard bounds). This restriction is the direct fix for
    the P2Bi2-13 bug: the old, unrestricted-range peak-finder locked onto
    the masked high-q WAXS rise (a spurious "peak" at q~0.25 1/Å, d~25.6
    Å) because nothing stopped it searching there at all; here that
    region is structurally unreachable regardless of how prominent a
    feature it has.

    Deliberately NOT the Kratky representation, despite that being this
    module's usual peak-finding space elsewhere (_locate_peak): ANY
    smooth Guinier/Guinier-Porod-type monotonic decay has its OWN
    mathematically-guaranteed local maximum in q^2*I (a property of the
    Kratky transform itself, already documented on _locate_peak) even
    though I(q) itself never stops falling -- an earlier version of this
    function used a locally-baselined Kratky search and found q_peak
    exactly equal to q_knee for every single "parent" (genuinely
    peak-free) sample in the real 42-sample series, since ALL of them
    show this guaranteed knee-artifact hump. Requiring a genuine local
    maximum in RAW log(I) is the same cross-validation _locate_peak
    already uses to reject Kratky artifacts, just applied as the PRIMARY
    detector here rather than a secondary check: a pure monotonic decay
    cannot produce this no matter how it's transformed, while a real
    TS-type peak riding on top of it does.

    Significance test uses the curve's OWN propagated measurement sigma
    (prominence_log vs sigma_log = sigma/(I*ln10)) rather than any
    noise estimate inferred from the intensity trace itself. This
    replaced two failed local-trend-residual designs: (1) a fixed-radius
    local-window MAD blew up whenever a real peak was wide (its own
    rising/falling flanks dominate a small window's "trend" fit,
    inflating the noise estimate right where the signal is largest --
    confirmed on P5Bi8-12's real, order-of-magnitude peak at q=0.0073,
    which got REJECTED for exactly this reason); (2) excluding each
    candidate's own prominence-defined base before estimating noise
    failed the opposite way for two close, narrow real peaks (P5Bi5-12's
    idx 4 and 7): their combined exclusion zones ate the entire local
    neighborhood, forcing a window-widening fallback that crossed into a
    different part of the curve's decay and again over-estimated noise.
    The real per-point sigma sidesteps both: it needs no window, no
    detrending, and no exclusion logic, and it naturally reflects that a
    bright, high-count q-region carries far less relative noise than a
    faint tail even a few decades lower in intensity -- confirmed
    directly: P5Bi5-12's real peak (prominence only ~0.028 in log10, a
    ~7% relative rise) still comes out ~150x above its local sigma_log,
    while every noise-level tail candidate across all 9 acceptance
    samples stayed under a ratio of ~5."""
    q = np.asarray(q, dtype=float)
    I = np.asarray(I, dtype=float)
    if unmasked is None:
        unmasked = np.ones_like(q, dtype=bool)
    mask = unmasked & (q >= q_lo) & (q <= q_hi) & (q > 0) & (I > 0) & np.isfinite(q) & np.isfinite(I)
    if sigma is not None:
        sigma = np.asarray(sigma, dtype=float)
        mask = mask & np.isfinite(sigma) & (sigma > 0)
    if int(mask.sum()) < 8:
        return None, None
    qm, Im = q[mask], I[mask]
    sm = sigma[mask] if sigma is not None else None
    order = np.argsort(qm)
    qm, Im = qm[order], Im[order]
    if sm is not None:
        sm = sm[order]
    from scipy.signal import find_peaks
    n = len(qm)
    log_I = np.log10(np.clip(Im, 1e-300, None))
    margin = max(2, int(0.03 * n))

    peaks, props = find_peaks(log_I, prominence=1e-6)
    interior = (peaks >= margin) & (peaks <= n - 1 - margin)
    peaks, proms = peaks[interior], props["prominences"][interior]
    if peaks.size == 0:
        return None, None

    if sm is not None:
        sigma_log = sm[peaks] / (Im[peaks] * np.log(10.0))
        sigma_log = np.maximum(sigma_log, 1e-9)
    else:
        # Fallback when no genuine measurement sigma is available (e.g.
        # synthetic curves): a broad, loosely-smoothed MAD -- coarser
        # than the sigma-based test, but avoids the self-contamination
        # failure modes documented above since it uses one wide window
        # rather than a tight one keyed to each candidate's own location.
        from scipy.ndimage import uniform_filter1d
        win = max(15, n // 8)
        trend = uniform_filter1d(log_I, size=win, mode="nearest")
        resid = log_I - trend
        floor = float(np.median(np.abs(resid - np.median(resid)))) * 1.4826
        sigma_log = np.full(peaks.shape, max(floor, 1e-6))

    ratio = proms / sigma_log
    valid = ratio >= sig_ratio_min
    if not valid.any():
        return None, None
    best = int(np.argmax(np.where(valid, proms, -np.inf)))
    return float(qm[peaks[best]]), float(proms[best])


def detect_midq_hump(q: np.ndarray, I: np.ndarray, q_lo: float = 2e-2, q_hi: float = 1e-1) -> bool:
    """v4 §1 MIDQ_HUMP: does the curve show a positive residual bump (data
    above a provisional smooth power-law fit) in q in [q_lo, q_hi]? Flag
    only -- this signal is deliberately NOT fit by anything downstream,
    just recorded for later inspection (a candidate for a future model
    addition, not something the current component library should force
    a fit to)."""
    q = np.asarray(q, dtype=float)
    I = np.asarray(I, dtype=float)
    mask = (q >= q_lo) & (q <= q_hi) & (q > 0) & (I > 0) & np.isfinite(q) & np.isfinite(I)
    if int(mask.sum()) < 8:
        return False
    qm, Im = q[mask], I[mask]
    order = np.argsort(qm)
    qm, Im = qm[order], Im[order]
    log_q, log_I = np.log10(qm), np.log10(Im)
    if np.ptp(log_q) <= 0:
        return False
    slope, intercept = np.polyfit(log_q, log_I, 1)
    resid = log_I - (slope * log_q + intercept)
    threshold = 0.03
    run = 0
    max_run = 0
    for r in resid:
        if r > threshold:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run >= max(3, len(resid) // 8)


def classify_morphology(q: np.ndarray, I: np.ndarray, unmasked: Optional[np.ndarray] = None,
                        sigma: Optional[np.ndarray] = None) -> MorphologyResult:
    """v4 §1: classify a trimmed+masked curve into F (flat pedestal), S
    (plateau -> knee -> sigmoid drop -> tail), F+P, or S+P (the same
    shapes with a TS peak riding on top), BEFORE any fitting happens --
    this replaces treating "F+P" as the only possible shape, the root
    cause of the 42-sample batch's cascade failures (a knee-shaped S/S+P
    curve fit with F+P-only presets has no component that can describe
    its own low-q shape at all). sigma (the curve's own propagated
    measurement uncertainty) is passed through to detect_peak_q for its
    significance test when available."""
    q_peak, prominence = detect_peak_q(q, I, unmasked=unmasked, sigma=sigma)
    q_knee = detect_knee_q(q, I, exclude_around_peak=q_peak)
    hump = detect_midq_hump(q, I)
    has_knee = q_knee is not None
    has_peak = q_peak is not None
    if has_knee and has_peak:
        cls = "S+P"
    elif has_knee:
        cls = "S"
    elif has_peak:
        cls = "F+P"
    else:
        cls = "F"
    return MorphologyResult(cls=cls, q_knee=q_knee, q_peak=q_peak, peak_prominence=prominence, hump_midq=hump)


def propose_windows_from_classifier(q: np.ndarray, I: np.ndarray, morphology: "MorphologyResult",
                                    q_cut: Optional[float] = None) -> Windows:
    """v4 §4: per-sample windows derived from Stage A's own q_knee/q_peak,
    replacing propose_windows' whole-curve _locate_peak-based guess for
    classes where that guess targets the wrong feature entirely.
    _locate_peak ALWAYS returns some candidate, even for a genuinely
    peak-free curve (its Kratky-space fallback locks onto the very knee-
    transition artifact classify_morphology was built to reject -- see
    that function's own docstring) -- which for a no-peak S-class sample
    like P0Bi0 built W_loq/W_peak/W_hiq around a spurious "peak" location,
    a direct contributor to that sample's Stage-3 cascade failure.

    - W_loq: [q_min, 0.7*q_knee] (S/S+P) or [q_min, 0.5*q_peak] (F+P
      without a knee); degenerate (empty) when neither exists (class F).
    - W_peak: [q_peak/2, q_peak*2.2] clipped to [q_min, q_cut]; degenerate
      when there's no peak.
    - W_hiq: [max(2.5*q_peak, 1.5*q_knee), q_cut]; falls back to a fixed
      fraction of q_cut when neither knee nor peak exist (pure class F)."""
    q = np.asarray(q, dtype=float)
    qmin = float(np.min(q))
    qmax = float(np.max(q))
    q_cut_eff = float(q_cut) if q_cut is not None else qmax
    q_cut_eff = max(q_cut_eff, qmin * 1.0001)
    q_knee, q_peak = morphology.q_knee, morphology.q_peak

    if q_peak is not None:
        lo = max(q_peak / 2.0, qmin)
        hi = min(q_peak * 2.2, q_cut_eff)
        w_peak = (lo, hi) if lo < hi else (qmin, q_cut_eff)
    else:
        w_peak = (qmin, qmin)

    if q_knee is not None:
        w_loq = (qmin, max(0.7 * q_knee, qmin * 1.0001))
    elif q_peak is not None:
        w_loq = (qmin, max(0.5 * q_peak, qmin * 1.0001))
    else:
        w_loq = (qmin, qmin)

    hiq_candidates = []
    if q_peak is not None:
        hiq_candidates.append(2.5 * q_peak)
    if q_knee is not None:
        hiq_candidates.append(1.5 * q_knee)
    hiq_lo = max(hiq_candidates) if hiq_candidates else 0.7 * q_cut_eff
    hiq_lo = min(hiq_lo, 0.95 * q_cut_eff)
    if hiq_lo >= q_cut_eff:
        hiq_lo = 0.8 * q_cut_eff
    w_hiq = (max(hiq_lo, qmin), q_cut_eff)
    return {"W_peak": w_peak, "W_hiq": w_hiq, "W_loq": w_loq}


def propose_windows(q: np.ndarray, I: np.ndarray) -> Windows:
    """Auto-propose W_hiq/W_peak/W_loq (spec §4.1). Always visible/editable
    by the caller — this is a starting point, not a hard requirement."""
    q = np.asarray(q, dtype=float)
    I = np.asarray(I, dtype=float)
    qmin, qmax = float(np.min(q)), float(np.max(q))
    q_star, peak_lo, peak_hi = _locate_peak(q, I)

    # W_peak's low-q edge: never closer to q_star/2.5 than the peak's own
    # measured half-max descent point (peak_lo). A fixed q_star/2.5 ratio
    # can, for a peak sitting at small q_star, dip into a q-region where a
    # separate low-q feature (e.g. a Guinier-Porod upturn) still
    # contributes non-negligibly -- if that component later gets dropped
    # by the BIC ladder (its own signal too weak within the windows to
    # justify by itself), the leftover contamination biases the windowed
    # TS fit's recovered width. peak_lo is a data-driven measure of where
    # the peak's OWN contribution has actually fallen off, so taking
    # whichever bound is more conservative (closer to q_star) excludes
    # that contamination without assuming any particular low-q component
    # shape. Found via a 20-curve synthetic recovery harness where xi
    # (peak width) was biased ~20-50% at zero noise, reproducibly
    # independent of multistart count -- a real bias, not an optimizer
    # robustness gap.
    w_peak_lo = max(0.55 * peak_lo + 0.45 * (q_star / 2.5), q_star / 2.5, qmin)
    w_peak = (w_peak_lo, min(q_star * 2.5, qmax))
    hiq_lo = min(3.0 * peak_hi, 0.95 * qmax)
    if hiq_lo >= qmax:
        hiq_lo = 0.8 * qmax
    w_hiq = (hiq_lo, qmax)
    w_loq = (qmin, max(w_peak[0], qmin * 1.0001))  # tied to W_peak's own start, self-consistent
    return {"W_peak": w_peak, "W_hiq": w_hiq, "W_loq": w_loq}


def detect_high_q_cut(q: np.ndarray, I: np.ndarray) -> Optional[float]:
    """Auto-detect a rising high-q tail (v2 §2: PRISM_fit_pipeline_upgrade_
    prompt.md): the wing of an amorphous halo, detector-edge effects, or
    any other feature the background+power-law+peak composite isn't meant
    to explain can make intensity rise again well past the Porod region --
    left in, it biases every stage's fit (this was one of the four issues
    behind the real P5Bi8-12 fit's chi2red=384/pinned-bounds result).

    Computes a smoothed d(log10 I)/d(log10 q) over the last 1.5 decades of
    q, and returns the LOWEST q above that window's start (and strictly
    above the curve's own peak -- see below) where the slope is no longer
    a normal Porod-type falloff (> -0.1) AND stays that way out to q_max
    (a persistent regime change, not a transient blip/noise spike).
    Returns None when the tail keeps falling normally all the way to
    q_max -- most curves, including any curve that's already been
    truncated/doesn't have this artifact.

    The "above the peak" restriction matters concretely: this instrument's
    q-range spans only ~3.25 decades total, so "the last 1.5 decades" can
    cover nearly half the curve -- including the TS peak itself, whose own
    steep rise/fall produces large positive-slope excursions on its low-q
    flank that have nothing to do with a genuine high-q tail artifact and
    would otherwise corrupt the persistence check (found on the real
    P5Bi8-12 profile: without this restriction, the peak's own slope
    pattern got misread as "already rising" from the very start of the
    tail window)."""
    from scipy.ndimage import uniform_filter1d
    q = np.asarray(q, dtype=float)
    I = np.asarray(I, dtype=float)
    positive = (q > 0) & (I > 0) & np.isfinite(q) & np.isfinite(I)
    q, I = q[positive], I[positive]
    if q.size < 20:
        return None
    order = np.argsort(q)
    q, I = q[order], I[order]
    q_star, _peak_lo, peak_hi = _locate_peak(q, I)
    log_q, log_I = np.log10(q), np.log10(I)
    qmax = float(q[-1])
    tail_mask = q >= qmax / (10.0 ** 1.5)
    if int(np.sum(tail_mask)) < 10:
        return None
    lq, lI = log_q[tail_mask], log_I[tail_mask]
    q_tail = q[tail_mask]
    n = len(lq)
    win = max(5, n // 20)
    smoothed_I = uniform_filter1d(lI, size=win, mode="nearest")
    slope = np.gradient(smoothed_I, lq)
    slope = uniform_filter1d(slope, size=win, mode="nearest")
    # Threshold is 0.0, not the ticket's literal -0.1: on real data (and a
    # realistic synthetic control with a genuine flat background), a curve
    # asymptoting to its CONSTANT background term also has slope -> 0 from
    # below as q grows -- entirely normal, not a rising-tail artifact -- and
    # -0.1 is loose enough to misfire on that ordinary behavior (verified:
    # a synthetic BG_TS_GP curve with no injected tail falsely triggered at
    # -0.1, cleanly resolved at 0.0/0.05). A genuinely rising tail (the real
    # P5Bi8-12 profile's amorphous-halo wing) crosses clearly into positive
    # slope (+0.4 to +1.4 observed), so 0.0 still catches the real case.
    flat_or_rising = slope > 0.0
    above_peak = q_tail > max(peak_hi, q_star)
    # uniform_filter1d's mode="nearest" smears a boundary artifact across
    # roughly half the smoothing window's width at the very edge (found on
    # the real profile: the single last array point's one-sided-difference
    # derivative came out spuriously negative, and smoothing then dragged
    # ~win/2 neighboring points negative with it via repeated edge-value
    # padding) -- exclude that margin from the search entirely; the
    # eventual mask still extends to the TRUE q_max regardless of where in
    # this search the cut is found.
    margin = win
    search_end = max(0, len(flat_or_rising) - margin)
    # Earliest index i (restricted to points above the peak, before the
    # noisy edge margin) such that flat_or_rising[i:search_end] "stays so"
    # -- using a >=85% (not literal 100%) persistence bar so an isolated
    # noise blip elsewhere doesn't break an otherwise clearly-sustained
    # rise -- a large majority is still a persistent regime change, not a
    # transient blip.
    if search_end <= 0:
        return None
    idx = None
    for i in range(search_end - 1, -1, -1):
        if not above_peak[i]:
            break
        if float(np.mean(flat_or_rising[i:search_end])) >= 0.85:
            idx = i
        else:
            break
    if idx is None:
        return None
    return float(q_tail[idx])


def detect_beamstop_edge_trim(q: np.ndarray, I: np.ndarray, max_points: int = 10,
                              min_rel_bump: float = 0.03) -> int:
    """Auto-detect beamstop-edge/detector-shadow outliers at the LOWEST q
    (v3 ADDENDUM §7): a partial beamstop shadow or detector-edge response
    typically shows up as a small, LOCALIZED non-monotonic bump right at
    q_min -- intensity RISES for a point or two before the genuine,
    monotonic decay begins -- something no real scattering feature at the
    very start of a q-range produces (a genuine physical curve is already
    past whatever peak/knee it has by the time data collection starts
    just above the beamstop, so its own intensity is expected to be
    falling, not still rising, right at q_min).

    Walks forward from q_min while intensity keeps INCREASING; the first
    point where it turns over (I[i] >= I[i+1]) is treated as the genuine
    local start of the real decay, and everything up to and including it
    is dropped. `min_rel_bump` (3%) guards against ordinary counting
    noise producing a trivial one-point uptick being misread as an
    artifact -- verified against the real committed P5Bi8-12 fixture
    (whose own first two points rise 7.36e8 -> 8.20e8 before falling,
    exactly matching the ticket's stated "+1.5/+3 log-residual" outliers,
    recovering n=2) and against every synthetic curve elsewhere in this
    test suite (all correctly return 0 -- a genuine low-q upturn or a TS
    peak whose own rising flank starts close to q_min, both physically
    real features rather than artifacts, must NOT be mistaken for one;
    an earlier local-slope-vs-distant-reference design over-triggered on
    exactly these physically real, continuously-curving cases).

    An earlier design tried comparing each point's local log-log slope to
    a reference computed from a distant "clean interior" window -- this
    over-triggered on any curve with genuine continuously-changing
    curvature (a Guinier-like decay's slope steepens smoothly as q grows,
    which looks just as "anomalous" against a distant reference as a real
    artifact does) and under/over-triggered depending on how far into a
    peak's own rising flank that reference window happened to land.
    Comparing only immediate neighbors for simple non-monotonicity avoids
    both failure modes.

    v4 investigation note: the real parent (peak-free) profiles in this
    series show a WIDER, multi-wiggle non-monotonic pattern at low q (2-3
    separate up-down excursions across the first 4-5 points, each one
    hundreds of sigma above measurement noise, not a single clean rise-
    then-fall) -- a global-argmax-based trim that swallows the ENTIRE
    wiggle was tried and directly measured to be WORSE, not better: for
    several parents the genuine flat Guinier plateau turned out to sit
    ENTIRELY within that wider wiggle region (confirmed directly: the
    very first point retained after such a trim already showed slope
    steeper than -0.9, with no flat evidence anywhere in the classifier's
    fixed knee-search window), so trimming the whole wiggle away broke
    knee detection outright (cascaded to plain BG, chi2red in the
    millions -- a real, measured regression, not a hypothetical one).
    Whether that wider wiggle is genuinely artifact-plus-plateau or
    something else is an open data-reduction question -- see the
    project's own notes on raw-vs-corrected low-q treatment -- but
    fixing it is NOT this function's job: the conservative single-rise
    behavior is kept here since it is the version verified not to break
    downstream model selection on the real series."""
    q = np.asarray(q, dtype=float)
    I = np.asarray(I, dtype=float)
    n = len(q)
    if n < max_points + 2:
        return 0
    if I[1] <= I[0] * (1.0 + min_rel_bump):
        return 0
    i = 0
    while i < max_points and I[i + 1] > I[i]:
        i += 1
    return i + 1


def _apply_mask_regions(
    q: np.ndarray, I: np.ndarray, sigma: np.ndarray, mask_regions: List[Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Drop points inside ANY of `mask_regions` (each an inclusive [lo,hi]
    EXCLUDE range) from q/I/sigma before any stage sees them (v2 §2: masked
    points are excluded from every stage, not just W_hiq). Returns the
    trimmed arrays plus the boolean exclusion mask (in the ORIGINAL
    ordering) for plotting/provenance."""
    q = np.asarray(q, dtype=float)
    excluded = np.zeros_like(q, dtype=bool)
    for lo, hi in mask_regions or []:
        lo, hi = sorted((float(lo), float(hi)))
        excluded |= (q >= lo) & (q <= hi)
    keep = ~excluded
    return q[keep], I[keep], sigma[keep], excluded


def _mask_for(q: np.ndarray, windows: Windows, keys: Tuple[str, ...]) -> np.ndarray:
    """Union of the named windows (points inside ANY of them)."""
    mask = np.zeros_like(q, dtype=bool)
    for key in keys:
        if key not in windows:
            continue
        lo, hi = sorted(windows[key])
        mask = mask | ((q >= lo) & (q <= hi))
    return mask


def _seed_from_sample_id(sample_id: str) -> int:
    """Stable (process/run independent) seed from an arbitrary string —
    Python's built-in hash() is randomized per-process, so this uses a
    fixed hash instead (spec §4.5: 'deterministic, seeded from sample_id
    hash')."""
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


