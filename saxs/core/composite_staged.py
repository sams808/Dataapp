"""
saxs_core/composite_staged.py — the staged, reproducible fitting pipeline
(spec §4, stages 0-4 in this file; stages 5-6 diagnostics/model-selection
follow in a later pass). General-purpose: works on ANY 1D SAXS Curve, not
tied to a specific sample series or naming scheme — `sample_id` is just an
arbitrary string used to seed the deterministic multistart RNG and to label
the result; callers can pass a filename, a UUID, or anything else.

Stage sequence (spec §4.2):
  0  hygiene: trim, sigma model, auto-window proposal, class guess (a/b/c)
  1  fit BG (flat_background + power_law) on W_hiq only; freeze pl_B/pl_p
  2  add teubner_strey, seeded from the peak window; fit TS+bg_C on
     W_peak ∪ W_hiq (pl_B/pl_p stay frozen); a class-a guardrail (pulled
     forward from spec's own Stage 6 rule) rejects a TS fit that isn't
     actually significant, falling back to BG alone for that sample
  3  add guinier_porod for a low-q upturn; fit GP+bg_C on W_loq with
     TS/pl frozen
  4  global: release ALL parameters with widened bounds around the
     stage 1-3 best-fit values, multistart (deterministic, seeded from
     sample_id), keep the lowest reduced chi-square

The pipeline never raises on a shoulder-only or featureless profile — a
failed later stage simply falls back to the best composite assembled so
far, and the function still returns a valid FitResult.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from saxs.core.composite_fit import PRESETS, CompositeModel, build_composite
from saxs.core.composite_models import regime_label
from saxs.core.curve import Curve

Windows = Dict[str, Tuple[float, float]]

CODE_VERSION = "composite_staged-v1"


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


# =============================================================================
# FitResult (spec §4.3)
# =============================================================================

@dataclass
class FitResult:
    sample_id: str
    preset_chosen: str
    residual_mode: str
    loss: str
    windows: Windows
    sigma_model: str
    params: Dict[str, Dict[str, Any]]
    derived: Dict[str, Any]
    gof: Dict[str, float]
    flags: List[str] = field(default_factory=list)
    seeds_used: Dict[str, float] = field(default_factory=dict)
    multistart_n: int = 0
    code_version: str = CODE_VERSION
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    no_peak: bool = False
    stages: Dict[str, Any] = field(default_factory=dict)
    # v2 (PRISM_fit_pipeline_upgrade_prompt.md) additive fields:
    rms_log: Optional[float] = None
    q_cut: Optional[float] = None
    mask_regions: List[Tuple[float, float]] = field(default_factory=list)
    pruned: List[str] = field(default_factory=list)
    # v3 (PRISM_fit_upgrade2_prompt.md) additive fields:
    sigma_scale: float = 1.0
    d_ci: Optional[Tuple[float, float]] = None
    xi_ci: Optional[Tuple[float, Optional[float]]] = None  # upper=None when xi_unidentifiable
    xi_unidentifiable: bool = False
    fa_bound: Optional[float] = None
    d_peak: Optional[float] = None
    xi_peak: Optional[float] = None
    xi_unreliable: bool = False
    d_unreliable: bool = False
    # consistency-fix additive fields: model-conditional ("stat", Delta-
    # chi2<=3.841 flat) CIs alongside the headline d_ci/xi_ci (rescaled by
    # chi2red_global) -- see compute_ts_profile_likelihood_cis's docstring.
    d_ci_stat: Optional[Tuple[float, float]] = None
    xi_ci_stat: Optional[Tuple[float, Optional[float]]] = None
    # v4 (PRISM_fit_upgrade4_prompt.md) additive fields: Stage A's own
    # morphology classification and the systematic-error floor.
    morphology_cls: Optional[str] = None  # "F" | "S" | "F+P" | "S+P"
    q_knee: Optional[float] = None
    q_peak: Optional[float] = None
    hump_midq: bool = False
    f_systematic: float = 0.0

    def to_json(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id, "preset_chosen": self.preset_chosen,
            "residual_mode": self.residual_mode, "loss": self.loss,
            "windows": {k: list(v) for k, v in self.windows.items()},
            "sigma_model": self.sigma_model, "params": self.params,
            "derived": self.derived, "gof": self.gof, "flags": list(self.flags),
            "seeds_used": self.seeds_used, "multistart_n": self.multistart_n,
            "code_version": self.code_version, "timestamp": self.timestamp,
            "no_peak": self.no_peak, "stages": self.stages,
            "rms_log": self.rms_log, "q_cut": self.q_cut,
            "mask_regions": [list(r) for r in self.mask_regions], "pruned": list(self.pruned),
            "sigma_scale": self.sigma_scale,
            "d_ci": list(self.d_ci) if self.d_ci is not None else None,
            "xi_ci": list(self.xi_ci) if self.xi_ci is not None else None,
            "xi_unidentifiable": self.xi_unidentifiable, "fa_bound": self.fa_bound,
            "d_peak": self.d_peak, "xi_peak": self.xi_peak,
            "xi_unreliable": self.xi_unreliable, "d_unreliable": self.d_unreliable,
            "d_ci_stat": list(self.d_ci_stat) if self.d_ci_stat is not None else None,
            "xi_ci_stat": list(self.xi_ci_stat) if self.xi_ci_stat is not None else None,
            "morphology_cls": self.morphology_cls, "q_knee": self.q_knee, "q_peak": self.q_peak,
            "hump_midq": self.hump_midq, "f_systematic": self.f_systematic,
        }

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> "FitResult":
        payload = dict(payload)
        payload["windows"] = {k: tuple(v) for k, v in payload.get("windows", {}).items()}
        if "mask_regions" in payload:
            payload["mask_regions"] = [tuple(r) for r in payload["mask_regions"]]
        if payload.get("d_ci") is not None:
            payload["d_ci"] = tuple(payload["d_ci"])
        if payload.get("xi_ci") is not None:
            payload["xi_ci"] = tuple(payload["xi_ci"])
        if payload.get("d_ci_stat") is not None:
            payload["d_ci_stat"] = tuple(payload["d_ci_stat"])
        if payload.get("xi_ci_stat") is not None:
            payload["xi_ci_stat"] = tuple(payload["xi_ci_stat"])
        return cls(**payload)

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, indent=2, default=str)

    @classmethod
    def load_json(cls, path: str) -> "FitResult":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_json(json.load(f))

    def to_csv_row(self) -> Dict[str, Any]:
        """One flat row for the batch CSV (Phase 5; v2 adds rms_log,
        at_bounds count, q_cut, pruned; v3 adds sigma_scale, d/xi CI-or-
        bound columns, and the xi/d reliability flags)."""
        row: Dict[str, Any] = {
            "sample_id": self.sample_id, "preset_chosen": self.preset_chosen,
            "residual_mode": self.residual_mode, "loss": self.loss,
            "sigma_model": self.sigma_model, "no_peak": self.no_peak,
            "flags": ";".join(self.flags), "code_version": self.code_version,
            "timestamp": self.timestamp, "rms_log": self.rms_log, "q_cut": self.q_cut,
            "pruned": ";".join(self.pruned),
            "at_bounds": sum(1 for f in self.flags if f.startswith("at_bound:")),
            "sigma_scale": self.sigma_scale,
            "d_ci_lo": self.d_ci[0] if self.d_ci is not None else None,
            "d_ci_hi": self.d_ci[1] if self.d_ci is not None else None,
            "xi_ci_lo": self.xi_ci[0] if self.xi_ci is not None else None,
            "xi_ci_hi": self.xi_ci[1] if self.xi_ci is not None else None,
            "d_ci_stat_lo": self.d_ci_stat[0] if self.d_ci_stat is not None else None,
            "d_ci_stat_hi": self.d_ci_stat[1] if self.d_ci_stat is not None else None,
            "xi_ci_stat_lo": self.xi_ci_stat[0] if self.xi_ci_stat is not None else None,
            "xi_ci_stat_hi": self.xi_ci_stat[1] if self.xi_ci_stat is not None else None,
            "xi_unidentifiable": self.xi_unidentifiable, "fa_bound": self.fa_bound,
            "d_peak": self.d_peak, "xi_peak": self.xi_peak,
            "xi_unreliable": self.xi_unreliable, "d_unreliable": self.d_unreliable,
            "class": self.morphology_cls, "q_knee": self.q_knee, "q_peak": self.q_peak,
            "hump_midq": self.hump_midq, "f": self.f_systematic,
        }
        row.update({f"gof_{k}": v for k, v in self.gof.items()})
        row.update({f"derived_{k}": v for k, v in self.derived.items() if not isinstance(v, dict)})
        return row


def _params_to_dict(lmfit_params: Any, chi2red: Optional[float] = None) -> Dict[str, Dict[str, Any]]:
    """v3 consistency fix: `chi2red` is accepted (kept for call-site/API
    compatibility) but no longer used to rescale stderr. v2 §1 added a
    manual `stderr *= sqrt(chi2red)` here on the (incorrect) assumption
    that lmfit's own covariance-based stderr was NOT chi2red-corrected --
    but every `CompositeModel.fit()` call in this codebase uses lmfit's
    default `scale_covar=True`, which ALREADY multiplies the covariance
    matrix (hence stderr) by redchi before reporting it. Doing it again
    here silently inflated every reported stderr by an EXTRA factor of
    sqrt(chi2red) on top of lmfit's own correction (confirmed directly:
    a controlled lmfit reproduction with scale_covar=True vs False shows
    the stderr ratio is exactly sqrt(redchi) -- the v2 code then applied
    that same factor a second time). Caught via a real-data cross-check:
    the real P5Bi8-12 fit's own ts_xi stderr (2926 Å) was ~sqrt(chi2red)
    times larger than the properly-scaled profile-likelihood CI half-
    width would predict. `p.stderr` is now reported as lmfit gives it --
    already the statistically standard, correctly-scaled value."""
    out = {}
    for name in lmfit_params:
        p = lmfit_params[name]
        stderr = None if p.stderr is None else float(p.stderr)
        out[name] = {
            "value": float(p.value), "stderr": stderr,
            "min": float(p.min), "max": float(p.max), "vary": bool(p.vary),
        }
    return out


def _build_derived(model: CompositeModel, result_params: Any,
                   pruned: Optional[List[str]] = None) -> Dict[str, Any]:
    """Per-component derived() (nested) PLUS the spec's flat, named
    top-level aliases (d, xi, fa, q_max, a2, c1, c2, Rg, p_pl, p_gp, p_pl2,
    xiL_oz) — whichever of those are actually present in this composite.

    `pruned` (v3 §5) is a list of COMPONENT NAMES (e.g. "power_law")
    Stage 1's prune-and-refit froze to a negligible contribution -- their
    values are never displayed here, structurally, not just filtered at
    the report layer (they're still nominally present in the model, but
    carry no real physical information once pruned: the previous behavior
    of showing e.g. p_pl despite "power_law" being pruned was confusing,
    reported as a real bug in v3 §5)."""
    pruned_set = set(pruned or [])
    nested_all = model.derived(result_params)
    prefixes = {prefix.rstrip("_") or comp.name: (prefix, comp.name) for prefix, comp in model.components}
    nested = {key: nested_all[key] for key, (_, name) in prefixes.items()
             if name not in pruned_set and key in nested_all}
    flat: Dict[str, Any] = {"components": nested}
    if "ts" in prefixes and prefixes["ts"][1] not in pruned_set:
        prefix, _ = prefixes["ts"]
        flat["d"] = result_params[prefix + "d"].value
        flat["xi"] = result_params[prefix + "xi"].value
        ts_derived = nested.get("ts", {})
        flat["fa"] = ts_derived.get("fa")
        flat["q_max"] = ts_derived.get("q_max")
        flat["a2"] = ts_derived.get("a2")
        flat["c1"] = ts_derived.get("c1")
        flat["c2"] = ts_derived.get("c2")
    if "pl" in prefixes and prefixes["pl"][1] not in pruned_set:
        prefix, _ = prefixes["pl"]
        flat["p_pl"] = result_params[prefix + "p"].value
    if "gp" in prefixes and prefixes["gp"][1] not in pruned_set:
        prefix, _ = prefixes["gp"]
        flat["Rg"] = result_params[prefix + "Rg"].value
        flat["p_gp"] = result_params[prefix + "p"].value
    if "bu" in prefixes and prefixes["bu"][1] not in pruned_set:
        # v5: beaucage_unified's own flat aliases, mirroring guinier_porod's
        # -- Rg/p share the same physical meaning (overall fluctuation size,
        # high-q Porod-type exponent) as GP's, just via the additive
        # Guinier+Porod form instead of a matched-crossover one. Falls back
        # to "Rg"/"p_gp" naming when gp itself isn't ALSO present (the two
        # are mutually exclusive class-anchored alternatives in the
        # ladder), so a report/CSV consumer doesn't need to know which of
        # the two knee-level components actually won.
        prefix, _ = prefixes["bu"]
        if "Rg" not in flat:
            flat["Rg"] = result_params[prefix + "Rg"].value
        if "p_gp" not in flat:
            flat["p_gp"] = result_params[prefix + "p"].value
        flat["B_bu"] = result_params[prefix + "B"].value
    if "pl2" in prefixes and prefixes["pl2"][1] not in pruned_set:
        prefix, _ = prefixes["pl2"]
        p2_value = result_params[prefix + "p2"].value
        flat["p_pl2"] = p2_value
        flat["pl2_regime"] = regime_label(p2_value)
    if "oz" in prefixes and prefixes["oz"][1] not in pruned_set:
        prefix, _ = prefixes["oz"]
        flat["xiL_oz"] = result_params[prefix + "xiL"].value
    return flat


def _rms_log10(model: CompositeModel, result_params: Any, q: np.ndarray, I: np.ndarray) -> float:
    """RMS of log10(model) - log10(data) (v2 §1: reported in EVERY fit
    regardless of residual_mode, so weighted-linear and log10-mode fits
    stay comparable on a common scale)."""
    q = np.asarray(q, dtype=float)
    I = np.asarray(I, dtype=float)
    if q.size == 0:
        return float("nan")
    total = model.eval(q, result_params)
    resid = np.log10(np.clip(total, 1e-300, None)) - np.log10(np.clip(I, 1e-300, None))
    return float(np.sqrt(np.mean(resid ** 2)))


def _gof(model: CompositeModel, result: Any, q: np.ndarray, I: np.ndarray) -> Dict[str, float]:
    return {
        "chi2red": float(result.redchi), "aic": float(result.aic), "bic": float(result.bic),
        "n_points": int(result.ndata), "rms_log": _rms_log10(model, result.params, q, I),
    }


# =============================================================================
# Stages 1-4
# =============================================================================

_STAGE1_PL_P_BOUNDS = (2.5, 4.3)  # v2 §4: tightened from power_law's own [1,4.5] default,
# specifically for Stage 1's BG role -- p<2.5 isn't a physically expected
# Porod-regime exponent for a genuine background/high-q tail; letting the
# optimizer wander there is usually degenerate with bg_C, not a real fit.


def _bg_c_plateau_bounds(q: np.ndarray, I: np.ndarray) -> Tuple[float, float]:
    """v4 §3: bg_C bounds = [0.2x, 5x] the median of I over the last
    half-decade of q in the data actually passed in (the W_hiq-masked
    array, i.e. the high-q plateau region up to q_cut). Fixes the
    confirmed P0Bi0 cascade bug: an UNBOUNDED flat_background (its own
    component default is [0, inf)) let the Stage-1 optimizer land on
    bg_C=1e-12 -- 14 decades below the actual data -- which every later
    stage then inherited as its background floor. A data-driven bound
    tied to the plateau's own scale makes that value structurally
    unreachable regardless of what local minimum the optimizer wanders
    into."""
    if q.size == 0:
        return (0.0, np.inf)
    qmax = float(np.max(q))
    half_decade_lo = qmax / (10.0 ** 0.5)
    sel = q >= half_decade_lo
    if int(np.sum(sel)) < 3:
        sel = np.ones_like(q, dtype=bool)
    med = float(np.median(I[sel]))
    if not math.isfinite(med) or med <= 0:
        return (0.0, np.inf)
    return (0.2 * med, 5.0 * med)


def _stage1_bg(q: np.ndarray, I: np.ndarray, sigma: np.ndarray, windows: Windows,
               sample_id: str = "stage1", n_tries: int = 3,
               residual_mode: str = "weighted_linear") -> Dict[str, Any]:
    """A single seeded fit here is fragile: when W_hiq's true power-law
    contribution is negligible (common — Porod tails are often tiny
    relative to a flat background), the log-log-regression seed for
    pl_B/pl_p is essentially fit to noise and can land the optimizer in a
    bad bg_C/pl_B/pl_p local minimum — one that Stage 2 then FREEZES
    pl_B/pl_p into, propagating a bad background estimate through the rest
    of the pipeline (discovered via the Phase 6 synthetic harness: some
    curves' d recovery failed entirely at even the best noise level,
    traced to exactly this). A small local multistart around the seed
    (deterministic, keyed off sample_id) is a targeted, low-cost fix.

    v2 §4 adds two more guards: pl_p's bounds are tightened to
    _STAGE1_PL_P_BOUNDS for this fit specifically (not power_law's own
    default), and if the optimizer still pins pl_p at either bound
    (degenerate with bg_C — no genuine Porod tail in W_hiq), the fit is
    retried with power_law effectively pruned: pl_B/pl_p FROZEN at values
    that make its contribution negligible (1e-12 * q^-4 is astronomically
    small for any q>1e-4) rather than removing the component structurally
    — Stage 2/3/4's existing code all assumes pl_B/pl_p exist, and this
    keeps that contract intact while achieving the same scientific outcome
    ("no power-law term"). Recorded in the returned dict's "pruned" list.

    v3 §8.3: this fit IS the "featureless high-q plateau" fit the sigma
    self-calibration factor `s` is derived from (fit_staged computes
    chi2red from this exact result before deciding whether to rescale
    sigma) -- keep this stage's own weighting/masking behavior stable
    since the calibration's meaning depends on it.
    """
    model = build_composite(["flat_background", "power_law"])
    mask = _mask_for(q, windows, ("W_hiq",))
    if int(mask.sum()) < 5:
        mask = np.ones_like(q, dtype=bool)  # degenerate window: fall back to everything
    seeds = model.seed(q[mask], I[mask], windows)
    bg_c_bounds = _bg_c_plateau_bounds(q[mask], I[mask])
    bound_overrides = {"pl_p": _STAGE1_PL_P_BOUNDS, "bg_C": bg_c_bounds}
    rng = np.random.default_rng(_seed_from_sample_id(sample_id + ":stage1"))
    best_result = None
    for _ in range(max(n_tries, 1)):
        perturbed = {name: v * math.exp(rng.uniform(-0.3, 0.3)) if v > 0 else v for name, v in seeds.items()}
        params = model.to_lmfit_parameters(seed_values=perturbed, bound_overrides=bound_overrides)
        try:
            result = model.fit(q[mask], I[mask], sigma=sigma[mask], params=params, residual_mode=residual_mode)
        except Exception:
            continue
        if best_result is None or result.redchi < best_result.redchi:
            best_result = result
    if best_result is None:
        params = model.to_lmfit_parameters(seed_values=seeds, bound_overrides=bound_overrides)
        best_result = model.fit(q[mask], I[mask], sigma=sigma[mask], params=params, residual_mode=residual_mode)

    pruned: List[str] = []
    pl_p = best_result.params["pl_p"]
    span = pl_p.max - pl_p.min
    if span > 0 and (abs(pl_p.value - pl_p.min) <= 0.01 * span or abs(pl_p.value - pl_p.max) <= 0.01 * span):
        retry_params = model.to_lmfit_parameters(
            seed_values={"bg_C": best_result.params["bg_C"].value, "pl_B": 1e-12, "pl_p": 4.0},
            bound_overrides=bound_overrides)
        model.fix(retry_params, "pl_B", 1e-12)
        model.fix(retry_params, "pl_p", 4.0)
        try:
            retry_result = model.fit(q[mask], I[mask], sigma=sigma[mask], params=retry_params,
                                     residual_mode=residual_mode)
            best_result = retry_result
            pruned.append("power_law")
        except Exception:
            pass
    return {"model": model, "result": best_result, "mask": mask, "seeds": seeds, "pruned": pruned,
           "bg_c_bounds": bg_c_bounds}


def _stage2_add_ts(q: np.ndarray, I: np.ndarray, sigma: np.ndarray, windows: Windows,
                   stage1: Dict[str, Any], residual_mode: str = "weighted_linear",
                   bg_c_bounds: Optional[Tuple[float, float]] = None) -> Optional[Dict[str, Any]]:
    bg_params = stage1["result"].params
    model = build_composite(["flat_background", "power_law", "teubner_strey"])
    mask = _mask_for(q, windows, ("W_peak", "W_hiq"))
    if int(mask.sum()) < 8:
        return None
    peak_mask = _mask_for(q, windows, ("W_peak",))
    q_win, I_win = (q[peak_mask], I[peak_mask]) if np.any(peak_mask) else (q[mask], I[mask])
    ts_seed = model.components[-1][1].seed(q_win, I_win, windows)
    # S seeded as I(q*) minus the Stage-1 background+power-law level there (spec §4.2)
    bg_at_qstar = (bg_params["bg_C"].value
                   + bg_params["pl_B"].value * max(ts_seed["d"] and (2 * math.pi / ts_seed["d"]), 1e-8) ** (-bg_params["pl_p"].value))
    ts_seed["S"] = max(ts_seed["S"] - bg_at_qstar, ts_seed["S"] * 0.1)

    seed_values = {"bg_C": bg_params["bg_C"].value, "pl_B": bg_params["pl_B"].value,
                  "pl_p": bg_params["pl_p"].value, **{f"ts_{k}": v for k, v in ts_seed.items()}}
    bound_overrides = {"bg_C": bg_c_bounds} if bg_c_bounds is not None else None
    params = model.to_lmfit_parameters(seed_values=seed_values, bound_overrides=bound_overrides)
    model.fix(params, "pl_B", bg_params["pl_B"].value)
    model.fix(params, "pl_p", bg_params["pl_p"].value)
    # narrow ts_d's bounds to the active window (spec §1.7): d in [2pi/q_hi, 2pi/q_lo]
    q_lo_win, q_hi_win = float(np.min(q[mask])), float(np.max(q[mask]))
    d_lo, d_hi = sorted([2 * math.pi / q_hi_win, 2 * math.pi / max(q_lo_win, 1e-8)])
    params["ts_d"].set(min=max(d_lo, 10.0), max=min(d_hi, 1e6))
    if not (params["ts_d"].min < params["ts_d"].value < params["ts_d"].max):
        params["ts_d"].set(value=(params["ts_d"].min + params["ts_d"].max) / 2.0)

    result = model.fit(q[mask], I[mask], sigma=sigma[mask], params=params, residual_mode=residual_mode)
    return {"model": model, "result": result, "mask": mask, "seeds": seed_values}


def ts_guardrail_ok(result: Any, sigma_local: np.ndarray, windows: Windows) -> Tuple[bool, str]:
    """Pulled forward from spec's own Stage 6 class-a guardrail: refuse a
    TS fit whose height isn't actually significant, or whose peak sits
    outside the peak window entirely — used in Stage 2 (Phase 3) AND
    applied to every TS-containing ladder candidate (v2 §3/§4's
    select_best_preset) so a nonsense peak never wins purely on BIC.

    The significance bar is 8*sigma_typ, not the more conventional 3:
    window position AND width are themselves searched (auto-proposed
    from the same noisy data, then multistart-refined), a real
    "look-elsewhere effect" — the same reason particle-physics discovery
    claims use 5-sigma rather than 3 when scanning a mass range. Found
    necessary via the 20-curve peak-free synthetic battery: even at 8x,
    this alone doesn't fully protect against a badly UNDERESTIMATED
    sigma_typ (a separate, real bug in the test's own noise-generation
    calibration, fixed at the source — see test_composite_synthetic.py's
    _peak_free_curve) — this threshold is a legitimate independent
    hardening on top of that fix, not a substitute for it."""
    S = result.params["ts_S"].value
    d = result.params["ts_d"].value
    xi = result.params["ts_xi"].value
    k, kappa = 2 * math.pi / d, 1.0 / xi
    disc = k ** 2 - kappa ** 2
    if disc <= 0:
        return False, "ts_no_finite_q_max"
    q_max = math.sqrt(disc)
    lo, hi = windows.get("W_peak", (0.0, np.inf))
    if not (lo <= q_max <= hi):
        return False, "ts_q_max_outside_w_peak"
    sigma_typ = float(np.median(sigma_local)) if sigma_local.size else 0.0
    if sigma_typ > 0 and S < 8.0 * sigma_typ:
        return False, "ts_not_significant"
    return True, ""


def fit_systematic_floor(model: "CompositeModel", result_params: Any, q: np.ndarray, I: np.ndarray,
                         sigma: np.ndarray, windows: Windows, f_max: float = 0.2) -> float:
    """v4 §5: systematic-error floor. sigma_eff^2 = sigma^2 + (f*I)^2 (the
    `s`-calibrated sigma is already baked into `sigma` by the time this
    runs -- v3 §8.3's plateau calibration). Solves for f in [0, f_max]
    such that chi2red (plain mean of normalized squared residuals, same
    convention as _window_chi2red) reaches 1.0 on W_hiq UNION W_loq
    EXCLUDING W_peak -- the series-wide "measures precision only"
    problem this section exists for is a smooth ~5-10% systematics
    contribution the propagated sigma doesn't capture, which should show
    up on the featureless windows, not the peak itself (where genuine
    model mismatch, not measurement systematics, is the expected and
    interesting signal). Returns 0.0 when the region is already <=1 at
    f=0 (no floor needed) or when there are too few points (<5) to judge;
    returns f_max unchanged (not clamped-and-silent) when even f_max
    can't reach chi2red=1 -- the caller's own f>0.12 flag surfaces this
    as data_systematics_high regardless of which branch produced it."""
    mask = (_mask_for(q, windows, ("W_hiq", "W_loq"))) & (~_mask_for(q, windows, ("W_peak",)))
    if int(mask.sum()) < 5:
        return 0.0
    total = model.eval(q[mask], result_params)
    resid = I[mask] - total
    sig = sigma[mask]
    Ivals = I[mask]

    def _chi2red_minus_1(f: float) -> float:
        sigma_eff = np.sqrt(sig ** 2 + (f * Ivals) ** 2)
        return float(np.mean((resid / sigma_eff) ** 2)) - 1.0

    if _chi2red_minus_1(0.0) <= 0.0:
        return 0.0
    if _chi2red_minus_1(f_max) > 0.0:
        return f_max
    from scipy.optimize import brentq
    return float(brentq(_chi2red_minus_1, 0.0, f_max))


def ts_window_local_delta_bic(q: np.ndarray, I: np.ndarray, sigma: np.ndarray, windows: Windows,
                              bg_params: Any, ts_model: "CompositeModel", ts_result: Any,
                              residual_mode: str = "weighted_linear") -> Optional[float]:
    """v4 §2: TS acceptance decided by residual improvement in W_peak
    ONLY (delta-BIC computed on W_peak points), not by global BIC --
    fixes the diagnosed P5Bi5-12 false negative, where ts_guardrail_ok's
    global-significance check (S vs the WHOLE curve's typical sigma)
    failed even though the peak is clearly visible locally, because the
    curve's global chi2red is poisoned by unmodeled low-q structure
    elsewhere entirely outside W_peak. Compares "bg(+pl) only" against
    "bg(+pl)+TS", both evaluated ONLY on the W_peak-masked points, using
    the SAME weighted/log10 statistic as everywhere else in this module
    (v3's one-statistic-everywhere consistency rule). Returns
    bic_without - bic_with (positive favors TS); None when W_peak has
    too few points to judge (<5, the same bar used throughout this
    module for "insufficient window")."""
    mask = _mask_for(q, windows, ("W_peak",))
    n = int(mask.sum())
    if n < 5:
        return None
    q_w, I_w = q[mask], I[mask]
    bg_only = (bg_params["bg_C"].value
              + bg_params["pl_B"].value * np.power(np.clip(q_w, 1e-300, None), -bg_params["pl_p"].value))
    ts_values = {name: p.value for name, p in ts_result.params.items()}
    ts_total = ts_model.eval(q_w, ts_values)
    if residual_mode == "log10":
        r_without = np.log10(np.clip(bg_only, 1e-300, None)) - np.log10(np.clip(I_w, 1e-300, None))
        r_with = np.log10(np.clip(ts_total, 1e-300, None)) - np.log10(np.clip(I_w, 1e-300, None))
        chi2_without = float(np.sum(r_without ** 2))
        chi2_with = float(np.sum(r_with ** 2))
    else:
        sigma_w = sigma[mask]
        chi2_without = float(np.sum(((I_w - bg_only) / sigma_w) ** 2))
        chi2_with = float(np.sum(((I_w - ts_total) / sigma_w) ** 2))
    k_ts = 3  # ts_S, ts_d, ts_xi -- the only parameters TS adds relative to bg(+pl) alone
    bic_without = chi2_without
    bic_with = chi2_with + k_ts * math.log(n)
    return bic_without - bic_with


def _stage3_add_gp(q: np.ndarray, I: np.ndarray, sigma: np.ndarray, windows: Windows,
                   prev: Dict[str, Any], had_ts: bool,
                   residual_mode: str = "weighted_linear",
                   bg_c_bounds: Optional[Tuple[float, float]] = None,
                   q_knee: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """v4 §2: gp_p bounds are [3.0, 4.3] (class-anchored, not the older
    [2.5, 4.3]). When q_knee is available (Stage A's own classifier), Rg
    is seeded from the SAME q1=(1/Rg)*sqrt(3*p/2) relationship this
    module's own GuinierPorod._q1_D already uses at fit time (Hammouda's
    Guinier-Porod crossover), evaluated at the seed-time default p=4.0:
    Rg_seed = sqrt(6)/q_knee ~= 2.449/q_knee. The ticket's own text
    proposes "1.9/q_knee" but flags it "[[cross-check against Hammouda
    q1]]" as unverified; sqrt(6) (not 1.9) is what falls out of solving
    this component's own crossover formula for Rg at p=4, so it's used
    here instead, self-consistent with how gp_p is actually fit."""
    prev_names = ["flat_background", "power_law"] + (["teubner_strey"] if had_ts else [])
    model = build_composite(prev_names + ["guinier_porod"])
    mask = _mask_for(q, windows, ("W_loq",))
    min_pts = 8  # GP has 3 free params (G, Rg, p) -- needs more than the
    # generic 5-point "insufficient window" bar used for 2-parameter fits
    # elsewhere in this file, or an under-determined fit can hit a wild,
    # overfit local optimum that matches its handful of in-window points
    # exactly but diverges badly everywhere else (found on the real
    # P5Bi0 profile: a 5-point GP fit reached rms_log=5.7, far worse
    # than plain BG, rather than failing outright).
    if int(mask.sum()) < min_pts and q_knee is not None and q_knee > 0:
        # v4 §4: W_loq (the pre-knee-only flat region) is derived as
        # [q_min, 0.7*q_knee] -- at this instrument's real point spacing
        # that can be only 1-2 points wide for a knee sitting close to
        # q_min (confirmed on the real P0Bi0 profile), too few to
        # constrain a GP fit at all despite the classifier having
        # already confirmed a genuine knee exists. Widen to include the
        # knee TRANSITION itself (up to W_hiq's own low edge, or a fixed
        # multiple of q_knee, whichever is smaller) rather than skip a
        # class-anchored component the classifier says belongs here --
        # Guinier-Porod's own p/Rg trade-off needs the steep side of the
        # crossover to be identifiable anyway, not just the flat part.
        hiq_lo = windows.get("W_hiq", (float(np.max(q)), float(np.max(q))))[0]
        wide_hi = min(hiq_lo, 3.0 * q_knee)
        mask = (q >= float(np.min(q))) & (q <= wide_hi)
    if int(mask.sum()) < min_pts:
        return None
    frozen = prev["result"].params
    seed_values = {name: frozen[name].value for name in frozen}
    gp_seed = model.components[-1][1].seed(q[mask], I[mask], windows)
    if q_knee is not None and q_knee > 0:
        gp_seed["Rg"] = math.sqrt(6.0) / q_knee
    seed_values.update({f"gp_{k}": v for k, v in gp_seed.items()})
    bound_overrides = {"bg_C": bg_c_bounds} if bg_c_bounds is not None else {}
    params = model.to_lmfit_parameters(seed_values=seed_values, bound_overrides=bound_overrides)
    for name in frozen:
        if name != "bg_C":
            model.fix(params, name, frozen[name].value)
    params["gp_p"].set(min=3.0, max=4.3)
    if not (params["gp_p"].min < params["gp_p"].value < params["gp_p"].max):
        params["gp_p"].set(value=4.0)
    result = model.fit(q[mask], I[mask], sigma=sigma[mask], params=params, residual_mode=residual_mode)
    return {"model": model, "result": result, "mask": mask, "seeds": seed_values}


def _stage3_add_beaucage(q: np.ndarray, I: np.ndarray, sigma: np.ndarray, windows: Windows,
                         prev: Dict[str, Any], had_ts: bool,
                         residual_mode: str = "weighted_linear",
                         bg_c_bounds: Optional[Tuple[float, float]] = None,
                         q_knee: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """v5 (Beaucage-augmented model library): the class-anchored knee-level
    ALTERNATIVE to _stage3_add_gp, given the SAME staged treatment (frozen
    bg/pl, class-anchored p bounds, q_knee-seeded Rg via the identical
    exp(-q^2 Rg^2/3) Guinier term both components share) so it competes
    fairly in the ladder rather than arriving as a hastily fresh-seeded
    sibling. Motivation: composite_models.BeaucageUnified was already
    implemented (Beaucage, J. Appl. Cryst. 28, 717, 1995) but never wired
    into any preset or stage -- Hammouda's guinier_porod hard-switches to
    a FIXED power-law asymptote at its own continuity point q1, which
    structurally cannot reproduce a local log-log slope steeper than its
    own bounded exponent p (<=4.3 here); the real knee-transition region
    in this series has been measured to show local slopes as steep as -8
    to -12 (see the per-window chi2red investigation on the real
    P5Bi8-12 profile). Beaucage's own additive Guinier+Porod form has no
    such ceiling in the transition itself, since the ever-steepening
    Guinier exponential contributes directly there rather than being cut
    off at a single matched point."""
    prev_names = ["flat_background", "power_law"] + (["teubner_strey"] if had_ts else [])
    model = build_composite(prev_names + ["beaucage_unified"])
    mask = _mask_for(q, windows, ("W_loq",))
    min_pts = 8  # matches _stage3_add_gp's own bar (3 free params: Rg, B, p)
    if int(mask.sum()) < min_pts and q_knee is not None and q_knee > 0:
        hiq_lo = windows.get("W_hiq", (float(np.max(q)), float(np.max(q))))[0]
        wide_hi = min(hiq_lo, 3.0 * q_knee)
        mask = (q >= float(np.min(q))) & (q <= wide_hi)
    if int(mask.sum()) < min_pts:
        return None
    frozen = prev["result"].params
    seed_values = {name: frozen[name].value for name in frozen}
    bc_seed = model.components[-1][1].seed(q[mask], I[mask], windows)
    if q_knee is not None and q_knee > 0:
        bc_seed["Rg"] = math.sqrt(6.0) / q_knee
    seed_values.update({f"bu_{k}": v for k, v in bc_seed.items()})
    bound_overrides = {"bg_C": bg_c_bounds} if bg_c_bounds is not None else {}
    params = model.to_lmfit_parameters(seed_values=seed_values, bound_overrides=bound_overrides)
    for name in frozen:
        if name != "bg_C":
            model.fix(params, name, frozen[name].value)
    params["bu_p"].set(min=3.0, max=4.3)
    if not (params["bu_p"].min < params["bu_p"].value < params["bu_p"].max):
        params["bu_p"].set(value=4.0)
    result = model.fit(q[mask], I[mask], sigma=sigma[mask], params=params, residual_mode=residual_mode)
    return {"model": model, "result": result, "mask": mask, "seeds": seed_values}


def detect_guinier_knee(q: np.ndarray, I: np.ndarray, windows: Windows) -> bool:
    """Does W_loq actually show a genuine, well-RESOLVED Guinier knee (v2
    §3)? Fits the local log-log slope s(q) = d(log I)/d(log q) across
    W_loq: a knee exists only if a MEANINGFUL number of points near q_min
    show a flat plateau (s > -0.5, Guinier-like) and a meaningful number
    near the window's far edge show a steep falloff (s < -2, power-law-
    like past Rg) -- the same qualitative signature guinier_porod's own q1
    crossover describes. Absent that signature, the low-q upturn is
    better described as a plain power law (power_law2) than an
    unconstrained guinier_porod Rg.

    Requires >=15 points (not just >=8) and >=2 points on EACH side
    clearing their respective threshold, not merely the single first/last
    point or a 1-2-point average: on the real P5Bi8-12 profile, W_loq has
    only 10 points total and its literal endpoints happen to straddle
    -0.5/-2 by coincidence (a single-point "plateau" immediately followed
    by a steep Porod-like drop), which a naive endpoint check misreads as
    a genuine, resolvable Guinier feature. With this few points there
    isn't enough independent evidence to trust Rg at all -- the ticket's
    own diagnosis (Rg~1000 Å unconstrained/at-bound) is exactly what a
    too-eager knee call produces."""
    from scipy.ndimage import uniform_filter1d
    lo, hi = windows.get("W_loq", (0.0, 0.0))
    mask = (q >= lo) & (q <= hi) & (q > 0) & (I > 0) & np.isfinite(q) & np.isfinite(I)
    if int(np.sum(mask)) < 15:
        return False
    qm, Im = q[mask], I[mask]
    order = np.argsort(qm)
    qm, Im = qm[order], Im[order]
    log_q, log_I = np.log10(qm), np.log10(Im)
    n = len(log_q)
    win = max(3, n // 8)
    smoothed = uniform_filter1d(log_I, size=win, mode="nearest")
    slope = np.gradient(smoothed, log_q)
    edge = max(2, n // 5)
    flat_count = int(np.sum(slope[:edge] > -0.5))
    steep_count = int(np.sum(slope[-edge:] < -2.0))
    min_count = max(2, edge // 2)
    return flat_count >= min_count and steep_count >= min_count


def _stage3_add_pl2(q: np.ndarray, I: np.ndarray, sigma: np.ndarray, windows: Windows,
                    prev: Dict[str, Any], had_ts: bool,
                    residual_mode: str = "weighted_linear",
                    bg_c_bounds: Optional[Tuple[float, float]] = None) -> Optional[Dict[str, Any]]:
    """The no-knee counterpart of _stage3_add_gp (v2 §3): fits power_law2
    instead of guinier_porod for the low-q role.

    v3 §4 (pl2 stabilization): the fitting mask is q < 0.6*W_peak_lo
    (a tighter, cleaner-upturn-only region than W_loq, computed directly
    from the peak window's own low edge) rather than W_loq -- W_loq's
    upper edge sits AT W_peak's low edge by construction (see
    propose_windows), so it can include peak-flank points that bias the
    low-q-only pl2 fit. p2 is seeded at 3.7 regardless of the component's
    own log-log-regression seed, per the ticket's explicit instruction --
    intended to land the Stage-4 global release (which then typically
    freezes p2, see fit_staged's pl2-sensitivity check) on a stable,
    reproducible starting point rather than whatever a possibly-noisy
    local regression happens to produce on a short, steep window.

    Falls back to W_loq when 0.6*W_peak_lo yields too few points -- this
    instrument's real peaks (large xi) routinely sit close to q_min (an
    established property throughout this module, see propose_windows'
    own W_peak_lo formula), which for many real/synthetic curves pushes
    0.6*W_peak_lo BELOW q_min entirely (zero points, not just "few").
    Skipping Stage 3 outright in that case would silently drop a real
    low-q feature from the model (confirmed on the real P5Bi8-12 profile
    itself, whose low-q tail is exactly what the v3 ADDENDUM's OZ
    component exists to fit) rather than the "clean, peak-free window"
    the tighter mask is meant to provide when the data geometry allows
    it -- falling back to the wider, already-established W_loq window
    is a strictly better outcome than no low-q component at all."""
    prev_names = ["flat_background", "power_law"] + (["teubner_strey"] if had_ts else [])
    model = build_composite(prev_names + ["power_law2"])
    peak_lo = windows.get("W_peak", (float(np.max(q)) if q.size else 1.0, 0.0))[0]
    mask = q < (0.6 * peak_lo)
    if int(mask.sum()) < 5:
        mask = _mask_for(q, windows, ("W_loq",))
    if int(mask.sum()) < 5:
        return None
    frozen = prev["result"].params
    seed_values = {name: frozen[name].value for name in frozen}
    pl2_seed = model.components[-1][1].seed(q[mask], I[mask], windows)
    pl2_seed["p2"] = 3.7
    seed_values.update({f"pl2_{k}": v for k, v in pl2_seed.items()})
    bound_overrides = {"bg_C": bg_c_bounds} if bg_c_bounds is not None else {}
    params = model.to_lmfit_parameters(seed_values=seed_values, bound_overrides=bound_overrides)
    for name in frozen:
        if name != "bg_C":
            model.fix(params, name, frozen[name].value)
    result = model.fit(q[mask], I[mask], sigma=sigma[mask], params=params, residual_mode=residual_mode)
    return {"model": model, "result": result, "mask": mask, "seeds": seed_values}


_SCALE_PARAM_SUFFIXES = ("_C", "_B", "_B2", "_S", "_G", "_A", "_C_lorentz")
_LENGTH_PARAM_SUFFIXES = ("_d", "_xi", "_Rg")


def _widen_bounds_for_global(model: CompositeModel, best_values: Dict[str, float],
                             hard_overrides: Optional[Dict[str, Tuple[float, float]]] = None
                             ) -> Dict[str, Tuple[float, float]]:
    """Spec §4.2 Stage 4: bounds = best ± (x/÷3 for scales, ±40% for
    d/xi/Rg); p (and power_law2's p2) stays within its component's own
    default bound.

    `hard_overrides` (v4 §3) takes precedence over ALL of the above for
    the named parameter -- specifically bg_C's plateau-derived bound.
    Without this, the x/÷3-from-best scheme falls back to the
    component's own UNBOUNDED default the moment best_values["bg_C"]
    is small (<=1e-8, the "too close to zero for a multiplicative
    bound to mean anything" guard below) -- exactly the state Stage 3
    can hand off after its own low-q component absorbs most of the
    background, which then lets Stage 4's global release re-discover
    the same unbounded-collapse failure mode Stage 1's own bound was
    built to prevent (confirmed on the real P0Bi0 profile: Stage 1
    alone lands bg_C safely inside its plateau bound, but Stage 4's
    global refit re-collapsed it to ~5e-18 before this fix)."""
    overrides: Dict[str, Tuple[float, float]] = {}
    for prefix, comp in model.components:
        for p in comp.params():
            full = prefix + p.name
            if hard_overrides and full in hard_overrides:
                overrides[full] = hard_overrides[full]
                continue
            best = best_values.get(full, p.value)
            if p.name in ("p", "p2") and not p.name.endswith(("_d", "xi")):
                overrides[full] = (p.min, p.max)
            elif any(full.endswith(suf) for suf in _LENGTH_PARAM_SUFFIXES):
                lo, hi = best * 0.6, best * 1.4
                overrides[full] = (max(min(lo, hi), p.min), min(max(lo, hi), p.max))
            elif (any(full.endswith(suf) for suf in _SCALE_PARAM_SUFFIXES)
                  or p.name in ("S", "B", "B2", "C", "G", "A", "C_lorentz")):
                # x/÷3 only means anything numerically when `best` is well
                # clear of zero; lmfit's own min==max safeguard uses an
                # absolute tolerance (1e-13), so a tiny-but-nonzero best
                # (pl_B is often ~1e-11 by design here) produces a min/max
                # pair BOTH below that floor, which lmfit then treats as
                # degenerate and raises. Fall back to the component's own
                # default bound in that regime instead.
                if best > 1e-8:
                    lo, hi = best / 3.0, best * 3.0
                    overrides[full] = (max(min(lo, hi), p.min), min(max(lo, hi), p.max))
                else:
                    overrides[full] = (p.min, p.max)
            else:
                overrides[full] = (p.min, p.max)
    return overrides


def _stage4_global(q: np.ndarray, I: np.ndarray, sigma: np.ndarray, model: CompositeModel,
                   best_values: Dict[str, float], sample_id: str, multistart_n: int,
                   residual_mode: str = "weighted_linear",
                   fixed_params: Optional[List[str]] = None,
                   bg_c_bounds: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
    """`fixed_params` (v2 §4) keeps Stage 1's prune-and-refit decision
    intact through the global release: without this, releasing EVERY
    parameter here would silently un-prune power_law right back (its
    bounds reverting to the component's wide default range) and undo the
    whole point of pruning it in the first place. Fixed parameters are
    held at their EXACT best_value (never perturbed) across every
    multistart try, matching how lmfit itself ignores a non-varying
    parameter's "value" for optimization purposes.

    `bg_c_bounds` (v4 §3) is passed as a hard override to
    _widen_bounds_for_global so bg_C's plateau-derived bound survives
    this stage's own bound-widening too, not just Stage 1's fit."""
    hard_overrides = {"bg_C": bg_c_bounds} if bg_c_bounds is not None else None
    bound_overrides = _widen_bounds_for_global(model, best_values, hard_overrides=hard_overrides)
    fixed_set = set(fixed_params or [])
    vary_overrides = {name: False for name in fixed_set}
    rng = np.random.default_rng(_seed_from_sample_id(sample_id))
    best_result = None
    for _ in range(max(multistart_n, 1)):
        perturbed = {name: (v if name in fixed_set else (v * math.exp(rng.uniform(-0.2, 0.2)) if v > 0 else v))
                    for name, v in best_values.items()}
        params = model.to_lmfit_parameters(seed_values=perturbed, bound_overrides=bound_overrides,
                                           vary_overrides=vary_overrides)
        try:
            result = model.fit(q, I, sigma=sigma, params=params, residual_mode=residual_mode)
        except Exception:
            continue
        if best_result is None or result.redchi < best_result.redchi:
            best_result = result
    if best_result is None:
        # last resort: fit once from the un-perturbed best values
        params = model.to_lmfit_parameters(seed_values=best_values, bound_overrides=bound_overrides,
                                           vary_overrides=vary_overrides)
        best_result = model.fit(q, I, sigma=sigma, params=params, residual_mode=residual_mode)
    return {"model": model, "result": best_result}


# =============================================================================
# Stage 5 — diagnostics
# =============================================================================

def _durbin_watson(residual_normalized: np.ndarray) -> float:
    """Durbin-Watson statistic on sigma-normalized residuals: ~2 means no
    autocorrelation; the spec flags DW < 1.3 (residuals trending, usually
    a sign the model shape is wrong somewhere)."""
    r = np.asarray(residual_normalized, dtype=float)
    if r.size < 2:
        return float("nan")
    denom = float(np.sum(r ** 2))
    if denom <= 0:
        return float("nan")
    return float(np.sum(np.diff(r) ** 2) / denom)


def cormap_longest_run(residual: np.ndarray) -> Tuple[int, float]:
    """CorMap-style sigma-free structure diagnostic (v3 §8.5; Franke,
    Jeffries & Svergun 2015, Nat. Methods 12, 419): the longest run of
    consecutive same-sign residuals, and its approximate one-sided
    p-value under the null that residual signs are iid coin flips (the
    standard longest-run-of-heads asymptotic, Schilling 1990): for n
    trials, P(longest run >= C) ~= 1 - exp(-(n-C+1)/2^C). A small p-value
    means the observed run is longer than chance alone would produce --
    real structure in the residuals -- independent of whatever sigma was
    used to fit, a useful cross-check alongside Durbin-Watson."""
    s = np.sign(np.asarray(residual, dtype=float))
    s = s[s != 0]
    n = int(s.size)
    if n < 2:
        return 0, 1.0
    longest = 1
    current = 1
    for i in range(1, n):
        if s[i] == s[i - 1]:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    if longest > 60:  # 2**60 overflows float precision meaningfully; p ~ 0 anyway
        return longest, 0.0
    p = 1.0 - math.exp(-(n - longest + 1) / (2.0 ** longest))
    return longest, float(np.clip(p, 0.0, 1.0))


def _correlation_flags(result: Any, threshold: float = 0.95) -> List[str]:
    """Flag any pair of varying parameters with |correlation| > threshold
    (lmfit computes result.params[name].correl automatically once stderrs
    are available)."""
    flags: List[str] = []
    seen = set()
    for name, par in result.params.items():
        if not par.vary or not getattr(par, "correl", None):
            continue
        for other, rho in par.correl.items():
            key = tuple(sorted((name, other)))
            if key in seen:
                continue
            seen.add(key)
            if rho is not None and np.isfinite(rho) and abs(rho) > threshold:
                flags.append(f"high_correlation:{key[0]}~{key[1]}:{rho:.3f}")
    return flags


def _at_bound_flags(result: Any, rel_tol: float = 0.01) -> List[str]:
    """v2 §4: flag every VARYING parameter within `rel_tol` (1%) of either
    of its bounds -- a clear sign the optimizer wants to go further than
    a physically-motivated range allows, usually meaning the composite is
    mis-specified for this data (the wrong low-/high-q model, a component
    that shouldn't be there, or a genuinely different regime) rather than
    a fit that just needs more iterations."""
    flags: List[str] = []
    for name, par in result.params.items():
        if not par.vary:
            continue
        lo, hi = par.min, par.max
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            continue
        span = hi - lo
        value = par.value
        if abs(value - lo) <= rel_tol * span or abs(value - hi) <= rel_tol * span:
            flags.append(f"at_bound:{name}")
    return flags


def _window_chi2red(model: CompositeModel, result_params: Any, q: np.ndarray, I: np.ndarray,
                    sigma: np.ndarray, windows: Windows) -> Dict[str, float]:
    """Per-window chi2red (consistency-fix addition): mean squared sigma-
    normalized residual within each of W_loq/W_peak/W_hiq. Deliberately
    NOT dof-corrected per window (the global fit's parameters are shared
    across every window, so there's no clean per-window "how many
    parameters does this window alone determine" to subtract) -- this is
    a diagnostic for WHERE the overall chi2red actually comes from, e.g.
    confirming a real fit's misfit is concentrated in the low-q region
    rather than the peak window itself (directly useful for a methods
    section arguing the peak parameters are trustworthy even when the
    global chi2red is elevated)."""
    q = np.asarray(q, dtype=float)
    I = np.asarray(I, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    total = model.eval(q, result_params)
    resid = np.where(sigma > 0, (I - total) / sigma, 0.0)
    out: Dict[str, float] = {}
    for key in ("W_loq", "W_peak", "W_hiq"):
        m = _mask_for(q, windows, (key,))
        if int(m.sum()) >= 1:
            out[f"chi2red_{key.lower()}"] = float(np.mean(resid[m] ** 2))
    return out


def compute_diagnostics(model: CompositeModel, result: Any, q: np.ndarray, I: np.ndarray,
                        windows: Windows, sigma: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Spec §4.2 Stage 5: chi2red/AIC/BIC (lmfit computes these already),
    rms_log (v2 §1, comparable across residual_mode), Durbin-Watson,
    parameter-correlation flags (specific pl-vs-bg/gp/pl2 pairs are always
    covered here since the check is over EVERY varying pair, not just a
    named subset), at-bounds flags (v2 §4), and physicality flags (q_max
    inside W_peak; xi vs d/2pi sanity; Rg vs 2pi/q_min warning).

    `sigma`, when given, adds per-window chi2red to gof (see
    _window_chi2red) -- optional (defaults None, skipping those keys) so
    existing callers/tests that don't have a sigma array handy keep
    working unmodified."""
    dw = _durbin_watson(np.asarray(result.residual, dtype=float))
    flags: List[str] = []
    if np.isfinite(dw) and dw < 1.3:
        flags.append(f"low_durbin_watson:{dw:.2f}")
    cormap_run, cormap_p = cormap_longest_run(np.asarray(result.residual, dtype=float))
    if cormap_p < 0.05:
        flags.append(f"cormap_structured:run={cormap_run}:p={cormap_p:.3g}")
    flags.extend(_correlation_flags(result))
    flags.extend(_at_bound_flags(result))

    prefixes = {prefix.rstrip("_") or comp.name: prefix for prefix, comp in model.components}
    if "ts" in prefixes:
        prefix = prefixes["ts"]
        d = result.params[prefix + "d"].value
        xi = result.params[prefix + "xi"].value
        k, kappa = 2 * math.pi / d, 1.0 / xi
        disc = k ** 2 - kappa ** 2
        q_max = math.sqrt(disc) if disc > 0 else None
        lo, hi = windows.get("W_peak", (0.0, float("inf")))
        if q_max is None or not (lo <= q_max <= hi):
            flags.append("ts_q_max_outside_w_peak")
        if not (xi > d / (2 * math.pi)):
            flags.append("ts_xi_not_greater_than_d_over_2pi")
    if "gp" in prefixes:
        prefix = prefixes["gp"]
        Rg = result.params[prefix + "Rg"].value
        qmin = float(np.min(q)) if q.size else 1e-8
        if Rg > 0.8 * (2 * math.pi / max(qmin, 1e-12)):
            flags.append("gp_rg_poorly_constrained_vs_qmin")

    gof = _gof(model, result, q, I)
    gof["durbin_watson"] = dw
    gof["cormap_longest_run"] = cormap_run
    gof["cormap_pvalue"] = cormap_p
    if sigma is not None:
        gof.update(_window_chi2red(model, result.params, q, I, sigma, windows))
    return {"gof": gof, "flags": flags}


def median_log_residual_by_window(model: CompositeModel, result_params: Any,
                                  q: np.ndarray, I: np.ndarray, windows: Windows) -> Dict[str, float]:
    """|median log10 residual| within each of W_loq/W_peak/W_hiq (v3
    ADDENDUM §7's visual-equivalence check) -- a window with too few
    points (<5, matching the "insufficient window" bar used elsewhere in
    this module for Stage 3's own fits) is simply absent from the
    returned dict, rather than gating model selection on a median of a
    literal handful of points where a single noisy sample can trip the
    threshold regardless of whether the model is actually right there."""
    q = np.asarray(q, dtype=float)
    I = np.asarray(I, dtype=float)
    total = model.eval(q, result_params)
    resid = np.log10(np.clip(total, 1e-300, None)) - np.log10(np.clip(I, 1e-300, None))
    out: Dict[str, float] = {}
    for key in ("W_loq", "W_peak", "W_hiq"):
        m = _mask_for(q, windows, (key,))
        if int(m.sum()) >= 5:
            out[key] = float(np.median(np.abs(resid[m])))
    return out


def visual_equivalence_ok(model: CompositeModel, result: Any, q: np.ndarray, I: np.ndarray,
                          windows: Windows, threshold: float = 0.15) -> Tuple[bool, Dict[str, float]]:
    """v3 ADDENDUM §7: a candidate whose |median log10 residual| is >=
    threshold in ANY of W_loq (post beamstop-trim)/W_peak/W_hiq visibly
    doesn't describe that region of the curve regardless of how good its
    BIC looks overall -- treated as a hard gate, not a soft flag, in the
    model-selection ladder (select_best_preset)."""
    medians = median_log_residual_by_window(model, result.params, q, I, windows)
    ok = all(v < threshold for v in medians.values())
    return ok, medians


# =============================================================================
# Stage 6 — model-selection ladder
# =============================================================================

def _fit_full_range(component_names: List[str], q: np.ndarray, I: np.ndarray, sigma: np.ndarray,
                    sample_id: str, multistart_n: int,
                    residual_mode: str = "weighted_linear",
                    windows: Optional[Windows] = None,
                    bg_c_bounds: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
    """`windows` matters for any TS/GP/PL2-containing candidate: their
    generic .seed() fallbacks locate the peak/low-q region from the WHOLE
    curve when no windows are given, which for a real SAXS profile (a huge
    low-q upturn dwarfing everything else) picks the wrong feature
    entirely -- the exact bug _locate_peak itself was fixed for in Phase 6.
    BG/BG_DAB don't need windows and are unaffected either way."""
    model = build_composite(component_names)
    seeds = model.seed(q, I, windows)
    return _stage4_global(q, I, sigma, model, seeds, sample_id + ":" + "_".join(component_names), multistart_n,
                          residual_mode=residual_mode, bg_c_bounds=bg_c_bounds)


def _walk_ladder(order: List[str], bics: Dict[str, float], aics: Dict[str, float]) -> Tuple[str, List[Dict[str, Any]]]:
    """Pure decision logic (no fitting): walk `order` left-to-right,
    replacing the current pick whenever the next candidate clears
    Delta-BIC > 10 (current's BIC minus candidate's BIC). Any disagreement
    with the Delta-AIC > 10 verdict is recorded, but BIC always decides —
    the spec's own explicit tiebreak rule."""
    current = order[0]
    disagreements: List[Dict[str, Any]] = []
    for candidate in order[1:]:
        d_bic = bics[current] - bics[candidate]
        d_aic = aics[current] - aics[candidate]
        prefer_bic = d_bic > 10.0
        prefer_aic = d_aic > 10.0
        if prefer_bic != prefer_aic:
            disagreements.append({"pair": [current, candidate], "d_bic": d_bic, "d_aic": d_aic})
        if prefer_bic:
            current = candidate
    return current, disagreements


def select_best_preset(
    q: np.ndarray, I: np.ndarray, sigma: np.ndarray, assembled_name: str,
    assembled_model: CompositeModel, assembled_result: Any,
    sample_id: str, multistart_n: int, residual_mode: str = "weighted_linear",
    had_ts: bool = False, has_knee: bool = False, windows: Optional[Windows] = None,
    bg_c_bounds: Optional[Tuple[float, float]] = None,
    precomputed: Optional[Dict[str, Tuple[CompositeModel, Any]]] = None,
) -> Dict[str, Any]:
    """Spec §4.2 Stage 6, extended per v2 §3 and v3 ADDENDUM §7: the ladder
    is BG -> BG_DAB -> BG_TS -> BG_TS_OZ -> BG_TS_PL2 -> BG_TS_GP, where
    BG_TS/BG_TS_OZ/BG_TS_PL2 only enter the walk when stages 1-4 actually
    found a significant TS peak (`had_ts`), and BG_TS_GP only enters when
    a genuine Guinier knee was ALSO detected (`has_knee`) -- otherwise
    BG_TS_OZ/BG_TS_PL2 (an Ornstein-Zernike tail or a plain low-q power
    law) are the richest candidates offered. Primary criterion is ΔBIC>10
    (lower BIC wins); ΔAIC is cross-checked and any disagreement is
    recorded, but BIC always decides ties (spec's own explicit tiebreak).
    Whatever stages 1-4 already assembled (`assembled_name`) is reused
    as-is rather than re-fit from scratch when it coincides with one of
    these rungs (it usually does) -- only the OTHER rungs get a fresh
    `_fit_full_range` call.

    Every candidate (not just TS-containing ones) must ALSO clear the v3
    ADDENDUM §7 visual-equivalence gate (|median log10 residual| < 0.15 in
    each of W_loq/W_peak/W_hiq) -- a candidate that fits the BIC criterion
    but is visibly wrong in some window is rejected exactly like a
    guardrail failure, and the ladder walk continues to the next rung.
    If literally EVERY candidate fails this gate (only possible when the
    data itself has more structure than any composite in this library can
    capture to that precision), falling back to plain BG would make
    things WORSE, not better -- BG is invariably the crudest, highest-
    chi2red option of the lot. Instead the single best-BIC candidate
    among everything actually tried (gate-passing or not) is reported,
    flagged `visual_equivalence_gate_bypassed` so this is never silent —
    reporting the best AVAILABLE description of the data, honestly
    flagged, beats reporting a worse one just to satisfy the gate."""
    candidates: Dict[str, Tuple[CompositeModel, Any]] = {}
    all_attempts: Dict[str, Tuple[CompositeModel, Any]] = {}
    ladder: Dict[str, Any] = {}

    def _passes_guardrail(name: str, component_names: List[str], model: CompositeModel, result: Any) -> bool:
        """A TS-containing ladder candidate must ALSO clear
        ts_guardrail_ok's significance/sanity check, exactly like the
        originally-staged model does in Stage 2 -- these fresh candidates
        (fit via _fit_full_range, specifically so the ladder can compare
        alternatives BIC couldn't otherwise see) bypass that guardrail
        entirely if left unchecked, and BIC alone cannot tell a genuine
        peak from a fit that's just interpolating noise with a physically
        nonsensical one (found via the 20-curve peak-free synthetic
        battery: a spurious candidate with d~20-25 Å and xi pinned at its
        bound still won on BIC alone, a real regression from adding these
        extra rungs in v2 without carrying the guardrail along).

        Then EVERY candidate (TS-containing or not) must clear the v3
        ADDENDUM §7 visual-equivalence gate."""
        all_attempts[name] = (model, result)
        entry: Dict[str, Any] = {"bic": float(result.bic), "aic": float(result.aic),
                                 "rms_log": _rms_log10(model, result.params, q, I)}
        if "teubner_strey" in component_names and windows:
            ok, reason = ts_guardrail_ok(result, sigma, windows)
            if not ok:
                entry["rejected"] = reason
                ladder[name] = entry
                return False
        if windows:
            veq_ok, medians = visual_equivalence_ok(model, result, q, I, windows)
            if not veq_ok:
                entry["rejected"] = "visual_equivalence_fail"
                entry["window_medians"] = medians
                ladder[name] = entry
                return False
        ladder[name] = entry
        return True

    assembled_components = [comp.name for _, comp in assembled_model.components]
    if _passes_guardrail(assembled_name, assembled_components, assembled_model, assembled_result):
        candidates[assembled_name] = (assembled_model, assembled_result)

    def _ensure(name: str, component_names: List[str]) -> bool:
        if name in candidates:
            return True
        if name in ladder:  # already tried and guardrail/visual-equivalence-rejected
            return False
        if precomputed and name in precomputed:
            # v5: a candidate already staged+globally-refined by the
            # caller (e.g. the non-primary knee-level alternative from
            # Stage 3, given the SAME frozen-bg/q_knee-seeded treatment
            # as the assembled winner) -- use it as-is rather than
            # re-fitting from a generic whole-range seed, which measurably
            # lands in a much worse local optimum for these components.
            model, result = precomputed[name]
        else:
            fit = _fit_full_range(component_names, q, I, sigma, sample_id, multistart_n,
                                  residual_mode=residual_mode, windows=windows, bg_c_bounds=bg_c_bounds)
            model, result = fit["model"], fit["result"]
        if not _passes_guardrail(name, component_names, model, result):
            return False
        candidates[name] = (model, result)
        return True

    order: List[str] = []
    if _ensure("BG", ["flat_background", "power_law"]):
        order.append("BG")
    if _ensure("BG_DAB", ["flat_background", "power_law", "dab"]):
        order.append("BG_DAB")
    if has_knee and _ensure("BG_BC", ["flat_background", "power_law", "beaucage_unified"]):
        # v5 (Beaucage-augmented model library): a class-anchored alternative
        # to BG_GP, tried whenever a knee exists -- Hammouda's guinier_porod
        # locks the high-q asymptote to a FIXED, bounded power law (p<=4.3),
        # but the real knee-transition region in this series has been
        # measured to show local log-log slopes as steep as -8 to -12 (see
        # _stage3_add_gp's own docstring history and the real per-window
        # chi2red investigation on P5Bi8-12) -- steeper than ANY fixed
        # power-law asymptote can produce. Beaucage's additive Guinier+Porod
        # form (already implemented in composite_models.py, previously
        # unused anywhere in this pipeline) has no such asymptotic ceiling
        # in its transition region, since the Guinier term's own ever-
        # steepening exponential decay contributes directly there rather
        # than being hard-switched off at a fixed crossover point.
        order.append("BG_BC")
    if has_knee and _ensure("BG_GP", ["flat_background", "power_law", "guinier_porod"]):
        # Explicit sibling to BG_BC above (mirrors its placement exactly):
        # without this, when Beaucage wins as the PRIMARY assembled path,
        # guinier_porod -- carried forward as the precomputed, fully-staged
        # alternate candidate -- was never actually entered into the
        # ladder for the no-TS-peak case (it only got a free ride in via
        # the assembled_name fallback below, which only fires when GP
        # itself was the assembled winner). Found via direct testing: the
        # BG_GP vs BG_BC comparison must run both ways, not just one.
        order.append("BG_GP")
    if had_ts:
        if _ensure("BG_TS", ["flat_background", "power_law", "teubner_strey"]):
            order.append("BG_TS")
        if _ensure("BG_TS_OZ", ["flat_background", "power_law", "teubner_strey", "lorentz_oz"]):
            order.append("BG_TS_OZ")
        if _ensure("BG_TS_PL2", ["flat_background", "power_law", "teubner_strey", "power_law2"]):
            order.append("BG_TS_PL2")
        if has_knee and _ensure("BG_TS_GP", ["flat_background", "power_law", "teubner_strey", "guinier_porod"]):
            order.append("BG_TS_GP")
        if has_knee and _ensure("BG_TS_BC", ["flat_background", "power_law", "teubner_strey", "beaucage_unified"]):
            order.append("BG_TS_BC")
    if assembled_name not in order and assembled_name in candidates:
        order.append(assembled_name)

    if not order:
        best_name = min(all_attempts, key=lambda name: all_attempts[name][1].bic)
        candidates[best_name] = all_attempts[best_name]
        ladder["visual_equivalence_gate_bypassed"] = best_name
        order = [best_name]

    bics = {name: candidates[name][1].bic for name in order}
    aics = {name: candidates[name][1].aic for name in order}
    current_name, disagreements = _walk_ladder(order, bics, aics)
    if disagreements:
        ladder["disagreements"] = disagreements

    final_model, final_result = candidates[current_name]
    return {"chosen": current_name, "model": final_model, "result": final_result, "ladder": ladder}


# v2 §4: "two or more at-bound params => auto-suggest the next-simpler
# preset from the ladder" -- the ladder's own order, one step back.
# v3 ADDENDUM §7 inserts BG_TS_OZ between BG_TS and BG_TS_PL2.
_SIMPLER_PRESET = {
    "BG_TS_BC": "BG_TS_GP",
    "BG_TS_GP": "BG_TS_PL2",
    "BG_TS_PL2": "BG_TS_OZ",
    "BG_TS_OZ": "BG_TS",
    "BG_TS": "BG_DAB",
    "BG_BC": "BG_DAB",
    "BG_DAB": "BG",
    "BG": "BG",
}


# =============================================================================
# v3 §4 — pl2 sensitivity check (freeze p2 for the global stage unless
# releasing it materially improves the fit)
# =============================================================================

def _pl2_sensitive(model: CompositeModel, best_values: Dict[str, float],
                   q: np.ndarray, I: np.ndarray, sigma: np.ndarray,
                   residual_mode: str = "weighted_linear") -> bool:
    """v3 §4: 'FREEZE p2 for the global stage (release only if its profile
    shows sensitivity)'. Lightweight, cheap check: nudge pl2_p2 +/-10% from
    its Stage-3 value, refit everything else once at each nudge, and see
    if chi2 changes non-trivially (>5% relative). If not, freezing p2 loses
    nothing scientifically and kills the pl2_B2~pl2_p2 anti-correlation
    that was leaking into the fitted TS width (the ticket's stated goal);
    if chi2 DOES change meaningfully, p2 carries real information the
    global fit needs and should stay free. Defaults to "release" (True)
    if the baseline fit itself fails, since freezing an untested parameter
    is the riskier default."""
    p2_val = best_values.get("pl2_p2")
    if p2_val is None or p2_val <= 0:
        return False
    base_params = model.to_lmfit_parameters(seed_values=best_values)
    try:
        base_result = model.fit(q, I, sigma=sigma, params=base_params, residual_mode=residual_mode)
        base_chi2 = float(base_result.chisqr)
    except Exception:
        return True
    if not (math.isfinite(base_chi2) and base_chi2 > 0):
        return True
    for factor in (0.9, 1.1):
        trial_values = dict(best_values)
        trial_values["pl2_p2"] = p2_val * factor
        params = model.to_lmfit_parameters(seed_values=trial_values, vary_overrides={"pl2_p2": False})
        try:
            r = model.fit(q, I, sigma=sigma, params=params, residual_mode=residual_mode)
        except Exception:
            continue
        if abs(float(r.chisqr) - base_chi2) / base_chi2 > 0.05:
            return True
    return False


# =============================================================================
# v3 §3 — Stage 2b: peak-focused cross-check
# =============================================================================

_TS_PARAM_NAMES = ("ts_S", "ts_d", "ts_xi")


def stage2b_peak_crosscheck(
    model: CompositeModel, final_result: Any, q: np.ndarray, I: np.ndarray, sigma: np.ndarray,
    windows: Windows, residual_mode: str = "weighted_linear",
) -> Optional[Dict[str, Any]]:
    """v3 §3: after the global fit, refit ONLY the TS parameters (S/d/xi)
    on W_peak with every other component FROZEN at its global best-fit
    value, mirroring _stage4_global's own fixed_params/vary_overrides
    mechanism. A cross-check on the peak region alone, independent of
    whatever the low-q/high-q components are doing globally -- if d/xi
    move a lot once everything else is held fixed and only the peak's own
    local shape is fit, the global values are less trustworthy than they
    look."""
    if "ts_d" not in final_result.params or "ts_xi" not in final_result.params:
        return None
    mask = _mask_for(q, windows, ("W_peak",))
    if int(mask.sum()) < 5:
        return None
    vary_overrides = {name: (name in _TS_PARAM_NAMES) for name in final_result.params}
    seed_values = {name: final_result.params[name].value for name in final_result.params}
    params = model.to_lmfit_parameters(seed_values=seed_values, vary_overrides=vary_overrides)
    for name in final_result.params:
        if name not in _TS_PARAM_NAMES:
            model.fix(params, name, final_result.params[name].value)
    try:
        result = model.fit(q[mask], I[mask], sigma=sigma[mask], params=params, residual_mode=residual_mode)
    except Exception:
        return None
    return {"result": result, "mask": mask}


# =============================================================================
# v3 §2 — Profile-likelihood confidence intervals for ts_d / ts_xi
# =============================================================================

def _find_crossing(xs: np.ndarray, ys: np.ndarray, threshold: float) -> Optional[float]:
    """`xs` ordered from closest-to-best (index 0) outward; returns the
    linearly-interpolated x where `ys` first crosses `threshold` moving
    outward, or None if it never does within the given range."""
    for i in range(len(xs) - 1):
        y0, y1 = ys[i], ys[i + 1]
        if not (np.isfinite(y0) and np.isfinite(y1)):
            continue
        if y0 < threshold <= y1:
            frac = (threshold - y0) / (y1 - y0) if y1 != y0 else 0.0
            return float(xs[i] + frac * (xs[i + 1] - xs[i]))
    return None


def _profile_delta_chi2(
    model: CompositeModel, best_result: Any, q: np.ndarray, I: np.ndarray, sigma: np.ndarray,
    param_name: str, grid: np.ndarray, residual_mode: str,
) -> np.ndarray:
    """Delta-chi2(grid point) vs. the global minimum, refitting every
    OTHER free parameter per grid point from the current best values -- a
    SMALL local multistart (3 tries: unperturbed + 2 mildly-perturbed
    restarts, keeping the lowest chi2), not a full production multistart
    burst (25 grid points x a full multistart each would be prohibitively
    expensive per sample for no real gain, since the global stage already
    did a proper multistart). The small multistart matters in practice:
    a single unperturbed local refit was found (via a synthetic BG_TS_PL2
    curve with a near-perfect pl2_B2~pl2_p2 anti-correlation, rho~-1.0) to
    occasionally get stuck in a much worse local configuration for a
    slightly-off grid value, producing a Delta-chi2 that jumps from ~0 to
    billions within a single grid step for a parameter that's actually
    perfectly well-behaved -- a fragile-optimizer artifact, not a real
    statistical feature of the fit."""
    best_chi2 = float(best_result.chisqr)
    dchi = np.full(len(grid), np.nan)
    base_values = {n: best_result.params[n].value for n in best_result.params}
    rng = np.random.default_rng(0)
    for i, val in enumerate(grid):
        if param_name not in best_result.params:
            continue
        best_trial_chi2 = None
        for attempt in range(3):
            perturbed = base_values if attempt == 0 else {
                n: (v * math.exp(rng.uniform(-0.15, 0.15)) if v > 0 else v) for n, v in base_values.items()
            }
            trial = model.to_lmfit_parameters(seed_values=perturbed)
            for n in best_result.params:
                p = best_result.params[n]
                trial[n].set(min=p.min, max=p.max, vary=p.vary)
            trial[param_name].set(value=float(val), vary=False)
            try:
                r = model.fit(q, I, sigma=sigma, params=trial, residual_mode=residual_mode)
                if best_trial_chi2 is None or float(r.chisqr) < best_trial_chi2:
                    best_trial_chi2 = float(r.chisqr)
            except Exception:
                continue
        if best_trial_chi2 is not None:
            dchi[i] = best_trial_chi2 - best_chi2
    return dchi


def _ci_from_profile(best_value: float, grid: np.ndarray, dchi: np.ndarray,
                     threshold: float = 1.0) -> Tuple[Optional[float], Optional[float]]:
    """Split the grid at `best_value` into a below-side and an above-side,
    each walked OUTWARD from the best value, and find where Delta-chi2
    crosses `threshold` on each side independently.

    Both sides are explicitly anchored at (best_value, 0.0) rather than
    relying on the grid itself containing an exact duplicate of
    best_value: the grid is built via geomspace/linspace from best_value,
    but floating-point rounding in that construction means its "middle"
    point can differ from best_value by a tiny epsilon. Without this
    anchor, that epsilon can push the true best-value point onto the
    WRONG side of a `<=`/`>=` split, leaving the correct side with a run
    of far-away, all-identical (clipped) grid values and no point near
    the actual minimum to interpolate from -- silently returning None
    for a perfectly well-identified parameter (found via a synthetic
    curve with a very tightly-constrained xi, where floating-point noise
    alone was enough to trigger this)."""
    order = np.argsort(grid)
    g, d = grid[order], dchi[order]
    below_mask = g < best_value
    above_mask = g > best_value
    below_g = np.concatenate(([best_value], g[below_mask][::-1]))
    below_d = np.concatenate(([0.0], d[below_mask][::-1]))
    lower = _find_crossing(below_g, below_d, threshold) if len(below_g) >= 2 else None
    above_g = np.concatenate(([best_value], g[above_mask]))
    above_d = np.concatenate(([0.0], d[above_mask]))
    upper = _find_crossing(above_g, above_d, threshold) if len(above_g) >= 2 else None
    return lower, upper


_CHI2_95_ONE_PARAM = 3.841458820694124  # scipy.stats.chi2.ppf(0.95, df=1)


def _log_spaced_grid_around(best: float, max_log_offset: float, n_per_side: int = 12) -> np.ndarray:
    """25-point grid (n_per_side below + best + n_per_side above) with
    LOG-spaced offsets in log10(param) from `best`, densest near `best`
    and sparsest at `max_log_offset` away. Consistency-fix addition: a
    naive UNIFORM grid across the whole ±range (the ticket's own literal
    "25 linear/log-spaced points") puts its first off-best sample far
    enough away that, for a real curve with a genuinely steep chi2
    landscape, the ACTUAL Delta-chi2 crossing (for either the stat or
    the rescaled threshold) falls within that single first segment --
    linear interpolation across one huge, unsampled gap then makes the
    reported CI half-width scale linearly with the threshold instead of
    with its square root (found via a real-data consistency check: the
    stat-vs-rescaled half-width ratio came out as chi2red instead of the
    theoretically-required sqrt(chi2red), for both d and xi). Densifying
    near the center (same overall range, same point BUDGET) ensures
    whichever threshold's crossing point is actually being asked for gets
    bracketed by nearby samples rather than one distant, uninformative
    pair."""
    if max_log_offset <= 0:
        return np.full(2 * n_per_side + 1, best)
    log_offsets = np.geomspace(max_log_offset * 1e-4, max_log_offset, n_per_side)
    below = best * np.power(10.0, -log_offsets[::-1])
    above = best * np.power(10.0, log_offsets)
    return np.concatenate([below, [best], above])


def compute_ts_profile_likelihood_cis(
    model: CompositeModel, best_result: Any, q: np.ndarray, I: np.ndarray, sigma: np.ndarray,
    residual_mode: str = "weighted_linear",
) -> Dict[str, Any]:
    """v3 §2, re-unified per a real-data consistency cross-check: profile-
    likelihood CIs for ts_d (out to +/-15%) and ts_xi (out to x/รท4), 25
    points each, LOG-spaced in offset from the best value (see
    _log_spaced_grid_around) rather than uniformly across the whole
    range -- computed ONCE per parameter (the expensive part -- refitting
    every other free parameter at each grid
    point) and then evaluated at TWO thresholds against that same
    Delta-chi2 profile:

    - `d_ci_stat`/`xi_ci_stat` ("model-conditional"): Delta-chi2 <= 3.841
      (the one-parameter 95%-confidence chi-square critical value),
      taking the calibrated sigma (v3 §8.3's plateau rescale) at face
      value -- i.e. assuming the model itself is correct and only
      measurement noise contributes.
    - `d_ci`/`xi_ci` (the HEADLINE values): Delta-chi2 <= 3.841 *
      max(1, chi2red_global) -- the standard goodness-of-fit correction
      (MINUIT/Numerical-Recipes practice) for a fit whose overall chi2red
      is still >>1 even after plateau calibration, meaning some region
      has real unmodeled structure (a TS peak riding on a complex low-q
      feature, say) beyond what a single global sigma rescale can fix.
      `stat`-vs-headline half-widths should differ by very close to
      sqrt(chi2red_global) BY CONSTRUCTION (both share the same Delta-chi2
      profile, only the threshold differs by that factor) -- this ratio
      is exactly what a real cross-check should verify stayed consistent
      with lmfit's own (correctly scale_covar=True-corrected) stderr.

    Returns {} when ts_d/ts_xi aren't in the model at all.

    xi_unidentifiable (v3 §2) is decided from the HEADLINE (rescaled)
    profile: set when the upper side never crosses that threshold within
    the grid, OR the crossing found lands within 1% of ts_xi's own hard
    upper bound -- either way, the data can't rule out an arbitrarily
    large xi, so it's reported as a lower bound ("xi > lower_ci") rather
    than a point value. fa_bound follows: since |fa| increases
    monotonically toward 1 as xi -> infinity (fa's own formula is a
    strictly monotonic function of xi at fixed d), an unidentifiable
    upper xi means the TRUE fa is bounded by whatever fa would be at
    (d_best, xi_lower_ci) -- reported as `fa < that value`."""
    from saxs.core.composite_models import ts_classic_from_physical
    names = best_result.params
    if "ts_d" not in names or "ts_xi" not in names:
        return {}
    d_best = float(names["ts_d"].value)
    xi_best = float(names["ts_xi"].value)
    d_lo_b, d_hi_b = float(names["ts_d"].min), float(names["ts_d"].max)
    xi_lo_b, xi_hi_b = float(names["ts_xi"].min), float(names["ts_xi"].max)
    chi2red_global = float(best_result.redchi) if math.isfinite(best_result.redchi) else 1.0
    threshold_stat = _CHI2_95_ONE_PARAM
    threshold_rescaled = _CHI2_95_ONE_PARAM * max(1.0, chi2red_global)

    d_grid = np.clip(_log_spaced_grid_around(d_best, math.log10(1.15), n_per_side=12), d_lo_b, d_hi_b)
    xi_grid = np.clip(_log_spaced_grid_around(xi_best, math.log10(4.0), n_per_side=12), xi_lo_b, xi_hi_b)

    d_dchi = _profile_delta_chi2(model, best_result, q, I, sigma, "ts_d", d_grid, residual_mode)
    d_lower, d_upper = _ci_from_profile(d_best, d_grid, d_dchi, threshold=threshold_rescaled)
    d_lower_stat, d_upper_stat = _ci_from_profile(d_best, d_grid, d_dchi, threshold=threshold_stat)

    xi_dchi = _profile_delta_chi2(model, best_result, q, I, sigma, "ts_xi", xi_grid, residual_mode)
    xi_lower, xi_upper = _ci_from_profile(xi_best, xi_grid, xi_dchi, threshold=threshold_rescaled)
    xi_lower_stat, xi_upper_stat = _ci_from_profile(xi_best, xi_grid, xi_dchi, threshold=threshold_stat)

    xi_unidentifiable = xi_upper is None or (
        xi_hi_b > 0 and abs(xi_upper - xi_hi_b) <= 0.01 * xi_hi_b
    )

    out: Dict[str, Any] = {
        "d_ci": (d_lower, d_upper) if (d_lower is not None and d_upper is not None) else None,
        "d_ci_stat": (d_lower_stat, d_upper_stat) if (d_lower_stat is not None and d_upper_stat is not None) else None,
        "xi_unidentifiable": xi_unidentifiable,
    }
    if xi_unidentifiable:
        out["xi_ci"] = (xi_lower, None) if xi_lower is not None else None
        out["xi_ci_stat"] = (xi_lower_stat, None) if xi_lower_stat is not None else None
        if xi_lower is not None:
            a2, c1, c2 = ts_classic_from_physical(d_best, xi_lower)
            out["fa_bound"] = c1 / math.sqrt(4.0 * a2 * c2)
        else:
            out["fa_bound"] = None
    else:
        out["xi_ci"] = (xi_lower, xi_upper) if (xi_lower is not None and xi_upper is not None) else None
        out["xi_ci_stat"] = (
            (xi_lower_stat, xi_upper_stat) if (xi_lower_stat is not None and xi_upper_stat is not None) else None
        )
        out["fa_bound"] = None
    return out


# =============================================================================
# Orchestrator
# =============================================================================

def fit_staged(
    curve: Curve,
    *,
    sample_id: Optional[str] = None,
    windows: Optional[Windows] = None,
    trim_n: int = 3,
    residual_mode: Optional[str] = None,
    data_type: Optional[str] = None,
    loss: str = "linear",
    multistart_n: int = 8,
    mask_regions: Optional[List[Tuple[float, float]]] = None,
    force_preset: Optional[str] = None,
    log: Callable[[str], None] = lambda *_: None,
) -> FitResult:
    """Run stages 0-4 on one profile. Never raises: a later stage that
    can't be fit (too few points in its window, a non-significant/
    nonsensical TS peak, an lmfit exception) simply falls back to the
    best composite assembled so far — the returned FitResult always
    reflects SOME valid fit, down to BG alone in the worst case.

    `residual_mode=None` (the default) picks per v3 §8.1/8.2: "weighted_
    linear" (the sigma-weighted linear objective, now primary) whenever
    the curve carries a genuinely measured/propagated sigma column (see
    apply_hygiene/FitResult.sigma_model=="measured"); "log10" (unweighted
    log-residual fitting) as the explicit last-resort fallback when the
    curve has NO sigma at all and one had to be ESTIMATED (Poisson-like
    for genuine counts data, detrended-MAD otherwise) -- an estimated
    sigma is a rough approximation that can itself be biased over a
    region the current composite doesn't yet model, letting that region
    dominate a weighted-linear objective out of proportion to its real
    reliability; log10 residuals are immune to this regardless of sigma
    quality. This replaces v2's "arbitrary units" auto-switch, which keyed
    off the DATA's own shape rather than whether a trustworthy sigma
    actually exists. Pass an explicit `residual_mode` to override either
    way.

    `data_type` ("counts" or "au") overrides detect_data_type's own
    inference for the sigma-estimation fallback in apply_hygiene (only
    relevant when the curve has no measured sigma at all) -- for a caller
    that knows the data's true nature better than the generic heuristic
    can (e.g. a curve that's genuinely Poisson-counting-consistent but
    happens to look non-integer purely from a unit rescaling).

    `mask_regions=None` (v2 §2) auto-detects a rising high-q tail via
    detect_high_q_cut and excludes [q_cut, q_max] from every stage; pass
    an explicit list of [lo,hi] exclude ranges to override (an empty list
    disables masking entirely). Either way the ranges actually used are
    recorded in FitResult.mask_regions/stages['stage0'] for provenance —
    always visible/editable by the caller, never silently applied. A
    beamstop-edge trim (v3 ADDENDUM §7) additionally drops up to 10 points
    from the LOW-q end when their local slope departs from the interior's
    own reference slope by more than 30% -- recorded in stages['stage0']
    as `beamstop_trimmed_n`/`beamstop_trim_qmax` for the UI to grey out.

    `force_preset` (v2 §4: a manually-picked preset must still go through
    the staged protocol, never a one-shot fit from a single generic seed)
    skips the BIC ladder's OWN choice and reports whichever preset is
    named instead -- but still via hygiene + masking + auto-proposed
    windows + the SAME thorough seeded-multistart global refinement
    (`_fit_full_range`) the ladder itself uses to evaluate candidates, not
    a naive single-seed fit. If `force_preset` already matches what stages
    1-4 assembled, that already-staged result is reused directly."""
    sample_id = sample_id or curve.name
    flags: List[str] = []

    hygiene = apply_hygiene(curve, trim_n=trim_n, data_type_override=data_type)
    q = np.asarray(hygiene.curve.q, dtype=float)
    I = np.asarray(hygiene.curve.intensity, dtype=float)
    sigma = np.asarray(hygiene.curve.sigma, dtype=float)

    n_beamstop = detect_beamstop_edge_trim(q, I)
    beamstop_trim_qmax = float(q[n_beamstop - 1]) if n_beamstop > 0 else None
    if n_beamstop > 0:
        q, I, sigma = q[n_beamstop:], I[n_beamstop:], sigma[n_beamstop:]
        flags.append(f"beamstop_trimmed:{n_beamstop}")

    if residual_mode is None:
        # v3 §8.1/8.2: sigma-weighted linear is primary WHEN there's a
        # real sigma to weight by -- either genuinely measured/propagated,
        # or "no sigma exists" is false. When the curve carries no sigma
        # at all, an ESTIMATED one (Poisson-like or detrended-MAD) is a
        # rough approximation that can itself be biased low over a region
        # the current composite doesn't yet model (e.g. an unmodeled
        # low-q upturn artificially depresses the local noise estimate
        # there), which then lets that region's points dominate a
        # weighted-linear objective out of proportion to their real
        # reliability -- exactly the failure mode "unweighted log fitting
        # survives as a last-resort fallback when no sigma exists" (v3
        # §8.2) is for: log10 residuals are immune to this since they
        # don't depend on sigma's magnitude at all. Confirmed empirically:
        # a synthetic no-sigma TS+Guinier-Porod curve recovers xi to
        # <0.1% error in log10 mode vs. >150% error in weighted_linear
        # mode with the estimated sigma.
        residual_mode = "weighted_linear" if hygiene.sigma_model == "measured" else "log10"

    q_cut = None if mask_regions is not None else detect_high_q_cut(q, I)
    if mask_regions is None:
        active_mask_regions = [(q_cut, float(np.max(q)))] if q_cut is not None else []
    else:
        active_mask_regions = list(mask_regions)
    q, I, sigma, excluded_mask = _apply_mask_regions(q, I, sigma, active_mask_regions)

    cls_guess, prominence = guess_class(q, I)
    # v4 §1/§4: Stage A classifies morphology BEFORE any fitting, using
    # the curve's own genuine measurement sigma for peak significance
    # (see detect_peak_q); windows are then derived from q_knee/q_peak
    # rather than propose_windows' whole-curve _locate_peak-based guess,
    # which always finds SOME candidate even on a genuinely peak-free
    # curve (its Kratky-space fallback locks onto the same knee-artifact
    # classify_morphology exists to reject) -- confirmed to build
    # nonsensical windows for a no-peak S-class sample like P0Bi0.
    # detect_peak_q's significance test needs a genuinely trustworthy
    # per-point sigma -- an ESTIMATED fallback (Poisson-like or
    # detrended-MAD) is calibrated well enough for aggregate chi2
    # weighting but not necessarily for this ratio-of-~5-to-hundreds
    # significance test on a single narrow feature, so only a real
    # measured/propagated sigma is passed through; detect_peak_q falls
    # back to its own coarse noise estimate otherwise.
    peak_sigma = sigma if hygiene.sigma_model == "measured" else None
    morphology = classify_morphology(q, I, sigma=peak_sigma)
    active_windows = dict(propose_windows_from_classifier(q, I, morphology, q_cut=q_cut))
    if windows:
        active_windows.update(windows)  # user overrides win

    stages: Dict[str, Any] = {
        "stage0": {"class_guess": cls_guess, "prominence": prominence,
                  "morphology_cls": morphology.cls, "q_knee": morphology.q_knee,
                  "q_peak": morphology.q_peak, "hump_midq": morphology.hump_midq,
                  "n_trimmed_edge": hygiene.n_trimmed_edge,
                  "n_dropped_nonfinite": hygiene.n_dropped_nonfinite,
                  "n_points": int(q.size), "q_cut": q_cut,
                  "beamstop_trimmed_n": n_beamstop, "beamstop_trim_qmax": beamstop_trim_qmax,
                  # mask_regions itself lives on FitResult.mask_regions (the
                  # single source of truth) -- not duplicated here as tuples,
                  # which would break to_json()/from_json() round-tripping
                  # (JSON has no tuple type, so a nested copy would silently
                  # become a list of lists after one save/load cycle while
                  # the top-level field stays tuples).
                  "n_masked": int(excluded_mask.sum())},
    }

    stage1 = _stage1_bg(q, I, sigma, active_windows, sample_id=sample_id, residual_mode=residual_mode)
    sigma_scale = 1.0
    chi2red_plateau = float(stage1["result"].redchi)
    if residual_mode == "weighted_linear" and math.isfinite(chi2red_plateau) and chi2red_plateau > 0:
        # v3 §8.3 plateau calibration: fit bg(+pl) alone on the featureless
        # high-q plateau (exactly what _stage1_bg already does) and check
        # its chi2red. In [0.8, 1.25]: sigma is already realistic, use it
        # as-is (s=1). Outside that band: rescale ALL sigma by a single
        # global s=sqrt(chi2red_plateau) so the plateau's OWN calibrated
        # chi2red becomes exactly 1.0 (a uniform sigma rescale can't move
        # any stage's best-fit VALUES -- weighted least squares is
        # invariant to a global weight scale -- so this only recalibrates
        # reported chi2/AIC/BIC/uncertainty, never the fit itself). An
        # extreme scale (s>2 or s<0.5) additionally raises a
        # reduction_warning flag rather than silently absorbing it --
        # that large a mismatch is itself worth the user's attention.
        if not (0.8 <= chi2red_plateau <= 1.25):
            sigma_scale = math.sqrt(chi2red_plateau)
            sigma = sigma * sigma_scale
            flags.append("sigma_scaled")
            if sigma_scale > 2.0 or sigma_scale < 0.5:
                flags.append(f"reduction_warning:sigma_scale_extreme:{sigma_scale:.3g}")
            stage1 = _stage1_bg(q, I, sigma, active_windows, sample_id=sample_id, residual_mode=residual_mode)
    stages["stage1"] = {"redchi": float(stage1["result"].redchi), "chi2red_plateau_raw": chi2red_plateau,
                        "mask_n": int(stage1["mask"].sum()),
                        "pruned": stage1["pruned"], "sigma_scale": sigma_scale,
                        "bg_c_bounds": list(stage1["bg_c_bounds"])}
    # v4 §3: bg_C's plateau-derived bound is threaded through every later
    # stage that re-fits it as a free parameter (Stage 2/3/4 and the
    # ladder's own candidate fits) -- Stage 1 alone landing safely inside
    # this bound isn't enough on its own; the diagnosed P0Bi0 cascade
    # bug re-appeared at Stage 4's global release when only Stage 1 had
    # the bound applied.
    bg_c_bounds = tuple(stage1["bg_c_bounds"])
    current_model, current_result = stage1["model"], stage1["result"]
    preset_names = ["flat_background", "power_law"]
    had_ts = False
    pruned: List[str] = list(stage1["pruned"])
    if pruned:
        flags.append(f"pl_pruned:{','.join(pruned)}")

    stage2 = _stage2_add_ts(q, I, sigma, active_windows, stage1, residual_mode=residual_mode,
                            bg_c_bounds=bg_c_bounds)
    if stage2 is not None:
        ok, reason = ts_guardrail_ok(stage2["result"], sigma[stage2["mask"]], active_windows)
        if not ok and reason == "ts_not_significant":
            # v4 §2: a global-significance rejection can be a false
            # negative when the curve's overall chi2red is poisoned by
            # unmodeled structure OUTSIDE W_peak (the diagnosed P5Bi5-12
            # case) -- check whether TS meaningfully improves the fit
            # WITHIN W_peak alone before giving up on it.
            delta_bic_local = ts_window_local_delta_bic(
                q, I, sigma, active_windows, stage1["result"].params,
                stage2["model"], stage2["result"], residual_mode=residual_mode)
            if delta_bic_local is not None and delta_bic_local > 10.0:
                ok, reason = True, f"ts_accepted_local_delta_bic:{delta_bic_local:.1f}"
        if ok and cls_guess == "a" and morphology.cls not in ("F+P", "S+P"):
            # Stage 0's class guess is itself now prominence-based via
            # scipy.signal.find_peaks (robust to noise-driven false
            # positives), and Stage A's own morphology classifier agrees
            # no peak was found -- an extra backstop alongside the
            # guardrail's own significance/q_max-in-window checks (and
            # the local-delta-BIC override above), not a replacement.
            ok, reason = False, "class_guess_featureless"
        stages["stage2"] = {"redchi": float(stage2["result"].redchi), "mask_n": int(stage2["mask"].sum()),
                            "guardrail_ok": ok, "guardrail_reason": reason}
        if ok:
            current_model, current_result = stage2["model"], stage2["result"]
            preset_names = ["flat_background", "power_law", "teubner_strey"]
            had_ts = True
        else:
            flags.append(f"ts_rejected:{reason}")
    else:
        stages["stage2"] = {"skipped": "insufficient_points_in_window"}
        flags.append("ts_skipped_insufficient_window")

    # v4 §2/§4: Stage A's own q_knee (fixed-range, runs before windows
    # exist) decides the low-q branch, replacing detect_guinier_knee's
    # W_loq-dependent check -- W_loq itself is now BUILT from q_knee (see
    # propose_windows_from_classifier), so gating on a window-dependent
    # re-detection here would be circular for a peak-free S-class curve,
    # and detect_guinier_knee's OWN window came from the old, generally
    # peak-artifact-prone propose_windows for exactly the samples (no
    # peak, no reference W_peak edge) where that circularity bites.
    has_knee = morphology.q_knee is not None
    stage3_alt_name = None
    stage3_alt = None
    stage3_alt_model = stage3_alt_result = None
    if has_knee:
        # v5: try BOTH class-anchored knee-level candidates with the SAME
        # staged (frozen bg/pl, q_knee-seeded) treatment, rather than
        # picking one by has_knee alone and leaving the other to arrive at
        # the ladder as a hastily fresh-seeded sibling (measured to fit
        # much worse purely from the weaker seeding, not genuine model
        # inferiority -- see _stage3_add_beaucage's own docstring). The
        # better of the two (by chi2) becomes the primary assembled path;
        # the other is carried forward and ALSO given its own Stage 4
        # global polish below, then handed to the ladder as a precomputed
        # candidate so BIC compares two fairly-optimized fits.
        stage3_gp = _stage3_add_gp(q, I, sigma, active_windows, {"result": current_result}, had_ts,
                                   residual_mode=residual_mode, bg_c_bounds=bg_c_bounds,
                                   q_knee=morphology.q_knee)
        stage3_bc = _stage3_add_beaucage(q, I, sigma, active_windows, {"result": current_result}, had_ts,
                                         residual_mode=residual_mode, bg_c_bounds=bg_c_bounds,
                                         q_knee=morphology.q_knee)
        knee_candidates = [(name, s) for name, s in
                          (("guinier_porod", stage3_gp), ("beaucage_unified", stage3_bc)) if s is not None]
        if knee_candidates:
            knee_candidates.sort(key=lambda t: t[1]["result"].redchi)
            stage3_component, stage3 = knee_candidates[0]
            if len(knee_candidates) > 1:
                stage3_alt_name, stage3_alt = knee_candidates[1]
        else:
            stage3, stage3_component = None, "guinier_porod"
    else:
        stage3 = _stage3_add_pl2(q, I, sigma, active_windows, {"result": current_result}, had_ts,
                                 residual_mode=residual_mode, bg_c_bounds=bg_c_bounds)
        stage3_component = "power_law2"
    if stage3 is not None:
        stages["stage3"] = {"redchi": float(stage3["result"].redchi), "mask_n": int(stage3["mask"].sum()),
                            "component": stage3_component, "has_knee": has_knee}
        current_model, current_result = stage3["model"], stage3["result"]
        preset_names = preset_names + [stage3_component]
    else:
        stages["stage3"] = {"skipped": "insufficient_points_in_window", "has_knee": has_knee}
        flags.append(f"{stage3_component}_skipped_insufficient_window")

    best_values = {name: current_result.params[name].value for name in current_result.params}
    fixed_params: List[str] = ["pl_B", "pl_p"] if "power_law" in pruned else []
    if stage3_component == "power_law2" and stage3 is not None:
        # v3 §4: freeze pl2_p2 for the global stage UNLESS its own profile
        # shows real sensitivity -- kills the pl2_B2~pl2_p2 anti-
        # correlation that was leaking into the fitted TS width, while
        # still releasing p2 on the rare curve where it actually matters.
        if _pl2_sensitive(current_model, best_values, q, I, sigma, residual_mode):
            flags.append("pl2_p2_sensitive_kept_free")
        else:
            fixed_params.append("pl2_p2")
            flags.append("pl2_p2_frozen")
    stage4 = _stage4_global(q, I, sigma, current_model, best_values, sample_id, multistart_n,
                            residual_mode=residual_mode, fixed_params=fixed_params or None,
                            bg_c_bounds=bg_c_bounds)
    stages["stage4"] = {"redchi": float(stage4["result"].redchi), "n_multistart": multistart_n}
    assembled_model, assembled_result = stage4["model"], stage4["result"]

    if stage3_alt is not None:
        alt_best_values = {name: stage3_alt["result"].params[name].value for name in stage3_alt["result"].params}
        alt_fixed_params: List[str] = ["pl_B", "pl_p"] if "power_law" in pruned else []
        alt_stage4 = _stage4_global(q, I, sigma, stage3_alt["model"], alt_best_values, sample_id, multistart_n,
                                    residual_mode=residual_mode, fixed_params=alt_fixed_params or None,
                                    bg_c_bounds=bg_c_bounds)
        stage3_alt_model, stage3_alt_result = alt_stage4["model"], alt_stage4["result"]

    assembled_name = {
        ("flat_background", "power_law"): "BG",
        ("flat_background", "power_law", "guinier_porod"): "BG_GP",
        ("flat_background", "power_law", "beaucage_unified"): "BG_BC",
        ("flat_background", "power_law", "power_law2"): "BG_PL2",
        ("flat_background", "power_law", "teubner_strey"): "BG_TS",
        ("flat_background", "power_law", "teubner_strey", "power_law2"): "BG_TS_PL2",
        ("flat_background", "power_law", "teubner_strey", "guinier_porod"): "BG_TS_GP",
        ("flat_background", "power_law", "teubner_strey", "beaucage_unified"): "BG_TS_BC",
    }.get(tuple(preset_names), "+".join(preset_names))

    if force_preset is not None:
        if force_preset == assembled_name:
            final_model, final_result = assembled_model, assembled_result
        else:
            forced_names = PRESETS.get(force_preset, force_preset.split("+"))
            forced_fit = _fit_full_range(forced_names, q, I, sigma, sample_id, multistart_n,
                                         residual_mode=residual_mode, windows=active_windows,
                                         bg_c_bounds=bg_c_bounds)
            final_model, final_result = forced_fit["model"], forced_fit["result"]
        preset_chosen = force_preset
        stages["stage6"] = {"forced": force_preset}
    else:
        precomputed: Dict[str, Tuple[CompositeModel, Any]] = {}
        if stage3_alt_result is not None:
            alt_component_name = {"guinier_porod": "GP", "beaucage_unified": "BC"}[stage3_alt_name]
            alt_preset_name = f"BG_TS_{alt_component_name}" if had_ts else f"BG_{alt_component_name}"
            precomputed[alt_preset_name] = (stage3_alt_model, stage3_alt_result)
        stage6 = select_best_preset(q, I, sigma, assembled_name, assembled_model, assembled_result,
                                    sample_id, multistart_n, residual_mode=residual_mode,
                                    had_ts=had_ts, has_knee=has_knee, windows=active_windows,
                                    bg_c_bounds=bg_c_bounds, precomputed=precomputed or None)
        stages["stage6"] = stage6["ladder"]
        preset_chosen = stage6["chosen"]
        final_model, final_result = stage6["model"], stage6["result"]
        if preset_chosen != assembled_name:
            flags.append(f"ladder_demoted:{assembled_name}->{preset_chosen}")
        if stage6["ladder"].get("visual_equivalence_gate_bypassed") == preset_chosen:
            flags.append("visual_equivalence_gate_bypassed")

    no_peak = "teubner_strey" not in {comp.name for _, comp in final_model.components}
    if no_peak and had_ts:
        flags.append("no_peak")  # TS was fit through stages 1-4 but the ladder rejected it on BIC

    # v4 §5: systematic-error floor, fit against the FINAL chosen model --
    # makes chi2red meaningful series-wide (propagated sigma measures
    # counting precision only; real curves here carry ~5-10% smooth
    # systematics the sigma column never captures). sigma_eff feeds EVERY
    # downstream chi2/AIC/BIC/profile-CI computation (v3's one-statistic-
    # everywhere consistency rule, now extended to include this term).
    f_systematic = fit_systematic_floor(final_model, final_result.params, q, I, sigma, active_windows)
    sigma_eff = np.sqrt(sigma ** 2 + (f_systematic * I) ** 2)
    if f_systematic > 0.12:
        flags.append(f"data_systematics_high:f={f_systematic:.3f}")

    diagnostics = compute_diagnostics(final_model, final_result, q, I, active_windows, sigma=sigma_eff)
    stages["stage5"] = diagnostics
    stages["stage5"]["f_systematic"] = f_systematic
    flags.extend(diagnostics["flags"])

    # v3 §5: at_bounds_suggest_simpler_preset now requires the simpler
    # preset to have actually been tried by the ladder AND be genuinely
    # comparable (within 10 BIC, rms_log not worse by more than 0.1) --
    # previously fired unconditionally on >=2 at-bound params, which could
    # suggest a preset that's actually far worse (e.g. BG_TS_PL2 wrongly
    # suggesting BG_TS when BG_TS was a much poorer fit).
    at_bound_count = sum(1 for f in diagnostics["flags"] if f.startswith("at_bound:"))
    if at_bound_count >= 2:
        suggestion = _SIMPLER_PRESET.get(preset_chosen, preset_chosen)
        ladder_info = stages["stage6"] if isinstance(stages["stage6"], dict) else {}
        current_entry = ladder_info.get(preset_chosen, {})
        simpler_entry = ladder_info.get(suggestion, {})
        current_bic, current_rms = current_entry.get("bic"), diagnostics["gof"]["rms_log"]
        simpler_bic, simpler_rms = simpler_entry.get("bic"), simpler_entry.get("rms_log")
        if (suggestion != preset_chosen and None not in (current_bic, simpler_bic, simpler_rms)
                and (simpler_bic - current_bic) < 10.0 and (simpler_rms - current_rms) < 0.1):
            flags.append(f"at_bounds_suggest_simpler_preset:{suggestion}")

    # v3 §2/§3: profile-likelihood CIs and the peak-focused Stage 2b
    # cross-check, only meaningful when the final model actually has a TS
    # peak.
    ci_info: Dict[str, Any] = {}
    d_peak = xi_peak = None
    d_unreliable = xi_unreliable = False
    if not no_peak:
        ci_info = compute_ts_profile_likelihood_cis(final_model, final_result, q, I, sigma_eff,
                                                    residual_mode=residual_mode)
        # d_ci/xi_ci (tuples) are NOT duplicated into `stages` here -- they
        # live solely on FitResult.d_ci/xi_ci (the single source of truth,
        # same pattern as mask_regions), since a tuple nested inside the
        # free-form `stages` dict silently becomes a list after one JSON
        # save/load round-trip while the typed top-level field stays a
        # tuple, breaking to_json()/from_json() equality.
        if ci_info.get("xi_unidentifiable"):
            flags.append("xi_unidentifiable")

        stage2b = stage2b_peak_crosscheck(final_model, final_result, q, I, sigma_eff, active_windows,
                                          residual_mode=residual_mode)
        if stage2b is not None:
            d_global = float(final_result.params["ts_d"].value)
            xi_global = float(final_result.params["ts_xi"].value)
            d_peak = float(stage2b["result"].params["ts_d"].value)
            xi_peak = float(stage2b["result"].params["ts_xi"].value)
            stages["stage2b"] = {"d_peak": d_peak, "xi_peak": xi_peak, "mask_n": int(stage2b["mask"].sum())}
            if xi_global and abs(xi_peak - xi_global) / abs(xi_global) > 0.3:
                xi_unreliable = True
                flags.append("xi_unreliable")
            if d_global and abs(d_peak - d_global) / abs(d_global) > 0.1:
                d_unreliable = True
                flags.append("d_unreliable")
        else:
            stages["stage2b"] = {"skipped": "insufficient_points_in_window"}

    return FitResult(
        sample_id=sample_id, preset_chosen=preset_chosen, residual_mode=residual_mode, loss=loss,
        windows=active_windows, sigma_model=hygiene.sigma_model,
        params=_params_to_dict(final_result.params, chi2red=diagnostics["gof"]["chi2red"]),
        derived=_build_derived(final_model, final_result.params, pruned=pruned),
        gof=diagnostics["gof"], flags=flags, seeds_used=best_values,
        multistart_n=multistart_n, no_peak=no_peak, stages=stages, pruned=pruned,
        rms_log=diagnostics["gof"]["rms_log"], q_cut=q_cut, mask_regions=active_mask_regions,
        sigma_scale=sigma_scale,
        d_ci=ci_info.get("d_ci"), xi_ci=ci_info.get("xi_ci"),
        xi_unidentifiable=bool(ci_info.get("xi_unidentifiable", False)), fa_bound=ci_info.get("fa_bound"),
        d_peak=d_peak, xi_peak=xi_peak, xi_unreliable=xi_unreliable, d_unreliable=d_unreliable,
        d_ci_stat=ci_info.get("d_ci_stat"), xi_ci_stat=ci_info.get("xi_ci_stat"),
        morphology_cls=morphology.cls, q_knee=morphology.q_knee, q_peak=morphology.q_peak,
        hump_midq=morphology.hump_midq, f_systematic=f_systematic,
    )
