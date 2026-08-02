"""
htxrd_science.py — high-temperature XRD series processing (M20),
framework-agnostic.

Scope per the plan (informed by Jana2020's "cyclic refinement" workflow and
GSAS-II's sequential refinement, deliberately scoped DOWN from full
Rietveld — see the M20 milestone text for the reasoning):
  - Directory import of an ordered pattern series, ramp variable
    (temperature) derived from each file's own metadata (.rasx Temp axis,
    already parsed by io_universal.parse_rasx) or from a Jana-style
    filename template ("NB-LM01MO_???.XRDML" — rewrite the varying
    substring with ?'s).
  - Peak tracking: fit one peak (fitting_science.fit_spectrum — the same
    single entry point everything else uses) independently per pattern
    inside a user-chosen 2theta window; report center/FWHM/amplitude/area
    + fit quality per pattern.
  - Phase-transition-window flagging: a fit-quality (chi2_red) series
    diagnostic — patterns where the single-peak model abruptly degrades
    are flagged, the lightweight analog of Jana's GOF-vs-temperature plot
    (literally how the CaTeO3 tutorial example located its dehydration
    transition windows).
"""
from __future__ import annotations

import ntpath
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import core.io_universal as io_universal


# =============================================================================
# Series loading
# =============================================================================

@dataclass
class HtxrdPattern:
    path: str
    name: str
    x: np.ndarray  # 2theta (deg)
    y: np.ndarray  # intensity
    ramp_value: Optional[float] = None  # temperature (deg C) or template-derived number
    ramp_source: str = "none"  # "metadata" | "filename" | "index" | "none"
    meta: Dict[str, Any] = field(default_factory=dict)


def ramp_value_from_template(filename: str, template: str) -> Optional[float]:
    """Jana-style filename template: the varying substring is written as
    one or more '?' characters (e.g. 'NB-LM01MO_???.XRDML' matches
    'NB-LM01MO_100.XRDML' -> 100.0). Matching is case-insensitive and only
    against the basename. Returns None when the filename doesn't fit the
    template or the varying part isn't numeric."""
    # ntpath (not os.path) deliberately: real usage is always a Windows
    # path (PRISM is Windows-only), and ntpath.basename() strips both /
    # and \ regardless of the HOST platform, unlike os.path.basename()
    # which only strips \ when actually running on Windows -- without
    # this, a Windows-style path is left un-stripped when this runs on
    # Linux (e.g. in CI), and the template match silently fails.
    base = ntpath.basename(filename)
    m = re.search(r"\?+", template)
    if not m:
        return None
    n_wild = m.end() - m.start()
    pattern = re.escape(template[: m.start()]) + rf"(.{{{n_wild}}})" + re.escape(template[m.end():]) + "$"
    match = re.match(pattern, base, flags=re.IGNORECASE)
    if not match:
        return None
    token = match.group(1).strip()
    token = token.replace(",", ".")
    try:
        return float(token)
    except ValueError:
        digits = re.sub(r"[^\d.+-]", "", token)
        try:
            return float(digits) if digits else None
        except ValueError:
            return None


def load_pattern(path: str) -> HtxrdPattern:
    """Load one diffraction pattern via io_universal's parser registry
    (so .rasx, .xy, generic text all work), picking up the .rasx Temp-axis
    midpoint as the ramp value when present. Each HTXRD scan is collected
    over a temperature RANGE while the sample keeps heating (Position ->
    EndPosition), so the midpoint is the single representative value."""
    df, meta = io_universal.load_any(path, return_meta=True)
    canon = meta.get("canonical_map", {}) or {}
    x_col = canon.get("X") or df.columns[0]
    y_col = canon.get("Y") or df.columns[1]
    x = df[x_col].astype(float).to_numpy()
    y = df[y_col].astype(float).to_numpy()
    order = np.argsort(x, kind="mergesort")
    x, y = x[order], y[order]

    ramp_value = None
    ramp_source = "none"
    t0, t1 = meta.get("temp_start_C"), meta.get("temp_end_C")
    if t0 is not None:
        ramp_value = (float(t0) + float(t1)) / 2.0 if t1 is not None else float(t0)
        ramp_source = "metadata"

    return HtxrdPattern(
        path=str(path), name=os.path.splitext(os.path.basename(path))[0],
        x=x, y=y, ramp_value=ramp_value, ramp_source=ramp_source, meta=meta,
    )


def load_series(
    paths: List[str], *, filename_template: str = "",
) -> List[HtxrdPattern]:
    """Load a whole ordered series. Ramp-value resolution per pattern, in
    priority order: file metadata (.rasx Temp axis) -> filename template
    (if provided) -> sequence index (fallback, so the series is still
    plottable/trackable even with no temperature information at all).
    Returned sorted by ramp value (stable for ties)."""
    patterns: List[HtxrdPattern] = []
    for i, p in enumerate(paths):
        pat = load_pattern(p)
        if pat.ramp_value is None and filename_template:
            v = ramp_value_from_template(p, filename_template)
            if v is not None:
                pat.ramp_value = v
                pat.ramp_source = "filename"
        if pat.ramp_value is None:
            pat.ramp_value = float(i)
            pat.ramp_source = "index"
        patterns.append(pat)

    patterns.sort(key=lambda p: p.ramp_value)
    return patterns


def find_series_files(folder: str, *, extensions: Tuple[str, ...] = (".rasx", ".xy", ".xrdml", ".txt", ".dat")) -> List[str]:
    """All candidate pattern files in a folder (non-recursive), name-sorted."""
    out = []
    try:
        for entry in os.scandir(folder):
            if entry.is_file() and entry.name.lower().endswith(tuple(e.lower() for e in extensions)):
                out.append(entry.path)
    except OSError:
        return []
    return sorted(out)


# =============================================================================
# Peak tracking across the series.
#
# Reworked after real use (user feedback): the first version fit one
# window-max-seeded Gaussian independently per pattern, which breaks when
# (a) the window contains several peaks — the seed jumps to whichever is
# strongest, (b) the tracked peak disappears — the fit chases noise and
# reports garbage values, or (c) a new peak appears. Now:
#   - sequential seeding (GSAS-II style): each pattern starts from the
#     previous pattern's accepted fit,
#   - an adaptive sub-window follows the seeded center, so a neighboring
#     peak in the wider user window doesn't capture the fit,
#   - an optional initial_center anchors WHICH peak in the window to track,
#   - an amplitude-significance test (vs the window's high-frequency noise)
#     marks the peak absent instead of reporting a nonsense fit, and lets
#     it reappear later (re-seeded from the anchor),
#   - several windows can be tracked in one run (track_peaks_multi).
# =============================================================================

@dataclass
class PeakTrackResult:
    pattern_name: str
    ramp_value: float
    center: float
    fwhm: float
    amplitude: float
    area: float
    chi2_red: float
    error: Optional[str] = None
    present: bool = True        # False: peak not significant vs noise here
    noise: float = float("nan")  # the high-frequency noise σ used for the test
    window_label: str = ""


def _local_hwhm_estimate(x: np.ndarray, y_net: np.ndarray, idx: int, span: float) -> float:
    """Rough HWHM of the peak whose apex is at index idx: walk down each
    side to half the apex height. The MIN of the two sides is used so an
    overlapping neighbor on one side doesn't inflate the estimate."""
    apex = float(y_net[idx])
    if apex <= 0:
        return span * 0.05
    half_level = apex / 2.0
    i = idx
    while i > 0 and y_net[i] > half_level:
        i -= 1
    j = idx
    while j < len(y_net) - 1 and y_net[j] > half_level:
        j += 1
    sides = [d for d in (float(x[idx] - x[i]), float(x[j] - x[idx])) if d > 0]
    est = min(sides) if sides else span * 0.05
    return float(np.clip(est, span * 0.01, span * 0.25))


def _peak_support(x: np.ndarray, y_net: np.ndarray, idx: int, noise: float,
                  hwhm: float, span: float) -> Tuple[float, float]:
    """Fit-region bounds around the apex at idx: extend each side while the
    signal keeps falling, and STOP where it starts climbing again (the
    valley before a neighboring peak) or at ±3 HWHM, whichever is tighter.
    This is what keeps a single-peak fit on the intended peak in a window
    that contains several."""
    apex = float(y_net[idx])
    rise_tol = max(3.0 * noise, 0.10 * abs(apex))
    cap = max(3.0 * hwhm, span * 0.03)

    i = idx
    run_min = apex
    while i > 0 and (x[idx] - x[i - 1]) <= cap:
        nxt = float(y_net[i - 1])
        if nxt < run_min:
            run_min = nxt
        elif nxt > run_min + rise_tol:
            break
        i -= 1
    j = idx
    run_min = apex
    while j < len(y_net) - 1 and (x[j + 1] - x[idx]) <= cap:
        nxt = float(y_net[j + 1])
        if nxt < run_min:
            run_min = nxt
        elif nxt > run_min + rise_tol:
            break
        j += 1
    return float(x[i]), float(x[j])


def estimate_window_noise(y: np.ndarray) -> float:
    """High-frequency noise σ from SECOND differences (variance 6σ² for
    white noise). First differences read a peak's smooth slope as noise —
    found the hard way when guided tracking's narrow windows sat entirely
    on a peak flank and declared clearly-present peaks absent; the second
    difference of a smooth curve is ~h²·y'' and stays near zero."""
    d2 = np.diff(np.asarray(y, float), n=2)
    d2 = d2[np.isfinite(d2)]
    if len(d2) < 4:
        return 0.0
    mad = float(np.median(np.abs(d2 - np.median(d2))))
    return 1.4826 * mad / np.sqrt(6.0)


def track_peak(
    patterns: List[HtxrdPattern], *, window_lo: float, window_hi: float,
    shape: str = "G", seed_from_previous: bool = True,
    initial_center: Optional[float] = None, absence_sigma: float = 3.0,
    window_label: str = "",
) -> List[PeakTrackResult]:
    """Fit ONE peak per pattern within [window_lo, window_hi] 2θ and report
    center/FWHM/amplitude/area + chi2_red per pattern — the tractable analog
    of Jana's cell-parameter-vs-temperature Graph tool. Per-pattern failures
    become error rows, absent peaks become present=False rows; neither
    aborts the series (a peak vanishing above a transition is an expected
    outcome, not an exceptional one).

    initial_center picks WHICH peak to track when the window holds several
    (seed = the data maximum near it, not the window-wide maximum);
    seed_from_previous then keeps the fit locked on it as it drifts."""
    from core.fitting_science import fit_spectrum

    if window_hi <= window_lo:
        raise ValueError("window_hi must be greater than window_lo")
    span = window_hi - window_lo

    results: List[PeakTrackResult] = []
    prev_fit: Optional[Tuple[float, float, float]] = None  # (center, hwhm, amp) of last accepted fit

    for pat in patterns:
        mask = (pat.x >= window_lo) & (pat.x <= window_hi)
        x_win, y_win = pat.x[mask], pat.y[mask]
        if len(x_win) < 8:
            results.append(PeakTrackResult(pat.name, pat.ramp_value, np.nan, np.nan, np.nan, np.nan, np.nan,
                                           error=f"only {len(x_win)} points in window", present=False,
                                           window_label=window_label))
            continue

        baseline = float(np.nanmin(y_win))
        y_net = y_win - baseline  # simple constant-baseline removal for the local window
        # Noise from the WHOLE pattern, not the window: a narrow window can
        # be all peak, leaving no baseline to measure noise on.
        noise = estimate_window_noise(pat.y)

        # ---- seed: previous accepted fit > initial_center anchor > window max
        if seed_from_previous and prev_fit is not None:
            seed_center, hwhm0, amp0 = prev_fit
            # apex = the data maximum near the previous center (the peak may
            # have drifted since the last pattern). ±1.5 HWHM: wide enough
            # for realistic per-step drift, tight enough that a neighboring
            # peak's flank can't out-shine the tracked apex at the range edge.
            near = np.abs(x_win - seed_center) <= max(1.5 * hwhm0, span * 0.02)
            if np.any(near):
                idx_local = int(np.nanargmax(np.where(near, y_net, -np.inf)))
            else:
                idx_local = int(np.nanargmax(y_net))
            fwhm_guess = hwhm0
        else:
            if initial_center is not None:
                near = np.abs(x_win - float(initial_center)) <= max(span * 0.1, float(np.diff(x_win).mean()) * 5)
                if np.any(near):
                    idx_local = int(np.nanargmax(np.where(near, y_net, -np.inf)))
                else:
                    idx_local = int(np.nanargmax(y_net))
            else:
                idx_local = int(np.nanargmax(y_net))
            fwhm_guess = _local_hwhm_estimate(x_win, y_net, idx_local, span)
        center0 = float(x_win[idx_local])
        amp_guess = max(float(y_net[idx_local]), 1e-9)

        # Fit region: from the apex out to the valley before any neighboring
        # peak (or ±3 HWHM) — a stronger neighbor elsewhere in the user
        # window can capture neither the seed nor the fit.
        lo_fit, hi_fit = _peak_support(x_win, y_net, idx_local, noise, fwhm_guess, span)
        sub = (x_win >= lo_fit) & (x_win <= hi_fit)
        if np.count_nonzero(sub) >= 8:
            x_fit, y_fit_in = x_win[sub], y_net[sub]
        else:
            x_fit, y_fit_in, lo_fit, hi_fit = x_win, y_net, window_lo, window_hi

        # ---- fast absence test before fitting: nothing above the noise
        # floor. Significance is measured above the window MEDIAN — y_net's
        # baseline is the window minimum, so pure noise alone already shows
        # max-minus-min ≈ 5σ and would always look "present" otherwise.
        floor = float(np.median(y_net))
        if noise > 0 and (float(np.nanmax(y_fit_in)) - floor) < absence_sigma * noise:
            results.append(PeakTrackResult(pat.name, pat.ramp_value, np.nan, np.nan, 0.0, 0.0, np.nan,
                                           present=False, noise=noise, window_label=window_label))
            prev_fit = None  # reappearing peak re-anchors on initial_center / window max
            continue

        comp = {
            "shape": shape,
            "shift_val": center0, "shift_min": lo_fit, "shift_max": hi_fit, "fit_shift": True,
            "fwhm_val": fwhm_guess, "fwhm_min": span * 0.005, "fwhm_max": span, "fit_fwhm": True,
            "eta_val": 0.5, "eta_min": 0.0, "eta_max": 1.0, "fit_eta": True,
            "amp_val": amp_guess, "fit_amp": True,
        }
        try:
            fr = fit_spectrum(x_fit, y_fit_in, [comp], mode="classic")
            params = fr.lmfit_result.params
            center = float(params["f0"].value)
            fwhm = float(params["l0"].value)
            amplitude = float(params["a0"].value)
            area = float(np.trapz(fr.peaks[0], x_fit))

            # ---- post-fit significance: a "fit" of noise is an absence
            if noise > 0 and (amplitude - floor) < absence_sigma * noise:
                results.append(PeakTrackResult(pat.name, pat.ramp_value, np.nan, np.nan, amplitude, 0.0,
                                               fr.chi2_red, present=False, noise=noise,
                                               window_label=window_label))
                prev_fit = None
                continue

            results.append(PeakTrackResult(pat.name, pat.ramp_value, center, fwhm, amplitude, area,
                                           fr.chi2_red, noise=noise, window_label=window_label))
            prev_fit = (center, fwhm, amplitude)
        except Exception as exc:
            results.append(PeakTrackResult(pat.name, pat.ramp_value, np.nan, np.nan, np.nan, np.nan, np.nan,
                                           error=str(exc), present=False, noise=noise,
                                           window_label=window_label))
            prev_fit = None
    return results


def parse_track_windows(text: str) -> List[Dict[str, Optional[float]]]:
    """Parse the tracking-windows field: windows separated by ';', each
    'lo-hi' with an optional '@ center' anchor picking which peak to track,
    e.g. '28.5-29.5 @ 28.98; 31-32'. Returns [{lo, hi, center}, ...]."""
    out: List[Dict[str, Optional[float]]] = []
    for part in (text or "").split(";"):
        part = part.strip()
        if not part:
            continue
        anchor: Optional[float] = None
        if "@" in part:
            part, anchor_text = part.split("@", 1)
            anchor = float(anchor_text.strip())
        m = re.match(r"^\s*([\d.]+)\s*-\s*([\d.]+)\s*$", part)
        if not m:
            raise ValueError(f"Cannot parse tracking window '{part.strip()}' (expected 'lo-hi' or 'lo-hi @ center')")
        lo, hi = float(m.group(1)), float(m.group(2))
        if hi <= lo:
            raise ValueError(f"Tracking window '{part.strip()}': hi must exceed lo")
        if anchor is not None and not (lo <= anchor <= hi):
            raise ValueError(f"Anchor {anchor:g} lies outside window {lo:g}-{hi:g}")
        out.append({"lo": lo, "hi": hi, "center": anchor})
    return out


def track_peaks_multi(
    patterns: List[HtxrdPattern], windows: List[Dict[str, Optional[float]]], *,
    shape: str = "G", seed_from_previous: bool = True, absence_sigma: float = 3.0,
) -> Dict[str, List[PeakTrackResult]]:
    """Track several windows in one run (user request: real series follow
    more than one reflection). Keys are display labels like '28.5-29.5'."""
    out: Dict[str, List[PeakTrackResult]] = {}
    for w in windows:
        label = f"{w['lo']:g}-{w['hi']:g}" + (f" @ {w['center']:g}" if w.get("center") is not None else "")
        out[label] = track_peak(
            patterns, window_lo=w["lo"], window_hi=w["hi"], shape=shape,
            seed_from_previous=seed_from_previous, initial_center=w.get("center"),
            absence_sigma=absence_sigma, window_label=label,
        )
    return out


def guide_centers_for_patterns(guide_points: List[Tuple[int, float]], n_patterns: int) -> np.ndarray:
    """Expected 2θ center per pattern from picked guide anchors
    [(slice_1based, x), ...]: linear interpolation between anchors, FLAT
    extrapolation beyond the first/last one — so an anchor picked at an
    intermediate temperature still defines a full-series guide (the absence
    test decides where the peak actually exists)."""
    if not guide_points:
        raise ValueError("No guide anchors given.")
    pts = sorted(guide_points, key=lambda t: t[0])
    slices = np.array([p[0] for p in pts], float)
    xs = np.array([p[1] for p in pts], float)
    grid = np.arange(1, n_patterns + 1, dtype=float)
    return np.interp(grid, slices, xs)  # np.interp clamps outside the anchor range


def track_peak_guided(
    patterns: List[HtxrdPattern], guide_points: List[Tuple[int, float]], *,
    half_window: float = 0.3, shape: str = "G", absence_sigma: float = 3.0,
    window_label: str = "",
) -> List[PeakTrackResult]:
    """Track a peak along a PICKED guide (user request: click the peak at a
    start and an end temperature on the waterfall instead of typing window
    numbers). Each pattern gets its own window, centered on the guide's
    interpolated position — the window follows the peak by construction, so
    drift can never walk out of a fixed window — with the guide position as
    the anchor. Works for appearing/dying peaks: the guide extends flat
    across the whole series and absence detection reports where the peak
    isn't."""
    centers = guide_centers_for_patterns(guide_points, len(patterns))
    results: List[PeakTrackResult] = []
    for pat, center in zip(patterns, centers):
        rows = track_peak(
            [pat], window_lo=float(center - half_window), window_hi=float(center + half_window),
            shape=shape, seed_from_previous=False, initial_center=float(center),
            absence_sigma=absence_sigma, window_label=window_label,
        )
        results.extend(rows)
    return results


def auto_track_windows(
    pattern: HtxrdPattern, *, max_peaks: int = 10, min_prominence_sigma: float = 3.0,
) -> List[Dict[str, Optional[float]]]:
    """Automatic tracking setup (user request): find the peaks of one
    reference pattern and build an anchored tracking window around each,
    sized from the peak's own local width. Feed the result to
    track_peaks_multi — or show it in the windows field for editing."""
    from core.fitting_science import find_peak_candidates

    x, y = np.asarray(pattern.x, float), np.asarray(pattern.y, float)
    span = float(x[-1] - x[0]) if len(x) > 1 else 1.0
    centers = find_peak_candidates(x, y, max_peaks=max_peaks, min_prominence_sigma=min_prominence_sigma)
    dx = float(np.mean(np.diff(x))) if len(x) > 1 else 0.01

    windows: List[Dict[str, Optional[float]]] = []
    # _local_hwhm_estimate clips to [1%, 25%] of the span it's given; the
    # full pattern's span would floor sharp XRD peaks at far too wide an
    # estimate, so give it a local one.
    local_span = min(span, 2.0)
    for center in sorted(centers):
        idx = int(np.argmin(np.abs(x - center)))
        y_net = y - float(np.nanmin(y))
        hwhm = _local_hwhm_estimate(x, y_net, idx, local_span)
        half = max(6.0 * hwhm, 8.0 * dx)
        windows.append({"lo": round(float(center - half), 3), "hi": round(float(center + half), 3),
                        "center": round(float(center), 3)})
    return windows


def flag_transition_candidates(
    results: List[PeakTrackResult], *, z: float = 3.0,
) -> List[Tuple[float, float, str]]:
    """Flag ramp values where the tracked fit quality (chi2_red) jumps
    anomalously versus the series median — the lightweight analog of
    Jana's GOF-vs-temperature plot for locating phase-transition windows.
    Presence flips (peak vanished / appeared — THE transition signatures)
    and outright fit failures are flagged too.
    Returns (ramp_value, chi2_red, reason) tuples."""
    flags: List[Tuple[float, float, str]] = []
    valid = [r for r in results if r.error is None and r.present and np.isfinite(r.chi2_red)]
    if len(valid) >= 3:
        chi2s = np.array([r.chi2_red for r in valid], float)
        med = float(np.median(chi2s))
        mad = float(np.median(np.abs(chi2s - med)))
        robust_sigma = 1.4826 * mad if mad > 1e-30 else (float(np.std(chi2s)) or 1.0)
        for r in valid:
            if (r.chi2_red - med) > z * robust_sigma:
                flags.append((r.ramp_value, r.chi2_red, f"chi2_red {r.chi2_red:.4g} >> series median {med:.4g}"))

    prefix = ""
    for a, b in zip(results[:-1], results[1:]):
        if a.window_label:
            prefix = f"[{a.window_label}] "
        if a.error is None and b.error is None:
            if a.present and not b.present:
                flags.append((b.ramp_value, np.nan, f"{prefix}peak vanished at {b.ramp_value:g}"))
            elif not a.present and b.present:
                flags.append((b.ramp_value, np.nan, f"{prefix}peak appeared at {b.ramp_value:g}"))
    for r in results:
        if r.error is not None:
            flags.append((r.ramp_value, np.nan, f"fit failed: {r.error}"))
    flags.sort(key=lambda t: t[0])
    return flags


# =============================================================================
# Common-grid maps and derived plots (ported from the user's own HT-XRD
# plotting notebook — XRD_HT.ipynb: heatmap, difference map/waterfall vs a
# reference pattern, time axis from a heating rate, and interpolated peak
# guide lines).
# =============================================================================

def build_common_grid(patterns: List[HtxrdPattern], *, npts: int = 2000,
                      xmin: Optional[float] = None, xmax: Optional[float] = None) -> np.ndarray:
    """Common 2θ grid over the overlap of every pattern's range."""
    if not patterns:
        raise ValueError("No patterns loaded.")
    lo = max(float(np.nanmin(p.x)) for p in patterns)
    hi = min(float(np.nanmax(p.x)) for p in patterns)
    if xmin is not None:
        lo = max(lo, float(xmin))
    if xmax is not None:
        hi = min(hi, float(xmax))
    if lo >= hi:
        raise ValueError("No common 2θ range across the series.")
    return np.linspace(lo, hi, int(npts))


def build_intensity_map(patterns: List[HtxrdPattern], x_common: np.ndarray, *,
                        normalize: bool = False) -> np.ndarray:
    """(n_patterns × n_points) intensity matrix on the common grid — rows
    ordered like `patterns` (sorted by ramp value by load_series)."""
    rows = []
    for p in patterns:
        y = np.interp(x_common, p.x, p.y, left=np.nan, right=np.nan)
        if normalize:
            ymax = float(np.nanmax(y))
            if ymax > 0:
                y = y / ymax
        rows.append(y)
    return np.vstack(rows)


def reference_index(patterns: List[HtxrdPattern], reference) -> int:
    """Resolve the difference-plot reference: 'first', an integer row index,
    or a float ramp value (nearest pattern wins)."""
    if reference == "first":
        return 0
    if isinstance(reference, int):
        if not (0 <= reference < len(patterns)):
            raise ValueError(f"Reference index {reference} out of range (0-{len(patterns) - 1}).")
        return reference
    if isinstance(reference, float):
        ramps = np.array([p.ramp_value for p in patterns], float)
        return int(np.argmin(np.abs(ramps - reference)))
    raise ValueError("reference must be 'first', an int index, or a float ramp value")


def compute_relative_time_minutes(ramp_values, rate_c_per_min: Optional[float],
                                  ref_c: Optional[float] = None) -> Optional[np.ndarray]:
    """Relative time axis from a constant heating rate; None when no rate."""
    if not rate_c_per_min:
        return None
    ramps = np.asarray(ramp_values, float)
    if ref_c is None:
        ref_c = float(np.nanmin(ramps))
    return (ramps - ref_c) / float(rate_c_per_min)


def parse_peak_guides(strings: List[str]) -> List[List[Tuple[int, float]]]:
    """Parse guide definitions like '{1:21.2; 38:19.6}' (the notebook's
    syntax): at 1-based slice 1 the guide sits at 21.2° 2θ, at slice 38 at
    19.6°, linearly interpolated between anchors. Invalid entries raise."""
    guides: List[List[Tuple[int, float]]] = []
    for s in strings:
        s = (s or "").strip()
        if not s:
            continue
        if not (s.startswith("{") and s.endswith("}")):
            raise ValueError(f"Invalid guide format: {s} (expected '{{slice:2theta; ...}}')")
        pts: List[Tuple[int, float]] = []
        for part in s[1:-1].split(";"):
            part = part.strip()
            if not part:
                continue
            k, v = part.split(":")
            pts.append((int(k.strip()), float(v.strip())))
        pts.sort(key=lambda t: t[0])
        if len(pts) >= 2:
            guides.append(pts)
    return guides


def evaluate_peak_guide(points: List[Tuple[int, float]], n_slices: int) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolated guide x-position per slice (1-based slices; NaN outside
    the defined segments) — drawn over the waterfall and the heatmap."""
    slices = np.arange(1, n_slices + 1, dtype=float)
    xpos = np.full(n_slices, np.nan)
    for (s1, x1), (s2, x2) in zip(points[:-1], points[1:]):
        if s2 == s1:
            continue
        lo = max(1, min(s1, s2))
        hi = min(n_slices, max(s1, s2))
        seg = np.arange(lo, hi + 1, dtype=float)
        xpos[lo - 1:hi] = x1 + (x2 - x1) * (seg - s1) / (s2 - s1)
    return slices, xpos


def estimate_waterfall_shift(data: np.ndarray) -> float:
    """Auto vertical offset for waterfall plots: a quarter of the robust
    (5th-99th percentile) intensity range."""
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 1.0
    rng = float(np.nanpercentile(finite, 99) - np.nanpercentile(finite, 5))
    return 0.25 * rng if rng > 0 else 1.0
