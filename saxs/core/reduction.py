from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple
import math

import numpy as np

from .curve import Curve
from .chemistry import SamplePhysicsConfig, CapillaryConfig, calculate_absorption, theoretical_empty_transmission, theoretical_sample_transmission


@dataclass
class CorrectionSettings:
    scale_mode: str = "manual"
    manual_scale: float = 1.0
    q_match: Optional[float] = None
    target_ratio: float = 1.0
    clip_negative_q: bool = False
    remove_aberrations: bool = False
    final_normalization_mode: str = "none"
    final_normalization_value: float = 1.0
    recenter_enabled: bool = False
    recenter_value: float = 0.0
    recenter_mode: str = "high_q"
    tail_cleanup_mode: str = "repair_tail"
    tail_cleanup_points: int = 8

    def to_json(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class PhysicsBundle:
    sample_comp: Dict[str, object]
    sample_abs: Dict[str, object]
    capillary_comp: Dict[str, object]
    capillary_abs: Dict[str, object]
    theoretical_empty_transmission: float
    theoretical_filled_transmission: float


@dataclass
class CorrectionResult:
    q: np.ndarray
    sample_aligned: np.ndarray
    empty_aligned: np.ndarray
    empty_scaled: np.ndarray
    corrected_raw: np.ndarray
    corrected: np.ndarray
    sigma_corrected: Optional[np.ndarray]
    residual: np.ndarray
    scale_factor: float
    overlap_qmin: float
    overlap_qmax: float
    effective_sample_thickness_mm: Optional[float]
    sample_mu_linear_mm_inv: Optional[float]
    capillary_mu_linear_mm_inv: Optional[float]
    sample_transmission: Optional[float]
    empty_transmission: Optional[float]
    warnings: list[str]
    physics: PhysicsBundle
    final_normalization_mode: str = "none"
    final_normalization_factor: float = 1.0

    def summary(self) -> Dict[str, object]:
        return {
            "scale_factor": self.scale_factor,
            "overlap_qmin": self.overlap_qmin,
            "overlap_qmax": self.overlap_qmax,
            "effective_sample_thickness_mm": self.effective_sample_thickness_mm,
            "sample_mu_linear_mm_inv": self.sample_mu_linear_mm_inv,
            "capillary_mu_linear_mm_inv": self.capillary_mu_linear_mm_inv,
            "sample_transmission": self.sample_transmission,
            "empty_transmission": self.empty_transmission,
            "warnings": "; ".join(self.warnings),
            "theoretical_empty_transmission": self.physics.theoretical_empty_transmission,
            "theoretical_filled_transmission": self.physics.theoretical_filled_transmission,
            "final_normalization_mode": self.final_normalization_mode,
            "final_normalization_factor": self.final_normalization_factor,
        }



def compute_physics(sample_cfg: SamplePhysicsConfig, capillary_cfg: CapillaryConfig, energy_ev: float) -> PhysicsBundle:
    sample_density = sample_cfg.effective_density()
    sample_comp, sample_abs = calculate_absorption(
        composition_text=sample_cfg.composition_text,
        density_g_cm3=sample_density,
        energy_ev=energy_ev,
        mode=sample_cfg.composition_mode,
        basis=sample_cfg.mixture_basis,
    )
    cap_comp, cap_abs = calculate_absorption(
        composition_text=capillary_cfg.composition_text,
        density_g_cm3=capillary_cfg.density_g_cm3,
        energy_ev=energy_ev,
        mode=capillary_cfg.composition_mode,
        basis=capillary_cfg.mixture_basis,
    )
    if capillary_cfg.manual_mu_linear_mm_inv is not None:
        mu = float(capillary_cfg.manual_mu_linear_mm_inv)
        cap_abs.mu_linear_mm_inv = mu
        cap_abs.mu_linear_cm_inv = mu * 10.0
        cap_abs.attenuation_length_mm = 1.0 / max(mu, 1e-30)
        cap_abs.transmission_for_1mm = math.exp(-mu)
        cap_abs.mu_mass_cm2_g = cap_abs.mu_linear_cm_inv / max(float(capillary_cfg.density_g_cm3), 1e-30)

    empty_t = theoretical_empty_transmission(cap_abs.mu_linear_mm_inv, capillary_cfg.wall_thickness_mm)
    filled_t = theoretical_sample_transmission(
        capillary_mu_mm_inv=cap_abs.mu_linear_mm_inv,
        wall_thickness_mm=capillary_cfg.wall_thickness_mm,
        sample_mu_mm_inv=sample_abs.mu_linear_mm_inv,
        inner_diameter_mm=capillary_cfg.inner_diameter_mm,
    )
    return PhysicsBundle(
        sample_comp=sample_comp.to_json(),
        sample_abs=sample_abs.to_json(),
        capillary_comp=cap_comp.to_json(),
        capillary_abs=cap_abs.to_json(),
        theoretical_empty_transmission=empty_t,
        theoretical_filled_transmission=filled_t,
    )



def align_to_overlap(sample: Curve, empty: Curve) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    qmin = max(float(sample.q.min()), float(empty.q.min()))
    qmax = min(float(sample.q.max()), float(empty.q.max()))
    if qmax <= qmin:
        raise ValueError("No overlapping q-range between sample and empty.")

    mask = (sample.q >= qmin) & (sample.q <= qmax)
    q = sample.q[mask]
    sample_i = sample.intensity[mask]
    empty_i = np.interp(q, empty.q, empty.intensity)
    sigma = None
    if sample.sigma is not None and len(sample.sigma) == len(sample.q):
        sigma = sample.sigma[mask]
    return q, sample_i, empty_i, sigma



def _despike_signal(y: np.ndarray) -> np.ndarray:
    if y.size < 5:
        return np.array(y, copy=True)
    out = np.array(y, copy=True)
    logy = np.log10(np.clip(np.abs(out), 1e-30, None))
    med = np.copy(logy)
    for i in range(1, len(out)-1):
        lo = max(0, i-2)
        hi = min(len(out), i+3)
        med[i] = np.median(logy[lo:hi])
    delta = logy - med
    mad = np.median(np.abs(delta - np.median(delta)))
    thresh = max(4.5 * max(mad, 1e-6), 0.30)
    bad = np.abs(delta) > thresh

    # Be stricter on the edges, especially the last point which often glitches low.
    edge_window = min(5, len(out)-1)
    if edge_window >= 2:
        tail_ref = np.median(np.clip(np.abs(out[-edge_window-1:-1]), 1e-30, None))
        if np.isfinite(tail_ref) and tail_ref > 0:
            if out[-1] < 0.2 * tail_ref or out[-1] > 5.0 * tail_ref:
                bad[-1] = True
        head_ref = np.median(np.clip(np.abs(out[1:1+edge_window]), 1e-30, None))
        if np.isfinite(head_ref) and head_ref > 0:
            if out[0] < 0.2 * head_ref or out[0] > 5.0 * head_ref:
                bad[0] = True

    idx = np.where(bad)[0]
    for i in idx:
        lo = max(0, i-3)
        hi = min(len(out), i+4)
        neigh = np.delete(out[lo:hi], i - lo)
        neigh = neigh[np.isfinite(neigh)]
        if neigh.size:
            out[i] = float(np.median(neigh))
    return out




def _detect_invalid_terminal_run(y: np.ndarray, max_points: int = 6) -> int:
    """Return number of invalid trailing points to repair/trim."""
    n = int(min(max_points, max(0, y.size - 1)))
    if y.size < 4 or n <= 0:
        return 0
    out = np.asarray(y, dtype=float)
    count = 0
    for j in range(1, n + 1):
        idx = len(out) - j
        prev_hi = idx
        prev_lo = max(0, idx - 8)
        prev = out[prev_lo:prev_hi]
        prev = prev[np.isfinite(prev)]
        if prev.size < 2:
            break
        ref_med = float(np.median(np.abs(prev[-min(6, prev.size):])))
        ref_last = float(np.median(prev[-min(4, prev.size):]))
        yi = float(out[idx])
        if not np.isfinite(yi):
            count = j
            continue
        low_collapse = abs(yi) <= max(1e-30, 0.12 * max(ref_med, 1e-30))
        severe_drop = abs(yi) < 0.25 * max(abs(ref_last), 1e-30)
        severe_jump = abs(yi) > 8.0 * max(ref_med, 1e-30)
        sign_flip_glitch = (ref_last > 0 and yi < 0 and abs(yi) < 0.50 * abs(ref_last))
        if low_collapse or severe_drop or severe_jump or sign_flip_glitch:
            count = j
        else:
            break
    return count


def _detect_tail_anomaly_indices(y: np.ndarray, scan_points: int = 12) -> list[int]:
    """Detect anomalous points anywhere in the last scan_points, not just the final run."""
    arr = np.asarray(y, dtype=float)
    n = arr.size
    if n < 6:
        return []
    start = max(0, n - max(4, scan_points))
    bad: list[int] = []
    for idx in range(start, n):
        yi = float(arr[idx])
        if not np.isfinite(yi):
            bad.append(idx)
            continue

        prev = arr[max(0, idx - 6):idx]
        prev = prev[np.isfinite(prev)]
        if prev.size < 2:
            continue

        ref_med = float(np.median(np.abs(prev[-min(5, prev.size):])))
        ref_last = float(np.median(prev[-min(3, prev.size):]))

        # forward neighborhood when available helps catch isolated interior glitches near the tail
        nxt = arr[idx + 1:min(n, idx + 4)]
        nxt = nxt[np.isfinite(nxt)]
        neigh = np.concatenate([prev[-min(3, prev.size):], nxt[:min(3, nxt.size)]]) if nxt.size else prev[-min(3, prev.size):]
        neigh_med = float(np.median(np.abs(neigh))) if neigh.size else ref_med

        low_collapse = abs(yi) <= max(1e-30, 0.15 * max(ref_med, 1e-30))
        severe_drop = abs(yi) < 0.28 * max(abs(ref_last), 1e-30)
        severe_jump = abs(yi) > 8.0 * max(ref_med, 1e-30)
        off_neighborhood = neigh.size >= 2 and (
            abs(yi) < 0.30 * max(neigh_med, 1e-30) or abs(yi) > 5.0 * max(neigh_med, 1e-30)
        )

        if low_collapse or severe_drop or severe_jump or off_neighborhood:
            bad.append(idx)

    # collapse to unique sorted indices; if several bad tail points exist, treat from earliest bad onward for trim mode
    return sorted(set(bad))


def _repair_tail_anomalies(y: np.ndarray, bad_indices: list[int]) -> np.ndarray:
    if not bad_indices:
        return np.array(y, copy=True, dtype=float)
    out = np.array(y, copy=True, dtype=float)
    n = out.size

    def nearest_healthy_left(i: int):
        j = i - 1
        while j >= 0 and j in bad_set:
            j -= 1
        return j

    def nearest_healthy_right(i: int):
        j = i + 1
        while j < n and j in bad_set:
            j += 1
        return j

    bad_set = set(bad_indices)
    for i in bad_indices:
        li = nearest_healthy_left(i)
        ri = nearest_healthy_right(i)
        left = out[li] if li >= 0 and np.isfinite(out[li]) else np.nan
        right = out[ri] if ri < n and np.isfinite(out[ri]) else np.nan
        if np.isfinite(left) and np.isfinite(right):
            # linear interpolation across the bad region
            frac = (i - li) / (ri - li) if ri != li else 0.5
            out[i] = left + frac * (right - left)
        elif np.isfinite(left):
            out[i] = left
        elif np.isfinite(right):
            out[i] = right
    return out


def _repair_invalid_terminal_run(y: np.ndarray, n_bad: int) -> np.ndarray:
    if n_bad <= 0 or y.size < 3:
        return np.array(y, copy=True, dtype=float)
    out = np.array(y, copy=True, dtype=float)
    start = len(out) - n_bad
    if start <= 0:
        return out
    healthy = out[max(0, start - 5):start]
    healthy = healthy[np.isfinite(healthy)]
    if healthy.size == 0:
        return out
    if healthy.size >= 2:
        slope = float(np.median(np.diff(healthy)))
    else:
        slope = 0.0
    base = float(np.median(healthy[-min(4, healthy.size):]))
    max_step = 0.20 * max(abs(base), 1e-30)
    slope = float(np.clip(slope, -max_step, max_step))
    for k in range(n_bad):
        cand = base + slope * (k + 1)
        if base > 0:
            cand = max(cand, 0.6 * base)
        out[start + k] = cand
    return out


def _trim_invalid_terminal_run(*arrays: np.ndarray, n_bad: int):
    if n_bad <= 0:
        return [np.array(a, copy=True) if a is not None else None for a in arrays]
    trimmed=[]
    end = -n_bad
    for arr in arrays:
        if arr is None:
            trimmed.append(None)
        else:
            trimmed.append(np.array(arr[:end], copy=True))
    return trimmed


def _apply_tail_cleanup(arr: np.ndarray, settings: CorrectionSettings) -> tuple[np.ndarray, int]:
    mode = getattr(settings, "tail_cleanup_mode", "repair_tail") or "repair_tail"
    max_points = max(int(getattr(settings, "tail_cleanup_points", 4) or 4), 6)
    arr2 = np.array(arr, copy=True, dtype=float)

    # First catch classic invalid terminal run.
    n_bad = _detect_invalid_terminal_run(arr2, max_points=max_points)
    # Then catch anomalies slightly earlier in the tail window.
    bad_idx = _detect_tail_anomaly_indices(arr2, scan_points=max(10, max_points * 2))
    earliest_bad = min(bad_idx) if bad_idx else None

    if n_bad <= 0 and earliest_bad is None:
        return arr2, 0

    if mode == "trim_invalid_tail":
        if earliest_bad is not None:
            n_trim = arr2.size - earliest_bad
            return np.array(arr2[:-n_trim], copy=True, dtype=float), n_trim
        return np.array(arr2[:-n_bad], copy=True, dtype=float), n_bad

    # repair mode
    if earliest_bad is not None:
        arr2 = _repair_tail_anomalies(arr2, bad_idx)
    if n_bad > 0:
        # run a second conservative terminal repair if the last points are still sick
        n_bad2 = _detect_invalid_terminal_run(arr2, max_points=max_points)
        if n_bad2 > 0:
            arr2 = _repair_invalid_terminal_run(arr2, n_bad2)
    return arr2, 0


def choose_scale_factor(
    q: np.ndarray,
    sample_i: np.ndarray,
    empty_i: np.ndarray,
    sample_sigma: Optional[np.ndarray],
    settings: CorrectionSettings,
    sample_transmission: Optional[float] = None,
    empty_transmission: Optional[float] = None,
    physics: Optional[PhysicsBundle] = None,
) -> float:
    mode = settings.scale_mode
    if mode == "max_empty":
        mode = "auto_robust"
    if mode == "manual":
        return max(float(settings.manual_scale), 0.0)

    if mode == "transmission":
        if sample_transmission is None or empty_transmission is None or empty_transmission <= 0:
            return max(float(settings.manual_scale), 0.0)
        return max(float(sample_transmission / empty_transmission), 0.0)

    if mode == "q_point_ratio":
        q0 = settings.q_match
        target = settings.target_ratio if settings.target_ratio and settings.target_ratio > 0 else 1.0
        if q0 is None:
            return max(float(settings.manual_scale), 0.0)
        s0 = float(np.interp(q0, q, sample_i))
        e0 = float(np.interp(q0, q, empty_i))
        if not np.isfinite(s0) or not np.isfinite(e0) or e0 <= 0:
            return max(float(settings.manual_scale), 0.0)
        return max(s0 / (target * e0), 0.0)

    if mode == "physics_based":
        if physics is not None:
            te = float(physics.theoretical_empty_transmission)
            tf = float(physics.theoretical_filled_transmission)
            if te > 0 and tf > 0:
                scale = tf / te
                valid = empty_i > 0
                if np.any(valid):
                    ratio = sample_i[valid] / empty_i[valid]
                    ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
                    if ratio.size:
                        scale = min(scale, float(np.quantile(ratio, 0.99)))
                return max(scale, 0.0)
        return max(float(settings.manual_scale), 0.0)

    if empty_i.size:
        idx = int(np.nanargmax(empty_i))
        emax = float(empty_i[idx])
        if np.isfinite(emax) and emax > 0:
            sample_at_empty_max = float(sample_i[idx])
            if np.isfinite(sample_at_empty_max):
                return max(sample_at_empty_max / emax, 0.0)
    return max(float(settings.manual_scale), 0.0)



def estimate_effective_sample_thickness_mm(sample_mu_linear_mm_inv: float, sample_transmission: Optional[float], empty_transmission: Optional[float]) -> Optional[float]:
    if sample_mu_linear_mm_inv is None or sample_mu_linear_mm_inv <= 0:
        return None
    if sample_transmission is None or empty_transmission is None:
        return None
    if sample_transmission <= 0 or empty_transmission <= 0:
        return None
    ratio = sample_transmission / empty_transmission
    if ratio <= 0 or ratio > 1.0:
        return None
    return -math.log(ratio) / sample_mu_linear_mm_inv



def _reference_signal_for_norm(corrected: np.ndarray, settings: CorrectionSettings) -> np.ndarray:
    ref = np.array(corrected, copy=True)
    if settings.remove_aberrations and ref.size >= 5:
        ref = _despike_signal(ref)
    return ref


def _final_normalize(corrected: np.ndarray, sigma: Optional[np.ndarray], settings: CorrectionSettings, sample_t: Optional[float], empty_t: Optional[float]) -> tuple[np.ndarray, Optional[np.ndarray], float]:
    mode = settings.final_normalization_mode
    factor = 1.0
    if mode == "none":
        return corrected, sigma, factor
    ref = _reference_signal_for_norm(corrected, settings)
    if mode == "factor":
        factor = float(settings.final_normalization_value or 1.0)
    elif mode == "range_to_value":
        rng = float(np.nanmax(ref) - np.nanmin(ref)) if ref.size else 0.0
        target = float(settings.final_normalization_value or 1.0)
        factor = target / rng if rng > 0 else 1.0
    elif mode == "transmission":
        if sample_t is not None and empty_t is not None and sample_t > 0 and empty_t > 0:
            factor = empty_t / sample_t
    elif mode == "delta_ends_to_value":
        n = min(5, max(1, ref.size // 2))
        low = float(np.nanmean(ref[:n])) if ref.size else 0.0
        high = float(np.nanmean(ref[-n:])) if ref.size else 0.0
        delta = high - low
        target = float(settings.final_normalization_value or 1.0)
        factor = target / delta if abs(delta) > 1e-30 else 1.0
    out = corrected * factor
    sig = None if sigma is None else sigma * factor
    return out, sig, factor


def _apply_recentering(corrected: np.ndarray, settings: CorrectionSettings) -> np.ndarray:
    if not settings.recenter_enabled or corrected.size == 0:
        return corrected
    ref = _reference_signal_for_norm(corrected, settings)
    n = min(5, max(1, ref.size // 2))
    low = float(np.nanmean(ref[:n]))
    high = float(np.nanmean(ref[-n:]))
    if settings.recenter_mode == "low_q":
        anchor = low
    elif settings.recenter_mode == "mean_ends":
        anchor = 0.5 * (low + high)
    else:
        anchor = high
    shift = float(settings.recenter_value or 0.0) - anchor
    return corrected + shift



def correct_sample(sample: Curve, empty: Curve, sample_cfg: SamplePhysicsConfig, capillary_cfg: CapillaryConfig, settings: CorrectionSettings, energy_ev: float) -> CorrectionResult:
    warnings: list[str] = []
    physics = compute_physics(sample_cfg, capillary_cfg, energy_ev)
    q, sample_i, empty_i, sigma = align_to_overlap(sample, empty)

    if settings.remove_aberrations:
        sample_i = _despike_signal(sample_i)
        empty_i = _despike_signal(empty_i)
        if sigma is not None:
            sigma = _despike_signal(np.maximum(sigma, 1e-30))
        # Tail cleanup on aligned raw curves: either repair or trim invalid terminal runs.
        sample_i2, n_trim = _apply_tail_cleanup(sample_i, settings)
        if n_trim > 0:
            q, sample_i, empty_i, sigma = _trim_invalid_terminal_run(q, sample_i, empty_i, sigma, n_bad=n_trim)
            warnings.append(f"Trimmed {n_trim} invalid terminal point(s) from aligned curves.")
        else:
            sample_i = sample_i2
            empty_i2, n_trim_e = _apply_tail_cleanup(empty_i, settings)
            if n_trim_e > 0:
                q, sample_i, empty_i, sigma = _trim_invalid_terminal_run(q, sample_i, empty_i, sigma, n_bad=n_trim_e)
                warnings.append(f"Trimmed {n_trim_e} invalid terminal point(s) from empty curve.")
            else:
                empty_i = empty_i2

    sample_t = sample_cfg.manual_transmission if sample_cfg.manual_transmission is not None else sample.transmission
    empty_t = empty.transmission

    scale = choose_scale_factor(q, sample_i, empty_i, sigma, settings, sample_t, empty_t, physics)
    empty_scaled = scale * empty_i
    corrected_raw = sample_i - empty_scaled

    sigma_corr = None
    if sigma is not None:
        sigma_empty = None
        if empty.sigma is not None and len(empty.sigma) == len(empty.q):
            sigma_empty = np.interp(q, empty.q, empty.sigma)
        if sigma_empty is not None:
            sigma_corr = np.sqrt(np.maximum(sigma, 1e-30) ** 2 + (scale * np.maximum(sigma_empty, 1e-30)) ** 2)
        else:
            sigma_corr = np.array(sigma, copy=True)

    corrected, sigma_corr, final_factor = _final_normalize(corrected_raw, sigma_corr, settings, sample_t, empty_t)

    if settings.remove_aberrations:
        # Second pass on corrected arrays catches any tail glitch introduced after subtraction/normalization.
        corrected2, n_trim_corr = _apply_tail_cleanup(corrected, settings)
        if n_trim_corr > 0:
            q, sample_i, empty_i, empty_scaled, corrected_raw, corrected, sigma_corr = _trim_invalid_terminal_run(
                q, sample_i, empty_i, empty_scaled, corrected_raw, corrected, sigma_corr, n_bad=n_trim_corr
            )
            warnings.append(f"Trimmed {n_trim_corr} invalid terminal point(s) after correction.")
        else:
            corrected = corrected2
            corrected_raw = _repair_invalid_terminal_run(corrected_raw, _detect_invalid_terminal_run(corrected_raw, max_points=settings.tail_cleanup_points))
            sample_i = _repair_invalid_terminal_run(sample_i, _detect_invalid_terminal_run(sample_i, max_points=settings.tail_cleanup_points))
            empty_scaled = _repair_invalid_terminal_run(empty_scaled, _detect_invalid_terminal_run(empty_scaled, max_points=settings.tail_cleanup_points))
            if sigma_corr is not None:
                sigma_corr = _repair_invalid_terminal_run(np.maximum(sigma_corr, 1e-30), _detect_invalid_terminal_run(np.maximum(sigma_corr,1e-30), max_points=settings.tail_cleanup_points))

    if sample_cfg.manual_mu_linear_mm_inv is not None:
        sample_mu_mm_inv = float(sample_cfg.manual_mu_linear_mm_inv)
    else:
        sample_mu_mm_inv = float(physics.sample_abs["mu_linear_mm_inv"])
    capillary_mu_mm_inv = float(physics.capillary_abs["mu_linear_mm_inv"])

    effective_thickness_mm = estimate_effective_sample_thickness_mm(sample_mu_mm_inv, sample_t, empty_t)

    if np.any(np.isnan(corrected)) or np.any(np.isinf(corrected)):
        warnings.append("Corrected curve contains NaN or inf values.")
    if np.sum(corrected_raw < 0) > 0:
        warnings.append("Corrected curve contains negative values; check scaling choice.")
    if sample_t is None:
        warnings.append("Sample transmission unavailable; transmission-based scaling could not be verified.")
    if empty_t is None:
        warnings.append("Empty transmission unavailable; transmission-based scaling is limited.")

    return CorrectionResult(
        q=q,
        sample_aligned=sample_i,
        empty_aligned=empty_i,
        empty_scaled=empty_scaled,
        corrected_raw=corrected_raw,
        corrected=corrected,
        sigma_corrected=sigma_corr,
        residual=corrected_raw.copy(),
        scale_factor=scale,
        overlap_qmin=float(q.min()),
        overlap_qmax=float(q.max()),
        effective_sample_thickness_mm=effective_thickness_mm,
        sample_mu_linear_mm_inv=sample_mu_mm_inv,
        capillary_mu_linear_mm_inv=capillary_mu_mm_inv,
        sample_transmission=sample_t,
        empty_transmission=empty_t,
        warnings=warnings,
        physics=physics,
        final_normalization_mode=settings.final_normalization_mode,
        final_normalization_factor=final_factor,
    )
