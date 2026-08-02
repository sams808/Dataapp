"""
saxs/core/_staged_profile_likelihood.py — internal implementation detail
of composite_staged.py: the pl2-sensitivity check, Stage 2b peak
crosscheck, and profile-likelihood confidence intervals for ts_d/ts_xi.

Not meant to be imported directly by anything outside this package —
import from saxs.core.composite_staged instead, which re-exports
everything here.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from saxs.core.composite_fit import CompositeModel
from ._staged_hygiene import Windows, _mask_for


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


