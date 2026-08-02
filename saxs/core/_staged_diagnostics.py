"""
saxs/core/_staged_diagnostics.py — internal implementation detail of
composite_staged.py: Stage 5 diagnostics (chi2red/AIC/BIC/Durbin-Watson/
CorMap/correlation/at-bound flags, per-window visual-equivalence check).

Not meant to be imported directly by anything outside this package —
import from saxs.core.composite_staged instead, which re-exports
everything here.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from saxs.core.composite_fit import CompositeModel
from ._staged_hygiene import Windows, _mask_for
from ._staged_result import _gof


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


