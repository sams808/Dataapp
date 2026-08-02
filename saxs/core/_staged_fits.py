"""
saxs/core/_staged_fits.py — internal implementation detail of
composite_staged.py: the Stage 1-4 staged-fitting functions (BG, +TS,
+GP/Beaucage/PL2, global multistart).

Not meant to be imported directly by anything outside this package —
import from saxs.core.composite_staged instead, which re-exports
everything here.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from saxs.core.composite_fit import CompositeModel, build_composite
from ._staged_hygiene import Windows, _mask_for, _seed_from_sample_id


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


