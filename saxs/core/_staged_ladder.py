"""
saxs/core/_staged_ladder.py — internal implementation detail of
composite_staged.py: Stage 6 model-selection ladder
(BG -> BG_DAB -> BG_TS -> BG_TS_OZ -> BG_TS_PL2 -> BG_TS_GP, with the
class-a guardrail and per-window visual-equivalence gate).

Not meant to be imported directly by anything outside this package —
import from saxs.core.composite_staged instead, which re-exports
everything here.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from saxs.core.composite_fit import CompositeModel, build_composite
from ._staged_hygiene import Windows
from ._staged_fits import _stage4_global, ts_guardrail_ok
from ._staged_diagnostics import visual_equivalence_ok
from ._staged_result import _rms_log10


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


