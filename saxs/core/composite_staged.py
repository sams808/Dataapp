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

Implementation split across saxs/core/_staged_*.py (each internal, not
meant to be imported directly — import from here instead, which re-exports
everything): _staged_hygiene.py (stage 0 + morphology classifier),
_staged_result.py (FitResult), _staged_fits.py (stages 1-4),
_staged_diagnostics.py (stage 5), _staged_ladder.py (stage 6),
_staged_profile_likelihood.py (pl2-sensitivity, stage 2b crosscheck,
ts_d/ts_xi profile-likelihood CIs). This file itself keeps only the
fit_staged orchestrator that ties every stage together.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from saxs.core.composite_fit import PRESETS, CompositeModel
from saxs.core.curve import Curve

from ._staged_hygiene import (
    Windows,
    estimate_sigma_model,
    detect_data_type,
    estimate_sigma_model_detrended,
    _log_rebin,
    HygieneResult,
    apply_hygiene,
    guess_class,
    _locate_peak,
    MorphologyResult,
    detect_knee_q,
    detect_peak_q,
    detect_midq_hump,
    classify_morphology,
    propose_windows_from_classifier,
    propose_windows,
    detect_high_q_cut,
    detect_beamstop_edge_trim,
    _apply_mask_regions,
    _mask_for,
    _seed_from_sample_id,
)
from ._staged_result import (
    CODE_VERSION,
    FitResult,
    _params_to_dict,
    _build_derived,
    _rms_log10,
    _gof,
)
from ._staged_fits import (
    _STAGE1_PL_P_BOUNDS,
    _bg_c_plateau_bounds,
    _stage1_bg,
    _stage2_add_ts,
    ts_guardrail_ok,
    fit_systematic_floor,
    ts_window_local_delta_bic,
    _stage3_add_gp,
    _stage3_add_beaucage,
    detect_guinier_knee,
    _stage3_add_pl2,
    _SCALE_PARAM_SUFFIXES,
    _LENGTH_PARAM_SUFFIXES,
    _widen_bounds_for_global,
    _stage4_global,
)
from ._staged_diagnostics import (
    _durbin_watson,
    cormap_longest_run,
    _correlation_flags,
    _at_bound_flags,
    _window_chi2red,
    compute_diagnostics,
    median_log_residual_by_window,
    visual_equivalence_ok,
)
from ._staged_ladder import (
    _fit_full_range,
    _walk_ladder,
    select_best_preset,
    _SIMPLER_PRESET,
)
from ._staged_profile_likelihood import (
    _pl2_sensitive,
    _TS_PARAM_NAMES,
    stage2b_peak_crosscheck,
    _find_crossing,
    _profile_delta_chi2,
    _ci_from_profile,
    _CHI2_95_ONE_PARAM,
    _log_spaced_grid_around,
    compute_ts_profile_likelihood_cis,
)

__all__ = [
    # _staged_hygiene
    "Windows", "estimate_sigma_model", "detect_data_type",
    "estimate_sigma_model_detrended", "_log_rebin", "HygieneResult",
    "apply_hygiene", "guess_class", "_locate_peak", "MorphologyResult",
    "detect_knee_q", "detect_peak_q", "detect_midq_hump",
    "classify_morphology", "propose_windows_from_classifier",
    "propose_windows", "detect_high_q_cut", "detect_beamstop_edge_trim",
    "_apply_mask_regions", "_mask_for", "_seed_from_sample_id",
    # _staged_result
    "CODE_VERSION", "FitResult", "_params_to_dict", "_build_derived",
    "_rms_log10", "_gof",
    # _staged_fits
    "_STAGE1_PL_P_BOUNDS", "_bg_c_plateau_bounds", "_stage1_bg",
    "_stage2_add_ts", "ts_guardrail_ok", "fit_systematic_floor",
    "ts_window_local_delta_bic", "_stage3_add_gp", "_stage3_add_beaucage",
    "detect_guinier_knee", "_stage3_add_pl2", "_SCALE_PARAM_SUFFIXES",
    "_LENGTH_PARAM_SUFFIXES", "_widen_bounds_for_global", "_stage4_global",
    # _staged_diagnostics
    "_durbin_watson", "cormap_longest_run", "_correlation_flags",
    "_at_bound_flags", "_window_chi2red", "compute_diagnostics",
    "median_log_residual_by_window", "visual_equivalence_ok",
    # _staged_ladder
    "_fit_full_range", "_walk_ladder", "select_best_preset",
    "_SIMPLER_PRESET",
    # _staged_profile_likelihood
    "_pl2_sensitive", "_TS_PARAM_NAMES", "stage2b_peak_crosscheck",
    "_find_crossing", "_profile_delta_chi2", "_ci_from_profile",
    "_CHI2_95_ONE_PARAM", "_log_spaced_grid_around",
    "compute_ts_profile_likelihood_cis",
    # this file
    "fit_staged",
]


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
