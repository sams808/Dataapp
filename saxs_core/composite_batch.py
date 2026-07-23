"""
saxs_core/composite_batch.py — batch fitting over a series of curves (spec
§4.4), with optional parameter continuation between neighbors. General-
purpose: "series" and "neighbor order" are whatever the CALLER says they
are — an explicit `group` label and `order_hint` on each BatchItem — this
module never parses sample names or assumes any particular composition
scheme. The first real campaign (spec §8, the user's own PBi glass series)
is just one CALLER supplying groups/order built from that project's own
naming convention; nothing here is specific to it.

Continuation (spec §4.4): within a group, each sample after the first is
ALSO tried with the previous sample's fitted values as the seed (instead of
fit_staged's own auto-seeding), refit via the same global-multistart
machinery Stage 4 uses. If that continuation-seeded fit is genuinely
better (lower chi2red) than the auto-seeded one, it's kept; if it degrades
by more than `chi2_degrade_factor`, the auto-seeded result is kept instead
— continuation is a targeted improvement attempt, never a way to force a
worse fit through.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from saxs_core.composite_fit import PRESETS, CompositeModel, build_composite, build_preset
from saxs_core.composite_staged import (
    FitResult, _build_derived, _params_to_dict, _stage4_global,
    apply_hygiene, compute_diagnostics, fit_staged,
)
from saxs_core.curve import Curve


@dataclass
class BatchItem:
    sample_id: str
    curve: Curve
    group: str = "default"     # caller-defined grouping label for continuation
    order_hint: float = 0.0    # caller-defined ordering key within the group


@dataclass
class BatchRunResult:
    fits: Dict[str, FitResult] = field(default_factory=dict)
    order: List[str] = field(default_factory=list)
    continuation_used: Dict[str, bool] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)


def _model_from_preset_name(name: str) -> CompositeModel:
    if name in PRESETS:
        return build_preset(name)
    return build_composite(name.split("+"))


def _continuation_seeded_result(
    prev_params: Dict[str, Dict[str, Any]], model: CompositeModel,
    q: np.ndarray, I: np.ndarray, sigma: np.ndarray, sample_id: str, multistart_n: int,
) -> Optional[Any]:
    """Refit `model` (assumed to be the SAME component set as the previous
    sample's accepted model) seeded from the previous sample's fitted
    values. Returns None if the parameter names don't line up (different
    component set — continuation isn't meaningful) or the fit raises."""
    names = model.param_names()
    if not all(name in prev_params for name in names):
        return None
    seed_values = {name: prev_params[name]["value"] for name in names}
    try:
        stage = _stage4_global(q, I, sigma, model, seed_values, sample_id + ":continuation", multistart_n)
        return stage["result"]
    except Exception:
        return None


def run_batch(
    items: Sequence[BatchItem],
    *,
    windows: Optional[Dict[str, Tuple[float, float]]] = None,
    trim_n: int = 3,
    multistart_n: int = 8,
    continuation: bool = True,
    chi2_degrade_factor: float = 3.0,
    log: Callable[[str], None] = lambda *_: None,
) -> BatchRunResult:
    """Fit every item, grouped/ordered exactly as given (sorted by
    `order_hint` within each `group`). Never raises: one sample's failure
    is recorded in `errors` and the rest of the batch continues (spec
    §4.6's "every sample stands alone")."""
    groups: Dict[str, List[BatchItem]] = {}
    for it in items:
        groups.setdefault(it.group, []).append(it)
    for g in groups:
        groups[g] = sorted(groups[g], key=lambda it: it.order_hint)

    result = BatchRunResult()
    for g, group_items in groups.items():
        prev_params: Optional[Dict[str, Dict[str, Any]]] = None
        for it in group_items:
            result.order.append(it.sample_id)
            log(f"[{g}] fitting {it.sample_id}...")
            try:
                baseline = fit_staged(it.curve, sample_id=it.sample_id, windows=windows,
                                      trim_n=trim_n, multistart_n=multistart_n, log=log)
            except Exception as exc:  # pragma: no cover - fit_staged itself never raises by design
                result.errors[it.sample_id] = str(exc)
                continue

            chosen = baseline
            used_continuation = False
            if continuation and prev_params is not None:
                model = _model_from_preset_name(baseline.preset_chosen)
                if set(model.param_names()) == set(prev_params):
                    hygiene = apply_hygiene(it.curve, trim_n=trim_n)
                    q = np.asarray(hygiene.curve.q, dtype=float)
                    I = np.asarray(hygiene.curve.intensity, dtype=float)
                    # Apply the SAME sigma_scale the baseline fit calibrated
                    # (v3 §8.4 consistency rule) so cont_result.redchi and
                    # baseline.gof["chi2red"] are on the same footing before
                    # being compared just below.
                    sigma = np.asarray(hygiene.curve.sigma, dtype=float) * baseline.sigma_scale
                    cont_result = _continuation_seeded_result(
                        prev_params, model, q, I, sigma, it.sample_id, multistart_n)
                    if cont_result is not None:
                        baseline_chi2 = baseline.gof["chi2red"]
                        if cont_result.redchi <= baseline_chi2 * chi2_degrade_factor and cont_result.redchi < baseline_chi2:
                            diagnostics = compute_diagnostics(model, cont_result, q, I, baseline.windows, sigma=sigma)
                            chosen = FitResult(
                                sample_id=it.sample_id, preset_chosen=baseline.preset_chosen,
                                residual_mode=baseline.residual_mode, loss=baseline.loss,
                                windows=baseline.windows, sigma_model=baseline.sigma_model,
                                params=_params_to_dict(cont_result.params, chi2red=float(cont_result.redchi)),
                                derived=_build_derived(model, cont_result.params, pruned=baseline.pruned),
                                gof=diagnostics["gof"],
                                flags=baseline.flags + ["continuation_seeded"] + diagnostics["flags"],
                                seeds_used={name: prev_params[name]["value"] for name in model.param_names()},
                                multistart_n=multistart_n, no_peak=baseline.no_peak,
                                stages={**baseline.stages, "continuation": {"chi2red": float(cont_result.redchi),
                                                                           "baseline_chi2red": float(baseline_chi2)}},
                                pruned=baseline.pruned, sigma_scale=baseline.sigma_scale,
                            )
                            used_continuation = True

            result.fits[it.sample_id] = chosen
            result.continuation_used[it.sample_id] = used_continuation
            prev_params = chosen.params
    return result


# =============================================================================
# Legacy Gaussian cross-check (spec §4.4)
# =============================================================================

def legacy_gaussian_comparison(curve: Curve, window: Tuple[float, float]) -> Dict[str, Optional[float]]:
    """d_gauss = 2*pi/q*_gauss, xi_gauss = 2*pi/FWHM_gauss from the
    existing Gaussian-on-power-law-baseline fit (saxs_core.analysis.
    fit_pseudo_bragg_peak — already the exact same formulas). Returns
    None values (not a raised exception) when the legacy fit itself fails
    on this window, so a batch row can always be written."""
    from saxs_core.analysis import fit_pseudo_bragg_peak
    try:
        r = fit_pseudo_bragg_peak(curve.q, curve.intensity, window[0], window[1])
        return {"d_gauss": r.d_spacing, "xi_gauss": r.xi_app}
    except Exception:
        return {"d_gauss": None, "xi_gauss": None}


def _row_for(sample_id: str, fit: FitResult, curve: Optional[Curve], continuation_used: bool) -> Dict[str, Any]:
    row = fit.to_csv_row()
    row["continuation_used"] = continuation_used
    if curve is not None and "d" in fit.derived and "W_peak" in fit.windows:
        legacy = legacy_gaussian_comparison(curve, tuple(fit.windows["W_peak"]))
        row.update(legacy)
        d_ts, d_gauss = fit.derived.get("d"), legacy.get("d_gauss")
        row["d_ts_over_d_gauss"] = (d_ts / d_gauss) if (d_ts and d_gauss) else None
    return row


def batch_to_csv_rows(batch: BatchRunResult, curves: Optional[Dict[str, Curve]] = None) -> List[Dict[str, Any]]:
    """One flat row per fitted sample, in processing order, including the
    d_TS/d_gauss legacy comparison when a source Curve is provided (spec
    §4.4's 'saxs_composite_fits.csv')."""
    curves = curves or {}
    return [_row_for(sid, batch.fits[sid], curves.get(sid), batch.continuation_used.get(sid, False))
            for sid in batch.order if sid in batch.fits]


def write_batch_csv(path: str, batch: BatchRunResult, curves: Optional[Dict[str, Curve]] = None) -> None:
    rows = batch_to_csv_rows(batch, curves)
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# Figures
# =============================================================================

def plot_sample_fit(fit: FitResult, model: CompositeModel, curve: Curve, fig=None):
    """Per-sample 2-panel figure (spec §4.4): data + total + component
    curves on log-log, sigma-normalized residuals below."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    q = np.asarray(curve.q, dtype=float)
    I = np.asarray(curve.intensity, dtype=float)
    params = {name: fit.params[name]["value"] for name in fit.params}
    total = model.eval(q, params)
    parts = model.eval_components(q, params)

    fig = fig or plt.figure(figsize=(7, 6))
    ax_top = fig.add_subplot(211)
    ax_bottom = fig.add_subplot(212, sharex=ax_top)

    ax_top.loglog(q, np.clip(I, 1e-12, None), ".", ms=2, color="black", label=fit.sample_id)
    ax_top.loglog(q, np.clip(total, 1e-12, None), lw=1.6, color="crimson", label="total")
    for name, curve_part in parts.items():
        ax_top.loglog(q, np.clip(curve_part, 1e-12, None), lw=1.0, alpha=0.7, label=name)
    ax_top.legend(fontsize=7)
    ax_top.set_ylabel("I (a.u.)")
    ax_top.grid(alpha=0.25, which="both")

    sigma = np.where(np.asarray(curve.sigma, dtype=float) > 0, curve.sigma, 1.0) if curve.sigma is not None else np.ones_like(q)
    residual = (I - total) / sigma
    ax_bottom.plot(q, residual, ".", ms=2, color="steelblue")
    ax_bottom.axhline(0.0, color="0.5", lw=0.8)
    ax_bottom.set_xscale("log")
    ax_bottom.set_xlabel("q (Å⁻¹)")
    ax_bottom.set_ylabel("residual / σ")
    ax_bottom.grid(alpha=0.25, which="both")
    fig.tight_layout()
    return fig


def plot_series_overview(batch: BatchRunResult, group: Sequence[str], x_values: Dict[str, float],
                         x_label: str = "x", fig=None):
    """d_TS / xi_TS / fa vs a caller-supplied x-value per sample (spec
    §4.4: 'd_TS, xi_TS, fa vs Bi content per P-series' — the x-axis
    quantity and which samples belong together are entirely the caller's
    choice; this function has no notion of composition itself)."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    xs, ds, xis, fas = [], [], [], []
    for sid in group:
        fit = batch.fits.get(sid)
        if fit is None or "d" not in fit.derived or sid not in x_values:
            continue
        xs.append(x_values[sid])
        ds.append(fit.derived["d"])
        xis.append(fit.derived["xi"])
        fas.append(fit.derived["fa"])

    fig = fig or plt.figure(figsize=(6, 8))
    ax_d = fig.add_subplot(311)
    ax_xi = fig.add_subplot(312, sharex=ax_d)
    ax_fa = fig.add_subplot(313, sharex=ax_d)
    for ax, ys, ylabel in ((ax_d, ds, "d (Å)"), (ax_xi, xis, "ξ (Å)"), (ax_fa, fas, "fa")):
        ax.plot(xs, ys, "o-", ms=4)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    ax_fa.set_xlabel(x_label)
    fig.tight_layout()
    return fig
