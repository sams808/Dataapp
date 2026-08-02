"""
saxs/core/_staged_result.py — internal implementation detail of
composite_staged.py: the FitResult dataclass and its supporting
serialization/derived-quantity helpers.

Not meant to be imported directly by anything outside this package —
import from saxs.core.composite_staged instead, which re-exports
everything here.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from saxs.core.composite_fit import CompositeModel
from saxs.core.composite_models import regime_label
from ._staged_hygiene import Windows

CODE_VERSION = "composite_staged-v1"


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


