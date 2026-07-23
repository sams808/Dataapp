"""
saxs_core/composite_fit.py — the composition engine: sums composite_models.py
components into fittable composite models (spec §2). General-purpose: any
ordered list of component TYPE names builds a valid composite, not just the
five named presets below.

Backend: lmfit (`lmfit.Model(comp.eval, prefix=...)` composed with `+`) —
gives sum-composition, prefixed parameter names, bounds, vary flags,
expressions/constraints, and fit covariance for free, per the spec's own
recommendation ("use lmfit as the backend rather than reimplementing").
Downstream callers (the staged pipeline, UI) are NOT required to touch
lmfit types themselves — `eval`/`eval_components`/`derived` all accept a
plain `{name: float}` dict just as readily as an `lmfit.Parameters`/
`ModelResult.params` object.

Composition is addition only (incoherent sum of scattering contributions).
No slit smearing; the optional Gaussian point-spread smearing hook
(`gaussian_smear`) is a standalone utility, OFF by default everywhere in
this module — call it explicitly if the instrument ever needs it.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from saxs_core.composite_models import COMPONENTS, Component

if TYPE_CHECKING:
    import lmfit

# One default prefix per component TYPE; repeated types in the same
# composite get numbered (ts_ / ts2_ / ts3_ ...) by build_composite().
DEFAULT_PREFIXES: Dict[str, str] = {
    "flat_background": "bg_",
    "power_law": "pl_",
    "power_law2": "pl2_",
    "guinier": "gu_",
    "guinier_porod": "gp_",
    "beaucage_unified": "bu_",
    "dab": "dab_",
    "lorentz_oz": "oz_",
    "teubner_strey": "ts_",
    "broad_peak": "bp_",
}

# spec §2.2 registered composite presets (BG_TS_PL2 added in v2:
# PRISM_fit_pipeline_upgrade_prompt.md §3 — the low-q-upturn-without-a-
# genuine-Guinier-knee case, preferred over BG_TS_GP by the model-
# selection ladder unless a knee is actually detected; BG_TS_OZ added in
# v3 ADDENDUM — an Ornstein-Zernike low-q tail, ladder-only between
# BG_TS and BG_TS_PL2)
PRESETS: Dict[str, List[str]] = {
    "BG": ["flat_background", "power_law"],
    "BG_DAB": ["flat_background", "power_law", "dab"],
    "BG_GP": ["flat_background", "power_law", "guinier_porod"],
    "BG_BC": ["flat_background", "power_law", "beaucage_unified"],
    "BG_PL2": ["flat_background", "power_law", "power_law2"],
    "BG_TS": ["flat_background", "power_law", "teubner_strey"],
    "BG_TS_OZ": ["flat_background", "power_law", "teubner_strey", "lorentz_oz"],
    "BG_TS_PL2": ["flat_background", "power_law", "teubner_strey", "power_law2"],
    "BG_TS_GP": ["flat_background", "power_law", "teubner_strey", "guinier_porod"],
    "BG_TS_BC": ["flat_background", "power_law", "teubner_strey", "beaucage_unified"],
    "BG_BP": ["flat_background", "power_law", "broad_peak"],
}


def _value_of(params: Any, key: str) -> float:
    """Read one parameter's value whether `params` is a plain
    {name: float} dict or an lmfit.Parameters/ModelResult.params object
    (whose entries are lmfit.Parameter, exposing `.value`)."""
    v = params[key]
    return float(getattr(v, "value", v))


class CompositeModel:
    """An ordered sum of (prefix, Component) pairs. NOTE: `to_lmfit_model()`
    composes via lmfit's own `+` operator, which returns an
    `lmfit.model.CompositeModel` internally — a same-named but unrelated
    lmfit implementation detail, never part of this class's public API."""

    def __init__(self, components: Sequence[Tuple[str, Component]]):
        self.components: List[Tuple[str, Component]] = list(components)
        prefixes = [p for p, _ in self.components]
        if len(set(prefixes)) != len(prefixes):
            raise ValueError(f"Duplicate component prefixes: {prefixes}")
        if not self.components:
            raise ValueError("CompositeModel needs at least one component.")

    # ------------------------------------------------------------------
    def _local_kwargs(self, prefix: str, comp: Component, params: Any) -> Dict[str, float]:
        return {p.name: _value_of(params, prefix + p.name) for p in comp.params()}

    def param_names(self) -> List[str]:
        names: List[str] = []
        for prefix, comp in self.components:
            names.extend(prefix + p.name for p in comp.params())
        return names

    def eval(self, q: np.ndarray, params: Any) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        total = np.zeros_like(q)
        for prefix, comp in self.components:
            total = total + comp.eval(q, **self._local_kwargs(prefix, comp, params))
        return total

    def eval_components(self, q: np.ndarray, params: Any) -> Dict[str, np.ndarray]:
        """One curve per component, keyed by its prefix (underscore
        stripped) — for overlay plotting."""
        q = np.asarray(q, dtype=float)
        out: Dict[str, np.ndarray] = {}
        for prefix, comp in self.components:
            key = prefix.rstrip("_") or comp.name
            out[key] = comp.eval(q, **self._local_kwargs(prefix, comp, params))
        return out

    def derived(self, params: Any) -> Dict[str, Dict[str, float]]:
        """Per-component derived() dicts (e.g. teubner_strey's fa/q_max),
        keyed the same way as eval_components()."""
        out: Dict[str, Dict[str, float]] = {}
        for prefix, comp in self.components:
            key = prefix.rstrip("_") or comp.name
            out[key] = comp.derived(**self._local_kwargs(prefix, comp, params))
        return out

    def seed(self, q: np.ndarray, I: np.ndarray, windows: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, float]:
        """Prefixed heuristic initial values from every component's own
        seed() (a simple, generic fallback — composite_staged.py's staged
        pipeline does more elaborate, stage-aware seeding on top)."""
        out: Dict[str, float] = {}
        for prefix, comp in self.components:
            for k, v in comp.seed(q, I, windows).items():
                out[prefix + k] = v
        return out

    # ------------------------------------------------------------------
    # lmfit backend
    # ------------------------------------------------------------------
    def to_lmfit_parameters(
        self,
        seed_values: Optional[Dict[str, float]] = None,
        bound_overrides: Optional[Dict[str, Tuple[float, float]]] = None,
        vary_overrides: Optional[Dict[str, bool]] = None,
    ) -> "lmfit.Parameters":
        import lmfit
        params = lmfit.Parameters()
        for prefix, comp in self.components:
            for p in comp.params():
                full = prefix + p.name
                value = (seed_values or {}).get(full, p.value)
                lo, hi = (bound_overrides or {}).get(full, (p.min, p.max))
                vary = (vary_overrides or {}).get(full, p.vary)
                params.add(full, value=value, min=lo, max=hi, vary=vary)
        return params

    def to_lmfit_model(self) -> "lmfit.Model":
        import lmfit
        model = None
        for prefix, comp in self.components:
            m = lmfit.Model(comp.eval, prefix=prefix, independent_vars=["q"])
            model = m if model is None else model + m
        return model

    def fit(
        self, q: np.ndarray, I: np.ndarray, sigma: Optional[np.ndarray] = None,
        params: Optional["lmfit.Parameters"] = None, method: str = "least_squares",
        residual_mode: str = "weighted_linear",
        **kwargs: Any,
    ) -> "lmfit.model.ModelResult":
        """Sigma-weighted linear least squares is the PRIMARY objective
        (v3 §8: weights = 1/sigma when sigma is given) — chi2 = sum(((I -
        model)/sigma)^2), the same statistic used for chi2red/AIC/BIC,
        model-selection, and profile-likelihood confidence intervals
        throughout saxs_core.composite_staged (the v3 §8.4 consistency
        rule: one weighted statistic, one space, everywhere).

        `residual_mode="log10"` fits log10(I_model) against log10(I_data)
        instead, UNWEIGHTED. This is no longer the auto-selected default
        for any real curve (v3 §8 removed the "arbitrary units" auto-
        switch: every curve either carries a genuine propagated sigma or
        gets one from composite_staged's detrended/Poisson-like estimator,
        so a sigma-weighted linear fit is always available) — kept only
        as an explicit, manually-invoked fallback/cross-check. Implemented
        via a raw lmfit.Minimizer (not Model.fit()) since fitting in
        log-space isn't just a weights change — the RESIDUAL definition
        itself changes; the returned MinimizerResult exposes the same
        .redchi/.aic/.bic/.params/.residual/.ndata attributes as a
        Model.fit() ModelResult, so every downstream caller works
        unmodified regardless of which mode ran.

        method="least_squares" (scipy's trust-region-reflective algorithm,
        per spec §4.5's own "lmfit/least_squares" recommendation) rather
        than classic MINPACK "leastsq": TS-peak parameters like `d`/`xi`
        sit at O(1e3) with a numerically narrow peak in q-space, and
        MINPACK's default finite-difference step (scaled to machine
        epsilon) is too small there to see real curvature, silently
        stalling the fit at its seed value while still reporting
        success=True. least_squares' native bounds handling (no internal
        parameter-transform hack) converges reliably on exactly this
        shape of problem."""
        q = np.asarray(q, dtype=float)
        I = np.asarray(I, dtype=float)
        params = params if params is not None else self.to_lmfit_parameters()

        if residual_mode == "log10":
            import lmfit
            log_I = np.log10(np.clip(I, 1e-300, None))

            def _log10_residual(p: "lmfit.Parameters") -> np.ndarray:
                total = self.eval(q, p)
                return np.log10(np.clip(total, 1e-300, None)) - log_I

            minimizer = lmfit.Minimizer(_log10_residual, params)
            return minimizer.minimize(method=method, **kwargs)

        model = self.to_lmfit_model()
        weights = None
        if sigma is not None:
            sigma = np.asarray(sigma, dtype=float)
            weights = np.where(sigma > 0, 1.0 / sigma, 0.0)
        return model.fit(I, params, q=q, weights=weights, method=method, **kwargs)

    @staticmethod
    def set_expr(params: "lmfit.Parameters", name: str, expr: str) -> None:
        """Tie one parameter to an expression of others, e.g. tying two
        components' xi together: `set_expr(params, 'ts2_xi', 'ts_xi')`."""
        params[name].set(expr=expr)

    @staticmethod
    def fix(params: "lmfit.Parameters", name: str, value: Optional[float] = None) -> None:
        """Freeze a parameter (e.g. `pl_p`/`pl_B` after Stage 1)."""
        if value is not None:
            params[name].set(value=value)
        params[name].set(vary=False)


# =============================================================================
# Building composites: arbitrary user-picked component lists, or presets
# =============================================================================

def build_composite(component_names: Sequence[str]) -> CompositeModel:
    """Build a CompositeModel from an ordered list of component TYPE names
    (e.g. ["flat_background", "power_law", "teubner_strey"]) — the general
    entry point the UI's "pick components from library" flow uses. Repeated
    component types get numbered prefixes (ts_, ts2_, ts3_, ...)."""
    counts: Dict[str, int] = {}
    for name in component_names:
        if name not in COMPONENTS:
            raise KeyError(f"Unknown component {name!r}. Known: {sorted(COMPONENTS)}")
        counts[name] = counts.get(name, 0) + 1
    seen: Dict[str, int] = {}
    comps: List[Tuple[str, Component]] = []
    for name in component_names:
        seen[name] = seen.get(name, 0) + 1
        base = DEFAULT_PREFIXES.get(name, name[:3] + "_")
        # first occurrence of ANY type (even a repeated one) keeps the bare
        # prefix; only the 2nd, 3rd, ... occurrences get numbered
        prefix = base if seen[name] == 1 else f"{base.rstrip('_')}{seen[name]}_"
        comps.append((prefix, COMPONENTS[name]()))
    return CompositeModel(comps)


def build_preset(name: str) -> CompositeModel:
    if name not in PRESETS:
        raise KeyError(f"Unknown preset {name!r}. Known: {sorted(PRESETS)}")
    return build_composite(PRESETS[name])


# =============================================================================
# Optional instrument smearing (default OFF everywhere; spec §2.1)
# =============================================================================

def gaussian_smear(q: np.ndarray, I: np.ndarray, sigma_q: float) -> np.ndarray:
    """I_smeared(q) = I(q) convolved with a Gaussian of width sigma_q
    (constant across q). Pinhole-collimation instruments (like this lab's)
    have negligible smearing at this resolution — this is provided only
    for the rare case it's ever needed; nothing in this module calls it
    automatically. Not slit smearing (explicitly out of scope, spec §2.1)."""
    if sigma_q <= 0:
        return np.asarray(I, dtype=float)
    q = np.asarray(q, dtype=float)
    I = np.asarray(I, dtype=float)
    dq = float(np.median(np.diff(q))) if q.size > 1 else 1.0
    sigma_samples = max(sigma_q / max(dq, 1e-300), 1e-6)
    half = max(int(math.ceil(4 * sigma_samples)), 1)
    x = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (x / sigma_samples) ** 2)
    kernel = kernel / kernel.sum()
    return np.convolve(I, kernel, mode="same")
