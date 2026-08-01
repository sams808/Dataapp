"""Bridge to sasmodels (form/structure factor library) + bumps (optimizers).

sasmodels gives ~140 small-angle scattering models with polydispersity and
P(Q)*S(Q) combinations; bumps provides Levenberg-Marquardt, amoeba and the
DREAM MCMC sampler for real uncertainty estimates. Both are optional at
import time so the rest of the suite works without them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

try:
    from sasmodels.core import load_model, list_models
    from sasmodels.bumps_model import Model as BumpsSasModel, Experiment
    from sasmodels.data import Data1D, empty_data1D
    from bumps.names import FitProblem
    from bumps.fitters import fit as bumps_fit

    HAVE_SASMODELS = True
    _IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # pragma: no cover - depends on environment
    HAVE_SASMODELS = False
    _IMPORT_ERROR = str(exc)


def sasmodels_available() -> bool:
    return HAVE_SASMODELS


def sasmodels_import_error() -> Optional[str]:
    return _IMPORT_ERROR


STRUCTURE_FACTORS = ["hardsphere", "stickyhardsphere", "squarewell", "hayter_msa"]


def available_models() -> List[str]:
    if not HAVE_SASMODELS:
        return []
    return sorted(list_models("all"))


def available_form_factors() -> List[str]:
    return [m for m in available_models() if m not in STRUCTURE_FACTORS]


@dataclass
class ParamState:
    name: str
    value: float
    vary: bool = False
    lower: float = -np.inf
    upper: float = np.inf
    stderr: Optional[float] = None
    units: str = ""
    description: str = ""


@dataclass
class ModelFitResult:
    model_name: str
    chisq: float
    params: Dict[str, ParamState]
    q: np.ndarray
    theory: np.ndarray
    residuals: Optional[np.ndarray]
    message: str = ""
    dream_uncertainties: Dict[str, float] = field(default_factory=dict)


class SasModelFit:
    """One fit session: a sasmodels kernel bound to one dataset via bumps."""

    def __init__(self, model_name: str, q: np.ndarray, intensity: np.ndarray,
                 dI: Optional[np.ndarray] = None, dq: Optional[np.ndarray] = None):
        if not HAVE_SASMODELS:
            raise RuntimeError(f"sasmodels/bumps not available: {_IMPORT_ERROR}")
        self.model_name = model_name
        q = np.asarray(q, dtype=float)
        intensity = np.asarray(intensity, dtype=float)
        mask = np.isfinite(q) & np.isfinite(intensity) & (q > 0)
        if dI is not None:
            dI = np.asarray(dI, dtype=float)
            mask &= np.isfinite(dI) & (dI > 0)
        self.q = q[mask]
        self.intensity = intensity[mask]
        self.dI = dI[mask] if dI is not None else None
        self.dq = np.asarray(dq, dtype=float)[mask] if dq is not None else None

        dy = self.dI if self.dI is not None else np.maximum(np.sqrt(np.abs(self.intensity)), 1e-12 * np.max(np.abs(self.intensity)))
        self.data = Data1D(x=self.q, y=self.intensity, dy=dy)
        if self.dq is not None:
            self.data.dx = self.dq

        self.kernel = load_model(model_name)
        self.model = BumpsSasModel(self.kernel)
        self.params: Dict[str, ParamState] = {}
        self._init_params()

    def _init_params(self) -> None:
        info = self.kernel.info
        defaults = {p.name: p.default for p in info.parameters.common_parameters + info.parameters.kernel_parameters}
        limits = {p.name: p.limits for p in info.parameters.common_parameters + info.parameters.kernel_parameters}
        units = {p.name: p.units for p in info.parameters.common_parameters + info.parameters.kernel_parameters}
        descr = {p.name: p.description for p in info.parameters.common_parameters + info.parameters.kernel_parameters}
        for name, value in sorted(defaults.items()):
            lo, hi = limits.get(name, (-np.inf, np.inf))
            self.params[name] = ParamState(
                name=name, value=float(value), vary=False,
                lower=float(lo) if np.isfinite(lo) else -np.inf,
                upper=float(hi) if np.isfinite(hi) else np.inf,
                units=str(units.get(name, "")),
                description=str(descr.get(name, "")),
            )

    def param_names(self) -> List[str]:
        return list(self.params.keys())

    def set_param(self, name: str, value: Optional[float] = None, vary: Optional[bool] = None,
                  lower: Optional[float] = None, upper: Optional[float] = None) -> None:
        state = self.params[name]
        if value is not None:
            state.value = float(value)
        if vary is not None:
            state.vary = bool(vary)
        if lower is not None:
            state.lower = float(lower)
        if upper is not None:
            state.upper = float(upper)

    def set_polydispersity(self, param: str, width: float, npts: int = 35,
                           nsigmas: float = 3.0, distribution: str = "gaussian") -> None:
        """Enable size dispersity on a polydisperse-capable parameter."""
        for suffix, value in (("_pd", width), ("_pd_n", npts), ("_pd_nsigma", nsigmas)):
            name = param + suffix
            bumps_par = getattr(self.model, name, None)
            if bumps_par is None:
                raise KeyError(f"{self.model_name} has no dispersity control {name}")
            bumps_par.value = value
        setattr(self.model, param + "_pd_type", distribution)

    def _sync_model(self) -> None:
        for name, state in self.params.items():
            par = getattr(self.model, name, None)
            if par is None:
                continue
            par.value = state.value
            if state.vary:
                lo = state.lower if np.isfinite(state.lower) else None
                hi = state.upper if np.isfinite(state.upper) else None
                par.range(lo, hi)
            else:
                par.fixed = True

    def theory(self) -> np.ndarray:
        self._sync_model()
        experiment = Experiment(data=self.data, model=self.model)
        return np.asarray(experiment.theory(), dtype=float)

    def fit(self, method: str = "lm", steps: int = 1000, dream_samples: int = 10000) -> ModelFitResult:
        self._sync_model()
        experiment = Experiment(data=self.data, model=self.model)
        problem = FitProblem(experiment)
        if not any(state.vary for state in self.params.values()):
            raise ValueError("No parameters marked as fitted (vary=True).")

        if method == "dream":
            result = bumps_fit(problem, method="dream", samples=dream_samples, verbose=False)
        else:
            result = bumps_fit(problem, method=method, steps=steps, verbose=False)

        labels = problem.labels()
        values = np.atleast_1d(result.x)
        errors = np.atleast_1d(result.dx) if result.dx is not None else np.full_like(values, np.nan)

        # Push fitted values back into problem/model, then read back all params.
        problem.setp(values)
        dream_unc: Dict[str, float] = {}
        for label, val, err in zip(labels, values, errors):
            pname = _strip_label(label)
            if pname in self.params:
                self.params[pname].value = float(val)
                self.params[pname].stderr = float(err) if np.isfinite(err) else None
                if method == "dream":
                    dream_unc[pname] = float(err)

        theory = np.asarray(experiment.theory(), dtype=float)
        dy = np.asarray(self.data.dy, dtype=float)
        residuals = (self.intensity - theory) / np.where(dy > 0, dy, 1.0)
        return ModelFitResult(
            model_name=self.model_name,
            chisq=float(problem.chisq()),
            params={k: v for k, v in self.params.items()},
            q=self.q,
            theory=theory,
            residuals=residuals,
            message=f"method={method}",
            dream_uncertainties=dream_unc,
        )


def _strip_label(label: str) -> str:
    """bumps labels parameters e.g. 'M1 radius' or 'radius'; keep the tail."""
    return label.split()[-1]


def compute_model_curve(model_name: str, q: np.ndarray, params: Optional[Dict[str, float]] = None) -> np.ndarray:
    """Evaluate a sasmodels model on a q grid without any data (previews)."""
    if not HAVE_SASMODELS:
        raise RuntimeError(f"sasmodels/bumps not available: {_IMPORT_ERROR}")
    q = np.asarray(q, dtype=float)
    data = empty_data1D(q)
    kernel = load_model(model_name)
    model = BumpsSasModel(kernel, **(params or {}))
    experiment = Experiment(data=data, model=model)
    return np.asarray(experiment.theory(), dtype=float)
