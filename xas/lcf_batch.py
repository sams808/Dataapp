"""
xas/lcf_batch.py — batch/combinatorial linear combination fitting
(Athena-inspired "Combinatorial LCF"): for many target (sample) spectra
and a pool of reference (standard) spectra, fits each target against
EVERY combination of references within a configurable component-count
range, scores each combination, and ranks them -- rather than requiring
the user to pre-decide which references belong in the fit, this
surfaces which combination(s) actually explain the data best, and how
close the runner-up combinations are (a real degeneracy/ambiguity signal
LCF users need, not just a single best-guess answer).

Deliberately independent of Qt and of linear_combination_fit's own
simpler (2-arg, NNLS-only, single-combination) API in xas_science.py --
that one stays as the interactive single-fit tool in the Analysis tab;
this is the batch/combinatorial companion, and of xas_lcf_report.py
(PDF/MD rendering), so the fitting math here is independently testable
and reusable from a plain script.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .xas_science import _interp_to_grid

# Metrics where a LOWER value is better (everything else -- currently
# just r2 -- is higher-is-better). Used to pick sort direction.
_LOWER_IS_BETTER = {"rms", "reduced_chi_square", "ss_res"}


def fit_weighted_sum(target_y: np.ndarray, ref_matrix: np.ndarray, *,
                     weight_bounds: Tuple[float, float] = (0.0, 1.0),
                     sum_to_one: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Fits target_y ~= ref_matrix @ weights, subject to per-weight bounds
    and an optional sum(weights) == 1 constraint (the "fraction of each
    phase" reading LCF weights are usually given; NOT enforced by default
    since it's a real modeling choice, not always appropriate -- e.g. when
    self-absorption or a normalization mismatch means weights genuinely
    shouldn't sum to exactly 1).

    No constraint beyond bounds -> scipy.optimize.lsq_linear (fast, exact
    bounded least squares). sum_to_one -> scipy.optimize.minimize (SLSQP;
    lsq_linear has no equality-constraint support)."""
    target_y = np.asarray(target_y, float)
    n_refs = ref_matrix.shape[1]
    lb, ub = weight_bounds

    if not sum_to_one:
        from scipy.optimize import lsq_linear
        res = lsq_linear(ref_matrix, target_y, bounds=(lb, ub))
        weights = res.x
    else:
        from scipy.optimize import minimize
        w0 = np.full(n_refs, 1.0 / n_refs)

        def objective(w):
            resid = ref_matrix @ w - target_y
            return float(np.sum(resid ** 2))

        cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
        bounds = [(lb, ub)] * n_refs
        res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=cons)
        weights = np.asarray(res.x, float)

    fit_y = ref_matrix @ weights
    return weights, fit_y


def score_fit(target_y: np.ndarray, fit_y: np.ndarray, n_params: int) -> Dict[str, float]:
    """R², RMS residual, and reduced chi-square (sum of squared residuals
    / degrees of freedom) -- the three ranking metrics Athena-style LCF
    conventionally reports; `sort_by` in combinatorial_lcf picks among
    them."""
    target_y = np.asarray(target_y, float)
    resid = target_y - fit_y
    n = int(target_y.size)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((target_y - np.mean(target_y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else float("nan")
    rms = float(np.sqrt(ss_res / n)) if n else float("nan")
    dof = max(n - n_params, 1)
    reduced_chi_square = ss_res / dof
    return {"r2": r2, "rms": rms, "reduced_chi_square": reduced_chi_square, "ss_res": ss_res, "n_points": n}


@dataclass
class LCFCombinationResult:
    target: str
    ref_names: Tuple[str, ...]
    weights: np.ndarray
    fit_y: np.ndarray
    r2: float
    rms: float
    reduced_chi_square: float
    n_points: int

    def weight_map(self) -> Dict[str, float]:
        return dict(zip(self.ref_names, (float(w) for w in self.weights)))


def combinatorial_lcf(target_name: str, target_energy: np.ndarray, target_y: np.ndarray,
                      references: Sequence[Tuple[str, np.ndarray, np.ndarray]], *,
                      min_components: int = 2, max_components: int = 3,
                      weight_bounds: Tuple[float, float] = (0.0, 1.0),
                      sum_to_one: bool = False,
                      required_refs: Optional[Sequence[str]] = None,
                      sort_by: str = "r2") -> List[LCFCombinationResult]:
    """Fits `target` against every combination of `references` (each a
    (name, energy, y) triple) sized min_components..max_components
    (inclusive, both clamped to len(references)), optionally required to
    include every name in `required_refs` (e.g. "always include the
    matrix/host phase, vary the rest"). Returns results sorted best-first
    by `sort_by` ("r2", "rms", or "reduced_chi_square")."""
    if sort_by not in ("r2", "rms", "reduced_chi_square"):
        raise ValueError(f"sort_by must be 'r2', 'rms', or 'reduced_chi_square', got {sort_by!r}")
    if min_components < 1:
        raise ValueError("min_components must be >= 1")
    if len(references) == 0:
        return []

    target_y = np.asarray(target_y, float)
    ref_names_all = [r[0] for r in references]
    if len(set(ref_names_all)) != len(ref_names_all):
        raise ValueError("Reference names must be unique.")
    ref_grids = {name: _interp_to_grid(e, y, target_energy) for name, e, y in references}
    required = set(required_refs or [])
    if not required.issubset(ref_names_all):
        raise ValueError(f"required_refs contains names not in references: {required - set(ref_names_all)}")

    max_components = min(max_components, len(references))
    min_components = min(min_components, max_components)

    results: List[LCFCombinationResult] = []
    for k in range(min_components, max_components + 1):
        for combo in itertools.combinations(ref_names_all, k):
            if required and not required.issubset(combo):
                continue
            A = np.column_stack([ref_grids[name] for name in combo])
            weights, fit_y = fit_weighted_sum(target_y, A, weight_bounds=weight_bounds, sum_to_one=sum_to_one)
            scores = score_fit(target_y, fit_y, n_params=k)
            results.append(LCFCombinationResult(
                target=target_name, ref_names=combo, weights=weights, fit_y=fit_y,
                r2=scores["r2"], rms=scores["rms"], reduced_chi_square=scores["reduced_chi_square"],
                n_points=scores["n_points"],
            ))

    reverse = sort_by not in _LOWER_IS_BETTER
    results.sort(key=lambda r: getattr(r, sort_by), reverse=reverse)
    return results


@dataclass
class BatchLCFParams:
    min_components: int = 2
    max_components: int = 3
    weight_bounds: Tuple[float, float] = (0.0, 1.0)
    sum_to_one: bool = False
    required_refs: Tuple[str, ...] = field(default_factory=tuple)
    sort_by: str = "r2"
    top_n_report: int = 10


def run_batch_lcf(targets: Sequence[Tuple[str, np.ndarray, np.ndarray]],
                  references: Sequence[Tuple[str, np.ndarray, np.ndarray]],
                  params: BatchLCFParams) -> Dict[str, List[LCFCombinationResult]]:
    """One combinatorial_lcf() run per target. Returns {target_name:
    ranked results list} -- the direct input to xas_lcf_report.build_*."""
    out: Dict[str, List[LCFCombinationResult]] = {}
    for name, energy, y in targets:
        out[name] = combinatorial_lcf(
            name, energy, y, references,
            min_components=params.min_components, max_components=params.max_components,
            weight_bounds=params.weight_bounds, sum_to_one=params.sum_to_one,
            required_refs=params.required_refs, sort_by=params.sort_by,
        )
    return out
