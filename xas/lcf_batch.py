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

Tuning knobs added after a real-data review found Bi_metal picking up a
substantial, chemically-implausible weight across nearly every oxide
glass sample: the residual was a sharp, localized spike right at the
edge/white-line, not diffuse EXAFS-level noise -- the signature of
energy misalignment between the target and references being absorbed
into an unphysical weight, not a real 4th phase (Athena's own docs are
explicit that LCF needs pre-aligned spectra for exactly this reason).
`align_e0` fixes the alignment; `per_ref_bounds` and `fit_range` are the
general fine-tuning controls that let a user encode real prior
expectations (e.g. "Bi2O3 should dominate, everything else minor")
instead of an unconstrained blind search.

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
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .xas_science import _interp_to_grid

# Metrics where a LOWER value is better (everything else -- currently
# just r2 -- is higher-is-better). Used to pick sort direction.
_LOWER_IS_BETTER = {"rms", "reduced_chi_square", "ss_res"}

BoundsSpec = Union[Tuple[float, float], Dict[str, Tuple[float, float]]]


def estimate_e0_deriv(energy: np.ndarray, y: np.ndarray) -> float:
    """Simple, Larch-free edge-energy estimate (max of dy/dE) -- good
    enough for ALIGNING spectra before LCF, not meant as a substitute for
    the pipeline's own careful e0 determination. Mirrors the
    e0_method="deriv" fallback already used elsewhere in xas_science.py,
    kept independent here so this module never requires Larch."""
    energy = np.asarray(energy, float)
    y = np.asarray(y, float)
    return float(energy[np.nanargmax(np.gradient(y, energy))])


def _resolve_bounds(names: Sequence[str], default_bounds: Tuple[float, float],
                    per_ref_bounds: Optional[Dict[str, Tuple[float, float]]]) -> Tuple[np.ndarray, np.ndarray]:
    """Per-name (lb, ub) arrays, one row per name in `names` (the order a
    given combination's reference matrix columns are built in) -- names
    not present in `per_ref_bounds` fall back to `default_bounds`."""
    per_ref_bounds = per_ref_bounds or {}
    lb = np.array([per_ref_bounds.get(n, default_bounds)[0] for n in names], dtype=float)
    ub = np.array([per_ref_bounds.get(n, default_bounds)[1] for n in names], dtype=float)
    if np.any(lb > ub):
        bad = [n for n, l, u in zip(names, lb, ub) if l > u]
        raise ValueError(f"lower bound exceeds upper bound for: {bad}")
    return lb, ub


def fit_weighted_sum(target_y: np.ndarray, ref_matrix: np.ndarray, *,
                     weight_bounds: Tuple[np.ndarray, np.ndarray] = (0.0, 1.0),
                     sum_to_one: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Fits target_y ~= ref_matrix @ weights, subject to per-weight bounds
    (a single (lb, ub) pair broadcast to every weight, or a pair of
    per-weight arrays -- see `_resolve_bounds`) and an optional
    sum(weights) == 1 constraint (the "fraction of each phase" reading
    LCF weights are usually given; NOT enforced by default since it's a
    real modeling choice, not always appropriate -- e.g. when self-
    absorption or a normalization mismatch means weights genuinely
    shouldn't sum to exactly 1).

    No constraint beyond bounds -> scipy.optimize.lsq_linear (fast, exact
    bounded least squares, natively supports per-variable bounds).
    sum_to_one -> scipy.optimize.minimize (SLSQP; lsq_linear has no
    equality-constraint support)."""
    target_y = np.asarray(target_y, float)
    n_refs = ref_matrix.shape[1]
    lb, ub = weight_bounds
    lb = np.broadcast_to(np.asarray(lb, float), (n_refs,))
    ub = np.broadcast_to(np.asarray(ub, float), (n_refs,))

    if not sum_to_one:
        from scipy.optimize import lsq_linear
        res = lsq_linear(ref_matrix, target_y, bounds=(lb, ub))
        weights = res.x
    else:
        from scipy.optimize import minimize
        w0 = np.clip(np.full(n_refs, 1.0 / n_refs), lb, ub)

        def objective(w):
            resid = ref_matrix @ w - target_y
            return float(np.sum(resid ** 2))

        cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
        bounds = list(zip(lb.tolist(), ub.tolist()))
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
    fit_energy: np.ndarray
    r2: float
    rms: float
    reduced_chi_square: float
    n_points: int
    e0_shifts_ev: Dict[str, float] = field(default_factory=dict)

    def weight_map(self) -> Dict[str, float]:
        return dict(zip(self.ref_names, (float(w) for w in self.weights)))


def combinatorial_lcf(target_name: str, target_energy: np.ndarray, target_y: np.ndarray,
                      references: Sequence[Tuple[str, np.ndarray, np.ndarray]], *,
                      min_components: int = 2, max_components: int = 3,
                      weight_bounds: Tuple[float, float] = (0.0, 1.0),
                      per_ref_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                      sum_to_one: bool = False,
                      required_refs: Optional[Sequence[str]] = None,
                      sort_by: str = "r2",
                      align_e0: bool = False,
                      fit_range: Optional[Tuple[float, float]] = None) -> List[LCFCombinationResult]:
    """Fits `target` against every combination of `references` (each a
    (name, energy, y) triple) sized min_components..max_components
    (inclusive, both clamped to len(references)), optionally required to
    include every name in `required_refs` (e.g. "always include the
    matrix/host phase, vary the rest"). Returns results sorted best-first
    by `sort_by` ("r2", "rms", or "reduced_chi_square").

    `align_e0`: before interpolating each reference onto the target's
    energy grid, shift the reference's own energy axis so its own
    (derivative-estimated) edge lines up with the target's -- LCF on
    un-aligned spectra shows up as a sharp, localized residual right at
    the edge that a linear weight can't actually correct, and gets
    wrongly absorbed into whichever reference's edge happens to sit
    closest by chance.

    `weight_bounds` is the default (lb, ub) for every reference; `per_ref
    _bounds` overrides it for named references only -- e.g. {"Bi2O3":
    (0.3, 1.0), "Bi_metal": (0.0, 0.2)} to encode "Bi2O3 should dominate,
    Bi_metal should stay minor" as an actual constraint instead of hoping
    the unconstrained fit lands there.

    `fit_range` restricts BOTH the fit itself and its R²/RMS/χ² scoring
    to an (lo, hi) energy window on the target's own (never-shifted)
    energy axis -- e.g. a XANES-only window -- instead of the full
    overlap range."""
    if sort_by not in ("r2", "rms", "reduced_chi_square"):
        raise ValueError(f"sort_by must be 'r2', 'rms', or 'reduced_chi_square', got {sort_by!r}")
    if min_components < 1:
        raise ValueError("min_components must be >= 1")
    if len(references) == 0:
        return []

    target_energy = np.asarray(target_energy, float)
    target_y = np.asarray(target_y, float)
    ref_names_all = [r[0] for r in references]
    if len(set(ref_names_all)) != len(ref_names_all):
        raise ValueError("Reference names must be unique.")
    required = set(required_refs or [])
    if not required.issubset(ref_names_all):
        raise ValueError(f"required_refs contains names not in references: {required - set(ref_names_all)}")

    e0_shifts: Dict[str, float] = {}
    if align_e0:
        target_e0 = estimate_e0_deriv(target_energy, target_y)
        ref_grids = {}
        for name, e, y in references:
            ref_e0 = estimate_e0_deriv(e, y)
            shift = target_e0 - ref_e0
            e0_shifts[name] = shift
            ref_grids[name] = _interp_to_grid(np.asarray(e, float) + shift, y, target_energy)
    else:
        ref_grids = {name: _interp_to_grid(e, y, target_energy) for name, e, y in references}

    if fit_range is not None:
        lo, hi = fit_range
        mask = (target_energy >= lo) & (target_energy <= hi)
        if not np.any(mask):
            raise ValueError(f"fit_range {fit_range} has no overlap with the target's energy axis.")
    else:
        mask = np.ones_like(target_energy, dtype=bool)
    fit_energy = target_energy[mask]
    fit_target_y = target_y[mask]
    fit_ref_grids = {name: y[mask] for name, y in ref_grids.items()}

    max_components = min(max_components, len(references))
    min_components = min(min_components, max_components)

    results: List[LCFCombinationResult] = []
    for k in range(min_components, max_components + 1):
        for combo in itertools.combinations(ref_names_all, k):
            if required and not required.issubset(combo):
                continue
            A = np.column_stack([fit_ref_grids[name] for name in combo])
            bounds = _resolve_bounds(combo, weight_bounds, per_ref_bounds)
            weights, fit_y = fit_weighted_sum(fit_target_y, A, weight_bounds=bounds, sum_to_one=sum_to_one)
            scores = score_fit(fit_target_y, fit_y, n_params=k)
            results.append(LCFCombinationResult(
                target=target_name, ref_names=combo, weights=weights, fit_y=fit_y, fit_energy=fit_energy,
                r2=scores["r2"], rms=scores["rms"], reduced_chi_square=scores["reduced_chi_square"],
                n_points=scores["n_points"],
                e0_shifts_ev={n: e0_shifts[n] for n in combo} if align_e0 else {},
            ))

    reverse = sort_by not in _LOWER_IS_BETTER
    results.sort(key=lambda r: getattr(r, sort_by), reverse=reverse)
    return results


@dataclass
class BatchLCFParams:
    min_components: int = 2
    max_components: int = 3
    weight_bounds: Tuple[float, float] = (0.0, 1.0)
    per_ref_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    sum_to_one: bool = False
    required_refs: Tuple[str, ...] = field(default_factory=tuple)
    sort_by: str = "r2"
    top_n_report: int = 10
    align_e0: bool = False
    fit_range: Optional[Tuple[float, float]] = None


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
            weight_bounds=params.weight_bounds, per_ref_bounds=params.per_ref_bounds,
            sum_to_one=params.sum_to_one, required_refs=params.required_refs, sort_by=params.sort_by,
            align_e0=params.align_e0, fit_range=params.fit_range,
        )
    return out
