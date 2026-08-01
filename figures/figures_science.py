"""
figures_science.py — the Figures engine (framework-agnostic): everything the
publication-figure workspace needs that isn't Qt.

- FIT_MODELS: the point-fitting library (Origin's most-used curve models):
  linear, polynomial 2/3, exponential decay/growth, power law, logarithmic,
  Boltzmann sigmoid, Gaussian, Lorentzian, Arrhenius. Each entry knows its
  formula text, builds an initial guess from the data, and fits via
  scipy.optimize.curve_fit — returning parameters ±1σ and R².
- Ternary geometry (built natively so the portable exe needs no extra
  package): barycentric → cartesian, frame/grid construction — the shape of
  the user's own P-Bi ternary notebooks.
- STYLE_PRESETS: rcParams bundles (publication / presentation / poster)
  and figure sizing in centimeters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


# =============================================================================
# Point fitting
# =============================================================================

@dataclass
class PointFitResult:
    model: str
    formula: str
    params: List[float]
    param_names: List[str]
    stderr: List[float]          # 1σ from the covariance (NaN when unavailable)
    r_squared: float
    x_fit: np.ndarray
    y_fit: np.ndarray

    def report(self) -> str:
        lines = [f"{self.model}: {self.formula}   (R² = {self.r_squared:.5f})"]
        for name, p, e in zip(self.param_names, self.params, self.stderr):
            err = f" ± {e:.4g}" if np.isfinite(e) else ""
            lines.append(f"  {name} = {p:.6g}{err}")
        return "\n".join(lines)


def _guess_linear(x, y):
    return [np.polyfit(x, y, 1)[0], float(np.mean(y))]


def _guess_exp(x, y, sign):
    y0 = float(np.min(y)) if sign < 0 else 0.0
    a = float(y[0] - y0) or 1.0
    span = float(x[-1] - x[0]) or 1.0
    return [a, sign * 3.0 / span, y0]


def _guess_peak(x, y):
    i = int(np.argmax(y))
    base = float(np.min(y))
    width = float(x[-1] - x[0]) / 10.0 or 1.0
    return [float(y[i] - base), float(x[i]), width, base]


def _guess_boltzmann(x, y):
    return [float(y[0]), float(y[-1]), float(np.median(x)), float(x[-1] - x[0]) / 10.0 or 1.0]


FIT_MODELS: Dict[str, Dict] = {
    "Linear (a·x + b)": {
        "f": lambda x, a, b: a * x + b,
        "names": ["a", "b"], "formula": "y = a·x + b",
        "guess": _guess_linear},
    "Polynomial 2 (a·x² + b·x + c)": {
        "f": lambda x, a, b, c: a * x ** 2 + b * x + c,
        "names": ["a", "b", "c"], "formula": "y = a·x² + b·x + c",
        "guess": lambda x, y: list(np.polyfit(x, y, 2))},
    "Polynomial 3": {
        "f": lambda x, a, b, c, d: a * x ** 3 + b * x ** 2 + c * x + d,
        "names": ["a", "b", "c", "d"], "formula": "y = a·x³ + b·x² + c·x + d",
        "guess": lambda x, y: list(np.polyfit(x, y, 3))},
    "Exponential decay (a·e^(−k·x) + y0)": {
        "f": lambda x, a, k, y0: a * np.exp(-np.abs(k) * (x - x.min())) + y0,
        "names": ["a", "k", "y0"], "formula": "y = a·e^(−k·(x−x₀)) + y0",
        "guess": lambda x, y: _guess_exp(x, y, +1)},
    "Exponential growth (a·e^(k·x) + y0)": {
        "f": lambda x, a, k, y0: a * np.exp(np.abs(k) * (x - x.min())) + y0,
        "names": ["a", "k", "y0"], "formula": "y = a·e^(k·(x−x₀)) + y0",
        "guess": lambda x, y: _guess_exp(x, y, +1)},
    "Power law (a·x^n)": {
        "f": lambda x, a, n: a * np.power(np.clip(x, 1e-12, None), n),
        "names": ["a", "n"], "formula": "y = a·xⁿ",
        "guess": lambda x, y: [float(np.max(np.abs(y))) or 1.0, 1.0]},
    "Logarithmic (a·ln(x) + b)": {
        "f": lambda x, a, b: a * np.log(np.clip(x, 1e-12, None)) + b,
        "names": ["a", "b"], "formula": "y = a·ln(x) + b",
        "guess": lambda x, y: [1.0, float(np.mean(y))]},
    "Boltzmann sigmoid": {
        "f": lambda x, a1, a2, x0, dx: a2 + (a1 - a2) / (1.0 + np.exp((x - x0) / dx)),
        "names": ["a1", "a2", "x0", "dx"], "formula": "y = a2 + (a1−a2)/(1+e^((x−x0)/dx))",
        "guess": _guess_boltzmann},
    "Gaussian peak": {
        "f": lambda x, a, x0, w, y0: a * np.exp(-0.5 * ((x - x0) / w) ** 2) + y0,
        "names": ["a", "x0", "w", "y0"], "formula": "y = a·e^(−(x−x0)²/2w²) + y0",
        "guess": _guess_peak},
    "Lorentzian peak": {
        "f": lambda x, a, x0, w, y0: a * w ** 2 / ((x - x0) ** 2 + w ** 2) + y0,
        "names": ["a", "x0", "w", "y0"], "formula": "y = a·w²/((x−x0)²+w²) + y0",
        "guess": _guess_peak},
    "Arrhenius (ln y = ln A − Ea/(R·x))": {
        "f": lambda x, lnA, Ea: lnA - Ea / (8.314462618 * np.clip(x, 1e-12, None)),
        "names": ["lnA", "Ea (J/mol)"], "formula": "ln y = ln A − Ea/(R·T) — fit y as ln(y), x as T (K)",
        "guess": lambda x, y: [float(np.max(y)), 50000.0]},
}


def fit_points(x: Sequence[float], y: Sequence[float], model: str,
               n_curve: int = 400) -> PointFitResult:
    """Fit one FIT_MODELS entry to (x, y) points; returns parameters ±1σ,
    R², and a smooth curve for plotting."""
    from scipy.optimize import curve_fit
    spec = FIT_MODELS[model]
    x = np.asarray(list(x), float)
    y = np.asarray(list(y), float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    order = np.argsort(x)
    x, y = x[order], y[order]
    if len(x) < len(spec["names"]):
        raise ValueError(f"'{model}' needs at least {len(spec['names'])} points.")

    p0 = spec["guess"](x, y)
    params, cov = curve_fit(spec["f"], x, y, p0=p0, maxfev=20000)
    with np.errstate(invalid="ignore"):
        stderr = np.sqrt(np.diag(cov)) if cov is not None and np.all(np.isfinite(cov)) else np.full(len(params), np.nan)

    y_model = spec["f"](x, *params)
    ss_res = float(np.sum((y - y_model) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    x_fit = np.linspace(x[0], x[-1], n_curve)
    return PointFitResult(
        model=model, formula=spec["formula"], params=[float(p) for p in params],
        param_names=list(spec["names"]), stderr=[float(e) for e in stderr],
        r_squared=r2, x_fit=x_fit, y_fit=spec["f"](x_fit, *params),
    )


# =============================================================================
# Ternary geometry (native barycentric — the user's P-Bi notebooks' shape)
# =============================================================================

_SQRT3_2 = np.sqrt(3.0) / 2.0


def ternary_to_xy(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """(a, b, c) compositions → cartesian. Normalized internally, so raw
    percentages or fractions both work. Corners: a=1 bottom-left, b=1
    bottom-right, c=1 top."""
    a = np.asarray(list(a), float)
    b = np.asarray(list(b), float)
    c = np.asarray(list(c), float)
    total = a + b + c
    total = np.where(total > 0, total, 1.0)
    a, b, c = a / total, b / total, c / total
    x = b + 0.5 * c
    y = _SQRT3_2 * c
    return x, y


def ternary_frame(n_grid: int = 5):
    """Triangle outline + gridlines for a ternary axes: returns
    (frame_segments, grid_segments, corner_xy) as line-coordinate arrays."""
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, _SQRT3_2]])
    frame = [(corners[i], corners[(i + 1) % 3]) for i in range(3)]
    grid = []
    for k in range(1, n_grid):
        f = k / n_grid
        # lines of constant a, b, c
        grid.append((ternary_to_xy([1 - f], [f], [0]), ternary_to_xy([1 - f], [0], [f])))
        grid.append((ternary_to_xy([f], [1 - f], [0]), ternary_to_xy([0], [1 - f], [f])))
        grid.append((ternary_to_xy([f], [0], [1 - f]), ternary_to_xy([0], [f], [1 - f])))
    grid_xy = [((float(p1[0][0]), float(p1[1][0])), (float(p2[0][0]), float(p2[1][0]))) for p1, p2 in grid]
    return frame, grid_xy, corners


def draw_ternary_axes(ax, labels: Tuple[str, str, str] = ("A", "B", "C"), n_grid: int = 5):
    """Set up a matplotlib axes as a ternary plot; returns the axes."""
    frame, grid, corners = ternary_frame(n_grid)
    for (p1, p2) in grid:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="0.8", lw=0.6, zorder=0)
    for (p1, p2) in frame:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="black", lw=1.2, zorder=1)
    ax.text(-0.04, -0.04, labels[0], ha="right", va="top", fontsize=11)
    ax.text(1.04, -0.04, labels[1], ha="left", va="top", fontsize=11)
    ax.text(0.5, _SQRT3_2 + 0.04, labels[2], ha="center", va="bottom", fontsize=11)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, _SQRT3_2 + 0.1)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


# =============================================================================
# Style presets + size helpers
# =============================================================================

STYLE_PRESETS: Dict[str, Dict] = {
    "Publication": {
        "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.linewidth": 0.8, "lines.linewidth": 1.0,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "axes.grid": False,
    },
    "Presentation": {
        "font.size": 14, "axes.labelsize": 16, "axes.titlesize": 16,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
        "axes.linewidth": 1.2, "lines.linewidth": 2.0,
        "xtick.direction": "out", "ytick.direction": "out", "axes.grid": True,
    },
    "Poster": {
        "font.size": 18, "axes.labelsize": 20, "axes.titlesize": 22,
        "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 16,
        "axes.linewidth": 1.6, "lines.linewidth": 2.5,
        "xtick.direction": "out", "ytick.direction": "out", "axes.grid": False,
    },
}


def cm_to_inches(cm: float) -> float:
    return float(cm) / 2.54


# Per-layer plot types the XY builder offers (label → renderer key)
LAYER_TYPES = ["Line", "Scatter", "Line + symbols", "Sticks (vlines)", "Filled area", "Bars", "Steps"]
