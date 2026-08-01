"""
xas_science.py — framework-agnostic XAS/XANES/EXAFS processing engine.

Consolidates what used to be three drifted copies (ui_xas_processing.py,
xas_processing_v10.py, EXAMPLES/EXAMPLE_xas_processing.py) into one science
module with no UI dependency at all — no tkinter, no matplotlib. GUI layers
(Tk today, Qt later) import from here; they never contain this logic inline.

Layers, roughly innermost-to-outermost:
  - Math/processing helpers (smoothing, baseline fits, mu-building)
  - Physics/domain logic (Bragg angle-energy correction, edge inference)
  - Larch wrappers (normalization, EXAFS/FT pipeline) — one optional-import shim
  - Data models: Operation/Spectrum/SpectrumStore (session-level), XASData/Bundle
    (single-dataset, main.py-facing compatibility layer)
  - I/O: EasyXAFS zip/csv, Athena .prj, generic table reader
"""
from __future__ import annotations

import json
import math
import re
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

_SCIPY_AVAILABLE = False
try:
    from scipy.interpolate import UnivariateSpline
    _SCIPY_AVAILABLE = True
except Exception:
    _SCIPY_AVAILABLE = False

_SCIPY_SPARSE_AVAILABLE = False
try:
    from scipy import sparse as _sparse
    from scipy.sparse.linalg import spsolve as _spsolve
    _SCIPY_SPARSE_AVAILABLE = True
except Exception:
    _SCIPY_SPARSE_AVAILABLE = False

HC_EV_ANG = 12398.4193  # eV*Å
DEFAULT_LATTICE_A_ANG: Dict[str, float] = {"si": 5.4310205, "ge": 5.6575}


# =====================================================================
# Larch optional dependency — ONE shim, used by everything below.
# Previously copy-pasted verbatim across all three XAS files.
# =====================================================================

def larch_available() -> bool:
    # find_spec: "is it installed" without paying larch's multi-second import
    # at app startup — the real import happens on first Larch-based step.
    import importlib.util
    try:
        return importlib.util.find_spec("larch") is not None
    except Exception:
        return False


LARCH_AVAILABLE = larch_available()


def require_larch():
    """Resolve (Group, xraydb, find_e0, pre_edge, autobk, xftf) or raise ImportError.

    Defends against a few historical larch API layouts (module reshuffles across
    larch versions) via fallback import paths — see the M4 "open items" note
    about whether to keep this or pin one modern larch version.
    """
    try:
        import larch  # noqa: F401
        try:
            from larch import Group
        except Exception:
            try:
                from larch.utils import Group
            except Exception:
                from larch.symboltable import Group

        xafs_mod = None
        try:
            import larch.xafs as xafs_mod
        except Exception:
            pass

        def _get_from_xafs(name: str):
            if xafs_mod is None:
                raise AttributeError(name)
            return getattr(xafs_mod, name)

        try:
            find_e0 = _get_from_xafs("find_e0")
            pre_edge = _get_from_xafs("pre_edge")
            autobk = _get_from_xafs("autobk")
            xftf = _get_from_xafs("xftf")
        except Exception:
            try:
                from larch.xafs import find_e0, pre_edge, autobk, xftf
            except Exception:
                from larch.xafs.xafsutils import find_e0
                from larch.xafs.pre_edge import pre_edge
                from larch.xafs.autobk import autobk
                try:
                    from larch.xafs import xftf
                except Exception:
                    from larch.xafs.xafsft import xftf

        xraydb_mod = None
        try:
            import xraydb as xraydb_mod
        except Exception:
            try:
                from larch import xraydb as xraydb_mod
            except Exception:
                xraydb_mod = None

        return Group, xraydb_mod, find_e0, pre_edge, autobk, xftf
    except Exception as exc:
        raise ImportError(
            "Larch is required for this step (normalization/EXAFS).\n"
            "Note: the portable PRISM.exe deliberately does not bundle Larch — "
            "run the app via PRISM.bat (Python install) for Larch-based steps.\n"
            f"Import error: {exc}"
        ) from exc


def _call_larch_func(func, group, **kwargs):
    try:
        return func(group, **kwargs)
    except TypeError:
        alt = dict(kwargs)
        if "window" in alt and "win" not in alt:
            alt["win"] = alt.pop("window")
        if "win" in alt and "window" not in alt:
            alt["window"] = alt.pop("win")
        if "kweight" in alt and "kw" not in alt:
            alt["kw"] = alt["kweight"]
        try:
            return func(group, **alt)
        except TypeError:
            for key in list(alt.keys()):
                try_alt = dict(alt)
                try_alt.pop(key, None)
                try:
                    return func(group, **try_alt)
                except TypeError:
                    continue
            raise


# =====================================================================
# Math & processing helpers
# =====================================================================

def _interp_to_grid(x_src: np.ndarray, y_src: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    x_src = np.asarray(x_src, float); y_src = np.asarray(y_src, float); x_new = np.asarray(x_new, float)
    m = np.isfinite(x_src) & np.isfinite(y_src)
    x_src, y_src = x_src[m], y_src[m]
    idx = np.argsort(x_src, kind="mergesort")
    x_src, y_src = x_src[idx], y_src[idx]
    return np.interp(x_new, x_src, y_src)


def savgol_coeffs(window: int, poly: int, deriv: int = 0) -> np.ndarray:
    window = int(window); poly = int(poly); deriv = int(deriv)
    if window < 3 or window % 2 == 0:
        raise ValueError("SG window must be odd >= 3")
    if poly < 1 or poly >= window:
        raise ValueError("poly must be >=1 and < window")
    half = window // 2
    x = np.arange(-half, half + 1, dtype=float)
    A = np.vander(x, poly + 1, increasing=True)
    ATA = A.T @ A
    pinv = np.linalg.pinv(ATA) @ A.T
    e = np.zeros(poly + 1)
    e[deriv] = math.factorial(deriv)
    return pinv.T @ e


def savgol_filter(y: np.ndarray, window: int, poly: int) -> np.ndarray:
    y = np.asarray(y, float)
    c = savgol_coeffs(window, poly, deriv=0)
    half = window // 2
    ypad = np.pad(y, (half, half), mode="reflect")
    return np.convolve(ypad, c[::-1], mode="valid")


def rolling_median(y: np.ndarray, window: int) -> np.ndarray:
    y = np.asarray(y, float)
    if window <= 1:
        return y.copy()
    if window % 2 == 0:
        window += 1
    return pd.Series(y).rolling(window=window, center=True, min_periods=1).median().to_numpy(float)


def whittaker_smooth(y: np.ndarray, lam: float = 1e5, d: int = 2) -> np.ndarray:
    """Whittaker-Eilers smoother.

    Uses a sparse banded solve (O(n), via scipy.sparse) rather than a dense
    n x n solve (the original O(n^3)/O(n^2)-memory implementation, which was
    fine for a few hundred points but would be slow/memory-heavy on real
    thousand-point EXAFS scans). Falls back to the dense solve only if scipy's
    sparse module is unavailable.
    """
    y = np.asarray(y, float)
    n = y.size
    if n < 5:
        return y.copy()
    lam = float(lam); d = int(d)

    if _SCIPY_SPARSE_AVAILABLE:
        E = _sparse.eye(n, format="csc")
        D = E
        for _ in range(d):
            D = D[1:] - D[:-1]
        A = (_sparse.eye(n, format="csc") + lam * (D.T @ D)).tocsc()
        try:
            return np.asarray(_spsolve(A, y), dtype=float)
        except Exception:
            pass  # fall through to dense solve below

    I = np.eye(n); D = np.eye(n)
    for _ in range(d):
        D = np.diff(D, axis=0)
    A = I + lam * (D.T @ D)
    try:
        return np.linalg.solve(A, y)
    except Exception:
        return np.linalg.lstsq(A, y, rcond=None)[0]


def smooth_spectrum(y: np.ndarray, method: str, params: Dict[str, Any]) -> np.ndarray:
    method = method.lower().strip()
    if method == "savitzky-golay":
        return savgol_filter(y, int(params["window"]), int(params["poly"]))
    if method == "median+sg":
        return savgol_filter(rolling_median(y, int(params["median_window"])), int(params["sg_window"]), int(params["sg_poly"]))
    if method == "whittaker":
        return whittaker_smooth(y, float(params["lam"]), int(params["d"]))
    if method == "spline" and _SCIPY_AVAILABLE:
        x = np.arange(len(y), dtype=float)
        return np.asarray(UnivariateSpline(x, y, s=float(params.get("s", 0.0)))(x), float)
    raise ValueError(f"Unknown/unavailable smoothing method: {method}")


def fit_chebyshev(energy: np.ndarray, y: np.ndarray, degree: int, mask: np.ndarray) -> np.ndarray:
    from numpy.polynomial import Chebyshev
    x = np.asarray(energy, float); yy = np.asarray(y, float)
    mask = mask & np.isfinite(x) & np.isfinite(yy)
    if mask.sum() < max(20, degree + 2):
        raise ValueError("Not enough points for Chebyshev fit")
    xfit, yfit = x[mask], yy[mask]
    model = Chebyshev.fit(xfit, yfit, int(degree), domain=[float(xfit.min()), float(xfit.max())])
    return np.asarray(model(x), float)


def fit_spline(energy: np.ndarray, y: np.ndarray, s: float, mask: np.ndarray) -> np.ndarray:
    if not _SCIPY_AVAILABLE:
        raise ImportError("SciPy not available for spline fit")
    x = np.asarray(energy, float); yy = np.asarray(y, float)
    mask = mask & np.isfinite(x) & np.isfinite(yy)
    return np.asarray(UnivariateSpline(x[mask], yy[mask], s=float(s))(x), float)


def mu_from_transmission(i0: np.ndarray, it: np.ndarray, logbase: str = "ln") -> np.ndarray:
    eps = 1e-12
    i0 = np.clip(np.asarray(i0, float), eps, np.inf)
    it = np.clip(np.asarray(it, float), eps, np.inf)
    return np.log10(i0 / it) if logbase.lower() == "log10" else np.log(i0 / it)


def build_mu(i0_energy: np.ndarray, i0: np.ndarray, it_energy: np.ndarray, it: np.ndarray, log_mode: str) -> np.ndarray:
    """Interpolate It to I0 grid and build transmission mu."""
    it_interp = _interp_to_grid(it_energy, it, i0_energy)
    return mu_from_transmission(i0, it_interp, logbase=log_mode)


# =====================================================================
# Physics & domain logic
# =====================================================================

def _periodic_table_symbols() -> List[str]:
    return ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
            "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",
            "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr",
            "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au",
            "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U"]


def _parse_crystal2d(crystal2d: str) -> Tuple[str, int, int, int]:
    m = re.search(r"^\s*([A-Za-z]+)\s*\(\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*\)\s*$", crystal2d)
    if not m:
        raise ValueError(f"Cannot parse crystal2d='{crystal2d}'")
    return m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))


def _d_spacing_cubic(a_ang: float, h: int, k: int, l: int) -> float:
    return float(a_ang) / float(np.sqrt(h * h + k * k + l * l))


def _energy_from_theta(theta_deg: np.ndarray, d_ang: float) -> np.ndarray:
    s = np.clip(np.sin(np.deg2rad(np.asarray(theta_deg, float))), 1e-12, 1.0)
    return HC_EV_ANG / (2.0 * float(d_ang) * s)


def angle_energy_correction_bragg(angle_deg: np.ndarray, energy_ref_ev: np.ndarray, scan_def: dict, *, mode: str = "A", fit_linear: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    if "crystal2d" not in scan_def:
        raise KeyError("scan_def missing crystal2d")
    mat, h, k, l = _parse_crystal2d(scan_def["crystal2d"])
    mat_key = mat.lower()
    if mat_key not in DEFAULT_LATTICE_A_ANG:
        raise ValueError(f"Unknown crystal '{mat}'")
    d = _d_spacing_cubic(DEFAULT_LATTICE_A_ANG[mat_key], h, k, l)
    theta_offset = float(scan_def.get("theta_offset", 0.0))

    ang = np.asarray(angle_deg, float)
    e_ref = np.asarray(energy_ref_ev, float)

    E_theta = _energy_from_theta(ang + theta_offset, d)
    E_2theta = _energy_from_theta(ang / 2.0 + theta_offset, d)

    m = np.isfinite(e_ref) & np.isfinite(E_theta) & np.isfinite(E_2theta)
    if m.sum() < 30:
        raise ValueError("Not enough points to infer theta vs 2theta")
    err_theta = float(np.nanmedian(np.abs(E_theta[m] - e_ref[m])))
    err_2theta = float(np.nanmedian(np.abs(E_2theta[m] - e_ref[m])))
    if err_theta <= err_2theta:
        interp, E_bragg, base_err = "theta", E_theta, err_theta
    else:
        interp, E_bragg, base_err = "2theta", E_2theta, err_2theta

    diag: Dict[str, Any] = {"mode": mode, "crystal2d": scan_def["crystal2d"], "angle_interpretation": interp, "median_abs_err_before": base_err, "theta_offset_deg": theta_offset}
    if mode.upper() == "B" or not fit_linear:
        return E_bragg, diag

    mm = np.isfinite(E_bragg) & np.isfinite(e_ref)
    A, B = np.polyfit(E_bragg[mm], e_ref[mm], 1)
    diag["calibration"] = {"A": float(A), "B": float(B)}
    diag["median_abs_err_after"] = float(np.nanmedian(np.abs((A * E_bragg + B)[mm] - e_ref[mm])))
    return A * E_bragg + B, diag


@dataclass
class TiePoint:
    e_before: float
    e_after: float


def apply_alignment_mode_c(e_after: np.ndarray, tiepoints: List[TiePoint], model: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    if len(tiepoints) < 1:
        raise ValueError("Add at least one tie point")
    ea = np.asarray(e_after, float)
    eb = np.array([tp.e_before for tp in tiepoints], float)
    ea_tp = np.array([tp.e_after for tp in tiepoints], float)
    if model == "shift":
        dE = np.nanmedian(eb - ea_tp)
        return ea + dE, {"model": "shift", "dE": float(dE), "n": len(tiepoints)}
    if model == "affine":
        if len(tiepoints) < 2:
            raise ValueError("Affine model needs at least 2 tie points")
        a, b = np.linalg.lstsq(np.vstack([ea_tp, np.ones_like(ea_tp)]).T, eb, rcond=None)[0]
        return a * ea + b, {"model": "affine", "a": float(a), "b": float(b), "n": len(tiepoints)}
    raise ValueError("Unknown model")


def roi_scaled_window(scan_def: dict) -> Optional[Tuple[float, float]]:
    rs = scan_def.get("ROI_Scaled") or {}
    try:
        e1, e2 = float(rs.get("roi_min")), float(rs.get("roi_max"))
        if np.isfinite(e1) and np.isfinite(e2) and e2 > e1:
            return (e1, e2)
    except Exception:
        pass
    return None


def find_e0_from_roi_scaled(energy: np.ndarray, mu: np.ndarray, scan_def: Dict[str, Any]) -> float:
    dmu = np.gradient(mu, energy)
    return float(energy[np.nanargmax(dmu)])


def infer_edge_label_from_roi_scaled(energy: np.ndarray, mu: np.ndarray, scan_def: Dict[str, Any], max_delta: float = 30.0, ambiguity_delta: float = 2.0) -> Tuple[str, Optional[float]]:
    energy = np.asarray(energy, float); mu = np.asarray(mu, float)
    if energy.size < 10 or mu.size < 10:
        return ("XAS(?)", None)

    e0 = float(find_e0_from_roi_scaled(energy, mu, scan_def))
    if not np.isfinite(e0):
        return ("XAS(?)", None)

    try:
        Group, xraydb, find_e0, pre_edge, autobk, xftf = require_larch()
        if xraydb is None:
            return ("XAS(?)", e0)
    except Exception:
        return ("XAS(?)", e0)

    cmin, cmax = float(np.nanmin(energy)), float(np.nanmax(energy))
    try:
        elems = list(getattr(xraydb, "atomic_symbols", []))
    except Exception:
        elems = _periodic_table_symbols()
    if not elems:
        elems = _periodic_table_symbols()

    cands: List[Tuple[float, str, str]] = []
    for sym in elems:
        try:
            edges = xraydb.xray_edges(sym)
            if not edges:
                continue
            for edge_name, edge_obj in edges.items():
                ee_edge = getattr(edge_obj, "energy", None)
                if ee_edge is not None and np.isfinite(float(ee_edge)) and cmin <= float(ee_edge) <= cmax:
                    if abs(float(ee_edge) - e0) <= max_delta:
                        cands.append((abs(float(ee_edge) - e0), sym, str(edge_name)))
        except Exception:
            continue

    if not cands:
        return ("XAS(?)", e0)
    cands.sort(key=lambda t: t[0])
    if len(cands) > 1 and (cands[1][0] - cands[0][0]) < ambiguity_delta:
        return ("XAS(? ?)", e0)
    return (f"XAS({cands[0][1]} {cands[0][2]})", e0)


def edge_text(label: str) -> str:
    m = re.match(r"^XAS\(([^\s\)\?]+)\s+([^\s\)\?]+)\)$", str(label or "").strip())
    if m:
        return f"{m.group(1)} {m.group(2)}"
    return "?"


# =====================================================================
# Larch-dependent higher-level operations
# =====================================================================

def larch_normalize(energy: np.ndarray, mu: np.ndarray, *, e0_method: str, e0_manual: Optional[float], pre1: float, pre2: float, norm1: float, norm2: float, nnorm: int, smooth_for_e0: Optional[Tuple[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    Group, xraydb, find_e0, pre_edge, autobk, xftf = require_larch()
    energy = np.asarray(energy, float); mu = np.asarray(mu, float)

    mu_use = mu
    if smooth_for_e0 is not None:
        try:
            mu_use = smooth_spectrum(mu, smooth_for_e0[0], smooth_for_e0[1])
        except Exception:
            pass

    if e0_method == "manual" and e0_manual is not None and np.isfinite(e0_manual):
        e0 = float(e0_manual)
    elif e0_method == "deriv":
        e0 = float(energy[np.nanargmax(np.gradient(mu_use, energy))])
    else:
        e0 = float(find_e0(energy=energy, mu=mu_use))

    g = Group(energy=energy, mu=mu, e0=e0)
    # pre_edge(g) alone only reads e0 from the group's attributes (see its
    # `if group is not None and e0 is None: e0 = getattr(group, 'e0', None)`)
    # — pre1/pre2/norm1/norm2/nnorm are NOT read from group attributes, so
    # setting them on the Group before calling silently did nothing; Larch
    # fell back to its own auto-computed defaults instead of ours. Must be
    # passed as explicit keyword arguments to the call itself. Found via
    # tests/test_xas_science.py (M11) — no prior test exercised this with
    # non-default pre/norm ranges, so it went uncaught since M4.
    pre_edge(g, e0=e0, pre1=float(pre1), pre2=float(pre2), norm1=float(norm1), norm2=float(norm2), nnorm=int(nnorm))

    return {
        "e0": e0,
        "norm": np.asarray(g.norm, float), "flat": np.asarray(g.flat, float), "deriv": np.asarray(np.gradient(mu_use, energy), float),
        "pre_edge_line": np.asarray(getattr(g, "pre_edge", np.full_like(mu, np.nan)), float),
        "post_edge_line": np.asarray(getattr(g, "post_edge", np.full_like(mu, np.nan)), float),
        "anchors": {"pre1": e0 + float(pre1), "pre2": e0 + float(pre2), "norm1": e0 + float(norm1), "norm2": e0 + float(norm2)},
    }


def larch_exafs_pipeline(energy: np.ndarray, mu: np.ndarray, *, e0_method: str, e0_manual: Optional[float], pre1: float, pre2: float, norm1: float, norm2: float, nnorm: int, rbkg: float, kmin: float, kmax: float, dk: float, kweight: int, window: str, rmax_out: float, smooth_for_e0: Optional[Tuple[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    Group, xraydb, find_e0, pre_edge, autobk, xftf = require_larch()
    energy = np.asarray(energy, float); mu = np.asarray(mu, float)

    mu_use = mu
    if smooth_for_e0 is not None:
        try:
            mu_use = smooth_spectrum(mu, smooth_for_e0[0], smooth_for_e0[1])
        except Exception:
            pass

    if e0_method == "manual" and e0_manual is not None and np.isfinite(e0_manual):
        e0 = float(e0_manual)
    elif e0_method == "deriv":
        e0 = float(energy[np.nanargmax(np.gradient(mu_use, energy))])
    else:
        e0 = float(find_e0(energy=energy, mu=mu_use))

    g = Group(energy=energy, mu=mu, e0=e0)
    # See larch_normalize's identical fix above: pre1/pre2/norm1/norm2/nnorm
    # must be passed as explicit kwargs to pre_edge() — setting them as
    # Group attributes beforehand was silently ignored.
    pre_edge(g, e0=e0, pre1=float(pre1), pre2=float(pre2), norm1=float(norm1), norm2=float(norm2), nnorm=int(nnorm))
    _call_larch_func(autobk, g, rbkg=float(rbkg), kmin=float(kmin), kmax=float(kmax), dk=float(dk))

    k = np.asarray(getattr(g, "k", []), float); chi = np.asarray(getattr(g, "chi", []), float)
    if k.size == 0 or chi.size == 0:
        raise RuntimeError("Larch autobk did not produce k/chi arrays.")

    kw = int(kweight)
    chi_kw = chi * np.power(np.clip(k, 0, np.inf), kw)
    _call_larch_func(xftf, g, kmin=float(kmin), kmax=float(kmax), dk=float(dk), kweight=kw, window=str(window), rmax_out=float(rmax_out))

    return {
        "e0": e0, "norm": np.asarray(getattr(g, "norm", np.full_like(mu, np.nan)), float),
        "flat": np.asarray(getattr(g, "flat", np.full_like(mu, np.nan)), float),
        "deriv": np.asarray(np.gradient(mu_use, energy), float),
        "k": k, "chi": chi, "chi_kw": chi_kw,
        "r": np.asarray(getattr(g, "r", []), float),
        "chir_mag": np.asarray(getattr(g, "chir_mag", []), float),
        "chir_re": np.asarray(getattr(g, "chir_re", []), float),
        "chir_im": np.asarray(getattr(g, "chir_im", []), float),
    }


# =====================================================================
# Data models — session level (SpectrumStore) and single-dataset
# (XASData/Bundle, the layer main.py's import buttons talk to)
# =====================================================================

def _now_ts() -> float:
    return time.time()


def _uid(prefix: str = "sp") -> str:
    return f"{prefix}_{int(_now_ts()*1000)}_{np.random.randint(1000,9999)}"


@dataclass
class Operation:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    when: float = field(default_factory=_now_ts)


@dataclass
class Spectrum:
    sid: str
    name: str
    kind: str
    energy: np.ndarray
    y: np.ndarray
    angle: Optional[np.ndarray] = None
    units: str = "a.u."
    label: str = "XAS(Unknown)"
    e0: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    parents: List[str] = field(default_factory=list)
    history: List[Operation] = field(default_factory=list)

    def copy(self, *, new_name: Optional[str] = None, new_kind: Optional[str] = None) -> "Spectrum":
        return Spectrum(
            sid=_uid("sp"), name=new_name if new_name is not None else self.name,
            kind=new_kind if new_kind is not None else self.kind,
            energy=np.array(self.energy, float).copy(), y=np.array(self.y, float).copy(),
            angle=np.array(self.angle, float).copy() if self.angle is not None else None,
            units=self.units, label=self.label, e0=self.e0,
            meta=dict(self.meta), parents=[self.sid] + list(self.parents), history=list(self.history),
        )


class SpectrumStore:
    def __init__(self):
        self._order: List[str] = []
        self._sp: Dict[str, Spectrum] = {}

    def clear(self):
        self._order.clear(); self._sp.clear()

    def add(self, sp: Spectrum):
        self._sp[sp.sid] = sp; self._order.append(sp.sid)

    def remove(self, sid: str):
        if sid in self._sp:
            del self._sp[sid]
        self._order = [x for x in self._order if x != sid]

    def get(self, sid: str) -> Spectrum:
        return self._sp[sid]

    def all(self) -> List[Spectrum]:
        return [self._sp[sid] for sid in self._order if sid in self._sp]

    def by_kind(self, kinds: Sequence[str]) -> List[Spectrum]:
        return [s for s in self.all() if s.kind in set(kinds)]

    def find_by_name(self, name: str) -> Optional[Spectrum]:
        for s in self.all():
            if s.name == name:
                return s
        return None


@dataclass
class XASData:
    """A single parsed dataset: energy + I0/It transmission channels."""
    path: str
    df: pd.DataFrame
    energy_col: str
    i0_col: str
    it_col: str
    energy: np.ndarray
    i0: np.ndarray
    it: np.ndarray
    scan_def: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass
class Bundle:
    """A raw EasyXAFS-style bundle (one CSV + optional scan_def/metadata json)."""
    name: str
    df: pd.DataFrame
    scan_def: dict
    metadata: dict
    path: str
    npz_bytes: Optional[bytes] = None


# =====================================================================
# I/O — raw table/bundle readers
# =====================================================================

def _safe_json_load_bytes(b: bytes) -> dict:
    return json.loads(b.decode("utf-8", errors="ignore"))


def _safe_json_load_path(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8", errors="ignore"))


def _guess_col(columns: Iterable[str], patterns: List[str]) -> Optional[str]:
    cols = [str(c) for c in columns]
    for pat in patterns:
        r = re.compile(pat, flags=re.IGNORECASE)
        for c in cols:
            if r.search(c):
                return c
    return None


def _extract_energy_angle_signal(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, str]]:
    e_col = _guess_col(df.columns, [r"energy.*ev", r"^energy$"])
    if e_col is None:
        raise ValueError("Could not infer Energy(eV) column.")
    a_col = _guess_col(df.columns, [r"angle.*deg", r"\bangle\b", r"\btheta\b", r"bragg"])
    s_col = _guess_col(df.columns, [r"roi.*countsperlive", r"roi.*c/s", r"roi.*counts"])

    if s_col is None:
        numeric_cols = [str(c) for c in df.columns if np.isfinite(pd.to_numeric(df[c], errors="coerce").to_numpy(float)).mean() > 0.8]
        if not numeric_cols:
            raise ValueError("Could not infer a numeric signal column.")

        def score(c: str) -> float:
            sc = 0.0; cl = c.lower()
            if "roi" in cl:
                sc += 40
            if "count" in cl:
                sc += 20
            if "perlive" in cl or "/s" in cl:
                sc += 10
            if "time" in cl or "dead" in cl:
                sc -= 25
            return sc

        s_col = max([c for c in numeric_cols if c.lower() != e_col.lower()], key=score)

    energy = pd.to_numeric(df[e_col], errors="coerce").to_numpy(float)
    signal = pd.to_numeric(df[s_col], errors="coerce").to_numpy(float)
    angle = pd.to_numeric(df[a_col], errors="coerce").to_numpy(float) if a_col else np.full_like(energy, np.nan)

    m = np.isfinite(energy) & np.isfinite(signal)
    if np.isfinite(angle).any():
        m &= np.isfinite(angle)
    energy, signal, angle = energy[m], signal[m], angle[m]

    idx = np.argsort(energy, kind="mergesort")
    return angle[idx], energy[idx], signal[idx], {"angle_col": a_col or "", "energy_col": e_col, "signal_col": s_col}


def _classify_kind_from_name(name: str) -> str:
    return "I0" if re.search(r"(^|[_\-\s])i0([_\-\s]|$)", name.lower()) else "It"


def read_easyxafs_zip(zip_path: Union[str, Path]) -> List[Dict[str, Any]]:
    zp = Path(zip_path)
    with zipfile.ZipFile(zp, "r") as z:
        csvs = [m for m in z.namelist() if re.search(r"_exd\.csv$", m, flags=re.IGNORECASE)]
        if not csvs:
            raise FileNotFoundError(f"No '*_exd.csv' found in zip: {zp}")

        groups: Dict[str, Dict[str, Optional[str]]] = {}
        for csv in csvs:
            key = str(Path(csv).parent).replace("\\", "/")
            if key == ".":
                key = ""
            groups.setdefault(key, {})["csv"] = csv

        def find_in_group(key: str, pattern: str) -> Optional[str]:
            prefix = (key.rstrip("/") + "/") if key else ""
            in_same = [m for m in z.namelist() if m.startswith(prefix) and re.search(pattern, m, re.IGNORECASE)]
            if in_same:
                return in_same[0]
            in_root = [m for m in z.namelist() if "/" not in m.strip("/") and re.search(pattern, m, re.IGNORECASE)]
            return in_root[0] if in_root else None

        for key in list(groups.keys()):
            groups[key]["scan_def"] = find_in_group(key, r"(?:^|/)scan_def\.json$")
            groups[key]["metadata"] = find_in_group(key, r"(?:^|/)metadata\.json$")

        out = []
        for key, files in groups.items():
            if not files.get("csv"):
                continue
            df = pd.read_csv(io_bytes(z.read(files["csv"])), sep=None, engine="python")
            sd = _safe_json_load_bytes(z.read(files["scan_def"])) if files.get("scan_def") else {}
            md = _safe_json_load_bytes(z.read(files["metadata"])) if files.get("metadata") else {}
            bname = Path(key).name if key else Path(files["csv"]).stem
            name = f"{zp.stem}__{bname}" if len(groups) > 1 else zp.stem
            out.append({"name": name, "df": df, "scan_def": sd, "metadata": md, "source": str(zp)})
        return out


def io_bytes(b: bytes):
    import io as _io
    return _io.BytesIO(b)


def read_csv_dataset(csv_path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(csv_path)
    df = pd.read_csv(p, sep=None, engine="python")
    scan_def = {}; metadata = {}
    for cand in (p.parent / "scan_def.json", p.parent / "metadata.json"):
        if cand.exists():
            try:
                if cand.name == "scan_def.json":
                    scan_def = _safe_json_load_path(cand)
                else:
                    metadata = _safe_json_load_path(cand)
            except Exception:
                pass
    return {"name": p.stem, "df": df, "scan_def": scan_def, "metadata": metadata, "source": str(p)}


def _read_prj_text(path: Path) -> str:
    """Athena .prj files may be gzipped or plain text."""
    import gzip
    try:
        with gzip.open(path, "rb") as fh:
            return fh.read().decode("utf-8", errors="replace")
    except OSError:
        return path.read_text(encoding="utf-8", errors="replace")


def _perl_value(line: str):
    """Evaluate the right-hand side of one Athena perl-format line
    (`@x = (1.0, 2.0, ...);` or `%args = ('label' => 'scan1', ...);`).
    Perl fat commas become plain commas, flattening hashes into the same
    alternating key/value list larch's own parser produces."""
    import ast
    txt = line.split("=", 1)[1].strip().rstrip(";").strip()
    txt = txt.replace("=>", ",")
    return ast.literal_eval(txt)


def parse_athena_prj_no_larch(prj_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Pure-Python Athena .prj reader (no larch dependency) — ported from
    larch.io.athena_project's essential read path so the portable exe (which
    excludes larch) can still IMPORT projects. Handles both the perl-style
    format (Demeter/Athena, larch's writer) and larch's JSON-style format.
    Returns [{name, label, energy, mu, i0?, signal?, e0?}, ...]."""
    p = Path(prj_path)
    text = _read_prj_text(p)
    if "Athena project file -- " not in text[:500]:
        raise ValueError(f"'{p.name}' is not an Athena project file.")

    records: List[Dict[str, Any]] = []
    if "____header" in text[:500]:  # JSON-style (larch's default save)
        import json
        jsdict = json.loads(text)
        order = next((v for k, v in jsdict.items() if k.startswith("_____order")), [])
        for name in order:
            dat = jsdict.get(name)
            if not isinstance(dat, dict) or "x" not in dat or "y" not in dat:
                continue
            rec = {"name": name, "args": dict(dat.get("args", {})),
                   "energy": np.asarray(dat["x"], float), "mu": np.asarray(dat["y"], float)}
            for extra in ("i0", "signal"):
                if extra in dat:
                    rec[extra] = np.asarray(dat[extra], float)
            records.append(rec)
    else:  # perl-style
        raw: Dict[str, Any] = {}
        for line in text.split("\n"):
            if line.startswith("#") or len(line) < 2 or "undef" in line:
                continue
            key = line.split()[0].strip().lstrip("$@%")
            try:
                if key == "old_group":
                    raw["name"] = _perl_value(line)
                elif key == "args":
                    flat = list(_perl_value(line))
                    raw["args"] = {flat[2 * i]: flat[2 * i + 1] for i in range(len(flat) // 2)}
                elif key in ("x", "y", "i0", "signal"):
                    raw[{"x": "energy", "y": "mu"}.get(key, key)] = np.asarray(_perl_value(line), float)
                elif key == "[record]":
                    if "energy" in raw and "mu" in raw:
                        records.append(raw)
                    raw = {}
            except (ValueError, SyntaxError):
                continue  # journal/plot_features lines that don't literal-eval

    for rec in records:
        args = rec.get("args", {})
        rec["label"] = str(args.get("label", rec.get("name", "athena_group")))
        try:
            rec["e0"] = float(args["bkg_e0"])
        except (KeyError, TypeError, ValueError):
            rec["e0"] = None
    return records


def read_athena_prj(prj_path: Union[str, Path]) -> List[Spectrum]:
    """Import an Athena .prj via the pure-Python parser — always, not just
    when larch is absent. (The old larch-reader path iterated read_athena's
    return value as if it were a dict/list; modern larch returns a Group, so
    the import silently yielded ZERO spectra. One tested code path now, and
    the portable exe gets .prj import for free; only normalization/EXAFS
    still need larch.)"""
    p = Path(prj_path)
    out: List[Spectrum] = []
    for rec in parse_athena_prj_no_larch(p):
        energy, mu = rec["energy"], rec["mu"]
        order = np.argsort(energy)
        energy, mu = energy[order], mu[order]
        keep = np.concatenate([[True], np.diff(energy) > 0])  # drop duplicate energies
        out.append(Spectrum(
            sid=_uid("sp"), name=rec["label"], kind="mu", energy=energy[keep], y=mu[keep],
            angle=None, units="a.u.", label="XAS(Imported)", e0=rec["e0"], meta={"source": str(p)},
        ))
    return out


def export_athena_column(path: Union[str, Path], energy: np.ndarray, y: np.ndarray, header_lines: List[str]):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(p, np.column_stack([np.asarray(energy, float), np.asarray(y, float)]), header="\n".join(header_lines + ["# energy  mu_or_y"]), comments="")
    return p


def export_athena_prj_best_effort(path: Union[str, Path], spectra: List[Spectrum]) -> bool:
    Group, xraydb, find_e0, pre_edge, autobk, xftf = require_larch()

    groups = []
    for sp in spectra:
        g = Group(); g.label = sp.name; g.filename = sp.name; g.energy = np.asarray(sp.energy, float); g.mu = np.asarray(sp.y, float)
        if sp.e0 is not None and np.isfinite(sp.e0):
            g.e0 = float(sp.e0)
        groups.append(g)

    writer = None
    try:
        from larch.io import write_athena as writer
    except Exception:
        try:
            from larch.io.athena import write_athena as writer
        except Exception:
            writer = None
    if writer is not None:
        writer(str(path), groups)
        return True

    # Modern larch (e.g. 2025.x) dropped write_athena in favor of the
    # AthenaProject API — without this branch export silently failed.
    try:
        from larch.io import create_athena
    except Exception:
        return False
    project = create_athena(str(path))
    for g in groups:
        project.add_group(g)
    project.save(str(path))
    return True


# =====================================================================
# Compat layer — the API main.py's import buttons talk to.
# Bundle -> XASData -> mu(E) -> edge inference, each its own function so a
# future UI can call incrementally rather than only at whole-pipeline
# granularity.
# =====================================================================

def _xasdata_from_df(df: pd.DataFrame, path: str, *, scan_def: Optional[dict] = None, metadata: Optional[dict] = None) -> XASData:
    df = df.dropna(how="all", axis=0).dropna(how="all", axis=1).copy()

    angle, energy, signal, names = _extract_energy_angle_signal(df)
    energy_col = names.get("energy_col", "Energy")

    i0_col = None
    it_col = None
    cols = [str(c) for c in df.columns]
    for c in cols:
        cl = c.lower()
        if i0_col is None and re.search(r"^i0\b|\bincident\b", cl):
            i0_col = c
        if it_col is None and re.search(r"^it\b|\btransmitted\b|\btrans\b|^if\b|\bfluor\b", cl):
            it_col = c

    if i0_col is None or it_col is None:
        numeric_cols = [
            c for c in cols
            if pd.to_numeric(df[c], errors="coerce").notna().sum() > max(8, int(len(df) * 0.25))
        ]
        numeric_wo_energy = [c for c in numeric_cols if c != energy_col]
        if i0_col is None and numeric_wo_energy:
            i0_col = numeric_wo_energy[0]
        if it_col is None and len(numeric_wo_energy) > 1:
            it_col = numeric_wo_energy[1]

    if not i0_col or not it_col:
        s_col = names.get("signal_col")
        if s_col and s_col in df.columns:
            i0_col = "_i0_synth"
            it_col = "_it_from_signal"
            df[i0_col] = 1.0
            sig = pd.to_numeric(df[s_col], errors="coerce").to_numpy(float)
            sig = np.clip(sig, 1e-12, np.inf)
            df[it_col] = 1.0 / sig
        else:
            raise ValueError("Could not infer I0/It columns from dataset.")

    e = pd.to_numeric(df[energy_col], errors="coerce").to_numpy(float)
    i0 = pd.to_numeric(df[i0_col], errors="coerce").to_numpy(float)
    it = pd.to_numeric(df[it_col], errors="coerce").to_numpy(float)

    mask = np.isfinite(e) & np.isfinite(i0) & np.isfinite(it)
    e = e[mask]; i0 = i0[mask]; it = it[mask]
    order = np.argsort(e, kind="mergesort")

    return XASData(
        path=path, df=df, energy_col=energy_col, i0_col=i0_col, it_col=it_col,
        energy=e[order], i0=i0[order], it=it[order],
        scan_def=scan_def or {}, metadata=metadata or {},
    )


def parse_xas_file(path: Union[str, Path]) -> XASData:
    rec = read_csv_dataset(path)
    return _xasdata_from_df(rec["df"], str(path), scan_def=rec.get("scan_def") or {}, metadata=rec.get("metadata") or {})


def read_athena_project(path: Union[str, Path]) -> List[XASData]:
    """Load Athena .prj groups into XASData-compatible datasets."""
    spectra = read_athena_prj(path)
    out: List[XASData] = []
    for idx, sp in enumerate(spectra):
        energy = np.asarray(getattr(sp, "energy", []), dtype=float)
        mu = np.asarray(getattr(sp, "y", []), dtype=float)
        mask = np.isfinite(energy) & np.isfinite(mu)
        energy = energy[mask]; mu = mu[mask]
        if energy.size == 0:
            continue

        i0 = np.ones_like(mu, dtype=float)
        mu_clipped = np.clip(mu, -700.0, 700.0)
        it = np.exp(-mu_clipped)

        name = getattr(sp, "name", "Athena") or f"Athena_{idx+1}"
        df = pd.DataFrame({"Energy": energy, "I0_synth": i0, "It_from_mu": it, "mu_imported": mu})
        out.append(
            XASData(
                path=f"{path}::{name}::{idx}", df=df, energy_col="Energy", i0_col="I0_synth", it_col="It_from_mu",
                energy=energy, i0=i0, it=it, scan_def={}, metadata={"source": str(path), "athena_name": name},
            )
        )
    return out


def read_bundles_from_zip(zip_path: Union[str, Path]) -> List[Bundle]:
    records = read_easyxafs_zip(zip_path)
    out: List[Bundle] = []
    for rec in records:
        out.append(
            Bundle(name=str(rec.get("name", Path(zip_path).stem)), df=rec["df"], scan_def=rec.get("scan_def") or {}, metadata=rec.get("metadata") or {}, path=str(zip_path))
        )
    return out


def read_bundle(path: Union[str, Path]) -> Bundle:
    p = Path(path)
    if p.is_file() and p.suffix.lower() == ".zip":
        bundles = read_bundles_from_zip(p)
        if len(bundles) != 1:
            raise ValueError(f"Zip contains {len(bundles)} datasets; use read_bundles().")
        return bundles[0]
    raise ValueError("Path must be a .zip EasyXAFS bundle.")


def read_bundles(paths: Union[str, Path, List[Union[str, Path]]]) -> List[Bundle]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    out: List[Bundle] = []
    for p in paths:
        p = Path(p)
        if p.is_file() and p.suffix.lower() == ".zip":
            out.extend(read_bundles_from_zip(p))
        else:
            raise ValueError(f"Unsupported path: {p} (expected .zip)")
    return out


def parse_xas_bundle(bundle: Bundle) -> XASData:
    return _xasdata_from_df(bundle.df.copy(), f"{bundle.path}::{bundle.name}", scan_def=bundle.scan_def, metadata=bundle.metadata)


def deglitch_mu(mu: np.ndarray, *, z: float = 6.0, window: int = 21) -> np.ndarray:
    """Robust glitch removal (M11 fix — see compute_mu below): flag points
    that deviate from a rolling median by more than `z` robust-local-sigma
    (MAD-based, so a handful of large outliers don't inflate the sigma
    estimate the way a plain rolling std would) and replace them with the
    local rolling-median baseline. Standard first-pass EXAFS deglitching."""
    mu = np.asarray(mu, float)
    n = mu.size
    if n < 5 or window < 3:
        return mu.copy()
    if window % 2 == 0:
        window += 1

    baseline = rolling_median(mu, window)
    resid = mu - baseline
    mad = (
        pd.Series(resid)
        .rolling(window=window, center=True, min_periods=1)
        .apply(lambda a: np.median(np.abs(a - np.median(a))), raw=True)
        .to_numpy(float)
    )
    robust_sigma = 1.4826 * mad  # MAD -> sigma conversion for a normal distribution
    positive = robust_sigma[robust_sigma > 1e-12]
    # A hardcoded constant (e.g. 1.0) is the wrong fallback here: mu(E) can
    # legitimately sit at any scale, and a near-noiseless region (MAD == 0
    # in every window) means the true local sigma really is ~0, so even a
    # modest deviation should count as anomalous — not be measured against
    # an arbitrary absolute unit that may swamp the signal's own scale.
    # Scale the floor to the data itself instead.
    data_scale = float(np.nanmedian(np.abs(mu))) or 1.0
    fallback = float(np.nanmedian(positive)) if positive.size else max(1e-9, 1e-6 * data_scale)
    robust_sigma = np.where(robust_sigma > 1e-12, robust_sigma, fallback)

    glitch_mask = np.abs(resid) > (float(z) * robust_sigma)
    out = mu.copy()
    out[glitch_mask] = baseline[glitch_mask]
    return out


def compute_mu(
    xas: XASData,
    *,
    log_base: str = "ln",
    deglitch: bool = False,
    deglitch_z: float = 6.0,
    deglitch_window: int = 21,
    smooth_window: int = 1,
) -> np.ndarray:
    mu = mu_from_transmission(xas.i0, xas.it, logbase=log_base)
    if deglitch:
        # Previously a dead parameter: compute_mu accepted deglitch/
        # deglitch_z/deglitch_window but never referenced them, so passing
        # deglitch=True silently did nothing. Found while auditing this
        # module for the M11 Qt port, per the plan's own note to "confirm
        # whether M11's Qt UI actually exposes them."
        mu = deglitch_mu(mu, z=deglitch_z, window=deglitch_window)
    if int(smooth_window) > 1:
        k = int(smooth_window)
        if k % 2 == 0:
            k += 1
        kernel = np.ones(k, dtype=float) / k
        mu = np.convolve(mu, kernel, mode="same")
    return mu


def infer_xas_edge_from_roi_scaled(energy_ev: np.ndarray, mu: np.ndarray, scan_def: dict, *, max_delta_ev: float = 80.0) -> Dict[str, Any]:
    label, e0 = infer_edge_label_from_roi_scaled(energy_ev, mu, scan_def, max_delta=float(max_delta_ev))
    out: Dict[str, Any] = {"label": "?"}
    m = re.match(r"^XAS\(([^\s\)\?]+)\s+([^\s\)\?]+)\)$", str(label))
    if m:
        out["element"] = m.group(1); out["edge"] = m.group(2); out["label"] = f"{m.group(1)} {m.group(2)}"
    if e0 is not None and np.isfinite(e0):
        out["e0"] = float(e0)
    return out


def infer_xas_edge_from_spectrum(energy_ev: np.ndarray, mu: np.ndarray, *, max_delta_ev: float = 80.0) -> Dict[str, Any]:
    label, e0 = infer_edge_label_from_roi_scaled(energy_ev, mu, {}, max_delta=float(max_delta_ev))
    out: Dict[str, Any] = {"label": "?"}
    m = re.match(r"^XAS\(([^\s\)\?]+)\s+([^\s\)\?]+)\)$", str(label))
    if m:
        out["element"] = m.group(1); out["edge"] = m.group(2); out["label"] = f"{m.group(1)} {m.group(2)}"
    if e0 is not None and np.isfinite(e0):
        out["e0"] = float(e0)
    return out


def load_records_from_paths(paths: Union[str, Path, List[Union[str, Path]]]) -> List[Dict[str, Any]]:
    if isinstance(paths, (str, Path)):
        paths = [paths]

    records: List[Dict[str, Any]] = []
    for p in paths:
        pp = Path(p)
        if pp.is_file() and pp.suffix.lower() == ".zip":
            for b in read_bundles_from_zip(pp):
                xas = parse_xas_bundle(b)
                mu = compute_mu(xas)
                edge = infer_xas_edge_from_roi_scaled(xas.energy, mu, xas.scan_def)
                records.append({"title": f"{edge.get('label', 'XAS(Unknown)')} — {b.name}", "xas": xas, "bundle": b, "edge": edge})
        elif pp.is_file():
            xas = parse_xas_file(pp)
            mu = compute_mu(xas)
            edge = infer_xas_edge_from_roi_scaled(xas.energy, mu, xas.scan_def)
            records.append({"title": f"{edge.get('label', 'XAS(Unknown)')} — {pp.stem}", "xas": xas, "bundle": None, "edge": edge})
        else:
            raise ValueError(f"Unsupported path: {p}")

    if not records:
        raise FileNotFoundError("No datasets found.")
    return records
