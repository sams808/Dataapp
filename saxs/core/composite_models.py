"""
saxs_core/composite_models.py — physically meaningful SAXS model components
for the composite-fitting engine (composite_fit.py / composite_staged.py).

General-purpose: usable on ANY 1D SAXS curve, not tied to a specific sample
series or naming scheme. Each component is a small class wrapping a pure
function I(q; local params) -> array (vectorized, float64), exposing:
  - `params()`                    -> list[Param] with defaults/bounds/units/docs
  - `eval(q, **local_kwargs)`     -> np.ndarray
  - `derived(**local_kwargs)`     -> dict of physically meaningful derived
                                     quantities (mostly relevant for teubner_strey)
  - `seed(q, I, windows=None)`    -> dict of heuristic initial values; a
                                     GENERIC, standalone fallback (assumes this
                                     component alone explains the given data).
                                     The staged pipeline (composite_staged.py)
                                     implements the more elaborate,
                                     multi-component seeding described in the
                                     spec's stage sequence on top of these.

Components never include their own flat background — background is its own
component (FlatBackground), summed in by the composition engine.

Formulas, parameter bounds, and citations follow
PRISM_composite_models_spec.md §1 verbatim.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
from scipy.special import erf

Windows = Dict[str, Tuple[float, float]]


@dataclass
class Param:
    name: str
    value: float
    min: float
    max: float
    vary: bool = True
    unit: str = ""
    doc: str = ""


def _window_mask(q: np.ndarray, windows: Optional[Windows], key: str) -> Optional[np.ndarray]:
    if not windows or key not in windows:
        return None
    lo, hi = sorted(windows[key])
    return (q >= lo) & (q <= hi)


def _exclude_mask(q: np.ndarray, windows: Optional[Windows], key: str) -> np.ndarray:
    """True where q is OUTSIDE the named window (all-True when absent)."""
    m = _window_mask(q, windows, key)
    return np.ones_like(q, dtype=bool) if m is None else ~m


class Component:
    """Base for one SAXS scattering contribution. Subclasses implement
    `params`/`eval`; `derived`/`seed` have generic fallbacks."""
    name: str = ""

    def params(self) -> List[Param]:
        raise NotImplementedError

    def eval(self, q: np.ndarray, **kw: float) -> np.ndarray:
        raise NotImplementedError

    def derived(self, **kw: float) -> Dict[str, float]:
        return {}

    def seed(self, q: np.ndarray, I: np.ndarray, windows: Optional[Windows] = None) -> Dict[str, float]:
        raise NotImplementedError


# =============================================================================
# 1.1 flat_background — I(q) = C
# =============================================================================

class FlatBackground(Component):
    name = "flat_background"

    def params(self) -> List[Param]:
        return [Param("C", 0.0, 0.0, np.inf, unit="a.u.", doc="Flat background level.")]

    def eval(self, q: np.ndarray, C: float = 0.0) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        return np.full_like(q, float(C))

    def seed(self, q, I, windows=None) -> Dict[str, float]:
        q = np.asarray(q, dtype=float)
        I = np.asarray(I, dtype=float)
        excl = _exclude_mask(q, windows, "W_peak")
        qmax = float(np.max(q)) if q.size else 1.0
        tail = (q >= qmax / 10.0) & excl
        if int(np.sum(tail)) < 3:
            tail = excl if np.any(excl) else np.ones_like(q, dtype=bool)
        vals = I[tail]
        return {"C": float(np.median(vals)) if vals.size else 0.0}


# =============================================================================
# 1.2 power_law — I(q) = B * q^(-p)
# =============================================================================

class PowerLaw(Component):
    name = "power_law"

    def params(self) -> List[Param]:
        return [
            Param("B", 1.0, 0.0, np.inf, unit="a.u.", doc="Power-law prefactor."),
            Param("p", 4.0, 1.0, 4.5, doc=(
                "Porod exponent: p=4 sharp interfaces, 3<p<4 rough/fractal "
                "interfaces, p<3 mass-fractal-like. At the lowest q this term "
                "can also absorb inter-particle (powder grinding) scattering — "
                "do not over-interpret p from the lowest decade.")),
        ]

    def eval(self, q: np.ndarray, B: float = 1.0, p: float = 4.0) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        return float(B) * np.power(np.clip(q, 1e-300, None), -float(p))

    def seed(self, q, I, windows=None) -> Dict[str, float]:
        q = np.asarray(q, dtype=float)
        I = np.asarray(I, dtype=float)
        excl = _exclude_mask(q, windows, "W_peak")
        hiq = _window_mask(q, windows, "W_hiq")
        mask = hiq if hiq is not None else ((q >= float(np.max(q)) / 10.0) & excl)
        mask = mask & (q > 0) & (I > 0)
        if int(np.sum(mask)) < 2:
            mask = (q > 0) & (I > 0)
        if int(np.sum(mask)) < 2:
            return {"B": float(np.max(I)) if I.size else 1.0, "p": 4.0}
        slope, intercept = np.polyfit(np.log(q[mask]), np.log(I[mask]), 1)
        p = float(np.clip(-slope, 1.0, 4.5))
        return {"B": max(float(np.exp(intercept)), 0.0), "p": p}


# =============================================================================
# 1.2b power_law2 — low-q upturn role (v2: PRISM_fit_pipeline_upgrade_
# prompt.md §3). Same math as power_law; a SEPARATE component (not just
# tighter bounds on the same one) because the BG_TS_PL2 preset uses BOTH
# power_law (Porod/background role, high-q side) AND power_law2 (low-q
# upturn role) simultaneously with independent parameters.
# =============================================================================

class PowerLaw2(Component):
    """Low-q upturn as a plain power law (v2 §3) instead of an
    unconstrained guinier_porod, for use when no genuine Guinier knee is
    present in the data (composite_staged.detect_guinier_knee) — for
    powdered/ground samples, the low-q upturn is routinely inter-particle
    or grain-surface (grinding) scattering, not a real finite-size Guinier
    feature, and fitting it with guinier_porod's Rg then goes
    unconstrained (found on the real P5Bi8-12 fit: Rg~1000 Å with no knee
    actually present in the data)."""
    name = "power_law2"

    def params(self) -> List[Param]:
        return [
            Param("B2", 1.0, 0.0, np.inf, unit="a.u.", doc="Low-q upturn prefactor."),
            Param("p2", 3.5, 1.2, 4.3, doc=(
                "Low-q upturn exponent — bounded to [1.2, 4.3] (v3 ADDENDUM: "
                "widened from [2.5, 4.3] after visual inspection showed the "
                "real P5Bi8-12 low-q tail falls as ~q^-1.5..-2, an "
                "Ornstein-Zernike-like large-scale-fluctuation regime, not a "
                "Porod/grain-surface one). See regime_label() for the "
                "fitted-p2 -> physical-regime mapping.")),
        ]

    def eval(self, q: np.ndarray, B2: float = 1.0, p2: float = 3.5) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        return float(B2) * np.power(np.clip(q, 1e-300, None), -float(p2))

    def seed(self, q, I, windows=None) -> Dict[str, float]:
        q = np.asarray(q, dtype=float)
        I = np.asarray(I, dtype=float)
        loq = _window_mask(q, windows, "W_loq")
        mask = loq if loq is not None else (q <= float(np.max(q)) / 10.0 if q.size else np.ones_like(q, dtype=bool))
        mask = mask & (q > 0) & (I > 0)
        if int(np.sum(mask)) < 2:
            mask = (q > 0) & (I > 0)
        if int(np.sum(mask)) < 2:
            return {"B2": float(np.max(I)) if I.size else 1.0, "p2": 3.5}
        slope, intercept = np.polyfit(np.log(q[mask]), np.log(I[mask]), 1)
        p2 = float(np.clip(-slope, 1.2, 4.3))
        return {"B2": max(float(np.exp(intercept)), 0.0), "p2": p2}


def regime_label(p2: float) -> str:
    """v3 ADDENDUM: physical-regime label from a fitted power_law2 exponent.
    <2.5 an Ornstein-Zernike-like large-scale-fluctuation tail (no sharp
    interface); 2.5-3.5 fractal-like (rough/branched interfaces); >3.5
    Porod (smooth, sharp interfaces) -- purely descriptive, not used for
    any fitting/selection decision."""
    p2 = float(p2)
    if p2 < 2.5:
        return "OZ-like fluctuation tail"
    if p2 <= 3.5:
        return "fractal-like"
    return "Porod (surface)"


# =============================================================================
# 1.3 guinier — I(q) = G * exp(-q^2 Rg^2 / 3)
# =============================================================================

class Guinier(Component):
    name = "guinier"

    def params(self) -> List[Param]:
        return [
            Param("G", 1.0, 0.0, np.inf, doc="Forward scattering intensity I(0)."),
            Param("Rg", 100.0, 10.0, 5000.0, unit="Å", doc="Radius of gyration."),
        ]

    def eval(self, q: np.ndarray, G: float = 1.0, Rg: float = 100.0) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        return float(G) * np.exp(-(q ** 2) * (float(Rg) ** 2) / 3.0)

    def seed(self, q, I, windows=None) -> Dict[str, float]:
        q = np.asarray(q, dtype=float)
        I = np.asarray(I, dtype=float)
        loq = _window_mask(q, windows, "W_loq")
        mask = loq if loq is not None else np.ones_like(q, dtype=bool)
        mask = mask & (q > 0) & (I > 0)
        if not np.any(mask):
            mask = (q > 0) & (I > 0)
        if not np.any(mask):
            return {"G": float(np.max(I)) if I.size else 1.0, "Rg": 100.0}
        qm, Im = q[mask], I[mask]
        i0 = int(np.argmin(qm))
        qmin = float(qm[i0])
        Rg = float(np.clip(2.0 * math.pi / max(qmin, 1e-8), 10.0, 5000.0))
        return {"G": max(float(Im[i0]), 0.0), "Rg": Rg}


# =============================================================================
# 1.4 guinier_porod (Hammouda 2010, J. Appl. Cryst. 43, 716 — s=0 case)
# =============================================================================

class GuinierPorod(Component):
    """Smooth Guinier -> power-law crossover in ONE component. Continuity
    of I and dI/dq at q1 is guaranteed by the q1/D construction (proved
    analytically and checked numerically in tests/test_composite_models.py)."""
    name = "guinier_porod"

    def params(self) -> List[Param]:
        return [
            Param("G", 1.0, 0.0, np.inf, doc="Forward scattering intensity I(0)."),
            Param("Rg", 100.0, 10.0, 5000.0, unit="Å", doc="Radius of gyration."),
            Param("p", 4.0, 1.0, 4.5, doc="High-q power-law exponent."),
        ]

    @staticmethod
    def _q1_D(G: float, Rg: float, p: float) -> Tuple[float, float]:
        Rg = max(float(Rg), 1e-12)
        p = float(p)
        q1 = (1.0 / Rg) * math.sqrt(3.0 * p / 2.0)
        D = float(G) * math.exp(-(q1 ** 2) * (Rg ** 2) / 3.0) * (q1 ** p)
        return q1, D

    def eval(self, q: np.ndarray, G: float = 1.0, Rg: float = 100.0, p: float = 4.0) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        q1, D = self._q1_D(G, Rg, p)
        out = np.empty_like(q)
        lo = q <= q1
        out[lo] = float(G) * np.exp(-(q[lo] ** 2) * (float(Rg) ** 2) / 3.0)
        hi = ~lo
        out[hi] = D / np.power(np.clip(q[hi], 1e-300, None), float(p))
        return out

    def derived(self, G: float = 1.0, Rg: float = 100.0, p: float = 4.0, **_) -> Dict[str, float]:
        q1, D = self._q1_D(G, Rg, p)
        return {"q1": q1, "D": D}

    def seed(self, q, I, windows=None) -> Dict[str, float]:
        base = Guinier().seed(q, I, windows)
        return {"G": base["G"], "Rg": base["Rg"], "p": 4.0}


# =============================================================================
# 1.5 beaucage_unified (Beaucage 1995, J. Appl. Cryst. 28, 717) — one level
# =============================================================================

class BeaucageUnified(Component):
    """Use when a level needs BOTH its own Guinier knee and its own
    power-law tail. Prefer guinier_porod (fewer parameters) unless
    residuals demand this."""
    name = "beaucage_unified"

    def params(self) -> List[Param]:
        return [
            Param("G", 1.0, 0.0, np.inf, doc="Guinier prefactor."),
            Param("Rg", 100.0, 10.0, 5000.0, unit="Å", doc="Radius of gyration."),
            Param("B", 1.0, 0.0, np.inf, doc="Porod-regime prefactor."),
            Param("p", 4.0, 1.0, 4.5, doc="Porod exponent."),
        ]

    def eval(self, q: np.ndarray, G: float = 1.0, Rg: float = 100.0,
             B: float = 1.0, p: float = 4.0) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        guinier_term = float(G) * np.exp(-(q ** 2) * (float(Rg) ** 2) / 3.0)
        erf_term = np.clip(erf(q * float(Rg) / math.sqrt(6.0)), 1e-15, None)
        porod_term = float(B) * np.power(erf_term, 3.0 * float(p)) / np.power(np.clip(q, 1e-300, None), float(p))
        return guinier_term + porod_term

    def seed(self, q, I, windows=None) -> Dict[str, float]:
        base = Guinier().seed(q, I, windows)
        pl = PowerLaw().seed(q, I, windows)
        return {"G": base["G"], "Rg": base["Rg"], "B": pl["B"], "p": pl["p"]}


# =============================================================================
# 1.6 dab (Debye–Anderson–Brumberger) — non-peaked random heterogeneity
# =============================================================================

class Dab(Component):
    """The right null model for featureless (no-peak) profiles: random
    two-phase heterogeneity with correlation length xi and NO preferred
    spacing (contrast with teubner_strey)."""
    name = "dab"

    def params(self) -> List[Param]:
        return [
            Param("A", 1.0, 0.0, np.inf, doc="DAB amplitude."),
            Param("xi", 100.0, 1.0, 1e5, unit="Å", doc="Correlation length."),
        ]

    def eval(self, q: np.ndarray, A: float = 1.0, xi: float = 100.0) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        return float(A) / (1.0 + (q ** 2) * (float(xi) ** 2)) ** 2

    def seed(self, q, I, windows=None) -> Dict[str, float]:
        q = np.asarray(q, dtype=float)
        I = np.asarray(I, dtype=float)
        mask = (q > 0) & (I > 0)
        if not np.any(mask):
            return {"A": 1.0, "xi": 100.0}
        order = np.argsort(q[mask])
        qm, Im = q[mask][order], I[mask][order]
        A = float(Im[0])
        below = np.where(Im <= A / 2.0)[0]
        q_half = float(qm[below[0]]) if below.size else float(qm[-1])
        xi = float(np.clip(1.0 / max(q_half, 1e-8), 1.0, 1e5))
        return {"A": max(A, 0.0), "xi": xi}


# =============================================================================
# 1.6b lorentz_oz (Ornstein-Zernike) — v3 ADDENDUM: a large-scale-
# fluctuation low-q tail, I(q) = A / (1 + q^2 xiL^2). Exponent 1 in the
# denominator (vs. Dab's exponent 2) makes this mathematically distinct:
# it falls off as ~q^-2 at high q within its own regime rather than Dab's
# ~q^-4, matching the P5Bi8-12 low-q tail's observed ~q^-1.5..-2 falloff
# (an OZ tail with its knee below q_min) better than a Porod-type term.
# =============================================================================

class LorentzOZ(Component):
    """Ornstein-Zernike-type Lorentzian tail: I(q) = A / (1 + q^2 xiL^2).
    Used as the low-q role in preset BG_TS_OZ, a ladder-only alternative to
    guinier_porod/power_law2 for large-scale-fluctuation-dominated tails
    whose knee sits below the instrument's q_min (so xiL itself may not be
    tightly constrained -- the tail's presence/shape is still meaningful)."""
    name = "lorentz_oz"

    def params(self) -> List[Param]:
        return [
            Param("A", 1.0, 0.0, np.inf, unit="a.u.", doc="OZ tail amplitude."),
            Param("xiL", 5000.0, 500.0, 1e5, unit="Å", doc=(
                "OZ correlation length (knee at q ~ 1/xiL). Bounds "
                "[500, 1e5] Å per the v3 ADDENDUM.")),
        ]

    def eval(self, q: np.ndarray, A: float = 1.0, xiL: float = 5000.0) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        return float(A) / (1.0 + (q ** 2) * (float(xiL) ** 2))

    def seed(self, q, I, windows=None) -> Dict[str, float]:
        # Reuses Dab's own seed heuristic (same qualitative shape:
        # amplitude at low q, half-max descent locates the correlation
        # length) -- the formula for xi from q_half is identical to what
        # this component needs, only the eval() exponent differs.
        base = Dab().seed(q, I, windows)
        return {"A": base["A"], "xiL": float(np.clip(base["xi"], 500.0, 1e5))}


# =============================================================================
# 1.7 teubner_strey — THE peak model (Teubner & Strey 1987, J. Chem. Phys.
# 87, 3195). Physical (S, d, xi) parametrization: far better conditioned
# for least-squares than the classic (a2, c1, c2) form.
# =============================================================================

def ts_classic_from_physical(d: float, xi: float) -> Tuple[float, float, float]:
    """(d, xi) -> classic Teubner-Strey coefficients (a2, c1, c2=1), s.t.
    D(q) == c2*q^4 + c1*q^2 + a2 exactly (see module tests)."""
    kappa = 1.0 / float(xi)
    k = 2.0 * math.pi / float(d)
    a2 = (k ** 2 + kappa ** 2) ** 2
    c1 = -2.0 * (k ** 2 - kappa ** 2)
    return float(a2), float(c1), 1.0


def ts_physical_from_classic(a2: float, c1: float, c2: float = 1.0) -> Tuple[float, float]:
    """Classic (a2, c1, c2) -> (d, xi) — spec §7's formulas, exact inverse
    of ts_classic_from_physical when c2=1."""
    sqrt_a2c2 = math.sqrt(float(a2) / float(c2))
    inv_xi2 = sqrt_a2c2 / 2.0 + float(c1) / (4.0 * float(c2))
    inv_k2 = sqrt_a2c2 / 2.0 - float(c1) / (4.0 * float(c2))
    xi = 1.0 / math.sqrt(max(inv_xi2, 1e-300))
    d = 2.0 * math.pi / math.sqrt(max(inv_k2, 1e-300))
    return float(d), float(xi)


class TeubnerStrey(Component):
    """The peak model for class b/c samples: a broad, Bragg-like maximum
    from a bicontinuous/microphase-separated structure with characteristic
    repeat distance d and correlation length xi."""
    name = "teubner_strey"

    def params(self) -> List[Param]:
        return [
            Param("S", 1.0, 0.0, np.inf, doc="Peak height: I(q_max) = S."),
            Param("d", 1000.0, 10.0, 1e5, unit="Å", doc=(
                "Repeat distance (modulation wavelength). Generic default "
                "bounds shown here — composite_staged.py narrows these to "
                "the active q-window (2π/q_hi, 2π/q_lo) at fit-setup time.")),
            Param("xi", 3000.0, 50.0, 20000.0, unit="Å", doc="Correlation length."),
        ]

    @staticmethod
    def _kkappa(d: float, xi: float) -> Tuple[float, float]:
        return 2.0 * math.pi / float(d), 1.0 / float(xi)

    def eval(self, q: np.ndarray, S: float = 1.0, d: float = 1000.0, xi: float = 3000.0) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        k, kappa = self._kkappa(d, xi)
        Dq = (q ** 2 - (k ** 2 - kappa ** 2)) ** 2 + 4.0 * (k ** 2) * (kappa ** 2)
        return float(S) * 4.0 * (k ** 2) * (kappa ** 2) / np.clip(Dq, 1e-300, None)

    def derived(self, S: float = 1.0, d: float = 1000.0, xi: float = 3000.0, **_) -> Dict[str, float]:
        """fa < 0 iff a peak exists; fa -> -1 = increasingly ordered/
        lamellar-like modulation. Report fa for every TS fit — the single
        most Tomita-comparable order metric."""
        k, kappa = self._kkappa(d, xi)
        a2, c1, c2 = ts_classic_from_physical(d, xi)
        disc = k ** 2 - kappa ** 2
        q_max = math.sqrt(disc) if disc > 0 else float("nan")
        fa = c1 / math.sqrt(4.0 * a2 * c2)
        return {"q_max": q_max, "a2": a2, "c1": c1, "c2": c2, "fa": fa, "has_peak": bool(disc > 0)}

    def seed(self, q, I, windows=None) -> Dict[str, float]:
        q = np.asarray(q, dtype=float)
        I = np.asarray(I, dtype=float)
        peak = _window_mask(q, windows, "W_peak")
        mask = peak if peak is not None else np.ones_like(q, dtype=bool)
        if not np.any(mask):
            mask = np.ones_like(q, dtype=bool)
        qw, Iw = q[mask], I[mask]
        if qw.size == 0:
            return {"S": 1.0, "d": 1000.0, "xi": 3000.0}
        i_peak = int(np.argmax(Iw))
        q_star = float(qw[i_peak])
        S = float(max(Iw[i_peak], 0.0)) or 1.0
        d = float(np.clip(2.0 * math.pi / max(q_star, 1e-8), 10.0, 1e5))
        half = Iw[i_peak] / 2.0
        above = np.where(Iw >= half)[0]
        fwhm_q = float(qw[above[-1]] - qw[above[0]]) if above.size >= 2 else max(qw[-1] - qw[0], 1e-6) / 4.0
        xi = float(np.clip(2.0 * math.pi / max(fwhm_q, 1e-8), 50.0, 20000.0))
        return {"S": S, "d": d, "xi": xi}


# =============================================================================
# 1.8 broad_peak (empirical fallback; sasmodels-compatible)
# =============================================================================

class BroadPeak(Component):
    """Empirical fallback ONLY when teubner_strey refuses to converge on
    shoulder-class profiles; fits using this must be flagged in results."""
    name = "broad_peak"

    def params(self) -> List[Param]:
        return [
            Param("C_lorentz", 1.0, 0.0, np.inf, doc="Peak amplitude."),
            Param("q0", 0.01, 1e-6, 10.0, unit="Å⁻¹", doc="Peak position."),
            Param("xi", 100.0, 1.0, 1e5, unit="Å", doc="Peak width parameter."),
            Param("m", 2.0, 1.5, 4.0, doc="Peak shape exponent."),
        ]

    def eval(self, q: np.ndarray, C_lorentz: float = 1.0, q0: float = 0.01,
             xi: float = 100.0, m: float = 2.0) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        return float(C_lorentz) / (1.0 + (np.abs(q - float(q0)) * float(xi)) ** float(m))

    def seed(self, q, I, windows=None) -> Dict[str, float]:
        base = TeubnerStrey().seed(q, I, windows)
        return {
            "C_lorentz": base["S"],
            "q0": 2.0 * math.pi / base["d"],
            "xi": base["xi"],
            "m": 2.0,
        }


COMPONENTS: Dict[str, Type[Component]] = {
    cls.name: cls for cls in (
        FlatBackground, PowerLaw, PowerLaw2, Guinier, GuinierPorod,
        BeaucageUnified, Dab, LorentzOZ, TeubnerStrey, BroadPeak,
    )
}
