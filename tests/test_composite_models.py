"""Tests for saxs_core/composite_models.py — the SAXS composite-model
component library (Phase 1 of the composite-fitting roadmap).

Covers the spec's own continuity/identity requirements (§5.2, §7): the
guinier_porod value/slope continuity at q1, the teubner_strey argmax /
classic-form round-trip / fa range / classic-form equivalence, plus generic
vectorization and bounds sanity for every component.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from saxs_core.composite_models import (
    COMPONENTS, BeaucageUnified, BroadPeak, Dab, FlatBackground, Guinier,
    GuinierPorod, PowerLaw, TeubnerStrey, ts_classic_from_physical,
    ts_physical_from_classic,
)

ALL_COMPONENTS = [
    FlatBackground(), PowerLaw(), Guinier(), GuinierPorod(),
    BeaucageUnified(), Dab(), TeubnerStrey(), BroadPeak(),
]


# ---------------------------------------------------------------------------
# Generic: every component vectorizes cleanly and honors its own defaults
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("comp", ALL_COMPONENTS, ids=lambda c: c.name)
def test_component_eval_is_vectorized_and_finite(comp):
    q = np.linspace(1e-3, 0.5, 400)
    defaults = {p.name: p.value for p in comp.params()}
    y = comp.eval(q, **defaults)
    assert y.shape == q.shape
    assert y.dtype == np.float64
    assert np.all(np.isfinite(y))


@pytest.mark.parametrize("comp", ALL_COMPONENTS, ids=lambda c: c.name)
def test_component_params_within_declared_bounds(comp):
    for p in comp.params():
        assert p.min <= p.value <= p.max


def test_registry_contains_all_ten_components():
    assert set(COMPONENTS) == {
        "flat_background", "power_law", "power_law2", "guinier", "guinier_porod",
        "beaucage_unified", "dab", "lorentz_oz", "teubner_strey", "broad_peak",
    }
    for name, cls in COMPONENTS.items():
        assert cls().name == name


def test_power_law_and_guinier_porod_and_beaucage_share_p_bounds():
    # Spec §1.2: p bounds [1, 4.5] on every power-law-tailed component.
    for comp in (PowerLaw(), GuinierPorod(), BeaucageUnified()):
        p_param = next(p for p in comp.params() if p.name == "p")
        assert p_param.min == pytest.approx(1.0)
        assert p_param.max == pytest.approx(4.5)


def test_broad_peak_m_bounds():
    m_param = next(p for p in BroadPeak().params() if p.name == "m")
    assert m_param.min == pytest.approx(1.5)
    assert m_param.max == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# guinier_porod: value + slope continuity at q1 (spec §1.4, §5.2)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("G,Rg,p", [(100.0, 200.0, 4.0), (5.0, 800.0, 3.2), (1e4, 50.0, 4.3)])
def test_guinier_porod_value_continuity_at_q1(G, Rg, p):
    """The construction claim (spec §1.4) is that BOTH branch formulas,
    evaluated at exactly q1, agree — i.e. D = G*exp(-q1^2 Rg^2/3)*q1^p is
    built so the high-q formula reproduces the low-q (Guinier) value at
    the crossover. Evaluate each branch's formula directly (bypassing
    eval()'s own q<=q1 branch dispatch) so this actually tests the D
    construction rather than trivially re-deriving eval()'s own choice."""
    comp = GuinierPorod()
    q1, D = comp._q1_D(G, Rg, p)
    i_lo_formula = G * math.exp(-(q1 ** 2) * (Rg ** 2) / 3.0)
    i_hi_formula = D / (q1 ** p)
    assert abs(i_hi_formula - i_lo_formula) / i_lo_formula < 1e-10


@pytest.mark.parametrize("G,Rg,p", [(100.0, 200.0, 4.0), (5.0, 800.0, 3.2)])
def test_guinier_porod_slope_continuity_at_q1(G, Rg, p):
    """Finite-difference dI/dq from both sides must agree (the spec's own
    claim: 'continuity of I and dI/dq is guaranteed by the q1/D
    construction')."""
    comp = GuinierPorod()
    q1, _D = comp._q1_D(G, Rg, p)
    h = q1 * 1e-4
    # one-sided finite differences approaching q1 from below/above
    slope_below = (comp.eval(np.array([q1]), G=G, Rg=Rg, p=p)[0]
                   - comp.eval(np.array([q1 - h]), G=G, Rg=Rg, p=p)[0]) / h
    slope_above = (comp.eval(np.array([q1 + h]), G=G, Rg=Rg, p=p)[0]
                   - comp.eval(np.array([q1]), G=G, Rg=Rg, p=p)[0]) / h
    assert slope_below == pytest.approx(slope_above, rel=1e-3)


def test_guinier_porod_derived_reports_q1_and_D():
    comp = GuinierPorod()
    d = comp.derived(G=10.0, Rg=300.0, p=4.0)
    q1, D = comp._q1_D(10.0, 300.0, 4.0)
    assert d["q1"] == pytest.approx(q1)
    assert d["D"] == pytest.approx(D)


# ---------------------------------------------------------------------------
# teubner_strey: argmax, classic round-trip, fa range, classic-form equivalence
# ---------------------------------------------------------------------------

def _refine_argmax(q: np.ndarray, y: np.ndarray) -> float:
    """3-point parabolic refinement around the discrete argmax — enough to
    resolve the true continuous maximum to well under 1e-6 relative."""
    i = int(np.argmax(y))
    if i == 0 or i == len(q) - 1:
        return float(q[i])
    x0, x1, x2 = q[i - 1], q[i], q[i + 1]
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if abs(denom) < 1e-300:
        return float(q[i])
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    b = (x2 ** 2 * (y0 - y1) + x1 ** 2 * (y2 - y0) + x0 ** 2 * (y1 - y2)) / denom
    if abs(a) < 1e-300:
        return float(q[i])
    return float(-b / (2 * a))


@pytest.mark.parametrize("d,xi", [(1200.0, 3000.0), (900.0, 4500.0), (1600.0, 2600.0)])
def test_teubner_strey_argmax_matches_analytic_q_max(d, xi):
    comp = TeubnerStrey()
    k, kappa = comp._kkappa(d, xi)
    q_max_analytic = math.sqrt(k ** 2 - kappa ** 2)
    q = np.linspace(max(q_max_analytic * 0.5, 1e-6), q_max_analytic * 1.5, 20000)
    y = comp.eval(q, S=1.0, d=d, xi=xi)
    q_max_numeric = _refine_argmax(q, y)
    assert q_max_numeric == pytest.approx(q_max_analytic, rel=1e-6)
    # cross-check against derived()
    assert comp.derived(S=1.0, d=d, xi=xi)["q_max"] == pytest.approx(q_max_analytic, rel=1e-9)


@pytest.mark.parametrize("d,xi", [(1200.0, 3000.0), (500.0, 10000.0), (2000.0, 5000.0)])
def test_teubner_strey_classic_form_round_trip_is_exact(d, xi):
    a2, c1, c2 = ts_classic_from_physical(d, xi)
    assert c2 == 1.0
    d_rt, xi_rt = ts_physical_from_classic(a2, c1, c2)
    assert d_rt == pytest.approx(d, rel=1e-9)
    assert xi_rt == pytest.approx(xi, rel=1e-9)


@pytest.mark.parametrize("d,xi", [(1200.0, 3000.0), (900.0, 4500.0), (300.0, 20000.0)])
def test_teubner_strey_fa_in_expected_range_when_peak_exists(d, xi):
    comp = TeubnerStrey()
    k, kappa = comp._kkappa(d, xi)
    assert k > kappa  # a peak exists for all the (d, xi) pairs chosen above
    derived = comp.derived(S=1.0, d=d, xi=xi)
    assert derived["has_peak"] is True
    assert -1.0 < derived["fa"] < 0.0


def test_teubner_strey_fa_and_q_max_are_nan_when_no_peak():
    """xi so short (kappa large) that k < kappa: no finite-q maximum."""
    comp = TeubnerStrey()
    derived = comp.derived(S=1.0, d=50000.0, xi=10.0)
    assert derived["has_peak"] is False
    assert math.isnan(derived["q_max"])


@pytest.mark.parametrize("d,xi", [(1200.0, 3000.0), (900.0, 4500.0), (1600.0, 2600.0)])
def test_teubner_strey_matches_classic_form_up_to_normalization(d, xi):
    """Spec §7: D(q) == c2*q^4 + c1*q^2 + a2 exactly with c2=1, so
    S*4k^2*kappa^2 / D(q) must be proportional to 1/(a2 + c1 q^2 + c2 q^4)
    with proportionality constant S*4k^2*kappa^2 — the classic-form
    equivalence the spec asks to keep and cross-check."""
    comp = TeubnerStrey()
    k, kappa = comp._kkappa(d, xi)
    a2, c1, c2 = ts_classic_from_physical(d, xi)
    q = np.linspace(1e-4, 0.05, 500)
    S = 7.5
    physical = comp.eval(q, S=S, d=d, xi=xi)
    classic_shape = 1.0 / (a2 + c1 * q ** 2 + c2 * q ** 4)
    proportionality = S * 4.0 * k ** 2 * kappa ** 2
    np.testing.assert_allclose(physical, proportionality * classic_shape, rtol=1e-10)


# ---------------------------------------------------------------------------
# Component-specific seed() sanity (generic, standalone heuristics)
# ---------------------------------------------------------------------------

def test_flat_background_seed_recovers_known_constant():
    q = np.linspace(1e-3, 0.5, 500)
    I = np.full_like(q, 3.7)
    assert FlatBackground().seed(q, I)["C"] == pytest.approx(3.7, rel=1e-6)


def test_power_law_seed_recovers_known_exponent():
    q = np.linspace(0.05, 0.5, 500)
    B_true, p_true = 2.0, 3.5
    I = B_true * q ** (-p_true)
    seeded = PowerLaw().seed(q, I)
    assert seeded["p"] == pytest.approx(p_true, rel=1e-3)
    assert seeded["B"] == pytest.approx(B_true, rel=1e-2)


def test_guinier_seed_recovers_known_rg_order_of_magnitude():
    q = np.linspace(1e-3, 0.05, 400)
    Rg_true = 300.0
    I = 1000.0 * np.exp(-(q ** 2) * (Rg_true ** 2) / 3.0)
    seeded = Guinier().seed(q, I)
    # seed() is a cheap heuristic (2*pi/q_min), not a fit — only order-of-
    # magnitude / bound sanity is asserted here, not tight recovery.
    assert 10.0 <= seeded["Rg"] <= 5000.0
    assert seeded["G"] > 0


def test_teubner_strey_seed_finds_peak_location():
    comp = TeubnerStrey()
    d_true, xi_true = 1200.0, 3000.0
    q = np.linspace(1e-4, 0.02, 3000)
    I = comp.eval(q, S=50.0, d=d_true, xi=xi_true)
    seeded = comp.seed(q, I)
    assert seeded["d"] == pytest.approx(d_true, rel=0.1)
    assert seeded["S"] > 0


def test_dab_seed_is_positive_and_bounded():
    q = np.linspace(1e-3, 0.5, 400)
    I = Dab().eval(q, A=10.0, xi=500.0)
    seeded = Dab().seed(q, I)
    assert seeded["A"] > 0
    assert 1.0 <= seeded["xi"] <= 1e5


def test_seed_respects_windows_argument():
    """Passing windows must not raise and must still return all required
    keys — the staged pipeline relies on this for stage-specific seeding."""
    q = np.linspace(1e-4, 0.5, 2000)
    I = (TeubnerStrey().eval(q, S=20.0, d=1000.0, xi=3000.0)
         + PowerLaw().eval(q, B=1e-6, p=4.0))
    windows = {"W_peak": (0.003, 0.01), "W_hiq": (0.1, 0.5), "W_loq": (1e-4, 0.002)}
    for comp in ALL_COMPONENTS:
        seeded = comp.seed(q, I, windows=windows)
        assert set(seeded) == {p.name for p in comp.params()}
