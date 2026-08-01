"""Tests for the SAXS/WAXS module (saxs_core port of pomme + qt_saxs) and
the Hephaestus-style XAS sample-mass calculator (xas_mass)."""
from __future__ import annotations

import numpy as np
import pytest

import xas.xas_mass as xas_mass
from core.qt_models import SpectrumLibrary
from qt_saxs import SaxsWorkspace
from core.qt_shell import MODULES, NAV_ITEMS, PrismMainWindow
from saxs_core.analysis import fit_guinier, fit_pseudo_bragg_peak
from saxs_core.curve import Curve
from saxs_core.waxs import auto_find_peaks, fit_waxs_peaks


def _sphere_curve(rg=30.0, name="sample"):
    q = np.linspace(0.005, 0.5, 600)
    intensity = 1000.0 * np.exp(-(q * rg) ** 2 / 3.0) + 2.0
    return Curve(q=q, intensity=intensity, sigma=None, name=name)


def test_guinier_recovers_rg_on_synthetic_curve():
    c = _sphere_curve(rg=30.0)
    r = fit_guinier(c.q, c.intensity, 0.006, 1.0 / 30.0)
    assert r.Rg == pytest.approx(30.0, rel=0.05)
    assert r.r2 > 0.99


def test_pseudo_bragg_peak_d_spacing():
    q = np.linspace(0.05, 0.6, 800)
    intensity = 50.0 * np.exp(-((q - 0.30) / 0.02) ** 2) + 10.0
    r = fit_pseudo_bragg_peak(q, intensity, 0.2, 0.4)
    assert r.q0 == pytest.approx(0.30, abs=0.005)
    assert r.d_spacing == pytest.approx(2 * np.pi / 0.30, rel=0.02)


def test_waxs_multi_peak_fit_and_crystallinity():
    q = np.linspace(0.5, 4.0, 1200)
    rng = np.random.default_rng(0)
    intensity = (200.0 * np.exp(-((q - 1.5) / 0.03) ** 2)
                 + 120.0 * np.exp(-((q - 2.2) / 0.04) ** 2)
                 + 40.0 * np.exp(-((q - 1.8) / 0.5) ** 2)  # amorphous hump
                 + 5.0 + rng.normal(0, 1.0, q.shape))
    specs = auto_find_peaks(q, intensity)
    assert len(specs) >= 2
    result = fit_waxs_peaks(q, intensity, specs)
    centers = sorted(p.center for p in result.peaks if not p.is_amorphous)
    assert any(abs(c - 1.5) < 0.05 for c in centers)
    assert any(abs(c - 2.2) < 0.05 for c in centers)
    assert result.r2 > 0.9


def test_saxs_workspace_reduction_and_send_to_library(qtbot):
    library = SpectrumLibrary()
    calls = []
    widget = SaxsWorkspace(library=library, on_derived_added=lambda ids: calls.append(ids))
    qtbot.addWidget(widget)
    sample = _sphere_curve(name="samp")
    empty = Curve(q=sample.q, intensity=np.full_like(sample.q, 2.0), sigma=None, name="empty")
    widget.add_curve(sample)
    widget.add_curve(empty)
    widget.red_sample_combo.setCurrentText("samp")
    widget.red_empty_combo.setCurrentText("empty")
    widget.red_mode_combo.setCurrentText("manual")
    widget.run_reduction()
    qtbot.wait(20)
    assert any(c.name == "samp_corr" for c in widget.curves)

    widget.curve_list.selectAll()
    widget.send_to_library()
    assert len(library) == 3
    assert calls and len(calls[0]) == 3


def test_saxs_workspace_analysis_tab_guinier(qtbot):
    widget = SaxsWorkspace()
    qtbot.addWidget(widget)
    widget.add_curve(_sphere_curve(rg=25.0, name="c"))
    widget.ana_combo.setCurrentText("c")
    widget.run_guinier()  # auto region detection fills the range
    qtbot.wait(20)
    assert "Rg" in widget.ana_report.toPlainText()


def test_saxs_module_registered_in_shell(qtbot):
    assert "SAXS/WAXS" in MODULES
    assert "SAXS/WAXS" in NAV_ITEMS
    window = PrismMainWindow()
    qtbot.addWidget(window)
    window.nav.setCurrentRow(NAV_ITEMS.index("SAXS/WAXS"))
    qtbot.wait(20)
    assert window.stack.currentWidget() is window.saxs_page


# --------------------------------------------------------------------------
# Sample-mass calculator
# --------------------------------------------------------------------------

def test_parse_components_formula_and_table():
    assert xas_mass.parse_components("Fe2O3") == [("Fe2O3", 1.0)]
    comps = xas_mass.parse_components("SiO2 58.8\nNa2O 19.6; Bi2O3 19.6")
    assert comps == [("SiO2", 58.8), ("Na2O", 19.6), ("Bi2O3", 19.6)]
    with pytest.raises(Exception):
        xas_mass.parse_components("NotAnElementZz9")


def test_element_mass_fractions_pure_compound():
    w = xas_mass.element_mass_fractions([("Fe2O3", 1.0)])
    assert w["Fe"] == pytest.approx(0.6994, abs=0.001)  # textbook value
    assert w["O"] == pytest.approx(0.3006, abs=0.001)


def test_sample_mass_report_fe2o3_is_physically_sane():
    r = xas_mass.sample_mass_report("Fe2O3", "Fe", "K", pellet_diameter_mm=13.0)
    assert r.edge_energy_ev == pytest.approx(7112.0, abs=5.0)
    assert r.edge_offset_ev == pytest.approx(3.0)  # new default, was hardcoded 50
    # cross-check against xraydb directly, using whatever offset the report used
    import xraydb
    mu_direct = sum(w * xraydb.mu_elam(el, r.edge_energy_ev + r.edge_offset_ev)
                    for el, w in xas_mass.element_mass_fractions([("Fe2O3", 1.0)]).items())
    assert r.mu_rho_above == pytest.approx(mu_direct, rel=1e-6)
    assert r.edge_step_mu_rho > 0
    # mass for mu*t = 2.5: target/mu * area — recompute independently
    area = np.pi * 0.65 ** 2
    assert r.mass_mut_25_mg == pytest.approx(2.5 * area / r.mu_rho_above * 1000, rel=1e-6)
    assert 5.0 < r.mass_mut_25_mg < 100.0  # tens of mg — the realistic pellet range


def test_sample_mass_report_oxide_mixture_bi_l3():
    """The lab's actual case: a mol% oxide composition at the Bi L3 edge."""
    comp = "SiO2 58.8\nNa2O 19.6\nBi2O3 19.6\nUO3 2.0"
    r = xas_mass.sample_mass_report(comp, "Bi", "L3", basis="mol")
    assert r.edge_energy_ev == pytest.approx(13419.0, abs=10.0)
    assert 0.2 < r.absorber_fraction < 0.6  # Bi-heavy glass
    assert r.edge_step_mu_rho > 0
    assert r.mass_step_1_mg > r.mass_mut_1_mg  # step target always needs more mass
    with pytest.raises(ValueError, match="not in the composition"):
        xas_mass.sample_mass_report("SiO2", "Bi", "L3")


def test_xas_workspace_mass_tab(qtbot):
    from xas.qt_xas import XasWorkspace
    widget = XasWorkspace()
    qtbot.addWidget(widget)
    widget._compute_sample_mass()  # defaults: Bi L3 on the Bi glass composition
    text = widget.mass_report_text.toPlainText()
    assert "Bi L3 edge" in text
    assert "μt = 2.5" in text


def test_sample_mass_report_default_target_matches_fixed_25_line():
    """Defaulting target_mut leaves the existing μt=1/2.5 reference values
    untouched — the new parameter is additive, not a behavior change."""
    r = xas_mass.sample_mass_report("Fe2O3", "Fe", "K", pellet_diameter_mm=13.0)
    assert r.target_mut == pytest.approx(2.5)
    assert r.mass_target_mut_mg == pytest.approx(r.mass_mut_25_mg, rel=1e-9)
    text = r.text("Fe", "K", 13.0)
    assert text.count("your target") == 0  # no redundant line when target == 2.5


def test_sample_mass_report_custom_target_absorption_length():
    r = xas_mass.sample_mass_report("Fe2O3", "Fe", "K", pellet_diameter_mm=13.0, target_mut=1.8)
    assert r.target_mut == pytest.approx(1.8)
    area = np.pi * 0.65 ** 2
    assert r.mass_target_mut_mg == pytest.approx(1.8 * area / r.mu_rho_above * 1000, rel=1e-6)
    # a custom target between the two fixed lines sits between their masses
    assert r.mass_mut_1_mg < r.mass_target_mut_mg < r.mass_mut_25_mg
    text = r.text("Fe", "K", 13.0)
    assert "μt = 1.8" in text and "your target" in text
    assert "μt = 1.0" in text and "μt = 2.5" in text  # references still shown


def test_sample_mass_report_rejects_nonpositive_target():
    with pytest.raises(ValueError, match="positive"):
        xas_mass.sample_mass_report("Fe2O3", "Fe", "K", target_mut=0.0)


def test_sample_mass_report_edge_offset_defaults_to_3ev():
    r = xas_mass.sample_mass_report("Fe2O3", "Fe", "K", pellet_diameter_mm=13.0)
    assert r.edge_offset_ev == pytest.approx(3.0)
    assert "E₀+3 eV" in r.text("Fe", "K", 13.0)


def test_sample_mass_report_custom_edge_offset():
    import xraydb
    r3 = xas_mass.sample_mass_report("Fe2O3", "Fe", "K", edge_offset_ev=3.0)
    r50 = xas_mass.sample_mass_report("Fe2O3", "Fe", "K", edge_offset_ev=50.0)
    assert r3.edge_offset_ev == pytest.approx(3.0)
    assert r50.edge_offset_ev == pytest.approx(50.0)
    # far from any edge structure, +3 eV and +50 eV should be close (~1%) --
    # a large mismatch against another program is not explained by this
    # choice alone (see the module docstring / SeO2 investigation).
    assert r3.mu_rho_above == pytest.approx(r50.mu_rho_above, rel=0.05)
    mu_direct_3 = sum(w * xraydb.mu_elam(el, r3.edge_energy_ev + 3.0)
                      for el, w in xas_mass.element_mass_fractions([("Fe2O3", 1.0)]).items())
    assert r3.mu_rho_above == pytest.approx(mu_direct_3, rel=1e-6)


def test_sample_mass_report_rejects_nonpositive_edge_offset():
    with pytest.raises(ValueError, match="positive"):
        xas_mass.sample_mass_report("Fe2O3", "Fe", "K", edge_offset_ev=0.0)


def test_sample_mass_report_seo2_offset_does_not_explain_hephaestus_gap():
    """Regression for the SeO2 investigation: switching the offset from
    50 eV to 3 eV (using PRISM's own xraydb-tabulated edge) barely moves
    mu_rho_above -- confirming the ~6x mass mismatch against a Hephaestus
    reference run at a different absolute energy is a tabulated-edge-energy
    disagreement between the two programs, not this offset choice."""
    r3 = xas_mass.sample_mass_report("SeO2", "Se", "K", edge_offset_ev=3.0)
    r50 = xas_mass.sample_mass_report("SeO2", "Se", "K", edge_offset_ev=50.0)
    assert r3.edge_energy_ev == pytest.approx(12658.0, abs=1.0)
    assert r3.absorber_fraction == pytest.approx(0.7116, abs=0.001)
    assert r3.mu_rho_above == pytest.approx(r50.mu_rho_above, rel=0.05)


def test_xas_workspace_mass_tab_custom_target(qtbot):
    from xas.qt_xas import XasWorkspace
    widget = XasWorkspace()
    qtbot.addWidget(widget)
    widget.mass_target_mut_edit.setText("1.5")
    widget._compute_sample_mass()
    text = widget.mass_report_text.toPlainText()
    assert "μt = 1.5" in text and "your target" in text


def test_xas_workspace_mass_tab_edge_offset_defaults_to_3(qtbot):
    from xas.qt_xas import XasWorkspace
    widget = XasWorkspace()
    qtbot.addWidget(widget)
    assert widget.mass_edge_offset_edit.text() == "3"
    widget._compute_sample_mass()  # defaults: Bi L3 on the Bi glass composition
    text = widget.mass_report_text.toPlainText()
    assert "E₀+3 eV" in text


def test_xas_workspace_mass_tab_custom_edge_offset(qtbot):
    from xas.qt_xas import XasWorkspace
    widget = XasWorkspace()
    qtbot.addWidget(widget)
    widget.mass_edge_offset_edit.setText("50")
    widget._compute_sample_mass()
    text = widget.mass_report_text.toPlainText()
    assert "E₀+50 eV" in text
