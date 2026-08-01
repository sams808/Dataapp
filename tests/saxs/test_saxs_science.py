"""Tests for the SAXS/WAXS module (saxs_core port of pomme + qt_saxs).

Split out of the former tests/test_saxs_and_mass.py, which bundled these
with the unrelated xas_mass tests (tests/xas/test_xas_mass.py) purely
because both features landed in the same historical commit."""
from __future__ import annotations

import numpy as np
import pytest

from core.qt_models import SpectrumLibrary
from saxs.qt_saxs import SaxsWorkspace
from core.qt_shell import MODULES, NAV_ITEMS, PrismMainWindow
from saxs.core.analysis import fit_guinier, fit_pseudo_bragg_peak
from saxs.core.curve import Curve
from saxs.core.waxs import auto_find_peaks, fit_waxs_peaks


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
