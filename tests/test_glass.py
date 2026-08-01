"""Tests for glass_science + qt_glass (optical basicity, GlassNet)."""
from __future__ import annotations

import numpy as np
import pytest

import glass.glass_science as gs


def test_optical_basicity_pure_oxides_match_table():
    assert gs.optical_basicity([("SiO2", 1.0)])["basicity"] == pytest.approx(0.48)
    assert gs.optical_basicity([("Bi2O3", 100.0)])["basicity"] == pytest.approx(1.19)


def test_optical_basicity_oxygen_weighted_mixing():
    # 50:50 mol Na2O:SiO2 -> (0.5*1*1.11 + 0.5*2*0.48) / (0.5*1 + 0.5*2)
    res = gs.optical_basicity([("Na2O", 50.0), ("SiO2", 50.0)])
    assert res["basicity"] == pytest.approx((0.5 * 1.11 + 1.0 * 0.48) / 1.5, abs=1e-6)


def test_optical_basicity_wt_basis_differs_from_mol():
    mol = gs.optical_basicity([("Na2O", 50.0), ("SiO2", 50.0)], basis="mol")["basicity"]
    wt = gs.optical_basicity([("Na2O", 50.0), ("SiO2", 50.0)], basis="wt")["basicity"]
    assert mol != pytest.approx(wt)


def test_optical_basicity_unknown_oxide_raises():
    with pytest.raises(ValueError, match="Known oxides"):
        gs.optical_basicity([("XeO4", 1.0)])


def test_parse_composition_table_with_and_without_names():
    df = gs.parse_composition_table("name SiO2 Na2O\nA 70 30\nB 60 40")
    assert list(df.index) == ["A", "B"]
    assert df.loc["A", "SiO2"] == 70.0
    df2 = gs.parse_composition_table("SiO2,Na2O\n70,30")
    assert df2.iloc[0, 1] == 30.0


@pytest.mark.skipif(not gs.glassnet_available(), reason="glasspy not installed")
def test_glassnet_predicts_finite_tg():
    import pandas as pd
    df = pd.DataFrame([{"SiO2": 70.0, "Na2O": 30.0}])
    pred = gs.glassnet_predict(df)
    tg_cols = [c for c in pred.columns if "tg" in str(c).lower()]
    assert tg_cols
    assert np.isfinite(float(pred[tg_cols[0]].iloc[0]))


def test_glass_workspace_basicity_flow(qtbot):
    from glass.qt_glass import GlassWorkspace
    widget = GlassWorkspace()
    qtbot.addWidget(widget)
    widget.comp_edit.setPlainText("name SiO2 Na2O\nA 70 30")
    widget.compute_basicity()
    assert widget.result_table.rowCount() == 1
    val = float(widget.result_table.item(0, 0).text())
    assert 0.5 < val < 0.8  # sodium silicate range
    assert "PNNL-20184" in widget.status_label.text()


def test_shell_has_glass_page(qtbot):
    from core.qt_shell import NAV_ITEMS, PrismMainWindow
    window = PrismMainWindow()
    qtbot.addWidget(window)
    window.nav.setCurrentRow(NAV_ITEMS.index("Glass"))
    qtbot.wait(20)
    assert window.stack.currentWidget() is window.glass_page
