"""Tests for the Composite fit tab (Phase 7) in qt_saxs.py's SaxsWorkspace:
window auto-fill on curve selection, both single-fit paths (auto BIC ladder
via fit_staged, and a manually-chosen preset via CompositeModel directly),
batch fitting across multiple curves, and CSV export. qt_worker runs
synchronously in tests (conftest's autouse fixture), so results are ready
immediately after the triggering call — no wait-loop needed."""
from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pytest

from saxs.qt_saxs import SaxsWorkspace
from saxs.core.composite_fit import build_preset
from saxs.core.curve import Curve


def _ts_curve(d=1200.0, xi=3000.0, name="synth_ts") -> Curve:
    """A well-conditioned synthetic BG_TS_GP curve (linear q-grid,
    realistic magnitudes) — reuses test_composite_staged.py's own
    _ts_curve parametrization (same defaults, d=1200/xi=3000 verified
    there to converge reliably) rather than picking untested (d, xi)
    values: Phase 6's synthetic harness found that specific (d, xi)
    combinations for a BG_TS_GP curve can be genuinely harder to recover
    (a real bias-variance tradeoff in the windowed fit, investigated and
    partly mitigated in composite_staged.py's propose_windows) — this
    tab's tests exist to prove the UI WIRING works, not to re-litigate
    that already-covered scientific accuracy question, so they stick to
    combinations already known to be robust."""
    q = np.linspace(1e-3, 0.3, 900)
    model = build_preset("BG_TS_GP")
    true = {"bg_C": 500.0, "pl_B": 1e-9, "pl_p": 4.0,
            "ts_S": 5e6, "ts_d": d, "ts_xi": xi,
            "gp_G": 4e8, "gp_Rg": 2000.0, "gp_p": 4.0}
    intensity = model.eval(q, true)
    return Curve(q=q, intensity=intensity, sigma=None, name=name)


def test_composite_tab_registered(qtbot):
    widget = SaxsWorkspace()
    qtbot.addWidget(widget)
    labels = [widget.tabs.tabText(i) for i in range(widget.tabs.count())]
    assert "Composite fit" in labels


def test_composite_tab_auto_fills_windows_on_curve_select(qtbot):
    widget = SaxsWorkspace()
    qtbot.addWidget(widget)
    widget.add_curve(_ts_curve())
    widget.comp_combo.setCurrentText("synth_ts")
    qtbot.wait(20)
    for key, (lo_edit, hi_edit) in widget.comp_window_edits.items():
        assert lo_edit.text(), f"{key} lo not auto-filled"
        assert hi_edit.text(), f"{key} hi not auto-filled"
        assert float(hi_edit.text()) > float(lo_edit.text())


def test_composite_fit_auto_ladder_recovers_ts_peak(qtbot):
    widget = SaxsWorkspace()
    qtbot.addWidget(widget)
    widget.add_curve(_ts_curve(d=1200.0, xi=3000.0))
    widget.comp_combo.setCurrentText("synth_ts")
    widget.comp_preset_combo.setCurrentText("Auto (BIC ladder)")
    widget.comp_multistart_edit.setText("4")
    widget.run_composite_fit()
    qtbot.wait(20)
    assert widget._comp_last is not None
    derived = widget._comp_last["derived"]
    assert derived["d"] == pytest.approx(1200.0, rel=0.15)
    report = widget.comp_report.toPlainText()
    assert "Preset:" in report and "d = " in report
    ax = widget.comp_plot.figure.get_axes()[0]
    assert len(ax.get_lines()) >= 2  # data + total, at minimum


def test_composite_fit_manual_preset_includes_flat_aliases(qtbot):
    """A manually-picked preset (bypassing the BIC ladder) must still
    surface the spec's flat d/xi aliases, not just the raw component
    derived() dict -- a real gap found and fixed while building this tab."""
    widget = SaxsWorkspace()
    qtbot.addWidget(widget)
    widget.add_curve(_ts_curve(d=1000.0, xi=3000.0))
    widget.comp_combo.setCurrentText("synth_ts")
    widget.comp_preset_combo.setCurrentText("BG_TS")
    widget.comp_multistart_edit.setText("3")
    widget.run_composite_fit()
    qtbot.wait(20)
    derived = widget._comp_last["derived"]
    assert "d" in derived and "xi" in derived
    assert derived["d"] == pytest.approx(1000.0, rel=0.05)
    assert widget._comp_last["preset"] == "BG_TS"


def test_composite_fit_warns_without_a_curve(qtbot):
    widget = SaxsWorkspace()
    qtbot.addWidget(widget)
    with patch("qt_saxs.QMessageBox.warning") as mock_warn:
        widget.run_composite_fit()
    mock_warn.assert_called_once()


def test_composite_batch_populates_table_and_never_averages_across_curves(qtbot):
    widget = SaxsWorkspace()
    qtbot.addWidget(widget)
    widget.add_curve(_ts_curve(d=1100.0, xi=2800.0, name="a"))
    widget.add_curve(_ts_curve(d=1200.0, xi=3000.0, name="b"))
    widget.comp_multistart_edit.setText("4")
    widget.comp_batch_list.selectAll()
    widget.run_composite_batch()
    qtbot.wait(20)
    assert widget.comp_batch_table.rowCount() == 2
    sample_ids = {widget.comp_batch_table.item(r, 0).text() for r in range(2)}
    assert sample_ids == {"a", "b"}
    d_by_id = {widget.comp_batch_table.item(r, 0).text(): float(widget.comp_batch_table.item(r, 2).text())
               for r in range(2)}
    assert d_by_id["a"] == pytest.approx(1100.0, rel=0.15)
    assert d_by_id["b"] == pytest.approx(1200.0, rel=0.15)


def test_composite_batch_csv_export_writes_a_file(qtbot, tmp_path):
    widget = SaxsWorkspace()
    qtbot.addWidget(widget)
    widget.add_curve(_ts_curve(name="a"))
    widget.comp_multistart_edit.setText("3")
    widget.comp_batch_list.selectAll()
    widget.run_composite_batch()
    qtbot.wait(20)

    out_path = str(tmp_path / "batch_out.csv")
    with patch("qt_saxs.QFileDialog.getSaveFileName", return_value=(out_path, "")):
        widget.export_composite_batch_csv()
    assert os.path.isfile(out_path)
    assert os.path.getsize(out_path) > 0


def test_composite_batch_export_without_a_run_shows_info(qtbot):
    widget = SaxsWorkspace()
    qtbot.addWidget(widget)
    with patch("qt_saxs.QMessageBox.information") as mock_info:
        widget.export_composite_batch_csv()
    mock_info.assert_called_once()


def test_composite_fit_real_profile_smoke(qtbot):
    """Sanity check against the same committed real fixture the frozen
    regression test uses -- confirms the UI path (not just fit_staged
    called directly) produces a sane, non-crashing result on real data."""
    from saxs.core.loader import load_curve
    fixture = os.path.join(os.path.dirname(__file__), "fixtures", "P5Bi8-12__corr.dat")
    widget = SaxsWorkspace()
    qtbot.addWidget(widget)
    widget.add_curve(load_curve(fixture))
    widget.comp_combo.setCurrentText(widget.curves[0].name)
    widget.comp_multistart_edit.setText("3")
    widget.run_composite_fit()
    qtbot.wait(20)
    assert widget._comp_last is not None
    assert widget._comp_last["gof"]["n_points"] > 0
