"""Tests for qt_htxrd.py (M20) — the HTXRD series-processing workspace.

Run separately from the default Tk-focused suite (see pytest.ini / conftest.py
for why): `pytest tests/test_qt_htxrd.py --override-ini="addopts="`
"""
from __future__ import annotations

import numpy as np
import pytest
import rampy as rp

from xrd.qt_htxrd import HtxrdWorkspace
from core.qt_shell import PrismMainWindow, NAV_ITEMS


def _write_series(tmp_path, temps=(100, 200, 300), center_by_temp=None, noise=0.0):
    x = np.linspace(20, 40, 600)
    rng = np.random.default_rng(0)
    paths = []
    for t in temps:
        center = center_by_temp(t) if center_by_temp else 30.0
        y = rp.gaussian(x, 500.0, center, 0.3) + 50.0 + (rng.normal(0, noise, x.shape) if noise else 0.0)
        p = tmp_path / f"scan_{t}.xy"
        np.savetxt(p, np.column_stack([x, y]))
        paths.append(str(p))
    return paths


def test_workspace_constructs_empty(qtbot):
    widget = HtxrdWorkspace()
    qtbot.addWidget(widget)
    assert widget.series == []
    assert widget.series_list.count() == 0


def test_import_files_with_template_orders_series_by_ramp(qtbot, tmp_path, monkeypatch):
    paths = _write_series(tmp_path, temps=(300, 100, 200))
    widget = HtxrdWorkspace()
    qtbot.addWidget(widget)
    widget.template_edit.setText("scan_???.xy")

    monkeypatch.setattr("xrd.qt_htxrd.QFileDialog.getOpenFileNames", staticmethod(lambda *a, **k: (paths, "")))
    widget.import_files()
    qtbot.wait(200)  # waterfall render is debounced via request_redraw

    assert len(widget.series) == 3
    assert [p.ramp_value for p in widget.series] == [100.0, 200.0, 300.0]
    assert widget.series_list.count() == 3
    assert len(widget.plot.figure.get_axes()) == 1
    assert len(widget.plot.figure.get_axes()[0].lines) == 3


def test_track_peak_populates_table_plot_and_flags(qtbot, tmp_path, monkeypatch):
    def center_by_temp(t):
        return 30.0 - 0.0005 * t  # linear thermal shift

    paths = _write_series(tmp_path, temps=(100, 200, 300, 400), center_by_temp=center_by_temp)
    widget = HtxrdWorkspace()
    qtbot.addWidget(widget)
    widget.template_edit.setText("scan_???.xy")
    monkeypatch.setattr("xrd.qt_htxrd.QFileDialog.getOpenFileNames", staticmethod(lambda *a, **k: (paths, "")))
    widget.import_files()
    qtbot.wait(200)

    widget.windows_edit.setText("29.0-31.0")
    widget.run_track_peak()
    qtbot.wait(20)

    assert list(widget.track_results.keys()) == ["29-31"]
    rows = widget.track_results["29-31"]
    assert len(rows) == 4
    assert all(r.error is None and r.present for r in rows)
    assert widget.track_table.rowCount() == 4
    # Centers should decrease with temperature (built-in linear shift).
    centers = [r.center for r in rows]
    assert centers == sorted(centers, reverse=True)
    # Three stacked subplots: center/FWHM/area vs ramp.
    assert len(widget.track_plot.figure.get_axes()) == 3
    assert "transition" in widget.transition_text.toPlainText().lower()


def test_track_peak_without_series_warns(qtbot):
    widget = HtxrdWorkspace()
    qtbot.addWidget(widget)
    widget.run_track_peak()  # QMessageBox.warning neutralized by conftest fixture
    assert widget.track_results == {}


def test_track_peak_invalid_window_warns(qtbot, tmp_path, monkeypatch):
    paths = _write_series(tmp_path)
    widget = HtxrdWorkspace()
    qtbot.addWidget(widget)
    monkeypatch.setattr("xrd.qt_htxrd.QFileDialog.getOpenFileNames", staticmethod(lambda *a, **k: (paths, "")))
    widget.import_files()
    qtbot.wait(200)

    widget.windows_edit.setText("31.0-29.0")  # inverted
    widget.run_track_peak()
    assert widget.track_results == {}


def test_export_track_csv_writes_file(qtbot, tmp_path, monkeypatch):
    paths = _write_series(tmp_path)
    widget = HtxrdWorkspace()
    qtbot.addWidget(widget)
    widget.template_edit.setText("scan_???.xy")
    monkeypatch.setattr("xrd.qt_htxrd.QFileDialog.getOpenFileNames", staticmethod(lambda *a, **k: (paths, "")))
    widget.import_files()
    qtbot.wait(200)
    widget.windows_edit.setText("29.0-31.0")
    widget.run_track_peak()
    qtbot.wait(20)

    out_path = tmp_path / "track.csv"
    monkeypatch.setattr("xrd.qt_htxrd.QFileDialog.getSaveFileName", staticmethod(lambda *a, **k: (str(out_path), "")))
    widget.export_track_csv()

    assert out_path.exists()
    content = out_path.read_text()
    assert "center_2theta" in content
    assert "present" in content
    assert "scan_100" in content


def test_multi_window_tracking_populates_labels_and_legend(qtbot, tmp_path, monkeypatch):
    # real noise so the absence test has a noise floor to compare against
    paths = _write_series(tmp_path, temps=(100, 200, 300), noise=2.0)
    widget = HtxrdWorkspace()
    qtbot.addWidget(widget)
    widget.template_edit.setText("scan_???.xy")
    monkeypatch.setattr("xrd.qt_htxrd.QFileDialog.getOpenFileNames", staticmethod(lambda *a, **k: (paths, "")))
    widget.import_files()
    qtbot.wait(200)

    widget.windows_edit.setText("29.0-31.0 @ 30.0; 20.5-25.0")
    widget.run_track_peak()
    qtbot.wait(20)

    assert set(widget.track_results.keys()) == {"29-31 @ 30", "20.5-25"}
    assert widget.track_table.rowCount() == 6  # 3 patterns × 2 windows
    # The second window holds no peak (flat baseline) -> absent, not error
    empty_rows = widget.track_results["20.5-25"]
    assert all(r.error is None for r in empty_rows)
    assert not any(r.present for r in empty_rows)


def test_maps_tab_renders_heatmap_difference_and_3d(qtbot, tmp_path, monkeypatch):
    """The notebook-ported Maps tab: every plot type renders on a real
    series without raising, and the difference map resolves its reference
    by temperature."""
    paths = _write_series(tmp_path, temps=(100, 200, 300))
    widget = HtxrdWorkspace()
    qtbot.addWidget(widget)
    widget.template_edit.setText("scan_???.xy")
    monkeypatch.setattr("xrd.qt_htxrd.QFileDialog.getOpenFileNames", staticmethod(lambda *a, **k: (paths, "")))
    widget.import_files()
    qtbot.wait(200)

    for plot_type in ("Heatmap", "Difference map", "Difference waterfall", "3D surface"):
        widget.map_type_combo.setCurrentText(plot_type)
        widget.render_maps()
        qtbot.wait(20)
        assert len(widget.maps_plot.figure.get_axes()) >= 1, plot_type

    # reference by temperature: '210' resolves to the 200 °C pattern
    widget.map_type_combo.setCurrentText("Difference map")
    widget.reference_edit.setText("210.0")
    widget.render_maps()
    qtbot.wait(20)
    title = widget.maps_plot.figure.get_axes()[0].get_title()
    assert "Difference map" in title

    # guides draw on the heatmap without error
    widget.map_type_combo.setCurrentText("Heatmap")
    widget.guides_edit.setText("{1:30.0; 3:29.8}")
    widget.render_maps()
    qtbot.wait(20)
    ax = widget.maps_plot.figure.get_axes()[0]
    assert any(ln.get_linestyle() == "--" for ln in ax.lines)


def test_time_axis_from_heating_rate(qtbot, tmp_path, monkeypatch):
    paths = _write_series(tmp_path, temps=(100, 200, 300))
    widget = HtxrdWorkspace()
    qtbot.addWidget(widget)
    widget.template_edit.setText("scan_???.xy")
    monkeypatch.setattr("xrd.qt_htxrd.QFileDialog.getOpenFileNames", staticmethod(lambda *a, **k: (paths, "")))
    widget.import_files()
    qtbot.wait(200)

    widget.heating_rate_edit.setText("10")
    widget.yaxis_combo.setCurrentText("Time (min)")
    vals, label = widget._y_axis(widget.series)
    assert label == "Time (min)"
    assert np.allclose(vals, [0.0, 10.0, 20.0])

    widget.heating_rate_edit.setText("")  # no rate -> falls back to ramp axis
    vals, label = widget._y_axis(widget.series)
    assert "Time" not in label


def test_real_rasx_example_loads_with_metadata_temperature(qtbot, monkeypatch, examples_dir):
    rasx = examples_dir / "HTXRD_example.rasx"
    widget = HtxrdWorkspace()
    qtbot.addWidget(widget)
    monkeypatch.setattr("xrd.qt_htxrd.QFileDialog.getOpenFileNames", staticmethod(lambda *a, **k: ([str(rasx)], "")))
    widget.import_files()
    qtbot.wait(200)

    assert len(widget.series) == 1
    assert widget.series[0].ramp_source == "metadata"
    assert "metadata" in widget.status_label.text()


def test_shell_htxrd_page_is_htxrd_workspace(qtbot):
    window = PrismMainWindow()
    qtbot.addWidget(window)
    qtbot.wait(20)
    assert isinstance(window.htxrd_page, HtxrdWorkspace)
    window.nav.setCurrentRow(NAV_ITEMS.index("HT-XRD"))
    qtbot.wait(20)
    assert window.stack.currentWidget() is window.htxrd_page


def test_pick_guide_on_waterfall_and_track(qtbot, tmp_path, monkeypatch):
    """User request: click the peak at a start and an end temperature on
    the waterfall instead of typing window numbers."""
    from types import SimpleNamespace

    def center_by_temp(t):
        return 30.0 - 0.001 * t  # drifts 29.9 -> 29.7 across 100..300

    paths = _write_series(tmp_path, temps=(100, 200, 300), center_by_temp=center_by_temp, noise=2.0)
    widget = HtxrdWorkspace()
    qtbot.addWidget(widget)
    widget.template_edit.setText("scan_???.xy")
    monkeypatch.setattr("xrd.qt_htxrd.QFileDialog.getOpenFileNames", staticmethod(lambda *a, **k: (paths, "")))
    widget.import_files()
    qtbot.wait(250)

    widget.pick_guide_btn.setChecked(True)
    assert widget._pick_cid is not None
    patterns, display_y, delta = widget._wf_state
    ax = widget.plot.figure.get_axes()[0]

    # click near the peak of the bottom (100 degC) curve and the top (300 degC) one
    def click(x, slice_i):
        y = float(np.interp(x, patterns[slice_i].x, display_y[slice_i])) + slice_i * delta
        widget._on_waterfall_pick(SimpleNamespace(inaxes=ax, xdata=x, ydata=y))

    click(29.9, 0)
    click(29.7, 2)
    qtbot.wait(250)
    assert widget._track_picks == [(1, 29.9), (3, 29.7)]

    widget.track_picked_guide()
    qtbot.wait(20)
    labels = list(widget.track_results.keys())
    assert labels == ["guide 29.90°→29.70°"]
    rows = widget.track_results[labels[0]]
    assert [r.present for r in rows] == [True, True, True]
    for r, expected in zip(rows, (29.9, 29.8, 29.7)):
        assert r.center == pytest.approx(expected, abs=0.05)

    # picks clear and the guide artist disappears on next render
    widget._clear_track_picks()
    qtbot.wait(250)
    assert widget._track_picks == []


def test_auto_track_all_fills_windows_and_tracks(qtbot, tmp_path, monkeypatch):
    paths = _write_series(tmp_path, temps=(100, 200, 300), noise=2.0)
    widget = HtxrdWorkspace()
    qtbot.addWidget(widget)
    widget.template_edit.setText("scan_???.xy")
    monkeypatch.setattr("xrd.qt_htxrd.QFileDialog.getOpenFileNames", staticmethod(lambda *a, **k: (paths, "")))
    widget.import_files()
    qtbot.wait(250)

    widget.auto_track_all()
    qtbot.wait(20)
    # the generated windows are shown for editing, anchored on the found peak
    assert "@ 30" in widget.windows_edit.text() or "@ 29.9" in widget.windows_edit.text()
    assert len(widget.track_results) >= 1
    # weak curvature candidates may come along as absent rows; the REAL peak
    # must be present in (at least) one window, tracked at its true center
    present_rows = [r for rows in widget.track_results.values() for r in rows if r.present]
    assert present_rows
    assert all(r.error is None for rows in widget.track_results.values() for r in rows)
    assert any(r.center == pytest.approx(30.0, abs=0.05) for r in present_rows)
