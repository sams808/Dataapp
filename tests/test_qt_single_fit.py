"""Tests for qt_single_fit.py + qt_fit_params.py (M8) — the Qt port of
single-spectrum peak fitting.

Run separately from the default Tk-focused suite (see pytest.ini / conftest.py
for why): `pytest tests/test_qt_single_fit.py --override-ini="addopts="`
"""
from __future__ import annotations

import numpy as np
import pytest
import rampy as rp

from qt_fit_params import FitParamDialog
from core.qt_models import Spectrum, SpectrumLibrary
from core.qt_shell import NAV_ITEMS, PrismMainWindow, _load_spectrum_from_path
from qt_single_fit import SingleFitWorkspace


def _synthetic_gaussian_spectrum() -> Spectrum:
    x = np.linspace(400, 600, 400)
    y = rp.gaussian(x, 80.0, 505.0, 25.0)
    return Spectrum(id=Spectrum.new_id(), title="synthetic", path="", kind="raman_xy", x=x, y=y)


def _component(center=500.0, fwhm=30.0, amp=100.0, shape="G"):
    return {
        "shape": shape,
        "shift_val": center, "shift_min": center - 100, "shift_max": center + 100, "fit_shift": True,
        "fwhm_val": fwhm, "fwhm_min": 1.0, "fwhm_max": 200.0, "fit_fwhm": True,
        "eta_val": 0.5, "eta_min": 0.0, "eta_max": 1.0, "fit_eta": True,
        "amp_val": amp, "fit_amp": True,
    }


def test_workspace_constructs_empty(qtbot):
    widget = SingleFitWorkspace()
    qtbot.addWidget(widget)
    assert widget.spec_combo.count() == 0


def test_set_spectra_populates_combo_and_plots_data_only(qtbot):
    library = SpectrumLibrary()
    spectrum = _synthetic_gaussian_spectrum()
    library.add(spectrum)

    widget = SingleFitWorkspace(library=library)
    qtbot.addWidget(widget)
    widget.set_spectra([spectrum.id])

    assert widget.spec_combo.count() == 1
    assert widget._current_spectrum_id == spectrum.id
    assert widget.chi2_label.text() == "--"  # no params set yet -> data-only plot


def test_fit_param_memory_is_keyed_by_id_not_title(qtbot):
    """Regression guard for the identity fix this port makes: two spectra
    with the SAME title must not collide in fit_param_memory."""
    library = SpectrumLibrary()
    s1 = _synthetic_gaussian_spectrum()
    s2 = Spectrum(id=Spectrum.new_id(), title=s1.title, path="", kind="raman_xy", x=s1.x.copy(), y=s1.y.copy())
    library.add(s1)
    library.add(s2)

    widget = SingleFitWorkspace(library=library)
    qtbot.addWidget(widget)
    widget.set_spectra([s1.id, s2.id])

    widget.fit_param_memory.set(s1.id, [_component(center=111.0)])
    assert widget.fit_param_memory.get(s2.id) == []
    assert widget.fit_param_memory.get(s1.id)[0]["shift_val"] == 111.0


def test_run_fit_classic_recovers_known_peak_and_shows_residual_subplot(qtbot):
    library = SpectrumLibrary()
    spectrum = _synthetic_gaussian_spectrum()
    library.add(spectrum)

    widget = SingleFitWorkspace(library=library)
    qtbot.addWidget(widget)
    widget.set_spectra([spectrum.id])
    widget.fit_param_memory.set(spectrum.id, [_component(center=500.0, fwhm=30.0, amp=100.0)])

    widget.run_fit()

    params = widget.fit_param_memory.get(spectrum.id)
    assert params[0]["shift_val"] == pytest.approx(505.0, abs=1.0)
    assert float(widget.chi2_label.text()) < 1.0
    assert float(widget.r2_label.text()) > 0.99

    # Dashed fit line + residual subplot (item 11): two axes once a fit exists.
    axes = widget.plot.figure.get_axes()
    assert len(axes) == 2
    fit_lines = [ln for ln in axes[0].lines if ln.get_linestyle() == "--"]
    assert len(fit_lines) == 1


def test_reset_to_snapshot_restores_pre_fit_params(qtbot):
    library = SpectrumLibrary()
    spectrum = _synthetic_gaussian_spectrum()
    library.add(spectrum)

    widget = SingleFitWorkspace(library=library)
    qtbot.addWidget(widget)
    widget.set_spectra([spectrum.id])
    widget.fit_param_memory.set(spectrum.id, [_component(center=480.0, fwhm=35.0, amp=60.0)])

    widget.run_fit()
    assert widget.fit_param_memory.get(spectrum.id)[0]["shift_val"] != pytest.approx(480.0)

    widget.reset_params_to_snapshot()
    assert widget.fit_param_memory.get(spectrum.id)[0]["shift_val"] == pytest.approx(480.0)


def test_export_components_csv_writes_per_component_and_residual_files(qtbot, tmp_path, monkeypatch):
    library = SpectrumLibrary()
    spectrum = _synthetic_gaussian_spectrum()
    library.add(spectrum)

    widget = SingleFitWorkspace(library=library)
    qtbot.addWidget(widget)
    widget.set_spectra([spectrum.id])
    widget.fit_param_memory.set(spectrum.id, [_component(center=500.0, fwhm=30.0, amp=100.0)])
    widget.run_fit()

    out_base = tmp_path / "myfit.csv"
    monkeypatch.setattr(
        "qt_single_fit.QFileDialog.getSaveFileName", staticmethod(lambda *a, **k: (str(out_base), "")),
    )
    # The success-path QMessageBox.information() this triggers is neutralized
    # by conftest.py's autouse _prevent_blocking_qt_dialogs fixture (see its
    # docstring for why that fixture exists).
    widget.export_components_csv()

    assert (tmp_path / "myfit_all.csv").exists()
    assert (tmp_path / "myfit_comp1.csv").exists()
    assert (tmp_path / "myfit_residual.csv").exists()


def test_named_components_flow_into_report_csv_and_legend(qtbot, tmp_path, monkeypatch):
    """A FAIRE item 9: a component's user-given name shows up in the fit
    report, the per-component CSV filenames/headers, and the plot legend."""
    library = SpectrumLibrary()
    x = np.linspace(400, 600, 400)
    y = rp.gaussian(x, 80.0, 505.0, 25.0)
    source_path = tmp_path / "named.txt"
    source_path.write_text("dummy")
    spectrum = Spectrum(id=Spectrum.new_id(), title="named", path=str(source_path), kind="raman_xy", x=x, y=y)
    library.add(spectrum)
    widget = SingleFitWorkspace(library=library)
    qtbot.addWidget(widget)
    widget.set_spectra([spectrum.id])
    comp = _component(center=500.0, fwhm=30.0, amp=100.0)
    comp["name"] = "nu1 PO4"
    widget.fit_param_memory.set(spectrum.id, [comp])
    widget.run_fit()

    # legend carries the name
    labels = [ln.get_label() for ln in widget.plot.figure.get_axes()[0].lines]
    assert "nu1 PO4" in labels

    # report carries the name next to the component number
    widget.generate_report(quick=True)
    report = next((tmp_path / "reports").glob("named_fit*.txt")).read_text()
    assert "1 (nu1 PO4)" in report

    # CSV export: named header + sanitized filename
    out_base = tmp_path / "named.csv"
    monkeypatch.setattr(
        "qt_single_fit.QFileDialog.getSaveFileName", staticmethod(lambda *a, **k: (str(out_base), "")),
    )
    widget.export_components_csv()
    assert (tmp_path / "named_comp1_nu1_PO4.csv").exists()
    header = (tmp_path / "named_all.csv").read_text().splitlines()[0]
    assert "nu1 PO4" in header


def test_fit_param_dialog_name_column_roundtrip(qtbot):
    """The Name column reads/writes params_struct['name']; unnamed rows
    stay clean (no empty 'name' keys), so old saved models are untouched."""
    from qt_fit_params import _NAME_COL, FitParamDialog
    captured = []
    comp = _component(center=500.0)
    comp["name"] = "D band"
    dlg = FitParamDialog(None, [comp, _component(center=550.0)], captured.append)
    qtbot.addWidget(dlg)
    assert dlg.table.item(0, _NAME_COL).text() == "D band"
    dlg.table.item(1, _NAME_COL).setText("  G band  ")
    dlg._on_accept_clicked()
    assert captured[0][0]["name"] == "D band"
    assert captured[0][1]["name"] == "G band"  # stripped

    captured.clear()
    dlg2 = FitParamDialog(None, [_component(center=500.0)], captured.append)
    qtbot.addWidget(dlg2)
    dlg2._on_accept_clicked()
    assert "name" not in captured[0][0]


def test_generate_report_quick_writes_report_next_to_source(qtbot, tmp_path):
    library = SpectrumLibrary()
    x = np.linspace(400, 600, 400)
    y = rp.gaussian(x, 80.0, 505.0, 25.0)
    source_path = tmp_path / "myspec.txt"
    source_path.write_text("dummy")
    spectrum = Spectrum(id=Spectrum.new_id(), title="myspec", path=str(source_path), kind="raman_xy", x=x, y=y)
    library.add(spectrum)

    widget = SingleFitWorkspace(library=library)
    qtbot.addWidget(widget)
    widget.set_spectra([spectrum.id])
    widget.fit_param_memory.set(spectrum.id, [_component(center=500.0, fwhm=30.0, amp=100.0)])
    widget.run_fit()

    widget.generate_report(quick=True)

    reports_dir = tmp_path / "reports"
    assert reports_dir.is_dir()
    files = list(reports_dir.glob("myspec_fit*.txt"))
    assert len(files) == 1
    content = files[0].read_text()
    assert "R2:" in content
    assert "Centroid" in content


def test_fit_param_dialog_round_trips_new_shapes_and_link(qtbot):
    """The extended schema (V/EMG shapes, skew bounds, link_fwhm) must
    survive the dialog's table rebuild/sync cycle unchanged."""
    params = [
        {"shape": "V", "shift_min": 400.0, "shift_val": 500.0, "shift_max": 600.0, "fit_shift": True,
         "fwhm_min": 1.0, "fwhm_val": 20.0, "fwhm_max": 100.0, "fit_fwhm": True,
         "eta_min": 0.0, "eta_val": 0.3, "eta_max": 1.0, "fit_eta": True,
         "amp_val": 50.0, "fit_amp": True},
        {"shape": "EMG", "shift_min": 700.0, "shift_val": 800.0, "shift_max": 900.0, "fit_shift": True,
         "fwhm_min": 1.0, "fwhm_val": 15.0, "fwhm_max": 100.0, "fit_fwhm": True,
         "skew_min": -50.0, "skew_val": 12.0, "skew_max": 50.0, "fit_skew": True,
         "amp_val": 30.0, "fit_amp": True, "link_fwhm": 0},
    ]
    accepted = {}
    dlg = FitParamDialog(None, params_struct=params, on_accept=lambda p: accepted.update(result=p))
    qtbot.addWidget(dlg)
    dlg._on_accept_clicked()

    out = accepted["result"]
    assert out[0]["shape"] == "V"
    assert out[0]["eta_val"] == 0.3
    assert out[1]["shape"] == "EMG"
    assert out[1]["skew_val"] == 12.0
    assert out[1]["link_fwhm"] == 0


def test_fit_param_dialog_auto_find_peaks_seeds_component(qtbot):
    x = np.linspace(0, 1000, 2000)
    y = rp.gaussian(x, 100.0, 300.0, 15.0)
    accepted = {}
    dlg = FitParamDialog(None, params_struct=None, on_accept=lambda p: accepted.update(result=p), x=x, y=y)
    qtbot.addWidget(dlg)

    dlg._auto_find_peaks()
    assert len(dlg.rows) >= 1
    assert any(abs(row["shift_val"] - 300.0) < 15 for row in dlg.rows)


def test_shell_fitting_page_picks_up_library_records(qtbot, raman_example_path):
    window = PrismMainWindow()
    qtbot.addWidget(window)

    spectrum = _load_spectrum_from_path(str(raman_example_path))
    window.library.add(spectrum)

    window.nav.setCurrentRow(NAV_ITEMS.index("Peak Fitting"))
    qtbot.wait(20)

    assert window.fitting_page.spec_combo.count() == 1


def test_pick_peaks_on_plot_adds_components(qtbot):
    """User request: manual peak picking — clicking the plot in pick mode
    adds a component at the clicked x with amplitude read from the data."""
    from types import SimpleNamespace

    library = SpectrumLibrary()
    spectrum = _synthetic_gaussian_spectrum()  # apex at x=505, height 80
    library.add(spectrum)
    widget = SingleFitWorkspace(library=library)
    qtbot.addWidget(widget)
    widget.set_spectra([spectrum.id])
    widget.norm_check.setChecked(False)  # picked amp is read in the fit's own scale
    qtbot.wait(20)

    widget.pick_peaks_btn.setChecked(True)
    assert widget._pick_cid is not None

    # Click slightly off-apex — amplitude must still come from the data max
    # near the click, not the raw click y.
    event = SimpleNamespace(inaxes=widget.plot.figure.gca(), xdata=503.0, ydata=12.3)
    widget._on_pick_click(event)
    qtbot.wait(20)

    params = widget.fit_param_memory.get(spectrum.id)
    assert len(params) == 1
    assert params[0]["shift_val"] == pytest.approx(503.0)
    assert params[0]["amp_val"] == pytest.approx(80.0, rel=0.05)
    assert params[0]["shift_min"] < 503.0 < params[0]["shift_max"]

    # A second click appends rather than replaces.
    widget._on_pick_click(SimpleNamespace(inaxes=widget.plot.figure.gca(), xdata=450.0, ydata=5.0))
    qtbot.wait(20)
    assert len(widget.fit_param_memory.get(spectrum.id)) == 2

    # Toggling off disconnects: further clicks add nothing.
    widget.pick_peaks_btn.setChecked(False)
    assert widget._pick_cid is None


def test_pick_peaks_ignores_clicks_while_toolbar_zooming(qtbot):
    from types import SimpleNamespace

    library = SpectrumLibrary()
    spectrum = _synthetic_gaussian_spectrum()
    library.add(spectrum)
    widget = SingleFitWorkspace(library=library)
    qtbot.addWidget(widget)
    widget.set_spectra([spectrum.id])
    qtbot.wait(20)

    widget.pick_peaks_btn.setChecked(True)
    widget.plot.toolbar.zoom()  # activate the zoom tool
    try:
        widget._on_pick_click(SimpleNamespace(inaxes=widget.plot.figure.gca(), xdata=505.0, ydata=80.0))
    finally:
        widget.plot.toolbar.zoom()  # deactivate again
    assert widget.fit_param_memory.get(spectrum.id) == []


def test_origin_stepwise_button_takes_visible_incremental_steps(qtbot):
    """User feedback: '1 iteration' used to jump almost straight to the
    converged answer. Each click must now be ONE LM update — progress, but
    multiple clicks still needed to converge."""
    library = SpectrumLibrary()
    spectrum = _synthetic_gaussian_spectrum()  # truth: center 505
    library.add(spectrum)
    widget = SingleFitWorkspace(library=library)
    qtbot.addWidget(widget)
    widget.set_spectra([spectrum.id])
    widget.norm_check.setChecked(False)
    widget.mode_origin.setChecked(True)
    widget.fit_param_memory.set(spectrum.id, [_component(center=480.0, fwhm=40.0, amp=60.0)])

    widget.run_fit_origin_stepwise(1)
    qtbot.wait(20)
    after_one = widget.fit_param_memory.get(spectrum.id)[0]["shift_val"]
    assert after_one != 480.0  # it moved…
    assert abs(after_one - 505.0) > 0.5  # …but did not converge in one click

    for _ in range(30):
        widget.run_fit_origin_stepwise(1)
    qtbot.wait(20)
    after_many = widget.fit_param_memory.get(spectrum.id)[0]["shift_val"]
    assert after_many == pytest.approx(505.0, abs=0.5)  # repeated clicks converge


def test_pick_peaks_toggle_deactivates_active_zoom_tool(qtbot):
    """User report: with the zoom tool left active, pick-mode clicks were
    silently swallowed (drag still zoomed). Toggling pick mode ON must
    deactivate the zoom/pan tool so clicks land."""
    from types import SimpleNamespace

    library = SpectrumLibrary()
    spectrum = _synthetic_gaussian_spectrum()
    library.add(spectrum)
    widget = SingleFitWorkspace(library=library)
    qtbot.addWidget(widget)
    widget.set_spectra([spectrum.id])
    qtbot.wait(20)

    widget.plot.toolbar.zoom()  # user left the zoom tool on
    assert widget.plot.toolbar.mode
    widget.pick_peaks_btn.setChecked(True)
    assert not widget.plot.toolbar.mode  # pick mode turned it off

    widget._on_pick_click(SimpleNamespace(inaxes=widget.plot.figure.gca(), xdata=505.0, ydata=1.0))
    qtbot.wait(20)
    assert len(widget.fit_param_memory.get(spectrum.id)) == 1
