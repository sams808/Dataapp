"""Tests for qt_dta.py (M6) — the Qt port of the DTA/Tg tool.

Run separately from the default Tk-focused suite (see pytest.ini / conftest.py
for why): `pytest tests/test_qt_dta.py --override-ini="addopts="`
"""
from __future__ import annotations

import pytest

from dta.qt_dta import DtaWorkspace
from core.qt_shell import PrismMainWindow, _load_spectrum_from_path


def _dta_record(dta_example_path):
    spectrum = _load_spectrum_from_path(str(dta_example_path))
    return {"title": spectrum.title, "path": spectrum.path, "df": spectrum.df, "meta": spectrum.meta}


def test_dta_workspace_constructs_standalone(qtbot):
    widget = DtaWorkspace()
    qtbot.addWidget(widget)
    assert widget.record_combo.count() == 0


def test_dta_workspace_loads_record_and_picks_default_columns(qtbot, dta_example_path):
    widget = DtaWorkspace(records=[_dta_record(dta_example_path)])
    qtbot.addWidget(widget)
    assert widget.df is not None
    assert len(widget.df) > 100
    assert "temp" in widget.x_combo.currentText().lower()
    assert widget.y_combo.currentText() != ""


def test_dta_workspace_compute_matches_known_good_values(qtbot, dta_example_path):
    """Pin the exact values the original Tk TgGuiApp produced for this file
    (verified directly against ui_dta_processing.py before this port existed):
    Double=354.4666531002641, Parallel=354.51135373692597, |dY|max=357.6214.
    Both implementations call the same tested dta_science.py functions, so an
    exact match here is the right bar, not just "physically plausible."
    """
    widget = DtaWorkspace(records=[_dta_record(dta_example_path)])
    qtbot.addWidget(widget)
    widget._compute()

    assert widget.res_double.tg == pytest.approx(354.4666531002641, abs=1e-4)
    assert widget.res_parallel.tg == pytest.approx(354.51135373692597, abs=1e-4)
    assert widget.tg_deriv == pytest.approx(357.6214, abs=1e-3)


def test_compute_shows_method_agreement_line(qtbot, dta_example_path):
    """The three Tg methods land within ~3.2 units of each other on the
    bundled example, so the agreement line must say the methods agree."""
    widget = DtaWorkspace(records=[_dta_record(dta_example_path)])
    qtbot.addWidget(widget)
    widget._compute()
    text = widget.result_label.text()
    assert "Methods agree" in text
    assert "spread" in text


def test_manual_toggle_off_ignores_typed_ranges(qtbot, dta_example_path):
    """Regression guard for the M2 bug fix: leftover typed manual ranges must
    not affect the result when Manual is unchecked."""
    widget = DtaWorkspace(records=[_dta_record(dta_example_path)])
    qtbot.addWidget(widget)

    widget.manual_compute_check.setChecked(False)
    widget.low_min_edit.setText("340")
    widget.low_max_edit.setText("348")
    widget._compute()
    tg_auto = widget.res_double.tg

    # Typed values are still sitting in the fields, Manual is still off.
    widget._compute()
    assert widget.res_double.tg == pytest.approx(tg_auto)
    assert widget.res_double.tg == pytest.approx(354.4666531002641, abs=1e-4)


def test_point_plus_point_reports_range_mode_not_point(qtbot, dta_example_path):
    """Regression guard for the M2 bug fix: when both LOW and HIGH are set to
    'use point' (under-defined), the parallel-tangent method falls back to
    AUTO ranges internally — the result must report that as 'range' mode,
    not misleadingly echo back 'point' mode with the typed point values."""
    widget = DtaWorkspace(records=[_dta_record(dta_example_path)])
    qtbot.addWidget(widget)

    widget.manual_compute_check.setChecked(True)
    widget.low_use_point_check.setChecked(True)
    widget.high_use_point_check.setChecked(True)
    widget.low_point_edit.setText("355")
    widget.high_point_edit.setText("370")
    widget._compute()

    assert widget.res_parallel is not None
    assert widget.res_parallel.low_mode == "range"
    assert widget.res_parallel.high_mode == "range"
    assert "point x=" not in widget.result_label.text()


def test_calc_integrate_and_find_max(qtbot, dta_example_path):
    widget = DtaWorkspace(records=[_dta_record(dta_example_path)])
    qtbot.addWidget(widget)

    widget.calc_xmin_edit.setText("300")
    widget.calc_xmax_edit.setText("400")
    widget._calc_integrate()
    assert "Integrate" in widget.calc_result_label.text()

    widget._calc_find_max()
    assert "Max" in widget.calc_result_label.text()


def test_temp_unit_switch_defaults_to_celsius_and_enables_for_temperature_x(qtbot, dta_example_path):
    widget = DtaWorkspace(records=[_dta_record(dta_example_path)])
    qtbot.addWidget(widget)
    assert widget.temp_c_btn.isChecked()
    assert not widget.temp_k_btn.isChecked()
    # The bundled example's default X is a recognized temperature column
    # (asserted by an existing test above), so the switch should be live.
    assert widget.temp_c_btn.isEnabled()
    assert widget.temp_k_btn.isEnabled()


def test_temp_unit_switch_disables_for_non_temperature_x(qtbot, dta_example_path):
    widget = DtaWorkspace(records=[_dta_record(dta_example_path)])
    qtbot.addWidget(widget)
    temperature_col = widget.x_combo.currentText()  # the default X, a real temperature column

    widget.x_combo.setCurrentText("Time (min)")  # plain ASCII, present verbatim in every DTA export
    qtbot.wait(20)
    assert not widget.temp_c_btn.isEnabled()
    assert not widget.temp_k_btn.isEnabled()

    widget.x_combo.setCurrentText(temperature_col)
    qtbot.wait(20)
    assert widget.temp_c_btn.isEnabled()
    assert widget.temp_k_btn.isEnabled()


def test_temp_unit_switch_converts_axis_window_and_results_to_kelvin(qtbot, dta_example_path):
    """Switching to K must shift the plotted axis data, the Tg window
    fields, and (since a result already exists) the recomputed Tg values
    all by the same +273.15 -- not just relabel what's already there."""
    widget = DtaWorkspace(records=[_dta_record(dta_example_path)])
    qtbot.addWidget(widget)
    widget._compute()

    td_c = widget.res_double.tg
    tp_c = widget.res_parallel.tg
    tx_c = widget.tg_deriv
    xmin_c = float(widget.xmin_edit.text())
    xmax_c = float(widget.xmax_edit.text())
    x_c = widget._x.copy()

    widget.temp_k_btn.setChecked(True)
    qtbot.wait(20)

    assert widget._temp_unit() == "K"
    assert float(widget.xmin_edit.text()) == pytest.approx(xmin_c + 273.15, abs=1e-6)
    assert float(widget.xmax_edit.text()) == pytest.approx(xmax_c + 273.15, abs=1e-6)
    # Recomputed automatically because a result already existed.
    assert widget.res_double.tg == pytest.approx(td_c + 273.15, abs=1e-4)
    assert widget.res_parallel.tg == pytest.approx(tp_c + 273.15, abs=1e-4)
    assert widget.tg_deriv == pytest.approx(tx_c + 273.15, abs=1e-3)
    assert widget._x == pytest.approx(x_c + 273.15, abs=1e-6)
    assert "K" in widget.result_label.text()
    assert widget.plot.ax.get_xlabel() == "Temperature (K)"


def test_temp_unit_switch_round_trip_returns_exact_original_values(qtbot, dta_example_path):
    """Switching to K and back to °C must reproduce the pinned known-good
    values exactly (within float tolerance) -- a round trip through the
    conversion must not drift."""
    widget = DtaWorkspace(records=[_dta_record(dta_example_path)])
    qtbot.addWidget(widget)
    widget._compute()

    widget.temp_k_btn.setChecked(True)
    qtbot.wait(20)
    widget.temp_c_btn.setChecked(True)
    qtbot.wait(20)

    assert widget._temp_unit() == "°C"
    assert widget.res_double.tg == pytest.approx(354.4666531002641, abs=1e-4)
    assert widget.res_parallel.tg == pytest.approx(354.51135373692597, abs=1e-4)
    assert widget.tg_deriv == pytest.approx(357.6214, abs=1e-3)


def test_temp_unit_switch_leaves_non_temperature_calc_range_untouched(qtbot, dta_example_path):
    """A Calculs range set against a non-temperature X (here, using the
    derivative d/dTime) must not get shifted by a °C/K toggle -- only the
    fields that actually window a recognized temperature axis should move."""
    widget = DtaWorkspace(records=[_dta_record(dta_example_path)])
    qtbot.addWidget(widget)

    widget.calc_use_deriv_check.setChecked(True)
    widget.calc_deriv_x_combo.setCurrentText("Time (min)")
    widget.calc_xmin_edit.setText("10")
    widget.calc_xmax_edit.setText("20")

    widget.temp_k_btn.setChecked(True)
    qtbot.wait(20)

    assert float(widget.calc_xmin_edit.text()) == pytest.approx(10.0)
    assert float(widget.calc_xmax_edit.text()) == pytest.approx(20.0)
    # The main Tg window (a real temperature axis) still converts normally.
    assert float(widget.xmin_edit.text()) == pytest.approx(623.15, abs=1e-2)


def test_shell_dta_page_picks_up_library_records(qtbot, dta_example_path):
    window = PrismMainWindow()
    qtbot.addWidget(window)

    spectrum = _load_spectrum_from_path(str(dta_example_path))
    window.library.add(spectrum)

    from core.qt_shell import NAV_ITEMS
    window.nav.setCurrentRow(NAV_ITEMS.index("DTA / Thermal"))
    qtbot.wait(20)

    assert window.dta_page.record_combo.count() == 1
    assert window.dta_page.df is not None
