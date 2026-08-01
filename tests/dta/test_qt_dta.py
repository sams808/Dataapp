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
