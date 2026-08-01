"""Tests for qt_baseline.py — the baseline-subtraction workspace."""
from __future__ import annotations

import numpy as np
import pytest
import rampy as rp

from processing.qt_baseline import BaselineWorkspace
from core.qt_models import Spectrum, SpectrumLibrary
from core.qt_shell import NAV_ITEMS, PrismMainWindow


def _library_with_baselined_spectrum():
    library = SpectrumLibrary()
    x = np.linspace(100, 1000, 600)
    y = 0.02 * x + 5.0 + rp.gaussian(x, 100.0, 550.0, 15.0)
    library.add(Spectrum(id=Spectrum.new_id(), title="withbase", path="", kind="raman_xy", x=x, y=y))
    return library


def test_workspace_constructs_empty(qtbot):
    widget = BaselineWorkspace()
    qtbot.addWidget(widget)
    assert widget.file_list.count() == 0


def test_preview_poly_draws_baseline_and_subtracted(qtbot):
    library = _library_with_baselined_spectrum()
    widget = BaselineWorkspace(library=library)
    qtbot.addWidget(widget)
    widget.set_spectra([s.id for s in library.all()])
    widget.file_list.item(0).setSelected(True)

    widget.method_combo.setCurrentText("poly")
    widget.param_edits[0].setText("1")
    widget.roi_edit.setText("100-450; 700-1000")
    widget.preview()
    qtbot.wait(20)

    axes = widget.plot.figure.get_axes()
    assert len(axes) == 2  # raw+baseline on top, subtracted below
    assert widget._last_preview is not None


def test_apply_creates_bl_spectrum_with_flat_background(qtbot):
    library = _library_with_baselined_spectrum()
    widget = BaselineWorkspace(library=library)
    qtbot.addWidget(widget)
    widget.set_spectra([s.id for s in library.all()])
    widget.file_list.item(0).setSelected(True)

    widget.method_combo.setCurrentText("poly")
    widget.param_edits[0].setText("1")
    widget.roi_edit.setText("100-450; 700-1000")
    widget.apply_selected()
    qtbot.wait(20)

    bl = [s for s in library.all() if s.title == "withbase_bl"]
    assert len(bl) == 1
    assert bl[0].status == "derived"
    # Background regions now sit near zero; the peak survives.
    x, y = bl[0].x, bl[0].y
    bg = y[(x < 400) | (x > 750)]
    assert np.abs(bg).max() < 2.0
    assert y.max() == pytest.approx(100.0, rel=0.05)


def test_apply_without_roi_for_poly_reports_error(qtbot):
    library = _library_with_baselined_spectrum()
    widget = BaselineWorkspace(library=library)
    qtbot.addWidget(widget)
    widget.set_spectra([s.id for s in library.all()])
    widget.file_list.item(0).setSelected(True)
    widget.method_combo.setCurrentText("poly")
    widget.roi_edit.setText("")
    widget.apply_selected()  # error dialog neutralized by conftest fixture
    assert not any(s.title.endswith("_bl") for s in library.all())


def test_per_spectrum_settings_persist_across_selection(qtbot):
    library = _library_with_baselined_spectrum()
    x = np.linspace(0, 100, 60)
    library.add(Spectrum(id=Spectrum.new_id(), title="other", path="", kind="raman_xy", x=x, y=np.ones(60)))
    widget = BaselineWorkspace(library=library)
    qtbot.addWidget(widget)
    widget.set_spectra([s.id for s in library.all()])

    widget.file_list.item(0).setSelected(True)
    widget.roi_edit.setText("100-450")
    widget.method_combo.setCurrentText("poly")

    # Switch to the second spectrum, change everything…
    widget.file_list.item(0).setSelected(False)
    widget.file_list.item(1).setSelected(True)
    widget.roi_edit.setText("0-10")
    widget.method_combo.setCurrentText("arPLS")

    # …and back: the first spectrum's settings come back.
    widget.file_list.item(1).setSelected(False)
    widget.file_list.item(0).setSelected(True)
    assert widget.roi_edit.text() == "100-450"
    assert widget.method_combo.currentText() == "poly"


def test_pick_roi_by_dragging_appends_segments(qtbot):
    library = _library_with_baselined_spectrum()
    widget = BaselineWorkspace(library=library)
    qtbot.addWidget(widget)
    widget.set_spectra([s.id for s in library.all()])
    widget.file_list.item(0).setSelected(True)

    widget.pick_roi_btn.setChecked(True)  # draws the raw spectrum + arms the selector
    qtbot.wait(20)
    assert widget._span_selector is not None

    widget._on_span_selected(100.0, 450.0)
    widget._on_span_selected(700.0, 1000.0)
    assert widget.roi_edit.text() == "100-450; 700-1000"

    # Un-toggling detaches the selector.
    widget.pick_roi_btn.setChecked(False)
    assert widget._span_selector is None

    # And the picked regions are directly usable by a poly preview.
    widget.method_combo.setCurrentText("poly")
    widget.param_edits[0].setText("1")
    widget.preview()
    qtbot.wait(20)
    assert widget._last_preview is not None


def test_shell_baseline_page_picks_up_library_records(qtbot, raman_example_path):
    from core.qt_shell import _load_spectrum_from_path

    window = PrismMainWindow()
    qtbot.addWidget(window)
    window.library.add(_load_spectrum_from_path(str(raman_example_path)))

    window.nav.setCurrentRow(NAV_ITEMS.index("Baseline"))
    qtbot.wait(20)
    assert window.baseline_page.file_list.count() == 1
