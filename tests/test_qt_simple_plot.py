"""Tests for qt_simple_plot.py (M7) — the Qt port of Simple Plot + CIF
Bragg overlay.

Run separately from the default Tk-focused suite (see pytest.ini / conftest.py
for why): `pytest tests/test_qt_simple_plot.py --override-ini="addopts="`
"""
from __future__ import annotations

import numpy as np
import pytest

from core.qt_models import Spectrum, SpectrumLibrary
from core.qt_shell import NAV_ITEMS, PrismMainWindow, _load_spectrum_from_path
from raman.qt_raman import RamanWorkspace

_CUBIC_CIF = """\
data_test_cubic
_cell_length_a 5.6400
_cell_length_b 5.6400
_cell_length_c 5.6400
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_diffrn_radiation_wavelength 1.5406
"""


@pytest.fixture
def cif_path(tmp_path):
    p = tmp_path / "cubic_test.cif"
    p.write_text(_CUBIC_CIF, encoding="utf-8")
    return p


def _library_with_raman(raman_example_path) -> SpectrumLibrary:
    library = SpectrumLibrary()
    library.add(_load_spectrum_from_path(str(raman_example_path)))
    return library


def test_workspace_constructs_empty(qtbot):
    widget = RamanWorkspace()
    qtbot.addWidget(widget)
    assert widget.file_list.count() == 0


def test_set_spectra_populates_list_and_selection_renders(qtbot, raman_example_path):
    library = _library_with_raman(raman_example_path)
    widget = RamanWorkspace(library=library)
    qtbot.addWidget(widget)

    widget.set_spectra([s.id for s in library.all()])
    assert widget.file_list.count() == 1

    widget.file_list.selectAll()
    qtbot.wait(200)  # let the debounced redraw fire
    assert len(widget.plot.figure.get_axes()) >= 1


def test_stacked_mode_with_two_spectra_offsets_lines(qtbot, raman_example_path):
    library = SpectrumLibrary()
    s1 = _load_spectrum_from_path(str(raman_example_path))
    s2 = Spectrum(id=Spectrum.new_id(), title="copy", path=s1.path, kind=s1.kind,
                  x=s1.x.copy(), y=s1.y.copy(), df=s1.df, meta=dict(s1.meta))
    library.add(s1)
    library.add(s2)

    widget = RamanWorkspace(library=library)
    qtbot.addWidget(widget)
    widget.set_spectra([s1.id, s2.id])
    widget.file_list.selectAll()
    widget.mode_stacked.setChecked(True)
    qtbot.wait(200)

    axes = widget.plot.figure.get_axes()
    assert len(axes) == 1
    assert len(axes[0].lines) == 2


def test_cif_import_adds_series_and_draws_markers(qtbot, raman_example_path, cif_path):
    library = _library_with_raman(raman_example_path)
    widget = RamanWorkspace(library=library)
    qtbot.addWidget(widget)
    widget.set_spectra([s.id for s in library.all()])
    widget.file_list.selectAll()
    qtbot.wait(200)

    peaks_before = len(widget.cif_series)
    from raman.cif_tools import bragg_peaks_from_cif_generic
    peaks = bragg_peaks_from_cif_generic(str(cif_path), two_theta_max=80.0, hkl_max=6)
    assert len(peaks) > 0

    widget.cif_series.append({
        "path": str(cif_path), "label": "cubic_test.cif", "plot_label": "",
        "peaks": peaks, "visible": True, "color": "crimson", "pad": 0.03,
    })
    widget.render()
    qtbot.wait(20)
    assert len(widget.cif_series) == peaks_before + 1
    ax = widget.plot.figure.get_axes()[0]
    # Bragg markers are drawn as vlines collections distinct from the data line.
    assert len(ax.collections) > 0


def test_cif_manager_toggle_visible_triggers_debounced_redraw(qtbot, raman_example_path, cif_path):
    library = _library_with_raman(raman_example_path)
    widget = RamanWorkspace(library=library)
    qtbot.addWidget(widget)
    widget.set_spectra([s.id for s in library.all()])
    widget.file_list.selectAll()

    from raman.cif_tools import bragg_peaks_from_cif_generic
    peaks = bragg_peaks_from_cif_generic(str(cif_path), two_theta_max=80.0, hkl_max=6)
    serie = {"path": str(cif_path), "label": "cubic_test.cif", "plot_label": "",
             "peaks": peaks, "visible": False, "color": "crimson", "pad": 0.03}
    widget.cif_series.append(serie)
    widget.render()
    qtbot.wait(20)
    ax = widget.plot.figure.get_axes()[0]
    assert len(ax.collections) == 0  # not visible yet

    widget.set_cif_field(serie, "visible", True)
    qtbot.wait(200)  # debounce interval is 120ms
    ax = widget.plot.figure.get_axes()[0]
    assert len(ax.collections) > 0


def test_color_scheme_change_does_not_crash(qtbot, raman_example_path):
    library = _library_with_raman(raman_example_path)
    widget = RamanWorkspace(library=library)
    qtbot.addWidget(widget)
    widget.set_spectra([s.id for s in library.all()])
    widget.file_list.selectAll()
    qtbot.wait(200)

    for scheme in ["Matplotlib cycle", "Distinct", "Hash by name", "Monochrome"]:
        widget.color_combo.setCurrentText(scheme)
        qtbot.wait(200)
    assert len(widget.plot.figure.get_axes()) >= 1


def test_difference_mode_plots_a_minus_b(qtbot):
    library = SpectrumLibrary()
    x = np.linspace(0, 100, 200)
    a = Spectrum(id=Spectrum.new_id(), title="a", path="", kind="raman_xy", x=x, y=np.full(200, 5.0))
    b = Spectrum(id=Spectrum.new_id(), title="b", path="", kind="raman_xy", x=x, y=np.full(200, 2.0))
    library.add(a)
    library.add(b)

    widget = RamanWorkspace(library=library)
    qtbot.addWidget(widget)
    widget.set_spectra([a.id, b.id])
    widget.file_list.selectAll()
    widget.diff_check.setChecked(True)
    qtbot.wait(200)  # debounced redraw

    axes = widget.plot.figure.get_axes()
    assert len(axes) == 1
    lines = axes[0].lines
    # a, b, and the difference curve (plus the y=0 axhline)
    labels = [ln.get_label() for ln in lines]
    assert any("−" in lbl for lbl in labels)
    diff_line = [ln for ln in lines if "−" in ln.get_label()][0]
    assert np.allclose(diff_line.get_ydata(), 3.0)


def test_difference_mode_with_wrong_selection_count_falls_back(qtbot, raman_example_path):
    library = _library_with_raman(raman_example_path)
    widget = RamanWorkspace(library=library)
    qtbot.addWidget(widget)
    widget.set_spectra([s.id for s in library.all()])
    widget.file_list.selectAll()  # only ONE spectrum
    widget.diff_check.setChecked(True)
    qtbot.wait(200)
    # Falls back to normal rendering rather than erroring or blanking.
    assert len(widget.plot.figure.get_axes()) == 1
    assert len(widget.plot.figure.get_axes()[0].lines) >= 1


def test_click_to_annotate_adds_and_clears_annotations(qtbot, raman_example_path):
    library = _library_with_raman(raman_example_path)
    widget = RamanWorkspace(library=library)
    qtbot.addWidget(widget)
    widget.set_spectra([s.id for s in library.all()])
    widget.file_list.selectAll()
    qtbot.wait(200)

    widget.annotate_check.setChecked(True)

    class _FakeClick:
        inaxes = widget.plot.figure.get_axes()[0]
        xdata = 981.4
        ydata = 100.0

    widget._on_plot_click(_FakeClick())
    qtbot.wait(200)  # debounced re-render
    assert len(widget.annotations) == 1

    ax = widget.plot.figure.get_axes()[0]
    assert any("981.4" in t.get_text() for t in ax.texts)
    # …and a dot marker right at the clicked point (A FAIRE item 16)
    assert any(len(ln.get_xdata()) == 1 and ln.get_marker() == "o" for ln in ax.lines)

    # Annotations persist across an unrelated re-render (the whole point of
    # storing them as data, not artists).
    widget.render()
    ax = widget.plot.figure.get_axes()[0]
    assert any("981.4" in t.get_text() for t in ax.texts)

    widget._clear_annotations()
    qtbot.wait(200)
    assert widget.annotations == []
    ax = widget.plot.figure.get_axes()[0]
    assert not any("981.4" in t.get_text() for t in ax.texts)


def test_annotate_click_ignored_when_toggle_off(qtbot, raman_example_path):
    library = _library_with_raman(raman_example_path)
    widget = RamanWorkspace(library=library)
    qtbot.addWidget(widget)
    widget.set_spectra([s.id for s in library.all()])
    widget.file_list.selectAll()
    qtbot.wait(200)

    class _FakeClick:
        inaxes = widget.plot.figure.get_axes()[0]
        xdata = 500.0
        ydata = 1.0

    widget._on_plot_click(_FakeClick())  # annotate_check is unchecked
    assert widget.annotations == []


def test_rapid_axis_typing_coalesces_into_one_render(qtbot, raman_example_path, cif_path, monkeypatch):
    """Regression guard for THE historical performance bug (multi-second
    per-keystroke redraws with several CIF overlays loaded) — its original
    Tk-era test was deleted with the Tk app, restored here for the Qt
    architecture: a burst of axis-title keystrokes must produce ONE
    debounced render, not one per keystroke."""
    library = _library_with_raman(raman_example_path)
    widget = RamanWorkspace(library=library)
    qtbot.addWidget(widget)
    widget.set_spectra([s.id for s in library.all()])
    widget.file_list.selectAll()

    from raman.cif_tools import bragg_peaks_from_cif_generic
    peaks = bragg_peaks_from_cif_generic(str(cif_path), two_theta_max=80.0, hkl_max=6)
    for i in range(5):  # several CIF overlays, like the original report
        widget.cif_series.append({
            "path": f"{cif_path}#{i}", "label": f"cif{i}", "plot_label": "",
            "peaks": peaks, "visible": True, "color": "crimson", "pad": 0.03,
        })
    qtbot.wait(250)  # flush the selection-triggered render

    calls = {"n": 0}
    real_render = widget.render

    def counting_render(*a, **k):
        calls["n"] += 1
        return real_render(*a, **k)

    monkeypatch.setattr(widget, "render", counting_render)

    # Simulate typing a 12-character axis title as 12 rapid setText events.
    for i in range(12):
        widget.x_title_edit.setText("Raman shift"[: i + 1])
    qtbot.wait(400)  # > debounce interval; let the single coalesced render run

    assert calls["n"] <= 2, f"expected coalesced redraws, got {calls['n']} renders for 12 keystrokes"


def test_plot_widget_mouse_readout_label_exists(qtbot):
    from core.qt_widgets import PlotWidget
    widget = PlotWidget()
    qtbot.addWidget(widget)
    assert widget.coords_label.text() == ""

    class _FakeEvent:
        inaxes = widget.ax
        xdata = 123.456
        ydata = 7.89

    widget._on_mouse_move(_FakeEvent())
    assert "123.5" in widget.coords_label.text()

    class _OutsideEvent:
        inaxes = None
        xdata = None
        ydata = None

    widget._on_mouse_move(_OutsideEvent())
    assert widget.coords_label.text() == ""


def test_shell_raman_page_picks_up_library_records(qtbot, raman_example_path):
    window = PrismMainWindow()
    qtbot.addWidget(window)

    spectrum = _load_spectrum_from_path(str(raman_example_path))
    window.library.add(spectrum)

    window.nav.setCurrentRow(NAV_ITEMS.index("Raman"))
    qtbot.wait(20)

    assert window.raman_page.file_list.count() == 1
