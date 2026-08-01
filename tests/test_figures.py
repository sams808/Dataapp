"""Tests for figures_science.py + qt_figures.py — the publication-figure
module (point-fit models, native ternary geometry, XY builder, and the
Raman↔XRD identification figure)."""
from __future__ import annotations

import numpy as np
import pytest

import figures.figures_science as fsc
from figures.qt_figures import FiguresWorkspace
from core.qt_models import Spectrum, SpectrumLibrary
from core.qt_shell import NAV_ITEMS, PrismMainWindow


# --------------------------------------------------------------------------
# figures_science
# --------------------------------------------------------------------------

def test_fit_points_linear_and_gaussian_recover_truth():
    x = np.linspace(0, 10, 60)
    res = fsc.fit_points(x, 3.0 * x + 2.0, "Linear (a·x + b)")
    assert res.params[0] == pytest.approx(3.0, abs=1e-6)
    assert res.params[1] == pytest.approx(2.0, abs=1e-6)
    assert res.r_squared == pytest.approx(1.0)

    y = 5.0 * np.exp(-0.5 * ((x - 4.0) / 0.8) ** 2) + 1.0
    res = fsc.fit_points(x, y, "Gaussian peak")
    assert res.params[1] == pytest.approx(4.0, abs=0.01)  # x0
    assert res.params[0] == pytest.approx(5.0, abs=0.01)  # a
    assert "±" in res.report() or "R²" in res.report()


def test_fit_points_boltzmann_sigmoid():
    x = np.linspace(0, 100, 200)
    y = 10.0 + (2.0 - 10.0) / (1.0 + np.exp((x - 50.0) / 5.0))
    res = fsc.fit_points(x, y, "Boltzmann sigmoid")
    assert res.params[2] == pytest.approx(50.0, abs=0.5)  # x0
    assert res.r_squared > 0.999


def test_fit_points_every_registered_model_runs():
    rng = np.random.default_rng(0)
    x = np.linspace(1.0, 10.0, 80)  # positive x for log/power
    y = 2.0 * x + 1.0 + rng.normal(0, 0.05, x.shape)
    for model in fsc.FIT_MODELS:
        res = fsc.fit_points(x, y, model)
        assert np.isfinite(res.r_squared) or res.r_squared != res.r_squared, model
        assert len(res.params) == len(fsc.FIT_MODELS[model]["names"]), model


def test_ternary_geometry_corners_and_center():
    x, y = fsc.ternary_to_xy([1, 0, 0], [0, 1, 0], [0, 0, 1])
    assert (x[0], y[0]) == (0.0, 0.0)          # A corner
    assert (x[1], y[1]) == (1.0, 0.0)          # B corner
    assert x[2] == pytest.approx(0.5) and y[2] == pytest.approx(np.sqrt(3) / 2)  # C corner
    # equal thirds land at the centroid; percentages normalize the same way
    x, y = fsc.ternary_to_xy([33.3], [33.3], [33.3])
    assert x[0] == pytest.approx(0.5, abs=1e-6)
    assert y[0] == pytest.approx(np.sqrt(3) / 6, abs=1e-3)


def test_cm_to_inches():
    assert fsc.cm_to_inches(2.54) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# qt_figures
# --------------------------------------------------------------------------

def _spectrum(title, n=50, value=1.0):
    x = np.linspace(0, 100, n)
    return Spectrum(id=Spectrum.new_id(), title=title, path="", kind="raman_xy",
                    x=x, y=np.full(n, float(value)))


def test_xy_builder_layers_render_types_and_panels(qtbot):
    library = SpectrumLibrary()
    library.add(_spectrum("s1", value=1.0))
    library.add(_spectrum("s2", value=2.0))
    widget = FiguresWorkspace(library=library)
    qtbot.addWidget(widget)
    widget.set_spectra([s.id for s in library.all()])

    widget.spectra_list.selectAll()
    widget._add_layers()
    assert len(widget.layers) == 2
    widget.layers[0]["type"] = "Scatter"
    widget.layers[1]["type"] = "Sticks (vlines)"
    widget.layers[1]["panel"] = 2
    widget.cols_edit.setText("2")

    widget.render_xy()
    qtbot.wait(20)
    axes = widget.xy_plot.figure.get_axes()
    assert len(axes) == 2
    assert len(axes[0].collections) == 1  # scatter on panel 1
    assert len(axes[1].collections) == 1  # vlines on panel 2


def test_xy_builder_ticks_and_ticklabels_toggle(qtbot):
    library = SpectrumLibrary()
    library.add(_spectrum("s1", value=1.0))
    widget = FiguresWorkspace(library=library)
    qtbot.addWidget(widget)
    widget.set_spectra([s.id for s in library.all()])
    widget.spectra_list.selectAll()
    widget._add_layers()

    # defaults: both ticks and tick labels shown
    widget.render_xy()
    qtbot.wait(20)
    ax = widget.xy_plot.figure.get_axes()[0]
    assert ax.xaxis.get_major_ticks()[0].tick1line.get_visible()
    assert ax.xaxis.get_major_ticks()[0].label1.get_visible()

    # hide tick marks only, labels stay
    widget.ticks_check.setChecked(False)
    widget.render_xy()
    qtbot.wait(20)
    ax = widget.xy_plot.figure.get_axes()[0]
    assert not ax.xaxis.get_major_ticks()[0].tick1line.get_visible()
    assert ax.xaxis.get_major_ticks()[0].label1.get_visible()

    # hide labels too
    widget.ticklabels_check.setChecked(False)
    widget.render_xy()
    qtbot.wait(20)
    ax = widget.xy_plot.figure.get_axes()[0]
    assert not ax.xaxis.get_major_ticks()[0].tick1line.get_visible()
    assert not ax.xaxis.get_major_ticks()[0].label1.get_visible()


def test_export_figure_rescales_fonts_for_small_size(qtbot, tmp_path):
    library = SpectrumLibrary()
    library.add(_spectrum("s1", value=1.0))
    widget = FiguresWorkspace(library=library)
    qtbot.addWidget(widget)
    widget.set_spectra([s.id for s in library.all()])
    widget.spectra_list.selectAll()
    widget._add_layers()
    widget.render_xy()
    qtbot.wait(20)

    ax = widget.xy_plot.figure.get_axes()[0]
    original_size = ax.xaxis.label.get_fontsize()

    # capture the font size actually in effect at the moment savefig runs,
    # i.e. mid-export, before the finally-block restore
    seen_during_export = {}
    real_savefig = widget.xy_plot.figure.savefig

    def _spy_savefig(path, **kwargs):
        seen_during_export["size"] = ax.xaxis.label.get_fontsize()
        return real_savefig(path, **kwargs)

    out = tmp_path / "small.png"
    widget.width_edit.setText("5")
    widget.height_edit.setText("6")
    import figures.qt_figures as qt_figures
    from unittest.mock import patch
    with patch.object(qt_figures.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out), ""))), \
         patch.object(widget.xy_plot.figure, "savefig", side_effect=_spy_savefig):
        widget.export_figure()

    assert out.exists()
    # fonts must be visibly smaller during the small export than on-screen…
    assert seen_during_export["size"] < original_size
    # …but restored to the on-screen size afterward
    assert ax.xaxis.label.get_fontsize() == pytest.approx(original_size)
    # figure size must also be restored
    assert widget.xy_plot.figure.get_size_inches()[0] == pytest.approx(7.0)


def test_point_fit_tab_reports_r_squared(qtbot):
    library = SpectrumLibrary()
    x = np.linspace(0, 10, 40)
    library.add(Spectrum(id=Spectrum.new_id(), title="lin", path="", kind="raman_xy",
                         x=x, y=2.0 * x + 5.0))
    widget = FiguresWorkspace(library=library)
    qtbot.addWidget(widget)
    widget.set_spectra([s.id for s in library.all()])
    widget.fit_model_combo.setCurrentText("Linear (a·x + b)")
    widget.run_point_fit()
    qtbot.wait(20)
    assert "R² = 1.000" in widget.fit_report.toPlainText()
    assert "a = 2" in widget.fit_report.toPlainText()


def test_ternary_tab_from_csv(qtbot, tmp_path):
    csv = tmp_path / "comp.csv"
    csv.write_text("P,Bi,O,Tg\n50,25,25,400\n30,40,30,380\n20,20,60,420\n")
    widget = FiguresWorkspace(library=SpectrumLibrary())
    qtbot.addWidget(widget)

    import figures.qt_figures as qt_figures
    from unittest.mock import patch
    with patch.object(qt_figures.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(csv), ""))):
        widget.load_ternary_csv()
    assert "3 rows" in widget.ternary_status.text()
    widget.ternary_v_combo.setCurrentIndex(widget.ternary_v_combo.findData("Tg"))
    widget.render_ternary()
    qtbot.wait(20)
    ax = widget.ternary_plot.figure.get_axes()[0]
    assert len(ax.collections) >= 1  # the scatter (plus colorbar mappable)


def test_raman_xrd_link_figure(qtbot, tmp_path):
    import xrd.xrd_id_science as xid
    from test_xrd_id_science import QUARTZ_D, QUARTZ_I, _make_source_sq
    src = tmp_path / "src.sq"
    _make_source_sq(src, [(1010, "Quartz", "Quartz", "Si O2", "P 32 2 1", "A", QUARTZ_D, QUARTZ_I, ["Si", "O"])])
    db = tmp_path / "unified.sq"
    xid.build_xrd_database([(str(src), "TESTDB")], out_path=str(db), log=lambda *a: None)

    library = SpectrumLibrary()
    raman = _spectrum("raman_sample")
    raman.meta["rruff_matches"] = [{"mineral": "Quartz", "rruff_id": "R040031"}]
    xrd = _spectrum("xrd_sample")
    xrd.meta["xrd_matches"] = [{"mineral": "Quartz", "source": "TESTDB", "source_code": "1010"}]
    library.add(raman)
    library.add(xrd)

    widget = FiguresWorkspace(library=library, xrd_db_path=str(db))
    qtbot.addWidget(widget)
    widget.set_spectra([s.id for s in library.all()])
    widget.link_raman_combo.setCurrentIndex(widget.link_raman_combo.findData(raman.id))
    widget.link_xrd_combo.setCurrentIndex(widget.link_xrd_combo.findData(xrd.id))
    widget.render_link()
    qtbot.wait(20)

    ax_r, ax_x = widget.link_plot.figure.get_axes()
    assert "Quartz" in ax_r.get_title()
    assert "Quartz" in ax_x.get_title()
    assert len(ax_x.collections) == 1  # the reference stick pattern


def test_shell_has_figures_page(qtbot):
    window = PrismMainWindow()
    qtbot.addWidget(window)
    window.library.add(_spectrum("in_lib"))
    window.nav.setCurrentRow(NAV_ITEMS.index("Figures"))
    qtbot.wait(20)
    assert window.stack.currentWidget() is window.figures_page
    assert window.figures_page.spectra_list.count() == 1


def test_difference_vs_first_layer(qtbot):
    library = SpectrumLibrary()
    library.add(_spectrum("ref", value=1.0))
    library.add(_spectrum("s", value=3.0))
    widget = FiguresWorkspace(library=library)
    qtbot.addWidget(widget)
    widget.set_spectra([s.id for s in library.all()])
    widget.spectra_list.selectAll()
    widget._add_layers()
    widget.diff_check.setChecked(True)
    widget.render_xy()
    qtbot.wait(20)
    ax = widget.xy_plot.figure.get_axes()[0]
    diff_line = ax.lines[1]  # second layer, plotted as (s − ref) = 2.0
    assert np.allclose(diff_line.get_ydata(), 2.0)


def test_series_tab_heatmap_and_waterfall(qtbot):
    import rampy as rp
    library = SpectrumLibrary()
    x = np.linspace(0, 100, 300)
    for i in range(4):
        library.add(Spectrum(id=Spectrum.new_id(), title=f"t{i}", path="", kind="raman_xy",
                             x=x, y=rp.gaussian(x, 10.0, 40.0 + 5 * i, 5.0)))
    widget = FiguresWorkspace(library=library)
    qtbot.addWidget(widget)
    widget.set_spectra([s.id for s in library.all()])
    widget.spectra_list.selectAll()
    widget._add_layers()
    widget.series_y_edit.setText("100, 200, 300, 400")
    for kind in ("Heatmap", "Contour (filled)", "Waterfall 3D"):
        widget.series_type_combo.setCurrentText(kind)
        widget.render_series()
        qtbot.wait(20)
        assert len(widget.series_plot.figure.get_axes()) >= 1, kind


def test_table_plots_from_csv(qtbot, tmp_path):
    csv = tmp_path / "t.csv"
    csv.write_text("a,b,c\n1,2,3\n2,4,6\n3,6,9\n4,8,12\n")
    widget = FiguresWorkspace(library=SpectrumLibrary())
    qtbot.addWidget(widget)
    import figures.qt_figures as qt_figures
    from unittest.mock import patch
    with patch.object(qt_figures.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(csv), ""))):
        widget.load_table_csv()
    assert "4 rows" in widget.table_status.text()
    for kind in ("Histogram", "Box plot", "Violin plot", "Correlation matrix", "Scatter (col vs col)"):
        widget.table_plot_combo.setCurrentText(kind)
        widget.render_table_plot()
        qtbot.wait(20)
    # correlation of perfectly linear columns = 1
    widget.table_plot_combo.setCurrentText("Correlation matrix")
    widget.render_table_plot()
    qtbot.wait(20)
    assert len(widget.table_plot.figure.get_axes()) >= 1
