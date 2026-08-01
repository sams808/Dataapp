"""Tests for qt_calc.py — the Calculations workspace (registry-driven UI
over calc_science)."""
from __future__ import annotations

import numpy as np

from qt_calc import CalcWorkspace
from core.qt_models import Spectrum, SpectrumLibrary
from core.qt_shell import NAV_ITEMS, PrismMainWindow


def _spectrum(title, value=1.0, n=101):
    x = np.linspace(0, 100, n)
    return Spectrum(id=Spectrum.new_id(), title=title, path="", kind="raman_xy",
                    x=x, y=np.full(n, float(value)))


def _workspace(qtbot, values=(6.0, 2.0)):
    library = SpectrumLibrary()
    for i, v in enumerate(values):
        library.add(_spectrum(f"s{i + 1}", v))
    widget = CalcWorkspace(library=library)
    qtbot.addWidget(widget)
    widget.set_spectra([s.id for s in library.all()])
    return widget, library


def test_subtract_two_spectra_adds_derived_to_library(qtbot):
    widget, library = _workspace(qtbot, values=(6.0, 2.0))
    widget.file_list.selectAll()
    widget.op_combo.setCurrentText("Subtract (A − rest)")
    widget.apply_selected()
    qtbot.wait(20)

    derived = [s for s in library.all() if s.status == "derived"]
    assert len(derived) == 1
    assert derived[0].title == "s1_sub"
    assert np.allclose(derived[0].y, 4.0)
    assert derived[0].meta["derived"].startswith("calc:")
    assert derived[0].meta["sources"] == ["s1", "s2"]


def test_batch_transform_applies_to_each_selected(qtbot):
    widget, library = _workspace(qtbot, values=(4.0, 9.0))
    widget.file_list.selectAll()
    widget.op_combo.setCurrentText("√y")
    widget.apply_selected()
    qtbot.wait(20)

    derived = [s for s in library.all() if s.status == "derived"]
    assert len(derived) == 2
    assert np.allclose(derived[0].y, 2.0)
    assert np.allclose(derived[1].y, 3.0)


def test_params_rebuild_with_operation(qtbot):
    widget, _ = _workspace(qtbot)
    widget.op_combo.setCurrentText("Scale / offset (a·y + b)")
    assert set(widget.param_edits.keys()) == {"factor", "offset"}
    widget.op_combo.setCurrentText("Normalize to max")
    assert widget.param_edits == {}


def test_scale_offset_uses_params(qtbot):
    widget, library = _workspace(qtbot, values=(3.0,))
    widget.file_list.selectAll()
    widget.op_combo.setCurrentText("Scale / offset (a·y + b)")
    widget.param_edits["factor"].setText("2")
    widget.param_edits["offset"].setText("1")
    widget.apply_selected()
    qtbot.wait(20)
    derived = [s for s in library.all() if s.status == "derived"]
    assert np.allclose(derived[0].y, 7.0)


def test_lcf_reports_coefficients_and_adds_fit_and_residual(qtbot):
    library = SpectrumLibrary()
    x = np.linspace(0, 100, 300)
    r1 = np.exp(-0.5 * ((x - 30) / 5.0) ** 2)
    r2 = np.exp(-0.5 * ((x - 70) / 8.0) ** 2)
    library.add(Spectrum(id=Spectrum.new_id(), title="mix", path="", kind="raman_xy", x=x, y=0.6 * r1 + 0.4 * r2))
    library.add(Spectrum(id=Spectrum.new_id(), title="refA", path="", kind="raman_xy", x=x, y=r1))
    library.add(Spectrum(id=Spectrum.new_id(), title="refB", path="", kind="raman_xy", x=x, y=r2))
    widget = CalcWorkspace(library=library)
    qtbot.addWidget(widget)
    widget.set_spectra([s.id for s in library.all()])
    widget.file_list.selectAll()
    widget.op_combo.setCurrentText("Linear-combination fit (target ≈ Σ cᵢ·refᵢ)")
    widget.apply_selected()
    qtbot.wait(20)

    report = widget.report_text.toPlainText()
    assert "R²" in report and "refA" in report and "refB" in report
    titles = [s.title for s in library.all()]
    assert "mix_lcf_fit" in titles and "mix_lcf_residual" in titles


def test_statistics_writes_report_and_adds_nothing(qtbot):
    widget, library = _workspace(qtbot, values=(5.0,))
    widget.file_list.selectAll()
    widget.op_combo.setCurrentText("Statistics")
    widget.apply_selected()
    qtbot.wait(20)
    assert "y_mean=5" in widget.report_text.toPlainText()
    assert not [s for s in library.all() if s.status == "derived"]


def test_derived_added_callback_feeds_undo(qtbot):
    calls = []
    library = SpectrumLibrary()
    library.add(_spectrum("a", 1.0))
    widget = CalcWorkspace(library=library, on_derived_added=lambda ids: calls.append(list(ids)))
    qtbot.addWidget(widget)
    widget.set_spectra([s.id for s in library.all()])
    widget.file_list.selectAll()
    widget.op_combo.setCurrentText("|y|")
    widget.apply_selected()
    qtbot.wait(20)
    assert len(calls) == 1 and len(calls[0]) == 1


def test_shell_has_calculations_page_and_nav_refreshes_it(qtbot):
    window = PrismMainWindow()
    qtbot.addWidget(window)
    window.library.add(_spectrum("in_lib", 2.0))
    window.nav.setCurrentRow(NAV_ITEMS.index("Calculations"))
    qtbot.wait(20)
    assert window.stack.currentWidget() is window.calc_page
    assert window.calc_page.file_list.count() == 1


def test_cluster_operation_reports_and_adds_means(qtbot):
    import rampy as rp
    library = SpectrumLibrary()
    x = np.linspace(0, 100, 200)
    for i, c in enumerate((30.0, 31.0, 70.0, 71.0)):
        library.add(Spectrum(id=Spectrum.new_id(), title=f"s{i}", path="", kind="raman_xy",
                             x=x, y=rp.gaussian(x, 10.0, c, 5.0)))
    widget = CalcWorkspace(library=library)
    qtbot.addWidget(widget)
    widget.set_spectra([s.id for s in library.all()])
    widget.file_list.selectAll()
    widget.op_combo.setCurrentText("Cluster (KMeans)")
    widget.param_edits["n_clusters"].setText("2")
    widget.apply_selected()
    qtbot.wait(20)

    report = widget.report_text.toPlainText()
    assert "Clustering" in report and "cluster" in report
    means = [s for s in library.all() if "cluster" in s.title and "_mean" in s.title]
    assert len(means) == 2


def test_despike_manual_positions_and_pick_click(qtbot):
    from types import SimpleNamespace
    library = SpectrumLibrary()
    x = np.linspace(0, 100, 201)
    y = np.sin(x / 10.0)
    y[100] += 50.0  # a spike the statistical test may or may not catch
    library.add(Spectrum(id=Spectrum.new_id(), title="spiky", path="", kind="raman_xy", x=x, y=y))
    widget = CalcWorkspace(library=library)
    qtbot.addWidget(widget)
    widget.set_spectra([s.id for s in library.all()])
    widget.file_list.selectAll()
    widget.op_combo.setCurrentText("Despike (cosmic rays)")
    widget.param_edits["z"].setText("999")  # disable the statistical test entirely

    # simulate a pick click at the spike position
    widget.pick_btn.setChecked(True)
    widget._on_pick_click(SimpleNamespace(inaxes=widget.plot.figure.gca(), xdata=50.0, ydata=0.0))
    assert widget.param_edits["positions"].text().startswith("50")
    widget.pick_btn.setChecked(False)

    widget.apply_selected()
    qtbot.wait(20)
    derived = [s for s in library.all() if s.status == "derived"][0]
    assert abs(derived.y[100]) < 2.0  # clicked spike removed despite z=999
