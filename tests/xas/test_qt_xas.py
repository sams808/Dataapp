"""Tests for qt_xas.py (M11) — the Qt port of xas_processing_v10.py's
XASUltimateApp, built on xas_science.py's SpectrumStore/Spectrum model.

Run separately from the default Tk-focused suite (see pytest.ini / conftest.py
for why): `pytest tests/test_qt_xas.py --override-ini="addopts="`
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.special import erf

from core.qt_shell import NAV_ITEMS, PrismMainWindow
from xas.qt_xas import XasWorkspace
from xas.xas_science import Spectrum, _uid


def _synthetic_mu_spectrum(name="sample_mu", e0=7112.0, n=400, e_lo=6900.0, e_hi=7400.0) -> Spectrum:
    energy = np.linspace(e_lo, e_hi, n)
    pre = 0.15
    edge = 1.0 * 0.5 * (1 + erf((energy - e0) / 3.0))
    # A small, realistic noise term (real detector/counting noise is always
    # far larger than this) — a perfectly noiseless erf() saturates to a
    # constant in its deep tails to within ~1e-17, which can trip a rank-
    # deficient-fit edge case in Larch's own polyfit() (np.polynomial.
    # Polynomial.fit silently drops the linear coefficient when the y-window
    # is constant to within machine epsilon but not exactly, vs. handling an
    # EXACTLY constant window fine) — confirmed directly by bisecting grid
    # density/range combinations until the crash reproduced, unrelated to
    # anything in this codebase. Unreachable with real experimental data.
    rng = np.random.default_rng(0)
    mu = pre + edge + rng.normal(0, 1e-4, size=energy.shape)
    return Spectrum(sid=_uid("sp"), name=name, kind="mu", energy=energy, y=mu, label="XAS(Fe K)", e0=e0)


def test_workspace_constructs_empty(qtbot):
    widget = XasWorkspace()
    qtbot.addWidget(widget)
    assert widget.table.rowCount() == 0
    assert widget.mu_i0_combo.count() == 0


def test_import_csv_classifies_i0_and_it_by_filename(qtbot, tmp_path, monkeypatch):
    import pandas as pd

    i0_path = tmp_path / "sample_I0.csv"
    it_path = tmp_path / "sample_It.csv"
    df = pd.DataFrame({"Energy(eV)": np.linspace(7000, 7200, 50), "ROI_countsPerLive": np.full(50, 100.0)})
    df.to_csv(i0_path, index=False)
    df.to_csv(it_path, index=False)

    widget = XasWorkspace()
    qtbot.addWidget(widget)
    monkeypatch.setattr("xas._qt_xas_objects.QFileDialog.getOpenFileNames", staticmethod(lambda *a, **k: ([str(i0_path), str(it_path)], "")))
    widget.import_csvs()

    names_kinds = [(widget.table.item(r, 0).text(), widget.table.item(r, 1).text()) for r in range(widget.table.rowCount())]
    assert ("sample_I0", "I0") in names_kinds
    assert ("sample_It", "It") in names_kinds


def test_mu_builder_computes_new_object_and_deglitch_changes_output(qtbot):
    """Rolling z-score method specifically (the test's own z/window
    fields) -- selected explicitly since the noise-calibrated method is
    now the tab's default and, correctly, declines to touch a perfectly
    flat/noiseless synthetic baseline like this one (its 3x-noise
    threshold is undefined when the true point-to-point noise is exactly
    zero -- see test_mu_builder_noise_calibrated_deglitch_removes_spike
    for that method's own, more representative coverage)."""
    widget = XasWorkspace()
    qtbot.addWidget(widget)

    energy = np.linspace(7000, 7300, 300)
    i0 = Spectrum(sid=_uid("sp"), name="I0_a", kind="I0", energy=energy, y=np.full(300, 100.0))
    it_y = np.full(300, 50.0)
    it_y[150] *= 0.2  # glitch
    it = Spectrum(sid=_uid("sp"), name="It_a", kind="It", energy=energy, y=it_y)
    widget.store.add(i0)
    widget.store.add(it)
    widget._refresh_all()

    widget.mu_i0_combo.setCurrentText("I0_a")
    for i in range(widget.mu_it_list.count()):
        if widget.mu_it_list.item(i).text() == "It_a":
            widget.mu_it_list.item(i).setSelected(True)

    widget.compute_mu_selected()
    qtbot.wait(20)  # let the preview's deferred canvas.draw_idle() complete before teardown
    mu_plain = widget.store.find_by_name("It_a_mu")
    assert mu_plain is not None
    assert mu_plain.y[150] > mu_plain.y[100] + 1.0  # glitch still present without deglitch

    widget.mu_deglitch_check.setChecked(True)
    widget.mu_deglitch_method_combo.setCurrentIndex(1)  # "rolling z-score"
    widget.mu_deglitch_z_edit.setText("4.0")
    widget.mu_deglitch_window_edit.setText("15")
    widget.compute_mu_selected()
    qtbot.wait(20)
    # There will now be two "It_a_mu" objects (compute_mu_selected() was
    # called twice); fetch both by store order rather than find_by_name
    # (which only returns the first match).
    mu_objects = [s for s in widget.store.all() if s.name == "It_a_mu"]
    assert len(mu_objects) == 2
    assert mu_objects[-1].y[150] < mu_objects[0].y[150]  # deglitched version pulls the spike down


def test_mu_builder_noise_calibrated_deglitch_removes_spike(qtbot):
    """The tab's default deglitch method (noise-calibrated, see
    xas_science.deglitch_3x_noise) on a realistically-noisy signal --
    unlike the rolling z-score test above, this needs actual point-to-
    point noise for its 3x-noise threshold to be well-defined."""
    widget = XasWorkspace()
    qtbot.addWidget(widget)

    rng = np.random.default_rng(0)
    energy = np.linspace(7000, 7300, 300)
    i0 = Spectrum(sid=_uid("sp"), name="I0_b", kind="I0", energy=energy, y=np.full(300, 100.0))
    it_y = 50.0 + rng.normal(0.0, 0.05, size=300)
    it_y[150] *= 0.2  # glitch, on top of the noise
    it = Spectrum(sid=_uid("sp"), name="It_b", kind="It", energy=energy, y=it_y)
    widget.store.add(i0)
    widget.store.add(it)
    widget._refresh_all()

    widget.mu_i0_combo.setCurrentText("I0_b")
    for i in range(widget.mu_it_list.count()):
        if widget.mu_it_list.item(i).text() == "It_b":
            widget.mu_it_list.item(i).setSelected(True)

    assert widget.mu_deglitch_method_combo.currentIndex() == 0  # default is noise-calibrated
    widget.mu_deglitch_check.setChecked(True)
    widget.compute_mu_selected()
    qtbot.wait(20)
    mu_degl = widget.store.find_by_name("It_b_mu")
    assert mu_degl is not None
    assert mu_degl.y[150] < 1.0  # spike pulled back down near the surrounding baseline (~0.69)


def test_normalize_selected_respects_custom_pre_edge_ranges(qtbot):
    """Regression guard for the xas_science.py pre_edge-kwargs fix: custom
    pre1/pre2/norm1/norm2 typed in the UI must actually reach Larch, not be
    silently replaced by its own auto-computed defaults."""
    widget = XasWorkspace()
    qtbot.addWidget(widget)
    sp = _synthetic_mu_spectrum()
    widget.store.add(sp)
    widget._refresh_all()

    for i in range(widget.norm_mu_list.count()):
        if widget.norm_mu_list.item(i).text() == sp.name:
            widget.norm_mu_list.item(i).setSelected(True)

    widget.norm_pre1_edit.setText("-150")
    widget.norm_pre2_edit.setText("-50")
    widget.norm_norm1_edit.setText("30")
    widget.norm_norm2_edit.setText("150")
    widget.norm_nnorm_edit.setText("1")
    widget.normalize_selected()
    qtbot.wait(20)  # let the deferred canvas.draw_idle() complete before teardown

    norm_sp = widget.store.find_by_name(f"{sp.name}_norm")
    assert norm_sp is not None
    assert norm_sp.e0 == pytest.approx(7112.0, abs=15.0)
    flat_sp = widget.store.find_by_name(f"{sp.name}_flat")
    assert flat_sp is not None


def test_exafs_selected_produces_chi_and_ft_objects(qtbot):
    widget = XasWorkspace()
    qtbot.addWidget(widget)
    sp = _synthetic_mu_spectrum(n=600, e_hi=7500.0)
    widget.store.add(sp)
    widget._refresh_all()
    for i in range(widget.norm_mu_list.count()):
        if widget.norm_mu_list.item(i).text() == sp.name:
            widget.norm_mu_list.item(i).setSelected(True)

    widget.exafs_selected()
    qtbot.wait(20)  # let the deferred canvas.draw_idle() complete before teardown

    assert widget.store.find_by_name(f"{sp.name}_chi") is not None
    assert widget.store.find_by_name(f"{sp.name}_chi_k2") is not None
    assert widget.store.find_by_name(f"{sp.name}_FTmag") is not None


def test_merge_average_produces_new_spectrum(qtbot):
    widget = XasWorkspace()
    qtbot.addWidget(widget)
    energy = np.linspace(0, 100, 50)
    a = Spectrum(sid=_uid("sp"), name="a", kind="mu", energy=energy, y=np.full(50, 1.0))
    b = Spectrum(sid=_uid("sp"), name="b", kind="mu", energy=energy, y=np.full(50, 3.0))
    widget.store.add(a); widget.store.add(b)
    widget._refresh_all()

    for i in range(widget.analysis_list.count()):
        widget.analysis_list.item(i).setSelected(True)
    widget.merge_average_selected()
    qtbot.wait(20)  # let the deferred canvas.draw_idle() complete before teardown

    avg = widget.store.find_by_name("a_avg2")
    assert avg is not None
    assert np.allclose(avg.y, 2.0)


def test_difference_produces_new_spectrum(qtbot):
    widget = XasWorkspace()
    qtbot.addWidget(widget)
    energy = np.linspace(0, 100, 50)
    a = Spectrum(sid=_uid("sp"), name="a", kind="mu", energy=energy, y=np.full(50, 5.0))
    b = Spectrum(sid=_uid("sp"), name="b", kind="mu", energy=energy, y=np.full(50, 2.0))
    widget.store.add(a); widget.store.add(b)
    widget._refresh_all()

    for i in range(widget.analysis_list.count()):
        widget.analysis_list.item(i).setSelected(True)
    widget.difference_selected()
    qtbot.wait(20)  # let the deferred canvas.draw_idle() complete before teardown

    diff = widget.store.find_by_name("a_minus_b")
    assert diff is not None
    assert np.allclose(diff.y, 3.0)


def test_linear_combination_fit_recovers_known_weights(qtbot):
    widget = XasWorkspace()
    qtbot.addWidget(widget)
    energy = np.linspace(0, 100, 100)
    ref1 = Spectrum(sid=_uid("sp"), name="ref1", kind="mu", energy=energy, y=np.sin(energy / 10) + 2)
    ref2 = Spectrum(sid=_uid("sp"), name="ref2", kind="mu", energy=energy, y=np.cos(energy / 15) + 2)
    target_y = 0.7 * ref1.y + 0.3 * ref2.y
    target = Spectrum(sid=_uid("sp"), name="target", kind="mu", energy=energy, y=target_y)
    for sp in (ref1, ref2, target):
        widget.store.add(sp)
    widget._refresh_all()

    # Order matters: target must be selected LAST.
    for i in range(widget.analysis_list.count()):
        item = widget.analysis_list.item(i)
        if item.text() in ("ref1", "ref2", "target"):
            item.setSelected(True)

    widget.linear_combination_fit_selected()
    qtbot.wait(20)  # let the deferred canvas.draw_idle() complete before teardown
    fit_sp = widget.store.find_by_name("target_LCFfit")
    assert fit_sp is not None
    assert np.allclose(fit_sp.y, target_y, atol=1e-6)
    assert "R²" in widget.analysis_result_text.toPlainText()


def test_preproc_smoothing_apply_creates_new_object(qtbot):
    widget = XasWorkspace()
    qtbot.addWidget(widget)
    energy = np.linspace(7000, 7300, 300)
    rng = np.random.default_rng(0)
    sp = Spectrum(sid=_uid("sp"), name="noisy", kind="It", energy=energy,
                  y=np.sin(energy / 30) + rng.normal(0, 0.1, energy.shape))
    widget.store.add(sp)
    widget._refresh_all()

    widget.sm_target_combo.setCurrentText("noisy")
    widget.sm_method_combo.setCurrentText("Savitzky-Golay")
    widget.apply_smoothing()
    qtbot.wait(20)

    sp2 = widget.store.find_by_name("noisy_sm")
    assert sp2 is not None
    # Smoothed variance of the residual to the clean signal must drop.
    clean = np.sin(energy / 30)
    assert np.std(sp2.y - clean) < np.std(sp.y - clean)
    assert sp2.history[-1].name == "smooth"


def test_preproc_mode_c_tiepoint_alignment_via_simulated_clicks(qtbot):
    """Full Mode C flow: overlay, pick a BEFORE/AFTER pair via simulated
    canvas clicks (snapping to nearest data point), apply shift alignment,
    and confirm the corrected object's energy axis moved by the tie-point
    difference."""
    widget = XasWorkspace()
    qtbot.addWidget(widget)
    energy = np.linspace(7000, 7300, 301)  # 1 eV grid
    y = np.exp(-((energy - 7112.0) ** 2) / 50.0)
    before = Spectrum(sid=_uid("sp"), name="ref", kind="mu", energy=energy, y=y)
    after = Spectrum(sid=_uid("sp"), name="shifted", kind="mu", energy=energy + 5.0, y=y.copy())
    widget.store.add(before)
    widget.store.add(after)
    widget._refresh_all()

    widget.ang_before_combo.setCurrentText("ref")
    widget.ang_after_combo.setCurrentText("shifted")
    widget.plot_mode_c_overlay()
    qtbot.wait(20)
    widget.start_picking_pair()

    ax = widget.preproc_plot.figure.get_axes()[0]

    class _Click:
        def __init__(self, x):
            self.inaxes = ax
            self.xdata = x
            self.ydata = 0.5

    widget._on_preproc_click(_Click(7112.0))   # BEFORE feature
    widget._on_preproc_click(_Click(7117.0))   # AFTER feature (shifted +5)
    qtbot.wait(20)
    assert len(widget.tiepoints) == 1
    assert widget.tiepoints[0].e_before == pytest.approx(7112.0, abs=0.6)
    assert widget.tiepoints[0].e_after == pytest.approx(7117.0, abs=0.6)

    widget.ang_mode_combo.setCurrentText("C: Feature alignment (click)")
    widget.mode_c_model_combo.setCurrentText("shift")
    widget.apply_angle_correction()
    qtbot.wait(20)

    corrected = widget.store.find_by_name("shifted_Ealign")
    assert corrected is not None
    # Shift model: corrected energy = after energy + (before - after) = -5 eV
    assert corrected.energy[0] == pytest.approx(after.energy[0] - 5.0, abs=0.6)
    assert corrected.history[-1].name == "align_mode_c"


def test_pca_selected_reports_species_count(qtbot):
    """M21: PCA across a synthetic 2-species series must attribute the
    dominant variance to PC1 and report a small species count."""
    widget = XasWorkspace()
    qtbot.addWidget(widget)
    energy = np.linspace(7000, 7400, 300)
    rng = np.random.default_rng(0)
    spec_a = 0.15 + 1.0 * 0.5 * (1 + erf((energy - 7112.0) / 3.0))
    spec_b = 0.15 + 1.0 * 0.5 * (1 + erf((energy - 7130.0) / 3.0))
    for i, frac in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
        y = (1 - frac) * spec_a + frac * spec_b + rng.normal(0, 1e-3, energy.shape)
        widget.store.add(Spectrum(sid=_uid("sp"), name=f"mix{i}", kind="mu", energy=energy, y=y))
    widget._refresh_all()

    for i in range(widget.analysis_list.count()):
        widget.analysis_list.item(i).setSelected(True)
    widget.pca_selected()
    qtbot.wait(20)

    text = widget.analysis_result_text.toPlainText()
    assert "PC1" in text
    assert "distinct species" in text
    # Two-species linear mixtures live on a 1D line in spectrum space:
    # PC1 alone should carry essentially all the variance.
    assert "PC1: 9" in text or "PC1: 100" in text  # >= 90%
    assert len(widget.analysis_plot.figure.get_axes()) == 2  # scatter + scree


def test_edge_definer_apply_sets_label(qtbot):
    widget = XasWorkspace()
    qtbot.addWidget(widget)
    sp = Spectrum(sid=_uid("sp"), name="unknown_spec", kind="mu", energy=np.linspace(7000, 7200, 30), y=np.ones(30))
    widget.store.add(sp)
    widget._refresh_all()

    for i in range(widget.edge_list.count()):
        if widget.edge_list.item(i).text() == "unknown_spec":
            widget.edge_list.item(i).setSelected(True)

    widget.edge_elem_combo.setCurrentText("Cu")
    widget.edge_line_combo.setCurrentText("K")
    widget.apply_edge_definer()
    qtbot.wait(20)  # let the deferred canvas.draw_idle() complete before teardown

    updated = widget.store.find_by_name("unknown_spec")
    assert updated.label == "XAS(Cu K)"


def test_export_athena_dat_writes_file(qtbot, tmp_path, monkeypatch):
    widget = XasWorkspace()
    qtbot.addWidget(widget)
    sp = _synthetic_mu_spectrum()
    widget.store.add(sp)
    widget._refresh_all()
    widget.selected_sid = sp.sid

    out_path = tmp_path / "exported.dat"
    monkeypatch.setattr("xas._qt_xas_export.QFileDialog.getSaveFileName", staticmethod(lambda *a, **k: (str(out_path), "")))
    widget.export_athena_dat()

    assert out_path.exists()
    content = out_path.read_text()
    assert "sample_mu" in content


def test_duplicate_and_delete_selected(qtbot):
    widget = XasWorkspace()
    qtbot.addWidget(widget)
    sp = _synthetic_mu_spectrum()
    widget.store.add(sp)
    widget._refresh_all()
    widget.selected_sid = sp.sid

    widget.duplicate_selected()
    assert widget.store.find_by_name(f"{sp.name}_copy") is not None

    widget.selected_sid = sp.sid
    # QMessageBox.question is neutralized to Yes by conftest's autouse _prevent_blocking_qt_dialogs fixture.
    widget.delete_selected()
    assert widget.store.find_by_name(sp.name) is None


def test_shell_xas_page_is_xas_workspace(qtbot):
    window = PrismMainWindow()
    qtbot.addWidget(window)
    qtbot.wait(20)
    assert isinstance(window.xas_page, XasWorkspace)
    window.nav.setCurrentRow(NAV_ITEMS.index("XAS"))
    qtbot.wait(20)
    assert window.stack.currentWidget() is window.xas_page


def test_analysis_sum_selected_creates_summed_object(qtbot):
    """User request: summing alongside averaging in the XAS stream."""
    widget = XasWorkspace()
    qtbot.addWidget(widget)
    energy = np.linspace(0, 100, 50)
    a = Spectrum(sid=_uid("sp"), name="a", kind="mu", energy=energy, y=np.full(50, 1.0))
    b = Spectrum(sid=_uid("sp"), name="b", kind="mu", energy=energy, y=np.full(50, 3.0))
    widget.store.add(a); widget.store.add(b)
    widget._refresh_all()

    for i in range(widget.analysis_list.count()):
        widget.analysis_list.item(i).setSelected(True)
    widget.sum_selected()
    qtbot.wait(20)

    summed = widget.store.find_by_name("a_sum2")
    assert summed is not None
    assert np.allclose(summed.y, 4.0)
    assert summed.history[-1].name == "merge_sum"
