"""Tests for qt_multi_fit.py (M9) — the fresh-built saved-recipe batch
fitting workspace.

Run separately from the default Tk-focused suite (see pytest.ini / conftest.py
for why): `pytest tests/test_qt_multi_fit.py --override-ini="addopts="`
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import rampy as rp

from core.qt_models import Spectrum, SpectrumLibrary
from processing.qt_multi_fit import MultiFitWorkspace
from core.qt_settings_store import PerItemSettingsStore
from core.qt_shell import NAV_ITEMS, PrismMainWindow, _load_spectrum_from_path
from processing.qt_single_fit import SingleFitWorkspace


def _component(center=500.0, fwhm=30.0, amp=100.0, shape="G"):
    return {
        "shape": shape,
        "shift_val": center, "shift_min": center - 100, "shift_max": center + 100, "fit_shift": True,
        "fwhm_val": fwhm, "fwhm_min": 1.0, "fwhm_max": 200.0, "fit_fwhm": True,
        "eta_val": 0.5, "eta_min": 0.0, "eta_max": 1.0, "fit_eta": True,
        "amp_val": amp, "fit_amp": True,
    }


def _synthetic_spectrum(title="s", true_center=505.0, true_fwhm=25.0, true_amp=80.0) -> Spectrum:
    x = np.linspace(400, 600, 400)
    y = rp.gaussian(x, true_amp, true_center, true_fwhm)
    return Spectrum(id=Spectrum.new_id(), title=title, path="", kind="raman_xy", x=x, y=y)


def _write_recipe(model_dir, name, params_struct):
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / f"{name}.json").write_text(json.dumps(params_struct))


def test_workspace_constructs_empty(qtbot, tmp_path):
    widget = MultiFitWorkspace(model_dir=str(tmp_path))
    qtbot.addWidget(widget)
    assert widget.file_list.count() == 0
    assert widget.recipe_combo.count() == 0


def test_set_spectra_populates_file_list(qtbot, tmp_path):
    library = SpectrumLibrary()
    s1 = _synthetic_spectrum("a")
    library.add(s1)
    widget = MultiFitWorkspace(library=library, model_dir=str(tmp_path))
    qtbot.addWidget(widget)
    widget.set_spectra([s1.id])
    assert widget.file_list.count() == 1


def test_refresh_recipe_list_picks_up_saved_models(qtbot, tmp_path):
    _write_recipe(tmp_path, "my_recipe", [_component()])
    widget = MultiFitWorkspace(model_dir=str(tmp_path))
    qtbot.addWidget(widget)
    assert widget.recipe_combo.count() == 1
    assert widget.recipe_combo.itemText(0) == "my_recipe"


def test_run_batch_fits_all_selected_spectra(qtbot, tmp_path):
    _write_recipe(tmp_path, "single_gaussian", [_component(center=500.0, fwhm=30.0, amp=100.0)])

    library = SpectrumLibrary()
    s1 = _synthetic_spectrum("spec1", true_center=505.0)
    s2 = _synthetic_spectrum("spec2", true_center=520.0)
    library.add(s1)
    library.add(s2)

    widget = MultiFitWorkspace(library=library, model_dir=str(tmp_path))
    qtbot.addWidget(widget)
    widget.set_spectra([s1.id, s2.id])
    widget.file_list.selectAll()
    widget.recipe_combo.setCurrentText("single_gaussian")

    widget.run_batch()
    qtbot.wait(20)  # let the preview's deferred canvas.draw_idle() complete before teardown

    assert len(widget._results) == 2
    assert widget.results_table.rowCount() == 2  # one component each
    centers = sorted(r.params_struct[0]["shift_val"] for r in widget._results)
    assert centers[0] == pytest.approx(505.0, abs=1.0)
    assert centers[1] == pytest.approx(520.0, abs=1.0)


def test_run_batch_writeback_is_visible_in_shared_store(qtbot, tmp_path):
    _write_recipe(tmp_path, "single_gaussian", [_component(center=500.0, fwhm=30.0, amp=100.0)])

    library = SpectrumLibrary()
    s1 = _synthetic_spectrum("spec1", true_center=505.0)
    library.add(s1)

    shared_store: PerItemSettingsStore = PerItemSettingsStore(list)
    multi = MultiFitWorkspace(library=library, fit_param_memory=shared_store, model_dir=str(tmp_path))
    single = SingleFitWorkspace(library=library, fit_param_memory=shared_store)
    qtbot.addWidget(multi)
    qtbot.addWidget(single)

    multi.set_spectra([s1.id])
    multi.file_list.selectAll()
    multi.recipe_combo.setCurrentText("single_gaussian")
    multi.run_batch()
    qtbot.wait(20)  # let the preview's deferred canvas.draw_idle() complete before teardown

    assert shared_store.has(s1.id)
    assert single.fit_param_memory.get(s1.id)[0]["shift_val"] == pytest.approx(505.0, abs=1.0)


def test_run_batch_without_recipe_warns_and_does_not_crash(qtbot, tmp_path):
    library = SpectrumLibrary()
    s1 = _synthetic_spectrum("spec1")
    library.add(s1)
    widget = MultiFitWorkspace(library=library, model_dir=str(tmp_path))
    qtbot.addWidget(widget)
    widget.set_spectra([s1.id])
    widget.file_list.selectAll()
    widget.run_batch()  # no recipe selected -> QMessageBox.warning, neutralized by conftest fixture
    assert widget._results == []


def test_export_results_csv_writes_file(qtbot, tmp_path, monkeypatch):
    _write_recipe(tmp_path, "single_gaussian", [_component(center=500.0, fwhm=30.0, amp=100.0)])
    library = SpectrumLibrary()
    s1 = _synthetic_spectrum("spec1", true_center=505.0)
    library.add(s1)

    widget = MultiFitWorkspace(library=library, model_dir=str(tmp_path))
    qtbot.addWidget(widget)
    widget.set_spectra([s1.id])
    widget.file_list.selectAll()
    widget.recipe_combo.setCurrentText("single_gaussian")
    widget.run_batch()
    qtbot.wait(20)  # let the preview's deferred canvas.draw_idle() complete before teardown

    out_path = tmp_path / "results.csv"
    monkeypatch.setattr("qt_multi_fit.QFileDialog.getSaveFileName", staticmethod(lambda *a, **k: (str(out_path), "")))
    widget.export_results_csv()

    assert out_path.exists()
    content = out_path.read_text()
    assert "spec1" in content
    assert "chi2_red" in content


def test_preview_plot_updates_on_row_selection(qtbot, tmp_path):
    _write_recipe(tmp_path, "single_gaussian", [_component(center=500.0, fwhm=30.0, amp=100.0)])
    library = SpectrumLibrary()
    s1 = _synthetic_spectrum("spec1", true_center=505.0)
    library.add(s1)

    widget = MultiFitWorkspace(library=library, model_dir=str(tmp_path))
    qtbot.addWidget(widget)
    widget.set_spectra([s1.id])
    widget.file_list.selectAll()
    widget.recipe_combo.setCurrentText("single_gaussian")
    widget.run_batch()
    qtbot.wait(20)  # let the preview's deferred canvas.draw_idle() complete before teardown

    assert len(widget.plot.figure.get_axes()) >= 1
    assert len(widget.plot.figure.get_axes()[0].lines) >= 2  # data + fit


def test_shell_refreshes_recipes_saved_after_construction_on_nav_switch(qtbot, tmp_path, monkeypatch, raman_example_path):
    """Regression guard, found via manual smoke-testing the real shell (not
    just the isolated widget): MultiFitWorkspace only scanned model_dir at
    its own __init__ time, so a recipe saved via Peak Fitting's "Save as
    model..." (or written directly, as here) AFTER the shell already exists
    silently didn't appear in the Multi-Fit combo until the user clicked its
    own manual Refresh button. qt_shell.py now re-scans on every nav switch
    into the Multi-Fit page, mirroring how spectra are already refreshed."""
    monkeypatch.setattr("qt_multi_fit._default_model_dir", lambda: str(tmp_path))

    window = PrismMainWindow()
    qtbot.addWidget(window)
    qtbot.wait(20)

    spectrum = _load_spectrum_from_path(str(raman_example_path))
    window.library.add(spectrum)

    _write_recipe(tmp_path, "late_recipe", [_component()])  # written AFTER construction

    window.nav.setCurrentRow(NAV_ITEMS.index("Multi-Fit"))
    qtbot.wait(20)

    names = [window.multifit_page.recipe_combo.itemText(i) for i in range(window.multifit_page.recipe_combo.count())]
    assert "late_recipe" in names


def test_shell_multifit_page_picks_up_library_records(qtbot, raman_example_path):
    window = PrismMainWindow()
    qtbot.addWidget(window)

    spectrum = _load_spectrum_from_path(str(raman_example_path))
    window.library.add(spectrum)

    window.nav.setCurrentRow(NAV_ITEMS.index("Multi-Fit"))
    qtbot.wait(20)

    assert window.multifit_page.file_list.count() == 1


def test_shell_shares_fit_param_memory_between_fitting_and_multifit_pages(qtbot):
    window = PrismMainWindow()
    qtbot.addWidget(window)
    qtbot.wait(20)  # flush any deferred draws queued while constructing all 6 pages
    assert window.fitting_page.fit_param_memory is window.fit_param_memory
    assert window.multifit_page.fit_param_memory is window.fit_param_memory
