"""Tests for the shell's project save/load wiring (M14)."""
from __future__ import annotations

import numpy as np
import rampy as rp

from core.qt_models import Spectrum
from core.qt_shell import PrismMainWindow


def _spectrum(title="s1"):
    x = np.linspace(400, 600, 200)
    y = rp.gaussian(x, 80.0, 505.0, 25.0)
    return Spectrum(id=Spectrum.new_id(), title=title, path="", kind="raman_xy", x=x, y=y)


def test_shell_project_round_trip(qtbot, tmp_path, monkeypatch):
    window = PrismMainWindow()
    qtbot.addWidget(window)
    qtbot.wait(20)

    sp = _spectrum("session_spectrum")
    window.library.add(sp)
    window.fit_param_memory.set(sp.id, [{"shape": "G", "shift_val": 505.0}])

    project_path = tmp_path / "roundtrip.prism"
    monkeypatch.setattr("qt_shell.QFileDialog.getSaveFileName", staticmethod(lambda *a, **k: (str(project_path), "")))
    window.save_project()
    assert project_path.exists()

    # Fresh window, load the project back.
    window2 = PrismMainWindow()
    qtbot.addWidget(window2)
    qtbot.wait(20)
    monkeypatch.setattr("qt_shell.QFileDialog.getOpenFileName", staticmethod(lambda *a, **k: (str(project_path), "")))
    window2.open_project()

    loaded = window2.library.all()
    assert len(loaded) == 1
    assert loaded[0].title == "session_spectrum"
    assert loaded[0].id == sp.id
    assert np.allclose(loaded[0].x, sp.x)
    assert window2.fit_param_memory.get(sp.id)[0]["shift_val"] == 505.0
    # Library table refreshed too.
    assert window2.library_page.table.rowCount() == 1


def test_save_project_appends_extension(qtbot, tmp_path, monkeypatch):
    window = PrismMainWindow()
    qtbot.addWidget(window)
    qtbot.wait(20)
    window.library.add(_spectrum())

    bare_path = tmp_path / "noext"
    monkeypatch.setattr("qt_shell.QFileDialog.getSaveFileName", staticmethod(lambda *a, **k: (str(bare_path), "")))
    window.save_project()
    assert (tmp_path / "noext.prism").exists()


_MINIMAL_CIF = """\
data_test
_chemical_name_mineral 'Quartz'
_cell_length_a 4.913
_cell_length_b 4.913
_cell_length_c 5.405
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 120
"""


def test_project_v3_round_trips_cif_overlays_and_baseline_settings(qtbot, tmp_path, monkeypatch):
    """Session-persistence promise: CIF overlays (recomputed from their
    paths on load) and per-spectrum Baseline settings survive save/open."""
    cif_path = tmp_path / "Quartz__0001.cif"
    cif_path.write_text(_MINIMAL_CIF, encoding="utf-8")

    window = PrismMainWindow()
    qtbot.addWidget(window)
    qtbot.wait(20)

    sp = _spectrum("with_settings")
    window.library.add(sp)
    window.raman_page.add_cif_files([str(cif_path)])
    qtbot.wait(200)
    window.raman_page.cif_series[0]["plot_label"] = "my quartz"
    window.raman_page.cif_series[0]["color"] = "royalblue"
    window.baseline_page.settings.set(sp.id, {"method": "poly", "roi_text": "100-400", "p0": "2", "p1": ""})

    project_path = tmp_path / "v3.prism"
    monkeypatch.setattr("qt_shell.QFileDialog.getSaveFileName", staticmethod(lambda *a, **k: (str(project_path), "")))
    window.save_project()

    window2 = PrismMainWindow()
    qtbot.addWidget(window2)
    qtbot.wait(20)
    monkeypatch.setattr("qt_shell.QFileDialog.getOpenFileName", staticmethod(lambda *a, **k: (str(project_path), "")))
    window2.open_project()
    qtbot.wait(250)  # restore_cif_overlays renders via the debounce

    assert len(window2.raman_page.cif_series) == 1
    restored = window2.raman_page.cif_series[0]
    assert restored["plot_label"] == "my quartz"
    assert restored["color"] == "royalblue"
    assert len(restored["peaks"]) > 0  # recomputed from the CIF path

    bl = window2.baseline_page.settings.get(sp.id)
    assert bl["method"] == "poly"
    assert bl["roi_text"] == "100-400"


def test_open_project_replaces_existing_library(qtbot, tmp_path, monkeypatch):
    window = PrismMainWindow()
    qtbot.addWidget(window)
    qtbot.wait(20)
    keeper = _spectrum("in_project")
    window.library.add(keeper)

    project_path = tmp_path / "p.prism"
    monkeypatch.setattr("qt_shell.QFileDialog.getSaveFileName", staticmethod(lambda *a, **k: (str(project_path), "")))
    window.save_project()

    # Add a second spectrum, then open the project — the pre-open library
    # contents must be replaced (question dialog auto-answers Yes via
    # conftest's autouse fixture).
    window.library.add(_spectrum("volatile"))
    assert len(window.library) == 2
    monkeypatch.setattr("qt_shell.QFileDialog.getOpenFileName", staticmethod(lambda *a, **k: (str(project_path), "")))
    window.open_project()

    titles = [s.title for s in window.library.all()]
    assert titles == ["in_project"]
