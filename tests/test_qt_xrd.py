"""Tests for qt_xrd.py — the QualX-style XRD phase-ID workspace.
Uses the same tiny synthetic QualX-format database as test_xrd_id_science."""
from __future__ import annotations

import numpy as np
import pytest

import xrd_id_science as xid
from core.qt_models import Spectrum, SpectrumLibrary
from core.qt_shell import NAV_ITEMS, PrismMainWindow
from qt_xrd import XrdIdWorkspace
from test_xrd_id_science import QUARTZ_D, QUARTZ_I, _make_source_sq


@pytest.fixture()
def db_path(tmp_path):
    src = tmp_path / "src.sq"
    _make_source_sq(src, [
        (1010, "Quartz low", "Quartz", "Si O2", "P 32 2 1", "A", QUARTZ_D, QUARTZ_I, ["Si", "O"]),
        (1020, "Calcite", "Calcite", "C Ca O3", "R -3 c", "A",
         [3.035, 2.285, 2.095, 1.913, 1.875, 3.86], [100.0, 18.0, 18.0, 17.0, 17.0, 12.0], ["C", "Ca", "O"]),
    ])
    out = tmp_path / "unified.sq"
    xid.build_xrd_database([(str(src), "TESTDB")], out_path=str(out), log=lambda *a: None)
    return str(out)


def _quartz_pattern() -> Spectrum:
    """Synthetic diffractogram with quartz's lines as narrow Gaussians."""
    tt = xid.d_to_two_theta(np.array(QUARTZ_D))
    x = np.linspace(10, 80, 4000)
    y = np.full_like(x, 50.0)
    for c, i in zip(tt, QUARTZ_I):
        y += 10.0 * i * np.exp(-0.5 * ((x - c) / 0.05) ** 2)
    return Spectrum(id=Spectrum.new_id(), title="my_xrd", path="", kind="xrd_xy", x=x, y=y)


def _workspace(qtbot, db_path, on_accept=None):
    library = SpectrumLibrary()
    sp = _quartz_pattern()
    library.add(sp)
    widget = XrdIdWorkspace(library=library, db_path=db_path, on_accept=on_accept)
    qtbot.addWidget(widget)
    widget.set_spectra([sp.id])
    # set_spectra queues the entry preview through the 120ms debounce —
    # let it complete INSIDE the test, or matplotlib's own singleShot(0)
    # idle-draw lands on a torn-down canvas in the next test.
    qtbot.wait(200)
    return widget, library, sp


def test_db_status_and_source_checkboxes(qtbot, db_path):
    widget, _, _ = _workspace(qtbot, db_path)
    assert "2 cards" in widget.db_status_label.text()
    assert set(widget.source_checks.keys()) == {"TESTDB"}


def test_auto_find_then_search_ranks_quartz_first(qtbot, db_path):
    widget, _, _ = _workspace(qtbot, db_path)
    widget.auto_find_peaks()
    assert widget.peaks_edit.text() != ""
    widget.run_search()
    qtbot.wait(20)
    assert widget.results_table.rowCount() >= 1
    assert widget.results_table.item(0, 1).text() == "Quartz"
    assert int(widget.results_table.item(0, 0).text()) > 80  # FoM %
    # provenance visible: source tag + original code
    assert widget.results_table.item(0, 3).text() == "TESTDB"
    assert widget.results_table.item(0, 4).text() == "1010"


def test_accept_records_phase_and_is_iterative(qtbot, db_path):
    calls = []
    widget, _, sp = _workspace(qtbot, db_path, on_accept=lambda sid, old: calls.append((sid, old)))
    widget.auto_find_peaks()
    widget.run_search()
    qtbot.wait(20)
    widget.results_table.selectRow(0)
    widget.accept_selected()
    qtbot.wait(20)

    assert sp.meta["xrd_match"]["mineral"] == "Quartz"
    assert sp.meta["xrd_match"]["source_code"] == "1010"
    assert [m["mineral"] for m in sp.meta["xrd_matches"]] == ["Quartz"]
    assert calls and calls[0][0] == sp.id
    assert calls[0][1] == {"xrd_match": None, "xrd_matches": None}


def test_check_raman_phases_overlays_reference_lines(qtbot, db_path):
    widget, library, sp = _workspace(qtbot, db_path)
    # a Raman spectrum of the same sample, with an accepted Mineral ID phase
    raman = Spectrum(id=Spectrum.new_id(), title="raman_sample", path="", kind="raman_xy",
                     x=np.linspace(100, 1200, 500), y=np.ones(500))
    raman.meta["rruff_matches"] = [{"mineral": "Quartz", "rruff_id": "R040031"}]
    library.add(raman)

    widget.check_raman_phases()
    qtbot.wait(20)
    assert widget.results_table.rowCount() == 1
    assert widget.results_table.item(0, 1).text() == "Quartz"
    assert "Quartz" in widget.db_status_label.text()


def test_card_browser_lookup(qtbot, db_path):
    widget, _, _ = _workspace(qtbot, db_path)
    widget.browse_edit.setText("calcite")
    widget.browse_cards()
    assert widget.browse_list.count() == 1
    assert "Calcite" in widget.browse_list.item(0).text()
    assert "TESTDB 1020" in widget.browse_list.item(0).text()


def test_database_list_shows_and_toggles(qtbot, db_path):
    from PySide6.QtCore import Qt
    widget, _, _ = _workspace(qtbot, db_path)
    assert widget.db_list.count() == 1
    assert widget.db_list.item(0).checkState() == Qt.Checked
    assert widget._enabled_paths() == [db_path]
    widget.db_list.item(0).setCheckState(Qt.Unchecked)
    assert widget._enabled_paths() == []  # unchecked databases are not probed
    qtbot.wait(200)  # flush the deferred list rebuild + debounced render before teardown


def test_add_database_and_search_probes_both(qtbot, db_path, tmp_path):
    """Register a second database at runtime; one search must probe both
    (the user's 'allow them to use multiple if wanted')."""
    src = tmp_path / "extra_src.sq"
    _make_source_sq(src, [
        (9, "Cristobalite", "Cristobalite", "Si O2", "P 41 21 2", "A",
         [4.05, 2.485, 2.841, 3.135], [100.0, 20.0, 13.0, 11.0], ["Si", "O"]),
    ])
    db2 = tmp_path / "extra.sq"
    xid.build_xrd_database([(str(src), "EXTRA")], out_path=str(db2), log=lambda *a: None)

    widget, _, _ = _workspace(qtbot, db_path)
    widget._register_paths([str(db2)])
    qtbot.wait(20)
    assert widget.db_list.count() == 2
    assert len(widget._enabled_paths()) == 2
    assert "Added: extra" in widget.db_status_label.text()

    # a quartz+cristobalite mixture: each phase only exists in one database.
    # No spectrum selected -> equal peak intensities, so the strong-line
    # prefilter probes both phases' lines (with a spectrum, weak second-phase
    # lines are found via the iterative Accept workflow instead).
    tt_q = xid.d_to_two_theta(np.array(QUARTZ_D))
    tt_c = xid.d_to_two_theta(np.array([4.05, 2.485, 2.841, 3.135]))
    widget.peaks_edit.setText(", ".join(f"{v:.3f}" for v in np.concatenate([tt_q, tt_c])))
    widget.spec_combo.setCurrentIndex(-1)
    widget.run_search()
    qtbot.wait(20)
    found = {widget.results_table.item(r, 1).text() for r in range(widget.results_table.rowCount())}
    assert {"Quartz", "Cristobalite"} <= found
    qtbot.wait(200)  # flush the debounced preview render before teardown


def test_add_rejects_non_database_file(qtbot, db_path, tmp_path):
    junk = tmp_path / "junk.sq"
    junk.write_text("not sqlite at all")
    widget, _, _ = _workspace(qtbot, db_path)
    widget._register_paths([str(junk)])
    qtbot.wait(20)
    assert widget.db_list.count() == 1  # nothing added
    assert widget.add_db_btn.isEnabled()  # buttons re-enabled after the error


def test_entry_preview_shows_diffractogram(qtbot, db_path):
    """Arriving on the page renders the query pattern immediately (user
    report: 'where tf is my diffractogram')."""
    widget, _, _ = _workspace(qtbot, db_path)
    axes = widget.plot.figure.get_axes()
    assert axes and axes[0].lines


def test_accept_keeps_overlay_and_clear_resets(qtbot, db_path):
    widget, _, sp = _workspace(qtbot, db_path)
    widget.auto_find_peaks()
    widget.run_search()
    qtbot.wait(20)
    widget.results_table.selectRow(0)
    widget.accept_selected()
    qtbot.wait(20)
    state = widget._state()
    assert len(state["accepted"]) == 1   # the accepted phase stays overlaid (QualX-style)
    assert state["explained"]            # its peaks turn gray instead of vanishing

    widget.clear_session()
    qtbot.wait(250)
    assert widget._state() == {"accepted": [], "explained": []}
    assert widget.peaks_edit.text() == ""
    assert widget.results_table.rowCount() == 0


def test_crystal_system_filter_narrows_results(qtbot, db_path):
    widget, _, _ = _workspace(qtbot, db_path)
    widget.auto_find_peaks()
    widget.system_filter._actions["cubic"].setChecked(True)  # quartz is trigonal
    widget.run_search()
    qtbot.wait(20)
    assert widget.results_table.rowCount() == 0
    widget.system_filter._actions["cubic"].setChecked(False)
    widget.system_filter._actions["trigonal"].setChecked(True)
    widget.run_search()
    qtbot.wait(20)
    assert widget.results_table.rowCount() >= 1


def test_quality_filter_populated_from_databases(qtbot, db_path):
    widget, _, _ = _workspace(qtbot, db_path)
    assert set(widget.quality_filter._actions) == {"A"}  # the fixture DB's only quality code


def test_zoom_preserved_when_testing_other_cards(qtbot, db_path):
    """User request: a zoom on some peaks must survive re-renders (e.g.
    selecting another candidate card); Clear autoscales again."""
    widget, _, _ = _workspace(qtbot, db_path)
    widget.auto_find_peaks()
    widget.run_search()
    qtbot.wait(200)  # results preview rendered
    ax = widget.plot.figure.get_axes()[0]
    widget.plot.toolbar.push_current()  # what a real toolbar zoom does first
    ax.set_xlim(25.0, 28.0)
    ax.set_ylim(-20.0, 60.0)

    widget.plot.request_redraw(widget.render_preview)  # another card clicked
    qtbot.wait(200)
    ax2 = widget.plot.figure.get_axes()[0]
    assert ax2.get_xlim() == (25.0, 28.0)
    assert ax2.get_ylim() == (-20.0, 60.0)

    widget.clear_session()  # starting over zooms back out
    qtbot.wait(250)
    assert widget.plot.figure.get_axes()[0].get_xlim() != (25.0, 28.0)


def test_shell_has_xrd_id_page(qtbot):
    window = PrismMainWindow()
    qtbot.addWidget(window)
    window.nav.setCurrentRow(NAV_ITEMS.index("XRD ID"))
    qtbot.wait(20)
    assert window.stack.currentWidget() is window.xrd_id_page


def test_xrd_accept_is_undoable_through_shell(qtbot, db_path):
    window = PrismMainWindow()
    qtbot.addWidget(window)
    sp = _quartz_pattern()
    window.library.add(sp)
    # simulate what the workspace's on_accept wiring records
    window.library_page.push_undo(("xrd_ident", sp.id, {"xrd_match": None, "xrd_matches": None}))
    sp.meta["xrd_match"] = {"mineral": "Quartz"}
    sp.meta["xrd_matches"] = [{"mineral": "Quartz"}]

    window.library_page._undo()
    assert "xrd_match" not in sp.meta
    assert "xrd_matches" not in sp.meta
