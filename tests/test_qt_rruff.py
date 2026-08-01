"""Tests for qt_rruff.py (M12) — the RRUFF match-assist UI.

Uses a small synthetic cache_dir (index.json + raw/*.txt), never the real
~/.raman_cache/rruff/ corpus, so these tests are fast, deterministic, and
don't depend on M10's ingest having been run on this machine.

Run separately from the default Tk-focused suite (see pytest.ini / conftest.py
for why): `pytest tests/test_qt_rruff.py --override-ini="addopts="`
"""
from __future__ import annotations

import json

import numpy as np
import rampy as rp

from core.qt_models import Spectrum, SpectrumLibrary
from qt_rruff import RruffMatchWorkspace
from core.qt_shell import PrismMainWindow


def _write_fake_cache(cache_dir, raw_dir):
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "Quartz__R040031__Raman__532__0-000____Raman_Data_Processed__abc.txt"
    raw_path.write_text(
        "##NAMES=Quartz\n##RRUFFID=R040031\n##RAMAN WAVELENGTH=532\n\n"
        "460.0, 10.0\n464.0, 500.0\n468.0, 12.0\n1080.0, 8.0\n1085.0, 300.0\n1090.0, 9.0\n",
        encoding="utf-8",
    )
    index = [
        {
            "mineral": "Quartz", "rruff_id": "R040031", "wavelength_nm": 532.0,
            "orientation_deg": None, "polarization": None, "data_kind": "Processed",
            "peaks": [464.0, 1085.0], "raw_path": str(raw_path), "category": "excellent_unoriented",
            "x_min": 460.0, "x_max": 1090.0,
        },
        {
            "mineral": "Calcite", "rruff_id": "R050128", "wavelength_nm": 785.0,
            "orientation_deg": None, "polarization": None, "data_kind": "Processed",
            "peaks": [1085.0], "raw_path": "", "category": "fair_unoriented",
            "x_min": 100.0, "x_max": 1200.0,
        },
        {
            "mineral": "NoPeaksMineral", "rruff_id": "R000000", "wavelength_nm": 514.0,
            "peaks": [], "raw_path": "", "category": "poor_unoriented", "x_min": 0.0, "x_max": 100.0,
        },
    ]
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "index.json").write_text(json.dumps(index), encoding="utf-8")


def _synthetic_query_spectrum() -> Spectrum:
    x = np.linspace(400, 1200, 1600)
    y = rp.gaussian(x, 100.0, 464.0, 8.0) + rp.gaussian(x, 60.0, 1085.0, 8.0)
    return Spectrum(id=Spectrum.new_id(), title="query_sample", path="", kind="raman_xy", x=x, y=y)


def test_workspace_constructs_empty(qtbot, tmp_path):
    widget = RruffMatchWorkspace(cache_dir=str(tmp_path))
    qtbot.addWidget(widget)
    assert widget.spec_combo.count() == 0
    assert widget.results_table.rowCount() == 0


def test_set_spectra_populates_combo(qtbot, tmp_path):
    library = SpectrumLibrary()
    sp = _synthetic_query_spectrum()
    library.add(sp)
    widget = RruffMatchWorkspace(library=library, cache_dir=str(tmp_path))
    qtbot.addWidget(widget)
    widget.set_spectra([sp.id])
    assert widget.spec_combo.count() == 1


def test_auto_find_peaks_populates_field(qtbot, tmp_path):
    library = SpectrumLibrary()
    sp = _synthetic_query_spectrum()
    library.add(sp)
    widget = RruffMatchWorkspace(library=library, cache_dir=str(tmp_path))
    qtbot.addWidget(widget)
    widget.set_spectra([sp.id])

    widget.auto_find_peaks()
    text = widget.peaks_edit.text()
    assert text != ""
    peaks = [float(p) for p in text.split(",")]
    assert any(abs(p - 464.0) < 15 for p in peaks)
    assert any(abs(p - 1085.0) < 15 for p in peaks)


def test_find_matches_without_peaks_warns_and_does_not_crash(qtbot, tmp_path):
    widget = RruffMatchWorkspace(cache_dir=str(tmp_path))
    qtbot.addWidget(widget)
    widget.find_matches()  # no peaks entered -> QMessageBox.warning, neutralized by conftest fixture
    assert widget.results_table.rowCount() == 0


def test_find_matches_missing_cache_updates_status(qtbot, tmp_path):
    widget = RruffMatchWorkspace(cache_dir=str(tmp_path / "does_not_exist"))
    qtbot.addWidget(widget)
    widget.peaks_edit.setText("464.0, 1085.0")
    widget.find_matches()
    assert "No RRUFF database found" in widget.db_status_label.text()


def test_download_database_button_flow(qtbot, tmp_path, monkeypatch):
    """No-Python-needed setup (user request): clicking the button disables
    both download buttons, drains the worker's progress log into the status
    label, and reloads the index once the (mocked) download finishes."""
    widget = RruffMatchWorkspace(cache_dir=str(tmp_path))
    qtbot.addWidget(widget)

    calls = []

    def fake_download(log=None):
        calls.append("called")
        log("downloading excellent_oriented.zip...")
        _write_fake_cache(tmp_path, tmp_path / "raw")  # the mock "builds" the cache
        return 3

    import rruff_science as rs
    monkeypatch.setattr(rs, "download_and_build_rruff_cache", fake_download)

    widget.download_database()  # QMessageBox.question -> Yes via conftest's autouse fixture

    assert calls == ["called"]
    assert widget.download_db_btn.isEnabled()
    assert widget.download_db_btn.text() == "Download RRUFF database…"
    assert widget.download_amcsd_btn.isEnabled()
    assert "3" in widget.db_status_label.text()
    assert widget._index is not None  # reloaded after the download


def test_download_database_declined_does_not_download(qtbot, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.No))
    widget = RruffMatchWorkspace(cache_dir=str(tmp_path))
    qtbot.addWidget(widget)

    calls = []
    import rruff_science as rs
    monkeypatch.setattr(rs, "download_and_build_rruff_cache", lambda **k: calls.append(1))
    widget.download_database()
    assert calls == []


def test_download_database_error_reenables_buttons(qtbot, tmp_path, monkeypatch):
    widget = RruffMatchWorkspace(cache_dir=str(tmp_path))
    qtbot.addWidget(widget)

    import rruff_science as rs
    monkeypatch.setattr(rs, "download_and_build_rruff_cache",
                        lambda **k: (_ for _ in ()).throw(RuntimeError("no internet")))
    widget.download_database()  # QMessageBox.critical is neutralized by conftest too

    assert widget.download_db_btn.isEnabled()
    assert widget.download_amcsd_btn.isEnabled()


def test_download_amcsd_button_flow(qtbot, tmp_path, monkeypatch):
    widget = RruffMatchWorkspace(cache_dir=str(tmp_path))
    qtbot.addWidget(widget)

    import rruff_science as rs
    monkeypatch.setattr(rs, "download_and_build_amcsd_cache", lambda **k: 7)
    widget.download_amcsd()

    assert widget.download_amcsd_btn.isEnabled()
    assert "7" in widget.db_status_label.text()


def test_accept_keeps_overlay_state_and_clear_resets(qtbot, tmp_path):
    """QualX-style session state (same as XRD ID): accepted phases stay
    overlaid, their peaks gray out; Clear starts over."""
    cache_dir = tmp_path / "rruff_cache"
    _write_fake_cache(cache_dir, cache_dir / "raw")
    library = SpectrumLibrary()
    sp = _synthetic_query_spectrum()
    library.add(sp)
    widget = RruffMatchWorkspace(library=library, cache_dir=str(cache_dir))
    qtbot.addWidget(widget)
    widget.set_spectra([sp.id])
    qtbot.wait(200)

    widget.peaks_edit.setText("464.0, 1085.0")
    widget.find_matches()
    qtbot.wait(20)
    widget.results_table.selectRow(0)
    widget.accept_selected_candidate()
    qtbot.wait(20)
    state = widget._state()
    assert len(state["accepted"]) == 1
    assert state["explained"]

    widget.clear_session()
    qtbot.wait(250)
    assert widget._state() == {"accepted": [], "explained": []}
    assert widget.peaks_edit.text() == ""
    assert widget.results_table.rowCount() == 0


def test_zoom_preserved_across_candidate_renders(qtbot, tmp_path):
    """Same zoom-preservation rule as XRD ID."""
    cache_dir = tmp_path / "rruff_cache"
    _write_fake_cache(cache_dir, cache_dir / "raw")
    library = SpectrumLibrary()
    sp = _synthetic_query_spectrum()
    library.add(sp)
    widget = RruffMatchWorkspace(library=library, cache_dir=str(cache_dir))
    qtbot.addWidget(widget)
    widget.set_spectra([sp.id])
    qtbot.wait(200)

    widget.peaks_edit.setText("464.0, 1085.0")
    widget.find_matches()
    qtbot.wait(20)
    ax = widget.plot.figure.get_axes()[0]
    widget.plot.toolbar.push_current()
    ax.set_xlim(440.0, 500.0)

    widget._render_preview(widget._selected_candidates())  # another card clicked
    qtbot.wait(20)
    assert widget.plot.figure.get_axes()[0].get_xlim() == (440.0, 500.0)


def test_entry_preview_renders_spectrum(qtbot, tmp_path):
    """Arriving on the page renders the query spectrum immediately."""
    library = SpectrumLibrary()
    sp = _synthetic_query_spectrum()
    library.add(sp)
    widget = RruffMatchWorkspace(library=library, cache_dir=str(tmp_path))
    qtbot.addWidget(widget)
    widget.set_spectra([sp.id])
    qtbot.wait(200)  # entry preview goes through the debounce
    axes = widget.plot.figure.get_axes()
    assert axes and axes[0].lines


def test_find_matches_ranks_best_candidate_first(qtbot, tmp_path):
    cache_dir = tmp_path / "rruff_cache"
    _write_fake_cache(cache_dir, cache_dir / "raw")

    library = SpectrumLibrary()
    sp = _synthetic_query_spectrum()
    library.add(sp)
    widget = RruffMatchWorkspace(library=library, cache_dir=str(cache_dir))
    qtbot.addWidget(widget)
    widget.set_spectra([sp.id])

    widget.peaks_edit.setText("464.0, 1085.0")
    widget.tolerance_edit.setText("10.0")
    widget.find_matches()
    qtbot.wait(20)  # let the preview's deferred canvas.draw_idle() complete before teardown

    assert widget.results_table.rowCount() == 2  # NoPeaksMineral excluded
    assert widget.results_table.item(0, 0).text() == "Quartz"
    assert widget.results_table.item(0, 3).text() == "2"  # matched
    assert "3 spectra" in widget.db_status_label.text() or "database" in widget.db_status_label.text().lower()


def test_accept_selected_candidate_writes_spectrum_metadata(qtbot, tmp_path):
    cache_dir = tmp_path / "rruff_cache"
    _write_fake_cache(cache_dir, cache_dir / "raw")

    library = SpectrumLibrary()
    sp = _synthetic_query_spectrum()
    library.add(sp)
    widget = RruffMatchWorkspace(library=library, cache_dir=str(cache_dir))
    qtbot.addWidget(widget)
    widget.set_spectra([sp.id])
    widget.peaks_edit.setText("464.0, 1085.0")
    widget.find_matches()
    qtbot.wait(20)

    widget.accept_selected_candidate()
    qtbot.wait(20)  # accept re-renders the preview; let draw_idle settle
    assert sp.meta["rruff_match"]["mineral"] == "Quartz"
    assert sp.meta["rruff_match"]["rruff_id"] == "R040031"
    assert sp.meta["rruff_match"]["wavelength_nm"] == 532.0


def test_accept_fires_on_accept_callback_with_previous_state(qtbot, tmp_path):
    """The shell hooks on_accept to the Library undo stack — it must receive
    the spectrum id and the full identification state BEFORE this accept
    (both the latest match and the accepted-phases list)."""
    cache_dir = tmp_path / "rruff_cache"
    _write_fake_cache(cache_dir, cache_dir / "raw")

    library = SpectrumLibrary()
    sp = _synthetic_query_spectrum()
    old_match = {"mineral": "Calcite", "rruff_id": "R040070"}
    sp.meta["rruff_match"] = dict(old_match)
    library.add(sp)
    calls = []
    widget = RruffMatchWorkspace(library=library, cache_dir=str(cache_dir),
                                 on_accept=lambda sid, prev: calls.append((sid, prev)))
    qtbot.addWidget(widget)
    widget.set_spectra([sp.id])
    widget.peaks_edit.setText("464.0, 1085.0")
    widget.find_matches()
    qtbot.wait(20)

    widget.accept_selected_candidate()
    qtbot.wait(20)  # accept re-renders the preview; let draw_idle settle
    assert calls == [(sp.id, {"rruff_match": old_match, "rruff_matches": None})]
    assert sp.meta["rruff_match"]["mineral"] == "Quartz"
    assert [m["mineral"] for m in sp.meta["rruff_matches"]] == ["Quartz"]


def test_accept_subtracts_matched_peaks_and_researches_remainder(qtbot, tmp_path):
    """User request: accept = keep the candidate as an identified phase and
    look for matches on the REMAINING peaks (mixture identification)."""
    cache_dir = tmp_path / "rruff_cache"
    _write_fake_cache(cache_dir, cache_dir / "raw")

    library = SpectrumLibrary()
    sp = _synthetic_query_spectrum()
    library.add(sp)
    widget = RruffMatchWorkspace(library=library, cache_dir=str(cache_dir))
    qtbot.addWidget(widget)
    widget.set_spectra([sp.id])
    # 464 + 1085 belong to Quartz; 700 is an extra unexplained peak
    widget.peaks_edit.setText("464.0, 1085.0, 700.0")
    widget.find_matches()
    qtbot.wait(20)
    assert widget.results_table.item(0, 0).text() == "Quartz"
    widget.results_table.selectRow(0)

    widget.accept_selected_candidate()
    qtbot.wait(20)
    # Quartz recorded; its peaks removed; the field now holds the remainder
    assert [m["mineral"] for m in sp.meta["rruff_matches"]] == ["Quartz"]
    assert widget.peaks_edit.text() == "700.0"
    # the re-search excludes accepted Quartz, and nothing matches 700 alone
    assert all(widget.results_table.item(r, 0).text() != "Quartz"
               for r in range(widget.results_table.rowCount()))

    # a second accept in a fully-explained scenario clears the query
    widget.peaks_edit.setText("1085.0")
    widget.find_matches()
    qtbot.wait(20)
    assert widget.results_table.item(0, 0).text() == "Calcite"  # Quartz excluded now
    widget.results_table.selectRow(0)
    widget.accept_selected_candidate()
    qtbot.wait(20)
    assert [m["mineral"] for m in sp.meta["rruff_matches"]] == ["Quartz", "Calcite"]
    assert widget.peaks_edit.text() == ""


def test_shift_click_multi_selection_overlays_candidates(qtbot, tmp_path):
    cache_dir = tmp_path / "rruff_cache"
    _write_fake_cache(cache_dir, cache_dir / "raw")

    library = SpectrumLibrary()
    sp = _synthetic_query_spectrum()
    library.add(sp)
    widget = RruffMatchWorkspace(library=library, cache_dir=str(cache_dir))
    qtbot.addWidget(widget)
    widget.set_spectra([sp.id])
    widget.peaks_edit.setText("464.0, 1085.0")
    widget.find_matches()
    qtbot.wait(20)
    assert widget.results_table.rowCount() == 2

    from PySide6.QtWidgets import QTableWidget
    assert widget.results_table.selectionMode() == QTableWidget.ExtendedSelection
    widget.results_table.selectAll()  # both candidates selected
    qtbot.wait(50)

    ax = widget.plot.figure.get_axes()[0]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("Quartz" in lbl for lbl in labels)
    assert any("Calcite" in lbl for lbl in labels)


def test_render_preview_overlays_measured_spectrum(qtbot, tmp_path):
    cache_dir = tmp_path / "rruff_cache"
    _write_fake_cache(cache_dir, cache_dir / "raw")

    library = SpectrumLibrary()
    sp = _synthetic_query_spectrum()
    library.add(sp)
    widget = RruffMatchWorkspace(library=library, cache_dir=str(cache_dir))
    qtbot.addWidget(widget)
    widget.set_spectra([sp.id])
    widget.peaks_edit.setText("464.0, 1085.0")
    widget.find_matches()
    qtbot.wait(20)

    ax = widget.plot.figure.get_axes()[0]
    assert len(ax.lines) >= 2  # query + candidate measured spectrum


_MINIMAL_CIF = """\
data_global
_chemical_name_mineral 'Quartz'
_cell_length_a 4.913
_cell_length_b 4.913
_cell_length_c 5.405
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 120
"""


def test_send_candidate_cifs_hands_structures_to_callback(qtbot, tmp_path):
    """RRUFF→CIF handoff: with a local AMCSD cache containing the matched
    mineral, the button must resolve its CIF(s) and pass them to the
    shell-provided callback."""
    import zipfile

    import rruff_science as rs

    amcsd_zip = tmp_path / "cif.zip"
    with zipfile.ZipFile(amcsd_zip, "w") as zf:
        zf.writestr("Quartz__0000789.cif", _MINIMAL_CIF)
    amcsd_cache = tmp_path / "amcsd"
    rs.ingest_amcsd_cif_zip(str(amcsd_zip), cache_dir=str(amcsd_cache))

    rruff_cache = tmp_path / "rruff_cache"
    _write_fake_cache(rruff_cache, rruff_cache / "raw")

    library = SpectrumLibrary()
    sp = _synthetic_query_spectrum()
    library.add(sp)

    received = {}
    widget = RruffMatchWorkspace(
        library=library, cache_dir=str(rruff_cache), amcsd_cache_dir=str(amcsd_cache),
        on_send_cifs=lambda paths: received.update(paths=list(paths)),
    )
    qtbot.addWidget(widget)
    widget.set_spectra([sp.id])
    widget.peaks_edit.setText("464.0, 1085.0")
    widget.find_matches()
    qtbot.wait(20)
    assert widget.results_table.item(0, 0).text() == "Quartz"

    widget.send_candidate_cifs()
    assert "paths" in received
    assert len(received["paths"]) == 1
    assert received["paths"][0].endswith(".cif")


def test_send_candidate_cifs_without_cache_informs_user(qtbot, tmp_path):
    rruff_cache = tmp_path / "rruff_cache"
    _write_fake_cache(rruff_cache, rruff_cache / "raw")
    library = SpectrumLibrary()
    sp = _synthetic_query_spectrum()
    library.add(sp)
    widget = RruffMatchWorkspace(
        library=library, cache_dir=str(rruff_cache),
        amcsd_cache_dir=str(tmp_path / "no_amcsd"),
        on_send_cifs=lambda paths: None,
    )
    qtbot.addWidget(widget)
    widget.set_spectra([sp.id])
    widget.peaks_edit.setText("464.0")
    widget.find_matches()
    qtbot.wait(20)
    widget.send_candidate_cifs()  # info dialog neutralized by conftest fixture; no crash


def test_shell_rruff_send_cifs_adds_to_raman_overlay(qtbot, tmp_path, monkeypatch):
    """Full handoff through the real shell: the callback adds the CIF to
    the Raman workspace's overlay series and switches nav to it."""
    cif_path = tmp_path / "Quartz__0000789.cif"
    cif_path.write_text(_MINIMAL_CIF, encoding="utf-8")

    window = PrismMainWindow()
    qtbot.addWidget(window)
    qtbot.wait(20)

    window._on_rruff_send_cifs([str(cif_path)])
    qtbot.wait(200)  # debounced render

    assert len(window.raman_page.cif_series) == 1
    assert window.raman_page.cif_series[0]["visible"] is True
    assert window.stack.currentWidget() is window.raman_page


def test_shell_rruff_page_picks_up_library_records(qtbot, raman_example_path):
    from core.qt_shell import _load_spectrum_from_path

    window = PrismMainWindow()
    qtbot.addWidget(window)
    spectrum = _load_spectrum_from_path(str(raman_example_path))
    window.library.add(spectrum)

    from core.qt_shell import NAV_ITEMS
    window.nav.setCurrentRow(NAV_ITEMS.index("Raman ID"))
    qtbot.wait(20)

    assert window.rruff_page.spec_combo.count() == 1


def test_wavelength_filter_restricts_candidates(qtbot, tmp_path):
    """User request: choose the laser wavelength (and orientation/scan-type/
    quality) instead of matching against the whole corpus."""
    cache_dir = tmp_path / "rruff_cache"
    _write_fake_cache(cache_dir, cache_dir / "raw")

    library = SpectrumLibrary()
    sp = _synthetic_query_spectrum()
    library.add(sp)
    widget = RruffMatchWorkspace(library=library, cache_dir=str(cache_dir))
    qtbot.addWidget(widget)
    widget.set_spectra([sp.id])
    widget.peaks_edit.setText("464.0, 1085.0")

    # Unfiltered: Quartz (532) and Calcite (785) both match
    widget.find_matches()
    qtbot.wait(20)
    assert widget.results_table.rowCount() == 2
    # The λ combo filled itself from the loaded index (Any + 3 wavelengths)
    assert widget.wavelength_combo.count() == 4

    # λ = 785 keeps only Calcite
    widget.wavelength_combo.setCurrentIndex(widget.wavelength_combo.findData(785.0))
    widget.find_matches()
    qtbot.wait(20)
    assert widget.results_table.rowCount() == 1
    assert widget.results_table.item(0, 0).text() == "Calcite"
    assert "Filters keep" in widget.db_status_label.text()

    # quality = excellent keeps only Quartz (dialog-free: back to Any λ first)
    widget.wavelength_combo.setCurrentIndex(0)
    widget.quality_combo.setCurrentIndex(widget.quality_combo.findData("excellent"))
    widget.find_matches()
    qtbot.wait(20)
    assert widget.results_table.rowCount() == 1
    assert widget.results_table.item(0, 0).text() == "Quartz"
