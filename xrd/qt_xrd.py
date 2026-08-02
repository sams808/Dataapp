"""
qt_xrd.py — the XRD phase-identification workspace: a QualX-style
search-match (xrd_id_science engine) over the user's OWN registered card
databases, plus a reference-card browser and the Raman↔XRD bridge.

PRISM ships no reference data: any QualX-format or PRISM-format .sq the
user has the rights to can be registered ("Add database…" / "Add folder…"),
several can be enabled at once, and every search probes all enabled ones.

Same philosophy as Raman ID: candidates are RANKED with their figure of
merit, sources, and codes — identification only happens on an explicit
Accept, which works iteratively for mixtures (matched peaks removed, the
remaining ones re-searched), mirroring the Raman workflow.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMessageBox, QPushButton, QSplitter,
    QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

import xrd.xrd_id_science as xid
from core.fitting_science import find_peak_candidates
from core.qt_models import SpectrumLibrary
from core.qt_widgets import PlotWidget, to_float as _to_float

RESULT_COLUMNS = ["FoM %", "Mineral / name", "Formula", "Source", "Code", "Q", "Space group", "Matched"]
CARD_COLORS = ["crimson", "royalblue", "seagreen", "darkorange", "purple", "teal"]
# Muted palette for phases already accepted (they stay overlaid, QualX-style)
ACCEPTED_COLORS = ["#9c6b74", "#6b7d9c", "#6b9c7d", "#9c8a6b", "#856b9c", "#6b969c"]


def _parse_elements(text: str) -> List[str]:
    return [t.strip() for t in (text or "").replace(";", ",").split(",") if t.strip()]


class XrdIdWorkspace(QWidget):
    def __init__(self, parent: Optional[QWidget] = None, library: Optional[SpectrumLibrary] = None,
                 db_path: Optional[str] = None, on_accept=None):
        super().__init__(parent)
        self.library = library if library is not None else SpectrumLibrary()
        # Database registry: with an explicit db_path (tests / embedding) the
        # workspace runs on that single file and never touches the on-disk
        # registry; otherwise the JSON registry drives the list and toggles
        # persist there.
        if db_path:
            self._registry_backed = False
            self._db_entries: List[Dict[str, Any]] = [
                {"name": os.path.splitext(os.path.basename(db_path))[0], "path": db_path, "enabled": True}]
        else:
            self._registry_backed = True
            self._db_entries = xid.load_registry()
        # Shell callback (spectrum_id, previous_state) — accepted phases join
        # the Library undo stack, same as Raman ID.
        self.on_accept = on_accept
        self._results: List[xid.XrdMatch] = []
        self._query_peaks: List[float] = []
        self._query_int: List[float] = []
        # Per-spectrum session state (QualX-style): accepted phases stay
        # overlaid across iterative searches, and the query peaks they
        # explain turn gray instead of vanishing from the plot.
        self._session: Dict[str, Dict[str, Any]] = {}
        self._build_ui()
        self._refresh_db_status()

    def _state(self) -> Dict[str, Any]:
        sid = self.spec_combo.currentData() or "__none__"
        return self._session.setdefault(sid, {"accepted": [], "explained": []})

    def _enabled_paths(self) -> List[str]:
        return [e["path"] for e in self._db_entries
                if e.get("enabled", True) and os.path.isfile(e.get("path", ""))]

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        splitter = QSplitter()
        root.addWidget(splitter)

        left = QWidget()
        left.setObjectName("Card")
        left.setMaximumWidth(340)
        left_layout = QVBoxLayout(left)

        left_layout.addWidget(QLabel("Query pattern"))
        self.spec_combo = QComboBox()
        self.spec_combo.currentIndexChanged.connect(
            lambda _=None: self.plot.request_redraw(self.render_preview))
        left_layout.addWidget(self.spec_combo)

        auto_row = QHBoxLayout()
        auto_btn = QPushButton("Auto-find peaks")
        auto_btn.clicked.connect(self.auto_find_peaks)
        auto_row.addWidget(auto_btn)
        auto_row.addWidget(QLabel("limit ×σ"))
        self.detection_edit = QLineEdit("3.0")
        self.detection_edit.setMaximumWidth(45)
        auto_row.addWidget(self.detection_edit)
        left_layout.addLayout(auto_row)

        self.pick_peaks_btn = QPushButton("Pick peaks on plot")
        self.pick_peaks_btn.setCheckable(True)
        self.pick_peaks_btn.setToolTip("Toggle, then click peaks on the Phase ID preview — each click appends a 2θ value.")
        self.pick_peaks_btn.toggled.connect(self._on_pick_peaks_toggled)
        left_layout.addWidget(self.pick_peaks_btn)

        left_layout.addWidget(QLabel("Peaks 2θ (comma-sep — editable)"))
        self.peaks_edit = QLineEdit()
        left_layout.addWidget(self.peaks_edit)

        row = QHBoxLayout()
        row.addWidget(QLabel("λ (Å)"))
        self.wavelength_edit = QLineEdit(f"{xid.CU_KA1}")
        self.wavelength_edit.setMaximumWidth(60)
        row.addWidget(self.wavelength_edit)
        row.addWidget(QLabel("tol (°2θ)"))
        self.tol_edit = QLineEdit("0.2")
        self.tol_edit.setMaximumWidth(45)
        row.addWidget(self.tol_edit)
        left_layout.addLayout(row)

        left_layout.addWidget(QLabel("Chemistry filter"))
        el_row = QHBoxLayout()
        el_row.addWidget(QLabel("contains all:"))
        self.elements_all_edit = QLineEdit()
        self.elements_all_edit.setPlaceholderText("e.g. Si, O")
        el_row.addWidget(self.elements_all_edit)
        left_layout.addLayout(el_row)
        el_row2 = QHBoxLayout()
        el_row2.addWidget(QLabel("excludes:"))
        self.elements_none_edit = QLineEdit()
        self.elements_none_edit.setPlaceholderText("e.g. Pb")
        el_row2.addWidget(self.elements_none_edit)
        left_layout.addLayout(el_row2)

        left_layout.addWidget(QLabel("Card filters"))
        from core.qt_widgets import CheckComboBox
        filt_row = QHBoxLayout()
        self.system_filter = CheckComboBox("systems")
        self.system_filter.set_items(xid.CRYSTAL_SYSTEMS)
        filt_row.addWidget(self.system_filter, 1)
        self.quality_filter = CheckComboBox("qualities")
        filt_row.addWidget(self.quality_filter, 1)
        left_layout.addLayout(filt_row)
        self.sg_filter_edit = QLineEdit()
        self.sg_filter_edit.setPlaceholderText("space group contains… (e.g. Pnma)")
        left_layout.addWidget(self.sg_filter_edit)

        self.source_checks: Dict[str, QCheckBox] = {}
        self.sources_holder = QHBoxLayout()
        left_layout.addLayout(self.sources_holder)

        self._search_btn = QPushButton("Search match")
        self._search_btn.setObjectName("Primary")
        self._search_btn.clicked.connect(self.run_search)
        left_layout.addWidget(self._search_btn)

        accept_btn = QPushButton("Accept selected phase")
        accept_btn.clicked.connect(self.accept_selected)
        left_layout.addWidget(accept_btn)

        clear_row = QHBoxLayout()
        update_btn = QPushButton("Update plot")
        update_btn.clicked.connect(lambda: self.plot.request_redraw(self.render_preview))
        clear_row.addWidget(update_btn)
        clear_btn = QPushButton("Clear")
        clear_btn.setToolTip("Start over on this pattern: clears peaks, results, and the accepted-phase overlays (accepted identifications stay recorded on the spectrum; Ctrl+Z in the Library to undo those).")
        clear_btn.clicked.connect(self.clear_session)
        clear_row.addWidget(clear_btn)
        left_layout.addLayout(clear_row)

        self.raman_link_btn = QPushButton("Check Raman-identified phases here")
        self.raman_link_btn.setToolTip(
            "For each phase accepted in Mineral ID on this spectrum's sample, look the mineral up in "
            "the XRD database and overlay its reference lines on the query pattern."
        )
        self.raman_link_btn.clicked.connect(self.check_raman_phases)
        left_layout.addWidget(self.raman_link_btn)

        left_layout.addWidget(QLabel("Databases (check = probe)"))
        self.db_list = QListWidget()
        self.db_list.setMaximumHeight(110)
        self.db_list.setToolTip(
            "Register any card database you have the rights to use: a QualX-format .sq is "
            "indexed once locally; a PRISM-format .sq is used in place. Check several to "
            "probe them all in one search."
        )
        self.db_list.itemChanged.connect(self._on_db_toggled)
        left_layout.addWidget(self.db_list)
        db_btn_row = QHBoxLayout()
        self.add_db_btn = QPushButton("Add database…")
        self.add_db_btn.clicked.connect(self.add_database)
        db_btn_row.addWidget(self.add_db_btn)
        self.add_folder_btn = QPushButton("Add folder…")
        self.add_folder_btn.clicked.connect(self.add_database_folder)
        db_btn_row.addWidget(self.add_folder_btn)
        self.remove_db_btn = QPushButton("Remove")
        self.remove_db_btn.setToolTip("Unregister the selected database (the .sq file itself is not deleted).")
        self.remove_db_btn.clicked.connect(self.remove_database)
        db_btn_row.addWidget(self.remove_db_btn)
        left_layout.addLayout(db_btn_row)

        self.db_status_label = QLabel("")
        self.db_status_label.setWordWrap(True)
        left_layout.addWidget(self.db_status_label)
        left_layout.addStretch(1)
        splitter.addWidget(left)

        right = QWidget()
        right.setObjectName("Card")
        right_layout = QVBoxLayout(right)
        self.tabs = QTabWidget()

        id_tab = QWidget()
        idl = QVBoxLayout(id_tab)
        self.results_table = QTableWidget(0, len(RESULT_COLUMNS))
        self.results_table.setHorizontalHeaderLabels(RESULT_COLUMNS)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.results_table.itemSelectionChanged.connect(self._on_result_selected)
        idl.addWidget(self.results_table, 1)
        self.plot = PlotWidget(figsize=(7, 4))
        idl.addWidget(self.plot, 1)
        self.tabs.addTab(id_tab, "Phase ID")

        browse_tab = QWidget()
        bl = QVBoxLayout(browse_tab)
        browse_row = QHBoxLayout()
        browse_row.addWidget(QLabel("Name / mineral / formula"))
        self.browse_edit = QLineEdit()
        self.browse_edit.returnPressed.connect(self.browse_cards)
        browse_row.addWidget(self.browse_edit, 1)
        browse_btn = QPushButton("Look up")
        browse_btn.clicked.connect(self.browse_cards)
        browse_row.addWidget(browse_btn)
        bl.addLayout(browse_row)
        self.browse_list = QListWidget()
        self.browse_list.itemSelectionChanged.connect(lambda: self.browse_plot.request_redraw(self.render_browse))
        bl.addWidget(self.browse_list, 1)
        self.browse_plot = PlotWidget(figsize=(7, 4))
        bl.addWidget(self.browse_plot, 1)
        self.tabs.addTab(browse_tab, "Card browser")

        right_layout.addWidget(self.tabs)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        self._browse_hits: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Database management
    # ------------------------------------------------------------------
    def _summary_cached(self, path: str):
        """database_summary scans the whole cards table — cache per file
        state so nav switches into this page stay instant."""
        try:
            st = os.stat(path)
            key = (path, st.st_mtime_ns, st.st_size)
        except OSError:
            return None
        cache = getattr(self, "_summary_cache", None)
        if cache is None:
            cache = self._summary_cache = {}
        if key not in cache:
            cache[key] = xid.database_summary(path)
        return cache[key]

    def _refresh_db_status(self) -> None:
        self.db_list.blockSignals(True)
        self.db_list.clear()
        total = 0
        by_source: Dict[str, int] = {}
        for e in self._db_entries:
            item = QListWidgetItem(e.get("name", "?"))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            missing = not os.path.isfile(e.get("path", ""))
            item.setCheckState(Qt.Checked if e.get("enabled", True) and not missing else Qt.Unchecked)
            tip = e.get("path", "")
            if missing:
                item.setText(f"{e.get('name', '?')} (file missing)")
                tip += "\nThe .sq file is missing — reconnect the drive or Remove the entry."
            summary = self._summary_cached(e["path"]) if not missing else None
            if summary is not None:
                tip += f"\n{summary['total_cards']} cards"
                if e.get("enabled", True):
                    total += summary["total_cards"]
                    for k, v in summary["by_source"].items():
                        by_source[k] = by_source.get(k, 0) + v
            item.setToolTip(tip)
            self.db_list.addItem(item)
        self.db_list.blockSignals(False)

        if not self._db_entries:
            self.db_status_label.setText(
                "No card database registered yet. Download any QualX-format .sq database "
                "you have the rights to use, then click 'Add database…'."
            )
        else:
            parts = ", ".join(f"{k}: {v}" for k, v in sorted(by_source.items()))
            self.db_status_label.setText(f"Enabled databases: {total} cards ({parts}).")

        # Source-tag checkboxes track the ENABLED databases' contents only:
        # tags from disabled/removed databases disappear (nobody sees tag
        # names their own files don't carry), new ones appear checked.
        for tag in list(self.source_checks):
            if tag not in by_source:
                cb = self.source_checks.pop(tag)
                self.sources_holder.removeWidget(cb)
                cb.deleteLater()
        for tag in sorted(by_source):
            if tag not in self.source_checks:
                cb = QCheckBox(tag)
                cb.setChecked(True)
                self.source_checks[tag] = cb
                self.sources_holder.addWidget(cb)

        # Quality filter entries = the quality codes actually present in
        # the enabled databases (check states preserved across refreshes).
        qualities: set = set()
        for e in self._db_entries:
            if e.get("enabled", True) and os.path.isfile(e.get("path", "")):
                summary = self._summary_cached(e["path"])
                qualities.update((summary or {}).get("qualities", []))
        self.quality_filter.set_items(sorted(qualities))

    def _on_db_toggled(self, item: QListWidgetItem) -> None:
        row = self.db_list.row(item)
        if not (0 <= row < len(self._db_entries)):
            return
        entry = self._db_entries[row]
        entry["enabled"] = item.checkState() == Qt.Checked
        if self._registry_backed:
            xid.set_database_enabled(entry["name"], entry["enabled"])
        # deferred: rebuilding the list from inside its own itemChanged
        # handler would delete the item mid-signal
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, self._refresh_db_status)

    def add_database(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Add card database (.sq)", "",
            "Card databases (*.sq);;All files (*.*)")
        if path:
            self._register_paths([path])

    def add_database_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Add every .sq database in a folder")
        if folder:
            self._register_paths([folder], folder_mode=True)

    def _register_paths(self, paths: List[str], folder_mode: bool = False) -> None:
        """Register in a worker thread — indexing a large QualX-format .sq
        takes minutes and must not freeze the UI."""
        self.add_db_btn.setEnabled(False)
        self.add_folder_btn.setEnabled(False)
        self.db_status_label.setText(
            "Registering… a QualX-format database is indexed once, which can take a few "
            "minutes for hundreds of thousands of cards. PRISM stays usable meanwhile."
        )

        registry_backed = self._registry_backed
        base_entries = list(self._db_entries)  # snapshot — the worker fn must not touch self

        def work():
            added = []
            if registry_backed:
                for p in paths:
                    if folder_mode:
                        added += xid.register_folder(p, log=lambda *a: None)
                    else:
                        added.append(xid.register_database(p, log=lambda *a: None))
                return xid.load_registry(), added
            # single-db (embedded/test) mode: keep everything in memory
            entries = base_entries
            for p in paths:
                fmt = xid.sniff_sq_format(p)
                if fmt is None:
                    raise ValueError(f"{os.path.basename(p)} is not a recognizable card database.")
                entry = {"name": os.path.splitext(os.path.basename(p))[0], "path": p, "enabled": True}
                entries.append(entry)
                added.append(entry)
            return entries, added

        from core.qt_worker import run_in_thread
        run_in_thread(work, self._on_register_done, self._on_register_error)

    def _on_register_error(self, traceback_text: str) -> None:
        self.add_db_btn.setEnabled(True)
        self.add_folder_btn.setEnabled(True)
        self._refresh_db_status()
        QMessageBox.critical(self, "Add database", traceback_text)

    def _on_register_done(self, result) -> None:
        entries, added = result
        self.add_db_btn.setEnabled(True)
        self.add_folder_btn.setEnabled(True)
        self._db_entries = entries
        self._refresh_db_status()
        if added:
            names = ", ".join(e["name"] for e in added if e)
            self.db_status_label.setText(self.db_status_label.text() + f" Added: {names}.")
        else:
            QMessageBox.information(self, "Add database", "No new database found (already registered, or no .sq files).")

    def remove_database(self) -> None:
        row = self.db_list.currentRow()
        if not (0 <= row < len(self._db_entries)):
            QMessageBox.information(self, "Remove database", "Select a database in the list first.")
            return
        entry = self._db_entries[row]
        if self._registry_backed:
            xid.unregister_database(entry["name"])
            self._db_entries = xid.load_registry()
        else:
            self._db_entries = [e for e in self._db_entries if e is not entry]
        self._refresh_db_status()

    def set_spectra(self, spectrum_ids: List[str]) -> None:
        current = self.spec_combo.currentData()
        self.spec_combo.blockSignals(True)
        self.spec_combo.clear()
        for sid in spectrum_ids:
            sp = self.library.get(sid)
            if sp is not None:
                self.spec_combo.addItem(sp.title, sid)
        if current is not None:
            idx = self.spec_combo.findData(current)
            if idx >= 0:
                self.spec_combo.setCurrentIndex(idx)
        self.spec_combo.blockSignals(False)
        self._refresh_db_status()
        # Arriving on the page must show the diffractogram immediately, not
        # an empty axes (user request — same rule on every workspace).
        self.plot.request_redraw(self.render_preview)

    def _current_spectrum(self):
        sid = self.spec_combo.currentData()
        return self.library.get(sid) if sid else None

    # ------------------------------------------------------------------
    def auto_find_peaks(self) -> None:
        sp = self._current_spectrum()
        if sp is None:
            QMessageBox.warning(self, "Auto-find peaks", "Select a query pattern first.")
            return
        limit = _to_float(self.detection_edit.text(), 3.0) or 3.0
        centers = find_peak_candidates(sp.x, sp.y, max_peaks=25, min_prominence_sigma=limit)
        if not centers:
            QMessageBox.information(self, "Auto-find peaks", "No clear peaks — lower the detection limit.")
            return
        self.peaks_edit.setText(", ".join(f"{c:.3f}" for c in sorted(centers)))

    def _on_pick_peaks_toggled(self, checked: bool) -> None:
        if checked:
            mode = str(self.plot.toolbar.mode)
            if "zoom" in mode:
                self.plot.toolbar.zoom()
            elif "pan" in mode:
                self.plot.toolbar.pan()
            self.plot.canvas.setCursor(Qt.CrossCursor)
            self._pick_cid = self.plot.canvas.mpl_connect("button_press_event", self._on_pick_peak_click)
            self.plot.request_redraw(self.render_preview)
        else:
            if getattr(self, "_pick_cid", None) is not None:
                self.plot.canvas.mpl_disconnect(self._pick_cid)
                self._pick_cid = None
            self.plot.canvas.unsetCursor()

    def _on_pick_peak_click(self, event) -> None:
        if event.inaxes is None or event.xdata is None or self.plot.toolbar.mode:
            return
        cur = self.peaks_edit.text().strip()
        xv = f"{float(event.xdata):.3f}"
        self.peaks_edit.setText(f"{cur}, {xv}" if cur else xv)

    def _parse_peaks(self):
        vals = []
        for tok in (self.peaks_edit.text() or "").split(","):
            v = _to_float(tok)
            if v is not None:
                vals.append(v)
        return vals

    def _peak_intensities(self, peaks: List[float]) -> List[float]:
        sp = self._current_spectrum()
        if sp is None:
            return [1.0] * len(peaks)
        x, y = np.asarray(sp.x, float), np.asarray(sp.y, float)
        out = []
        for p in peaks:
            near = np.abs(x - p) <= max(0.15, float(np.mean(np.diff(x))) * 3)
            out.append(float(np.nanmax(y[near])) if np.any(near) else 1.0)
        floor = min(out) if out else 0.0
        return [max(v - floor, 1.0) for v in out]

    def run_search(self) -> None:
        peaks = self._parse_peaks()
        if not peaks:
            QMessageBox.warning(self, "Search match", "Enter or auto-find at least one 2θ peak.")
            return
        db_paths = self._enabled_paths()
        if not db_paths:
            QMessageBox.information(
                self, "Search match",
                "No card database enabled. Register one with 'Add database…' (any "
                "QualX-format .sq you have the rights to use) and check it in the list.")
            return
        self._query_peaks = peaks
        self._query_int = self._peak_intensities(peaks)
        sp = self._current_spectrum()
        tt_range = (float(np.nanmin(sp.x)), float(np.nanmax(sp.x))) if sp is not None else None
        sources = [tag for tag, cb in self.source_checks.items() if cb.isChecked()]

        self._search_btn.setEnabled(False)
        self._search_btn.setText("Searching…")
        kwargs = dict(
            wavelength=_to_float(self.wavelength_edit.text(), xid.CU_KA1),
            tol_two_theta=_to_float(self.tol_edit.text(), 0.2),
            elements_all=_parse_elements(self.elements_all_edit.text()),
            elements_none=_parse_elements(self.elements_none_edit.text()),
            sources=sources if sources and len(sources) < len(self.source_checks) else (),
            two_theta_range=tt_range, db_paths=db_paths,
            qualities=self.quality_filter.checked(),
            crystal_systems=self.system_filter.checked(),
            spacegroup_contains=self.sg_filter_edit.text().strip(),
        )
        from core.qt_worker import run_in_thread
        run_in_thread(
            lambda: xid.search_match(list(peaks), list(self._query_int), **kwargs),
            self._on_search_done, self._on_search_error,
        )

    def _on_search_error(self, traceback_text: str) -> None:
        self._search_btn.setEnabled(True)
        self._search_btn.setText("Search match")
        QMessageBox.critical(self, "Search match error", traceback_text)

    def _on_search_done(self, results: List[xid.XrdMatch]) -> None:
        self._search_btn.setEnabled(True)
        self._search_btn.setText("Search match")
        self._results = results
        self.results_table.setRowCount(len(results))
        for row, r in enumerate(results):
            display = r.mineral or r.name or r.formula
            values = [f"{100 * r.fom:.0f}", display, r.formula, r.source, r.source_code,
                      r.quality, r.spacegroup, str(r.n_matched)]
            for col, val in enumerate(values):
                self.results_table.setItem(row, col, QTableWidgetItem(str(val)))
        self.results_table.resizeColumnsToContents()
        if results:
            self.results_table.selectRow(0)
        else:
            self.plot.request_redraw(self.render_preview)

    def _selected_matches(self) -> List[xid.XrdMatch]:
        rows = sorted({i.row() for i in self.results_table.selectionModel().selectedRows()})
        return [self._results[r] for r in rows if 0 <= r < len(self._results)]

    def _on_result_selected(self) -> None:
        self.plot.request_redraw(self.render_preview)

    def render_preview(self) -> None:
        # keep the user's zoom across candidate-card re-renders (same spectrum)
        self.plot.preserve_zoom(("phase_id", self.spec_combo.currentData()))
        fig = self.plot.figure
        fig.clf()
        ax = fig.add_subplot(111)
        sp = self._current_spectrum()
        ymax = 1.0
        if sp is not None:
            y = np.asarray(sp.y, float)
            ymax = float(np.nanmax(y)) or 1.0
            ax.plot(sp.x, y / ymax * 100.0, color="black", lw=1.0, label=f"query: {sp.title}")

        # Query peak bars (QualX-style): the current (unexplained) peaks in
        # black; peaks already explained by an accepted phase turn gray.
        state = self._state()
        peaks = self._parse_peaks()
        if peaks:
            ax.vlines(peaks, 0, 104, color="black", lw=0.8, alpha=0.30,
                      label="query peaks")
        if state["explained"]:
            ax.vlines(state["explained"], 0, 104, color="gray", lw=0.8,
                      alpha=0.45, ls="--", label="explained by accepted phases")

        wavelength = _to_float(self.wavelength_edit.text(), xid.CU_KA1)
        # Accepted phases stay overlaid across iterative searches (muted).
        for k, r in enumerate(state["accepted"]):
            tt = xid.d_to_two_theta(r.d, wavelength)
            ok = np.isfinite(tt)
            label = f"accepted: {r.mineral or r.name or r.formula} [{r.source} {r.source_code}]"
            ax.vlines(tt[ok], 0, -r.i[ok], color=ACCEPTED_COLORS[k % len(ACCEPTED_COLORS)],
                      lw=1.1, alpha=0.75, label=label)
        for k, r in enumerate(self._selected_matches()):
            color = CARD_COLORS[k % len(CARD_COLORS)]
            tt = xid.d_to_two_theta(r.d, wavelength)
            ok = np.isfinite(tt)
            label = f"{r.mineral or r.name or r.formula} [{r.source} {r.source_code}] FoM {100 * r.fom:.0f}%"
            ax.vlines(tt[ok], 0, -r.i[ok], color=color, lw=1.2, label=label)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_xlabel("2θ (deg)")
        ax.set_ylabel("query (up) / card lines (down)")
        if sp is not None or peaks or state["accepted"]:
            ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
        self.plot.restore_zoom(ax)
        fig.tight_layout()
        self.plot.canvas.draw_idle()

    def clear_session(self) -> None:
        """Start over on the current pattern: peaks, results, and the
        accepted-overlay/explained-peak session state (the identifications
        recorded on the spectrum itself are untouched — undo those from
        the Library)."""
        sid = self.spec_combo.currentData() or "__none__"
        self._session.pop(sid, None)
        self.peaks_edit.setText("")
        self._results = []
        self._query_peaks = []
        self._query_int = []
        self.results_table.setRowCount(0)
        self.plot.reset_zoom_memory()  # starting over autoscales again
        self.plot.request_redraw(self.render_preview)

    # ------------------------------------------------------------------
    def accept_selected(self) -> None:
        """Iterative accept, mirroring Mineral ID: record the phase, drop
        the peaks it explains, re-search the remainder."""
        sp = self._current_spectrum()
        matches = self._selected_matches()
        if sp is None or not matches:
            QMessageBox.warning(self, "Accept phase", "Select a query pattern and a result row first.")
            return
        r = matches[0]
        previous_state = {
            "xrd_match": sp.meta.get("xrd_match"),
            "xrd_matches": list(sp.meta["xrd_matches"]) if sp.meta.get("xrd_matches") else None,
        }
        record = {
            "mineral": r.mineral, "name": r.name, "formula": r.formula,
            "source": r.source, "source_code": r.source_code, "spacegroup": r.spacegroup,
            "fom": r.fom, "n_matched": r.n_matched, "accepted_at": time.time(),
        }
        sp.meta["xrd_match"] = record
        accepted = list(sp.meta.get("xrd_matches", []))
        accepted.append(record)
        sp.meta["xrd_matches"] = accepted
        if self.on_accept is not None:
            self.on_accept(sp.id, previous_state)

        matched_q = {round(q, 4) for q, _ in r.matched_pairs}
        remaining = [p for p in self._query_peaks if round(p, 4) not in matched_q]
        # session overlays: the accepted phase's bars stay, its peaks gray out
        state = self._state()
        state["accepted"].append(r)
        state["explained"].extend(p for p in self._query_peaks if round(p, 4) in matched_q)
        phase_list = ", ".join(f"{m['mineral'] or m['name'] or m['formula']} [{m['source']} {m['source_code']}]"
                               for m in accepted)
        if remaining:
            self.peaks_edit.setText(", ".join(f"{p:.3f}" for p in sorted(remaining)))
            QMessageBox.information(
                self, "Phase accepted",
                f"Recorded '{record['mineral'] or record['name'] or record['formula']}' "
                f"[{r.source} {r.source_code}] for '{sp.title}'.\n"
                f"{len(remaining)} peak(s) remain unexplained — searching those now.\n"
                f"Accepted so far: {phase_list}",
            )
            self.run_search()
        else:
            self.peaks_edit.setText("")
            self._results = []
            self.results_table.setRowCount(0)
            self.plot.request_redraw(self.render_preview)
            QMessageBox.information(
                self, "All peaks explained",
                f"Every query peak of '{sp.title}' is explained by {len(accepted)} phase(s): {phase_list}.",
            )

    # ------------------------------------------------------------------
    # Raman ↔ XRD bridge
    # ------------------------------------------------------------------
    def check_raman_phases(self) -> None:
        """Overlay the XRD reference lines of every phase accepted in
        Mineral ID (meta['rruff_matches'] on any library spectrum sharing…
        for now: on the selected query spectrum itself or any spectrum in
        the library) against the current pattern."""
        sp = self._current_spectrum()
        if sp is None:
            QMessageBox.warning(self, "Raman link", "Select a query pattern first.")
            return
        # accepted Raman phases anywhere in the library (typically the Raman
        # spectrum of the same sample)
        minerals: List[str] = []
        for s in self.library.all():
            for m in s.meta.get("rruff_matches", []) or ([s.meta["rruff_match"]] if s.meta.get("rruff_match") else []):
                name = (m or {}).get("mineral")
                if name and name not in minerals:
                    minerals.append(name)
        if not minerals:
            QMessageBox.information(self, "Raman link",
                                    "No accepted Raman identifications found in the Library (Mineral ID → Accept).")
            return
        found: List[xid.XrdMatch] = []
        missing = []
        for name in minerals:
            hits = xid.find_cards_by_text(name, limit=3, db_paths=self._enabled_paths())
            if not hits:
                missing.append(name)
                continue
            h = hits[0]
            found.append(xid.XrdMatch(
                card_id=h["card_id"], source=h["source"], source_code=h["source_code"],
                name=h["name"], mineral=h["mineral"] or name, formula=h["formula"],
                spacegroup=h["spacegroup"], quality=h["quality"], rir=None,
                fom=0.0, cov_card=0.0, cov_query=0.0, n_matched=0, d=h["d"], i=h["i"],
            ))
        self._results = found
        self._on_search_done(found)
        if found:
            self.results_table.selectAll()
        msg = f"Overlaid XRD reference lines for {len(found)} Raman-identified phase(s): " \
              f"{', '.join(r.mineral or r.name for r in found)}."
        if missing:
            msg += f"\nNot in the XRD database: {', '.join(missing)}."
        self.db_status_label.setText(msg)

    # ------------------------------------------------------------------
    def browse_cards(self) -> None:
        """Text lookup PLUS element-set lookup: 'TiO2' also finds cards whose
        formula is written 'O2 Ti' (user-reported gap) — exact element set
        first, then contains-these-elements, then plain text; deduplicated."""
        text = self.browse_edit.text().strip()
        if not text:
            return
        db_paths = self._enabled_paths()
        hits, seen = [], set()
        for batch in (xid.find_cards_by_elements(text, mode="exact", db_paths=db_paths),
                      xid.find_cards_by_elements(text, mode="contains", limit=40, db_paths=db_paths),
                      xid.find_cards_by_text(text, limit=50, db_paths=db_paths)):
            for h in batch:
                # content-level dedup: the same card carried by several
                # databases/sources appears once (same rule as search results)
                key = xid._dedup_key(h["source_code"], h["name"], h["mineral"],
                                     h["formula"], h["spacegroup"], h["quality"])
                if key not in seen:
                    seen.add(key)
                    hits.append(h)
        self._browse_hits = hits[:100]
        self.browse_list.clear()
        for h in self._browse_hits:
            label = h["mineral"] or h["name"] or h["formula"]
            self.browse_list.addItem(f"{label}  |  {h['formula']}  [{h['source']} {h['source_code']}]  {h['spacegroup']}")
        if not self._browse_hits:
            self.browse_list.addItem("(no cards found)")

    def render_browse(self) -> None:
        self.browse_plot.preserve_zoom("browse")  # keep zoom across card clicks
        fig = self.browse_plot.figure
        fig.clf()
        ax = fig.add_subplot(111)
        rows = sorted({i.row() for i in self.browse_list.selectedIndexes()})
        wavelength = _to_float(self.wavelength_edit.text(), xid.CU_KA1)
        for k, row in enumerate(rows):
            if 0 <= row < len(self._browse_hits):
                h = self._browse_hits[row]
                tt = xid.d_to_two_theta(h["d"], wavelength)
                ok = np.isfinite(tt)
                ax.vlines(tt[ok], 0, h["i"][ok], color=CARD_COLORS[k % len(CARD_COLORS)], lw=1.2,
                          label=f"{h['mineral'] or h['name']} [{h['source']} {h['source_code']}]")
        if rows:
            ax.legend(fontsize=7)
        ax.set_xlabel("2θ (deg)")
        ax.set_ylabel("relative intensity")
        ax.grid(alpha=0.2)
        self.browse_plot.restore_zoom(ax)
        fig.tight_layout()
        self.browse_plot.canvas.draw_idle()
