"""
qt_rruff.py — RRUFF match-assist UI (M12), built on M10's local RRUFF cache
and M7's CIF-Bragg-overlay rendering pattern (reused here for candidate
peak markers).

This is an ASSIST tool, never an auto-labeler — the user's own explicit
original requirement, since RRUFF spectra span many laser excitation
wavelengths (affecting relative peak intensities/fluorescence even though
Raman shift in cm^-1 is nominally wavelength-independent): every candidate
row shows its own wavelength, and nothing is written back to a spectrum's
metadata until the user clicks "Accept as identification" for a specific,
selected candidate.

The RRUFF index (~28k spectra, ~19MB JSON) is loaded lazily on first use,
not at workspace construction — confirmed to take ~0.25s for the real
corpus, which is fine for an on-demand click but not worth paying at every
app startup regardless of whether this workspace is ever opened.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget,
)

from core.fitting_science import find_peak_candidates
from core.qt_models import SpectrumLibrary
from core.qt_widgets import PlotWidget, to_float as _to_float
from raman.rruff_science import (
    RRUFF_ATTRIBUTION_NOTE,
    RRUFF_CACHE_DIR,
    RRUFF_CITATION,
    filter_rruff_index,
    find_cifs_for_mineral,
    index_summary,
    load_index,
    parse_rruff_txt,
    rank_rruff_matches,
)

RESULT_COLUMNS = ["Mineral", "RRUFF ID", "λ (nm)", "Matched", "Match %", "Category"]


class RruffMatchWorkspace(QWidget):
    def __init__(
        self, parent: Optional[QWidget] = None, library: Optional[SpectrumLibrary] = None,
        cache_dir: Optional[str] = None, amcsd_cache_dir: Optional[str] = None,
        on_send_cifs=None, on_accept=None,
    ):
        super().__init__(parent)
        self.library = library if library is not None else SpectrumLibrary()
        # Shell-provided callback (spectrum_id, previous_match_or_None) fired
        # when an identification is accepted, so Accept joins the undo stack.
        self.on_accept = on_accept
        self.cache_dir = cache_dir or RRUFF_CACHE_DIR
        self.amcsd_cache_dir = amcsd_cache_dir  # None -> rruff_science default
        # Shell-provided callback taking a list of CIF paths (the RRUFF→CIF
        # overlay handoff into the Raman workspace).
        self.on_send_cifs = on_send_cifs
        self._index: Optional[List[Dict[str, Any]]] = None
        self._query_peaks: List[float] = []
        self._results: List[Dict[str, Any]] = []
        # Per-spectrum session state (QualX-style, same as XRD ID): accepted
        # phases stay overlaid across iterative searches, and the query
        # peaks they explain turn gray instead of vanishing.
        self._session: Dict[str, Dict[str, Any]] = {}
        # Background-download progress plumbing (queue.Queue is thread-safe;
        # the worker thread only ever puts() into it — a QTimer on the main
        # thread drains it, so no Qt widget is ever touched off-thread).
        import queue
        self._dl_queue: "queue.Queue[str]" = queue.Queue()
        self._dl_timer = QTimer(self)
        self._dl_timer.setInterval(200)
        self._dl_timer.timeout.connect(self._drain_download_log)
        self._build_ui()

    def _state(self) -> Dict[str, Any]:
        sid = self.spec_combo.currentData() or "__none__"
        return self._session.setdefault(sid, {"accepted": [], "explained": []})

    def _selected_candidates(self) -> List[Dict[str, Any]]:
        rows = self.results_table.selectionModel().selectedRows()
        return [self._results[r.row()] for r in sorted(rows, key=lambda r: r.row())
                if 0 <= r.row() < len(self._results)]

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

        left_layout.addWidget(QLabel("Query spectrum"))
        self.spec_combo = QComboBox()
        self.spec_combo.currentIndexChanged.connect(
            lambda _=None: self.plot.request_redraw(lambda: self._render_preview(self._selected_candidates())))
        left_layout.addWidget(self.spec_combo)

        auto_btn = QPushButton("Auto-find peaks from spectrum")
        auto_btn.clicked.connect(self.auto_find_peaks)
        left_layout.addWidget(auto_btn)

        left_layout.addWidget(QLabel("Peak positions (cm⁻¹, comma-separated — editable)"))
        self.peaks_edit = QLineEdit()
        left_layout.addWidget(self.peaks_edit)

        tol_row = QHBoxLayout()
        tol_row.addWidget(QLabel("Tolerance (cm⁻¹)"))
        self.tolerance_edit = QLineEdit("10.0")
        tol_row.addWidget(self.tolerance_edit)
        left_layout.addLayout(tol_row)

        # Database filters (user request): restrict candidates by laser
        # wavelength / orientation / scan type / quality before ranking.
        filters_label = QLabel("Database filters")
        filters_label.setObjectName("SectionTitle")
        left_layout.addWidget(filters_label)

        wl_row = QHBoxLayout()
        wl_row.addWidget(QLabel("Laser λ (nm)"))
        self.wavelength_combo = QComboBox()
        self.wavelength_combo.addItem("Any", None)  # real values fill in once the index loads
        wl_row.addWidget(self.wavelength_combo, 1)
        left_layout.addLayout(wl_row)

        orient_row = QHBoxLayout()
        orient_row.addWidget(QLabel("Orientation"))
        self.orientation_combo = QComboBox()
        for label, value in (("Any", None), ("Oriented", True), ("Unoriented", False)):
            self.orientation_combo.addItem(label, value)
        orient_row.addWidget(self.orientation_combo, 1)
        left_layout.addLayout(orient_row)

        scan_row = QHBoxLayout()
        scan_row.addWidget(QLabel("Scan type"))
        self.scan_type_combo = QComboBox()
        for label, value in (("Any", None), ("Raman (high-res)", "Raman"), ("Broad scan (LR survey)", "Broad_Scan")):
            self.scan_type_combo.addItem(label, value)
        scan_row.addWidget(self.scan_type_combo, 1)
        left_layout.addLayout(scan_row)

        quality_row = QHBoxLayout()
        quality_row.addWidget(QLabel("Quality"))
        self.quality_combo = QComboBox()
        for label, value in (("Any", None), ("Excellent", "excellent"), ("Fair", "fair"), ("Poor", "poor"), ("Unrated", "unrated")):
            self.quality_combo.addItem(label, value)
        quality_row.addWidget(self.quality_combo, 1)
        left_layout.addLayout(quality_row)

        find_btn = QPushButton("Find matches")
        find_btn.setObjectName("Primary")
        find_btn.clicked.connect(self.find_matches)
        left_layout.addWidget(find_btn)

        self.db_status_label = QLabel("RRUFF database not loaded yet.")
        self.db_status_label.setWordWrap(True)
        left_layout.addWidget(self.db_status_label)

        # No Python needed: downloads + indexes the RRUFF database straight
        # from rruff.net, from inside the app — the same thing the shipped
        # Download-RRUFF-database.bat/.ps1 scripts do via the CLI flag, for
        # a colleague who only has the portable exe.
        dl_row = QHBoxLayout()
        self.download_db_btn = QPushButton("Download RRUFF database…")
        self.download_db_btn.setToolTip(
            "Downloads ~400-500 MB from rruff.net and builds the local search index "
            "— no Python needed. Takes several minutes; safe to re-run if interrupted."
        )
        self.download_db_btn.clicked.connect(self.download_database)
        dl_row.addWidget(self.download_db_btn)
        self.download_amcsd_btn = QPushButton("Download AMCSD structures…")
        self.download_amcsd_btn.setToolTip(
            "Downloads ~66 MB of crystal structures, needed only for "
            "'Overlay candidate's XRD (CIF)' below."
        )
        self.download_amcsd_btn.clicked.connect(self.download_amcsd)
        dl_row.addWidget(self.download_amcsd_btn)
        left_layout.addLayout(dl_row)

        self.overlay_raw_check = QCheckBox("Overlay candidate's measured spectrum (not just peaks)")
        self.overlay_raw_check.setChecked(True)
        left_layout.addWidget(self.overlay_raw_check)

        accept_btn = QPushButton("Accept selected candidate as identification")
        accept_btn.clicked.connect(self.accept_selected_candidate)
        left_layout.addWidget(accept_btn)

        send_cif_btn = QPushButton("Overlay candidate's XRD (CIF) in Raman workspace")
        send_cif_btn.clicked.connect(self.send_candidate_cifs)
        left_layout.addWidget(send_cif_btn)

        clear_row = QHBoxLayout()
        update_btn = QPushButton("Update plot")
        update_btn.clicked.connect(
            lambda: self.plot.request_redraw(lambda: self._render_preview(self._selected_candidates())))
        clear_row.addWidget(update_btn)
        clear_btn = QPushButton("Clear")
        clear_btn.setToolTip("Start over on this spectrum: clears peaks, results, and the accepted-phase overlays (accepted identifications stay recorded; Ctrl+Z in the Library to undo those).")
        clear_btn.clicked.connect(self.clear_session)
        clear_row.addWidget(clear_btn)
        left_layout.addLayout(clear_row)

        citation_label = QLabel(f"{RRUFF_CITATION}\n\n{RRUFF_ATTRIBUTION_NOTE}")
        citation_label.setWordWrap(True)
        citation_label.setObjectName("SectionNote")
        left_layout.addWidget(citation_label)
        left_layout.addStretch(1)
        splitter.addWidget(left)

        right = QWidget()
        right.setObjectName("Card")
        right_layout = QVBoxLayout(right)

        self.results_table = QTableWidget(0, len(RESULT_COLUMNS))
        self.results_table.setHorizontalHeaderLabels(RESULT_COLUMNS)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setSelectionMode(QTableWidget.ExtendedSelection)  # shift/ctrl-click overlays several candidates
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.results_table.itemSelectionChanged.connect(self._on_result_selected)
        right_layout.addWidget(self.results_table, 1)

        self.plot = PlotWidget(figsize=(7, 4.5))
        right_layout.addWidget(self.plot, 1)

        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

    # ------------------------------------------------------------------
    # No-Python-needed database setup (downloads from rruff.net)
    # ------------------------------------------------------------------
    def download_database(self) -> None:
        resp = QMessageBox.question(
            self, "Download RRUFF database",
            "This downloads roughly 400-500 MB from rruff.net and builds the local search "
            "index — no Python needed, but it takes several minutes and needs an internet "
            "connection. Continue?",
        )
        if resp != QMessageBox.Yes:
            return
        import raman.rruff_science as rs
        self._start_download(
            self.download_db_btn, "Downloading…",
            lambda: rs.download_and_build_rruff_cache(log=self._dl_queue.put),
            self._on_database_downloaded,
        )

    def _on_database_downloaded(self, n_spectra: int) -> None:
        self._index = None  # force a reload of the freshly-built index
        self._index_load_attempted = False
        self._ensure_index_loaded()
        self.db_status_label.setText(f"Downloaded and indexed {n_spectra} spectra. " + self.db_status_label.text())

    def download_amcsd(self) -> None:
        resp = QMessageBox.question(
            self, "Download AMCSD structures",
            "This downloads roughly 66 MB from rruff.net for the 'Overlay candidate's XRD (CIF)' "
            "button — no Python needed. Continue?",
        )
        if resp != QMessageBox.Yes:
            return
        import raman.rruff_science as rs
        self._start_download(
            self.download_amcsd_btn, "Downloading…",
            lambda: rs.download_and_build_amcsd_cache(log=self._dl_queue.put),
            lambda n: self.db_status_label.setText(f"Downloaded and indexed {n} AMCSD structures."),
        )

    def _start_download(self, button: QPushButton, busy_text: str, work, on_done) -> None:
        """Both download buttons disable together — the RRUFF and AMCSD
        downloads share one progress queue/status label, so running them
        concurrently would interleave their log lines."""
        self.download_db_btn.setEnabled(False)
        self.download_amcsd_btn.setEnabled(False)
        button.setText(busy_text)
        self.db_status_label.setText("Starting download…")
        self._dl_timer.start()
        from core.qt_worker import run_in_thread
        run_in_thread(work, lambda result: self._on_download_done(on_done, result),
                     self._on_download_error)

    def _drain_download_log(self) -> None:
        """Runs on the main thread every 200ms while a download is active;
        the worker thread only ever puts() into the queue (thread-safe),
        never touches a widget directly."""
        last = None
        while not self._dl_queue.empty():
            last = self._dl_queue.get_nowait()
        if last is not None:
            self.db_status_label.setText(last)

    def _reset_download_buttons(self) -> None:
        self._dl_timer.stop()
        self.download_db_btn.setEnabled(True)
        self.download_db_btn.setText("Download RRUFF database…")
        self.download_amcsd_btn.setEnabled(True)
        self.download_amcsd_btn.setText("Download AMCSD structures…")

    def _on_download_done(self, on_done, result) -> None:
        self._reset_download_buttons()
        on_done(result)

    def _on_download_error(self, traceback_text: str) -> None:
        self._reset_download_buttons()
        QMessageBox.critical(self, "Download failed", traceback_text)

    # ------------------------------------------------------------------
    def set_spectra(self, spectrum_ids: List[str]) -> None:
        current = self.spec_combo.currentData()
        self.spec_combo.blockSignals(True)
        self.spec_combo.clear()
        for sid in spectrum_ids:
            spectrum = self.library.get(sid)
            if spectrum is not None:
                self.spec_combo.addItem(spectrum.title, sid)
        if current is not None:
            idx = self.spec_combo.findData(current)
            if idx >= 0:
                self.spec_combo.setCurrentIndex(idx)
        self.spec_combo.blockSignals(False)
        # Eager index load on page entry (user report: the λ filter was
        # empty until the first search, because the index only loaded then).
        if self._index is None and not getattr(self, "_index_load_attempted", False):
            self._index_load_attempted = True
            self._ensure_index_loaded()
        # Arriving on the page must show the spectrum immediately, not an
        # empty axes (user request — same rule on every workspace).
        self.plot.request_redraw(lambda: self._render_preview(self._selected_candidates()))

    def _current_spectrum(self):
        sid = self.spec_combo.currentData()
        return self.library.get(sid) if sid else None

    # ------------------------------------------------------------------
    def _ensure_index_loaded(self) -> bool:
        if self._index is not None:
            return True
        t0 = time.time()
        self._index = load_index(cache_dir=self.cache_dir)
        elapsed = time.time() - t0
        if not self._index:
            self.db_status_label.setText(
                f"No RRUFF database found at {self.cache_dir}. Run rruff_science.build_index() first."
            )
            return False
        summary = index_summary(self._index)
        self.db_status_label.setText(
            f"RRUFF database: {summary['n_spectra']} spectra, {summary['n_minerals']} minerals, "
            f"{len(summary['wavelengths_nm'])} wavelengths (loaded in {elapsed:.2f}s)."
        )
        # Fill the λ filter with the wavelengths actually present, keeping
        # any selection the user already made.
        current = self.wavelength_combo.currentData()
        self.wavelength_combo.blockSignals(True)
        self.wavelength_combo.clear()
        self.wavelength_combo.addItem("Any", None)
        for wl in summary["wavelengths_nm"]:
            self.wavelength_combo.addItem(f"{wl:g} nm", float(wl))
        if current is not None:
            idx = self.wavelength_combo.findData(current)
            if idx >= 0:
                self.wavelength_combo.setCurrentIndex(idx)
        self.wavelength_combo.blockSignals(False)
        return True

    # ------------------------------------------------------------------
    def auto_find_peaks(self) -> None:
        spectrum = self._current_spectrum()
        if spectrum is None:
            QMessageBox.warning(self, "Auto-find peaks", "Select a query spectrum first.")
            return
        peaks = find_peak_candidates(spectrum.x, spectrum.y, max_peaks=15)
        if not peaks:
            QMessageBox.information(self, "Auto-find peaks", "No clear peak candidates were found.")
            return
        self.peaks_edit.setText(", ".join(f"{p:.1f}" for p in sorted(peaks)))

    def _parse_peaks(self) -> List[float]:
        text = self.peaks_edit.text().strip()
        if not text:
            return []
        out = []
        for part in text.split(","):
            v = _to_float(part)
            if v is not None:
                out.append(v)
        return out

    def find_matches(self) -> None:
        if not self._ensure_index_loaded():
            return
        peaks = self._parse_peaks()
        if not peaks:
            QMessageBox.warning(self, "Find matches", "Enter or auto-find at least one peak position.")
            return
        tolerance = _to_float(self.tolerance_edit.text(), 10.0)
        self._query_peaks = peaks
        searchable = filter_rruff_index(
            self._index,
            wavelength_nm=self.wavelength_combo.currentData(),
            oriented=self.orientation_combo.currentData(),
            scan_type=self.scan_type_combo.currentData(),
            quality=self.quality_combo.currentData(),
        )
        if not searchable:
            QMessageBox.information(self, "Find matches", "No database spectra pass the current filters — relax them and retry.")
            self._results = []
            self._populate_results_table()
            return
        # Already-accepted references for this spectrum drop out of further
        # searches — accept means "this phase is identified", so the search
        # continues over the remaining candidates/peaks.
        spectrum = self._current_spectrum()
        accepted_ids = {m.get("rruff_id") for m in (spectrum.meta.get("rruff_matches", []) if spectrum else [])}
        if accepted_ids:
            searchable = [r for r in searchable if r.get("rruff_id") not in accepted_ids]
        if len(searchable) < len(self._index):
            self.db_status_label.setText(
                f"Filters keep {len(searchable)} of {len(self._index)} database spectra"
                + (f" ({len(accepted_ids)} already-accepted excluded)." if accepted_ids else ".")
            )
        self._results = rank_rruff_matches(peaks, searchable, tolerance=tolerance, top_n=25)
        self._populate_results_table()
        self._render_preview([self._results[0]] if self._results else [])

    def _populate_results_table(self) -> None:
        self.results_table.setRowCount(len(self._results))
        for row, rec in enumerate(self._results):
            values = [
                rec.get("mineral", ""), rec.get("rruff_id", ""),
                f"{rec.get('wavelength_nm', ''):.0f}" if rec.get("wavelength_nm") else "",
                str(rec.get("matched_peaks", "")),
                f"{rec.get('match_fraction', 0.0) * 100:.0f}%",
                rec.get("category", ""),
            ]
            for col, val in enumerate(values):
                self.results_table.setItem(row, col, QTableWidgetItem(str(val)))
        self.results_table.resizeColumnsToContents()
        if self._results:
            self.results_table.selectRow(0)

    def _on_result_selected(self) -> None:
        rows = self.results_table.selectionModel().selectedRows()
        if not rows or not self._results:
            return
        self._render_preview(self._selected_candidates())

    # ------------------------------------------------------------------
    CANDIDATE_COLORS = ["crimson", "royalblue", "seagreen", "darkorange", "purple", "teal"]
    # Muted palette for phases already accepted (they stay overlaid)
    ACCEPTED_COLORS = ["#9c6b74", "#6b7d9c", "#6b9c7d", "#9c8a6b", "#856b9c", "#6b969c"]

    def _render_preview(self, candidates: List[Dict[str, Any]]) -> None:
        """Overlay the query against one or several candidates (shift/ctrl-
        click rows to compare references side by side)."""
        self.plot.cancel_pending()  # direct draws supersede any queued entry render
        # keep the user's zoom across candidate re-renders (same spectrum)
        self.plot.preserve_zoom(("raman_id", self.spec_combo.currentData()))
        fig = self.plot.figure
        fig.clf()
        ax = fig.add_subplot(111)

        spectrum = self._current_spectrum()
        if spectrum is not None:
            y = spectrum.y
            y_norm = y / np.nanmax(np.abs(y)) if np.nanmax(np.abs(y)) > 0 else y
            ax.plot(spectrum.x, y_norm, color="black", lw=1.1, label=f"query: {spectrum.title}")

        # Query peak bars (same convention as XRD ID): current unexplained
        # peaks in black, peaks explained by accepted phases in gray.
        state = self._state()
        peaks = self._parse_peaks()
        if peaks:
            ax.vlines(peaks, 0, 1.04, color="black", lw=0.8, alpha=0.30,
                      label="query peaks")
        if state["explained"]:
            ax.vlines(state["explained"], 0, 1.04, color="gray", lw=0.8,
                      alpha=0.45, ls="--", label="explained by accepted phases")

        # Accepted phases stay overlaid across iterative searches (muted).
        for k, acc in enumerate(state["accepted"]):
            color = self.ACCEPTED_COLORS[k % len(self.ACCEPTED_COLORS)]
            acc_label = f"accepted: {acc.get('mineral', '?')} ({acc.get('rruff_id', '?')})"
            for j, peak in enumerate(acc.get("peaks", [])):
                ax.axvline(peak, color=color, ls=":", lw=0.9, alpha=0.6,
                           label=acc_label if j == 0 else None)

        for k, candidate in enumerate(candidates):
            color = self.CANDIDATE_COLORS[k % len(self.CANDIDATE_COLORS)]
            label = f"{candidate.get('mineral', '?')} ({candidate.get('rruff_id', '?')}, {candidate.get('wavelength_nm', '?')} nm)"
            drew_curve = False
            if self.overlay_raw_check.isChecked() and candidate.get("raw_path") and os.path.isfile(candidate["raw_path"]):
                try:
                    with open(candidate["raw_path"], "r", encoding="utf-8", errors="replace") as f:
                        text = f.read()
                    ref = parse_rruff_txt(text, source_filename=os.path.basename(candidate["raw_path"]))
                    if len(ref.y):
                        ref_norm = ref.y / np.nanmax(np.abs(ref.y)) if np.nanmax(np.abs(ref.y)) > 0 else ref.y
                        ax.plot(ref.x, ref_norm, color=color, lw=1.0, alpha=0.8, label=label)
                        drew_curve = True
                except OSError:
                    pass
            for j, peak in enumerate(candidate.get("peaks", [])):
                ax.axvline(peak, color=color, ls="--", lw=0.7, alpha=0.5,
                           label=label if (not drew_curve and j == 0) else None)

        ax.set_xlabel("Raman shift (cm⁻¹)")
        ax.set_ylabel("Normalized intensity")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        self.plot.restore_zoom(ax)
        fig.tight_layout()
        self.plot.canvas.draw_idle()

    # ------------------------------------------------------------------
    def clear_session(self) -> None:
        """Start over on the current spectrum: peaks, results, and the
        accepted-overlay/explained-peak session state (identifications
        recorded on the spectrum itself are untouched — undo those from
        the Library)."""
        sid = self.spec_combo.currentData() or "__none__"
        self._session.pop(sid, None)
        self.peaks_edit.setText("")
        self._query_peaks = []
        self._results = []
        self._populate_results_table()
        self.plot.reset_zoom_memory()  # starting over autoscales again
        self.plot.request_redraw(lambda: self._render_preview([]))

    # ------------------------------------------------------------------
    def send_candidate_cifs(self) -> None:
        """RRUFF→CIF overlay handoff: look up the selected candidate's
        mineral in the local AMCSD CIF cache and send its structure(s) to
        the Raman workspace's CIF overlay — Raman identification to
        predicted-XRD verification in one click."""
        rows = self.results_table.selectionModel().selectedRows()
        if not rows or not self._results:
            QMessageBox.warning(self, "CIF overlay", "Select a candidate row first.")
            return
        candidate = self._results[rows[0].row()]
        mineral = candidate.get("mineral", "")
        kwargs = {"cache_dir": self.amcsd_cache_dir} if self.amcsd_cache_dir else {}
        cif_paths = find_cifs_for_mineral(mineral, **kwargs)
        if not cif_paths:
            QMessageBox.information(
                self, "CIF overlay",
                f"No AMCSD structure found for '{mineral}' in the local cache.\n"
                "Build it once with rruff_science.ingest_amcsd_cif_zip() "
                "(cif.zip from https://www.rruff.net/AMS/zipped_files/).",
            )
            return
        if self.on_send_cifs is None:
            QMessageBox.information(self, "CIF overlay", "No Raman workspace is wired to receive CIFs here.")
            return
        self.on_send_cifs(cif_paths)
        self.db_status_label.setText(
            f"Sent {len(cif_paths)} AMCSD structure(s) for '{mineral}' to the Raman workspace CIF overlay."
        )

    def accept_selected_candidate(self) -> None:
        """Accept = 'this phase is identified': record it, remove the peaks
        it explains from the query, and immediately re-search the REMAINING
        peaks for further phases (user request — mixtures are the norm, one
        spectrum can hold several minerals). Accepted references are
        excluded from the follow-up search; every accept lands on the undo
        stack (Ctrl+Z in the Library)."""
        rows = self.results_table.selectionModel().selectedRows()
        spectrum = self._current_spectrum()
        if spectrum is None or not rows or not self._results:
            QMessageBox.warning(self, "Accept identification", "Select a query spectrum and a candidate row first.")
            return
        candidate = self._results[rows[0].row()]
        tolerance = _to_float(self.tolerance_edit.text(), 10.0)

        previous_state = {
            "rruff_match": spectrum.meta.get("rruff_match"),
            "rruff_matches": list(spectrum.meta["rruff_matches"]) if spectrum.meta.get("rruff_matches") else None,
        }
        match_record = {
            "mineral": candidate.get("mineral"),
            "rruff_id": candidate.get("rruff_id"),
            "wavelength_nm": candidate.get("wavelength_nm"),
            "matched_peaks": candidate.get("matched_peaks"),
            "match_fraction": candidate.get("match_fraction"),
            "accepted_at": time.time(),
        }
        spectrum.meta["rruff_match"] = match_record  # latest accept (back-compat: reports/projects read this)
        accepted = list(spectrum.meta.get("rruff_matches", []))
        accepted.append(match_record)
        spectrum.meta["rruff_matches"] = accepted
        if self.on_accept is not None:
            self.on_accept(spectrum.id, previous_state)

        cand_peaks = candidate.get("peaks", []) or []
        remaining = [p for p in self._query_peaks
                     if not any(abs(p - cp) <= tolerance for cp in cand_peaks)]
        # session overlays: the accepted phase's bars stay, its peaks gray out
        state = self._state()
        state["accepted"].append(dict(candidate))
        state["explained"].extend(p for p in self._query_peaks if p not in remaining)
        phase_list = ", ".join(f"{m['mineral']} ({m['rruff_id']})" for m in accepted)

        if remaining:
            self.peaks_edit.setText(", ".join(f"{p:.1f}" for p in sorted(remaining)))
            QMessageBox.information(
                self, "Phase accepted",
                f"Recorded '{candidate.get('mineral')}' ({candidate.get('rruff_id')}) for '{spectrum.title}'.\n"
                f"{len(remaining)} peak(s) remain unexplained — the table now shows matches for those.\n"
                f"Accepted so far: {phase_list}\n\n{RRUFF_CITATION}",
            )
            self.find_matches()
        else:
            self.peaks_edit.setText("")
            self._results = []
            self._populate_results_table()
            self._render_preview([])
            QMessageBox.information(
                self, "All peaks explained",
                f"Every query peak of '{spectrum.title}' is now explained by {len(accepted)} accepted phase(s): "
                f"{phase_list}.\n\n{RRUFF_CITATION}",
            )
