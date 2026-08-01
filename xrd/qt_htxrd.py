"""
qt_htxrd.py — HTXRD series-processing workspace (M20, reworked after real
use).

Thin Qt layer over htxrd_science.py: import a whole folder (or file
selection) of diffraction patterns as one temperature-ordered series
(.rasx Temp metadata -> Jana-style filename template -> sequence index),
then:
  - Waterfall view (temperature-colored, optional peak guide lines),
  - Maps view ported from the user's own XRD_HT.ipynb: 2D heatmap
    (linear/log/sqrt/power color scaling), difference map / difference
    waterfall vs a reference pattern, 3D surface, temperature-or-time
    y-axis from a heating rate,
  - Multi-window peak tracking with sequential seeding, per-window "@"
    anchors picking WHICH peak to track, and absence detection (peaks that
    vanish/appear are flagged as transition signatures, not fit garbage).
"""
from __future__ import annotations

import csv
import re
from typing import Dict, List, Optional

import numpy as np

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QMessageBox, QPushButton, QSplitter, QTabWidget,
    QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)

from xrd.htxrd_science import (
    HtxrdPattern,
    PeakTrackResult,
    auto_track_windows,
    build_common_grid,
    build_intensity_map,
    compute_relative_time_minutes,
    estimate_waterfall_shift,
    evaluate_peak_guide,
    find_series_files,
    flag_transition_candidates,
    guide_centers_for_patterns,
    load_series,
    parse_peak_guides,
    parse_track_windows,
    reference_index,
    track_peak_guided,
    track_peaks_multi,
)
from core.qt_widgets import PlotWidget

MAP_CMAPS = ["magma", "inferno", "viridis", "plasma", "jet", "Greys", "RdBu_r"]
TRACK_COLORS = ["royalblue", "crimson", "seagreen", "darkorange", "purple", "brown", "teal"]


def _to_float(text: str, default: Optional[float] = None) -> Optional[float]:
    try:
        return float((text or "").strip())
    except (TypeError, ValueError):
        return default


class HtxrdWorkspace(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.series: List[HtxrdPattern] = []
        self.track_results: Dict[str, List[PeakTrackResult]] = {}
        self._track_picks: List[tuple] = []  # (slice_1based, x) waterfall click anchors
        self._wf_state: Optional[tuple] = None  # (patterns, display_y, delta) of the last waterfall render
        self._pick_cid = None
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        splitter = QSplitter()
        root.addWidget(splitter)

        left = QWidget()
        left.setObjectName("Card")
        left.setMaximumWidth(330)
        left_layout = QVBoxLayout(left)

        import_row = QHBoxLayout()
        folder_btn = QPushButton("Import folder…")
        folder_btn.clicked.connect(self.import_folder)
        files_btn = QPushButton("Import files…")
        files_btn.clicked.connect(self.import_files)
        import_row.addWidget(folder_btn)
        import_row.addWidget(files_btn)
        left_layout.addLayout(import_row)

        left_layout.addWidget(QLabel("Filename template (Jana-style, e.g. scan_???.xy)\nUsed only when a file has no temperature metadata"))
        self.template_edit = QLineEdit()
        left_layout.addWidget(self.template_edit)

        left_layout.addWidget(QLabel("Series (ordered by ramp value)"))
        self.series_list = QListWidget()
        self.series_list.itemSelectionChanged.connect(lambda: self.plot.request_redraw(self.render_waterfall))
        left_layout.addWidget(self.series_list, 1)

        shift_row = QHBoxLayout()
        shift_row.addWidget(QLabel("Waterfall shift ×"))
        self.shift_edit = QLineEdit("1.0")
        self.shift_edit.setMaximumWidth(60)
        self.shift_edit.textChanged.connect(lambda: self.plot.request_redraw(self.render_waterfall))
        shift_row.addWidget(self.shift_edit)
        self.normalize_check = QCheckBox("Normalize")
        self.normalize_check.toggled.connect(lambda: self.plot.request_redraw(self.render_waterfall))
        shift_row.addWidget(self.normalize_check)
        shift_row.addStretch(1)
        left_layout.addLayout(shift_row)

        rate_row = QHBoxLayout()
        rate_row.addWidget(QLabel("Heating rate (°C/min)"))
        self.heating_rate_edit = QLineEdit()
        self.heating_rate_edit.setPlaceholderText("optional")
        self.heating_rate_edit.setMaximumWidth(70)
        self.heating_rate_edit.setToolTip("Enables the 'Time (min)' y-axis on the Maps tab.")
        rate_row.addWidget(self.heating_rate_edit)
        left_layout.addLayout(rate_row)

        left_layout.addWidget(QLabel("Peak tracking — 2θ windows\n('lo-hi @ center; lo-hi'; the @ anchor picks WHICH\npeak to track when a window holds several)"))
        self.windows_edit = QLineEdit()
        self.windows_edit.setPlaceholderText("28.5-29.5 @ 28.98; 31-32")
        left_layout.addWidget(self.windows_edit)

        track_opts = QHBoxLayout()
        self.seed_check = QCheckBox("Seed from previous")
        self.seed_check.setChecked(True)
        self.seed_check.setToolTip("Each pattern's fit starts from the previous pattern's result (sequential refinement) — keeps the fit on a drifting peak.")
        track_opts.addWidget(self.seed_check)
        track_opts.addWidget(QLabel("Absence σ"))
        self.absence_edit = QLineEdit("3.0")
        self.absence_edit.setMaximumWidth(45)
        self.absence_edit.setToolTip("A fitted peak weaker than this many noise σ is reported as ABSENT instead of a garbage fit.")
        track_opts.addWidget(self.absence_edit)
        left_layout.addLayout(track_opts)

        shape_row = QHBoxLayout()
        shape_row.addWidget(QLabel("Peak shape"))
        self.shape_combo = QComboBox()
        self.shape_combo.addItems(["G", "GL"])
        shape_row.addWidget(self.shape_combo)
        left_layout.addLayout(shape_row)

        self._track_btn = QPushButton("Track peaks across series")
        self._track_btn.setObjectName("Primary")
        self._track_btn.clicked.connect(self.run_track_peak)
        left_layout.addWidget(self._track_btn)

        self._auto_track_btn = QPushButton("Auto-track all peaks")
        self._auto_track_btn.setToolTip(
            "Find every peak of the first selected pattern, build an anchored window around each "
            "(shown in the windows field for editing), and track them all."
        )
        self._auto_track_btn.clicked.connect(self.auto_track_all)
        left_layout.addWidget(self._auto_track_btn)

        export_btn = QPushButton("Export tracking results to CSV…")
        export_btn.clicked.connect(self.export_track_csv)
        left_layout.addWidget(export_btn)

        self.status_label = QLabel("Import a series to begin.")
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()

        waterfall_tab = QWidget()
        wl = QVBoxLayout(waterfall_tab)
        pick_row = QHBoxLayout()
        self.pick_guide_btn = QPushButton("Pick track guide")
        self.pick_guide_btn.setCheckable(True)
        self.pick_guide_btn.setToolTip(
            "Toggle, then click the SAME peak at two (or more) temperatures on the waterfall — "
            "e.g. once near the bottom and once near the top. 'Track picked guide' then follows "
            "the peak along that line. One click works too (a constant-2θ guide), including from "
            "an intermediate temperature for a peak that appears or dies."
        )
        self.pick_guide_btn.toggled.connect(self._on_pick_guide_toggled)
        pick_row.addWidget(self.pick_guide_btn)
        pick_row.addWidget(QLabel("± window (°2θ)"))
        self.guide_half_edit = QLineEdit("0.3")
        self.guide_half_edit.setMaximumWidth(50)
        pick_row.addWidget(self.guide_half_edit)
        self._track_guide_btn = QPushButton("Track picked guide")
        self._track_guide_btn.clicked.connect(self.track_picked_guide)
        pick_row.addWidget(self._track_guide_btn)
        clear_picks_btn = QPushButton("Clear picks")
        clear_picks_btn.clicked.connect(self._clear_track_picks)
        pick_row.addWidget(clear_picks_btn)
        pick_row.addStretch(1)
        wl.addLayout(pick_row)
        self.plot = PlotWidget(figsize=(7, 5.5))
        wl.addWidget(self.plot)
        self.tabs.addTab(waterfall_tab, "Waterfall")

        maps_tab = QWidget()
        ml = QVBoxLayout(maps_tab)
        maps_controls = QHBoxLayout()
        self.map_type_combo = QComboBox()
        self.map_type_combo.addItems(["Heatmap", "Difference map", "Difference waterfall", "3D surface"])
        maps_controls.addWidget(self.map_type_combo)
        self.map_intensity_combo = QComboBox()
        self.map_intensity_combo.addItems(["raw", "normalized"])
        maps_controls.addWidget(self.map_intensity_combo)
        self.map_scale_combo = QComboBox()
        self.map_scale_combo.addItems(["linear", "log", "sqrt", "power"])
        maps_controls.addWidget(self.map_scale_combo)
        self.map_cmap_combo = QComboBox()
        self.map_cmap_combo.addItems(MAP_CMAPS)
        maps_controls.addWidget(self.map_cmap_combo)
        self.yaxis_combo = QComboBox()
        self.yaxis_combo.addItems(["Temperature", "Time (min)"])
        maps_controls.addWidget(self.yaxis_combo)
        ml.addLayout(maps_controls)

        maps_controls2 = QHBoxLayout()
        maps_controls2.addWidget(QLabel("Diff. reference"))
        self.reference_edit = QLineEdit("first")
        self.reference_edit.setMaximumWidth(60)
        self.reference_edit.setToolTip("'first', a pattern index (e.g. 3), or a temperature (e.g. 610.5 — nearest pattern wins).")
        maps_controls2.addWidget(self.reference_edit)
        self.diff_mode_combo = QComboBox()
        self.diff_mode_combo.addItems(["signed", "absolute"])
        maps_controls2.addWidget(self.diff_mode_combo)
        maps_controls2.addWidget(QLabel("Guides"))
        self.guides_edit = QLineEdit()
        self.guides_edit.setPlaceholderText("{1:29; 57:28.65} {39:29.63; 57:29.4}")
        self.guides_edit.setToolTip("Peak guide lines: {slice:2θ; slice:2θ} anchors, linearly interpolated (1-based slices, sorted by ramp). Drawn on the heatmap and waterfall.")
        maps_controls2.addWidget(self.guides_edit, 1)
        ml.addLayout(maps_controls2)

        for combo in (self.map_type_combo, self.map_intensity_combo, self.map_scale_combo,
                      self.map_cmap_combo, self.yaxis_combo, self.diff_mode_combo):
            combo.currentIndexChanged.connect(lambda *_: self.maps_plot.request_redraw(self.render_maps))
        self.reference_edit.editingFinished.connect(lambda: self.maps_plot.request_redraw(self.render_maps))
        self.guides_edit.editingFinished.connect(lambda: self.maps_plot.request_redraw(self.render_maps))

        self.maps_plot = PlotWidget(figsize=(7, 5.5))
        ml.addWidget(self.maps_plot, 1)
        self.tabs.addTab(maps_tab, "Maps")

        track_tab = QWidget()
        tl = QVBoxLayout(track_tab)
        self.track_plot = PlotWidget(figsize=(7, 5))
        tl.addWidget(self.track_plot, 2)
        self.track_table = QTableWidget(0, 9)
        self.track_table.setHorizontalHeaderLabels(["Window", "Pattern", "Ramp", "Center", "FWHM", "Amplitude", "Area", "chi2_red", "Present"])
        self.track_table.setEditTriggers(QTableWidget.NoEditTriggers)
        tl.addWidget(self.track_table, 1)
        self.transition_text = QTextEdit()
        self.transition_text.setReadOnly(True)
        self.transition_text.setMaximumHeight(90)
        tl.addWidget(self.transition_text)
        self.tabs.addTab(track_tab, "Peak tracking")

        right_layout.addWidget(self.tabs)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

    # ------------------------------------------------------------------
    def import_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select folder of diffraction patterns")
        if not folder:
            return
        paths = find_series_files(folder)
        if not paths:
            QMessageBox.information(self, "Import series", "No pattern files (.rasx/.xy/.xrdml/.txt/.dat) found in that folder.")
            return
        self._load(paths)

    def import_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select diffraction patterns", "",
            "Diffraction patterns (*.rasx *.xy *.xrdml *.txt *.dat);;All files (*.*)",
        )
        if not paths:
            return
        self._load(list(paths))

    def _load(self, paths: List[str]) -> None:
        template = self.template_edit.text().strip()
        try:
            series = load_series(paths, filename_template=template)
        except Exception as exc:
            QMessageBox.critical(self, "Import series error", str(exc))
            return
        self.set_series(series)

    def set_series(self, series: List[HtxrdPattern]) -> None:
        """Adopt an already-loaded series (fresh import or project restore)
        and refresh the list + waterfall."""
        self.series = list(series)
        self.series_list.clear()
        sources = set()
        for pat in self.series:
            self.series_list.addItem(f"{pat.name}   [{pat.ramp_value:g}{'°C' if pat.ramp_source == 'metadata' else ''}, {pat.ramp_source}]")
            sources.add(pat.ramp_source)
        self.series_list.selectAll()
        self.status_label.setText(
            f"Loaded {len(self.series)} patterns. Ramp source(s): {', '.join(sorted(sources))}."
            + (" WARNING: some ramp values are just sequence indices — add a filename template or use files with temperature metadata."
               if "index" in sources else "")
        )
        self.render_waterfall()
        self.plot.canvas.draw_idle()
        self.maps_plot.request_redraw(self.render_maps)

    def _selected_patterns(self) -> List[HtxrdPattern]:
        selected_rows = {i.row() for i in self.series_list.selectedIndexes()}
        if not selected_rows:
            return list(self.series)
        return [p for i, p in enumerate(self.series) if i in selected_rows]

    def _parse_guides(self) -> list:
        """Guides field: any number of '{slice:2θ; slice:2θ}' groups."""
        text = self.guides_edit.text().strip()
        if not text:
            return []
        try:
            return parse_peak_guides(re.findall(r"\{[^}]*\}", text))
        except (ValueError, TypeError):
            return []

    def _y_axis(self, patterns: List[HtxrdPattern]):
        """(values, label) for map y-axes: temperature, or minutes when a
        heating rate is given and 'Time (min)' is selected."""
        ramps = np.array([p.ramp_value for p in patterns], float)
        if self.yaxis_combo.currentText().startswith("Time"):
            times = compute_relative_time_minutes(ramps, _to_float(self.heating_rate_edit.text()))
            if times is not None:
                return times, "Time (min)"
        return ramps, "Temperature (°C)" if any(p.ramp_source == "metadata" for p in patterns) else "Ramp value"

    # ------------------------------------------------------------------
    def render_waterfall(self) -> None:
        patterns = self._selected_patterns()
        fig = self.plot.figure
        fig.clf()
        if not patterns:
            self.plot.canvas.draw_idle()
            return
        ax = fig.add_subplot(111)

        shift_factor = _to_float(self.shift_edit.text(), 1.0) or 1.0
        max_dy = max(((np.nanmax(p.y) - np.nanmin(p.y)) for p in patterns), default=1.0)
        delta = 0.5 * shift_factor * max_dy

        cmap = None
        try:
            import matplotlib
            import matplotlib.colors as mcolors
            ramps = [p.ramp_value for p in patterns]
            norm = mcolors.Normalize(vmin=min(ramps), vmax=max(ramps) or 1.0)
            cmap = matplotlib.colormaps["inferno"]
        except Exception:
            norm = None

        display_y = []
        for i, pat in enumerate(patterns):
            y = pat.y.astype(float)
            if self.normalize_check.isChecked():
                rng = np.nanmax(y) - np.nanmin(y)
                if rng > 0:
                    y = (y - np.nanmin(y)) / rng * max_dy
            display_y.append(y)
            color = cmap(norm(pat.ramp_value)) if (cmap is not None and norm is not None) else None
            label = f"{pat.ramp_value:g}" + ("°C" if pat.ramp_source == "metadata" else "")
            ax.plot(pat.x, y + i * delta, lw=0.8, color=color, label=label if len(patterns) <= 15 else None)

        self._wf_state = (patterns, display_y, delta)

        # Peak guide lines (notebook feature): interpolated 2θ anchors drawn
        # at each slice's own intensity + offset.
        for guide in self._parse_guides():
            _, xpos = evaluate_peak_guide(guide, len(patterns))
            gx, gy = [], []
            for i, xg in enumerate(xpos):
                if not np.isfinite(xg):
                    continue
                yg = float(np.interp(xg, patterns[i].x, display_y[i]))
                gx.append(xg)
                gy.append(yg + i * delta + 0.02 * delta)
            if gx:
                ax.plot(gx, gy, "k--", lw=1.0)

        # Picked track-guide anchors (X markers) + the interpolated guide
        # they define across the whole series.
        if self._track_picks:
            centers = guide_centers_for_patterns(self._track_picks, len(patterns))
            gx, gy = [], []
            for i, xg in enumerate(centers):
                yg = float(np.interp(xg, patterns[i].x, display_y[i]))
                gx.append(float(xg))
                gy.append(yg + i * delta + 0.02 * delta)
            ax.plot(gx, gy, "-", color="magenta", lw=1.2, alpha=0.8)
            for s, xp in self._track_picks:
                i = int(np.clip(s - 1, 0, len(patterns) - 1))
                yp = float(np.interp(xp, patterns[i].x, display_y[i])) + i * delta
                ax.plot([xp], [yp], "x", color="magenta", ms=10, mew=2)

        ax.set_xlabel("2θ (deg)")
        ax.set_ylabel("Intensity (offset)")
        ax.set_title(f"HTXRD series — {len(patterns)} patterns (bottom = lowest ramp)")
        if len(patterns) <= 15:
            ax.legend(fontsize=7, title="Ramp")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        self.plot.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Pick-to-track (user request: click the peak at a start and an end
    # temperature on the waterfall instead of typing window numbers)
    # ------------------------------------------------------------------
    def _on_pick_guide_toggled(self, checked: bool) -> None:
        from PySide6.QtCore import Qt
        if checked:
            if not self.series:
                self.pick_guide_btn.setChecked(False)
                QMessageBox.information(self, "Pick track guide", "Import a series first.")
                return
            mode = str(self.plot.toolbar.mode)
            if "zoom" in mode:
                self.plot.toolbar.zoom()
            elif "pan" in mode:
                self.plot.toolbar.pan()
            self.plot.canvas.setCursor(Qt.CrossCursor)
            self._pick_cid = self.plot.canvas.mpl_connect("button_press_event", self._on_waterfall_pick)
            self.status_label.setText("Pick mode: click the same peak at different temperatures (bottom = lowest).")
        else:
            if self._pick_cid is not None:
                self.plot.canvas.mpl_disconnect(self._pick_cid)
                self._pick_cid = None
            self.plot.canvas.unsetCursor()

    def _on_waterfall_pick(self, event) -> None:
        if event.inaxes is None or event.xdata is None or self.plot.toolbar.mode:
            return
        if self._wf_state is None:
            return
        patterns, display_y, delta = self._wf_state
        x_click, y_click = float(event.xdata), float(event.ydata)
        # Which slice? The one whose displayed curve (offset included) passes
        # closest to the click at this 2θ.
        best_i, best_d = 0, np.inf
        for i, pat in enumerate(patterns):
            y_at = float(np.interp(x_click, pat.x, display_y[i])) + i * delta
            d = abs(y_click - y_at)
            if d < best_d:
                best_i, best_d = i, d
        self._track_picks.append((best_i + 1, x_click))
        self._track_picks.sort(key=lambda t: t[0])
        self.status_label.setText(
            f"Guide anchors: {', '.join(f'slice {s} @ {x:.2f}°' for s, x in self._track_picks)} — "
            "add more or click 'Track picked guide'."
        )
        self.plot.request_redraw(self.render_waterfall)

    def _clear_track_picks(self) -> None:
        self._track_picks = []
        self.plot.request_redraw(self.render_waterfall)

    def track_picked_guide(self) -> None:
        if not self.series:
            QMessageBox.warning(self, "Track picked guide", "Import a series first.")
            return
        if not self._track_picks:
            QMessageBox.warning(self, "Track picked guide", "Pick at least one anchor on the waterfall first ('Pick track guide').")
            return
        patterns = self._selected_patterns()
        picks = list(self._track_picks)
        half = _to_float(self.guide_half_edit.text(), 0.3) or 0.3
        shape = self.shape_combo.currentText()
        absence_sigma = _to_float(self.absence_edit.text(), 3.0) or 3.0
        label = f"guide {picks[0][1]:.2f}°" + (f"→{picks[-1][1]:.2f}°" if len(picks) > 1 else "")

        self._track_guide_btn.setEnabled(False)
        self._track_guide_btn.setText("Tracking…")
        from core.qt_worker import run_in_thread
        run_in_thread(
            lambda: track_peak_guided(patterns, picks, half_window=half, shape=shape,
                                      absence_sigma=absence_sigma, window_label=label),
            lambda rows, label=label: self._on_guided_done(label, rows),
            self._on_guided_error,
        )

    def _on_guided_error(self, traceback_text: str) -> None:
        self._track_guide_btn.setEnabled(True)
        self._track_guide_btn.setText("Track picked guide")
        QMessageBox.critical(self, "Track picked guide error", traceback_text)

    def _on_guided_done(self, label: str, rows: List[PeakTrackResult]) -> None:
        self._track_guide_btn.setEnabled(True)
        self._track_guide_btn.setText("Track picked guide")
        # merge, so several guides can be picked and tracked one after another
        self.track_results = {**self.track_results, label: rows}
        self._refresh_tracking_views()

    def auto_track_all(self) -> None:
        """Automatic tracking setup: peaks of the first selected pattern →
        anchored windows (left in the windows field for editing) → full run."""
        if not self.series:
            QMessageBox.warning(self, "Auto-track", "Import a series first.")
            return
        patterns = self._selected_patterns()
        windows = auto_track_windows(patterns[0])
        if not windows:
            QMessageBox.information(self, "Auto-track", "No clear peaks found in the first selected pattern.")
            return
        self.windows_edit.setText("; ".join(
            f"{w['lo']:g}-{w['hi']:g} @ {w['center']:g}" for w in windows
        ))
        self.status_label.setText(f"Auto-track: {len(windows)} peak window(s) from '{patterns[0].name}' — tracking…")
        self.run_track_peak()

    # ------------------------------------------------------------------
    # Maps tab (ported from the user's XRD_HT.ipynb)
    # ------------------------------------------------------------------
    def _color_norm(self, data: np.ndarray, scale: str, signed: bool = False):
        import matplotlib.colors as mcolors
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return mcolors.Normalize(vmin=0, vmax=1)
        if signed:
            vmax = float(np.nanmax(np.abs(finite)))
            return mcolors.Normalize(vmin=-vmax, vmax=vmax)
        if scale == "log":
            pos = finite[finite > 0]
            if pos.size:
                return mcolors.LogNorm(vmin=float(pos.min()), vmax=float(pos.max()))
            scale = "linear"
        if scale == "sqrt":
            return mcolors.PowerNorm(gamma=0.5, vmin=float(finite.min()), vmax=float(finite.max()))
        if scale == "power":
            return mcolors.PowerNorm(gamma=0.8, vmin=float(finite.min()), vmax=float(finite.max()))
        return mcolors.Normalize(vmin=float(finite.min()), vmax=float(finite.max()))

    def _parse_reference(self):
        text = self.reference_edit.text().strip() or "first"
        if text.lower() == "first":
            return "first"
        try:
            return int(text) if re.fullmatch(r"[+-]?\d+", text) else float(text)
        except ValueError:
            return "first"

    def render_maps(self) -> None:
        patterns = self._selected_patterns()
        fig = self.maps_plot.figure
        fig.clf()
        if len(patterns) < 2:
            ax = fig.add_subplot(111)
            ax.set_title("Maps need at least 2 patterns")
            self.maps_plot.canvas.draw_idle()
            return

        try:
            grid = build_common_grid(patterns, npts=2000)
        except ValueError as exc:
            ax = fig.add_subplot(111)
            ax.set_title(str(exc))
            self.maps_plot.canvas.draw_idle()
            return
        normalize = self.map_intensity_combo.currentText() == "normalized"
        data = build_intensity_map(patterns, grid, normalize=normalize)
        yvals, ylabel = self._y_axis(patterns)
        cmap_name = self.map_cmap_combo.currentText()
        scale = self.map_scale_combo.currentText()
        plot_type = self.map_type_combo.currentText()
        temps = np.array([p.ramp_value for p in patterns], float)

        if plot_type == "Heatmap":
            ax = fig.add_subplot(111)
            show = data.copy()
            if scale == "log":
                show = np.where(show > 0, show, np.nan)
            mesh = ax.pcolormesh(grid, yvals, show, cmap=cmap_name, norm=self._color_norm(show, scale), shading="nearest")
            fig.colorbar(mesh, ax=ax, label=f"{'normalized ' if normalize else ''}intensity ({scale})")
            for guide in self._parse_guides():
                _, xpos = evaluate_peak_guide(guide, len(patterns))
                ok = np.isfinite(xpos)
                if np.any(ok):
                    ax.plot(xpos[ok], yvals[ok], "k--", lw=1.0)
            ax.set_xlabel("2θ (deg)")
            ax.set_ylabel(ylabel)
            ax.set_title("Intensity heatmap")

        elif plot_type == "Difference map":
            ax = fig.add_subplot(111)
            try:
                ref_idx = reference_index(patterns, self._parse_reference())
            except ValueError as exc:
                ax.set_title(str(exc))
                self.maps_plot.canvas.draw_idle()
                return
            diff = data - data[ref_idx][None, :]
            signed = self.diff_mode_combo.currentText() == "signed"
            if not signed:
                diff = np.abs(diff)
            use_cmap = "RdBu_r" if signed else cmap_name
            mesh = ax.pcolormesh(grid, yvals, diff, cmap=use_cmap,
                                 norm=self._color_norm(diff, scale if not signed else "linear", signed=signed),
                                 shading="nearest")
            fig.colorbar(mesh, ax=ax, label=f"Δ intensity vs slice {ref_idx + 1} ({patterns[ref_idx].ramp_value:g})")
            ax.set_xlabel("2θ (deg)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"Difference map ({'signed' if signed else 'absolute'})")

        elif plot_type == "Difference waterfall":
            ax = fig.add_subplot(111)
            try:
                ref_idx = reference_index(patterns, self._parse_reference())
            except ValueError as exc:
                ax.set_title(str(exc))
                self.maps_plot.canvas.draw_idle()
                return
            diff = data - data[ref_idx][None, :]
            shift = estimate_waterfall_shift(diff) * (_to_float(self.shift_edit.text(), 1.0) or 1.0)
            import matplotlib
            import matplotlib.colors as mcolors
            norm_t = mcolors.Normalize(vmin=float(temps.min()), vmax=float(temps.max()) or 1.0)
            cm = matplotlib.colormaps[cmap_name]
            for i in range(len(patterns)):
                ax.plot(grid, diff[i] + i * shift, lw=0.8, color=cm(norm_t(temps[i])))
            sm = matplotlib.cm.ScalarMappable(norm=norm_t, cmap=cm)
            fig.colorbar(sm, ax=ax, label="Temperature (°C)")
            ax.set_yticks([])
            ax.set_xlabel("2θ (deg)")
            ax.set_ylabel("Δ intensity + offset")
            ax.set_title(f"Difference waterfall vs slice {ref_idx + 1} ({patterns[ref_idx].ramp_value:g})")

        else:  # 3D surface
            ax = fig.add_subplot(111, projection="3d")
            X, Y = np.meshgrid(grid, yvals)
            Z = np.where(np.isfinite(data), data, 0.0)
            surf_norm = self._color_norm(Z, scale)
            ax.plot_surface(X, Y, Z, cmap=cmap_name, norm=surf_norm, linewidth=0, antialiased=True)
            ax.set_xlabel("2θ (deg)")
            ax.set_ylabel(ylabel)
            ax.set_zticks([])
            ax.view_init(elev=25, azim=-70)
            ax.set_title("3D surface")

        fig.tight_layout()
        self.maps_plot.canvas.draw_idle()

    # ------------------------------------------------------------------
    def run_track_peak(self) -> None:
        if not self.series:
            QMessageBox.warning(self, "Peak tracking", "Import a series first.")
            return
        try:
            windows = parse_track_windows(self.windows_edit.text())
        except ValueError as exc:
            QMessageBox.warning(self, "Peak tracking", str(exc))
            return
        if not windows:
            QMessageBox.warning(self, "Peak tracking", "Enter at least one 2θ window, e.g. '28.5-29.5 @ 28.98; 31-32'.")
            return

        patterns = self._selected_patterns()
        shape = self.shape_combo.currentText()
        seed = self.seed_check.isChecked()
        absence_sigma = _to_float(self.absence_edit.text(), 3.0) or 3.0

        self._track_btn.setEnabled(False)
        self._track_btn.setText("Tracking…")

        from core.qt_worker import run_in_thread
        run_in_thread(
            lambda: track_peaks_multi(patterns, windows, shape=shape,
                                      seed_from_previous=seed, absence_sigma=absence_sigma),
            self._on_track_done, self._on_track_error,
        )

    def _on_track_error(self, traceback_text: str) -> None:
        self._track_btn.setEnabled(True)
        self._track_btn.setText("Track peaks across series")
        QMessageBox.critical(self, "Peak tracking error", traceback_text)

    def _on_track_done(self, results: Dict[str, List[PeakTrackResult]]) -> None:
        self._track_btn.setEnabled(True)
        self._track_btn.setText("Track peaks across series")
        self.track_results = results
        self._refresh_tracking_views()

    def _refresh_tracking_views(self) -> None:
        """Table + plot + transition flags from self.track_results — shared
        by full runs and incremental guided-tracking merges."""
        self._populate_track_table()
        self._render_track_plot()

        lines = []
        for label, rows in self.track_results.items():
            flags = flag_transition_candidates(rows)
            for ramp, chi2, reason in flags:
                prefix = "" if reason.startswith("[") else f"[{label}] "
                lines.append(f"  ramp={ramp:g}: {prefix}{reason}")
        if lines:
            self.transition_text.setPlainText(
                "Candidate phase-transition signatures (fit-quality anomalies, vanished/appeared peaks):\n"
                + "\n".join(lines)
            )
        else:
            self.transition_text.setPlainText("No transition signatures flagged across the series.")
        self.tabs.setCurrentIndex(2)

    def _populate_track_table(self) -> None:
        rows_flat = [r for rows in self.track_results.values() for r in rows]
        self.track_table.setRowCount(len(rows_flat))
        for row, r in enumerate(rows_flat):
            values = [r.window_label, r.pattern_name, f"{r.ramp_value:g}"]
            if r.error is not None:
                values += ["—", "—", "—", "—", f"ERROR: {r.error}", "—"]
            elif not r.present:
                values += ["—", "—", f"{r.amplitude:.4g}", "—", "—", "absent"]
            else:
                values += [f"{r.center:.4f}", f"{r.fwhm:.4f}", f"{r.amplitude:.4g}", f"{r.area:.4g}", f"{r.chi2_red:.4g}", "yes"]
            for col, val in enumerate(values):
                self.track_table.setItem(row, col, QTableWidgetItem(str(val)))
        self.track_table.resizeColumnsToContents()

    def _render_track_plot(self) -> None:
        fig = self.track_plot.figure
        fig.clf()
        any_ok = any(r.error is None and r.present for rows in self.track_results.values() for r in rows)
        if not any_ok:
            self.track_plot.canvas.draw_idle()
            return
        ax_c, ax_w, ax_a = fig.subplots(3, 1, sharex=True)
        for k, (label, rows) in enumerate(self.track_results.items()):
            ok = [r for r in rows if r.error is None]
            if not ok:
                continue
            ramps = [r.ramp_value for r in ok]
            # absent -> NaN so the line breaks across the gap instead of bridging it
            centers = [r.center if r.present else np.nan for r in ok]
            fwhms = [r.fwhm if r.present else np.nan for r in ok]
            areas = [r.area if r.present else np.nan for r in ok]
            color = TRACK_COLORS[k % len(TRACK_COLORS)]
            ax_c.plot(ramps, centers, "o-", ms=4, lw=1, color=color, label=label)
            ax_w.plot(ramps, fwhms, "s-", ms=4, lw=1, color=color)
            ax_a.plot(ramps, areas, "^-", ms=4, lw=1, color=color)
        ax_c.set_ylabel("Center (°2θ)")
        ax_c.grid(alpha=0.25)
        if len(self.track_results) > 1:
            ax_c.legend(fontsize=7)
        ax_w.set_ylabel("FWHM (°2θ)")
        ax_w.grid(alpha=0.25)
        ax_a.set_ylabel("Area")
        ax_a.set_xlabel("Ramp value (temperature)")
        ax_a.grid(alpha=0.25)
        fig.suptitle("Tracked peak(s) vs. ramp", fontsize=10)
        fig.tight_layout()
        self.track_plot.canvas.draw_idle()

    # ------------------------------------------------------------------
    def export_track_csv(self) -> None:
        if not self.track_results:
            QMessageBox.information(self, "Export", "Run peak tracking first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export tracking results", "", "CSV (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["window", "pattern", "ramp_value", "center_2theta", "fwhm_2theta",
                                 "amplitude", "area", "chi2_red", "present", "noise_sigma", "error"])
                for label, rows in self.track_results.items():
                    for r in rows:
                        writer.writerow([label, r.pattern_name, r.ramp_value, r.center, r.fwhm,
                                         r.amplitude, r.area, r.chi2_red, r.present, r.noise, r.error or ""])
        except OSError as exc:
            QMessageBox.critical(self, "Export error", str(exc))
            return
        QMessageBox.information(self, "Export", f"Tracking results exported to:\n{path}")
