"""
qt_simple_plot.py — Simple Plot + CIF Bragg overlay, ported from
ui_simple_plot.py's SimplePlotWindow (M7, core slice).

Reuses the shared, debounced PlotWidget (qt_widgets.py) — the CIF-overlay
lag the user hit directly (see the M7 planning conversation) was caused by
an undebounced redraw-on-every-keystroke in the Tk version; here, every
control that triggers a redraw goes through PlotWidget.request_redraw(),
so the same class of bug is structurally avoided rather than patched after
the fact.

Deferred to a follow-up (not in this slice): DTA-specific derivative
plotting within Simple Plot, spectral-difference mode, click-to-annotate,
mouse-move coordinate readout, dark mode. FitParamWindow (ui_fit_params.py)
is scoped to M8 instead, since that's where single-spectrum fitting lives.

Reads spectra directly from the shared SpectrumLibrary/Spectrum model (M5)
rather than re-loading via injected loader callables the way the Tk version
needed to when it was decoupled from main.py.
"""
from __future__ import annotations

import colorsys
import os
from typing import Any, Dict, List, Optional

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDialog, QFileDialog, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
    QRadioButton, QScrollArea, QSplitter, QVBoxLayout, QWidget,
)

from cif_tools import bragg_peaks_from_cif_generic, list_cif_files_case_insensitive
from core.qt_models import SpectrumLibrary
from core.qt_widgets import PlotWidget

CIF_COLORS = ["crimson", "royalblue", "seagreen", "darkorange", "purple", "goldenrod"]
LINE_COLORS = ["navy", "darkred", "seagreen", "darkorange", "purple", "teal", "brown", "indigo"]


def _to_float(text: str) -> Optional[float]:
    try:
        return float((text or "").strip().replace(",", "."))
    except (TypeError, ValueError):
        return None


class CifManagerDialog(QDialog):
    """Per-CIF visibility/label/color/pad controls — a modeless-feeling
    dialog whose edits redraw the parent's plot via the shared debounce."""

    def __init__(self, workspace: "SimplePlotWorkspace"):
        super().__init__(workspace)
        self.workspace = workspace
        self.setWindowTitle("CIF manager")
        self.resize(680, 320)

        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        folder_btn = QPushButton("Set CIF folder…")
        folder_btn.clicked.connect(workspace.choose_cif_folder_and_reload)
        top.addWidget(folder_btn)

        top.addWidget(QLabel("Label pos."))
        self.pos_combo = QComboBox()
        self.pos_combo.addItems(["right_out", "left_out", "right_in", "left_in"])
        self.pos_combo.setCurrentText(workspace.cif_label_pos)
        self.pos_combo.currentTextChanged.connect(self._on_pos_changed)
        top.addWidget(self.pos_combo)

        top.addWidget(QLabel("Bragg height ×"))
        self.height_edit = QLineEdit(str(workspace.bragg_height_scale))
        self.height_edit.textChanged.connect(self._on_height_changed)
        top.addWidget(self.height_edit)
        layout.addLayout(top)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.list_container = QWidget()
        self.list_layout = QVBoxLayout(self.list_container)
        self.list_layout.setAlignment(Qt.AlignTop)
        scroll.setWidget(self.list_container)
        layout.addWidget(scroll)

        self.rebuild()

    def _on_pos_changed(self, text: str) -> None:
        self.workspace.cif_label_pos = text
        self.workspace.plot.request_redraw(self.workspace.render)

    def _on_height_changed(self, text: str) -> None:
        v = _to_float(text)
        self.workspace.bragg_height_scale = v if (v is not None and v > 0.1) else 1.0
        self.workspace.plot.request_redraw(self.workspace.render)

    def rebuild(self) -> None:
        while self.list_layout.count():
            item = self.list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for serie in self.workspace.cif_series:
            row = QHBoxLayout()
            show_check = QCheckBox(serie.get("label", os.path.basename(serie.get("path", ""))))
            show_check.setChecked(serie.get("visible", False))
            show_check.toggled.connect(lambda checked, s=serie: self.workspace.set_cif_field(s, "visible", checked))
            row.addWidget(show_check)

            name_edit = QLineEdit(serie.get("plot_label", ""))
            name_edit.setPlaceholderText("display label")
            name_edit.textChanged.connect(lambda text, s=serie: self.workspace.set_cif_field(s, "plot_label", text))
            row.addWidget(name_edit)

            color_edit = QLineEdit(serie.get("color", "crimson"))
            color_edit.setMaximumWidth(80)
            color_edit.textChanged.connect(lambda text, s=serie: self.workspace.set_cif_field(s, "color", text or "crimson"))
            row.addWidget(color_edit)

            pad_edit = QLineEdit(str(serie.get("pad", 0.03)))
            pad_edit.setMaximumWidth(50)
            pad_edit.textChanged.connect(lambda text, s=serie: self.workspace.set_cif_field(s, "pad", _to_float(text) or 0.03))
            row.addWidget(pad_edit)

            wrapper = QWidget()
            wrapper.setLayout(row)
            self.list_layout.addWidget(wrapper)


class SimplePlotWorkspace(QWidget):
    def __init__(self, parent: Optional[QWidget] = None, library: Optional[SpectrumLibrary] = None):
        super().__init__(parent)
        self.library = library if library is not None else SpectrumLibrary()
        self.cif_series: List[Dict[str, Any]] = []
        self.cif_label_pos = "right_out"
        self.bragg_height_scale = 1.0
        self._cif_manager: Optional[CifManagerDialog] = None
        self.annotations: List[Dict[str, float]] = []  # persists across re-renders

        self._build_ui()
        self.plot.canvas.mpl_connect("button_press_event", self._on_plot_click)

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        left = QWidget()
        left.setObjectName("Card")
        left.setMaximumWidth(320)
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("Select from imported"))

        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.file_list.itemSelectionChanged.connect(lambda: self.plot.request_redraw(self.render))
        left_layout.addWidget(self.file_list, 1)

        self.norm_check = QCheckBox("Normalize")
        self.norm_check.toggled.connect(lambda: self.plot.request_redraw(self.render))
        left_layout.addWidget(self.norm_check)

        self.zero_line_check = QCheckBox("Zero line (y=0)")
        self.zero_line_check.toggled.connect(lambda: self.plot.request_redraw(self.render))
        left_layout.addWidget(self.zero_line_check)

        self.diff_check = QCheckBox("Difference (1st − 2nd selected)")
        self.diff_check.toggled.connect(lambda: self.plot.request_redraw(self.render))
        left_layout.addWidget(self.diff_check)

        annotate_row = QHBoxLayout()
        self.annotate_check = QCheckBox("Annotate on click")
        annotate_row.addWidget(self.annotate_check)
        clear_annot_btn = QPushButton("Clear")
        clear_annot_btn.setMaximumWidth(60)
        clear_annot_btn.clicked.connect(self._clear_annotations)
        annotate_row.addWidget(clear_annot_btn)
        left_layout.addLayout(annotate_row)

        smooth_row = QHBoxLayout()
        smooth_row.addWidget(QLabel("Smoothing"))
        self.smooth_combo = QComboBox()
        self.smooth_combo.addItems(["None", "SG", "MA(5)"])
        self.smooth_combo.currentTextChanged.connect(lambda: self.plot.request_redraw(self.render))
        smooth_row.addWidget(self.smooth_combo)
        left_layout.addLayout(smooth_row)

        cif_row = QHBoxLayout()
        import_cif_btn = QPushButton("Import CIF…")
        import_cif_btn.clicked.connect(self.import_cif)
        manage_cif_btn = QPushButton("CIF manager…")
        manage_cif_btn.clicked.connect(self.open_cif_manager)
        cif_row.addWidget(import_cif_btn)
        cif_row.addWidget(manage_cif_btn)
        left_layout.addLayout(cif_row)
        left_layout.addStretch(1)
        splitter.addWidget(left)

        right = QWidget()
        right.setObjectName("Card")
        right_layout = QVBoxLayout(right)

        mode_row = QHBoxLayout()
        self.mode_group = QButtonGroup(self)
        self.mode_separate = QRadioButton("Separate")
        self.mode_stacked = QRadioButton("Stacked")
        self.mode_separate.setChecked(True)
        self.mode_group.addButton(self.mode_separate, 0)
        self.mode_group.addButton(self.mode_stacked, 1)
        self.mode_separate.toggled.connect(lambda: self.plot.request_redraw(self.render))
        mode_row.addWidget(self.mode_separate)
        mode_row.addWidget(self.mode_stacked)
        mode_row.addWidget(QLabel("Shift:"))
        self.shift_edit = QLineEdit("0.5")
        self.shift_edit.setMaximumWidth(50)
        self.shift_edit.textChanged.connect(lambda: self.plot.request_redraw(self.render))
        mode_row.addWidget(self.shift_edit)
        mode_row.addWidget(QLabel("Colors:"))
        self.color_combo = QComboBox()
        self.color_combo.addItems(["Matplotlib cycle", "Distinct", "Hash by name", "Monochrome"])
        self.color_combo.setCurrentText("Distinct")
        self.color_combo.currentTextChanged.connect(lambda: self.plot.request_redraw(self.render))
        mode_row.addWidget(self.color_combo)
        mode_row.addStretch(1)
        right_layout.addLayout(mode_row)

        axes_row = QHBoxLayout()
        axes_row.addWidget(QLabel("Title"))
        self.title_edit = QLineEdit()
        self.title_edit.textChanged.connect(lambda: self.plot.request_redraw(self.render))
        axes_row.addWidget(self.title_edit)
        axes_row.addWidget(QLabel("X title"))
        self.x_title_edit = QLineEdit("Raman Shift (cm⁻¹)")
        self.x_title_edit.textChanged.connect(lambda: self.plot.request_redraw(self.render))
        axes_row.addWidget(self.x_title_edit)
        axes_row.addWidget(QLabel("Y title"))
        self.y_title_edit = QLineEdit("Intensity (a.u.)")
        self.y_title_edit.textChanged.connect(lambda: self.plot.request_redraw(self.render))
        axes_row.addWidget(self.y_title_edit)
        right_layout.addLayout(axes_row)

        limits_row = QHBoxLayout()
        limits_row.addWidget(QLabel("X limits"))
        self.xmin_edit = QLineEdit()
        self.xmax_edit = QLineEdit()
        for e in (self.xmin_edit, self.xmax_edit):
            e.setMaximumWidth(70)
            e.textChanged.connect(lambda: self.plot.request_redraw(self._apply_axis_limits))
        limits_row.addWidget(self.xmin_edit)
        limits_row.addWidget(self.xmax_edit)
        limits_row.addWidget(QLabel("Y limits"))
        self.ymin_edit = QLineEdit()
        self.ymax_edit = QLineEdit()
        for e in (self.ymin_edit, self.ymax_edit):
            e.setMaximumWidth(70)
            e.textChanged.connect(lambda: self.plot.request_redraw(self._apply_axis_limits))
        limits_row.addWidget(self.ymin_edit)
        limits_row.addWidget(self.ymax_edit)

        self.no_title_check = QCheckBox("No title")
        self.no_yticks_check = QCheckBox("No Y ticks")
        self.grid_check = QCheckBox("Grid")
        for cb in (self.no_title_check, self.no_yticks_check, self.grid_check):
            cb.toggled.connect(lambda: self.plot.request_redraw(self.render))
            limits_row.addWidget(cb)
        limits_row.addStretch(1)
        right_layout.addLayout(limits_row)

        self.plot = PlotWidget(figsize=(8, 5))
        right_layout.addWidget(self.plot, 1)

        export_row = QHBoxLayout()
        self.export_png_check = QCheckBox("PNG")
        self.export_png_check.setChecked(True)
        self.export_svg_check = QCheckBox("SVG")
        self.export_pdf_check = QCheckBox("PDF")
        for cb in (self.export_png_check, self.export_svg_check, self.export_pdf_check):
            export_row.addWidget(cb)
        export_btn = QPushButton("Export")
        export_btn.clicked.connect(self.export_all_formats)
        export_row.addWidget(export_btn)
        export_row.addStretch(1)
        right_layout.addLayout(export_row)

        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

    # ------------------------------------------------------------------
    def set_spectra(self, spectrum_ids: List[str]) -> None:
        """Populate the file list from the given library spectrum ids,
        called by the shell when this workspace becomes active. The previous
        selection is preserved, and the first spectrum is auto-selected when
        nothing was — arriving on the page always shows a plot, never an
        empty axes (user request)."""
        selected = {self.file_list.item(i).data(Qt.UserRole)
                    for i in range(self.file_list.count()) if self.file_list.item(i).isSelected()}
        self.file_list.blockSignals(True)
        self.file_list.clear()
        for sid in spectrum_ids:
            spectrum = self.library.get(sid)
            if spectrum is None:
                continue
            item = QListWidgetItem(spectrum.title)
            item.setData(Qt.UserRole, sid)
            self.file_list.addItem(item)
            if sid in selected:
                item.setSelected(True)
        self.file_list.blockSignals(False)
        if not self.file_list.selectedItems() and self.file_list.count():
            self.file_list.item(0).setSelected(True)  # fires the render hook
        else:
            self.plot.request_redraw(self.render)

    def _selected_spectra(self):
        out = []
        for item in self.file_list.selectedItems():
            spectrum = self.library.get(item.data(Qt.UserRole))
            if spectrum is not None:
                out.append(spectrum)
        return out

    def _apply_smoothing(self, y: np.ndarray) -> np.ndarray:
        method = self.smooth_combo.currentText()
        if method == "None":
            return y
        if method.startswith("MA"):
            kernel = np.ones(5, dtype=float) / 5.0
            return np.convolve(y, kernel, mode="same")
        if method == "SG":
            try:
                from scipy.signal import savgol_filter
                return savgol_filter(y, window_length=7, polyorder=2, mode="interp")
            except Exception:
                kernel = np.ones(5, dtype=float) / 5.0
                return np.convolve(y, kernel, mode="same")
        return y

    def _color_from_scheme(self, i: int, label: str, total: int):
        scheme = self.color_combo.currentText()
        if scheme == "Matplotlib cycle":
            return None
        if scheme == "Distinct":
            hue = i / max(total, 1)
            return colorsys.hsv_to_rgb(hue, 0.75, 0.9)
        if scheme == "Hash by name":
            h = (hash(label) % 360) / 360.0
            return colorsys.hsv_to_rgb(h, 0.70, 0.9)
        if scheme == "Monochrome":
            return "#3b82f6"
        return LINE_COLORS[i % len(LINE_COLORS)]

    # ------------------------------------------------------------------
    # Click-to-annotate (deferred M7 item). Annotations are stored as data
    # coordinates on the workspace (not as matplotlib artists), so they
    # survive every debounced fig.clf()+render cycle and are re-drawn by
    # render() until explicitly cleared.
    # ------------------------------------------------------------------

    def _on_plot_click(self, event) -> None:
        if not self.annotate_check.isChecked():
            return
        if event.inaxes is None or event.xdata is None:
            return
        if self.plot.toolbar.mode:  # zoom/pan tool active — don't hijack the click
            return
        self.annotations.append({"x": float(event.xdata), "y": float(event.ydata)})
        self.plot.request_redraw(self.render)

    def _clear_annotations(self) -> None:
        self.annotations.clear()
        self.plot.request_redraw(self.render)

    def _draw_annotations(self) -> None:
        axes = self.plot.figure.get_axes()
        if not axes or not self.annotations:
            return
        for ann in self.annotations:
            for ax in axes:
                xlo, xhi = sorted(ax.get_xlim())
                if not (xlo <= ann["x"] <= xhi):
                    continue
                # a visible dot right where the user clicked (A FAIRE item 16)
                ax.plot([ann["x"]], [ann["y"]], marker="o", ms=6, mew=0,
                        color="#8a3033", zorder=5)
                ax.annotate(
                    f"{ann['x']:.1f}", xy=(ann["x"], ann["y"]),
                    xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=8, color="#8a3033",
                    arrowprops=dict(arrowstyle="-", color="#8a3033", lw=0.8),
                )
                break  # annotate in the first axes whose x-range contains it

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> None:
        spectra = self._selected_spectra()
        fig = self.plot.figure
        fig.clf()
        if not spectra:
            self.plot.canvas.draw_idle()
            return

        norm_xmin = _to_float(self.xmin_edit.text())
        norm_xmax = _to_float(self.xmax_edit.text())
        global_title = self.title_edit.text().strip()
        mode = "stacked" if self.mode_stacked.isChecked() else "separate"

        prepared = []
        for s in spectra:
            x, y = np.asarray(s.x, dtype=float), np.asarray(s.y, dtype=float)
            if self.norm_check.isChecked():
                mask = np.isfinite(x) & np.isfinite(y)
                if norm_xmin is not None:
                    mask &= x >= norm_xmin
                if norm_xmax is not None:
                    mask &= x <= norm_xmax
                if mask.sum() > 1:
                    area = np.trapz(y[mask], x[mask])
                    if abs(area) > 1e-12:
                        y = y / area * 100
            y = self._apply_smoothing(y)
            prepared.append((x, y, s.title))

        if self.diff_check.isChecked():
            # Spectral-difference mode (deferred M7 item): with exactly two
            # spectra selected, plot A, B, and A − B (B interpolated onto
            # A's grid) on one axes, overriding separate/stacked layout.
            if len(prepared) == 2:
                (xa, ya, la), (xb, yb, lb) = prepared
                yb_on_a = np.interp(xa, xb, yb)
                ax = fig.add_subplot(111)
                ax.plot(xa, ya, lw=1, label=la)
                ax.plot(xa, yb_on_a, lw=1, label=lb)
                ax.plot(xa, ya - yb_on_a, lw=1.4, color="black", label=f"{la} − {lb}")
                ax.axhline(0.0, lw=0.8, alpha=0.6, color="0.6")
                ax.set_xlabel(self.x_title_edit.text())
                ax.set_ylabel(self.y_title_edit.text())
                if not self.no_title_check.isChecked():
                    ax.set_title(global_title or f"Difference: {la} − {lb}")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3) if self.grid_check.isChecked() else ax.grid(False)
                self._draw_cif_bragg_markers()
                self._apply_axis_limits(skip_redraw=True)
                self._draw_annotations()
                fig.tight_layout()
                self.plot.canvas.draw_idle()
                return
            # Wrong selection count: fall through to normal rendering so the
            # user still sees their data (checkbox simply has no effect).

        if mode == "separate":
            axes = np.atleast_1d(fig.subplots(1, len(prepared), sharey=True)).ravel()
            for i, (ax, (x, y, label)) in enumerate(zip(axes, prepared)):
                color = self._color_from_scheme(i, label, len(prepared))
                ax.plot(x, y, color=color, lw=1) if color is not None else ax.plot(x, y, lw=1)
                ax.set_xlabel(self.x_title_edit.text())
                if not self.no_title_check.isChecked():
                    ax.set_title(global_title or label)
                if i == 0 and not self.no_yticks_check.isChecked():
                    ax.set_ylabel(self.y_title_edit.text())
                else:
                    ax.tick_params(labelleft=False)
                if self.zero_line_check.isChecked():
                    ax.axhline(0.0, lw=0.8, alpha=0.6, color="0.6")
                ax.grid(True, alpha=0.3) if self.grid_check.isChecked() else ax.grid(False)
        else:
            ax = fig.add_subplot(111)
            try:
                shift_factor = float(self.shift_edit.text())
            except (TypeError, ValueError):
                shift_factor = 0.5
            max_dy = 0.0
            for x, y, _ in prepared:
                mask = np.ones_like(x, dtype=bool)
                if norm_xmin is not None:
                    mask &= x >= norm_xmin
                if norm_xmax is not None:
                    mask &= x <= norm_xmax
                yv = y[mask] if np.any(mask) else y
                if len(yv):
                    max_dy = max(max_dy, (yv.max() - yv.min()) * 0.5)
            delta = shift_factor * max_dy
            for i, (x, y, label) in enumerate(prepared):
                color = self._color_from_scheme(i, label, len(prepared))
                y_plot = y + i * delta
                ax.plot(x, y_plot, color=color, lw=1, label=label) if color is not None else ax.plot(x, y_plot, lw=1, label=label)
            if len(prepared) > 1:
                ax.legend(fontsize=9)
            ax.set_xlabel(self.x_title_edit.text())
            ax.set_ylabel(self.y_title_edit.text())
            if self.no_yticks_check.isChecked():
                ax.tick_params(left=False, labelleft=False)
            if self.zero_line_check.isChecked():
                ax.axhline(0.0, lw=0.8, alpha=0.6, color="0.6")
            if not self.no_title_check.isChecked():
                ax.set_title(global_title or prepared[-1][2])
            ax.grid(True, alpha=0.3) if self.grid_check.isChecked() else ax.grid(False)

        self._draw_cif_bragg_markers()
        self._apply_axis_limits(skip_redraw=True)
        self._draw_annotations()
        fig.tight_layout()
        self.plot.canvas.draw_idle()

    def _apply_axis_limits(self, skip_redraw: bool = False) -> None:
        xmin, xmax = _to_float(self.xmin_edit.text()), _to_float(self.xmax_edit.text())
        ymin, ymax = _to_float(self.ymin_edit.text()), _to_float(self.ymax_edit.text())
        for ax in self.plot.figure.get_axes():
            if xmin is not None or xmax is not None:
                lo, hi = ax.get_xlim()
                new_lo = xmin if xmin is not None else lo
                new_hi = xmax if xmax is not None else hi
                if new_lo != new_hi:
                    ax.set_xlim(new_lo, new_hi)
            if ymin is not None or ymax is not None:
                lo, hi = ax.get_ylim()
                new_lo = ymin if ymin is not None else lo
                new_hi = ymax if ymax is not None else hi
                if new_lo != new_hi:
                    ax.set_ylim(new_lo, new_hi)
        if not skip_redraw:
            self.plot.canvas.draw_idle()

    def autoscale_y(self) -> None:
        for ax in self.plot.figure.get_axes():
            ax.relim()
            ax.autoscale(axis="y")
        self.ymin_edit.clear()
        self.ymax_edit.clear()
        self.plot.canvas.draw_idle()

    # ------------------------------------------------------------------
    # CIF Bragg overlay
    # ------------------------------------------------------------------

    def import_cif(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select CIF file", "", "CIF files (*.cif);;All files (*.*)")
        if not path:
            return
        for s in self.cif_series:
            if os.path.abspath(s["path"]) == os.path.abspath(path):
                s["visible"] = True
                self.render()
                return
        try:
            peaks = bragg_peaks_from_cif_generic(path, two_theta_max=80.0, hkl_max=6)
        except Exception as exc:
            QMessageBox.critical(self, "CIF import error", f"Could not read CIF:\n{exc}")
            return
        if not peaks:
            QMessageBox.information(self, "CIF import", "No Bragg peak found below 80° 2θ.")
            return
        color = CIF_COLORS[len(self.cif_series) % len(CIF_COLORS)]
        self.cif_series.append({
            "path": path, "label": os.path.basename(path), "plot_label": "",
            "peaks": peaks, "visible": True, "color": color, "pad": 0.03,
        })
        self.render()

    def restore_cif_overlays(self, records: List[Dict[str, Any]]) -> int:
        """Project-restore path: rebuild the overlay series from saved
        display fields, recomputing Bragg peaks from each CIF path (cheap —
        cif_tools disk-caches them). Overlays whose CIF file no longer
        exists are skipped rather than erroring."""
        restored = 0
        self.cif_series = []
        for rec in records:
            path = rec.get("path", "")
            if not path or not os.path.isfile(path):
                continue
            try:
                peaks = bragg_peaks_from_cif_generic(path, two_theta_max=80.0, hkl_max=6)
            except Exception:
                continue
            self.cif_series.append({
                "path": path,
                "label": rec.get("label", os.path.basename(path)),
                "plot_label": rec.get("plot_label", ""),
                "peaks": peaks,
                "visible": bool(rec.get("visible", True)),
                "color": rec.get("color", "crimson"),
                "pad": float(rec.get("pad", 0.03)),
            })
            restored += 1
        if self._cif_manager is not None:
            self._cif_manager.rebuild()
        self.plot.request_redraw(self.render)
        return restored

    def add_cif_files(self, paths: List[str]) -> int:
        """Programmatic CIF import (used by the RRUFF→CIF overlay handoff):
        add each CIF as a visible series, skipping ones already loaded.
        Returns how many were added."""
        added = 0
        for path in paths:
            ap = os.path.abspath(path)
            existing = next((s for s in self.cif_series if os.path.abspath(s.get("path", "")) == ap), None)
            if existing is not None:
                existing["visible"] = True
                continue
            try:
                peaks = bragg_peaks_from_cif_generic(path, two_theta_max=80.0, hkl_max=6)
            except Exception:
                continue
            if not peaks:
                continue
            color = CIF_COLORS[len(self.cif_series) % len(CIF_COLORS)]
            self.cif_series.append({
                "path": path, "label": os.path.basename(path), "plot_label": "",
                "peaks": peaks, "visible": True, "color": color, "pad": 0.03,
            })
            added += 1
        if self._cif_manager is not None:
            self._cif_manager.rebuild()
        self.plot.request_redraw(self.render)
        return added

    def choose_cif_folder_and_reload(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select CIF folder")
        if not folder:
            return
        cif_paths = list_cif_files_case_insensitive(folder)
        if not cif_paths:
            QMessageBox.information(self, "CIF manager", "No CIF files found in the selected folder.")
            return
        added = 0
        for p in cif_paths:
            ap = os.path.abspath(p)
            if any(os.path.abspath(s.get("path", "")) == ap for s in self.cif_series):
                continue
            try:
                peaks = bragg_peaks_from_cif_generic(p, two_theta_max=80.0, hkl_max=6)
            except Exception:
                continue
            color = CIF_COLORS[len(self.cif_series) % len(CIF_COLORS)]
            self.cif_series.append({
                "path": p, "label": os.path.basename(p), "plot_label": "",
                "peaks": peaks, "visible": False, "color": color, "pad": 0.03,
            })
            added += 1
        if self._cif_manager is not None:
            self._cif_manager.rebuild()
        self.render()
        QMessageBox.information(self, "CIF manager", f"Found {len(cif_paths)} CIF files.\nAdded {added} new entries.")

    def open_cif_manager(self) -> None:
        if self._cif_manager is None:
            self._cif_manager = CifManagerDialog(self)
        else:
            self._cif_manager.rebuild()
        self._cif_manager.show()
        self._cif_manager.raise_()

    def set_cif_field(self, serie: Dict[str, Any], key: str, value: Any) -> None:
        serie[key] = value
        # Same debounce mechanism as every other redraw-triggering control —
        # the whole point of this port is that per-field CIF edits (color,
        # label, pad, visibility) can never again cause an undebounced
        # full-artist-recreation redraw per keystroke.
        self.plot.request_redraw(self.render)

    def _effective_xlim(self, ax):
        xmin_ax, xmax_ax = ax.get_xlim()
        xmin_in, xmax_in = _to_float(self.xmin_edit.text()), _to_float(self.xmax_edit.text())
        if xmin_in is not None and xmax_in is not None and xmin_in < xmax_in:
            return xmin_in, xmax_in
        if xmin_in is not None:
            return xmin_in, xmax_ax
        if xmax_in is not None:
            return xmin_ax, xmax_in
        return xmin_ax, xmax_ax

    def _draw_cif_bragg_markers(self) -> None:
        visible_series = [c for c in self.cif_series if c.get("visible", True)]
        axes = self.plot.figure.get_axes()
        if not visible_series or not axes:
            return

        hmul = max(0.1, self.bragg_height_scale)

        for ax in axes:
            xmin, xmax = self._effective_xlim(ax)
            ymin_orig, ymax_orig = ax.get_ylim()
            height = (ymax_orig - ymin_orig) or 1.0

            n_in_view = [len([tt for (tt, _hkl, _d) in s["peaks"] if xmin <= tt <= xmax]) for s in visible_series]
            max_n = max(n_in_view) if n_in_view else 0
            row_space_factor = 1.0 + min(2.0, 0.02 * max_n)

            base_band_h = 0.03 * height
            base_spacing = 1.8 * base_band_h
            base_offset = 0.02 * height
            band_h = base_band_h * hmul
            spacing = base_spacing * row_space_factor
            off_down = base_offset * row_space_factor

            extra_down = off_down + len(visible_series) * spacing
            ax.set_ylim(ymin_orig - extra_down, ymax_orig)

            pos = self.cif_label_pos
            xspan = (xmax - xmin) or 1.0
            if pos in ("right_out", "left_out"):
                max_pad = max(float(c.get("pad", 0.03)) for c in visible_series)
                pad_abs = max_pad * xspan
                if pos == "right_out":
                    ax.set_xlim(xmin, xmax + pad_abs)
                else:
                    ax.set_xlim(xmin - pad_abs, xmax)
                xmin, xmax = ax.get_xlim()
                xspan = (xmax - xmin) or 1.0

            for i, serie in enumerate(visible_series):
                color = serie.get("color", "crimson")
                disp_label = serie.get("plot_label") or serie.get("label") or f"CIF {i + 1}"
                cif_pad = float(serie.get("pad", 0.03))
                y0 = (ymin_orig - off_down) - i * spacing
                y1 = y0 + band_h * 0.6

                for tt, hkl, d in serie["peaks"]:
                    if xmin <= tt <= xmax:
                        ax.vlines(tt, y0, y1, colors=color, linewidth=1.2)

                if pos == "right_out":
                    x_text, dytext, ha = xmax - cif_pad * xspan, band_h * 0.75, "right"
                elif pos == "left_out":
                    x_text, dytext, ha = xmin + cif_pad * xspan, band_h * 0.75, "left"
                elif pos == "right_in":
                    x_text, dytext, ha = xmax - cif_pad * xspan, 0, "right"
                else:
                    x_text, dytext, ha = xmin + cif_pad * xspan, 0, "left"

                ax.text(
                    x_text, y0 + dytext, disp_label, va="bottom", ha=ha, fontsize=7, color=color,
                    bbox=dict(boxstyle="round,pad=0.1", edgecolor="none", fc="white", lw=0.6, alpha=0.9),
                    clip_on=False,
                )

    # ------------------------------------------------------------------
    def export_all_formats(self) -> None:
        base, _ = QFileDialog.getSaveFileName(self, "Export plot as…", "", "PNG (*.png)")
        if not base:
            return
        base = os.path.splitext(base)[0]
        errors = []
        if self.export_png_check.isChecked():
            try:
                self.plot.figure.savefig(base + ".png")
            except Exception as exc:
                errors.append(f"PNG: {exc}")
        if self.export_svg_check.isChecked():
            try:
                self.plot.figure.savefig(base + ".svg")
            except Exception as exc:
                errors.append(f"SVG: {exc}")
        if self.export_pdf_check.isChecked():
            try:
                self.plot.figure.savefig(base + ".pdf")
            except Exception as exc:
                errors.append(f"PDF: {exc}")
        if errors:
            QMessageBox.critical(self, "Export", "Some exports failed:\n" + "\n".join(errors))
        else:
            QMessageBox.information(self, "Export", "Plot exported successfully.")
