"""
qt_xas.py — XAS/XANES/EXAFS processing ported to Qt (M11), built on
xas_science.py's SpectrumStore/Spectrum/Operation object model (already
identity-stable via `Spectrum.sid` — no redesign needed here, unlike
qt_models.SpectrumLibrary which replaced main.py's four-parallel-list
anti-pattern from scratch).

Core slice ported faithfully from xas_processing_v10.py's XASUltimateApp:
object list (import ZIP/CSV/.prj, rename/duplicate/delete/export), Preview,
mu(E) Builder, Normalization + EXAFS/FT (Larch), Tools (edge definer),
Export (Athena .dat/.prj).

Two real bugs found and fixed in xas_science.py while building this (see
its own comments): (1) larch_normalize/larch_exafs_pipeline set pre1/pre2/
norm1/norm2/nnorm as Group attributes but never passed them as explicit
kwargs to pre_edge(), which only reads e0 that way — Larch silently used
its own auto-computed defaults instead for every call until now; (2)
compute_mu()'s deglitch/deglitch_z/deglitch_window parameters were dead
(accepted, never referenced) — exactly the gap the plan asked M11 to
"confirm." Fixed there; exposed here as real deglitch controls in the
mu(E) Builder tab (applied as an optional post-step after build_mu(), since
the interactive builder aligns two possibly-different-grid I0/It spectra
via build_mu()'s own interpolation — compute_mu() assumes a shared grid
already and isn't a drop-in replacement for that).

Athena-inspired additions (new "Analysis" tab, cheap/self-contained ones
only — PCA needs a new scikit-learn dependency better co-scoped with M16;
self-absorption correction is a substantial standalone physics feature):
merge/average repeat scans, difference spectra, linear combination fitting
(NNLS-based, 2+ references).

Deliberately deferred to a documented follow-up (not half-implemented):
the Pre-processing tab (smoothing preview/apply, Bragg angle/energy
correction, Mode C interactive click-based feature alignment — all fairly
beamline/angle-dispersive-specific), the separate I0 baseline-"Fit" tab,
PCA, self-absorption correction, the CSV Builder export tool.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMenu, QMessageBox, QPushButton,
    QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QTextEdit,
    QVBoxLayout, QWidget,
)

from core.qt_widgets import PlotWidget
from xas.xas_science import (
    _SCIPY_AVAILABLE,
    Operation,
    Spectrum,
    SpectrumStore,
    TiePoint,
    angle_energy_correction_bragg,
    apply_alignment_mode_c,
    smooth_spectrum,
    _classify_kind_from_name,
    _extract_energy_angle_signal,
    _interp_to_grid,
    _periodic_table_symbols,
    _uid,
    build_mu,
    deglitch_mu,
    edge_text,
    export_athena_column,
    export_athena_prj_best_effort,
    infer_edge_label_from_roi_scaled,
    larch_exafs_pipeline,
    larch_normalize,
    read_athena_prj,
    read_csv_dataset,
    read_easyxafs_zip,
    require_larch,
)

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]


def _to_float(text: str, default: Optional[float] = None) -> Optional[float]:
    try:
        return float((text or "").strip())
    except (TypeError, ValueError):
        return default


def _to_int(text: str, default: int) -> int:
    try:
        return int(float((text or "").strip()))
    except (TypeError, ValueError):
        return default


class XasWorkspace(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.store = SpectrumStore()
        self.selected_sid: Optional[str] = None
        self._build_ui()
        self._refresh_all()

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

        import_row = QHBoxLayout()
        for label, handler in [
            ("ZIP…", self.import_zips), ("CSV…", self.import_csvs),
            (".prj…", self.import_prj), ("Clear", self.clear_all),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(handler)
            import_row.addWidget(btn)
        left_layout.addLayout(import_row)

        left_layout.addWidget(QLabel("Imported spectra (objects)"))
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Name", "Kind", "Edge", "E0", "E range"])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_table_context_menu)
        left_layout.addWidget(self.table, 1)

        self.status_label = QLabel("Ready.")
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_preview_tab(), "Preview")
        self.tabs.addTab(self._build_preproc_tab(), "Pre-processing")
        self.tabs.addTab(self._build_mu_tab(), "μ(E) Builder")
        self.tabs.addTab(self._build_norm_tab(), "Normalization / EXAFS")
        self.tabs.addTab(self._build_analysis_tab(), "Analysis")
        self.tabs.addTab(self._build_tools_tab(), "Tools")
        self.tabs.addTab(self._build_export_tab(), "Export")
        self.tabs.addTab(self._build_mass_tab(), "Sample mass")
        right_layout.addWidget(self.tabs)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

    def _set_status(self, msg: str) -> None:
        self.status_label.setText(msg)

    # ------------------------------------------------------------------
    # Preview tab
    # ------------------------------------------------------------------
    def _build_preview_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        self.preview_plot = PlotWidget(figsize=(7, 5))
        layout.addWidget(self.preview_plot)
        return w

    def _plot_selected_preview(self) -> None:
        if self.selected_sid is None:
            self.preview_plot.clear("Preview")
            return
        sp = self.store.get(self.selected_sid)
        ax = self.preview_plot.ax
        ax.clear()
        ax.plot(sp.energy, sp.y, lw=1.2, color=COLORS[0], label=sp.name)
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel(sp.units)
        ax.set_title(f"{sp.label} — {sp.name} [{sp.kind}]")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        self.preview_plot.figure.tight_layout()
        self.preview_plot.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Pre-processing tab: smoothing + angle/energy correction (port of the
    # Tk XASUltimateApp's Pre-processing tab; the deferred M11 slice).
    # ------------------------------------------------------------------
    _SM_PARAM_DEFS = {
        # method -> [(label, default), ...] driving the generic param fields
        "Savitzky-Golay": [("window", "11"), ("poly", "3")],
        "Median+SG": [("median", "9"), ("sg window", "11"), ("sg poly", "3")],
        "Whittaker": [("λ", "1e5"), ("d", "2")],
        "Spline": [("s", "0.0")],
    }

    def _build_preproc_tab(self) -> QWidget:
        w = QWidget()
        layout = QHBoxLayout(w)

        ctrl = QWidget()
        ctrl.setMaximumWidth(360)
        ctrl_layout = QVBoxLayout(ctrl)

        ctrl_layout.addWidget(QLabel("Smoothing"))
        self.sm_target_combo = QComboBox()
        ctrl_layout.addWidget(self.sm_target_combo)

        sm_method_row = QHBoxLayout()
        sm_method_row.addWidget(QLabel("Method"))
        self.sm_method_combo = QComboBox()
        methods = ["Savitzky-Golay", "Median+SG", "Whittaker"] + (["Spline"] if _SCIPY_AVAILABLE else [])
        self.sm_method_combo.addItems(methods)
        self.sm_method_combo.currentTextChanged.connect(self._rebuild_sm_params)
        sm_method_row.addWidget(self.sm_method_combo)
        ctrl_layout.addLayout(sm_method_row)

        self.sm_param_labels = [QLabel("") for _ in range(3)]
        self.sm_param_edits = [QLineEdit() for _ in range(3)]
        params_row = QHBoxLayout()
        for lbl, edit in zip(self.sm_param_labels, self.sm_param_edits):
            edit.setMaximumWidth(60)
            params_row.addWidget(lbl)
            params_row.addWidget(edit)
        ctrl_layout.addLayout(params_row)
        self._rebuild_sm_params(self.sm_method_combo.currentText())

        sm_btn_row = QHBoxLayout()
        sm_preview_btn = QPushButton("Preview smoothing")
        sm_preview_btn.clicked.connect(self.preview_smoothing)
        sm_apply_btn = QPushButton("Apply → new object")
        sm_apply_btn.clicked.connect(self.apply_smoothing)
        sm_btn_row.addWidget(sm_preview_btn)
        sm_btn_row.addWidget(sm_apply_btn)
        ctrl_layout.addLayout(sm_btn_row)

        ctrl_layout.addWidget(QLabel("Angle/E correction"))
        self.ang_mode_combo = QComboBox()
        self.ang_mode_combo.addItems(["A: Bragg+Linear", "B: Bragg only", "C: Feature alignment (click)"])
        ctrl_layout.addWidget(self.ang_mode_combo)

        before_row = QHBoxLayout()
        before_row.addWidget(QLabel("Before"))
        self.ang_before_combo = QComboBox()
        before_row.addWidget(self.ang_before_combo, 1)
        ctrl_layout.addLayout(before_row)

        after_row = QHBoxLayout()
        after_row.addWidget(QLabel("After"))
        self.ang_after_combo = QComboBox()
        after_row.addWidget(self.ang_after_combo, 1)
        ctrl_layout.addLayout(after_row)

        c_row = QHBoxLayout()
        self.ang_fit_linear_check = QCheckBox("Fit linear calibration (Mode A)")
        self.ang_fit_linear_check.setChecked(True)
        c_row.addWidget(self.ang_fit_linear_check)
        c_row.addWidget(QLabel("Mode C model"))
        self.mode_c_model_combo = QComboBox()
        self.mode_c_model_combo.addItems(["shift", "affine"])
        c_row.addWidget(self.mode_c_model_combo)
        ctrl_layout.addLayout(c_row)

        ang_btn_row = QHBoxLayout()
        overlay_btn = QPushButton("Plot overlay")
        overlay_btn.clicked.connect(self.plot_mode_c_overlay)
        pick_btn = QPushButton("Pick pair")
        pick_btn.clicked.connect(self.start_picking_pair)
        apply_ang_btn = QPushButton("Apply → new object")
        apply_ang_btn.clicked.connect(self.apply_angle_correction)
        ang_btn_row.addWidget(overlay_btn)
        ang_btn_row.addWidget(pick_btn)
        ang_btn_row.addWidget(apply_ang_btn)
        ctrl_layout.addLayout(ang_btn_row)

        ctrl_layout.addWidget(QLabel("Mode C tie points (click BEFORE, then AFTER)"))
        self.tp_table = QTableWidget(0, 3)
        self.tp_table.setHorizontalHeaderLabels(["E before", "E after", "ΔE"])
        self.tp_table.setEditTriggers(QTableWidget.NoEditTriggers)
        ctrl_layout.addWidget(self.tp_table, 1)
        clear_tp_btn = QPushButton("Clear tie points")
        clear_tp_btn.clicked.connect(self.clear_tiepoints)
        ctrl_layout.addWidget(clear_tp_btn)

        layout.addWidget(ctrl)
        self.preproc_plot = PlotWidget(figsize=(7, 5))
        self.preproc_plot.canvas.mpl_connect("button_press_event", self._on_preproc_click)
        layout.addWidget(self.preproc_plot, 1)

        self.tiepoints: List[TiePoint] = []
        self._pick_state = {"active": False, "waiting": "before", "last_before": None}
        self._pick_lines: Dict[str, Any] = {}
        return w

    def _rebuild_sm_params(self, method: str) -> None:
        defs = self._SM_PARAM_DEFS.get(method, [])
        for i, (lbl, edit) in enumerate(zip(self.sm_param_labels, self.sm_param_edits)):
            if i < len(defs):
                lbl.setText(defs[i][0])
                edit.setText(defs[i][1])
                lbl.show()
                edit.show()
            else:
                lbl.hide()
                edit.hide()

    def _sm_method_and_params(self):
        method_ui = self.sm_method_combo.currentText()
        vals = [e.text() for e in self.sm_param_edits]
        if method_ui == "Savitzky-Golay":
            return "savitzky-golay", {"window": _to_int(vals[0], 11), "poly": _to_int(vals[1], 3)}
        if method_ui == "Median+SG":
            return "median+sg", {"median_window": _to_int(vals[0], 9), "sg_window": _to_int(vals[1], 11), "sg_poly": _to_int(vals[2], 3)}
        if method_ui == "Whittaker":
            return "whittaker", {"lam": _to_float(vals[0], 1e5), "d": _to_int(vals[1], 2)}
        return "spline", {"s": _to_float(vals[0], 0.0)}

    def preview_smoothing(self) -> None:
        sp = self.store.find_by_name(self.sm_target_combo.currentText())
        if sp is None:
            QMessageBox.warning(self, "Smoothing", "Select a target spectrum.")
            return
        method, params = self._sm_method_and_params()
        try:
            smoothed = smooth_spectrum(sp.y, method, params)
        except Exception as exc:
            QMessageBox.critical(self, "Smoothing preview error", str(exc))
            return
        fig = self.preproc_plot.figure
        fig.clf()
        ax = fig.add_subplot(111)
        self.preproc_plot.ax = ax
        ax.plot(sp.energy, sp.y, lw=1.0, alpha=0.6, label="raw")
        ax.plot(sp.energy, smoothed, lw=1.4, label="smoothed")
        ax.set_xlabel("Energy (eV)"); ax.set_ylabel(sp.units)
        ax.set_title(f"Smoothing preview — {sp.name} ({self.sm_method_combo.currentText()})")
        ax.legend(fontsize=8); ax.grid(alpha=0.25)
        fig.tight_layout()
        self.preproc_plot.canvas.draw_idle()

    def apply_smoothing(self) -> None:
        sp = self.store.find_by_name(self.sm_target_combo.currentText())
        if sp is None:
            QMessageBox.warning(self, "Smoothing", "Select a target spectrum.")
            return
        method, params = self._sm_method_and_params()
        try:
            smoothed = smooth_spectrum(sp.y, method, params)
        except Exception as exc:
            QMessageBox.critical(self, "Smoothing error", str(exc))
            return
        sp2 = sp.copy(new_name=f"{sp.name}_sm", new_kind=sp.kind)
        sp2.y = np.asarray(smoothed, float)
        sp2.history.append(Operation("smooth", {"method": method, **params}))
        self.store.add(sp2)
        self._refresh_all()
        self._set_status(f"Smoothed → {sp2.name}")

    # ---- Angle/E correction ----
    def plot_mode_c_overlay(self) -> None:
        b = self.store.find_by_name(self.ang_before_combo.currentText())
        a = self.store.find_by_name(self.ang_after_combo.currentText())
        if b is None or a is None:
            QMessageBox.warning(self, "Overlay", "Select BEFORE and AFTER spectra.")
            return
        fig = self.preproc_plot.figure
        fig.clf()
        ax = fig.add_subplot(111)
        self.preproc_plot.ax = ax
        self._pick_lines["before"], = ax.plot(b.energy, b.y, lw=1.2, label=f"before: {b.name}")
        self._pick_lines["after"], = ax.plot(a.energy, a.y, lw=1.2, label=f"after: {a.name}")
        ax.set_xlabel("Energy (eV)"); ax.set_ylabel(a.units)
        ax.set_title("Mode C overlay")
        ax.legend(fontsize=8); ax.grid(alpha=0.25)
        fig.tight_layout()
        self.preproc_plot.canvas.draw_idle()

    def start_picking_pair(self) -> None:
        if "before" not in self._pick_lines or "after" not in self._pick_lines:
            QMessageBox.information(self, "Mode C", "Plot the overlay first.")
            return
        self._pick_state.update({"active": True, "waiting": "before", "last_before": None})
        self._set_status("Picking: click the BEFORE feature point.")

    def _on_preproc_click(self, event) -> None:
        st = self._pick_state
        if not st.get("active") or event.inaxes is None or event.xdata is None:
            return
        role = st.get("waiting", "before")
        line = self._pick_lines.get(role)
        if line is None:
            return
        x = np.asarray(line.get_xdata(), float)
        y = np.asarray(line.get_ydata(), float)
        if x.size == 0:
            return
        idx = int(np.nanargmin(np.abs(x - float(event.xdata))))
        ex, ey = float(x[idx]), float(y[idx])
        event.inaxes.plot([ex], [ey], marker="o", ms=6, linestyle="None", color="black")
        self.preproc_plot.canvas.draw_idle()

        if role == "before":
            st["last_before"] = ex
            st["waiting"] = "after"
            self._set_status(f"Picked BEFORE at {ex:.3f} eV. Now click AFTER.")
        else:
            eb = st.get("last_before")
            if eb is None:
                st["waiting"] = "before"
                return
            self.tiepoints.append(TiePoint(e_before=float(eb), e_after=ex))
            self._refresh_tiepoints_table()
            st.update({"waiting": "before", "active": False})
            self._set_status(f"Added tie point: {eb:.3f} → {ex:.3f} (ΔE = {ex - eb:+.3f} eV).")

    def _refresh_tiepoints_table(self) -> None:
        self.tp_table.setRowCount(len(self.tiepoints))
        for row, tp in enumerate(self.tiepoints):
            self.tp_table.setItem(row, 0, QTableWidgetItem(f"{tp.e_before:.3f}"))
            self.tp_table.setItem(row, 1, QTableWidgetItem(f"{tp.e_after:.3f}"))
            self.tp_table.setItem(row, 2, QTableWidgetItem(f"{tp.e_before - tp.e_after:+.3f}"))

    def clear_tiepoints(self) -> None:
        self.tiepoints.clear()
        self._refresh_tiepoints_table()
        self._set_status("Cleared tie points.")

    def apply_angle_correction(self) -> None:
        mode_ui = self.ang_mode_combo.currentText()
        sp_before = self.store.find_by_name(self.ang_before_combo.currentText())
        if sp_before is None:
            QMessageBox.warning(self, "Angle/E correction", "Select a BEFORE spectrum.")
            return

        if mode_ui.startswith("C"):
            sp_after = self.store.find_by_name(self.ang_after_combo.currentText())
            if sp_after is None:
                QMessageBox.warning(self, "Mode C", "Select an AFTER spectrum for alignment.")
                return
            if not self.tiepoints:
                QMessageBox.warning(self, "Mode C", "Add at least one tie point first.")
                return
            try:
                e_corr, diag = apply_alignment_mode_c(sp_after.energy, self.tiepoints, model=self.mode_c_model_combo.currentText())
            except Exception as exc:
                QMessageBox.critical(self, "Mode C error", str(exc))
                return
            sp2 = sp_after.copy(new_name=f"{sp_after.name}_Ealign", new_kind=f"corrected_{sp_after.kind}")
            sp2.energy = np.asarray(e_corr, float)
            sp2.history.append(Operation("align_mode_c", {"before": sp_before.name, "after": sp_after.name, **diag}))
            self.store.add(sp2)
            self._refresh_all()
            self._set_status(f"Mode C alignment → {sp2.name}")
            return

        if sp_before.angle is None or not np.isfinite(sp_before.angle).any():
            QMessageBox.critical(self, "Angle/E correction", "BEFORE spectrum has no valid angle column.")
            return
        mode = "A" if mode_ui.startswith("A") else "B"
        try:
            e_corr, diag = angle_energy_correction_bragg(
                sp_before.angle, sp_before.energy, sp_before.meta.get("scan_def", {}) or {},
                mode=mode, fit_linear=self.ang_fit_linear_check.isChecked(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Angle/E correction error", str(exc))
            return
        sp2 = sp_before.copy(new_name=f"{sp_before.name}_Ebragg{mode}", new_kind=f"corrected_{sp_before.kind}")
        sp2.energy = np.asarray(e_corr, float)
        sp2.history.append(Operation("angle_energy_correction", {"mode": mode, **diag}))
        self.store.add(sp2)
        self._refresh_all()
        self._set_status(f"Bragg correction Mode {mode} → {sp2.name}")

    # ------------------------------------------------------------------
    # mu(E) Builder tab
    # ------------------------------------------------------------------
    def _build_mu_tab(self) -> QWidget:
        w = QWidget()
        layout = QHBoxLayout(w)

        ctrl = QWidget()
        ctrl.setMaximumWidth(320)
        ctrl_layout = QVBoxLayout(ctrl)

        ctrl_layout.addWidget(QLabel("I0 selection"))
        self.mu_i0_combo = QComboBox()
        self.mu_i0_combo.currentIndexChanged.connect(self._preview_mu)
        ctrl_layout.addWidget(self.mu_i0_combo)

        ctrl_layout.addWidget(QLabel("It spectra (multi-select)"))
        self.mu_it_list = QListWidget()
        self.mu_it_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.mu_it_list.itemSelectionChanged.connect(self._preview_mu)
        ctrl_layout.addWidget(self.mu_it_list, 1)

        log_row = QHBoxLayout()
        log_row.addWidget(QLabel("log"))
        self.mu_log_combo = QComboBox()
        self.mu_log_combo.addItems(["ln", "log10"])
        self.mu_log_combo.currentTextChanged.connect(self._preview_mu)
        log_row.addWidget(self.mu_log_combo)
        ctrl_layout.addLayout(log_row)

        self.mu_deglitch_check = QCheckBox("Deglitch")
        ctrl_layout.addWidget(self.mu_deglitch_check)
        degl_row = QHBoxLayout()
        degl_row.addWidget(QLabel("z"))
        self.mu_deglitch_z_edit = QLineEdit("6.0")
        self.mu_deglitch_z_edit.setMaximumWidth(50)
        degl_row.addWidget(self.mu_deglitch_z_edit)
        degl_row.addWidget(QLabel("window"))
        self.mu_deglitch_window_edit = QLineEdit("21")
        self.mu_deglitch_window_edit.setMaximumWidth(50)
        degl_row.addWidget(self.mu_deglitch_window_edit)
        ctrl_layout.addLayout(degl_row)

        preview_btn = QPushButton("Preview μ")
        preview_btn.clicked.connect(self._preview_mu)
        ctrl_layout.addWidget(preview_btn)
        compute_btn = QPushButton("Compute μ → new objects")
        compute_btn.setObjectName("Primary")
        compute_btn.clicked.connect(self.compute_mu_selected)
        ctrl_layout.addWidget(compute_btn)
        ctrl_layout.addStretch(1)
        layout.addWidget(ctrl)

        self.mu_plot = PlotWidget(figsize=(7, 5))
        layout.addWidget(self.mu_plot, 1)
        return w

    def _mu_deglitch_params(self):
        enabled = self.mu_deglitch_check.isChecked()
        z = _to_float(self.mu_deglitch_z_edit.text(), 6.0)
        window = _to_int(self.mu_deglitch_window_edit.text(), 21)
        return enabled, z, window

    def _build_mu(self, i0: Spectrum, it: Spectrum) -> np.ndarray:
        mu = build_mu(i0_energy=i0.energy, i0=i0.y, it_energy=it.energy, it=it.y, log_mode=self.mu_log_combo.currentText())
        enabled, z, window = self._mu_deglitch_params()
        if enabled:
            mu = deglitch_mu(mu, z=z, window=window)
        return mu

    def _preview_mu(self) -> None:
        i0_name = self.mu_i0_combo.currentText()
        if not i0_name:
            self.mu_plot.clear("μ preview")
            return
        i0 = self.store.find_by_name(i0_name)
        it_items = self.mu_it_list.selectedItems()
        if i0 is None or not it_items:
            self.mu_plot.clear("μ preview")
            return
        it = self.store.find_by_name(it_items[0].text())
        if it is None:
            return
        try:
            mu = self._build_mu(i0, it)
        except Exception as exc:
            QMessageBox.critical(self, "μ preview error", str(exc))
            return
        ax = self.mu_plot.ax
        ax.clear()
        ax.plot(i0.energy, mu, lw=1.2, color=COLORS[0], label=f"μ from {it.name}")
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("arb.")
        ax.set_title(f"μ preview — I0={i0.name}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        self.mu_plot.figure.tight_layout()
        self.mu_plot.canvas.draw_idle()

    def compute_mu_selected(self) -> None:
        i0_name = self.mu_i0_combo.currentText()
        if not i0_name:
            QMessageBox.warning(self, "μ builder", "Select an I0 spectrum.")
            return
        i0 = self.store.find_by_name(i0_name)
        if i0 is None:
            return
        it_items = self.mu_it_list.selectedItems()
        if not it_items:
            QMessageBox.warning(self, "μ builder", "Select at least one It spectrum.")
            return

        last = None
        for item in it_items:
            it = self.store.find_by_name(item.text())
            if it is None:
                continue
            try:
                mu = self._build_mu(i0, it)
            except Exception as exc:
                QMessageBox.critical(self, "μ builder error", str(exc))
                continue
            sp_mu = it.copy(new_name=f"{it.name}_mu", new_kind="mu")
            sp_mu.energy = np.asarray(i0.energy, float)
            sp_mu.y = np.asarray(mu, float)
            sp_mu.label = it.label
            sp_mu.e0 = it.e0
            enabled, z, window = self._mu_deglitch_params()
            sp_mu.history.append(Operation("mu_builder", {"I0": i0.name, "It": it.name, "log": self.mu_log_combo.currentText(), "deglitch": enabled, "deglitch_z": z, "deglitch_window": window}))
            self.store.add(sp_mu)
            last = sp_mu

        self._refresh_all()
        if last is not None:
            self._set_status(f"Computed μ for {len(it_items)} It spectrum/spectra.")

    # ------------------------------------------------------------------
    # Normalization / EXAFS tab
    # ------------------------------------------------------------------
    def _build_norm_tab(self) -> QWidget:
        w = QWidget()
        layout = QHBoxLayout(w)

        ctrl = QWidget()
        ctrl.setMaximumWidth(360)
        ctrl_layout = QVBoxLayout(ctrl)

        e0_row = QHBoxLayout()
        e0_row.addWidget(QLabel("E0 method"))
        self.norm_e0_combo = QComboBox()
        self.norm_e0_combo.addItems(["larch", "deriv", "manual"])
        e0_row.addWidget(self.norm_e0_combo)
        e0_row.addWidget(QLabel("manual"))
        self.norm_e0_manual_edit = QLineEdit()
        self.norm_e0_manual_edit.setMaximumWidth(70)
        e0_row.addWidget(self.norm_e0_manual_edit)
        ctrl_layout.addLayout(e0_row)

        grid_row1 = QHBoxLayout()
        grid_row1.addWidget(QLabel("pre1"))
        self.norm_pre1_edit = QLineEdit("-150")
        grid_row1.addWidget(self.norm_pre1_edit)
        grid_row1.addWidget(QLabel("pre2"))
        self.norm_pre2_edit = QLineEdit("-50")
        grid_row1.addWidget(self.norm_pre2_edit)
        ctrl_layout.addLayout(grid_row1)

        grid_row2 = QHBoxLayout()
        grid_row2.addWidget(QLabel("norm1"))
        self.norm_norm1_edit = QLineEdit("30")
        grid_row2.addWidget(self.norm_norm1_edit)
        grid_row2.addWidget(QLabel("norm2"))
        self.norm_norm2_edit = QLineEdit("150")
        grid_row2.addWidget(self.norm_norm2_edit)
        grid_row2.addWidget(QLabel("nnorm"))
        self.norm_nnorm_edit = QLineEdit("1")
        grid_row2.addWidget(self.norm_nnorm_edit)
        ctrl_layout.addLayout(grid_row2)

        self.norm_smooth_check = QCheckBox("Smooth for E0/derivative only")
        self.norm_smooth_check.setChecked(True)
        ctrl_layout.addWidget(self.norm_smooth_check)

        normalize_btn = QPushButton("Normalize selected μ → new objects")
        normalize_btn.setObjectName("Primary")
        normalize_btn.clicked.connect(self.normalize_selected)
        ctrl_layout.addWidget(normalize_btn)

        ctrl_layout.addWidget(QLabel("EXAFS / FT (Larch autobk + xftf)"))
        ex_row1 = QHBoxLayout()
        ex_row1.addWidget(QLabel("rbkg"))
        self.exafs_rbkg_edit = QLineEdit("1.0")
        ex_row1.addWidget(self.exafs_rbkg_edit)
        ex_row1.addWidget(QLabel("kmin"))
        self.exafs_kmin_edit = QLineEdit("0")
        ex_row1.addWidget(self.exafs_kmin_edit)
        ex_row1.addWidget(QLabel("kmax"))
        self.exafs_kmax_edit = QLineEdit("15")
        ex_row1.addWidget(self.exafs_kmax_edit)
        ctrl_layout.addLayout(ex_row1)

        ex_row2 = QHBoxLayout()
        ex_row2.addWidget(QLabel("dk"))
        self.exafs_dk_edit = QLineEdit("0.1")
        ex_row2.addWidget(self.exafs_dk_edit)
        ex_row2.addWidget(QLabel("k-weight"))
        self.exafs_kweight_edit = QLineEdit("2")
        ex_row2.addWidget(self.exafs_kweight_edit)
        ctrl_layout.addLayout(ex_row2)

        ex_row3 = QHBoxLayout()
        ex_row3.addWidget(QLabel("window"))
        self.exafs_window_combo = QComboBox()
        self.exafs_window_combo.addItems(["hanning", "kaiser", "parzen", "welch", "sine", "gaussian"])
        ex_row3.addWidget(self.exafs_window_combo)
        ex_row3.addWidget(QLabel("rmax_out"))
        self.exafs_rmax_edit = QLineEdit("10")
        ex_row3.addWidget(self.exafs_rmax_edit)
        ctrl_layout.addLayout(ex_row3)

        exafs_btn = QPushButton("Compute χ(k) + FT → new objects")
        exafs_btn.clicked.connect(self.exafs_selected)
        ctrl_layout.addWidget(exafs_btn)

        ctrl_layout.addWidget(QLabel("Select μ spectra"))
        self.norm_mu_list = QListWidget()
        self.norm_mu_list.setSelectionMode(QListWidget.ExtendedSelection)
        ctrl_layout.addWidget(self.norm_mu_list, 1)

        layout.addWidget(ctrl)
        self.norm_plot = PlotWidget(figsize=(7, 5))
        layout.addWidget(self.norm_plot, 1)
        return w

    def _norm_params(self):
        e0_method = self.norm_e0_combo.currentText()
        e0_manual = _to_float(self.norm_e0_manual_edit.text()) if e0_method == "manual" else None
        pre1 = _to_float(self.norm_pre1_edit.text(), -150.0)
        pre2 = _to_float(self.norm_pre2_edit.text(), -50.0)
        norm1 = _to_float(self.norm_norm1_edit.text(), 30.0)
        norm2 = _to_float(self.norm_norm2_edit.text(), 150.0)
        nnorm = _to_int(self.norm_nnorm_edit.text(), 1)
        smooth_for_e0 = ("savitzky-golay", {"window": 11, "poly": 3}) if self.norm_smooth_check.isChecked() else None
        return e0_method, e0_manual, pre1, pre2, norm1, norm2, nnorm, smooth_for_e0

    def normalize_selected(self) -> None:
        items = self.norm_mu_list.selectedItems()
        if not items:
            QMessageBox.warning(self, "Normalization", "Select μ spectra.")
            return
        e0_method, e0_manual, pre1, pre2, norm1, norm2, nnorm, smooth_for_e0 = self._norm_params()

        last = None
        for item in items:
            sp = self.store.find_by_name(item.text())
            if sp is None:
                continue
            try:
                out = larch_normalize(sp.energy, sp.y, e0_method=e0_method, e0_manual=e0_manual, pre1=pre1, pre2=pre2, norm1=norm1, norm2=norm2, nnorm=nnorm, smooth_for_e0=smooth_for_e0)
            except Exception as exc:
                QMessageBox.critical(self, "Normalization error", str(exc))
                continue
            sp_norm = sp.copy(new_name=f"{sp.name}_norm", new_kind="norm")
            sp_norm.y = out["norm"]; sp_norm.e0 = out["e0"]
            sp_norm.history.append(Operation("normalize", {"e0_method": e0_method, "e0": out["e0"]}))
            sp_flat = sp.copy(new_name=f"{sp.name}_flat", new_kind="flat")
            sp_flat.y = out["flat"]; sp_flat.e0 = out["e0"]
            sp_flat.history.append(Operation("normalize_flat", {"e0": out["e0"]}))
            self.store.add(sp_norm); self.store.add(sp_flat)
            last = (sp, out)

        self._refresh_all()
        if last is not None:
            sp, out = last
            ax = self.norm_plot.ax
            ax.clear()
            ax.plot(sp.energy, sp.y, lw=1.1, label="μ(E)")
            ax.plot(sp.energy, out["norm"], lw=1.1, label="norm")
            ax.plot(sp.energy, out["deriv"], lw=1.0, label="dμ/dE", alpha=0.6)
            for val in (out["anchors"].get(k) for k in ("pre1", "pre2", "norm1", "norm2")):
                if val is not None:
                    ax.axvline(float(val), ls="--", lw=1.0, alpha=0.7)
            ax.set_xlabel("Energy (eV)"); ax.set_ylabel("arb.")
            ax.set_title(f"{sp.label} — E0={out['e0']:.2f}")
            ax.legend(fontsize=8); ax.grid(alpha=0.25)
            self.norm_plot.figure.tight_layout()
            self.norm_plot.canvas.draw_idle()

    def exafs_selected(self) -> None:
        items = self.norm_mu_list.selectedItems()
        if not items:
            QMessageBox.warning(self, "EXAFS/FT", "Select μ spectra.")
            return
        e0_method, e0_manual, pre1, pre2, norm1, norm2, nnorm, smooth_for_e0 = self._norm_params()
        rbkg = _to_float(self.exafs_rbkg_edit.text(), 1.0)
        kmin = _to_float(self.exafs_kmin_edit.text(), 0.0)
        kmax = _to_float(self.exafs_kmax_edit.text(), 15.0)
        dk = _to_float(self.exafs_dk_edit.text(), 0.1)
        kweight = _to_int(self.exafs_kweight_edit.text(), 2)
        window = self.exafs_window_combo.currentText()
        rmax_out = _to_float(self.exafs_rmax_edit.text(), 10.0)

        last = None
        for item in items:
            sp = self.store.find_by_name(item.text())
            if sp is None:
                continue
            try:
                out = larch_exafs_pipeline(
                    sp.energy, sp.y, e0_method=e0_method, e0_manual=e0_manual, pre1=pre1, pre2=pre2,
                    norm1=norm1, norm2=norm2, nnorm=nnorm, rbkg=rbkg, kmin=kmin, kmax=kmax, dk=dk,
                    kweight=kweight, window=window, rmax_out=rmax_out, smooth_for_e0=smooth_for_e0,
                )
            except Exception as exc:
                QMessageBox.critical(self, "EXAFS/FT error", str(exc))
                continue

            sp_norm = sp.copy(new_name=f"{sp.name}_norm", new_kind="norm"); sp_norm.y = out["norm"]; sp_norm.e0 = out["e0"]
            sp_norm.history.append(Operation("normalize", {"e0_method": e0_method}))
            sp_flat = sp.copy(new_name=f"{sp.name}_flat", new_kind="flat"); sp_flat.y = out["flat"]; sp_flat.e0 = out["e0"]
            sp_flat.history.append(Operation("normalize_flat", {"e0": out["e0"]}))
            self.store.add(sp_norm); self.store.add(sp_flat)

            sp_chi = sp.copy(new_name=f"{sp.name}_chi", new_kind="chi(k)"); sp_chi.energy = out["k"]; sp_chi.y = out["chi"]; sp_chi.e0 = out["e0"]
            sp_chi.history.append(Operation("autobk", {"rbkg": rbkg, "kmin": kmin, "kmax": kmax, "dk": dk}))
            self.store.add(sp_chi)

            sp_chikw = sp.copy(new_name=f"{sp.name}_chi_k{kweight}", new_kind=f"chi(k)*k^{kweight}")
            sp_chikw.energy = out["k"]; sp_chikw.y = out["chi_kw"]; sp_chikw.e0 = out["e0"]
            sp_chikw.history.append(Operation("kweight", {"kweight": kweight}))
            self.store.add(sp_chikw)

            sp_ft = sp.copy(new_name=f"{sp.name}_FTmag", new_kind="FT|chi|"); sp_ft.energy = out["r"]; sp_ft.y = out["chir_mag"]; sp_ft.e0 = out["e0"]
            sp_ft.history.append(Operation("xftf", {"kmin": kmin, "kmax": kmax, "dk": dk, "kweight": kweight, "window": window, "rmax_out": rmax_out}))
            self.store.add(sp_ft)
            last = (sp, out, kweight)

        self._refresh_all()
        if last is not None:
            sp, out, kweight = last
            ax = self.norm_plot.ax
            ax.clear()
            ax.plot(out["k"], out["chi"], lw=1.1, label="χ(k)")
            ax.plot(out["k"], out["chi_kw"], lw=1.1, label=f"χ(k)*k^{kweight}")
            ax.plot(out["r"], out["chir_mag"], lw=1.1, label="|FT|")
            ax.set_xlabel("k (1/Å) / R (Å)"); ax.set_ylabel("arb.")
            ax.set_title(f"{sp.label} — E0={out['e0']:.2f}")
            ax.legend(fontsize=8); ax.grid(alpha=0.25)
            self.norm_plot.figure.tight_layout()
            self.norm_plot.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Analysis tab (Athena-inspired additions)
    # ------------------------------------------------------------------
    def _build_analysis_tab(self) -> QWidget:
        w = QWidget()
        layout = QHBoxLayout(w)

        ctrl = QWidget()
        ctrl.setMaximumWidth(340)
        ctrl_layout = QVBoxLayout(ctrl)

        ctrl_layout.addWidget(QLabel("Select spectra (multi-select)"))
        self.analysis_list = QListWidget()
        self.analysis_list.setSelectionMode(QListWidget.ExtendedSelection)
        ctrl_layout.addWidget(self.analysis_list, 1)

        merge_row = QHBoxLayout()
        merge_btn = QPushButton("Average selected → new object")
        merge_btn.clicked.connect(self.merge_average_selected)
        merge_row.addWidget(merge_btn)
        sum_btn = QPushButton("Sum selected → new object")
        sum_btn.clicked.connect(self.sum_selected)
        merge_row.addWidget(sum_btn)
        ctrl_layout.addLayout(merge_row)

        diff_btn = QPushButton("Difference (1st − 2nd selected) → new object")
        diff_btn.clicked.connect(self.difference_selected)
        ctrl_layout.addWidget(diff_btn)

        ctrl_layout.addWidget(QLabel("Linear combination fit: last selected = target,\nothers = references"))
        lcf_btn = QPushButton("Fit linear combination")
        lcf_btn.clicked.connect(self.linear_combination_fit_selected)
        ctrl_layout.addWidget(lcf_btn)

        pca_btn = QPushButton("PCA across selected (species count)")
        pca_btn.clicked.connect(self.pca_selected)
        ctrl_layout.addWidget(pca_btn)

        self.analysis_result_text = QTextEdit()
        self.analysis_result_text.setReadOnly(True)
        self.analysis_result_text.setMaximumHeight(140)
        ctrl_layout.addWidget(self.analysis_result_text)
        ctrl_layout.addStretch(1)
        layout.addWidget(ctrl)

        self.analysis_plot = PlotWidget(figsize=(7, 5))
        layout.addWidget(self.analysis_plot, 1)
        return w

    def _analysis_selected_spectra(self) -> List[Spectrum]:
        out = []
        for item in self.analysis_list.selectedItems():
            sp = self.store.find_by_name(item.text())
            if sp is not None:
                out.append(sp)
        return out

    def merge_average_selected(self) -> None:
        self._combine_selected("average")

    def sum_selected(self) -> None:
        self._combine_selected("sum")

    def _combine_selected(self, op: str) -> None:
        """Average (merge of repeat scans) or sum (e.g. adding up detector
        channels / partial acquisitions) of the selected spectra, on the
        first-selected spectrum's energy grid."""
        specs = self._analysis_selected_spectra()
        if len(specs) < 2:
            QMessageBox.warning(self, "Combine", f"Select at least 2 spectra to {op}.")
            return
        ref = specs[0]
        stacked = [ref.y]
        for sp in specs[1:]:
            stacked.append(_interp_to_grid(sp.energy, sp.y, ref.energy))
        combined = np.sum(np.vstack(stacked), axis=0) if op == "sum" else np.mean(np.vstack(stacked), axis=0)

        suffix = "sum" if op == "sum" else "avg"
        sp_new = ref.copy(new_name=f"{ref.name}_{suffix}{len(specs)}", new_kind=ref.kind)
        sp_new.y = combined
        sp_new.history.append(Operation(f"merge_{op}", {"members": [s.name for s in specs]}))
        self.store.add(sp_new)
        self._refresh_all()

        ax = self.analysis_plot.ax
        ax.clear()
        for i, sp in enumerate(specs):
            ax.plot(sp.energy, sp.y, lw=0.9, alpha=0.5, color=COLORS[i % len(COLORS)], label=sp.name)
        ax.plot(ref.energy, combined, lw=1.8, color="black", label=op)
        ax.set_xlabel("Energy (eV)"); ax.set_ylabel(ref.units)
        ax.set_title(f"{op.capitalize()} of {len(specs)} spectra")
        ax.legend(fontsize=7); ax.grid(alpha=0.25)
        self.analysis_plot.figure.tight_layout()
        self.analysis_plot.canvas.draw_idle()
        self._set_status(f"{op.capitalize()} of {len(specs)} spectra → {sp_new.name}")

    def difference_selected(self) -> None:
        specs = self._analysis_selected_spectra()
        if len(specs) != 2:
            QMessageBox.warning(self, "Difference", "Select exactly 2 spectra (A then B; result is A − B).")
            return
        a, b = specs
        b_interp = _interp_to_grid(b.energy, b.y, a.energy)
        diff_y = a.y - b_interp

        sp_diff = a.copy(new_name=f"{a.name}_minus_{b.name}", new_kind=f"diff({a.kind})")
        sp_diff.y = diff_y
        sp_diff.history.append(Operation("difference", {"a": a.name, "b": b.name}))
        self.store.add(sp_diff)
        self._refresh_all()

        ax = self.analysis_plot.ax
        ax.clear()
        ax.plot(a.energy, a.y, lw=1.1, label=a.name)
        ax.plot(a.energy, b_interp, lw=1.1, label=b.name)
        ax.plot(a.energy, diff_y, lw=1.4, color="black", label="difference (A − B)")
        ax.set_xlabel("Energy (eV)"); ax.set_ylabel(a.units)
        ax.set_title(f"Difference: {a.name} − {b.name}")
        ax.legend(fontsize=8); ax.grid(alpha=0.25)
        self.analysis_plot.figure.tight_layout()
        self.analysis_plot.canvas.draw_idle()
        self._set_status(f"Difference spectrum → {sp_diff.name}")

    def linear_combination_fit_selected(self) -> None:
        """Athena-style linear combination fitting: fit the LAST selected
        spectrum (target) as a non-negative weighted sum of the other
        selected spectra (references), via NNLS."""
        from scipy.optimize import nnls

        specs = self._analysis_selected_spectra()
        if len(specs) < 3:
            QMessageBox.warning(self, "Linear combination fit", "Select 2+ references and 1 target (3+ total; target = last selected).")
            return
        target = specs[-1]
        refs = specs[:-1]

        A = np.column_stack([_interp_to_grid(r.energy, r.y, target.energy) for r in refs])
        weights, residual_norm = nnls(A, target.y)
        fit_y = A @ weights
        ss_res = float(np.sum((target.y - fit_y) ** 2))
        ss_tot = float(np.sum((target.y - np.mean(target.y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else float("nan")

        lines = [f"Target: {target.name}", f"R² = {r2:.5f}", ""]
        for r, w_ in zip(refs, weights):
            lines.append(f"  {r.name}: weight = {w_:.4f}")
        lines.append(f"\nSum of weights = {sum(weights):.4f} (Athena-style LCF is often constrained to sum to 1; not enforced here, shown for reference)")
        self.analysis_result_text.setPlainText("\n".join(lines))

        sp_fit = target.copy(new_name=f"{target.name}_LCFfit", new_kind="lcf_fit")
        sp_fit.y = fit_y
        sp_fit.history.append(Operation("linear_combination_fit", {"target": target.name, "refs": [r.name for r in refs], "weights": list(map(float, weights)), "r2": r2}))
        self.store.add(sp_fit)
        self._refresh_all()

        ax = self.analysis_plot.ax
        ax.clear()
        ax.plot(target.energy, target.y, lw=1.3, color="black", label=f"target: {target.name}")
        ax.plot(target.energy, fit_y, lw=1.5, ls="--", color="red", label="LCF fit")
        for i, r in enumerate(refs):
            ax.plot(r.energy, r.y, lw=0.8, alpha=0.4, color=COLORS[i % len(COLORS)], label=r.name)
        ax.set_xlabel("Energy (eV)"); ax.set_ylabel(target.units)
        ax.set_title(f"Linear combination fit — R²={r2:.4f}")
        ax.legend(fontsize=7); ax.grid(alpha=0.25)
        self.analysis_plot.figure.tight_layout()
        self.analysis_plot.canvas.draw_idle()

    def pca_selected(self) -> None:
        """Athena-inspired PCA across a spectral series (M21): the explained-
        variance profile indicates how many distinct chemical species/
        environments the series contains (components with non-trivial
        variance ≈ independent spectral signatures)."""
        from core.cluster_science import build_feature_matrix, pca_scores

        specs = self._analysis_selected_spectra()
        if len(specs) < 3:
            QMessageBox.warning(self, "PCA", "Select at least 3 spectra.")
            return
        try:
            matrix, grid = build_feature_matrix([(sp.energy, sp.y) for sp in specs], normalize=None)
            out = pca_scores(matrix, n_components=min(5, len(specs) - 1))
        except (ValueError, ImportError) as exc:
            QMessageBox.critical(self, "PCA error", str(exc))
            return

        var = out["explained_variance_ratio"]
        lines = [f"PCA across {len(specs)} spectra:", ""]
        cumulative = 0.0
        for i, v in enumerate(var):
            cumulative += v
            lines.append(f"  PC{i + 1}: {v * 100:.1f}%  (cumulative {cumulative * 100:.1f}%)")
        n_significant = int(np.sum(var > 0.01))
        lines.append("")
        lines.append(f"Components above 1% variance: {n_significant} — a rough lower bound on the number of distinct species present.")
        self.analysis_result_text.setPlainText("\n".join(lines))

        fig = self.analysis_plot.figure
        fig.clf()
        ax_sc = fig.add_subplot(121)
        scores = out["scores"]  # n_components >= 2 given the 3-spectrum minimum above
        ax_sc.scatter(scores[:, 0], scores[:, 1], s=36, color=COLORS[0])
        for sp, (px, py) in zip(specs, scores[:, :2]):
            ax_sc.annotate(sp.name, (px, py), fontsize=6, alpha=0.7)
        ax_sc.set_xlabel(f"PC1 ({var[0] * 100:.0f}%)")
        ax_sc.set_ylabel(f"PC2 ({var[1] * 100:.0f}%)" if len(var) > 1 else "PC2")
        ax_sc.grid(alpha=0.25)

        ax_scree = fig.add_subplot(122)
        ax_scree.bar(np.arange(1, len(var) + 1), var * 100, color=COLORS[1])
        ax_scree.set_xlabel("Component")
        ax_scree.set_ylabel("Explained variance (%)")
        ax_scree.set_title("Scree", fontsize=9)
        ax_scree.grid(alpha=0.25, axis="y")
        fig.tight_layout()
        self.analysis_plot.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Tools tab (edge definer)
    # ------------------------------------------------------------------
    def _build_tools_tab(self) -> QWidget:
        w = QWidget()
        layout = QHBoxLayout(w)

        ctrl = QWidget()
        ctrl.setMaximumWidth(320)
        ctrl_layout = QVBoxLayout(ctrl)

        ctrl_layout.addWidget(QLabel("Select spectra (multi-select)"))
        self.edge_list = QListWidget()
        self.edge_list.setSelectionMode(QListWidget.ExtendedSelection)
        ctrl_layout.addWidget(self.edge_list, 1)

        elem_row = QHBoxLayout()
        elem_row.addWidget(QLabel("Element"))
        self.edge_elem_combo = QComboBox()
        self.edge_elem_combo.addItems(_periodic_table_symbols())
        self.edge_elem_combo.setCurrentText("Fe")
        elem_row.addWidget(self.edge_elem_combo)
        ctrl_layout.addLayout(elem_row)

        edge_row = QHBoxLayout()
        edge_row.addWidget(QLabel("Edge"))
        self.edge_line_combo = QComboBox()
        self.edge_line_combo.addItems(["K", "L1", "L2", "L3", "M1", "M2", "M3", "M4", "M5"])
        edge_row.addWidget(self.edge_line_combo)
        ctrl_layout.addLayout(edge_row)

        self.edge_set_e0_check = QCheckBox("Also set E0 to tabulated edge energy (xraydb)")
        ctrl_layout.addWidget(self.edge_set_e0_check)

        preview_btn = QPushButton("Preview")
        preview_btn.clicked.connect(self.preview_edge_definer)
        ctrl_layout.addWidget(preview_btn)
        apply_btn = QPushButton("Apply to selected spectra")
        apply_btn.setObjectName("Primary")
        apply_btn.clicked.connect(self.apply_edge_definer)
        ctrl_layout.addWidget(apply_btn)
        ctrl_layout.addStretch(1)
        layout.addWidget(ctrl)

        self.tools_plot = PlotWidget(figsize=(7, 5))
        layout.addWidget(self.tools_plot, 1)
        return w

    def _edge_selected_spectra(self) -> List[Spectrum]:
        out = []
        for item in self.edge_list.selectedItems():
            sp = self.store.find_by_name(item.text())
            if sp is not None:
                out.append(sp)
        return out

    def preview_edge_definer(self) -> None:
        specs = self._edge_selected_spectra()
        if not specs:
            self.tools_plot.clear("Edge definer preview")
            return
        elem = self.edge_elem_combo.currentText()
        edge = self.edge_line_combo.currentText()
        e_edge = None
        try:
            Group, xraydb, find_e0, pre_edge, autobk, xftf = require_larch()
            if xraydb is not None:
                edges = xraydb.xray_edges(elem)
                if edge in edges and getattr(edges[edge], "energy", None) is not None:
                    e_edge = float(edges[edge].energy)
        except Exception:
            pass

        ax = self.tools_plot.ax
        ax.clear()
        for i, sp in enumerate(specs):
            ax.plot(sp.energy, sp.y, lw=1.1, color=COLORS[i % len(COLORS)], label=sp.name)
        ax.set_xlabel("Energy (eV)"); ax.set_ylabel("arb.")
        ax.set_title(f"Edge definer preview — {elem} {edge}")
        ax.legend(fontsize=8); ax.grid(alpha=0.25)
        if e_edge is not None and np.isfinite(e_edge):
            ax.axvline(e_edge, ls="--", lw=1.2)
        self.tools_plot.figure.tight_layout()
        self.tools_plot.canvas.draw_idle()

    def apply_edge_definer(self) -> None:
        specs = self._edge_selected_spectra()
        if not specs:
            QMessageBox.warning(self, "Edge definer", "Select at least one spectrum.")
            return
        elem = self.edge_elem_combo.currentText()
        edge = self.edge_line_combo.currentText()
        e_edge = None
        if self.edge_set_e0_check.isChecked():
            try:
                Group, xraydb, find_e0, pre_edge, autobk, xftf = require_larch()
                if xraydb is None:
                    raise ValueError("xraydb not available in this Larch install; cannot set tabulated E0.")
                edges = xraydb.xray_edges(elem)
                if edge not in edges or getattr(edges[edge], "energy", None) is None:
                    raise ValueError("Unknown element/edge in xraydb.")
                e_edge = float(edges[edge].energy)
            except Exception as exc:
                QMessageBox.critical(self, "Edge definer error", str(exc))
                return

        for sp in specs:
            sp.label = f"XAS({elem} {edge})"
            if e_edge is not None:
                sp.e0 = e_edge
            sp.history.append(Operation("edge_definer", {"element": elem, "edge": edge, "set_e0": e_edge is not None}))

        self._refresh_all()
        self.preview_edge_definer()
        self._set_status(f"Applied manual edge label to {len(specs)} spectrum/spectra.")

    # ------------------------------------------------------------------
    # Export tab
    # ------------------------------------------------------------------
    def _build_mass_tab(self) -> QWidget:
        """Hephaestus-style sample-mass calculator (xas_mass), accepting the
        lab's oxide-composition tables (mol%/wt%) or a single formula."""
        from PySide6.QtWidgets import QPlainTextEdit
        tab = QWidget()
        layout = QHBoxLayout(tab)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.addWidget(QLabel("Composition — one component per line:\n'SiO2 58.8' (fraction optional; a lone\nformula means the pure compound)"))
        self.mass_comp_edit = QPlainTextEdit()
        self.mass_comp_edit.setPlainText("SiO2 58.8\nNa2O 19.6\nBi2O3 19.6\nUO3 2.0")
        ll.addWidget(self.mass_comp_edit, 1)
        row = QHBoxLayout()
        row.addWidget(QLabel("basis"))
        self.mass_basis_combo = QComboBox()
        self.mass_basis_combo.addItems(["mol", "wt"])
        row.addWidget(self.mass_basis_combo)
        row.addWidget(QLabel("element"))
        self.mass_element_edit = QLineEdit("Bi")
        self.mass_element_edit.setMaximumWidth(40)
        row.addWidget(self.mass_element_edit)
        row.addWidget(QLabel("edge"))
        self.mass_edge_combo = QComboBox()
        self.mass_edge_combo.addItems(["K", "L3", "L2", "L1", "M5"])
        self.mass_edge_combo.setCurrentText("L3")
        row.addWidget(self.mass_edge_combo)
        row.addWidget(QLabel("⌀ (mm)"))
        self.mass_diam_edit = QLineEdit("13")
        self.mass_diam_edit.setMaximumWidth(40)
        row.addWidget(self.mass_diam_edit)
        row.addWidget(QLabel("target μt"))
        self.mass_target_mut_edit = QLineEdit("2.5")
        self.mass_target_mut_edit.setMaximumWidth(40)
        self.mass_target_mut_edit.setToolTip(
            "Target absorption length: the sample thickness you want, in units "
            "of 1/μ (μt = thickness / absorption length). 2.5 is Hephaestus' "
            "transmission rule of thumb — lower it for a thinner/more dilute "
            "sample, raise it for a thicker one; the mass for μt = 1.0 and "
            "μt = 2.5 are always shown too, for reference."
        )
        row.addWidget(self.mass_target_mut_edit)
        row.addWidget(QLabel("eV above edge"))
        self.mass_edge_offset_edit = QLineEdit("3")
        self.mass_edge_offset_edit.setMaximumWidth(40)
        self.mass_edge_offset_edit.setToolTip(
            "How far above (and, for the edge step, below) E0 to evaluate "
            "μ/ρ, in eV. Defaults to 3, matching Hephaestus' own convention. "
            "Changing this rarely moves the mass much by itself (μ/ρ shifts "
            "~1% between +3 and +50 eV once clearly past the jump) — a big "
            "mismatch against another program is more likely a disagreement "
            "in tabulated edge energy, not this offset."
        )
        row.addWidget(self.mass_edge_offset_edit)
        ll.addLayout(row)
        calc_btn = QPushButton("Compute sample mass")
        calc_btn.clicked.connect(self._compute_sample_mass)
        ll.addWidget(calc_btn)
        layout.addWidget(left, 1)
        self.mass_report_text = QPlainTextEdit()
        self.mass_report_text.setReadOnly(True)
        layout.addWidget(self.mass_report_text, 1)
        return tab

    def _compute_sample_mass(self) -> None:
        import xas.xas_mass as xas_mass
        try:
            element = self.mass_element_edit.text().strip().capitalize()
            edge = self.mass_edge_combo.currentText()
            diameter = float(self.mass_diam_edit.text() or 13.0)
            target_mut = float(self.mass_target_mut_edit.text() or 2.5)
            edge_offset_ev = float(self.mass_edge_offset_edit.text() or 3.0)
            report = xas_mass.sample_mass_report(
                self.mass_comp_edit.toPlainText(), element, edge,
                basis=self.mass_basis_combo.currentText(), pellet_diameter_mm=diameter,
                target_mut=target_mut, edge_offset_ev=edge_offset_ev,
            )
        except Exception as exc:
            QMessageBox.warning(self, "Sample mass", str(exc))
            return
        self.mass_report_text.setPlainText(report.text(element, edge, diameter))

    def _build_export_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(QLabel("Athena export"))
        row = QHBoxLayout()
        dat_btn = QPushButton("Export selected as Athena column (.dat)")
        dat_btn.clicked.connect(self.export_athena_dat)
        row.addWidget(dat_btn)
        prj_btn = QPushButton("Export ALL mu/norm/flat as Athena project (.prj)")
        prj_btn.clicked.connect(self.export_athena_prj)
        row.addWidget(prj_btn)
        layout.addLayout(row)
        layout.addStretch(1)
        return w

    def export_athena_dat(self) -> None:
        if self.selected_sid is None:
            QMessageBox.information(self, "Export", "Select a spectrum first.")
            return
        sp = self.store.get(self.selected_sid)
        path, _ = QFileDialog.getSaveFileName(self, "Save Athena column file", "", "Athena column file (*.dat)")
        if not path:
            return
        header = ["# Athena column file exported from PRISM", f"# name = {sp.name}", f"# kind = {sp.kind}", f"# label = {sp.label}"]
        if sp.e0 is not None and np.isfinite(sp.e0):
            header.append(f"# e0 = {sp.e0:.6f}")
        try:
            export_athena_column(path, sp.energy, sp.y, header)
            self._set_status(f"Saved: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export .dat error", str(exc))

    def export_athena_prj(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save Athena project (.prj)", "", "Athena project (*.prj)")
        if not path:
            return
        try:
            ok = export_athena_prj_best_effort(path, [s for s in self.store.all() if s.kind in ("mu", "norm", "flat")])
            if not ok:
                QMessageBox.warning(self, "Export .prj", "This Larch install exposes neither write_athena nor create_athena. Export .dat instead.")
            else:
                self._set_status(f"Saved .prj: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export .prj error", str(exc))

    # ------------------------------------------------------------------
    # Import / object management
    # ------------------------------------------------------------------
    def _spectrum_from_record(self, rec: Dict[str, Any]) -> Spectrum:
        angle, energy, signal, cols = _extract_energy_angle_signal(rec["df"])
        kind = _classify_kind_from_name(rec["name"])
        scan_def = rec.get("scan_def", {}) or {}
        label, e0 = infer_edge_label_from_roi_scaled(energy, signal, scan_def)
        return Spectrum(
            sid=_uid("sp"), name=rec["name"], kind=kind, energy=energy, y=signal,
            angle=angle if np.isfinite(angle).any() else None, units="counts/s", label=label, e0=e0,
            meta={"source": rec.get("source", ""), "columns": cols, "scan_def": scan_def, "metadata": rec.get("metadata", {})},
        )

    def import_zips(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Select EasyXAFS ZIP(s)", "", "ZIP files (*.zip);;All files (*.*)")
        if not paths:
            return
        try:
            n = 0
            for zp in paths:
                for rec in read_easyxafs_zip(zp):
                    sp = self._spectrum_from_record(rec)
                    sp.history.append(Operation("import", {"source": rec.get("source", "")}))
                    self.store.add(sp)
                    n += 1
            self._refresh_all()
            self._set_status(f"Imported {n} dataset(s) from ZIP(s).")
        except Exception as exc:
            QMessageBox.critical(self, "Import ZIP error", str(exc))

    def import_csvs(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Select CSV(s)", "", "CSV files (*.csv);;All files (*.*)")
        if not paths:
            return
        try:
            n = 0
            for p in paths:
                rec = read_csv_dataset(p)
                sp = self._spectrum_from_record(rec)
                sp.history.append(Operation("import", {"source": rec.get("source", "")}))
                self.store.add(sp)
                n += 1
            self._refresh_all()
            self._set_status(f"Imported {n} CSV dataset(s).")
        except Exception as exc:
            QMessageBox.critical(self, "Import CSV error", str(exc))

    def import_prj(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Athena project (.prj)", "", "Athena project (*.prj);;All files (*.*)")
        if not path:
            return
        try:
            specs = read_athena_prj(path)
            for sp in specs:
                self.store.add(sp)
            self._refresh_all()
            self._set_status(f"Imported {len(specs)} group(s) from .prj.")
        except Exception as exc:
            QMessageBox.critical(self, "Import .prj error", str(exc))

    def clear_all(self) -> None:
        self.store.clear()
        self.selected_sid = None
        self._refresh_all()
        self._set_status("Cleared all spectra.")

    def load_initial_spectra(self, specs: List[Spectrum]) -> None:
        for sp in specs:
            self.store.add(sp)
        self._refresh_all()

    # ------------------------------------------------------------------
    def _refresh_all(self) -> None:
        self._refresh_table()
        self._refresh_lists()

    def _refresh_table(self) -> None:
        specs = self.store.all()
        self.table.setRowCount(len(specs))
        for row, sp in enumerate(specs):
            e0_txt = "" if sp.e0 is None or not np.isfinite(sp.e0) else f"{sp.e0:.1f}"
            er_txt = f"{np.nanmin(sp.energy):.1f}–{np.nanmax(sp.energy):.1f}" if sp.energy.size else ""
            name_item = QTableWidgetItem(sp.name)
            name_item.setData(Qt.UserRole, sp.sid)
            self.table.setItem(row, 0, name_item)
            self.table.setItem(row, 1, QTableWidgetItem(sp.kind))
            self.table.setItem(row, 2, QTableWidgetItem(edge_text(sp.label)))
            self.table.setItem(row, 3, QTableWidgetItem(e0_txt))
            self.table.setItem(row, 4, QTableWidgetItem(er_txt))
        self.table.resizeColumnsToContents()

    def _refresh_lists(self) -> None:
        all_names = [s.name for s in self.store.all()]
        i0_names = [s.name for s in self.store.all() if s.kind in ("I0", "fit", "I0_fit")]
        it_names = [s.name for s in self.store.all() if s.kind == "It"]
        mu_names = [s.name for s in self.store.all() if s.kind == "mu"]

        self.mu_i0_combo.blockSignals(True)
        current_i0 = self.mu_i0_combo.currentText()
        self.mu_i0_combo.clear()
        self.mu_i0_combo.addItems(i0_names)
        if current_i0 in i0_names:
            self.mu_i0_combo.setCurrentText(current_i0)
        self.mu_i0_combo.blockSignals(False)

        for combo in (self.sm_target_combo, self.ang_before_combo, self.ang_after_combo):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(all_names)
            if current in all_names:
                combo.setCurrentText(current)
            combo.blockSignals(False)

        self._fill_list(self.mu_it_list, it_names)
        self._fill_list(self.norm_mu_list, mu_names)
        self._fill_list(self.edge_list, all_names)
        self._fill_list(self.analysis_list, all_names)

    @staticmethod
    def _fill_list(listwidget: QListWidget, names: List[str]) -> None:
        selected = {listwidget.item(i).text() for i in range(listwidget.count()) if listwidget.item(i).isSelected()}
        listwidget.clear()
        for n in names:
            item = QListWidgetItem(n)
            listwidget.addItem(item)
            if n in selected:
                item.setSelected(True)

    def _on_selection_changed(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            self.selected_sid = None
            return
        item = self.table.item(rows[0].row(), 0)
        self.selected_sid = item.data(Qt.UserRole)
        self._plot_selected_preview()

    def _on_table_context_menu(self, pos) -> None:
        item = self.table.itemAt(pos)
        if item is None:
            return
        row = item.row()
        sid = self.table.item(row, 0).data(Qt.UserRole)
        self.table.selectRow(row)
        self.selected_sid = sid

        menu = QMenu(self)
        menu.addAction("Rename…", self.rename_selected)
        menu.addAction("Duplicate", self.duplicate_selected)
        menu.addSeparator()
        menu.addAction("Delete", self.delete_selected)
        menu.addSeparator()
        menu.addAction("Export selected as .dat", self.export_athena_dat)
        menu.exec(self.table.viewport().mapToGlobal(pos))

    def rename_selected(self) -> None:
        if self.selected_sid is None:
            return
        sp = self.store.get(self.selected_sid)
        from PySide6.QtWidgets import QInputDialog
        new_name, ok = QInputDialog.getText(self, "Rename", "New name:", text=sp.name)
        if ok and new_name.strip():
            sp.name = new_name.strip()
            self._refresh_all()

    def duplicate_selected(self) -> None:
        if self.selected_sid is None:
            return
        sp = self.store.get(self.selected_sid)
        sp2 = sp.copy(new_name=f"{sp.name}_copy")
        sp2.history.append(Operation("duplicate", {"from": sp.sid}))
        self.store.add(sp2)
        self._refresh_all()

    def delete_selected(self) -> None:
        if self.selected_sid is None:
            return
        sp = self.store.get(self.selected_sid)
        resp = QMessageBox.question(self, "Delete", f"Delete '{sp.name}'?")
        if resp == QMessageBox.Yes:
            self.store.remove(self.selected_sid)
            self.selected_sid = None
            self._refresh_all()
