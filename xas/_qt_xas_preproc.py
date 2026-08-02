"""
xas/_qt_xas_preproc.py — internal implementation detail of qt_xas.py:
XasWorkspace's Pre-processing tab (smoothing preview/apply, Bragg
angle/energy correction, Mode C interactive click-based feature
alignment). Mixed into XasWorkspace, not meant to be used standalone.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from core.qt_widgets import PlotWidget, to_float as _to_float
from xas.xas_science import (
    _SCIPY_AVAILABLE,
    Operation,
    TiePoint,
    angle_energy_correction_bragg,
    apply_alignment_mode_c,
    smooth_spectrum,
)
from ._qt_xas_shared import _to_int


class PreprocTabMixin:
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
