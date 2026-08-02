"""
qt_single_fit.py — Qt port of main.py's SingleFitWindow (M8): single-spectrum
peak fitting (Classic one-shot LM, or Origin-like stepwise relaxation).

Per-spectrum fit parameters are now keyed by Spectrum.id via
PerItemSettingsStore instead of by display title (qt_settings_store.py) —
main.py's SingleFitWindow keyed fit_param_memory by
`self.spec_var.get()` (the title string), which is exactly the
title-instead-of-identity pattern the M5 architecture principles call out
as a live bug class (two imports sharing a title silently collide). Fixed
here rather than carried forward.

New in this port, beyond a faithful port of the Tk original (see the plan's
M8 milestone, "layer 1-5" additive items):
  - Dashed fit line + a residual subplot underneath, drawn whenever a fit/
    model exists (item 11).
  - Per-component peak CSV export, plus a residual column (item 12).
  - A second, one-click "Quick report" button (auto-named, no dialog) next
    to the existing dialog-based "Report as..." button (item 3).
  - Peak centroid (trapz-weighted, fitting_science.peak_centroid) and
    overall R^2 added to the exported report (item 10 + a cheap report
    enhancement noted alongside the competitive-research follow-ups).
  - "Auto-find peaks" in the fit-parameter dialog (qt_fit_params.py),
    backed by fitting_science.find_peak_candidates (2nd-derivative finder).

All four items originally deferred here — true Voigt/EMG peak shapes,
cross-component parameter linking, and lmfit conf_interval()-based
confidence intervals — are now implemented (fitting_science.voigt_peak/
emg_peak, the link_fwhm/link_eta constraint mechanism, and
show_confidence_intervals below).
"""
from __future__ import annotations

import copy
import datetime
import os
from typing import Any, Dict, List, Optional

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QFileDialog, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QRadioButton,
    QSplitter, QVBoxLayout, QWidget,
)

from core.fitting_science import (
    build_lmfit_parameters, compute_chi2, compute_model, compute_r_squared,
    fit_spectrum, origin_lm_iteration, peak_centroid,
)
from processing.qt_fit_params import FitParamDialog
from core.qt_models import SpectrumLibrary
from core.qt_settings_store import PerItemSettingsStore
from core.qt_widgets import PlotWidget

COLORS = ["black", "red", "seagreen", "royalblue", "orange", "purple", "brown", "indigo"]


def _to_float(text: str, default: Optional[float] = None) -> Optional[float]:
    try:
        return float((text or "").strip())
    except (TypeError, ValueError):
        return default


class SingleFitWorkspace(QWidget):
    def __init__(
        self, parent: Optional[QWidget] = None, library: Optional[SpectrumLibrary] = None,
        fit_param_memory: Optional[PerItemSettingsStore] = None,
    ):
        super().__init__(parent)
        self.library = library if library is not None else SpectrumLibrary()
        # Shared with MultiFitWorkspace (M9) when both are created by qt_shell
        # so a batch fit's write-back is immediately visible here too, and
        # vice versa — one PerItemSettingsStore per concept, per the M5
        # architecture principles, not one already-fresh copy per window.
        self.fit_param_memory: PerItemSettingsStore[List[Dict[str, Any]]] = (
            fit_param_memory if fit_param_memory is not None else PerItemSettingsStore(list)
        )
        self._last_snapshot_by_id: Dict[str, list] = {}

        self._current_spectrum_id: Optional[str] = None
        self._current_fit = None
        self._current_fit_result = None  # FitResult from the last classic fit
        self._current_yfit: Optional[np.ndarray] = None
        self._current_peaks: Optional[List[np.ndarray]] = None
        self._current_x: Optional[np.ndarray] = None
        self._current_y: Optional[np.ndarray] = None

        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        splitter = QSplitter()
        root.addWidget(splitter)

        left = QWidget()
        left.setObjectName("Card")
        left.setMaximumWidth(300)
        left_layout = QVBoxLayout(left)

        left_layout.addWidget(QLabel("Select from imported"))
        self.spec_combo_labels: List[str] = []
        from PySide6.QtWidgets import QComboBox
        self.spec_combo = QComboBox()
        self.spec_combo.currentIndexChanged.connect(self._on_spec_changed)
        left_layout.addWidget(self.spec_combo)

        self.norm_check = QCheckBox("Norm.")
        self.norm_check.setChecked(True)
        self.norm_check.toggled.connect(self._request_render)
        left_layout.addWidget(self.norm_check)

        plot_x_row = QHBoxLayout()
        plot_x_row.addWidget(QLabel("Plot X"))
        self.xmin_edit = QLineEdit()
        self.xmax_edit = QLineEdit()
        for e in (self.xmin_edit, self.xmax_edit):
            e.setMaximumWidth(60)
            e.textChanged.connect(self._request_render)
        plot_x_row.addWidget(self.xmin_edit)
        plot_x_row.addWidget(self.xmax_edit)
        left_layout.addLayout(plot_x_row)

        param_btn = QPushButton("Fit param.")
        param_btn.clicked.connect(self.open_param_window)
        left_layout.addWidget(param_btn)

        self.pick_peaks_btn = QPushButton("Pick peaks on plot")
        self.pick_peaks_btn.setCheckable(True)
        self.pick_peaks_btn.setToolTip(
            "Toggle on, then click peak apexes on the plot — each click adds a fit component "
            "centered at the clicked position with its amplitude read from the data. Toggle off when done."
        )
        self.pick_peaks_btn.toggled.connect(self._on_pick_peaks_toggled)
        left_layout.addWidget(self.pick_peaks_btn)

        fit_btn = QPushButton("Fit !")
        fit_btn.setObjectName("Primary")
        fit_btn.clicked.connect(self.run_fit)
        left_layout.addWidget(fit_btn)

        reset_btn = QPushButton("Reset params to snapshot")
        reset_btn.clicked.connect(self.reset_params_to_snapshot)
        left_layout.addWidget(reset_btn)

        mode_row = QHBoxLayout()
        self.mode_group = QButtonGroup(self)
        self.mode_classic = QRadioButton("Classic")
        self.mode_origin = QRadioButton("Origin-like")
        self.mode_classic.setChecked(True)
        self.mode_group.addButton(self.mode_classic, 0)
        self.mode_group.addButton(self.mode_origin, 1)
        self.mode_origin.toggled.connect(self._toggle_origin_controls)
        mode_row.addWidget(self.mode_classic)
        mode_row.addWidget(self.mode_origin)
        left_layout.addLayout(mode_row)

        self.origin_panel = QWidget()
        origin_layout = QVBoxLayout(self.origin_panel)
        origin_layout.setContentsMargins(0, 4, 0, 4)

        tol_row = QHBoxLayout()
        tol_row.addWidget(QLabel("Δχ² tol:"))
        self.origin_tol_edit = QLineEdit("1e-9")
        self.origin_tol_edit.setToolTip("Relative χ² change below which the fit counts as converged (Origin's default is 1e-9).")
        tol_row.addWidget(self.origin_tol_edit)
        tol_row.addWidget(QLabel("Max iter:"))
        self.origin_max_iter_edit = QLineEdit("200")
        self.origin_max_iter_edit.setMaximumWidth(50)
        tol_row.addWidget(self.origin_max_iter_edit)
        origin_layout.addLayout(tol_row)

        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Iterate:"))
        for n in (1, 2, 5, 10):
            btn = QPushButton(f"{n}×" if n > 1 else "1 iteration")
            btn.setToolTip("One Levenberg-Marquardt parameter update per iteration — watch the curve move, like Origin's NLFit.")
            btn.clicked.connect(lambda _checked=False, n=n: self.run_fit_origin_stepwise(n))
            step_row.addWidget(btn)
        origin_layout.addLayout(step_row)

        full_btn = QPushButton("Fit until converged")
        full_btn.clicked.connect(self.run_fit_origin_full)
        origin_layout.addWidget(full_btn)

        self.origin_status_label = QLabel("λ = 1e-3 (damping resets when parameters change)")
        self.origin_status_label.setObjectName("SectionNote")
        origin_layout.addWidget(self.origin_status_label)

        left_layout.addWidget(self.origin_panel)
        self.origin_panel.setVisible(False)
        left_layout.addStretch(1)
        splitter.addWidget(left)

        right = QWidget()
        right.setObjectName("Card")
        right_layout = QVBoxLayout(right)

        axes_row = QHBoxLayout()
        axes_row.addWidget(QLabel("X title"))
        self.x_title_edit = QLineEdit("Raman Shift (cm⁻¹)")
        self.x_title_edit.textChanged.connect(self._request_render)
        axes_row.addWidget(self.x_title_edit)
        axes_row.addWidget(QLabel("Y title"))
        self.y_title_edit = QLineEdit("Intensity (a.u.)")
        self.y_title_edit.textChanged.connect(self._request_render)
        axes_row.addWidget(self.y_title_edit)
        right_layout.addLayout(axes_row)

        self.plot = PlotWidget(figsize=(7, 5))
        right_layout.addWidget(self.plot, 1)

        chi2_row = QHBoxLayout()
        chi2_row.addWidget(QLabel("Chi²:"))
        self.chi2_label = QLabel("--")
        chi2_row.addWidget(self.chi2_label)
        chi2_row.addWidget(QLabel("R²:"))
        self.r2_label = QLabel("--")
        chi2_row.addWidget(self.r2_label)

        ci_btn = QPushButton("Conf. intervals")
        ci_btn.clicked.connect(self.show_confidence_intervals)
        chi2_row.addWidget(ci_btn)
        quick_report_btn = QPushButton("Quick report")
        quick_report_btn.clicked.connect(lambda: self.generate_report(quick=True))
        chi2_row.addWidget(quick_report_btn)
        report_btn = QPushButton("Report as…")
        report_btn.clicked.connect(lambda: self.generate_report(quick=False))
        chi2_row.addWidget(report_btn)
        components_btn = QPushButton("Export components…")
        components_btn.clicked.connect(self.export_components_csv)
        chi2_row.addWidget(components_btn)
        chi2_row.addStretch(1)
        right_layout.addLayout(chi2_row)

        export_row = QHBoxLayout()
        self.export_png_check = QCheckBox("PNG")
        self.export_png_check.setChecked(True)
        self.export_svg_check = QCheckBox("SVG")
        self.export_pdf_check = QCheckBox("PDF")
        for cb in (self.export_png_check, self.export_svg_check, self.export_pdf_check):
            export_row.addWidget(cb)
        export_btn = QPushButton("Export plot")
        export_btn.clicked.connect(self.export_plot)
        export_row.addWidget(export_btn)
        export_row.addStretch(1)
        right_layout.addLayout(export_row)

        right_layout.addWidget(QLabel("Log"))
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(120)
        right_layout.addWidget(self.log)

        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

    # ------------------------------------------------------------------
    def set_spectra(self, spectrum_ids: List[str]) -> None:
        previous_id = self._current_spectrum_id
        self.spec_combo.blockSignals(True)
        self.spec_combo.clear()
        for sid in spectrum_ids:
            spectrum = self.library.get(sid)
            if spectrum is not None:
                self.spec_combo.addItem(spectrum.title, sid)
        self.spec_combo.blockSignals(False)

        if previous_id in spectrum_ids and self.spec_combo.count():
            idx = spectrum_ids.index(previous_id)
            self.spec_combo.setCurrentIndex(idx)
        elif self.spec_combo.count():
            self.spec_combo.setCurrentIndex(0)
        self._on_spec_changed(self.spec_combo.currentIndex())

    def _on_spec_changed(self, index: int) -> None:
        self._current_spectrum_id = self.spec_combo.itemData(index) if index >= 0 else None
        self._current_fit = None
        self._current_yfit = None
        self._current_peaks = None
        self._origin_reset_damping()
        self.render()

    def _current_spectrum(self):
        if self._current_spectrum_id is None:
            return None
        return self.library.get(self._current_spectrum_id)

    def _toggle_origin_controls(self) -> None:
        self.origin_panel.setVisible(self.mode_origin.isChecked())

    def _append_log(self, msg: str) -> None:
        self.log.appendPlainText(msg)

    def _request_render(self) -> None:
        self.plot.request_redraw(self.render)

    # ------------------------------------------------------------------
    def get_xy(self):
        spectrum = self._current_spectrum()
        if spectrum is None:
            return np.array([]), np.array([])
        x = np.asarray(spectrum.x, dtype=float)
        y = np.asarray(spectrum.y, dtype=float)

        xmin, xmax = _to_float(self.xmin_edit.text()), _to_float(self.xmax_edit.text())
        if xmin is not None and xmax is not None:
            mask = (x >= xmin) & (x <= xmax)
            x, y = x[mask], y[mask]

        if self.norm_check.isChecked() and len(x) and len(y):
            area = float(np.trapz(y, x))
            if abs(area) > 1e-10:
                y = y / area * 100.0
        return x, y

    def get_current_params(self) -> List[Dict[str, Any]]:
        if self._current_spectrum_id is None:
            return []
        return self.fit_param_memory.get(self._current_spectrum_id)

    def _snapshot_current_params(self) -> None:
        if self._current_spectrum_id is None:
            return
        cur = self.get_current_params()
        if cur:
            self._last_snapshot_by_id[self._current_spectrum_id] = copy.deepcopy(cur)
            self._append_log(f"[Snapshot] Saved params for '{self.spec_combo.currentText()}' before fit.")

    def reset_params_to_snapshot(self) -> None:
        if self._current_spectrum_id is None:
            return
        snap = self._last_snapshot_by_id.get(self._current_spectrum_id)
        if snap is None:
            QMessageBox.information(self, "Reset", "No pre-fit snapshot found for this spectrum yet.")
            return
        self.fit_param_memory.set(self._current_spectrum_id, copy.deepcopy(snap))
        self._append_log(f"[Reset] Restored last pre-fit snapshot for '{self.spec_combo.currentText()}'.")
        self.render()

    # ------------------------------------------------------------------
    def open_param_window(self) -> None:
        spectrum = self._current_spectrum()
        if spectrum is None:
            return
        x, y = self.get_xy()
        existing = self.fit_param_memory.get(spectrum.id) if self.fit_param_memory.has(spectrum.id) else None
        dlg = FitParamDialog(
            self, params_struct=existing,
            on_accept=lambda params, sid=spectrum.id: self._on_params_saved(sid, params),
            x=x, y=y,
        )
        dlg.exec()

    def _on_params_saved(self, spectrum_id: str, params: List[Dict[str, Any]]) -> None:
        self.fit_param_memory.set(spectrum_id, params)
        self._origin_reset_damping()  # new starting point → restart LM damping
        self.render()

    # ------------------------------------------------------------------
    # Manual peak picking (user request): click apexes on the plot to add
    # components — the intuitive complement to typing centers in the table
    # or trusting the auto-finder. Same click plumbing as Simple Plot's
    # annotations and XAS's Mode C tie-points.
    # ------------------------------------------------------------------
    def _on_pick_peaks_toggled(self, checked: bool) -> None:
        if checked:
            if self._current_spectrum() is None:
                self.pick_peaks_btn.setChecked(False)
                QMessageBox.information(self, "Pick peaks", "Select a spectrum first.")
                return
            # A still-active zoom/pan tool would swallow every click (the
            # user's report: "clicking does nothing but drag-zoom works").
            # Entering pick mode deactivates it instead of silently losing.
            mode = str(self.plot.toolbar.mode)
            if "zoom" in mode:
                self.plot.toolbar.zoom()
            elif "pan" in mode:
                self.plot.toolbar.pan()
            self.plot.canvas.setCursor(Qt.CrossCursor)
            self._pick_cid = self.plot.canvas.mpl_connect("button_press_event", self._on_pick_click)
            self._append_log("[Pick peaks] ON — click peak apexes; each click adds a component.")
        else:
            if getattr(self, "_pick_cid", None) is not None:
                self.plot.canvas.mpl_disconnect(self._pick_cid)
                self._pick_cid = None
            self.plot.canvas.unsetCursor()
            n = len(self.get_current_params())
            self._append_log(f"[Pick peaks] OFF — model now has {n} component(s); open Fit param. to refine.")

    def _on_pick_click(self, event) -> None:
        if event.inaxes is None or event.xdata is None:
            return
        if self.plot.toolbar.mode:  # zoom/pan re-activated mid-pick — say so instead of silently eating clicks
            self._append_log("[Pick peaks] click ignored — the zoom/pan tool is active; deactivate it in the toolbar to pick.")
            return
        x, y = self.get_xy()
        if not len(x):
            return
        span = float(np.nanmax(x) - np.nanmin(x))
        half_width = max(span * 0.02, 1.0)
        center = float(event.xdata)
        # Amplitude from the data around the click, not the raw click y
        # (the user aims at the apex but rarely lands exactly on it).
        near = np.abs(np.asarray(x, float) - center) <= half_width
        amp = float(np.nanmax(np.asarray(y, float)[near])) if np.any(near) else float(event.ydata)

        from processing.qt_fit_params import _DEFAULTS
        row = dict(_DEFAULTS)
        row["shift_val"] = center
        row["shift_min"] = center - 10 * half_width
        row["shift_max"] = center + 10 * half_width
        row["amp_val"] = amp

        sid = self._current_spectrum_id
        params = list(self.fit_param_memory.get(sid)) if self.fit_param_memory.has(sid) else []
        params.append(row)
        self.fit_param_memory.set(sid, params)
        self._append_log(f"[Pick peaks] + component {len(params)} at {center:.2f} (amp≈{amp:.3g})")
        self._request_render()

    # ------------------------------------------------------------------
    # Rendering — dashed fit line + residual subplot (M8 item 11)
    # ------------------------------------------------------------------
    def render(self) -> None:
        x, y = self.get_xy()
        fig = self.plot.figure
        fig.clf()
        if len(x) == 0:
            self.chi2_label.setText("--")
            self.r2_label.setText("--")
            self.plot.canvas.draw_idle()
            return

        params_struct = self.get_current_params()
        show_components = bool(params_struct)

        y_fit, peaks = None, None
        if show_components:
            lm_params = build_lmfit_parameters(params_struct)
            y_fit, peaks = compute_model(x, lm_params, params_struct)
            self._current_yfit, self._current_peaks = y_fit, peaks
            chi2 = compute_chi2(y, y_fit, lm_params)
            r2 = compute_r_squared(y, y_fit)
            self.chi2_label.setText(f"{chi2:.3g}")
            self.r2_label.setText(f"{r2:.4f}")
        else:
            self._current_yfit, self._current_peaks = None, None
            self.chi2_label.setText("--")
            self.r2_label.setText("--")

        if y_fit is not None:
            ax_main, ax_resid = fig.subplots(2, 1, sharex=True, height_ratios=[3, 1])
        else:
            ax_main = fig.add_subplot(111)
            ax_resid = None

        ax_main.plot(x, y, color="black", lw=1.2, label="Data")
        if y_fit is not None:
            ax_main.plot(x, y_fit, color="red", lw=2, ls="--", label="Fit")
            for i, pk in enumerate(peaks):
                # named components (A FAIRE item 9) join the legend
                name = params_struct[i].get("name") if i < len(params_struct) else None
                ax_main.plot(x, pk, lw=1.1, color=COLORS[(i + 2) % len(COLORS)], alpha=0.7,
                             label=name or None)
        ax_main.set_ylabel(self.y_title_edit.text())
        ax_main.set_title("Fit preview")
        ax_main.legend(fontsize=8)
        ax_main.grid(alpha=0.25)

        if ax_resid is not None:
            residual = y - y_fit
            ax_resid.axhline(0.0, color="0.5", lw=0.8)
            ax_resid.plot(x, residual, color="royalblue", lw=1.0)
            ax_resid.set_ylabel("Resid.")
            ax_resid.set_xlabel(self.x_title_edit.text())
            ax_resid.grid(alpha=0.25)
        else:
            ax_main.set_xlabel(self.x_title_edit.text())

        fig.tight_layout()
        if ax_resid is not None:
            fig.subplots_adjust(hspace=0.06)  # after tight_layout, which owns margins but not inter-axes spacing
        self.plot.canvas.draw_idle()
        self._current_x, self._current_y = x, y

    # ------------------------------------------------------------------
    # Classic fit
    # ------------------------------------------------------------------
    def run_fit(self) -> None:
        self._snapshot_current_params()
        x, y = self.get_xy()
        params_struct = self.get_current_params()
        if not params_struct:
            QMessageBox.warning(self, "No parameters", "No fit parameters found for this spectrum.")
            return

        try:
            fr = fit_spectrum(x, y, params_struct, mode="classic")
            self._current_fit = fr.lmfit_result
            self._current_fit_result = fr  # full FitResult incl. minimizer, for conf_interval
            self._current_yfit = fr.y_fit
            self._current_peaks = fr.peaks
            self._current_x, self._current_y = x, y

            self._writeback_params_from_result(fr.lmfit_result, params_struct)
            self.render()
            self._append_log(f"[Classic] chi2_red={fr.chi2_red:.6g} spec='{self.spec_combo.currentText()}'")
        except Exception as exc:
            self.chi2_label.setText("FAIL")
            QMessageBox.critical(self, "Fit error", str(exc))

    # ------------------------------------------------------------------
    # Origin-like (stepwise) mode
    # ------------------------------------------------------------------
    def _origin_common(self):
        x, y = self.get_xy()
        params_struct = self.get_current_params()
        if not params_struct:
            raise RuntimeError("No fit parameters defined for this spectrum.")
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        lm_params = build_lmfit_parameters(params_struct)
        return x, y, params_struct, lm_params

    def _origin_reset_damping(self) -> None:
        self._origin_lambda = 1e-3
        self._origin_iter_count = 0

    def _run_origin_iterations(self, max_iters: int, *, stop_on_converge: bool) -> None:
        """Shared engine for the Iterate-N buttons and Fit-until-converged:
        one true LM parameter update per iteration (origin_lm_iteration),
        redrawn after every step so the curve visibly walks toward the data —
        the Origin NLFit behavior the first implementation only imitated."""
        self._snapshot_current_params()
        tol = _to_float(self.origin_tol_edit.text(), 1e-9)

        try:
            x, y, params_struct, lm_params = self._origin_common()
        except RuntimeError as exc:
            QMessageBox.warning(self, "No parameters", str(exc))
            return

        if not hasattr(self, "_origin_lambda"):
            self._origin_reset_damping()

        current = lm_params
        for _ in range(max_iters):
            try:
                step = origin_lm_iteration(x, y, params_struct, current, lambda_lm=self._origin_lambda)
            except ValueError as exc:
                QMessageBox.warning(self, "Origin-like fit", str(exc))
                return
            current = step.params
            self._origin_lambda = step.next_lambda
            self._origin_iter_count += 1

            rel_drop = (step.chisq_before - step.chisq_after) / max(step.chisq_before, 1e-30)
            self._append_log(
                f"[Origin iter {self._origin_iter_count}] chisq {step.chisq_before:.6g} → {step.chisq_after:.6g} "
                f"(Δrel={rel_drop:.3g}, λ={step.lambda_used:.1e}{'' if step.accepted else ', no improving step'})"
            )
            self.origin_status_label.setText(
                f"iter {self._origin_iter_count}: χ²={step.chisq_after:.6g}, λ={self._origin_lambda:.1e}"
            )

            self._current_yfit, self._current_peaks = step.y_fit, step.peaks
            self._current_x, self._current_y = x, y
            self.chi2_label.setText(f"{step.chi2_red:.6g}")
            self.render()
            QApplication.processEvents()  # let each step actually paint

            self._writeback_params_from_struct_values(current, params_struct)

            if not step.accepted:
                self._append_log("[Origin] no damping level improves χ² — at a (local) minimum.")
                break
            if stop_on_converge and rel_drop < tol:
                self._append_log(f"[Origin] converged: Δrel={rel_drop:.3g} < tol={tol:g} after {self._origin_iter_count} iterations.")
                break

    def run_fit_origin_stepwise(self, step_iters: int = 1) -> None:
        if not self.mode_origin.isChecked():
            QMessageBox.information(self, "Info", "Select 'Origin-like' mode to use stepwise buttons.")
            return
        self._run_origin_iterations(step_iters, stop_on_converge=False)

    def run_fit_origin_full(self) -> None:
        if not self.mode_origin.isChecked():
            QMessageBox.information(self, "Info", "Select 'Origin-like' mode first.")
            return
        max_iters = int(_to_float(self.origin_max_iter_edit.text(), 200) or 200)
        self._run_origin_iterations(max_iters, stop_on_converge=True)

    # ------------------------------------------------------------------
    def _writeback_params_from_result(self, result, params_struct: List[Dict[str, Any]]) -> None:
        self._writeback_params_from_struct_values(result.params, params_struct)

    def _writeback_params_from_struct_values(self, params, params_struct: List[Dict[str, Any]]) -> None:
        if self._current_spectrum_id is None:
            return
        for i, d in enumerate(params_struct):
            d["shift_val"] = float(params[f"f{i}"].value)
            d["fwhm_val"] = float(params[f"l{i}"].value)
            if f"a{i}" in params:
                d["amp_val"] = float(params[f"a{i}"].value)
            if d.get("shape", "G") in ("GL", "V") and f"eta{i}" in params:
                d["eta_val"] = float(params[f"eta{i}"].value)
            if d.get("shape", "G") == "EMG" and f"s{i}" in params:
                d["skew_val"] = float(params[f"s{i}"].value)
        self.fit_param_memory.set(self._current_spectrum_id, params_struct)

    # ------------------------------------------------------------------
    # Export / report
    # ------------------------------------------------------------------
    def export_plot(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export plot as…", "", "PNG (*.png)")
        if not path:
            return
        base = os.path.splitext(path)[0]
        errors = []
        for ext, enabled in (("png", self.export_png_check.isChecked()),
                              ("svg", self.export_svg_check.isChecked()),
                              ("pdf", self.export_pdf_check.isChecked())):
            if enabled:
                try:
                    self.plot.figure.savefig(f"{base}.{ext}")
                except Exception as exc:
                    errors.append(f"{ext.upper()}: {exc}")
        if errors:
            QMessageBox.critical(self, "Export", "Some exports failed:\n" + "\n".join(errors))
        else:
            QMessageBox.information(self, "Export", "Plot exported successfully.")

    def export_components_csv(self) -> None:
        """Per-component peak CSV export + residual column (A FAIRE item 12)."""
        if self._current_x is None or self._current_peaks is None:
            QMessageBox.information(self, "Export components", "No fit available. Run a fit first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export components as…", "", "CSV (*.csv)")
        if not path:
            return
        base = os.path.splitext(path)[0]
        x, y = self._current_x, self._current_y
        y_fit = self._current_yfit
        residual = y - y_fit if y_fit is not None else np.full_like(x, np.nan)

        params_struct = self.get_current_params() or []

        def _comp_label(i: int) -> str:
            name = params_struct[i].get("name", "") if i < len(params_struct) else ""
            return name or f"comp{i + 1}"

        def _comp_slug(i: int) -> str:
            import re
            name = params_struct[i].get("name", "") if i < len(params_struct) else ""
            slug = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
            return f"comp{i + 1}" + (f"_{slug}" if slug else "")

        try:
            header = ["x", "data", "fit", "residual"] + [_comp_label(i) for i in range(len(self._current_peaks))]
            cols = [x, y, y_fit if y_fit is not None else np.full_like(x, np.nan), residual, *self._current_peaks]
            data = np.column_stack(cols)
            np.savetxt(f"{base}_all.csv", data, delimiter=",", header=",".join(header), comments="")

            for i, pk in enumerate(self._current_peaks):
                comp_data = np.column_stack([x, pk])
                np.savetxt(f"{base}_{_comp_slug(i)}.csv", comp_data, delimiter=",", header="x,y", comments="")

            np.savetxt(f"{base}_residual.csv", np.column_stack([x, residual]), delimiter=",", header="x,residual", comments="")
            QMessageBox.information(self, "Export components", f"Exported {len(self._current_peaks)} component(s) + residual.")
        except OSError as exc:
            QMessageBox.critical(self, "Export error", str(exc))

    def show_confidence_intervals(self) -> None:
        """F-test confidence-interval profiling (lmfit conf_interval) for
        the last classic fit — the rigorous complement to the covariance
        ±1σ values in reports."""
        from core.fitting_science import compute_confidence_intervals
        if self._current_fit_result is None:
            QMessageBox.information(self, "Confidence intervals", "Run a classic fit ('Fit !') first.")
            return
        try:
            report = compute_confidence_intervals(self._current_fit_result)
        except ValueError as exc:
            QMessageBox.warning(self, "Confidence intervals", str(exc))
            return
        from PySide6.QtWidgets import QDialog, QPlainTextEdit as _QPT, QVBoxLayout as _QVB
        dlg = QDialog(self)
        dlg.setWindowTitle("Confidence intervals (F-test profiling)")
        dlg.resize(560, 320)
        lay = _QVB(dlg)
        text = _QPT()
        text.setReadOnly(True)
        text.setPlainText(report)
        lay.addWidget(text)
        dlg.exec()
        self._append_log("[Conf. intervals]\n" + report)

    def generate_report(self, quick: bool = False) -> None:
        x, y = self._current_x, self._current_y
        params_struct = self.get_current_params()
        y_fit, peaks = self._current_yfit, self._current_peaks

        if x is None or y is None or not params_struct or peaks is None:
            QMessageBox.information(self, "Report", "No fit available. Run fit or set parameters first.")
            return

        spectrum = self._current_spectrum()
        specname = spectrum.title if spectrum else "spectrum"
        now = datetime.datetime.now()
        datestr = now.strftime("%Y%m%d-%H%M%S")
        default_name = f"{specname}_fit{datestr}.txt"

        if quick:
            base_dir = os.path.join(os.path.dirname(spectrum.path), "reports") if spectrum and spectrum.path else "reports"
            os.makedirs(base_dir, exist_ok=True)
            filename = os.path.join(base_dir, default_name)
        else:
            filename, _ = QFileDialog.getSaveFileName(self, "Generate report", default_name, "Text (*.txt)")
            if not filename:
                return

        chi2 = compute_chi2(y, y_fit, build_lmfit_parameters(params_struct))
        r2 = compute_r_squared(y, y_fit)
        areas = [float(np.trapz(pk, x)) for pk in peaks]
        centroids = [peak_centroid(x, pk) for pk in peaks]

        def _stderr(pname: str) -> str:
            """1-sigma uncertainty from the last fit's covariance matrix
            (lmfit's leastsq computes these for free — the pragmatic
            uncertainty report; full conf_interval() profiling deferred)."""
            if self._current_fit is None:
                return ""
            par = self._current_fit.params.get(pname)
            if par is None or par.stderr is None or not np.isfinite(par.stderr):
                return ""
            return f" ± {par.stderr:.3g}"

        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write("# Raman Fit Report\n")
                f.write(f"# Spectrum: {specname}\n")
                f.write(f"# Date: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Chi2: {chi2:.5g}\n")
                f.write(f"# R2: {r2:.5g}\n")
                f.write("# Uncertainties (when shown) are 1-sigma standard errors from the fit covariance matrix.\n")
                f.write("#\n")
                f.write("# Component\tCenter\tFWHM\tAmplitude\tArea\tCentroid\tShape\tEta\tSkew\n")
                for i, d in enumerate(params_struct):
                    center = d["shift_val"]
                    fwhm = d["fwhm_val"]
                    amp = d.get("amp_val", 1.0)
                    area = areas[i] if i < len(areas) else float("nan")
                    centroid = centroids[i] if i < len(centroids) else float("nan")
                    shape = d.get("shape", "G")
                    eta = d.get("eta_val", "--") if shape in ("GL", "V") else "--"
                    skew = d.get("skew_val", "--") if shape == "EMG" else "--"
                    comp_label = f"{i + 1} ({d['name']})" if d.get("name") else f"{i + 1}"
                    f.write(
                        f"{comp_label}\t{center:.2f}{_stderr(f'f{i}')}\t{fwhm:.2f}{_stderr(f'l{i}')}\t"
                        f"{amp}{_stderr(f'a{i}')}\t{area:.2f}\t{centroid:.2f}\t{shape}\t{eta}\t{skew}\n"
                    )
                f.write("#\n# End of report\n")
        except OSError as exc:
            QMessageBox.critical(self, "Report error", str(exc))
            return

        if not quick:
            QMessageBox.information(self, "Report", f"Report exported to:\n{filename}")
        else:
            self._append_log(f"[Quick report] {filename}")
