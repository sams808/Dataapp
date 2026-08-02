"""
qt_multi_fit.py — Multi-spectrum fitting (M9), built fresh rather than
ported: main.py's multi_fit() was a true no-op stub (`pass`).

Ships as a saved-recipe/template system (the Origin Themes / Spectragryph
Automate-tabs equivalent named in the plan), not just "loop the current fit
over N files": a recipe is a named, reusable peak-model configuration that
can be built once and re-applied, in one click, to any batch of spectra.

Recipes ARE param_models/*.json — the exact same save/load format M8's
FitParamDialog already writes — so a model built and refined interactively
on one spectrum in the Peak Fitting workspace is immediately usable here as
a batch recipe, and vice versa. No second file format.

Reuses fitting_science.fit_spectrum() per spectrum (mode="classic" only —
the Origin-like stepwise mode is inherently interactive/one-spectrum-at-a-
time, not a natural fit for unattended batch application), and shares one
PerItemSettingsStore with SingleFitWorkspace (qt_shell.py wires the same
instance into both) so a batch run's write-back is immediately visible in
Peak Fitting, and vice versa.
"""
from __future__ import annotations

import copy
import csv
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMessageBox, QPushButton, QSplitter,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from core.fitting_science import (
    compute_r_squared,
    fit_spectrum, peak_centroid,
)
from processing.qt_fit_params import FitParamDialog, list_model_names, load_model, _default_model_dir
from core.qt_models import SpectrumLibrary
from core.qt_settings_store import PerItemSettingsStore
from core.qt_widgets import PlotWidget, to_float as _to_float

COLORS = ["black", "red", "seagreen", "royalblue", "orange", "purple", "brown", "indigo"]


@dataclass
class _BatchFitResult:
    spectrum_id: str
    spectrum_title: str
    params_struct: List[Dict[str, Any]]
    chi2_red: float
    r2: float
    x: np.ndarray
    y: np.ndarray
    y_fit: np.ndarray
    peaks: List[np.ndarray] = field(default_factory=list)
    error: Optional[str] = None


class MultiFitWorkspace(QWidget):
    def __init__(
        self, parent: Optional[QWidget] = None, library: Optional[SpectrumLibrary] = None,
        fit_param_memory: Optional[PerItemSettingsStore] = None, model_dir: Optional[str] = None,
    ):
        super().__init__(parent)
        self.library = library if library is not None else SpectrumLibrary()
        self.fit_param_memory: PerItemSettingsStore[List[Dict[str, Any]]] = (
            fit_param_memory if fit_param_memory is not None else PerItemSettingsStore(list)
        )
        self.model_dir = model_dir or _default_model_dir()
        self._results: List[_BatchFitResult] = []

        self._build_ui()
        self._refresh_recipe_list()

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

        left_layout.addWidget(QLabel("Select spectra to batch-fit"))
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.file_list.itemSelectionChanged.connect(
            lambda: self.plot.request_redraw(self._draw_inputs_preview))
        left_layout.addWidget(self.file_list, 1)

        recipe_row = QHBoxLayout()
        recipe_row.addWidget(QLabel("Recipe"))
        self.recipe_combo = QComboBox()
        recipe_row.addWidget(self.recipe_combo, 1)
        left_layout.addLayout(recipe_row)

        recipe_btn_row = QHBoxLayout()
        new_recipe_btn = QPushButton("New…")
        new_recipe_btn.clicked.connect(self._new_recipe)
        edit_recipe_btn = QPushButton("Edit…")
        edit_recipe_btn.clicked.connect(self._edit_recipe)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_recipe_list)
        recipe_btn_row.addWidget(new_recipe_btn)
        recipe_btn_row.addWidget(edit_recipe_btn)
        recipe_btn_row.addWidget(refresh_btn)
        left_layout.addLayout(recipe_btn_row)

        self.norm_check = QCheckBox("Normalize")
        left_layout.addWidget(self.norm_check)

        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("Fit X range"))
        self.xmin_edit = QLineEdit()
        self.xmax_edit = QLineEdit()
        range_row.addWidget(self.xmin_edit)
        range_row.addWidget(self.xmax_edit)
        left_layout.addLayout(range_row)

        self.writeback_check = QCheckBox("Write results into per-spectrum fit params")
        self.writeback_check.setChecked(True)
        left_layout.addWidget(self.writeback_check)

        self._run_btn = QPushButton("Run batch fit")
        self._run_btn.setObjectName("Primary")
        self._run_btn.clicked.connect(self.run_batch)
        left_layout.addWidget(self._run_btn)
        left_layout.addStretch(1)
        splitter.addWidget(left)

        right = QWidget()
        right.setObjectName("Card")
        right_layout = QVBoxLayout(right)

        right_layout.addWidget(QLabel("Results (one row per fitted component)"))
        self.results_table = QTableWidget(0, 9)
        self.results_table.setHorizontalHeaderLabels(
            ["Spectrum", "Comp", "Center", "FWHM", "Amplitude", "Area", "Centroid", "Chi2_red", "R2"]
        )
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.results_table.itemSelectionChanged.connect(self._on_result_row_selected)
        right_layout.addWidget(self.results_table, 1)

        export_row = QHBoxLayout()
        export_btn = QPushButton("Export results to CSV…")
        export_btn.clicked.connect(self.export_results_csv)
        export_row.addWidget(export_btn)
        export_row.addStretch(1)
        right_layout.addLayout(export_row)

        right_layout.addWidget(QLabel("Preview (selected row's spectrum)"))
        self.plot = PlotWidget(figsize=(7, 3.5))
        right_layout.addWidget(self.plot, 1)

        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

    # ------------------------------------------------------------------
    def set_spectra(self, spectrum_ids: List[str]) -> None:
        selected_ids = {
            self.file_list.item(i).data(Qt.UserRole)
            for i in range(self.file_list.count())
            if self.file_list.item(i).isSelected()
        }
        self.file_list.clear()
        for sid in spectrum_ids:
            spectrum = self.library.get(sid)
            if spectrum is None:
                continue
            item = QListWidgetItem(spectrum.title)
            item.setData(Qt.UserRole, sid)
            self.file_list.addItem(item)
            if sid in selected_ids:
                item.setSelected(True)
        # Page entry never opens blank (user request): show the selected —
        # or first — spectra raw until a batch result takes over the plot.
        if not self.file_list.selectedItems() and self.file_list.count():
            self.file_list.item(0).setSelected(True)  # fires the preview hook
        else:
            self.plot.request_redraw(self._draw_inputs_preview)

    def _draw_inputs_preview(self) -> None:
        """Pre-batch view (page entry / selection change): the selected
        input spectra, raw. A computed batch preview is never clobbered."""
        if self._results:
            return
        fig = self.plot.figure
        fig.clf()
        ax = fig.add_subplot(111)
        spectra = self._selected_spectra()
        for i, sp in enumerate(spectra):
            ax.plot(sp.x, sp.y, lw=0.9, color=COLORS[i % len(COLORS)], alpha=0.85, label=sp.title)
        if spectra:
            ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        self.plot.canvas.draw_idle()

    def _selected_spectra(self):
        out = []
        for item in self.file_list.selectedItems():
            spectrum = self.library.get(item.data(Qt.UserRole))
            if spectrum is not None:
                out.append(spectrum)
        return out

    # ------------------------------------------------------------------
    # Recipes (== param_models/*.json — same format as Peak Fitting's
    # "Save as model" / "Load model" buttons)
    # ------------------------------------------------------------------
    def _refresh_recipe_list(self) -> None:
        current = self.recipe_combo.currentText()
        names = list_model_names(self.model_dir)
        self.recipe_combo.blockSignals(True)
        self.recipe_combo.clear()
        self.recipe_combo.addItems(names)
        if current in names:
            self.recipe_combo.setCurrentText(current)
        self.recipe_combo.blockSignals(False)

    def _new_recipe(self) -> None:
        dlg = FitParamDialog(self, params_struct=None, on_accept=lambda _params: None, model_dir=self.model_dir)
        dlg.exec()
        self._refresh_recipe_list()

    def _edit_recipe(self) -> None:
        name = self.recipe_combo.currentText()
        if not name:
            QMessageBox.information(self, "Edit recipe", "No recipe selected.")
            return
        try:
            params = load_model(name, self.model_dir)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Edit recipe", str(exc))
            return
        dlg = FitParamDialog(self, params_struct=params, on_accept=lambda _params: None, model_dir=self.model_dir)
        dlg.exec()
        self._refresh_recipe_list()

    # ------------------------------------------------------------------
    def _get_xy(self, spectrum):
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

    def run_batch(self) -> None:
        """Validates inputs on the UI thread, then runs the fits on a
        background worker (qt_worker) so a long batch never freezes the
        app; results land back on the main thread in _on_batch_done."""
        from core.qt_worker import run_in_thread

        recipe_name = self.recipe_combo.currentText()
        if not recipe_name:
            QMessageBox.warning(self, "No recipe", "Create or select a recipe first.")
            return
        try:
            template = load_model(recipe_name, self.model_dir)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Recipe error", str(exc))
            return

        spectra = self._selected_spectra()
        if not spectra:
            QMessageBox.warning(self, "No spectra", "Select at least one spectrum to fit.")
            return

        # Snapshot everything the worker needs NOW — it must not touch
        # widgets or live library objects from the background thread.
        jobs = [(sp.id, sp.title, *self._get_xy(sp)) for sp in spectra]
        writeback_wanted = self.writeback_check.isChecked()

        self._run_btn.setEnabled(False)
        self._run_btn.setText("Running…")

        def compute(jobs=jobs, template=template):
            results, errors = [], []
            for sid, title, x, y in jobs:
                params_struct = copy.deepcopy(template)
                try:
                    fr = fit_spectrum(x, y, params_struct, mode="classic")
                    self._writeback(params_struct, fr.lmfit_result)  # pure dict mutation
                    r2 = compute_r_squared(y, fr.y_fit)
                    results.append(_BatchFitResult(
                        spectrum_id=sid, spectrum_title=title,
                        params_struct=params_struct, chi2_red=fr.chi2_red, r2=r2,
                        x=x, y=y, y_fit=fr.y_fit, peaks=fr.peaks,
                    ))
                except Exception as exc:
                    errors.append(f"{title}: {exc}")
                    results.append(_BatchFitResult(
                        spectrum_id=sid, spectrum_title=title,
                        params_struct=params_struct, chi2_red=float("nan"), r2=float("nan"),
                        x=x, y=y, y_fit=np.full_like(x, np.nan), peaks=[], error=str(exc),
                    ))
            return results, errors, writeback_wanted

        run_in_thread(compute, self._on_batch_done, self._on_batch_error)

    def _on_batch_error(self, traceback_text: str) -> None:
        self._run_btn.setEnabled(True)
        self._run_btn.setText("Run batch fit")
        QMessageBox.critical(self, "Batch fit error", traceback_text)

    def _on_batch_done(self, payload) -> None:
        results, errors, writeback_wanted = payload
        self._run_btn.setEnabled(True)
        self._run_btn.setText("Run batch fit")
        self._results = results
        if writeback_wanted:
            for res in results:
                if res.error is None:
                    self.fit_param_memory.set(res.spectrum_id, res.params_struct)

        self._populate_results_table()
        if errors:
            QMessageBox.warning(self, "Batch fit", "Some spectra failed to fit:\n" + "\n".join(errors))

    def _writeback(self, params_struct: List[Dict[str, Any]], result) -> None:
        for i, d in enumerate(params_struct):
            d["shift_val"] = float(result.params[f"f{i}"].value)
            d["fwhm_val"] = float(result.params[f"l{i}"].value)
            if f"a{i}" in result.params:
                d["amp_val"] = float(result.params[f"a{i}"].value)
            if d.get("shape", "G") in ("GL", "V") and f"eta{i}" in result.params:
                d["eta_val"] = float(result.params[f"eta{i}"].value)
            if d.get("shape", "G") == "EMG" and f"s{i}" in result.params:
                d["skew_val"] = float(result.params[f"s{i}"].value)

    # ------------------------------------------------------------------
    def _populate_results_table(self) -> None:
        rows = []
        for res in self._results:
            if res.error is not None:
                rows.append((res.spectrum_title, "ERROR", res.error, "", "", "", "", "", ""))
                continue
            for i, d in enumerate(res.params_struct):
                area = float(np.trapz(res.peaks[i], res.x)) if i < len(res.peaks) else float("nan")
                centroid = peak_centroid(res.x, res.peaks[i]) if i < len(res.peaks) else float("nan")
                comp_label = f"{i + 1} ({d['name']})" if d.get("name") else str(i + 1)
                rows.append((
                    res.spectrum_title, comp_label, f"{d['shift_val']:.2f}", f"{d['fwhm_val']:.2f}",
                    f"{d.get('amp_val', float('nan')):.4g}", f"{area:.2f}", f"{centroid:.2f}",
                    f"{res.chi2_red:.4g}", f"{res.r2:.4f}",
                ))

        self.results_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                self.results_table.setItem(r, c, QTableWidgetItem(str(val)))
        self.results_table.resizeColumnsToContents()
        if self._results:
            self.results_table.selectRow(0)

    def _on_result_row_selected(self) -> None:
        rows = self.results_table.selectionModel().selectedRows()
        if not rows or not self._results:
            return
        row_idx = rows[0].row()
        # Map the selected table row back to its spectrum (rows are one-per-
        # component, spectra can have multiple components).
        cursor = 0
        target = None
        for res in self._results:
            n = 1 if res.error is not None else max(len(res.params_struct), 1)
            if cursor <= row_idx < cursor + n:
                target = res
                break
            cursor += n
        if target is None or target.error is not None:
            return
        self._draw_preview(target)

    def _draw_preview(self, res: _BatchFitResult) -> None:
        self.plot.cancel_pending()  # direct draw supersedes any queued inputs preview
        fig = self.plot.figure
        fig.clf()
        ax = fig.add_subplot(111)
        ax.plot(res.x, res.y, color="black", lw=1.1, label="Data")
        ax.plot(res.x, res.y_fit, color="red", lw=1.8, ls="--", label="Fit")
        for i, pk in enumerate(res.peaks):
            name = res.params_struct[i].get("name") if i < len(res.params_struct) else None
            ax.plot(res.x, pk, lw=1.0, color=COLORS[(i + 2) % len(COLORS)], alpha=0.7,
                    label=name or None)
        ax.set_title(f"{res.spectrum_title}  (chi2_red={res.chi2_red:.3g}, R2={res.r2:.3f})", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        self.plot.canvas.draw_idle()

    # ------------------------------------------------------------------
    def export_results_csv(self) -> None:
        if not self._results:
            QMessageBox.information(self, "Export", "No batch results to export yet.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export results as…", "", "CSV (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["spectrum", "component", "center", "fwhm", "amplitude", "area", "centroid", "shape", "eta", "chi2_red", "r2", "error"])
                for res in self._results:
                    if res.error is not None:
                        writer.writerow([res.spectrum_title, "", "", "", "", "", "", "", "", "", "", res.error])
                        continue
                    for i, d in enumerate(res.params_struct):
                        area = float(np.trapz(res.peaks[i], res.x)) if i < len(res.peaks) else float("nan")
                        centroid = peak_centroid(res.x, res.peaks[i]) if i < len(res.peaks) else float("nan")
                        eta = d.get("eta_val", "") if d.get("shape", "G") == "GL" else ""
                        writer.writerow([
                            res.spectrum_title, d.get("name") or i + 1, d["shift_val"], d["fwhm_val"],
                            d.get("amp_val", ""), area, centroid, d.get("shape", "G"), eta,
                            res.chi2_red, res.r2, "",
                        ])
        except OSError as exc:
            QMessageBox.critical(self, "Export error", str(exc))
            return
        QMessageBox.information(self, "Export", f"Results exported to:\n{path}")
