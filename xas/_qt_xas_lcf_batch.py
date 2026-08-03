"""
xas/_qt_xas_lcf_batch.py — internal implementation detail of qt_xas.py:
XasWorkspace's Batch LCF tab (combinatorial linear combination fitting
across many samples at once, with a PDF+MD report). Mixed into
XasWorkspace, not meant to be used standalone.

Targets and references are both selected from the SAME shared object
list every other tab reads from (self.store) -- same convention as the
Analysis tab's single-fit LCF, so "import a lot of normalized spectra"
just means importing them via the existing CSV.../ZIP... buttons like
any other spectra, then picking which ones are targets vs. references
here.

Tuning controls (align e0, per-reference weight bounds, fit range) were
added after a real-data review found Bi_metal picking up a substantial,
chemically-implausible weight across nearly every oxide glass sample --
the residual was a sharp, localized spike right at the edge, the
signature of energy-misaligned references being absorbed into an
unphysical weight, not a real 4th phase. See xas/lcf_batch.py's module
docstring for the full story.
"""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QMessageBox, QPushButton, QSpinBox, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from core.qt_widgets import PlotWidget, to_float as _to_float
from xas.lcf_batch import BatchLCFParams, run_batch_lcf
from xas.lcf_report import build_batch_report
from xas.xas_science import Spectrum, _interp_to_grid
from ._qt_xas_shared import COLORS

_MAX_FITS_WITHOUT_CONFIRM = 5000


class LcfBatchTabMixin:
    def _build_lcf_batch_tab(self) -> QWidget:
        w = QWidget()
        layout = QHBoxLayout(w)

        ctrl = QWidget()
        ctrl.setMaximumWidth(380)
        ctrl_layout = QVBoxLayout(ctrl)

        ctrl_layout.addWidget(QLabel("Targets (samples, multi-select)"))
        self.lcf_targets_list = QListWidget()
        self.lcf_targets_list.setSelectionMode(QListWidget.ExtendedSelection)
        ctrl_layout.addWidget(self.lcf_targets_list, 1)

        ctrl_layout.addWidget(QLabel("References (standards, multi-select)"))
        self.lcf_refs_list = QListWidget()
        self.lcf_refs_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.lcf_refs_list.itemSelectionChanged.connect(self._on_lcf_refs_selection_changed)
        ctrl_layout.addWidget(self.lcf_refs_list, 1)

        ctrl_layout.addWidget(QLabel("Always include (optional subset of references above)"))
        self.lcf_required_list = QListWidget()
        self.lcf_required_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.lcf_required_list.setMaximumHeight(60)
        ctrl_layout.addWidget(self.lcf_required_list)

        comp_row = QHBoxLayout()
        comp_row.addWidget(QLabel("components min"))
        self.lcf_min_components_spin = QSpinBox()
        self.lcf_min_components_spin.setRange(1, 10)
        self.lcf_min_components_spin.setValue(2)
        comp_row.addWidget(self.lcf_min_components_spin)
        comp_row.addWidget(QLabel("max"))
        self.lcf_max_components_spin = QSpinBox()
        self.lcf_max_components_spin.setRange(1, 10)
        self.lcf_max_components_spin.setValue(3)
        comp_row.addWidget(self.lcf_max_components_spin)
        ctrl_layout.addLayout(comp_row)

        bounds_row = QHBoxLayout()
        bounds_row.addWidget(QLabel("default weight min"))
        self.lcf_weight_lb_edit = QLineEdit("0.0")
        self.lcf_weight_lb_edit.setMaximumWidth(50)
        bounds_row.addWidget(self.lcf_weight_lb_edit)
        bounds_row.addWidget(QLabel("max"))
        self.lcf_weight_ub_edit = QLineEdit("1.0")
        self.lcf_weight_ub_edit.setMaximumWidth(50)
        bounds_row.addWidget(self.lcf_weight_ub_edit)
        ctrl_layout.addLayout(bounds_row)

        ctrl_layout.addWidget(QLabel(
            "Per-reference weight bounds (blank cell = use the default above).\n"
            "Encode real expectations here, e.g. Bi2O3 min=0.3 to force it\n"
            "dominant, Bi_metal max=0.15 to keep it minor."
        ))
        self.lcf_bounds_table = QTableWidget(0, 3)
        self.lcf_bounds_table.setHorizontalHeaderLabels(["Reference", "min", "max"])
        self.lcf_bounds_table.setMaximumHeight(130)
        self.lcf_bounds_table.cellChanged.connect(self._on_lcf_bounds_table_changed)
        ctrl_layout.addWidget(self.lcf_bounds_table)
        self._lcf_bounds_cache: Dict[str, Tuple[str, str]] = {}

        self.lcf_sum_to_one_check = QCheckBox("Constrain weights to sum to 1")
        ctrl_layout.addWidget(self.lcf_sum_to_one_check)

        self.lcf_align_e0_check = QCheckBox("Align references to target e0 before fitting")
        self.lcf_align_e0_check.setToolTip(
            "Shifts each reference's own energy axis (derivative-estimated "
            "edge position) onto the target's own edge before fitting. Turn "
            "this on if a residual shows a sharp spike right at the edge -- "
            "that's energy misalignment being absorbed into a weight, not a "
            "real extra phase."
        )
        ctrl_layout.addWidget(self.lcf_align_e0_check)

        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("fit range (eV)"))
        self.lcf_fit_range_lo_edit = QLineEdit("")
        self.lcf_fit_range_lo_edit.setPlaceholderText("min (blank = full)")
        range_row.addWidget(self.lcf_fit_range_lo_edit)
        self.lcf_fit_range_hi_edit = QLineEdit("")
        self.lcf_fit_range_hi_edit.setPlaceholderText("max (blank = full)")
        range_row.addWidget(self.lcf_fit_range_hi_edit)
        ctrl_layout.addLayout(range_row)

        sort_row = QHBoxLayout()
        sort_row.addWidget(QLabel("Rank by"))
        self.lcf_sort_combo = QComboBox()
        self.lcf_sort_combo.addItems(["r2", "rms", "reduced_chi_square"])
        sort_row.addWidget(self.lcf_sort_combo)
        sort_row.addWidget(QLabel("top N"))
        self.lcf_top_n_spin = QSpinBox()
        self.lcf_top_n_spin.setRange(1, 200)
        self.lcf_top_n_spin.setValue(10)
        sort_row.addWidget(self.lcf_top_n_spin)
        ctrl_layout.addLayout(sort_row)

        run_btn = QPushButton("Run batch LCF")
        run_btn.setObjectName("Primary")
        run_btn.clicked.connect(self.run_batch_lcf_clicked)
        ctrl_layout.addWidget(run_btn)

        ctrl_layout.addWidget(QLabel("Report output"))
        report_row = QHBoxLayout()
        self.lcf_save_pdf_check = QCheckBox("PDF")
        self.lcf_save_pdf_check.setChecked(True)
        report_row.addWidget(self.lcf_save_pdf_check)
        self.lcf_save_md_check = QCheckBox("Markdown")
        self.lcf_save_md_check.setChecked(True)
        report_row.addWidget(self.lcf_save_md_check)
        ctrl_layout.addLayout(report_row)

        ctrl_layout.addWidget(QLabel("Individual page images"))
        image_row = QHBoxLayout()
        self.lcf_save_png_check = QCheckBox("PNG")
        self.lcf_save_png_check.setChecked(True)
        image_row.addWidget(self.lcf_save_png_check)
        self.lcf_save_svg_check = QCheckBox("SVG")
        image_row.addWidget(self.lcf_save_svg_check)
        ctrl_layout.addLayout(image_row)

        report_btn = QPushButton("Generate report…")
        report_btn.clicked.connect(self.generate_batch_lcf_report_clicked)
        ctrl_layout.addWidget(report_btn)

        self.lcf_status_label = QLabel("")
        self.lcf_status_label.setWordWrap(True)
        ctrl_layout.addWidget(self.lcf_status_label)
        ctrl_layout.addStretch(1)
        layout.addWidget(ctrl)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel("Per-sample summary (best combination) — click a row to preview"))
        self.lcf_summary_table = QTableWidget(0, 5)
        self.lcf_summary_table.setHorizontalHeaderLabels(["Target", "Best combination", "R²", "RMS", "reduced χ²"])
        self.lcf_summary_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.lcf_summary_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.lcf_summary_table.itemSelectionChanged.connect(self._preview_lcf_selected_target)
        self.lcf_summary_table.setMaximumHeight(220)
        right_layout.addWidget(self.lcf_summary_table)

        self.lcf_plot = PlotWidget(figsize=(7, 6))
        right_layout.addWidget(self.lcf_plot, 1)
        layout.addWidget(right, 1)

        self._lcf_batch_results = {}
        self._lcf_target_lookup = {}
        self._lcf_ref_lookup = {}
        return w

    # ------------------------------------------------------------------
    def _on_lcf_refs_selection_changed(self) -> None:
        self._refresh_lcf_required_list()
        self._refresh_lcf_bounds_table()

    def _on_lcf_bounds_table_changed(self, row: int, _col: int) -> None:
        name_item = self.lcf_bounds_table.item(row, 0)
        if name_item is None:
            return
        min_item = self.lcf_bounds_table.item(row, 1)
        max_item = self.lcf_bounds_table.item(row, 2)
        self._lcf_bounds_cache[name_item.text()] = (
            min_item.text().strip() if min_item else "",
            max_item.text().strip() if max_item else "",
        )

    def _refresh_lcf_bounds_table(self) -> None:
        selected_refs = [item.text() for item in self.lcf_refs_list.selectedItems()]
        self.lcf_bounds_table.blockSignals(True)
        self.lcf_bounds_table.setRowCount(len(selected_refs))
        for row, name in enumerate(selected_refs):
            name_item = QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.lcf_bounds_table.setItem(row, 0, name_item)
            min_text, max_text = self._lcf_bounds_cache.get(name, ("", ""))
            self.lcf_bounds_table.setItem(row, 1, QTableWidgetItem(min_text))
            self.lcf_bounds_table.setItem(row, 2, QTableWidgetItem(max_text))
        self.lcf_bounds_table.blockSignals(False)
        self.lcf_bounds_table.resizeColumnsToContents()

    def _lcf_per_ref_bounds(self, default_lb: float, default_ub: float) -> Dict[str, Tuple[float, float]]:
        out: Dict[str, Tuple[float, float]] = {}
        for row in range(self.lcf_bounds_table.rowCount()):
            name_item = self.lcf_bounds_table.item(row, 0)
            if name_item is None:
                continue
            min_item = self.lcf_bounds_table.item(row, 1)
            max_item = self.lcf_bounds_table.item(row, 2)
            min_text = min_item.text().strip() if min_item else ""
            max_text = max_item.text().strip() if max_item else ""
            if not min_text and not max_text:
                continue  # both blank -> use the global default, no override needed
            lb = _to_float(min_text, default_lb) if min_text else default_lb
            ub = _to_float(max_text, default_ub) if max_text else default_ub
            out[name_item.text()] = (lb, ub)
        return out

    def _lcf_params(self) -> BatchLCFParams:
        lb = _to_float(self.lcf_weight_lb_edit.text(), 0.0)
        ub = _to_float(self.lcf_weight_ub_edit.text(), 1.0)
        required = tuple(item.text() for item in self.lcf_required_list.selectedItems())
        fit_lo = _to_float(self.lcf_fit_range_lo_edit.text())
        fit_hi = _to_float(self.lcf_fit_range_hi_edit.text())
        fit_range = (fit_lo, fit_hi) if (fit_lo is not None and fit_hi is not None) else None
        return BatchLCFParams(
            min_components=self.lcf_min_components_spin.value(),
            max_components=self.lcf_max_components_spin.value(),
            weight_bounds=(lb, ub),
            per_ref_bounds=self._lcf_per_ref_bounds(lb, ub),
            sum_to_one=self.lcf_sum_to_one_check.isChecked(),
            required_refs=required,
            sort_by=self.lcf_sort_combo.currentText(),
            top_n_report=self.lcf_top_n_spin.value(),
            align_e0=self.lcf_align_e0_check.isChecked(),
            fit_range=fit_range,
        )

    def _estimate_fit_count(self, n_targets: int, n_refs: int, params: BatchLCFParams) -> int:
        max_c = min(params.max_components, n_refs)
        min_c = min(params.min_components, max_c)
        n_combos = sum(len(list(itertools.combinations(range(n_refs), k))) for k in range(min_c, max_c + 1))
        return n_targets * n_combos

    def run_batch_lcf_clicked(self) -> None:
        target_items = self.lcf_targets_list.selectedItems()
        ref_items = self.lcf_refs_list.selectedItems()
        if not target_items:
            QMessageBox.warning(self, "Batch LCF", "Select at least one target spectrum.")
            return
        if len(ref_items) < self.lcf_min_components_spin.value():
            QMessageBox.warning(self, "Batch LCF", "Select at least as many references as the minimum component count.")
            return

        targets = []
        target_lookup = {}
        for item in target_items:
            sp = self.store.find_by_name(item.text())
            if sp is None:
                continue
            targets.append((sp.name, sp.energy, sp.y))
            target_lookup[sp.name] = (sp.energy, sp.y)

        refs = []
        ref_lookup = {}
        for item in ref_items:
            sp = self.store.find_by_name(item.text())
            if sp is None:
                continue
            refs.append((sp.name, sp.energy, sp.y))
            ref_lookup[sp.name] = (sp.energy, sp.y)

        try:
            params = self._lcf_params()
        except ValueError as exc:
            QMessageBox.critical(self, "Batch LCF", f"Bad parameter: {exc}")
            return
        n_fits = self._estimate_fit_count(len(targets), len(refs), params)
        if n_fits > _MAX_FITS_WITHOUT_CONFIRM:
            resp = QMessageBox.question(
                self, "Batch LCF",
                f"This will run {n_fits} fits (min_components={params.min_components}, "
                f"max_components={params.max_components}, {len(refs)} references, {len(targets)} targets). "
                f"That's a lot -- continue?",
            )
            if resp != QMessageBox.Yes:
                return

        self.lcf_status_label.setText(f"Running {n_fits} fit(s)…")
        try:
            results = run_batch_lcf(targets, refs, params)
        except Exception as exc:
            QMessageBox.critical(self, "Batch LCF error", str(exc))
            self.lcf_status_label.setText("")
            return

        self._lcf_batch_results = results
        self._lcf_target_lookup = target_lookup
        self._lcf_ref_lookup = ref_lookup
        self._refresh_lcf_summary_table()
        n_ok = sum(1 for v in results.values() if v)
        self.lcf_status_label.setText(f"Done: {n_ok}/{len(results)} target(s) fit successfully ({n_fits} combinations tried in total).")

    def _refresh_lcf_summary_table(self) -> None:
        rows = list(self._lcf_batch_results.items())
        self.lcf_summary_table.setRowCount(len(rows))
        for row, (name, results) in enumerate(rows):
            self.lcf_summary_table.setItem(row, 0, QTableWidgetItem(name))
            if not results:
                for col in range(1, 5):
                    self.lcf_summary_table.setItem(row, col, QTableWidgetItem("—"))
                continue
            best = results[0]
            self.lcf_summary_table.setItem(row, 1, QTableWidgetItem(" + ".join(best.ref_names)))
            self.lcf_summary_table.setItem(row, 2, QTableWidgetItem(f"{best.r2:.4f}"))
            self.lcf_summary_table.setItem(row, 3, QTableWidgetItem(f"{best.rms:.5f}"))
            self.lcf_summary_table.setItem(row, 4, QTableWidgetItem(f"{best.reduced_chi_square:.5f}"))
        self.lcf_summary_table.resizeColumnsToContents()

    def _preview_lcf_selected_target(self) -> None:
        """Quick on-screen check before generating the full report -- data,
        best-fit combination, and each weighted reference contribution, all
        on PlotWidget's one persistent axes (same ax.clear()-and-replot
        convention as every other tab's preview; the full 2-axes
        fit-plus-residual page is what the PDF/MD report itself renders,
        via build_fit_overlay_figure, for a proper standalone figure).
        Uses the result's own fit_energy (not the target's full range) and
        applies any align_e0 shift, so the preview matches the report even
        when fit_range/align_e0 are in play."""
        rows = self.lcf_summary_table.selectionModel().selectedRows()
        if not rows:
            return
        name = self.lcf_summary_table.item(rows[0].row(), 0).text()
        results = self._lcf_batch_results.get(name)
        ax = self.lcf_plot.ax
        ax.clear()
        if not results:
            ax.set_title(f"{name} — no successful fit")
            self.lcf_plot.canvas.draw_idle()
            return
        best = results[0]
        target_energy, target_y = self._lcf_target_lookup[name]

        fit_lo, fit_hi = float(np.min(best.fit_energy)), float(np.max(best.fit_energy))
        full_lo, full_hi = float(np.min(target_energy)), float(np.max(target_energy))
        if fit_lo > full_lo + 1e-6 or fit_hi < full_hi - 1e-6:
            ax.axvspan(fit_lo, fit_hi, color="0.85", zorder=0)

        ax.plot(target_energy, target_y, color="black", lw=1.5, label=f"{name} (data)", zorder=5)
        ax.plot(best.fit_energy, best.fit_y, color="red", lw=1.6, ls="--", label=f"fit (R²={best.r2:.4f})", zorder=4)
        colors = COLORS
        for i, (ref_name, weight) in enumerate(zip(best.ref_names, best.weights)):
            ref_e, ref_y = self._lcf_ref_lookup[ref_name]
            shift = best.e0_shifts_ev.get(ref_name, 0.0)
            ref_interp = _interp_to_grid(np.asarray(ref_e, float) + shift, ref_y, best.fit_energy)
            shift_note = f", e0 {shift:+.1f} eV" if shift else ""
            ax.plot(best.fit_energy, weight * ref_interp, color=colors[i % len(colors)], lw=1.0, alpha=0.8,
                    label=f"{ref_name} × {weight:.3f}{shift_note}")
        ax.set_xlabel("Energy (eV)"); ax.set_ylabel("Normalized signal")
        ax.set_title(f"{name} — best: {' + '.join(best.ref_names)}", fontsize=10)
        ax.legend(fontsize=7.5); ax.grid(alpha=0.25)
        self.lcf_plot.figure.tight_layout()
        self.lcf_plot.canvas.draw_idle()

    def generate_batch_lcf_report_clicked(self) -> None:
        if not self._lcf_batch_results:
            QMessageBox.warning(self, "Batch LCF report", "Run the batch fit first.")
            return
        save_pdf = self.lcf_save_pdf_check.isChecked()
        save_md = self.lcf_save_md_check.isChecked()
        image_formats = tuple(fmt for fmt, check in
                              [("png", self.lcf_save_png_check), ("svg", self.lcf_save_svg_check)]
                              if check.isChecked())
        if not (save_pdf or save_md):
            QMessageBox.warning(self, "Batch LCF report", "Check at least PDF or Markdown to save.")
            return

        # One "Save As" dialog for both name and location, matching how
        # every other export in this app works, rather than a folder
        # picker plus a separate fixed filename -- the dialog's own
        # extension is just a suggestion; whichever formats are checked
        # above get written next to whatever base name/location is chosen.
        if save_pdf:
            name_filter, default_suffix = "PDF files (*.pdf)", ".pdf"
        else:
            name_filter, default_suffix = "Markdown files (*.md)", ".md"
        default_path = str(Path.home() / f"batch_lcf_report{default_suffix}")
        chosen_path_str, _ = QFileDialog.getSaveFileName(self, "Save batch LCF report as…", default_path, name_filter)
        if not chosen_path_str:
            return
        chosen_path = Path(chosen_path_str)
        report_name = chosen_path.stem if chosen_path.suffix.lower() in (".pdf", ".md") else chosen_path.name
        out_dir = chosen_path.parent

        params = self._lcf_params()
        try:
            out = build_batch_report(
                self._lcf_batch_results, self._lcf_target_lookup, self._lcf_ref_lookup,
                sort_by=params.sort_by, top_n=params.top_n_report, out_dir=out_dir, report_name=report_name,
                save_pdf=save_pdf, save_md=save_md, image_formats=image_formats,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Batch LCF report error", str(exc))
            return

        written = "\n".join(str(p) for p in out.values())
        self.lcf_status_label.setText(f"Report written: {', '.join(p.name for p in out.values())}")
        QMessageBox.information(self, "Batch LCF report", f"Wrote:\n{written}")
