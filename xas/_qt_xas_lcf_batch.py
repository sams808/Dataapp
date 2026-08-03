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
"""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import List

import numpy as np

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
        ctrl.setMaximumWidth(360)
        ctrl_layout = QVBoxLayout(ctrl)

        ctrl_layout.addWidget(QLabel("Targets (samples, multi-select)"))
        self.lcf_targets_list = QListWidget()
        self.lcf_targets_list.setSelectionMode(QListWidget.ExtendedSelection)
        ctrl_layout.addWidget(self.lcf_targets_list, 1)

        ctrl_layout.addWidget(QLabel("References (standards, multi-select)"))
        self.lcf_refs_list = QListWidget()
        self.lcf_refs_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.lcf_refs_list.itemSelectionChanged.connect(self._refresh_lcf_required_list)
        ctrl_layout.addWidget(self.lcf_refs_list, 1)

        ctrl_layout.addWidget(QLabel("Always include (optional subset of references above)"))
        self.lcf_required_list = QListWidget()
        self.lcf_required_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.lcf_required_list.setMaximumHeight(70)
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
        bounds_row.addWidget(QLabel("weight min"))
        self.lcf_weight_lb_edit = QLineEdit("0.0")
        self.lcf_weight_lb_edit.setMaximumWidth(55)
        bounds_row.addWidget(self.lcf_weight_lb_edit)
        bounds_row.addWidget(QLabel("max"))
        self.lcf_weight_ub_edit = QLineEdit("1.0")
        self.lcf_weight_ub_edit.setMaximumWidth(55)
        bounds_row.addWidget(self.lcf_weight_ub_edit)
        ctrl_layout.addLayout(bounds_row)

        self.lcf_sum_to_one_check = QCheckBox("Constrain weights to sum to 1")
        ctrl_layout.addWidget(self.lcf_sum_to_one_check)

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

        report_btn = QPushButton("Generate report (PDF + MD)…")
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
    def _lcf_params(self) -> BatchLCFParams:
        lb = _to_float(self.lcf_weight_lb_edit.text(), 0.0)
        ub = _to_float(self.lcf_weight_ub_edit.text(), 1.0)
        required = tuple(item.text() for item in self.lcf_required_list.selectedItems())
        return BatchLCFParams(
            min_components=self.lcf_min_components_spin.value(),
            max_components=self.lcf_max_components_spin.value(),
            weight_bounds=(lb, ub),
            sum_to_one=self.lcf_sum_to_one_check.isChecked(),
            required_refs=required,
            sort_by=self.lcf_sort_combo.currentText(),
            top_n_report=self.lcf_top_n_spin.value(),
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

        params = self._lcf_params()
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
        via build_fit_overlay_figure, for a proper standalone figure)."""
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

        ax.plot(target_energy, target_y, color="black", lw=1.5, label=f"{name} (data)", zorder=5)
        ax.plot(target_energy, best.fit_y, color="red", lw=1.6, ls="--", label=f"fit (R²={best.r2:.4f})", zorder=4)
        colors = COLORS
        for i, (ref_name, weight) in enumerate(zip(best.ref_names, best.weights)):
            ref_e, ref_y = self._lcf_ref_lookup[ref_name]
            ref_interp = _interp_to_grid(ref_e, ref_y, target_energy)
            ax.plot(target_energy, weight * ref_interp, color=colors[i % len(colors)], lw=1.0, alpha=0.8,
                    label=f"{ref_name} × {weight:.3f}")
        ax.set_xlabel("Energy (eV)"); ax.set_ylabel("Normalized signal")
        ax.set_title(f"{name} — best: {' + '.join(best.ref_names)}", fontsize=10)
        ax.legend(fontsize=7.5); ax.grid(alpha=0.25)
        self.lcf_plot.figure.tight_layout()
        self.lcf_plot.canvas.draw_idle()

    def generate_batch_lcf_report_clicked(self) -> None:
        if not self._lcf_batch_results:
            QMessageBox.warning(self, "Batch LCF report", "Run the batch fit first.")
            return
        out_dir_str = QFileDialog.getExistingDirectory(self, "Select output folder for the report")
        if not out_dir_str:
            return
        params = self._lcf_params()
        try:
            out = build_batch_report(
                self._lcf_batch_results, self._lcf_target_lookup, self._lcf_ref_lookup,
                sort_by=params.sort_by, top_n=params.top_n_report, out_dir=Path(out_dir_str),
                report_name="batch_lcf_report",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Batch LCF report error", str(exc))
            return
        self.lcf_status_label.setText(f"Report written: {out['pdf'].name}, {out['md'].name}")
        QMessageBox.information(self, "Batch LCF report", f"Wrote:\n{out['pdf']}\n{out['md']}")
