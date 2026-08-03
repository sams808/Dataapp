"""
xas/_qt_xas_analysis.py — internal implementation detail of qt_xas.py:
XasWorkspace's Analysis tab (Athena-inspired additions: merge/average,
difference spectra, PCA). Mixed into XasWorkspace, not meant to be used
standalone.

Linear combination fitting used to live here too (a single target vs. one
fixed set of references, via plain NNLS). It was removed once the LCF tab's
combinatorial engine made it fully redundant — running the LCF tab with one
target selected and min_components == max_components == (number of
references) reproduces the old single-fit behavior exactly, plus gives
weight bounds, e0 alignment, fit-range restriction, and PDF/MD reporting
the old button never had. See ._qt_xas_lcf_batch.
"""
from __future__ import annotations

from typing import List

import numpy as np

from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QListWidget, QMessageBox, QPushButton, QTextEdit,
    QVBoxLayout, QWidget,
)

from core.qt_widgets import PlotWidget
from xas.xas_science import (
    Operation,
    Spectrum,
    combine_spectra,
    difference_spectra,
)
from ._qt_xas_shared import COLORS


class AnalysisTabMixin:
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

        lcf_pointer = QLabel("Looking for linear combination fitting? See the LCF tab.")
        lcf_pointer.setStyleSheet("color: #777; font-style: italic;")
        lcf_pointer.setWordWrap(True)
        ctrl_layout.addWidget(lcf_pointer)

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
        try:
            combined = combine_spectra(ref.energy, ref.y, [(sp.energy, sp.y) for sp in specs[1:]], op)
        except ValueError as exc:
            # combine_spectra raises when a selected spectrum doesn't cover
            # the reference's full energy range (e.g. a repeat scan cut
            # short by an instrument fault) -- a real scenario found via
            # the Bi L3 real-data review, not a hypothetical.
            QMessageBox.critical(self, "Combine error", str(exc))
            return

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
        b_interp, diff_y = difference_spectra(a.energy, a.y, b.energy, b.y)

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
