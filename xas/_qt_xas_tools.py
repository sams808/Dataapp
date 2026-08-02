"""
xas/_qt_xas_tools.py — internal implementation detail of qt_xas.py:
XasWorkspace's Tools tab (edge definer). Mixed into XasWorkspace, not
meant to be used standalone.
"""
from __future__ import annotations

from typing import List

import numpy as np

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QListWidget, QMessageBox,
    QPushButton, QVBoxLayout, QWidget,
)

from core.qt_widgets import PlotWidget
from xas.xas_science import Operation, Spectrum, _periodic_table_symbols, require_larch
from ._qt_xas_shared import COLORS


class ToolsTabMixin:
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
