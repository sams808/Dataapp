"""
xas/_qt_xas_export.py — internal implementation detail of qt_xas.py:
XasWorkspace's Export tab (Athena .dat/.prj export). Mixed into
XasWorkspace, not meant to be used standalone.
"""
from __future__ import annotations

import numpy as np

from PySide6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout,
    QWidget,
)

from xas.xas_science import export_athena_column, export_athena_prj_best_effort


class ExportTabMixin:
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
