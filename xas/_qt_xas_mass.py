"""
xas/_qt_xas_mass.py — internal implementation detail of qt_xas.py:
XasWorkspace's Sample mass tab (Hephaestus-style sample-mass calculator,
xas_mass.py). Mixed into XasWorkspace, not meant to be used standalone.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
    QVBoxLayout, QWidget,
)


class MassTabMixin:
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
