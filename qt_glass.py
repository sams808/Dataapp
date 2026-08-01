"""
qt_glass.py — the Glass workspace: composition-based property calculation.

Input: paste a composition table (header = oxides, rows = samples, optional
first name column) or load a CSV — in mol% or wt%.
Outputs: optical basicity Λ (PNNL-20184 Table B.1 recommended values,
Duffy oxygen-weighted mixing) for every sample, and — when GlassPy is
installed — GlassNet machine-learning predictions (Tg, viscosity
parameters, density, refractive index, moduli and ~80 more properties;
Cassar 2023, trained on SciGlass). Results export to CSV.
"""
from __future__ import annotations

from typing import List, Optional

from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPlainTextEdit, QPushButton, QSplitter, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

import glass_science as gs

SOURCES_NOTE = (
    "Λ: Rodriguez & McCloy, PNNL-20184 (2011) Table B.1 (Duffy & Ingram framework). "
    "ML predictions: GlassNet — Cassar, Ceram. Int. 49 (2023) 36013, trained on SciGlass. "
    "Compare also: SciGlass (github.com/epam/SciGlass), INTERGLAD (newglass.jp), "
    "glassproperties.com."
)


class GlassWorkspace(QWidget):
    def __init__(self, parent: Optional[QWidget] = None, library=None):
        super().__init__(parent)
        self._df = None
        self._results = None
        self._build_ui()

    def set_spectra(self, spectrum_ids: List[str]) -> None:  # nav hook (table-based workspace)
        pass

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        splitter = QSplitter()
        root.addWidget(splitter)

        left = QWidget()
        left.setObjectName("Card")
        left.setMaximumWidth(380)
        ll = QVBoxLayout(left)
        ll.addWidget(QLabel("Composition table — paste (header = oxides,\nrows = samples, optional name column) or load CSV"))
        self.comp_edit = QPlainTextEdit()
        self.comp_edit.setPlainText("name SiO2 Na2O Al2O3 B2O3 Bi2O3\n1Bi 49.5 29.7 4.9 14.8 1.0\n3Bi 48.5 29.1 4.9 14.6 3.0")
        ll.addWidget(self.comp_edit, 1)
        row = QHBoxLayout()
        load_btn = QPushButton("Load CSV…")
        load_btn.clicked.connect(self.load_csv)
        row.addWidget(load_btn)
        row.addWidget(QLabel("basis"))
        self.basis_combo = QComboBox()
        self.basis_combo.addItems(["mol", "wt"])
        row.addWidget(self.basis_combo)
        ll.addLayout(row)

        ob_btn = QPushButton("Optical basicity Λ")
        ob_btn.setObjectName("Primary")
        ob_btn.clicked.connect(self.compute_basicity)
        ll.addWidget(ob_btn)

        gn_row = QHBoxLayout()
        self.gn_btn = QPushButton("GlassNet predict")
        self.gn_btn.clicked.connect(self.run_glassnet)
        gn_row.addWidget(self.gn_btn)
        self.gn_filter_edit = QLineEdit()
        self.gn_filter_edit.setPlaceholderText("property filter (e.g. Tg, density)")
        gn_row.addWidget(self.gn_filter_edit)
        ll.addLayout(gn_row)
        if not gs.glassnet_available():
            self.gn_btn.setEnabled(False)
            self.gn_btn.setToolTip("pip install glasspy (needs PyTorch) to enable GlassNet predictions.")

        export_btn = QPushButton("Export results CSV…")
        export_btn.clicked.connect(self.export_csv)
        ll.addWidget(export_btn)

        src = QLabel(SOURCES_NOTE)
        src.setWordWrap(True)
        src.setObjectName("SectionNote")
        ll.addWidget(src)
        splitter.addWidget(left)

        right = QWidget()
        right.setObjectName("Card")
        rl = QVBoxLayout(right)
        self.result_table = QTableWidget(0, 0)
        self.result_table.setEditTriggers(QTableWidget.NoEditTriggers)
        rl.addWidget(self.result_table)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        rl.addWidget(self.status_label)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

    # ------------------------------------------------------------------
    def load_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Composition CSV", "", "CSV (*.csv *.txt);;All files (*.*)")
        if not path:
            return
        with open(path, encoding="utf-8", errors="replace") as f:
            self.comp_edit.setPlainText(f.read())

    def _parse(self):
        return gs.parse_composition_table(self.comp_edit.toPlainText())

    def _show_df(self, df) -> None:
        self._results = df
        self.result_table.setRowCount(len(df))
        self.result_table.setColumnCount(len(df.columns))
        self.result_table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        self.result_table.setVerticalHeaderLabels([str(i) for i in df.index])
        for r in range(len(df)):
            for c in range(len(df.columns)):
                v = df.iloc[r, c]
                self.result_table.setItem(r, c, QTableWidgetItem(f"{v:.5g}" if isinstance(v, float) else str(v)))
        self.result_table.resizeColumnsToContents()

    def compute_basicity(self) -> None:
        import pandas as pd
        try:
            df = self._parse()
            basis = self.basis_combo.currentText()
            rows = {}
            for name, row in df.iterrows():
                comps = [(str(c), float(v)) for c, v in row.items() if float(v) > 0]
                rows[name] = gs.optical_basicity(comps, basis=basis)["basicity"]
        except Exception as exc:
            QMessageBox.warning(self, "Optical basicity", str(exc))
            return
        out = pd.DataFrame({"optical_basicity": rows})
        self._show_df(out)
        self.status_label.setText(gs.OPTICAL_BASICITY_SOURCE)

    def run_glassnet(self) -> None:
        # The first call imports glasspy/PyTorch and loads the model (tens of
        # seconds cold) — run it in a worker so the UI never freezes.
        try:
            df = self._parse()
        except Exception as exc:
            QMessageBox.warning(self, "GlassNet", str(exc))
            return
        self.gn_btn.setEnabled(False)
        self.gn_btn.setText("Predicting… (first run loads the model)")
        from core.qt_worker import run_in_thread
        run_in_thread(lambda: gs.glassnet_predict(df), self._on_glassnet_done, self._on_glassnet_error)

    def _on_glassnet_error(self, traceback_text: str) -> None:
        self.gn_btn.setEnabled(True)
        self.gn_btn.setText("GlassNet predict")
        QMessageBox.critical(self, "GlassNet", traceback_text)

    def _on_glassnet_done(self, pred) -> None:
        self.gn_btn.setEnabled(True)
        self.gn_btn.setText("GlassNet predict")
        filt = self.gn_filter_edit.text().strip().lower()
        if filt:
            keep = [c for c in pred.columns if filt in str(c).lower()]
            if keep:
                pred = pred[keep]
        self._show_df(pred)
        self.status_label.setText("GlassNet (Cassar 2023, SciGlass-trained) — treat as estimates; validate critical values experimentally.")

    def export_csv(self) -> None:
        if self._results is None:
            QMessageBox.information(self, "Export", "Compute something first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export results", "", "CSV (*.csv)")
        if path:
            self._results.to_csv(path)
            self.status_label.setText(f"Saved {path}")
