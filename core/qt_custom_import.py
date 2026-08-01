"""
qt_custom_import.py — the Custom Import dialog: parser override + manual
X/Y column selection with a live preview. Restores the old Tk app's
"Custom import" (the fallback for files the auto-detection guesses wrong
on), which the first Qt pass dropped.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from PySide6.QtWidgets import (
    QComboBox, QDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QVBoxLayout,
)

import core.io_universal as io_universal
from core.qt_models import Spectrum
from core.qt_widgets import PlotWidget

AUTO = "(auto-detect)"


class CustomImportDialog(QDialog):
    """Import ONE file with explicit control: pick the parser, then pick
    the X and Y columns from whatever that parser produced, with a live
    preview. On accept, `spectrum` holds the resulting Spectrum."""

    def __init__(self, parent, path: str):
        super().__init__(parent)
        self.setWindowTitle(f"Custom import — {os.path.basename(path)}")
        self.resize(760, 480)
        self.path = path
        self.spectrum: Optional[Spectrum] = None
        self._df = None
        self._meta = None

        layout = QVBoxLayout(self)

        parser_row = QHBoxLayout()
        parser_row.addWidget(QLabel("Parser"))
        self.parser_combo = QComboBox()
        self.parser_combo.addItems([AUTO] + io_universal.available_parsers())
        self.parser_combo.currentTextChanged.connect(self._reparse)
        parser_row.addWidget(self.parser_combo)
        self.detected_label = QLabel("")
        self.detected_label.setObjectName("SectionNote")
        parser_row.addWidget(self.detected_label, 1)
        layout.addLayout(parser_row)

        cols_row = QHBoxLayout()
        cols_row.addWidget(QLabel("X column"))
        self.x_combo = QComboBox()
        self.x_combo.currentTextChanged.connect(self._update_preview)
        cols_row.addWidget(self.x_combo, 1)
        cols_row.addWidget(QLabel("Y column"))
        self.y_combo = QComboBox()
        self.y_combo.currentTextChanged.connect(self._update_preview)
        cols_row.addWidget(self.y_combo, 1)
        layout.addLayout(cols_row)

        self.plot = PlotWidget(figsize=(7, 3.2))
        layout.addWidget(self.plot, 1)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(cancel_btn)
        ok_btn = QPushButton("Import")
        ok_btn.setObjectName("Primary")
        ok_btn.clicked.connect(self._on_import)
        buttons.addWidget(ok_btn)
        layout.addLayout(buttons)

        self._reparse()

    # ------------------------------------------------------------------
    def _reparse(self, *_args) -> None:
        choice = self.parser_combo.currentText()
        prefer = None if choice == AUTO else choice
        try:
            df, meta = io_universal.load_any(self.path, prefer=prefer, return_meta=True)
        except Exception as exc:
            self._df, self._meta = None, None
            self.x_combo.clear()
            self.y_combo.clear()
            self.detected_label.setText(f"parse failed: {exc}")
            self.plot.clear("Parse failed")
            return
        self._df, self._meta = df, meta
        self.detected_label.setText(
            f"parsed as '{meta.get('selected_parser')}' — {len(df)} rows, {len(df.columns)} columns"
        )

        canon = meta.get("canonical_map", {}) or {}
        cols = [str(c) for c in df.columns]
        for combo, default in ((self.x_combo, canon.get("X")), (self.y_combo, canon.get("Y"))):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(cols)
            if default in cols:
                combo.setCurrentText(str(default))
            combo.blockSignals(False)
        if self.y_combo.currentText() == self.x_combo.currentText() and len(cols) > 1:
            self.y_combo.setCurrentIndex(1 if self.x_combo.currentIndex() == 0 else 0)
        self._update_preview()

    def _current_xy(self):
        if self._df is None:
            return None
        x_col, y_col = self.x_combo.currentText(), self.y_combo.currentText()
        if not x_col or not y_col:
            return None
        try:
            x = self._df[x_col].astype(float).to_numpy()
            y = self._df[y_col].astype(float).to_numpy()
        except (KeyError, TypeError, ValueError):
            return None
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 2:
            return None
        order = np.argsort(x[m], kind="mergesort")
        return x[m][order], y[m][order], x_col, y_col

    def _update_preview(self, *_args) -> None:
        got = self._current_xy()
        if got is None:
            self.plot.clear("No plottable X/Y with this selection")
            return
        x, y, x_col, y_col = got
        ax = self.plot.ax
        ax.clear()
        ax.plot(x, y, lw=1.0, color="#3c6e71")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(alpha=0.25)
        self.plot.figure.tight_layout()
        self.plot.canvas.draw_idle()

    def _on_import(self) -> None:
        got = self._current_xy()
        if got is None:
            QMessageBox.warning(self, "Custom import", "The current X/Y selection doesn't produce plottable numeric data.")
            return
        x, y, x_col, y_col = got
        meta = dict(self._meta or {})
        meta["custom_import"] = {"x_col": x_col, "y_col": y_col, "parser": meta.get("selected_parser")}
        self.spectrum = Spectrum(
            id=Spectrum.new_id(),
            title=os.path.splitext(os.path.basename(self.path))[0],
            path=self.path,
            kind=meta.get("selected_parser", "generic_xy"),
            x=x, y=y, df=self._df, meta=meta, status="imported",
        )
        self.accept()
