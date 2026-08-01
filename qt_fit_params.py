"""
qt_fit_params.py — Qt port of ui_fit_params.py's FitParamWindow (M8): the
per-component peak-parameter table (shift/FWHM/eta ranges, shape, amplitude)
used by single-spectrum fitting.

Adds one new capability over the Tk original: an "Auto-find peaks" button
that seeds component centers from fitting_science.find_peak_candidates()
(2nd-derivative peak-finder), instead of the user always typing every guess
by hand — the params_struct schema and save/load-model JSON format are
otherwise unchanged, so existing saved *.json models remain compatible.
"""
from __future__ import annotations

import copy
import json
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QHBoxLayout,
    QInputDialog, QLabel, QMessageBox, QPushButton, QScrollArea, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

# NOTE: the "FWHM" labels are kept for continuity with every existing saved
# model, but the value has always been passed to rampy's HWHM slot — see the
# width-convention note in fitting_science.py.
COLUMNS = [
    "Comp", "Name", "Shift min", "Shift val", "Shift max", "Vary",
    "FWHM min", "FWHM val", "FWHM max", "Vary",
    "Eta min", "Eta val", "Eta max", "Vary",
    "Skew min", "Skew val", "Skew max", "Vary",
    "Shape", "Amp val", "Vary", "FWHM=#",
]

_DEFAULTS: Dict[str, Any] = {
    "shift_min": 900.0, "shift_val": 1000.0, "shift_max": 1100.0, "fit_shift": True,
    "fwhm_min": 1.0, "fwhm_val": 50.0, "fwhm_max": 200.0, "fit_fwhm": True,
    "eta_min": 0.0, "eta_val": 0.5, "eta_max": 1.0, "fit_eta": True,
    "skew_min": -100.0, "skew_val": 1.0, "skew_max": 100.0, "fit_skew": True,
    "shape": "G", "amp_val": 1.0, "fit_amp": True,
}

_NAME_COL = 1   # A FAIRE item 9: optional user name per component ("ν1 PO4", …)
_NUMERIC_COLS = {
    2: "shift_min", 3: "shift_val", 4: "shift_max",
    6: "fwhm_min", 7: "fwhm_val", 8: "fwhm_max",
    10: "eta_min", 11: "eta_val", 12: "eta_max",
    14: "skew_min", 15: "skew_val", 16: "skew_max",
    19: "amp_val",
}
_VARY_COLS = {5: "fit_shift", 9: "fit_fwhm", 13: "fit_eta", 17: "fit_skew", 20: "fit_amp"}
_SHAPE_COL = 18
_LINK_COL = 21  # Origin-style "share this FWHM with peak N" (1-based in UI, blank = no link)
SHAPES = ["G", "GL", "V", "EMG"]


def _default_model_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "param_models")


def list_model_names(model_dir: Optional[str] = None) -> List[str]:
    """Saved parameter models (== M9's "recipes" — same JSON files, same
    directory, no separate format) available for batch fitting."""
    model_dir = model_dir or _default_model_dir()
    if not os.path.isdir(model_dir):
        return []
    return sorted(f[:-5] for f in os.listdir(model_dir) if f.lower().endswith(".json"))


def load_model(name: str, model_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    model_dir = model_dir or _default_model_dir()
    path = os.path.join(model_dir, name + ".json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class FitParamDialog(QDialog):
    """Edit the peak-component table for one spectrum. Calls on_accept(params)
    with the collected params_struct list when the user clicks Accept."""

    def __init__(
        self,
        parent: Optional[QWidget],
        params_struct: Optional[List[Dict[str, Any]]],
        on_accept: Callable[[List[Dict[str, Any]]], None],
        *,
        x: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        model_dir: Optional[str] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Fit parameters")
        self.resize(980, 420)
        self.on_accept = on_accept
        self.model_dir = model_dir or _default_model_dir()
        self._x, self._y = x, y

        params_struct = copy.deepcopy(params_struct) if params_struct else []
        self.rows: List[Dict[str, Any]] = params_struct if params_struct else [dict(_DEFAULTS)]

        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        add_btn = QPushButton("+ Component")
        add_btn.clicked.connect(self._add_component)
        remove_btn = QPushButton("- Component")
        remove_btn.clicked.connect(self._remove_component)
        top.addWidget(add_btn)
        top.addWidget(remove_btn)
        top.addStretch(1)
        if x is not None and y is not None:
            auto_btn = QPushButton("Auto-find peaks")
            auto_btn.clicked.connect(self._auto_find_peaks)
            top.addWidget(auto_btn)
            top.addWidget(QLabel("detection limit (×σ)"))
            self.detection_limit_spin = QDoubleSpinBox()
            self.detection_limit_spin.setRange(0.0, 20.0)
            self.detection_limit_spin.setSingleStep(0.1)
            self.detection_limit_spin.setDecimals(1)
            self.detection_limit_spin.setValue(0.5)
            self.detection_limit_spin.setToolTip(
                "Curvature prominence a candidate needs, in units of the noise σ.\n"
                "Lower finds weaker/broader peaks (and more noise); higher keeps only the strongest; 0 disables the threshold."
            )
            top.addWidget(self.detection_limit_spin)
        save_model_btn = QPushButton("Save as model…")
        save_model_btn.clicked.connect(self._save_model)
        load_model_btn = QPushButton("Load model…")
        load_model_btn.clicked.connect(self._load_model)
        top.addWidget(save_model_btn)
        top.addWidget(load_model_btn)
        layout.addLayout(top)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.table = QTableWidget()
        self.table.setColumnCount(len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        scroll.setWidget(self.table)
        layout.addWidget(scroll, 1)

        accept_btn = QPushButton("Accept")
        accept_btn.setObjectName("Primary")
        accept_btn.clicked.connect(self._on_accept_clicked)
        layout.addWidget(accept_btn)

        self._rebuild_table()

    # ------------------------------------------------------------------
    def _rebuild_table(self) -> None:
        self.table.setRowCount(len(self.rows))
        for i, row in enumerate(self.rows):
            label_item = QTableWidgetItem(f"#{i + 1}")
            label_item.setFlags(label_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(i, 0, label_item)

            self.table.setItem(i, _NAME_COL, QTableWidgetItem(str(row.get("name", ""))))

            for col, key in _NUMERIC_COLS.items():
                item = QTableWidgetItem(f"{row.get(key, _DEFAULTS.get(key, 0.0)):.4g}")
                self.table.setItem(i, col, item)

            for col, key in _VARY_COLS.items():
                cb = QCheckBox()
                cb.setChecked(bool(row.get(key, True)))
                self.table.setCellWidget(i, col, cb)

            shape_combo = QComboBox()
            shape_combo.addItems(SHAPES)
            shape_combo.setCurrentText(row.get("shape", "G"))
            self.table.setCellWidget(i, _SHAPE_COL, shape_combo)

            link = row.get("link_fwhm")
            link_item = QTableWidgetItem("" if link is None else str(int(link) + 1))  # 1-based in the UI
            self.table.setItem(i, _LINK_COL, link_item)

        self.table.resizeColumnsToContents()

    def _sync_rows_from_table(self) -> None:
        for i in range(self.table.rowCount()):
            row = self.rows[i]
            name_item = self.table.item(i, _NAME_COL)
            name_text = (name_item.text() if name_item is not None else "").strip()
            if name_text:
                row["name"] = name_text
            else:
                row.pop("name", None)
            for col, key in _NUMERIC_COLS.items():
                item = self.table.item(i, col)
                try:
                    row[key] = float(item.text())
                except (TypeError, ValueError):
                    row[key] = _DEFAULTS.get(key, 0.0)
            for col, key in _VARY_COLS.items():
                cb = self.table.cellWidget(i, col)
                row[key] = bool(cb.isChecked()) if cb is not None else True
            combo = self.table.cellWidget(i, _SHAPE_COL)
            row["shape"] = combo.currentText() if combo is not None else "G"

            link_item = self.table.item(i, _LINK_COL)
            link_text = (link_item.text() if link_item is not None else "").strip()
            row.pop("link_fwhm", None)
            if link_text:
                try:
                    j = int(float(link_text)) - 1  # UI is 1-based; params_struct is 0-based
                    if j != i and 0 <= j < self.table.rowCount():
                        row["link_fwhm"] = j
                except (TypeError, ValueError):
                    pass

    # ------------------------------------------------------------------
    def _add_component(self) -> None:
        self._sync_rows_from_table()
        self.rows.append(dict(_DEFAULTS))
        self._rebuild_table()

    def _remove_component(self) -> None:
        if len(self.rows) <= 1:
            return
        self._sync_rows_from_table()
        self.rows.pop()
        self._rebuild_table()

    def _auto_find_peaks(self) -> None:
        if self._x is None or self._y is None:
            return
        from core.fitting_science import find_peak_candidates
        centers = find_peak_candidates(
            self._x, self._y, max_peaks=10,
            min_prominence_sigma=float(self.detection_limit_spin.value()),
        )
        if not centers:
            QMessageBox.information(
                self, "Auto-find peaks",
                "No clear peak candidates were found — try lowering the detection limit.",
            )
            return
        span = float(np.nanmax(self._x) - np.nanmin(self._x)) if len(self._x) else 100.0
        half_width = max(span * 0.02, 1.0)
        new_rows = []
        for center in centers:
            row = dict(_DEFAULTS)
            row["shift_val"] = center
            row["shift_min"] = center - 10 * half_width
            row["shift_max"] = center + 10 * half_width
            new_rows.append(row)
        self.rows = new_rows
        self._rebuild_table()

    # ------------------------------------------------------------------
    def _save_model(self) -> None:
        self._sync_rows_from_table()
        name, ok = QInputDialog.getText(self, "Model name", "Name for this parameter model:")
        if not ok or not name.strip():
            return
        os.makedirs(self.model_dir, exist_ok=True)
        path = os.path.join(self.model_dir, name.strip() + ".json")
        if os.path.exists(path):
            resp = QMessageBox.question(self, "Overwrite?", f"'{os.path.basename(path)}' already exists. Overwrite?")
            if resp != QMessageBox.Yes:
                return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.rows, f, indent=2)
            QMessageBox.information(self, "Saved", f"Model '{name}' saved.")
        except OSError as exc:
            QMessageBox.critical(self, "Save error", str(exc))

    def _load_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load model", self.model_dir, "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                params = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            QMessageBox.critical(self, "Load error", str(exc))
            return
        self.rows = params if params else [dict(_DEFAULTS)]
        self._rebuild_table()

    # ------------------------------------------------------------------
    def _on_accept_clicked(self) -> None:
        self._sync_rows_from_table()
        self.on_accept(copy.deepcopy(self.rows))
        self.accept()
