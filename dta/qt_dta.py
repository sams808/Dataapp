"""
qt_dta.py — DTA/DSC/TGA Tg tool, ported from ui_dta_processing.py's TgGuiApp.

All math delegates to dta_science.py (extracted and pytest-covered in M2) —
this file is purely the Qt presentation layer. Reuses the shared PlotWidget
(qt_widgets.py) instead of embedding matplotlib independently, and the shared
PerItemSettingsStore (qt_settings_store.py) for per-record settings instead
of re-solving persistence ad hoc.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
    QRadioButton, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from dta.dta_science import (
    BaselineParams, TgDoubleTangentResult, TgParallelTangentResult,
    compute_derivative, compute_tg_derivative, compute_tg_double_tangent,
    compute_tg_parallel_improved, moving_average_10, resolve_baseline_params,
    _line_y,
)
from core.io_universal import parse_ta_sdt_txt
from core.qt_widgets import PlotWidget
from core.qt_widgets import to_float as _to_float


def _unit_from_col(col: str) -> str:
    cl = col.lower()
    # Kelvin markers are checked FIRST and returned immediately: the older
    # ordering checked "temp"/"temperature" first, which matches a column
    # literally named "Temperature (K)" too and always misclassified it as
    # °C before the Kelvin branch ever ran. Real DTA exports are almost
    # always °C (TA Instruments' own convention, see io_universal.py's
    # canonical "T_C" key), so °C stays the fallback for an ambiguous
    # generic "Temp"/"Temperature" name with no explicit unit marker.
    if "kelvin" in cl or "(k)" in cl or cl.strip().endswith(" k"):
        return "K"
    if "°c" in cl or "temperature" in cl or "temp" in cl or "c)" in cl:
        return "°C"
    return ""


def _shift_temperature(value, old_unit: str, new_unit: str):
    """Convert a scalar float or ndarray from old_unit to new_unit ('°C'
    or 'K'). A no-op when the units already match or either is blank/
    unrecognized. Kelvin and Celsius differ only by a constant +273.15
    offset, so this is the one place that offset is applied -- everything
    else (axis data, window/baseline fields, exported numbers) routes
    through this so a switch can never apply it inconsistently in two
    places. Deliberately NOT used for differences/spreads (e.g. the Tg
    methods' agreement threshold): a difference in °C already equals the
    same difference in K, since the constant offset cancels out."""
    if not old_unit or not new_unit or old_unit == new_unit:
        return value
    if old_unit == "°C" and new_unit == "K":
        return value + 273.15
    if old_unit == "K" and new_unit == "°C":
        return value - 273.15
    return value


class DtaWorkspace(QWidget):
    def __init__(self, parent: Optional[QWidget] = None, records: Optional[List[Dict[str, Any]]] = None):
        super().__init__(parent)
        self.records: List[Dict[str, Any]] = records or []
        self._record_lookup: Dict[str, Dict[str, Any]] = {}
        for i, rec in enumerate(self.records):
            key = rec.get("title") or f"Record {i + 1}"
            base, suffix = key, 2
            while key in self._record_lookup:
                key = f"{base} ({suffix})"
                suffix += 1
            self._record_lookup[key] = rec

        self.path: Optional[Path] = None
        self.header: Dict[str, str] = {}
        self.colnames: List[str] = []
        self.df: Optional[pd.DataFrame] = None
        self._x: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None

        self.res_double: Optional[TgDoubleTangentResult] = None
        self.res_parallel: Optional[TgParallelTangentResult] = None
        self.tg_deriv: Optional[float] = None
        self._prev_temp_unit = "°C"  # tracked so the toggle handler always
        # knows the OLD unit to convert range fields from, regardless of
        # which of the two buttons Qt happens to signal.

        self._build_ui()
        if self.records:
            self.record_combo.setCurrentIndex(0)
            self._load_selected_record()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        left = QWidget()
        left.setObjectName("Card")
        left.setMaximumWidth(440)
        left_layout = QVBoxLayout(left)

        self.tabs = QTabWidget()
        left_layout.addWidget(self.tabs)

        self.tabs.addTab(self._build_tg_tab(), "Tg")
        self.tabs.addTab(self._build_calc_tab(), "Calculs")
        self.tabs.addTab(self._build_batch_tab(), "Export / Batch")

        self.status_label = QLabel("")
        self.status_label.setObjectName("SectionNote")
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)

        splitter.addWidget(left)

        right = QWidget()
        right.setObjectName("Card")
        right_layout = QVBoxLayout(right)
        self.plot = PlotWidget(figsize=(7.8, 5.6))
        self.ax2 = None
        right_layout.addWidget(self.plot)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

        self._set_status(
            "Parallel (improved): each baseline is either a range (linear fit) or a point "
            "(a line parallel to the other baseline passing through that point)."
        )

    def _build_tg_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        data_box = QGroupBox("Data")
        data_form = QVBoxLayout(data_box)
        self.file_label = QLabel("No file loaded")
        self.file_label.setWordWrap(True)
        data_form.addWidget(self.file_label)

        rec_row = QHBoxLayout()
        rec_row.addWidget(QLabel("Imported"))
        self.record_combo = QComboBox()
        self.record_combo.addItems(list(self._record_lookup.keys()))
        self.record_combo.currentIndexChanged.connect(self._load_selected_record)
        rec_row.addWidget(self.record_combo, 1)
        data_form.addLayout(rec_row)

        btn_row = QHBoxLayout()
        open_btn = QPushButton("Open…")
        open_btn.clicked.connect(self._on_open)
        reload_btn = QPushButton("Reload")
        reload_btn.clicked.connect(self._on_reload)
        btn_row.addWidget(open_btn)
        btn_row.addWidget(reload_btn)
        data_form.addLayout(btn_row)
        layout.addWidget(data_box)

        plot_box = QGroupBox("Plot")
        plot_form = QFormLayout(plot_box)
        self.x_combo = QComboBox()
        self.y_combo = QComboBox()
        self.dy_combo = QComboBox()
        self.x_combo.currentTextChanged.connect(self._refresh_plot)
        self.y_combo.currentTextChanged.connect(self._on_y_change)
        self.dy_combo.currentTextChanged.connect(self._refresh_plot)
        plot_form.addRow("X", self.x_combo)

        unit_row = QHBoxLayout()
        self.temp_c_btn = QPushButton("°C")
        self.temp_k_btn = QPushButton("K")
        switch_tip = (
            "Switch the temperature axis, Tg results, and window/baseline "
            "ranges between °C and K (applies workspace-wide, including the "
            "Calculs tab). Existing range values and any computed Tg are "
            "converted in place, not reinterpreted. Disabled when the "
            "current X column isn't a recognized temperature axis."
        )
        for btn, obj_name in ((self.temp_c_btn, "UnitSwitchLeft"), (self.temp_k_btn, "UnitSwitchRight")):
            btn.setCheckable(True)
            btn.setObjectName(obj_name)
            btn.setToolTip(switch_tip)
            btn.setEnabled(False)  # enabled once a recognized temperature column is loaded
        self.temp_c_btn.setChecked(True)
        self.temp_unit_group = QButtonGroup(self)
        self.temp_unit_group.setExclusive(True)
        self.temp_unit_group.addButton(self.temp_c_btn, 0)
        self.temp_unit_group.addButton(self.temp_k_btn, 1)
        self.temp_c_btn.toggled.connect(self._on_temp_unit_toggled)
        self.temp_k_btn.toggled.connect(self._on_temp_unit_toggled)
        unit_row.addWidget(self.temp_c_btn)
        unit_row.addWidget(self.temp_k_btn)
        unit_row.addStretch(1)
        plot_form.addRow("Temp. unit", unit_row)

        plot_form.addRow("Y", self.y_combo)
        plot_form.addRow("dY source", self.dy_combo)

        opt_row = QHBoxLayout()
        self.invert_y_check = QCheckBox("Invert Y")
        self.show_deriv_check = QCheckBox("dY overlay")
        self.explicit_check = QCheckBox("Explicit")
        self.explicit_check.setChecked(True)
        for cb in (self.invert_y_check, self.show_deriv_check, self.explicit_check):
            cb.stateChanged.connect(self._refresh_plot)
            opt_row.addWidget(cb)
        plot_form.addRow(opt_row)

        smooth_row = QHBoxLayout()
        smooth_row.addWidget(QLabel("Smooth dY (MA10)"))
        self.smooth_group = QButtonGroup(self)
        smooth_off = QRadioButton("Off")
        smooth_on = QRadioButton("On")
        smooth_off.setChecked(True)
        self.smooth_group.addButton(smooth_off, 0)
        self.smooth_group.addButton(smooth_on, 1)
        smooth_off.toggled.connect(self._refresh_plot)
        smooth_row.addWidget(smooth_off)
        smooth_row.addWidget(smooth_on)
        plot_form.addRow(smooth_row)
        layout.addWidget(plot_box)

        tg_box = QGroupBox("Tg")
        tg_layout = QVBoxLayout(tg_box)

        method_row = QHBoxLayout()
        self.method_group = QButtonGroup(self)
        self.method_double = QRadioButton("Double")
        self.method_parallel = QRadioButton("Parallel (improved)")
        self.method_deriv = QRadioButton("|dY| max")
        self.method_parallel.setChecked(True)
        for i, btn in enumerate((self.method_double, self.method_parallel, self.method_deriv)):
            self.method_group.addButton(btn, i)
            btn.toggled.connect(self._refresh_plot)
            method_row.addWidget(btn)
        tg_layout.addLayout(method_row)

        window_row = QHBoxLayout()
        window_row.addWidget(QLabel("Window"))
        self.xmin_edit = QLineEdit("350")
        self.xmax_edit = QLineEdit("700")
        window_row.addWidget(self.xmin_edit)
        window_row.addWidget(QLabel("→"))
        window_row.addWidget(self.xmax_edit)
        tg_layout.addLayout(window_row)

        ranges_box = QGroupBox("Manual ranges (shared)")
        ranges_layout = QVBoxLayout(ranges_box)

        self.low_min_edit, self.low_max_edit, self.low_point_edit, self.low_use_point_check = self._baseline_row(
            ranges_layout, "LOW"
        )
        slope_row = QHBoxLayout()
        slope_row.addWidget(QLabel("SLOPE:"))
        self.slope_min_edit = QLineEdit()
        self.slope_max_edit = QLineEdit()
        slope_row.addWidget(self.slope_min_edit)
        slope_row.addWidget(QLabel("→"))
        slope_row.addWidget(self.slope_max_edit)
        slope_row.addWidget(QLabel("(Double Tangent only)"))
        ranges_layout.addLayout(slope_row)

        self.high_min_edit, self.high_max_edit, self.high_point_edit, self.high_use_point_check = self._baseline_row(
            ranges_layout, "HIGH"
        )
        tg_layout.addWidget(ranges_box)

        action_row = QHBoxLayout()
        self.manual_compute_check = QCheckBox("Manual")
        action_row.addWidget(self.manual_compute_check)
        compute_btn = QPushButton("Compute")
        compute_btn.setObjectName("Primary")
        compute_btn.clicked.connect(self._compute)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_results)
        redraw_btn = QPushButton("Redraw")
        redraw_btn.clicked.connect(self._refresh_plot)
        action_row.addWidget(compute_btn)
        action_row.addWidget(clear_btn)
        action_row.addWidget(redraw_btn)
        tg_layout.addLayout(action_row)

        self.result_label = QLabel("Tg: —")
        self.result_label.setWordWrap(True)
        tg_layout.addWidget(self.result_label)
        layout.addWidget(tg_box)
        layout.addStretch(1)
        return tab

    def _baseline_row(self, parent_layout: QVBoxLayout, label: str):
        row = QHBoxLayout()
        row.addWidget(QLabel(f"{label}:"))
        vmin = QLineEdit()
        vmax = QLineEdit()
        row.addWidget(vmin)
        row.addWidget(QLabel("→"))
        row.addWidget(vmax)
        use_point = QCheckBox("Use point")
        row.addWidget(use_point)
        row.addWidget(QLabel("Point:"))
        vpt = QLineEdit()
        row.addWidget(vpt)
        parent_layout.addLayout(row)

        def _toggle(checked: bool):
            vmin.setEnabled(not checked)
            vmax.setEnabled(not checked)
            vpt.setEnabled(checked)
            self._refresh_plot()

        use_point.toggled.connect(_toggle)
        _toggle(False)
        return vmin, vmax, vpt, use_point

    def _build_calc_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        box = QGroupBox("Calculs")
        form = QFormLayout(box)

        self.calc_y_combo = QComboBox()
        form.addRow("Y", self.calc_y_combo)

        deriv_row = QHBoxLayout()
        self.calc_use_deriv_check = QCheckBox("Use derivative")
        self.calc_use_deriv_check.toggled.connect(self._calc_clear)
        deriv_row.addWidget(self.calc_use_deriv_check)
        deriv_row.addWidget(QLabel("d/dX:"))
        self.calc_deriv_x_combo = QComboBox()
        deriv_row.addWidget(self.calc_deriv_x_combo, 1)
        form.addRow(deriv_row)

        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("Xmin"))
        self.calc_xmin_edit = QLineEdit()
        range_row.addWidget(self.calc_xmin_edit)
        range_row.addWidget(QLabel("Xmax"))
        self.calc_xmax_edit = QLineEdit()
        range_row.addWidget(self.calc_xmax_edit)
        form.addRow(range_row)

        btn_row = QHBoxLayout()
        integrate_btn = QPushButton("Integrate")
        integrate_btn.clicked.connect(self._calc_integrate)
        max_btn = QPushButton("Find Max")
        max_btn.clicked.connect(self._calc_find_max)
        min_btn = QPushButton("Find Min")
        min_btn.clicked.connect(self._calc_find_min)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._calc_clear)
        for b in (integrate_btn, max_btn, min_btn, clear_btn):
            btn_row.addWidget(b)
        form.addRow(btn_row)

        self.calc_result_label = QLabel("")
        self.calc_result_label.setWordWrap(True)
        form.addRow(self.calc_result_label)
        layout.addWidget(box)
        layout.addStretch(1)
        return tab

    def _build_batch_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        export_box = QGroupBox("Export current")
        export_layout = QVBoxLayout(export_box)
        export_csv_btn = QPushButton("Export result CSV…")
        export_csv_btn.setObjectName("Primary")
        export_csv_btn.clicked.connect(self._export_current_csv)
        save_png_btn = QPushButton("Save figure PNG…")
        save_png_btn.clicked.connect(self._save_figure_png)
        export_layout.addWidget(export_csv_btn)
        export_layout.addWidget(save_png_btn)
        layout.addWidget(export_box)

        batch_box = QGroupBox("Batch")
        batch_layout = QVBoxLayout(batch_box)
        note = QLabel("Batch uses CURRENT settings (window, manual ranges, smoothing).")
        note.setWordWrap(True)
        batch_layout.addWidget(note)

        snap_row = QHBoxLayout()
        self.batch_snapshot_group = QButtonGroup(self)
        self.batch_snapshot_no = QRadioButton("No snapshots")
        self.batch_snapshot_png = QRadioButton("PNG snapshots")
        self.batch_snapshot_no.setChecked(True)
        self.batch_snapshot_group.addButton(self.batch_snapshot_no, 0)
        self.batch_snapshot_group.addButton(self.batch_snapshot_png, 1)
        snap_row.addWidget(self.batch_snapshot_no)
        snap_row.addWidget(self.batch_snapshot_png)
        batch_layout.addLayout(snap_row)

        run_btn = QPushButton("Run batch…")
        run_btn.setObjectName("Primary")
        run_btn.clicked.connect(self._batch_run)
        batch_layout.addWidget(run_btn)
        layout.addWidget(batch_box)
        layout.addStretch(1)
        return tab

    # ------------------------------------------------------------------
    # Status / data loading
    # ------------------------------------------------------------------

    def _set_status(self, msg: str) -> None:
        self.status_label.setText(msg)

    def _load_selected_record(self) -> None:
        title = self.record_combo.currentText().strip()
        if not title or title not in self._record_lookup:
            return
        rec = self._record_lookup[title]
        df = rec.get("df")
        meta = rec.get("meta") or {}
        path = rec.get("path")

        if df is None and path:
            try:
                hdr, cols, df = parse_ta_sdt_txt(Path(path))
                meta = {**meta, "raw_header": hdr, "signals": cols}
            except Exception as exc:
                QMessageBox.critical(self, "Load error", f"Could not reload {path}:\n{exc}")
                return
        if df is None:
            QMessageBox.critical(self, "Load error", "Selected record has no data attached.")
            return

        display_title = rec.get("title") or (Path(path).stem if path else "DTA data")
        self._apply_dataframe(df, meta=meta, path=path, display_title=display_title)

    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open DTA file", "", "Text files (*.txt *.dat *.asc);;All files (*.*)")
        if path:
            self.load_file(Path(path))

    def _on_reload(self) -> None:
        if self.path:
            self.load_file(self.path)

    def load_file(self, path: Path) -> None:
        self.path = Path(path)
        self.header, self.colnames, self.df = parse_ta_sdt_txt(self.path)
        self._apply_dataframe(self.df, meta={"raw_header": self.header}, path=str(self.path), display_title=self.path.name)

    def _suggest_default_columns(self, meta: Optional[Dict[str, Any]]) -> Tuple[str, str]:
        canonical = (meta or {}).get("canonical_map") or {}

        def pick_from_canon(keys):
            for key in keys:
                col = canonical.get(key)
                if col in self.colnames:
                    return col
            return None

        def pick_by_keyword(keywords):
            for kw in keywords:
                for c in self.colnames:
                    if kw.lower() in str(c).lower():
                        return c
            return None

        xdef = pick_from_canon(["T_C", "time_min", "X"]) or pick_by_keyword(["temperature", "temp"]) or (self.colnames[0] if self.colnames else "")
        ydef = pick_from_canon(["DSC_mW_mg", "HF_mW", "TG_pct", "mass_mg", "Y"]) or pick_by_keyword(
            ["heat flow", "dsc", "dta", "heat", "signal", "tg"]
        ) or (self.colnames[1] if len(self.colnames) > 1 else (self.colnames[0] if self.colnames else ""))
        return xdef or "", ydef or ""

    def _apply_dataframe(self, df: pd.DataFrame, *, meta: Optional[Dict[str, Any]], path: Optional[str], display_title: str) -> None:
        self.path = Path(path) if path else None
        self.header = (meta or {}).get("raw_header", {}) or {}
        self.colnames = list(df.columns)
        self.df = df.copy()

        for combo in (self.x_combo, self.y_combo, self.calc_y_combo, self.calc_deriv_x_combo):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(self.colnames)
            combo.blockSignals(False)
        self.dy_combo.blockSignals(True)
        self.dy_combo.clear()
        self.dy_combo.addItems(["(same)"] + self.colnames)
        self.dy_combo.setCurrentText("(same)")
        self.dy_combo.blockSignals(False)

        xdef, ydef = self._suggest_default_columns(meta)
        self.x_combo.setCurrentText(xdef)
        self.y_combo.setCurrentText(ydef)
        self.calc_y_combo.setCurrentText(ydef)
        self.calc_deriv_x_combo.setCurrentText(xdef)

        self.xmin_edit.setText(self._format_temp_default(350.0))
        self.xmax_edit.setText(self._format_temp_default(700.0))
        self.calc_xmin_edit.setText(self._format_temp_default(350.0))
        self.calc_xmax_edit.setText(self._format_temp_default(700.0))

        sample = self.header.get("Sample", self.path.stem if self.path else display_title)
        suffix = self.path.name if self.path else display_title
        self.file_label.setText(f"{suffix}\nSample: {sample}")

        self._clear_results()
        self._refresh_plot()

    def set_records(self, records: List[Dict[str, Any]]) -> None:
        """Refresh the "Imported" dropdown from the library's current DTA-kind
        spectra — called by the shell whenever the DTA workspace becomes
        active, so newly imported files show up without rebuilding the whole
        widget."""
        self.records = records or []
        self._record_lookup = {}
        for i, rec in enumerate(self.records):
            key = rec.get("title") or f"Record {i + 1}"
            base, suffix = key, 2
            while key in self._record_lookup:
                key = f"{base} ({suffix})"
                suffix += 1
            self._record_lookup[key] = rec

        current = self.record_combo.currentText()
        self.record_combo.blockSignals(True)
        self.record_combo.clear()
        self.record_combo.addItems(list(self._record_lookup.keys()))
        self.record_combo.blockSignals(False)
        if current in self._record_lookup:
            self.record_combo.setCurrentText(current)
        elif self._record_lookup and self.df is None:
            self.record_combo.setCurrentIndex(0)
            self._load_selected_record()

    def _on_y_change(self) -> None:
        if not self.calc_y_combo.currentText():
            self.calc_y_combo.setCurrentText(self.y_combo.currentText())
        self._refresh_plot()

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def _smooth_derivative_enabled(self) -> bool:
        return self.smooth_group.checkedId() == 1

    # ---- Temperature unit switch --------------------------------------

    def _temp_unit(self) -> str:
        return "K" if self.temp_unit_group.checkedId() == 1 else "°C"

    def _convert_x_to_display(self, x: np.ndarray, x_col: str) -> np.ndarray:
        """Convert x (as read straight from the dataframe, in whatever unit
        the column's own header claims) to the currently selected display
        unit. A no-op for any column _unit_from_col doesn't recognize as
        temperature, so switching °C/K never touches a Time or other
        non-temperature axis even if it happens to be the current X."""
        native = _unit_from_col(x_col)
        if not native:
            return x
        return _shift_temperature(x, native, self._temp_unit())

    def _result_unit_label(self, x_col: str) -> str:
        """Unit suffix for displayed/exported numbers: the switch's current
        setting for a recognized temperature column, otherwise whatever
        _unit_from_col already returns (typically '' for e.g. Time)."""
        native = _unit_from_col(x_col)
        return self._temp_unit() if native else native

    def _display_x_label(self, x_col: str) -> str:
        if not _unit_from_col(x_col):
            return x_col
        return f"Temperature ({self._temp_unit()})"

    def _format_temp_default(self, celsius_value: float) -> str:
        """Tg-window/Calculs-range defaults are authored in °C (the
        canonical DTA unit); convert to whatever unit is currently active
        so re-loading a file while working in Kelvin doesn't reset the
        window to a physically wrong value."""
        v = _shift_temperature(celsius_value, "°C", self._temp_unit())
        return f"{v:.6g}"

    def _update_temp_switch_enabled(self, x_col: str) -> None:
        enabled = bool(_unit_from_col(x_col))
        self.temp_c_btn.setEnabled(enabled)
        self.temp_k_btn.setEnabled(enabled)

    def _convert_range_fields(self, old_unit: str, new_unit: str) -> None:
        """Keep every typed range/point field physically meaningful across
        a unit switch, converting in place rather than leaving the same
        typed number to mean something different. Only touches a group of
        fields when the X column that group actually windows is currently
        recognized as temperature -- so switching units while X is, say,
        Time never reinterprets a time window as degrees."""
        def _shift_field(edit: QLineEdit) -> None:
            v = _to_float(edit.text())
            if v is not None:
                edit.setText(f"{_shift_temperature(v, old_unit, new_unit):.6g}")

        if _unit_from_col(self.x_combo.currentText()):
            for edit in (
                self.xmin_edit, self.xmax_edit,
                self.low_min_edit, self.low_max_edit, self.low_point_edit,
                self.slope_min_edit, self.slope_max_edit,
                self.high_min_edit, self.high_max_edit, self.high_point_edit,
            ):
                _shift_field(edit)

        calc_x_col = self.calc_deriv_x_combo.currentText() if self.calc_use_deriv_check.isChecked() else self.x_combo.currentText()
        if _unit_from_col(calc_x_col):
            _shift_field(self.calc_xmin_edit)
            _shift_field(self.calc_xmax_edit)

    def _on_temp_unit_toggled(self, checked: bool) -> None:
        if not checked:
            return  # exclusive group fires toggled(False) on the outgoing button too; act once
        new_unit = self._temp_unit()
        if new_unit == self._prev_temp_unit:
            return
        old_unit = self._prev_temp_unit
        self._prev_temp_unit = new_unit
        self._convert_range_fields(old_unit, new_unit)
        if self.df is None:
            return
        if self.res_double is not None or self.res_parallel is not None or self.tg_deriv is not None:
            self._compute()
        else:
            self._refresh_plot()

    def _get_window(self) -> Tuple[float, float]:
        a = _to_float(self.xmin_edit.text())
        b = _to_float(self.xmax_edit.text())
        if a is None or b is None:
            raise ValueError("Provide a numeric Tg window.")
        return (min(a, b), max(a, b))

    def _get_xy(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, str, str]:
        if self.df is None:
            raise ValueError("No data loaded.")
        x_col = self.x_combo.currentText()
        y_col = self.y_combo.currentText()
        dy_sel = self.dy_combo.currentText()
        if not x_col or not y_col:
            raise ValueError("Pick X and Y columns first.")

        x = pd.to_numeric(self.df[x_col], errors="coerce").to_numpy(dtype=float)
        x = self._convert_x_to_display(x, x_col)
        y = pd.to_numeric(self.df[y_col], errors="coerce").to_numpy(dtype=float)
        if dy_sel and dy_sel != "(same)" and dy_sel in self.df.columns:
            y_dy = pd.to_numeric(self.df[dy_sel], errors="coerce").to_numpy(dtype=float)
            dy_col = dy_sel
        else:
            y_dy = y.copy()
            dy_col = "(same)"

        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(y_dy)
        x, y, y_dy = x[m], y[m], y_dy[m]
        order = np.argsort(x)
        x, y, y_dy = x[order], y[order], y_dy[order]
        if self.invert_y_check.isChecked():
            y, y_dy = -y, -y_dy
        return x, y, y_dy, x_col, y_col, dy_col

    def _raw_baseline_inputs(self) -> Dict[str, Any]:
        return dict(
            low_use_point=self.low_use_point_check.isChecked(),
            low_point_x=_to_float(self.low_point_edit.text()),
            low_min=_to_float(self.low_min_edit.text()),
            low_max=_to_float(self.low_max_edit.text()),
            high_use_point=self.high_use_point_check.isChecked(),
            high_point_x=_to_float(self.high_point_edit.text()),
            high_min=_to_float(self.high_min_edit.text()),
            high_max=_to_float(self.high_max_edit.text()),
            slope_min=_to_float(self.slope_min_edit.text()),
            slope_max=_to_float(self.slope_max_edit.text()),
        )

    def _resolved_baseline_params(self) -> BaselineParams:
        return resolve_baseline_params(manual_enabled=self.manual_compute_check.isChecked(), **self._raw_baseline_inputs())

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------

    def _compute(self) -> None:
        try:
            x, y, y_dy, x_col, y_col, dy_col = self._get_xy()
            xmin, xmax = self._get_window()
            smooth_d = self._smooth_derivative_enabled()
            bp = self._resolved_baseline_params()

            self.tg_deriv = compute_tg_derivative(x, y_dy, xmin, xmax, smooth_derivative=smooth_d, restrict_range=None)

            try:
                self.res_double = compute_tg_double_tangent(
                    x, y, xmin, xmax, threshold=0.20, guard_frac=0.0,
                    smooth_derivative=smooth_d, manual_low=bp.low_range, manual_slope=bp.manual_slope,
                )
            except Exception:
                self.res_double = None

            try:
                self.res_parallel = compute_tg_parallel_improved(
                    x, y, xmin, xmax, smooth_derivative=smooth_d,
                    manual_low_range=bp.low_range, manual_high_range=bp.high_range,
                    manual_low_point=bp.low_point, manual_high_point=bp.high_point,
                )
            except Exception:
                self.res_parallel = None

            unit = self._result_unit_label(x_col)
            td = self.res_double.tg if self.res_double is not None else float("nan")
            tp = self.res_parallel.tg if self.res_parallel is not None else float("nan")
            tx = self.tg_deriv if self.tg_deriv is not None else float("nan")

            lines = [
                f"Y: {y_col}",
                f"Double: {td:.2f} {unit}" if np.isfinite(td) else "Double: —",
                f"Parallel: {tp:.2f} {unit}" if np.isfinite(tp) else "Parallel: —",
                f"|dY| max: {tx:.2f} {unit}" if np.isfinite(tx) else "|dY| max: —",
            ]
            if dy_col != "(same)":
                lines.append(f"dY src: {dy_col}")
            if self.res_parallel is not None:
                lp = self.res_parallel.low_used
                lines.append(f"LOW: point x={lp[0]:.6g}" if self.res_parallel.low_mode == "point" else f"LOW: {lp[0]:.6g}..{lp[1]:.6g}")
                hp = self.res_parallel.high_used
                lines.append(f"HIGH: point x={hp[0]:.6g}" if self.res_parallel.high_mode == "point" else f"HIGH: {hp[0]:.6g}..{hp[1]:.6g}")

            from dta.dta_science import tg_agreement
            agreement = tg_agreement({"Double": td, "Parallel": tp, "|dY| max": tx}, threshold=5.0)
            if agreement["agree"] is True:
                lines.append(f"✓ Methods agree (spread {agreement['spread']:.2f} {unit})")
            elif agreement["agree"] is False:
                lo, hi = agreement["extremes"]
                lines.append(f"⚠ Methods disagree: spread {agreement['spread']:.2f} {unit} ({lo} vs {hi}) — inspect the baselines")

            self.result_label.setText("\n".join(lines))
            self._refresh_plot()
        except Exception as exc:
            QMessageBox.critical(self, "Tg computation failed", str(exc))

    def _clear_results(self) -> None:
        self.res_double = None
        self.res_parallel = None
        self.tg_deriv = None
        self.result_label.setText("Tg: —")
        self._set_status("Cleared results.")
        if self.df is not None:
            self._refresh_plot()

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _refresh_plot(self) -> None:
        if self.df is None:
            return
        try:
            x, y, y_dy, x_col, y_col, dy_col = self._get_xy()
        except Exception:
            return
        self._x, self._y = x, y
        self._update_temp_switch_enabled(x_col)

        if self.ax2 is not None:
            try:
                self.ax2.remove()
            except Exception:
                pass
            self.ax2 = None

        ax = self.plot.ax
        ax.clear()
        ax.plot(x, y, linewidth=2.0)
        ax.set_xlabel(self._display_x_label(x_col))
        ax.set_ylabel(y_col + (" (inv)" if self.invert_y_check.isChecked() else ""))
        sample = self.header.get("Sample", self.path.stem if self.path else "")
        ax.set_title(sample)
        ax.grid(True, alpha=0.25)

        try:
            xmin, xmax = self._get_window()
            ax.axvspan(xmin, xmax, alpha=0.04)
        except Exception:
            pass

        if self.show_deriv_check.isChecked():
            try:
                xmin, xmax = self._get_window()
                w = (x >= xmin) & (x <= xmax)
                if w.sum() >= 20:
                    dy = compute_derivative(y_dy[w], x[w])
                    if self._smooth_derivative_enabled():
                        dy = moving_average_10(dy)
                    self.ax2 = ax.twinx()
                    self.ax2.plot(x[w], dy, linewidth=1.2, alpha=0.85)
                    self.ax2.set_ylabel(f"dY/d{x_col}")
                    self.ax2.grid(False)
            except Exception:
                pass

        self._draw_overlays(dy_col=dy_col)
        self.plot.figure.tight_layout()
        self.plot.canvas.draw_idle()

    def _draw_overlays(self, dy_col: str) -> None:
        if self._x is None or self._y is None:
            return
        x, y = self._x, self._y
        ax = self.plot.ax
        explicit = self.explicit_check.isChecked()
        unit = self._result_unit_label(self.x_combo.currentText())
        method = ("double", "parallel", "deriv")[self.method_group.checkedId()] if self.method_group.checkedId() >= 0 else "parallel"

        td = self.res_double.tg if self.res_double is not None else float("nan")
        tp = self.res_parallel.tg if self.res_parallel is not None else float("nan")
        tx = self.tg_deriv if self.tg_deriv is not None else float("nan")
        info_lines = [
            f"Double: {td:.2f}{unit}" if np.isfinite(td) else "Double: —",
            f"Parallel: {tp:.2f}{unit}" if np.isfinite(tp) else "Parallel: —",
            f"|dY| max: {tx:.2f}{unit}" if np.isfinite(tx) else "|dY| max: —",
        ]
        if dy_col != "(same)":
            info_lines.append(f"dY src: {dy_col}")

        def draw_tg_marker(tg: float):
            if not np.isfinite(tg):
                return
            ax.axvline(tg, linestyle="--", linewidth=2.0)
            yy = float(np.interp(tg, x, y))
            ax.plot([tg], [yy], marker="o")

        try:
            xmin, xmax = self._get_window()
        except Exception:
            xmin, xmax = float(x[0]), float(x[-1])
        xx = np.array([xmin, xmax], dtype=float)

        if method == "double" and self.res_double is not None:
            r = self.res_double
            if explicit:
                ax.plot(xx, _line_y(r.m_low, r.b_low, xx), linestyle="--", linewidth=1.3)
                ax.plot(xx, _line_y(r.m_slope, r.b_slope, xx), linestyle="-.", linewidth=1.3)
                ax.axvline(r.x_left, alpha=0.15, linestyle=":")
                ax.axvline(r.x_right, alpha=0.15, linestyle=":")
            draw_tg_marker(r.tg)
        elif method == "parallel" and self.res_parallel is not None:
            r = self.res_parallel
            if explicit:
                ax.plot(xx, _line_y(r.m_par, r.b_low, xx), linestyle="--", linewidth=1.2)
                ax.plot(xx, _line_y(r.m_par, r.b_high, xx), linestyle="--", linewidth=1.2)
                ax.plot(xx, _line_y(r.m_par, r.b_mid, xx), linestyle="-", linewidth=1.3)
                ax.axvline(r.x_left, alpha=0.15, linestyle=":")
                ax.axvline(r.x_right, alpha=0.15, linestyle=":")
            draw_tg_marker(r.tg)
        elif method == "deriv" and np.isfinite(tx):
            draw_tg_marker(tx)

        ax.text(
            0.02, 0.98, "\n".join(info_lines), transform=ax.transAxes,
            va="top", ha="left", fontsize=10, bbox=dict(boxstyle="round,pad=0.35", alpha=0.15),
        )

    # ------------------------------------------------------------------
    # Calculs
    # ------------------------------------------------------------------

    def _get_calc_range(self) -> Tuple[float, float]:
        a = _to_float(self.calc_xmin_edit.text())
        b = _to_float(self.calc_xmax_edit.text())
        if a is None or b is None:
            raise ValueError("Provide numeric Xmin / Xmax for Calculs.")
        return (min(a, b), max(a, b))

    def _get_calc_y(self) -> Tuple[np.ndarray, np.ndarray, str]:
        if self.df is None:
            raise ValueError("No data loaded.")
        y_col = self.calc_y_combo.currentText() or self.y_combo.currentText()
        if not y_col:
            raise ValueError("Select a Y for Calculs.")
        use_deriv = self.calc_use_deriv_check.isChecked()
        x_col = self.calc_deriv_x_combo.currentText() if use_deriv else self.x_combo.currentText()
        x_col = x_col or self.x_combo.currentText()

        x = pd.to_numeric(self.df[x_col], errors="coerce").to_numpy(dtype=float)
        x = self._convert_x_to_display(x, x_col)
        y = pd.to_numeric(self.df[y_col], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        order = np.argsort(x)
        x, y = x[order], y[order]
        if self.invert_y_check.isChecked():
            y = -y

        if use_deriv:
            if x.size < 3:
                raise ValueError("Not enough points for derivative.")
            if np.any(np.diff(x) == 0):
                _, idx = np.unique(x, return_index=True)
                idx = np.sort(idx)
                x, y = x[idx], y[idx]
            return x, np.gradient(y, x), f"d({y_col})/d({x_col})"
        return x, y, y_col

    def _calc_axis(self):
        """Secondary axis when Calculs Y differs from Tg Y, or derivative is used."""
        use_deriv = self.calc_use_deriv_check.isChecked()
        tg_y = self.y_combo.currentText()
        calc_y = self.calc_y_combo.currentText()
        is_alt = bool(calc_y and tg_y and calc_y != tg_y)
        if not use_deriv and not is_alt:
            return self.plot.ax
        if self.ax2 is None:
            self.ax2 = self.plot.ax.twinx()
        if is_alt:
            self.ax2.spines["right"].set_color("red")
            self.ax2.tick_params(axis="y", colors="red")
        self.ax2.set_ylabel("Calculs" if is_alt else "Derivative")
        return self.ax2

    def _calc_clear(self) -> None:
        self._refresh_plot()
        self.calc_result_label.setText("")
        self._set_status("Calculs cleared.")

    def _calc_integrate(self) -> None:
        try:
            x, y, y_col = self._get_calc_y()
            xmin, xmax = self._get_calc_range()
            w = (x >= xmin) & (x <= xmax)
            if w.sum() < 2:
                raise ValueError("Range too small for integration.")
            area = float(np.trapz(y[w], x[w]))
            self.calc_result_label.setText(f"Integrate ({y_col}) from {xmin:.4g} to {xmax:.4g} = {area:.6g}")
            ax = self._calc_axis()
            self.plot.ax.axvspan(xmin, xmax, alpha=0.08)
            ax.fill_between(x[w], y[w], 0.0, alpha=0.12)
            self.plot.ax.text(0.01, 0.01, f"Intg = {area:.6g}", transform=self.plot.ax.transAxes, va="bottom", ha="left")
            self.plot.canvas.draw_idle()
        except Exception as exc:
            QMessageBox.critical(self, "Integrate failed", str(exc))

    def _calc_find_extremum(self, mode: str) -> None:
        try:
            x, y, y_col = self._get_calc_y()
            xmin, xmax = self._get_calc_range()
            w = (x >= xmin) & (x <= xmax)
            if w.sum() < 2:
                raise ValueError("Range too small.")
            xw, yw = x[w], y[w]
            j = int(np.nanargmax(yw)) if mode == "max" else int(np.nanargmin(yw))
            xv, yv = float(xw[j]), float(yw[j])
            label = "Max" if mode == "max" else "Min"
            self.calc_result_label.setText(f"{label} ({y_col}) in [{xmin:.4g}, {xmax:.4g}] = {yv:.6g} at x={xv:.6g}")
            ax = self._calc_axis()
            self.plot.ax.axvspan(xmin, xmax, alpha=0.06)
            ax.plot([xv], [yv], marker="o")
            ax.axvline(xv, linestyle="--", linewidth=1.5)
            self.plot.ax.text(0.01, 0.01, f"{label} = {yv:.6g} @ x={xv:.6g}", transform=self.plot.ax.transAxes, va="bottom", ha="left")
            self.plot.canvas.draw_idle()
        except Exception as exc:
            QMessageBox.critical(self, f"Find {mode.title()} failed", str(exc))

    def _calc_find_max(self) -> None:
        self._calc_find_extremum("max")

    def _calc_find_min(self) -> None:
        self._calc_find_extremum("min")

    # ------------------------------------------------------------------
    # Export / batch
    # ------------------------------------------------------------------

    def _current_result_row(self) -> Dict[str, Any]:
        x, y, y_dy, x_col, y_col, dy_col = self._get_xy()
        xmin, xmax = self._get_window()
        smooth_d = self._smooth_derivative_enabled()
        bp = self._resolved_baseline_params()

        def _fmt_rng(rng):
            return "" if rng is None else f"{rng[0]:.6g}..{rng[1]:.6g}"

        sample = self.header.get("Sample", self.path.stem if self.path else "")
        td = self.res_double.tg if self.res_double is not None else np.nan
        tp = self.res_parallel.tg if self.res_parallel is not None else np.nan
        tx = self.tg_deriv if self.tg_deriv is not None else np.nan

        if self.res_parallel is not None:
            low_mode, high_mode = self.res_parallel.low_mode, self.res_parallel.high_mode
        else:
            low_mode = "point" if bp.low_point is not None else ("range" if bp.low_range is not None else "auto")
            high_mode = "point" if bp.high_point is not None else ("range" if bp.high_range is not None else "auto")

        return dict(
            file=str(self.path) if self.path else "", sample=sample, x_col=x_col, y_col=y_col,
            dy_source=(y_col if dy_col == "(same)" else dy_col),
            temp_unit=self._result_unit_label(x_col),
            window_min=xmin, window_max=xmax, smooth_derivative=int(smooth_d),
            low_mode=low_mode, low_range=_fmt_rng(bp.low_range),
            low_point_x=("" if bp.low_point is None else float(bp.low_point)),
            high_mode=high_mode, high_range=_fmt_rng(bp.high_range),
            high_point_x=("" if bp.high_point is None else float(bp.high_point)),
            Tg_double=td, Tg_parallel=tp, Tg_derivative=tx,
        )

    def _export_current_csv(self) -> None:
        if self.df is None:
            QMessageBox.critical(self, "Export", "No data loaded.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save result as CSV", "", "CSV (*.csv)")
        if not path:
            return
        pd.DataFrame([self._current_result_row()]).to_csv(path, index=False)
        QMessageBox.information(self, "Export", f"Saved:\n{path}")

    def _save_figure_png(self) -> None:
        if self.df is None:
            QMessageBox.critical(self, "Save figure", "No data loaded.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save figure as PNG", "", "PNG (*.png)")
        if not path:
            return
        self.plot.figure.savefig(path, dpi=200, bbox_inches="tight")
        QMessageBox.information(self, "Save figure", f"Saved:\n{path}")

    def _batch_run(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Select files for batch Tg", "", "Text files (*.txt *.dat *.asc);;All files (*.*)")
        if not files:
            return
        out_csv, _ = QFileDialog.getSaveFileName(self, "Save batch results as CSV", "", "CSV (*.csv)")
        if not out_csv:
            return

        x_name = self.x_combo.currentText()
        y_name = self.y_combo.currentText()
        dy_sel = self.dy_combo.currentText()
        xmin, xmax = self._get_window()
        smooth_d = self._smooth_derivative_enabled()
        invert = self.invert_y_check.isChecked()
        bp = self._resolved_baseline_params()

        rows: List[Dict[str, Any]] = []
        for f in files:
            p = Path(f)
            try:
                header, cols, df = parse_ta_sdt_txt(p)

                def _find(name):
                    for c in df.columns:
                        if c.lower() == name.lower():
                            return c
                    return None

                x_col = x_name if x_name in df.columns else (_find(x_name) or x_name)
                y_col = y_name if y_name in df.columns else (_find(y_name) or y_name)
                if x_col not in df.columns or y_col not in df.columns:
                    raise ValueError(f"Missing columns: {x_name} or {y_name}")

                x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
                x = self._convert_x_to_display(x, x_col)
                y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
                if dy_sel and dy_sel != "(same)" and dy_sel in df.columns:
                    y_dy = pd.to_numeric(df[dy_sel], errors="coerce").to_numpy(dtype=float)
                    dy_col = dy_sel
                else:
                    y_dy = y.copy()
                    dy_col = "(same)"

                m = np.isfinite(x) & np.isfinite(y) & np.isfinite(y_dy)
                x, y, y_dy = x[m], y[m], y_dy[m]
                order = np.argsort(x)
                x, y, y_dy = x[order], y[order], y_dy[order]
                if invert:
                    y, y_dy = -y, -y_dy

                tg_der = compute_tg_derivative(x, y_dy, xmin, xmax, smooth_derivative=smooth_d, restrict_range=None)
                try:
                    rp = compute_tg_parallel_improved(
                        x, y, xmin, xmax, smooth_derivative=smooth_d,
                        manual_low_range=bp.low_range, manual_high_range=bp.high_range,
                        manual_low_point=bp.low_point, manual_high_point=bp.high_point,
                    )
                    tg_par = rp.tg
                except Exception:
                    rp = None
                    tg_par = np.nan
                try:
                    rd = compute_tg_double_tangent(
                        x, y, xmin, xmax, threshold=0.20, guard_frac=0.0,
                        smooth_derivative=smooth_d, manual_low=bp.low_range, manual_slope=bp.manual_slope,
                    )
                    tg_double = rd.tg
                except Exception:
                    tg_double = np.nan

                sample = header.get("Sample", p.stem)
                default_low_mode = "point" if bp.low_point is not None else ("range" if bp.low_range is not None else "auto")
                default_high_mode = "point" if bp.high_point is not None else ("range" if bp.high_range is not None else "auto")

                def _fmt_rng(rng):
                    return "" if rng is None else f"{rng[0]:.6g}..{rng[1]:.6g}"

                rows.append(dict(
                    file=str(p), sample=sample, x_col=x_col, y_col=y_col,
                    dy_source=(y_col if dy_col == "(same)" else dy_col),
                    temp_unit=self._result_unit_label(x_col),
                    window_min=xmin, window_max=xmax, smooth_derivative=int(smooth_d),
                    low_mode=(rp.low_mode if rp is not None else default_low_mode),
                    low_range=_fmt_rng(bp.low_range),
                    low_point_x=("" if bp.low_point is None else float(bp.low_point)),
                    high_mode=(rp.high_mode if rp is not None else default_high_mode),
                    high_range=_fmt_rng(bp.high_range),
                    high_point_x=("" if bp.high_point is None else float(bp.high_point)),
                    Tg_double=tg_double, Tg_parallel=tg_par, Tg_derivative=tg_der,
                ))
            except Exception as exc:
                rows.append(dict(file=str(p), sample="", error=str(exc)))

        pd.DataFrame(rows).to_csv(out_csv, index=False)
        QMessageBox.information(self, "Batch complete", f"Saved batch CSV:\n{out_csv}")
