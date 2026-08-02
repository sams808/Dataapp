"""
qt_calc.py — the Calculations workspace: one place for every spectrum
operation (calc_science.CALC_OPERATIONS registry): arithmetic between
spectra, modulated addition, transforms, smoothing/despiking, derivatives,
integrals, resampling, linear-combination fitting, and statistics.

Selection semantics per operation group:
  multi / modulated / lcf — first selected spectrum is A (or the LCF
    target), the rest are B/rest/references, in selection order.
  everything else — applied to EACH selected spectrum (batch).
Preview shows inputs (faint) + result (bold); Apply adds derived spectra
to the Library (undoable through the shell's undo stack).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QMessageBox,
    QPlainTextEdit, QPushButton, QSplitter, QVBoxLayout, QWidget,
)

import processing.calc_science as cs
from core.qt_models import Spectrum, SpectrumLibrary
from core.qt_widgets import PlotWidget, to_float as _to_float

_SUFFIX = {
    "add": "add", "subtract": "sub", "multiply": "mul", "divide": "div",
    "average": "avg", "weighted_sum": "wsum", "modulated": "modadd",
    "scale_offset": "scaled", "normalize_max": "normmax", "normalize_area": "normarea",
    "normalize_minmax": "norm01", "log10": "log10", "ln": "ln", "exp": "exp",
    "sqrt": "sqrt", "power": "pow", "reciprocal": "inv", "absolute": "abs",
    "x_shift": "xshift", "x_scale": "xscale", "crop": "crop", "resample": "resamp",
    "savgol": "sg", "moving_average": "ma", "median": "med", "despike": "despiked",
    "1": "d1", "2": "d2", "cumulative": "integ", "lcf": "lcf",
}


class CalcWorkspace(QWidget):
    def __init__(self, parent: Optional[QWidget] = None, library: Optional[SpectrumLibrary] = None,
                 on_derived_added=None):
        super().__init__(parent)
        self.library = library if library is not None else SpectrumLibrary()
        self.on_derived_added = on_derived_added
        self._preview: List[tuple] = []  # (x, y, title) result curves of the last preview
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        splitter = QSplitter()
        root.addWidget(splitter)

        left = QWidget()
        left.setObjectName("Card")
        left.setMaximumWidth(340)
        left_layout = QVBoxLayout(left)

        left_layout.addWidget(QLabel("Spectra (selection order matters:\nfirst = A / LCF target)"))
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)
        left_layout.addWidget(self.file_list, 1)

        left_layout.addWidget(QLabel("Operation"))
        self.op_combo = QComboBox()
        for label in cs.CALC_OPERATIONS:
            self.op_combo.addItem(label)
        self.op_combo.currentTextChanged.connect(self._rebuild_params)
        left_layout.addWidget(self.op_combo)

        self.param_rows_holder = QWidget()
        self.param_rows_layout = QVBoxLayout(self.param_rows_holder)
        self.param_rows_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.param_rows_holder)
        self.param_edits: Dict[str, QLineEdit] = {}
        self._rebuild_params(self.op_combo.currentText())

        self.pick_btn = QPushButton("Pick on plot")
        self.pick_btn.setCheckable(True)
        self.pick_btn.setToolTip("Click the plot to fill this operation's position parameters: "
                                 "Despike: each click adds a spike position. Crop/ranges: clicks fill min then max.")
        self.pick_btn.toggled.connect(self._on_pick_toggled)
        left_layout.addWidget(self.pick_btn)

        preview_btn = QPushButton("Preview")
        preview_btn.clicked.connect(self.preview)
        left_layout.addWidget(preview_btn)
        apply_btn = QPushButton("Apply → Library")
        apply_btn.setObjectName("Primary")
        apply_btn.clicked.connect(self.apply_selected)
        left_layout.addWidget(apply_btn)

        self.status_label = QLabel("Select spectra, choose an operation, Preview, then Apply.")
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)
        splitter.addWidget(left)

        right = QWidget()
        right.setObjectName("Card")
        right_layout = QVBoxLayout(right)
        self.plot = PlotWidget(figsize=(7, 4.6))
        right_layout.addWidget(self.plot, 2)
        self.report_text = QPlainTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setMaximumHeight(150)
        right_layout.addWidget(self.report_text)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

    def _on_pick_toggled(self, checked: bool) -> None:
        from PySide6.QtCore import Qt
        if checked:
            mode = str(self.plot.toolbar.mode)
            if "zoom" in mode:
                self.plot.toolbar.zoom()
            elif "pan" in mode:
                self.plot.toolbar.pan()
            self.plot.canvas.setCursor(Qt.CrossCursor)
            self._pick_cid = self.plot.canvas.mpl_connect("button_press_event", self._on_pick_click)
        else:
            if getattr(self, "_pick_cid", None) is not None:
                self.plot.canvas.mpl_disconnect(self._pick_cid)
                self._pick_cid = None
            self.plot.canvas.unsetCursor()

    def _on_pick_click(self, event) -> None:
        if event.inaxes is None or event.xdata is None or self.plot.toolbar.mode:
            return
        xv = f"{float(event.xdata):.4g}"
        if "positions" in self.param_edits:  # despike: accumulate clicks
            cur = self.param_edits["positions"].text().strip()
            self.param_edits["positions"].setText(f"{cur}, {xv}" if cur else xv)
        elif "xmin" in self.param_edits and "xmax" in self.param_edits:  # ranges: fill min then max
            target = "xmin" if not self.param_edits["xmin"].text().strip() or self.param_edits["xmax"].text().strip() else "xmax"
            self.param_edits[target].setText(xv)
        elif "x1" in self.param_edits and "x2" in self.param_edits:
            target = "x1" if not self.param_edits["x1"].text().strip() or self.param_edits["x2"].text().strip() else "x2"
            self.param_edits[target].setText(xv)
        self.status_label.setText(f"Picked x = {xv}")

    def _rebuild_params(self, label: str) -> None:
        while self.param_rows_layout.count():
            item = self.param_rows_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self.param_edits = {}
        spec = cs.CALC_OPERATIONS.get(label, {"params": []})
        for key, text, default in spec["params"]:
            row = QWidget()
            lay = QHBoxLayout(row)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(QLabel(text))
            edit = QLineEdit(str(default))
            lay.addWidget(edit)
            self.param_edits[key] = edit
            self.param_rows_layout.addWidget(row)

    # ------------------------------------------------------------------
    def set_spectra(self, spectrum_ids: List[str]) -> None:
        selected = {i.row() for i in self.file_list.selectedIndexes()}
        self.file_list.clear()
        for sid in spectrum_ids:
            sp = self.library.get(sid)
            if sp is not None:
                self.file_list.addItem(sp.title)
        for row in selected:
            if row < self.file_list.count():
                self.file_list.item(row).setSelected(True)
        # Arriving on the page must show a spectrum immediately (user
        # request): auto-select the first one when nothing is selected yet;
        # render_preview draws selected inputs even before any computation.
        if not self.file_list.selectedItems() and self.file_list.count():
            self.file_list.item(0).setSelected(True)
        self.plot.request_redraw(self.render_preview)

    def _selected_spectra(self) -> List[Spectrum]:
        """In SELECTION order (click order), not list order — A/target first."""
        rows = [i.row() for i in self.file_list.selectedIndexes()]
        all_specs = self.library.all()
        # selectedIndexes preserves selection order for list widgets
        out = []
        for row in rows:
            if 0 <= row < len(all_specs):
                out.append(all_specs[row])
        return out

    def _param(self, key: str, default=None):
        edit = self.param_edits.get(key)
        return edit.text().strip() if edit is not None else default

    # ------------------------------------------------------------------
    def _compute(self) -> tuple:
        """Returns (results, report) where results = [(x, y, title)]."""
        label = self.op_combo.currentText()
        spec = cs.CALC_OPERATIONS[label]
        group, op = spec["group"], spec["op"]
        selected = self._selected_spectra()
        results, report = [], ""

        if group == "cluster":
            if len(selected) < 3:
                raise ValueError("Clustering needs at least 3 spectra.")
        elif group in ("multi", "modulated", "lcf"):
            need = 2
            if len(selected) < need:
                raise ValueError(f"'{label}' needs at least {need} selected spectra (first = A/target).")
        elif not selected:
            raise ValueError("Select at least one spectrum.")

        if group == "multi":
            weights = None
            if op == "weighted_sum":
                weights = [float(w) for w in (self._param("weights") or "").split(",")]
                if len(weights) != len(selected):
                    raise ValueError(f"Need {len(selected)} weights, got {len(weights)}.")
            x, y = cs.multi_op([(s.x, s.y) for s in selected], op, weights=weights)
            results = [(x, y, f"{selected[0].title}_{_SUFFIX[op]}")]

        elif group == "modulated":
            if len(selected) != 2:
                raise ValueError("Modulated addition needs exactly 2 spectra (A, then B).")
            kwargs = {"envelope": (self._param("envelope") or "constant").lower(), "k": _to_float(self._param("k"), 1.0)}
            for key in ("x1", "x2", "center", "width"):
                v = _to_float(self._param(key))
                if v is not None:
                    kwargs[key] = v
            x, y = cs.modulated_addition((selected[0].x, selected[0].y), (selected[1].x, selected[1].y), **kwargs)
            results = [(x, y, f"{selected[0].title}_{_SUFFIX[op]}")]

        elif group == "lcf":
            non_neg = (self._param("non_negative") or "1").strip() not in ("0", "false", "no")
            target, refs = selected[0], selected[1:]
            res = cs.linear_combination_fit(
                (target.x, target.y), [(r.x, r.y) for r in refs],
                ref_names=[r.title for r in refs], non_negative=non_neg,
            )
            total = sum(res.coefficients) or 1.0
            lines = [f"LCF: {target.title} ≈ Σ cᵢ·refᵢ   (R² = {res.r_squared:.5f})"]
            for name, c in zip(res.names, res.coefficients):
                lines.append(f"  {name}: c = {c:.4f}   ({100.0 * c / total:.1f} % of Σc)")
            report = "\n".join(lines)
            results = [(res.grid, res.y_fit, f"{target.title}_lcf_fit"),
                       (res.grid, res.residual, f"{target.title}_lcf_residual")]

        elif group == "cluster":
            import core.cluster_science as cl
            n = int(_to_float(self._param("n_clusters"), 3) or 3)
            matrix, cgrid = cl.build_feature_matrix([(s.x, s.y) for s in selected])
            res = cl.cluster_spectra(matrix, method=op, n_clusters=n)
            sil = res["silhouette"]
            lines = [f"Clustering ({op}, k={n}): silhouette = " + (f"{sil:.3f}" if sil is not None else "n/a")]
            for s_obj, lab in zip(selected, res["labels"]):
                lines.append(f"  {s_obj.title}: cluster {lab}")
            report = "\n".join(lines)
            for k, mean_y in cl.cluster_means(matrix, res["labels"]).items():
                results.append((cgrid, mean_y, f"cluster{k}_mean"))

        elif group == "stats":
            lines = []
            for s in selected:
                st = cs.statistics_report(s.x, s.y)
                lines.append(f"{s.title}: " + ", ".join(f"{k}={v:.6g}" for k, v in st.items()))
            report = "\n".join(lines)

        else:  # per-spectrum batch groups
            for s in selected:
                if group == "transform":
                    x, y = cs.transform(s.x, s.y, op,
                                        factor=_to_float(self._param("factor"), 1.0),
                                        offset=_to_float(self._param("offset"), 0.0),
                                        power=_to_float(self._param("power"), 2.0))
                elif group == "crop":
                    x, y = cs.crop(s.x, s.y, xmin=_to_float(self._param("xmin"), 0.0),
                                   xmax=_to_float(self._param("xmax"), 0.0))
                elif group == "resample":
                    x, y = cs.resample(s.x, s.y, n_points=int(_to_float(self._param("n_points"), 1000)),
                                       xmin=_to_float(self._param("xmin")), xmax=_to_float(self._param("xmax")))
                elif group == "smooth":
                    x, y = cs.smooth(s.x, s.y, method=op, window=int(_to_float(self._param("window"), 11)),
                                     polyorder=int(_to_float(self._param("polyorder"), 3)))
                elif group == "despike":
                    pos = [float(t) for t in (self._param("positions") or "").split(",") if t.strip()]
                    x, y = cs.despike(s.x, s.y, z=_to_float(self._param("z"), 6.0),
                                      window=int(_to_float(self._param("window"), 7)),
                                      positions=pos or None)
                elif group == "derivative":
                    x, y = cs.derivative(s.x, s.y, order=int(op),
                                         window=int(_to_float(self._param("window"), 11)),
                                         polyorder=int(_to_float(self._param("polyorder"), 3)))
                elif group == "integral":
                    x, y = cs.cumulative_integral(s.x, s.y)
                else:
                    raise ValueError(f"Unhandled group {group!r}")
                results.append((x, y, f"{s.title}_{_SUFFIX[op]}"))

        return results, report

    # ------------------------------------------------------------------
    def preview(self) -> None:
        try:
            results, report = self._compute()
        except Exception as exc:
            QMessageBox.warning(self, "Calculation", str(exc))
            return
        self._preview = results
        self.report_text.setPlainText(report)
        self.plot.request_redraw(self.render_preview)
        n = len(results)
        self.status_label.setText(f"Preview: {n} result curve(s)." if n else "Done (report below).")

    def render_preview(self) -> None:
        fig = self.plot.figure
        fig.clf()
        ax = fig.add_subplot(111)
        for s in self._selected_spectra():
            ax.plot(s.x, s.y, lw=0.8, alpha=0.35, label=s.title)
        for x, y, title in self._preview:
            ax.plot(x, y, lw=1.6, label=title)
        if self._preview or self._selected_spectra():
            ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.tight_layout()
        self.plot.canvas.draw_idle()

    def apply_selected(self) -> None:
        try:
            results, report = self._compute()
        except Exception as exc:
            QMessageBox.warning(self, "Calculation", str(exc))
            return
        self.report_text.setPlainText(report)
        label = self.op_combo.currentText()
        selected = self._selected_spectra()
        new_ids = []
        for x, y, title in results:
            derived = Spectrum(
                id=Spectrum.new_id(), title=title, path="", kind=selected[0].kind if selected else "generic_xy",
                x=np.asarray(x, float), y=np.asarray(y, float), df=None,
                meta={"derived": f"calc:{label}", "sources": [s.title for s in selected],
                      "params": {k: e.text() for k, e in self.param_edits.items()}},
                status="derived",
            )
            self.library.add(derived)
            new_ids.append(derived.id)
        if new_ids and self.on_derived_added is not None:
            self.on_derived_added(new_ids)
        self._preview = results
        self.plot.request_redraw(self.render_preview)
        self.set_spectra([s.id for s in self.library.all()])
        self.status_label.setText(
            f"Added {len(new_ids)} derived spectrum/spectra to the Library." if new_ids
            else "Done (no curve output — see the report)."
        )
