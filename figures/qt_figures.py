"""
qt_figures.py — the Figures workspace: publication-grade plot building
(the Origin-inspired module).

Tabs:
  XY builder    — layered plotting of library spectra: per-layer plot type
                  (line/scatter/symbols/sticks/area/bars/steps), color,
                  offset, multi-panel assignment; style presets
                  (Publication/Presentation/Poster), log axes, legend,
                  export at exact size in cm and dpi.
  Point fitting — Origin's classic curve models (linear, polynomials,
                  exponentials, power, log, Boltzmann, Gaussian,
                  Lorentzian, Arrhenius) fit to any spectrum with ±1σ and
                  R² (figures_science.FIT_MODELS).
  Ternary       — native barycentric ternary scatter from a CSV
                  composition table (the user's P-Bi notebooks' plot),
                  optional value column mapped to color.
  Raman ↔ XRD   — the cross-technique identification figure: Raman
                  spectrum with its accepted Mineral-ID phases over the
                  XRD pattern with its accepted phase-ID stick patterns.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QMessageBox, QPlainTextEdit, QPushButton, QSplitter,
    QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

import figures.figures_science as fsc
from core.qt_models import SpectrumLibrary
from core.qt_widgets import PlotWidget

LAYER_COLORS = ["black", "crimson", "royalblue", "seagreen", "darkorange", "purple", "teal", "brown"]


def _to_float(text: str, default: Optional[float] = None) -> Optional[float]:
    try:
        return float((text or "").strip())
    except (TypeError, ValueError):
        return default


class FiguresWorkspace(QWidget):
    def __init__(self, parent: Optional[QWidget] = None, library: Optional[SpectrumLibrary] = None,
                 xrd_db_path: Optional[str] = None):
        super().__init__(parent)
        self.library = library if library is not None else SpectrumLibrary()
        self.xrd_db_path = xrd_db_path  # None -> xrd_id_science default
        self.layers: List[Dict[str, Any]] = []
        self._ternary_df = None
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        self.tabs = QTabWidget()
        root.addWidget(self.tabs)
        self._build_xy_tab()
        self._build_series_tab()
        self._build_table_tab()
        self._build_fit_tab()
        self._build_ternary_tab()
        self._build_link_tab()

    # ------------------------------------------------------------------
    # XY builder
    # ------------------------------------------------------------------
    def _build_xy_tab(self) -> None:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        splitter = QSplitter()
        layout.addWidget(splitter)

        left = QWidget()
        left.setObjectName("Card")
        left.setMaximumWidth(360)
        ll = QVBoxLayout(left)

        ll.addWidget(QLabel("Library spectra"))
        self.spectra_list = QListWidget()
        self.spectra_list.setSelectionMode(QListWidget.ExtendedSelection)
        ll.addWidget(self.spectra_list, 1)
        add_btn = QPushButton("Add selected as layer(s)")
        add_btn.clicked.connect(self._add_layers)
        ll.addWidget(add_btn)

        ll.addWidget(QLabel("Layers (edit Type/Color/Offset/Panel in place)"))
        self.layers_table = QTableWidget(0, 6)
        self.layers_table.setHorizontalHeaderLabels(["Spectrum", "Type", "Color", "Offset", "Panel", "Axis"])
        self.layers_table.cellChanged.connect(lambda *_: self._sync_layers_from_table())
        ll.addWidget(self.layers_table, 1)
        rm_btn = QPushButton("Remove selected layer")
        rm_btn.clicked.connect(self._remove_layer)
        ll.addWidget(rm_btn)

        grid_row = QHBoxLayout()
        grid_row.addWidget(QLabel("Panels rows×cols"))
        self.rows_edit = QLineEdit("1")
        self.rows_edit.setMaximumWidth(35)
        self.cols_edit = QLineEdit("1")
        self.cols_edit.setMaximumWidth(35)
        grid_row.addWidget(self.rows_edit)
        grid_row.addWidget(self.cols_edit)
        grid_row.addWidget(QLabel("Preset"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(fsc.STYLE_PRESETS.keys()))
        grid_row.addWidget(self.preset_combo, 1)
        ll.addLayout(grid_row)

        lab_row = QHBoxLayout()
        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("title")
        self.xlabel_edit = QLineEdit("x")
        self.ylabel_edit = QLineEdit("y")
        lab_row.addWidget(self.title_edit)
        lab_row.addWidget(self.xlabel_edit)
        lab_row.addWidget(self.ylabel_edit)
        ll.addLayout(lab_row)

        opt_row = QHBoxLayout()
        self.diff_check = QCheckBox("difference vs 1st layer")
        self.diff_check.setToolTip("Plot every layer after the first as (layer − first), interpolated onto the first layer's grid — difference-spectra figures in one click.")
        opt_row.addWidget(self.diff_check)
        self.logx_check = QCheckBox("log x")
        self.logy_check = QCheckBox("log y")
        self.legend_check = QCheckBox("legend")
        self.legend_check.setChecked(True)
        opt_row.addWidget(self.logx_check)
        opt_row.addWidget(self.logy_check)
        opt_row.addWidget(self.legend_check)
        ll.addLayout(opt_row)

        tick_row = QHBoxLayout()
        self.ticks_check = QCheckBox("ticks")
        self.ticks_check.setChecked(True)
        self.ticklabels_check = QCheckBox("tick labels")
        self.ticklabels_check.setChecked(True)
        tick_row.addWidget(self.ticks_check)
        tick_row.addWidget(self.ticklabels_check)
        ll.addLayout(tick_row)

        render_btn = QPushButton("Render")
        render_btn.setObjectName("Primary")
        render_btn.clicked.connect(lambda: self.xy_plot.request_redraw(self.render_xy))
        ll.addWidget(render_btn)

        exp_row = QHBoxLayout()
        exp_row.addWidget(QLabel("W×H (cm)"))
        self.width_edit = QLineEdit("16")
        self.width_edit.setMaximumWidth(40)
        self.height_edit = QLineEdit("10")
        self.height_edit.setMaximumWidth(40)
        exp_row.addWidget(self.width_edit)
        exp_row.addWidget(self.height_edit)
        exp_row.addWidget(QLabel("dpi"))
        self.dpi_edit = QLineEdit("600")
        self.dpi_edit.setMaximumWidth(45)
        exp_row.addWidget(self.dpi_edit)
        export_btn = QPushButton("Export…")
        export_btn.clicked.connect(self.export_figure)
        exp_row.addWidget(export_btn)
        ll.addLayout(exp_row)
        splitter.addWidget(left)

        right = QWidget()
        right.setObjectName("Card")
        rl = QVBoxLayout(right)
        self.xy_plot = PlotWidget(figsize=(7, 5))
        rl.addWidget(self.xy_plot)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        self.tabs.addTab(tab, "XY builder")

    def set_spectra(self, spectrum_ids: List[str]) -> None:
        self.spectra_list.clear()
        for sid in spectrum_ids:
            sp = self.library.get(sid)
            if sp is not None:
                self.spectra_list.addItem(sp.title)
        for combo in (self.fit_spec_combo, self.link_raman_combo, self.link_xrd_combo):
            current = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            for sid in spectrum_ids:
                sp = self.library.get(sid)
                if sp is not None:
                    combo.addItem(sp.title, sid)
            idx = combo.findData(current) if current else -1
            if idx >= 0:
                combo.setCurrentIndex(idx)
            combo.blockSignals(False)

    def _add_layers(self) -> None:
        rows = sorted({i.row() for i in self.spectra_list.selectedIndexes()})
        all_specs = self.library.all()
        for r in rows:
            if 0 <= r < len(all_specs):
                sp = all_specs[r]
                self.layers.append({"id": sp.id, "title": sp.title, "type": "Line",
                                    "color": LAYER_COLORS[len(self.layers) % len(LAYER_COLORS)],
                                    "offset": 0.0, "panel": 1, "axis": "L"})
        self._rebuild_layers_table()

    def _remove_layer(self) -> None:
        rows = sorted({i.row() for i in self.layers_table.selectedIndexes()}, reverse=True)
        for r in rows:
            if 0 <= r < len(self.layers):
                self.layers.pop(r)
        self._rebuild_layers_table()

    def _rebuild_layers_table(self) -> None:
        self.layers_table.blockSignals(True)
        self.layers_table.setRowCount(len(self.layers))
        for row, layer in enumerate(self.layers):
            for col, val in enumerate([layer["title"], layer["type"], layer["color"],
                                       f"{layer['offset']:g}", str(layer["panel"]),
                                       layer.get("axis", "L")]):
                self.layers_table.setItem(row, col, QTableWidgetItem(val))
        self.layers_table.blockSignals(False)

    def _sync_layers_from_table(self) -> None:
        for row, layer in enumerate(self.layers):
            t = self.layers_table.item(row, 1)
            c = self.layers_table.item(row, 2)
            o = self.layers_table.item(row, 3)
            p = self.layers_table.item(row, 4)
            if t is not None and t.text().strip():
                layer["type"] = t.text().strip()
            if c is not None and c.text().strip():
                layer["color"] = c.text().strip()
            if o is not None:
                layer["offset"] = _to_float(o.text(), 0.0) or 0.0
            if p is not None:
                try:
                    layer["panel"] = max(1, int(float(p.text())))
                except (TypeError, ValueError):
                    pass
            a = self.layers_table.item(row, 5)
            if a is not None and a.text().strip().upper() in ("L", "R"):
                layer["axis"] = a.text().strip().upper()

    def render_xy(self) -> None:
        import matplotlib
        fig = self.xy_plot.figure
        fig.clf()
        preset = fsc.STYLE_PRESETS.get(self.preset_combo.currentText(), {})
        n_rows = max(1, int(_to_float(self.rows_edit.text(), 1)))
        n_cols = max(1, int(_to_float(self.cols_edit.text(), 1)))
        n_panels = n_rows * n_cols
        with matplotlib.rc_context(preset):
            axes = fig.subplots(n_rows, n_cols, squeeze=False)
            flat = axes.ravel()
            twins = {}
            first_sp = self.library.get(self.layers[0]["id"]) if self.layers else None
            for li, layer in enumerate(self.layers):
                sp = self.library.get(layer["id"])
                if sp is None:
                    continue
                ax = flat[min(layer["panel"] - 1, n_panels - 1)]
                if layer.get("axis", "L") == "R":
                    if id(ax) not in twins:
                        twins[id(ax)] = ax.twinx()
                    ax = twins[id(ax)]
                x = np.asarray(sp.x, float)
                y = np.asarray(sp.y, float)
                if self.diff_check.isChecked() and li > 0 and first_sp is not None:
                    x0 = np.asarray(first_sp.x, float)
                    y = np.interp(x0, x, y) - np.asarray(first_sp.y, float)
                    x = x0
                y = y + float(layer["offset"])
                kind, color, label = layer["type"].lower(), layer["color"], layer["title"]
                try:
                    if kind.startswith("scatter"):
                        ax.scatter(x, y, s=8, color=color, label=label)
                    elif kind.startswith("line +"):
                        ax.plot(x, y, "-o", ms=3, lw=1.0, color=color, label=label)
                    elif kind.startswith("stick"):
                        ax.vlines(x, float(layer["offset"]), y, color=color, lw=0.8, label=label)
                    elif kind.startswith("fill"):
                        ax.fill_between(x, float(layer["offset"]), y, color=color, alpha=0.4, label=label)
                    elif kind.startswith("bar"):
                        w = float(np.mean(np.diff(x))) * 0.8 if len(x) > 1 else 1.0
                        ax.bar(x, y - float(layer["offset"]), bottom=float(layer["offset"]), width=w,
                               color=color, label=label)
                    elif kind.startswith("step"):
                        ax.step(x, y, where="mid", color=color, label=label)
                    else:
                        ax.plot(x, y, lw=None, color=color, label=label)
                except (ValueError, TypeError) as exc:  # e.g. bad color name
                    ax.plot(x, y, label=f"{label} ({exc})")
            for ax in flat:
                if self.logx_check.isChecked():
                    ax.set_xscale("log")
                if self.logy_check.isChecked():
                    ax.set_yscale("log")
                if not self.ticks_check.isChecked():
                    ax.tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False)
                if not self.ticklabels_check.isChecked():
                    ax.tick_params(axis="both", which="both", labelbottom=False, labelleft=False,
                                   labeltop=False, labelright=False)
                ax.set_xlabel(self.xlabel_edit.text())
                ax.set_ylabel(self.ylabel_edit.text())
                if self.legend_check.isChecked():
                    handles, labels = ax.get_legend_handles_labels()
                    if handles:
                        ax.legend()
            if self.title_edit.text().strip():
                fig.suptitle(self.title_edit.text())
        fig.tight_layout()
        self.xy_plot.canvas.draw_idle()

    def export_figure(self) -> None:
        """Export whichever tab's figure is current, at exact cm size + dpi."""
        plots = {0: self.xy_plot, 1: self.series_plot, 2: self.table_plot,
                 3: self.fit_plot, 4: self.ternary_plot, 5: self.link_plot}
        plot = plots.get(self.tabs.currentIndex(), self.xy_plot)
        path, _ = QFileDialog.getSaveFileName(self, "Export figure", "",
                                              "PNG (*.png);;SVG (*.svg);;PDF (*.pdf);;TIFF (*.tiff)")
        if not path:
            return
        w = fsc.cm_to_inches(_to_float(self.width_edit.text(), 16.0))
        h = fsc.cm_to_inches(_to_float(self.height_edit.text(), 10.0))
        dpi = int(_to_float(self.dpi_edit.text(), 600))
        import matplotlib.text as mtext

        fig = plot.figure
        old = fig.get_size_inches()
        old_area = float(old[0]) * float(old[1])
        scale = float(np.sqrt((w * h) / old_area)) if old_area > 0 else 1.0
        texts = fig.findobj(mtext.Text)
        original_sizes = [(t, t.get_fontsize()) for t in texts]
        try:
            fig.set_size_inches(w, h)
            for t, size in original_sizes:
                t.set_fontsize(size * scale)
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
        finally:
            fig.set_size_inches(*old)
            for t, size in original_sizes:
                t.set_fontsize(size)
            plot.canvas.draw_idle()
        QMessageBox.information(self, "Export", f"Saved {os.path.basename(path)} at {dpi} dpi.")

    # ------------------------------------------------------------------
    # 2D / Series (user request: heatmaps, projections… of spectral series)
    # ------------------------------------------------------------------
    def _build_series_tab(self) -> None:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        left = QWidget()
        left.setObjectName("Card")
        left.setMaximumWidth(320)
        ll = QVBoxLayout(left)
        ll.addWidget(QLabel("Series = the XY-builder layer list, in order.\nY axis = series index (or values below)"))
        self.series_type_combo = QComboBox()
        self.series_type_combo.addItems(["Heatmap", "Contour (filled)", "Contour (lines)", "Waterfall 3D"])
        ll.addWidget(self.series_type_combo)
        self.series_cmap_combo = QComboBox()
        self.series_cmap_combo.addItems(["viridis", "magma", "inferno", "plasma", "jet", "RdBu_r", "Greys"])
        ll.addWidget(self.series_cmap_combo)
        self.series_scale_combo = QComboBox()
        self.series_scale_combo.addItems(["linear", "log", "sqrt"])
        ll.addWidget(self.series_scale_combo)
        ll.addWidget(QLabel("Series y values (opt, comma-sep\ne.g. temperatures / compositions)"))
        self.series_y_edit = QLineEdit()
        ll.addWidget(self.series_y_edit)
        render_btn = QPushButton("Render series plot")
        render_btn.setObjectName("Primary")
        render_btn.clicked.connect(lambda: self.series_plot.request_redraw(self.render_series))
        ll.addWidget(render_btn)
        ll.addStretch(1)
        layout.addWidget(left)
        self.series_plot = PlotWidget(figsize=(7, 5))
        layout.addWidget(self.series_plot, 1)
        self.tabs.addTab(tab, "2D / Series")

    def render_series(self) -> None:
        import matplotlib.colors as mcolors
        fig = self.series_plot.figure
        fig.clf()
        specs = [self.library.get(l["id"]) for l in self.layers]
        specs = [s for s in specs if s is not None]
        if len(specs) < 2:
            ax = fig.add_subplot(111)
            ax.set_title("Add at least 2 layers in the XY builder first")
            self.series_plot.canvas.draw_idle()
            return
        lo = max(float(np.nanmin(s.x)) for s in specs)
        hi = min(float(np.nanmax(s.x)) for s in specs)
        if hi <= lo:
            ax = fig.add_subplot(111)
            ax.set_title("Layers share no common x range")
            self.series_plot.canvas.draw_idle()
            return
        grid = np.linspace(lo, hi, 1500)
        data = np.vstack([np.interp(grid, np.asarray(s.x, float), np.asarray(s.y, float)) for s in specs])
        yvals_txt = [t for t in self.series_y_edit.text().split(",") if t.strip()]
        try:
            yvals = np.array([float(t) for t in yvals_txt], float) if len(yvals_txt) == len(specs) else np.arange(len(specs), dtype=float)
        except ValueError:
            yvals = np.arange(len(specs), dtype=float)
        cmap = self.series_cmap_combo.currentText()
        scale = self.series_scale_combo.currentText()
        if scale == "log":
            pos = data[data > 0]
            norm = mcolors.LogNorm(vmin=float(pos.min()), vmax=float(pos.max())) if pos.size else None
            show = np.where(data > 0, data, np.nan)
        elif scale == "sqrt":
            norm = mcolors.PowerNorm(gamma=0.5)
            show = data
        else:
            norm = None
            show = data
        kind = self.series_type_combo.currentText()
        if kind == "Waterfall 3D":
            ax = fig.add_subplot(111, projection="3d")
            X, Y = np.meshgrid(grid, yvals)
            ax.plot_surface(X, Y, np.where(np.isfinite(show), show, 0.0), cmap=cmap, linewidth=0)
            ax.set_zticks([])
        else:
            ax = fig.add_subplot(111)
            if kind == "Heatmap":
                mesh = ax.pcolormesh(grid, yvals, show, cmap=cmap, norm=norm, shading="nearest")
                fig.colorbar(mesh, ax=ax)
            elif kind == "Contour (filled)":
                cs_ = ax.contourf(grid, yvals, np.where(np.isfinite(show), show, 0.0), levels=30, cmap=cmap, norm=norm)
                fig.colorbar(cs_, ax=ax)
            else:
                ax.contour(grid, yvals, np.where(np.isfinite(show), show, 0.0), levels=20, cmap=cmap, norm=norm)
        ax.set_xlabel(self.xlabel_edit.text())
        ax.set_ylabel("series value")
        fig.tight_layout()
        self.series_plot.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Table plots (histograms / box / violin / correlation from any CSV)
    # ------------------------------------------------------------------
    def _build_table_tab(self) -> None:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        left = QWidget()
        left.setObjectName("Card")
        left.setMaximumWidth(320)
        ll = QVBoxLayout(left)
        load_btn = QPushButton("Load table (CSV)…")
        load_btn.clicked.connect(self.load_table_csv)
        ll.addWidget(load_btn)
        self.table_status = QLabel("No table loaded.")
        self.table_status.setWordWrap(True)
        ll.addWidget(self.table_status)
        self.table_plot_combo = QComboBox()
        self.table_plot_combo.addItems(["Histogram", "Box plot", "Violin plot", "Correlation matrix", "Scatter (col vs col)"])
        ll.addWidget(self.table_plot_combo)
        ll.addWidget(QLabel("Columns (comma-sep; scatter = X,Y)"))
        self.table_cols_edit = QLineEdit()
        ll.addWidget(self.table_cols_edit)
        render_btn = QPushButton("Render table plot")
        render_btn.setObjectName("Primary")
        render_btn.clicked.connect(lambda: self.table_plot.request_redraw(self.render_table_plot))
        ll.addWidget(render_btn)
        ll.addStretch(1)
        layout.addWidget(left)
        self.table_plot = PlotWidget(figsize=(7, 5))
        layout.addWidget(self.table_plot, 1)
        self.tabs.addTab(tab, "Table plots")
        self._table_df = None

    def load_table_csv(self) -> None:
        import pandas as pd
        path, _ = QFileDialog.getOpenFileName(self, "Load table", "", "CSV (*.csv *.txt);;All files (*.*)")
        if not path:
            return
        try:
            self._table_df = pd.read_csv(path, sep=None, engine="python")
        except Exception as exc:
            QMessageBox.critical(self, "Table", str(exc))
            return
        numeric = [c for c in self._table_df.columns if np.issubdtype(self._table_df[c].dtype, np.number)]
        self.table_cols_edit.setText(", ".join(str(c) for c in numeric[:4]))
        self.table_status.setText(f"{os.path.basename(path)}: {len(self._table_df)} rows; numeric: {numeric}")

    def render_table_plot(self) -> None:
        fig = self.table_plot.figure
        fig.clf()
        ax = fig.add_subplot(111)
        df = self._table_df
        if df is None:
            ax.set_title("Load a table first")
            self.table_plot.canvas.draw_idle()
            return
        cols = [c.strip() for c in self.table_cols_edit.text().split(",") if c.strip() in df.columns]
        kind = self.table_plot_combo.currentText()
        if not cols:
            ax.set_title("Pick at least one valid column")
        elif kind == "Histogram":
            for c in cols:
                ax.hist(df[c].dropna(), bins=25, alpha=0.6, label=str(c))
            ax.legend(fontsize=8)
        elif kind == "Box plot":
            ax.boxplot([df[c].dropna() for c in cols], labels=cols)
        elif kind == "Violin plot":
            ax.violinplot([df[c].dropna() for c in cols], showmedians=True)
            ax.set_xticks(range(1, len(cols) + 1), cols)
        elif kind == "Correlation matrix":
            corr = df[cols].corr()
            im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
            ax.set_yticks(range(len(cols)), cols)
            for i in range(len(cols)):
                for j in range(len(cols)):
                    ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
            fig.colorbar(im, ax=ax)
        elif kind == "Scatter (col vs col)" and len(cols) >= 2:
            ax.scatter(df[cols[0]], df[cols[1]], s=18, edgecolor="black", lw=0.3)
            ax.set_xlabel(cols[0])
            ax.set_ylabel(cols[1])
        ax.grid(alpha=0.25)
        fig.tight_layout()
        self.table_plot.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Point fitting
    # ------------------------------------------------------------------
    def _build_fit_tab(self) -> None:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        left = QWidget()
        left.setObjectName("Card")
        left.setMaximumWidth(340)
        ll = QVBoxLayout(left)
        ll.addWidget(QLabel("Data"))
        self.fit_spec_combo = QComboBox()
        ll.addWidget(self.fit_spec_combo)
        ll.addWidget(QLabel("Model"))
        self.fit_model_combo = QComboBox()
        self.fit_model_combo.addItems(list(fsc.FIT_MODELS.keys()))
        ll.addWidget(self.fit_model_combo)
        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("x range (opt)"))
        self.fit_xmin_edit = QLineEdit()
        self.fit_xmax_edit = QLineEdit()
        range_row.addWidget(self.fit_xmin_edit)
        range_row.addWidget(self.fit_xmax_edit)
        ll.addLayout(range_row)
        fit_btn = QPushButton("Fit")
        fit_btn.setObjectName("Primary")
        fit_btn.clicked.connect(self.run_point_fit)
        ll.addWidget(fit_btn)
        self.fit_report = QPlainTextEdit()
        self.fit_report.setReadOnly(True)
        ll.addWidget(self.fit_report, 1)
        layout.addWidget(left)
        self.fit_plot = PlotWidget(figsize=(7, 5))
        layout.addWidget(self.fit_plot, 1)
        self.tabs.addTab(tab, "Point fitting")

    def run_point_fit(self) -> None:
        sid = self.fit_spec_combo.currentData()
        sp = self.library.get(sid) if sid else None
        if sp is None:
            QMessageBox.warning(self, "Point fitting", "Select a spectrum.")
            return
        x, y = np.asarray(sp.x, float), np.asarray(sp.y, float)
        xmin, xmax = _to_float(self.fit_xmin_edit.text()), _to_float(self.fit_xmax_edit.text())
        if xmin is not None and xmax is not None and xmax > xmin:
            m = (x >= xmin) & (x <= xmax)
            x, y = x[m], y[m]
        try:
            result = fsc.fit_points(x, y, self.fit_model_combo.currentText())
        except Exception as exc:
            QMessageBox.warning(self, "Point fitting", str(exc))
            return
        self.fit_report.setPlainText(result.report())
        self._last_fit = (x, y, result, sp.title)
        self.fit_plot.request_redraw(self.render_fit)

    def render_fit(self) -> None:
        fig = self.fit_plot.figure
        fig.clf()
        ax = fig.add_subplot(111)
        if getattr(self, "_last_fit", None):
            x, y, result, title = self._last_fit
            ax.plot(x, y, "o", ms=4, color="black", label=title)
            ax.plot(result.x_fit, result.y_fit, "-", lw=1.5, color="crimson",
                    label=f"{result.model} (R²={result.r_squared:.4f})")
            ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        self.fit_plot.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Ternary
    # ------------------------------------------------------------------
    def _build_ternary_tab(self) -> None:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        left = QWidget()
        left.setObjectName("Card")
        left.setMaximumWidth(340)
        ll = QVBoxLayout(left)
        load_btn = QPushButton("Load composition table (CSV)…")
        load_btn.clicked.connect(self.load_ternary_csv)
        ll.addWidget(load_btn)
        self.ternary_status = QLabel("No table loaded.")
        self.ternary_status.setWordWrap(True)
        ll.addWidget(self.ternary_status)
        for key, label in (("a", "A (bottom-left)"), ("b", "B (bottom-right)"), ("c", "C (top)"), ("v", "color value (optional)")):
            ll.addWidget(QLabel(label))
            combo = QComboBox()
            setattr(self, f"ternary_{key}_combo", combo)
            ll.addWidget(combo)
        render_btn = QPushButton("Render ternary")
        render_btn.setObjectName("Primary")
        render_btn.clicked.connect(lambda: self.ternary_plot.request_redraw(self.render_ternary))
        ll.addWidget(render_btn)
        ll.addStretch(1)
        layout.addWidget(left)
        self.ternary_plot = PlotWidget(figsize=(6.5, 6))
        layout.addWidget(self.ternary_plot, 1)
        self.tabs.addTab(tab, "Ternary")

    def load_ternary_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Composition table", "", "CSV (*.csv *.txt);;All files (*.*)")
        if not path:
            return
        import pandas as pd
        try:
            df = pd.read_csv(path, sep=None, engine="python")
        except Exception as exc:
            QMessageBox.critical(self, "Ternary", f"Could not read the table: {exc}")
            return
        self._ternary_df = df
        numeric = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        for key in ("a", "b", "c", "v"):
            combo = getattr(self, f"ternary_{key}_combo")
            combo.clear()
            if key == "v":
                combo.addItem("(none)", None)
            for c in numeric:
                combo.addItem(str(c), str(c))
        for i, key in enumerate(("a", "b", "c")):
            combo = getattr(self, f"ternary_{key}_combo")
            if i < len(numeric):
                combo.setCurrentIndex(i)
        self.ternary_status.setText(f"{os.path.basename(path)}: {len(df)} rows, columns {list(df.columns)}")

    def render_ternary(self) -> None:
        fig = self.ternary_plot.figure
        fig.clf()
        ax = fig.add_subplot(111)
        if self._ternary_df is None:
            ax.set_title("Load a composition table first")
            self.ternary_plot.canvas.draw_idle()
            return
        df = self._ternary_df
        cols = [getattr(self, f"ternary_{k}_combo").currentText() for k in ("a", "b", "c")]
        if any(c not in df.columns for c in cols):
            ax.set_title("Pick the three composition columns")
            self.ternary_plot.canvas.draw_idle()
            return
        fsc.draw_ternary_axes(ax, labels=tuple(cols))
        x, y = fsc.ternary_to_xy(df[cols[0]], df[cols[1]], df[cols[2]])
        vcol = self.ternary_v_combo.currentData()
        if vcol:
            sc = ax.scatter(x, y, c=df[vcol], cmap="viridis", s=30, edgecolor="black", lw=0.3, zorder=3)
            fig.colorbar(sc, ax=ax, label=vcol, shrink=0.7)
        else:
            ax.scatter(x, y, color="crimson", s=30, edgecolor="black", lw=0.3, zorder=3)
        fig.tight_layout()
        self.ternary_plot.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Raman ↔ XRD identification figure
    # ------------------------------------------------------------------
    def _build_link_tab(self) -> None:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        left = QWidget()
        left.setObjectName("Card")
        left.setMaximumWidth(340)
        ll = QVBoxLayout(left)
        ll.addWidget(QLabel("Raman spectrum (with accepted Mineral-ID phases)"))
        self.link_raman_combo = QComboBox()
        ll.addWidget(self.link_raman_combo)
        ll.addWidget(QLabel("XRD pattern (with accepted XRD-ID phases)"))
        self.link_xrd_combo = QComboBox()
        ll.addWidget(self.link_xrd_combo)
        render_btn = QPushButton("Render identification figure")
        render_btn.setObjectName("Primary")
        render_btn.clicked.connect(lambda: self.link_plot.request_redraw(self.render_link))
        ll.addWidget(render_btn)
        note = QLabel("Two panels: the Raman spectrum with its accepted phases, and the "
                      "XRD pattern with each accepted phase's reference stick pattern — "
                      "the cross-technique identification in one publishable figure.")
        note.setWordWrap(True)
        note.setObjectName("SectionNote")
        ll.addWidget(note)
        ll.addStretch(1)
        layout.addWidget(left)
        self.link_plot = PlotWidget(figsize=(7, 6))
        layout.addWidget(self.link_plot, 1)
        self.tabs.addTab(tab, "Raman ↔ XRD")

    def render_link(self) -> None:
        import xrd.xrd_id_science as xid
        fig = self.link_plot.figure
        fig.clf()
        ax_r, ax_x = fig.subplots(2, 1)

        raman = self.library.get(self.link_raman_combo.currentData()) if self.link_raman_combo.currentData() else None
        xrd = self.library.get(self.link_xrd_combo.currentData()) if self.link_xrd_combo.currentData() else None

        if raman is not None:
            ax_r.plot(raman.x, raman.y, color="black", lw=0.9)
            phases = raman.meta.get("rruff_matches") or ([raman.meta["rruff_match"]] if raman.meta.get("rruff_match") else [])
            names = [p.get("mineral", "?") for p in phases]
            ax_r.set_title(f"Raman — {raman.title}" + (f"   [{', '.join(names)}]" if names else "   [no accepted phases]"),
                           fontsize=10)
            ax_r.set_xlabel("Raman shift (cm⁻¹)")
            ax_r.set_ylabel("Intensity")
        else:
            ax_r.set_title("Select a Raman spectrum")

        if xrd is not None:
            y = np.asarray(xrd.y, float)
            ymax = float(np.nanmax(y)) or 1.0
            ax_x.plot(xrd.x, y / ymax * 100.0, color="black", lw=0.9)
            phases = xrd.meta.get("xrd_matches") or ([xrd.meta["xrd_match"]] if xrd.meta.get("xrd_match") else [])
            db_path = self.xrd_db_path or xid.XRD_ID_DB_PATH
            shown = []
            for k, p in enumerate(phases):
                label = p.get("mineral") or p.get("name") or p.get("formula") or "?"
                hits = xid.find_cards_by_text(p.get("mineral") or p.get("name") or p.get("formula") or "",
                                              limit=1, db_path=db_path) if os.path.isfile(db_path) else []
                if hits:
                    h = hits[0]
                    tt = xid.d_to_two_theta(h["d"])
                    ok = np.isfinite(tt)
                    color = LAYER_COLORS[(k + 1) % len(LAYER_COLORS)]
                    ax_x.vlines(tt[ok], 0, -h["i"][ok], color=color, lw=1.0,
                                label=f"{label} [{p.get('source', h['source'])} {p.get('source_code', h['source_code'])}]")
                shown.append(label)
            ax_x.axhline(0, color="grey", lw=0.5)
            ax_x.set_title(f"XRD — {xrd.title}" + (f"   [{', '.join(shown)}]" if shown else "   [no accepted phases]"),
                           fontsize=10)
            ax_x.set_xlabel("2θ (deg)")
            ax_x.set_ylabel("norm. intensity / card lines")
            if shown:
                ax_x.legend(fontsize=7)
        else:
            ax_x.set_title("Select an XRD pattern")

        fig.tight_layout()
        self.link_plot.canvas.draw_idle()
