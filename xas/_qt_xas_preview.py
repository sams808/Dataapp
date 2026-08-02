"""
xas/_qt_xas_preview.py — internal implementation detail of qt_xas.py:
XasWorkspace's Preview tab (mixed into XasWorkspace, not meant to be
used standalone).
"""
from __future__ import annotations

from PySide6.QtWidgets import QVBoxLayout, QWidget

from core.qt_widgets import PlotWidget
from ._qt_xas_shared import COLORS


class PreviewTabMixin:
    def _build_preview_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        self.preview_plot = PlotWidget(figsize=(7, 5))
        layout.addWidget(self.preview_plot)
        return w

    def _plot_selected_preview(self) -> None:
        if self.selected_sid is None:
            self.preview_plot.clear("Preview")
            return
        sp = self.store.get(self.selected_sid)
        ax = self.preview_plot.ax
        ax.clear()
        ax.plot(sp.energy, sp.y, lw=1.2, color=COLORS[0], label=sp.name)
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel(sp.units)
        ax.set_title(f"{sp.label} — {sp.name} [{sp.kind}]")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        self.preview_plot.figure.tight_layout()
        self.preview_plot.canvas.draw_idle()
