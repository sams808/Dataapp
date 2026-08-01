"""
qt_widgets.py — shared Qt widgets used by every tool window, instead of each
one reinventing matplotlib-embedding boilerplate. Today's Tk app embeds
matplotlib independently 12 times across 5 files — exactly the kind of
duplication that produced the 3-way XAS split; one shared PlotWidget here is
the fix for that pattern in the new architecture.

Also implements the two concrete performance fixes named in the rewrite
plan's "Performance" principle: debounced redraws (the historical Simple
Plot + CIF-overlay lag was almost certainly undebounced redraw-on-every-event)
and export-at-a-physical-size (A FAIRE item 14, "figure export size in cm").
"""
from __future__ import annotations

from typing import Callable, Optional

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class PlotWidget(QWidget):
    """Figure + canvas + navigation toolbar, with a debounced redraw helper.

    Update existing artists in place (line.set_data / set_offsets) wherever
    the underlying data hasn't changed, rather than ax.clear() + replot —
    that + undebounced redraws is the likely root cause of the historical
    CIF-overlay slowdown. request_redraw() below handles the debounce half
    of that fix; the "update in place" half is each caller's responsibility
    when it draws.
    """

    def __init__(self, parent: Optional[QWidget] = None, figsize=(6.0, 4.5), dpi: int = 100,
                 debounce_ms: int = 120):
        super().__init__(parent)
        self.figure = Figure(figsize=figsize, dpi=dpi)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # Live cursor readout (deferred M7 item, implemented once here so
        # every workspace gets it): x/y of the data point under the mouse.
        self.coords_label = QLabel("")
        self.coords_label.setObjectName("SectionNote")
        self.coords_label.setAlignment(Qt.AlignRight)
        layout.addWidget(self.coords_label)
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(debounce_ms)
        self._debounce.timeout.connect(self._flush_redraw)
        self._pending: Optional[tuple] = None
        self._zoom_key = None
        self._zoom_saved: Optional[tuple] = None

    def _on_mouse_move(self, event) -> None:
        if event.inaxes is None or event.xdata is None:
            self.coords_label.setText("")
            return
        self.coords_label.setText(f"x = {event.xdata:.4g}   y = {event.ydata:.4g}")

    def request_redraw(self, fn: Callable, *args, **kwargs) -> None:
        """Coalesce rapid-fire redraw requests (slider drags, live text-entry,
        a burst of selection-changed signals) into ONE redraw shortly after
        the last call, instead of one redraw per event. `fn` should do the
        actual plotting/artist updates; canvas.draw_idle() is called after.
        """
        self._pending = (fn, args, kwargs)
        self._debounce.start()

    def preserve_zoom(self, key) -> None:
        """Call at the START of a render fn, before clearing the figure:
        remembers the current axes limits when the user has zoomed/panned
        AND the content key (e.g. the selected spectrum's id) is unchanged
        — so re-rendering to preview another candidate card keeps the view
        instead of zooming back out. Apply with restore_zoom(ax) after
        drawing. A changed key (different spectrum) autoscales again."""
        axes = self.figure.get_axes()
        try:
            navigated = len(self.toolbar._nav_stack._elements) > 0
        except Exception:  # private-API drift: keep preserving once a view was saved
            navigated = self._zoom_saved is not None
        if axes and navigated and key == self._zoom_key:
            self._zoom_saved = (axes[0].get_xlim(), axes[0].get_ylim())
        else:
            self._zoom_saved = None
        self._zoom_key = key

    def restore_zoom(self, ax) -> None:
        if self._zoom_saved is not None:
            ax.set_xlim(self._zoom_saved[0])
            ax.set_ylim(self._zoom_saved[1])

    def reset_zoom_memory(self) -> None:
        """Forget the preserved view AND the toolbar's zoom history — the
        next render autoscales (Clear buttons, start-over flows)."""
        self._zoom_key = None
        self._zoom_saved = None
        try:
            self.toolbar._nav_stack.clear()
        except Exception:
            pass

    def cancel_pending(self) -> None:
        """Drop any queued debounced render. Call before drawing the figure
        DIRECTLY (not via request_redraw), or a stale queued render fires up
        to debounce_ms later and overwrites what was just drawn."""
        self._debounce.stop()
        self._pending = None

    def _flush_redraw(self) -> None:
        if self._pending is None:
            return
        fn, args, kwargs = self._pending
        self._pending = None
        try:
            fn(*args, **kwargs)
            self.canvas.draw_idle()
        except RuntimeError:
            # The widget is mid-destruction (deleteLater processed between
            # the debounce firing and the draw — pytest-qt teardown does
            # exactly this): there is nothing left to redraw.
            return

    def clear(self, title: str = "") -> None:
        self.ax.clear()
        self.ax.grid(alpha=0.25)
        if title:
            self.ax.set_title(title)
        self.canvas.draw_idle()

    def export_at_size_cm(self, path: str, width_cm: float, height_cm: float, dpi: int = 300) -> None:
        """Export the current figure at an exact physical size (A FAIRE item 14:
        a popup letting the user pick figure size in cm on export)."""
        w_in, h_in = width_cm / 2.54, height_cm / 2.54
        old_size = self.figure.get_size_inches()
        try:
            self.figure.set_size_inches(w_in, h_in)
            self.figure.savefig(path, dpi=dpi)
        finally:
            self.figure.set_size_inches(*old_size)
            self.canvas.draw_idle()


class CheckComboBox(QWidget):
    """A drop-list with checkboxes (multi-select filter), QualX-style.

    Reads as "All <label>" when nothing is checked (= no filtering), or a
    comma list of the checked entries. checked() returns [] for
    "no filter" so callers can pass it straight to the science layer.
    """

    def __init__(self, label: str = "values", parent: Optional[QWidget] = None):
        from PySide6.QtWidgets import QHBoxLayout, QMenu, QSizePolicy, QToolButton
        super().__init__(parent)
        self._label = label
        self._button = QToolButton()
        self._button.setPopupMode(QToolButton.InstantPopup)
        self._button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self._button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._menu = QMenu(self._button)
        self._button.setMenu(self._menu)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._button)
        self._actions: dict = {}
        self._update_text()

    def set_items(self, items) -> None:
        """Replace the entries, preserving check states for survivors.
        Entries are QCheckBoxes inside QWidgetActions so the menu stays
        open while ticking several."""
        from PySide6.QtWidgets import QCheckBox, QWidgetAction
        prev = {name for name, cb in self._actions.items() if cb.isChecked()}
        self._menu.clear()
        self._actions = {}
        for name in items:
            cb = QCheckBox(str(name))
            cb.setChecked(str(name) in prev)
            cb.toggled.connect(self._update_text)
            wa = QWidgetAction(self._menu)
            wa.setDefaultWidget(cb)
            self._menu.addAction(wa)
            self._actions[str(name)] = cb
        self._update_text()

    def checked(self) -> list:
        return [name for name, cb in self._actions.items() if cb.isChecked()]

    def clear_checks(self) -> None:
        for cb in self._actions.values():
            cb.setChecked(False)

    def _update_text(self, *_):
        sel = self.checked()
        self._button.setText(f"All {self._label}" if not sel else ", ".join(sel[:3]) + ("…" if len(sel) > 3 else ""))
