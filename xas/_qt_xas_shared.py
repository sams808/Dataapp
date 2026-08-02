"""
xas/_qt_xas_shared.py — small constants and helpers shared across
qt_xas.py and its per-tab mixin modules (_qt_xas_*.py). Kept in its own
module (rather than qt_xas.py itself) so the mixins can import it
without a circular import back to the module that imports them.
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional

from PySide6.QtWidgets import QMessageBox

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]


def _to_int(text: str, default: int) -> int:
    try:
        return int(float((text or "").strip()))
    except (TypeError, ValueError):
        return default


def for_each_selected_spectrum(widget: Any, store: Any, items: List[Any],
                               process_fn: Callable[[Any], Any], error_title: str) -> Optional[Any]:
    """Resolve each of `items` (QListWidgetItems) to a Spectrum via
    `store.find_by_name`, then call `process_fn(spectrum)`. A per-item
    exception shows a QMessageBox.critical(error_title, ...) and moves on
    to the next item rather than aborting the whole batch. Returns
    process_fn's return value for the last item that succeeded (or None),
    which the mu/normalize/EXAFS tabs use to plot/report on the
    most-recently-processed spectrum."""
    last = None
    for item in items:
        sp = store.find_by_name(item.text())
        if sp is None:
            continue
        try:
            result = process_fn(sp)
        except Exception as exc:
            QMessageBox.critical(widget, error_title, str(exc))
            continue
        last = result
    return last
