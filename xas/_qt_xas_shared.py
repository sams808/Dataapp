"""
xas/_qt_xas_shared.py — small constants shared across qt_xas.py and its
per-tab mixin modules (_qt_xas_*.py). Kept in its own module (rather than
qt_xas.py itself) so the mixins can import it without a circular import
back to the module that imports them.
"""
from __future__ import annotations

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]


def _to_int(text: str, default: int) -> int:
    try:
        return int(float((text or "").strip()))
    except (TypeError, ValueError):
        return default
