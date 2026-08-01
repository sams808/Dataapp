"""
qt_exception_hook.py — a global exception hook that shows a dialog and logs
the traceback, instead of an uncaught exception failing silently.

The two M0 showstopper bugs (the broken XAS import, the undefined edge_label
in _remember_payload) were invisible to a user double-clicking the app with
no console attached — Tkinter's default behavior for an exception raised
inside a callback is to print to stderr and otherwise do nothing visible.
Installing this in the Qt shell means that failure mode can't recur silently.
"""
from __future__ import annotations

import logging
import sys
import traceback

logger = logging.getLogger("prism")


def install(app) -> None:
    from PySide6.QtWidgets import QMessageBox

    def _hook(exc_type, exc_value, exc_tb):
        tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        logger.error("Unhandled exception:\n%s", tb_text)

        box = QMessageBox()
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("Unexpected error")
        box.setText(f"{exc_type.__name__}: {exc_value}")
        box.setDetailedText(tb_text)
        box.setStandardButtons(QMessageBox.Ok)
        box.exec()

    sys.excepthook = _hook
