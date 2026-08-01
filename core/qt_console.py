"""
qt_console.py — embedded Python console (M18), the Origin-LabTalk /
Spectragryph-Automate analog: a dockable interactive interpreter with the
app's live data model in scope, for the power-user one-liners no GUI ever
covers ("np.trapz over every library spectrum between 900 and 1100",
"dump all accepted RRUFF matches to a CSV", ...).

Built on code.InteractiveConsole; stdout/stderr are captured per command.
The namespace is provided by the shell and includes the live window,
library, XAS store, HT-XRD series accessor, fit params, and numpy/pandas —
objects, not copies: whatever the console mutates is immediately live in
the GUI (that's the point, and also the reason the dock is opt-in via the
View menu rather than always open).
"""
from __future__ import annotations

import code
import contextlib
import io
import sys
import traceback
from typing import Any, Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QKeyEvent
from PySide6.QtWidgets import QDockWidget, QLineEdit, QPlainTextEdit, QVBoxLayout, QWidget

_BANNER = (
    "PRISM Python console — the live app is in scope:\n"
    "  window, library, xas_store, htxrd_series, fit_params, np, pd\n"
    "Example: [s.title for s in library.all()]\n"
)


class _EmbeddedConsole(code.InteractiveConsole):
    """InteractiveConsole that ALWAYS prints tracebacks to stderr.

    Python 3.11+'s showtraceback() delegates to sys.excepthook whenever a
    custom one is installed — and this app installs a global QMessageBox
    excepthook (qt_exception_hook), so without this override every typo in
    the console would pop the app-wide error dialog instead of printing in
    the console. (Caught by test_console_error_is_shown_not_raised, where
    pytest-qt's own excepthook produced the same escape.)"""

    def showtraceback(self) -> None:
        etype, value, tb = sys.exc_info()
        lines = traceback.format_exception(etype, value, tb.tb_next if tb else tb)
        self.write("".join(lines))


class _HistoryLineEdit(QLineEdit):
    """QLineEdit with shell-style up/down history."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.history: list[str] = []
        self._pos = 0

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Up and self.history:
            self._pos = max(0, self._pos - 1)
            self.setText(self.history[self._pos])
            return
        if event.key() == Qt.Key_Down and self.history:
            self._pos = min(len(self.history), self._pos + 1)
            self.setText(self.history[self._pos] if self._pos < len(self.history) else "")
            return
        super().keyPressEvent(event)

    def remember(self, text: str) -> None:
        if text and (not self.history or self.history[-1] != text):
            self.history.append(text)
        self._pos = len(self.history)


class ConsoleDock(QDockWidget):
    def __init__(self, namespace: Dict[str, Any], parent: Optional[QWidget] = None):
        super().__init__("Python console", parent)
        self.setObjectName("PythonConsoleDock")
        self._interp = _EmbeddedConsole(locals=dict(namespace))

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        mono = QFont("Consolas")
        mono.setStyleHint(QFont.Monospace)

        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setFont(mono)
        self.output.setPlainText(_BANNER)
        layout.addWidget(self.output, 1)

        self.input = _HistoryLineEdit()
        self.input.setFont(mono)
        self.input.setPlaceholderText(">>> ")
        self.input.returnPressed.connect(self._on_enter)
        layout.addWidget(self.input)

        self.setWidget(body)

    def _on_enter(self) -> None:
        source = self.input.text()
        self.input.remember(source)
        self.input.clear()
        self.output.appendPlainText(f">>> {source}")

        buf = io.StringIO()
        needs_more = False
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                needs_more = self._interp.push(source)
            except SystemExit:
                buf.write("(exit() is disabled inside the embedded console)\n")
        text = buf.getvalue()
        if text:
            self.output.appendPlainText(text.rstrip("\n"))
        if needs_more:
            self.output.appendPlainText("... (continuation lines: finish the block, or enter an empty line)")
        sb = self.output.verticalScrollBar()
        sb.setValue(sb.maximum())
