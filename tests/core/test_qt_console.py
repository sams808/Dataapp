"""Tests for qt_console.py (M18) — the embedded Python console dock."""
from __future__ import annotations

import numpy as np

from core.qt_console import ConsoleDock
from core.qt_models import Spectrum, SpectrumLibrary
from core.qt_shell import PrismMainWindow


def test_console_executes_and_shows_output(qtbot):
    dock = ConsoleDock({"x": 21})
    qtbot.addWidget(dock)
    dock.input.setText("print(x * 2)")
    dock._on_enter()
    assert "42" in dock.output.toPlainText()


def test_console_namespace_objects_are_live_not_copies(qtbot):
    library = SpectrumLibrary()
    dock = ConsoleDock({"library": library, "Spectrum": Spectrum, "np": np})
    qtbot.addWidget(dock)
    dock.input.setText("library.add(Spectrum(id='cons1', title='from_console', path='', kind='raman_xy', x=np.array([1.0]), y=np.array([2.0])))")
    dock._on_enter()
    assert library.get("cons1") is not None  # mutation visible outside the console


def test_console_error_is_shown_not_raised(qtbot):
    dock = ConsoleDock({})
    qtbot.addWidget(dock)
    dock.input.setText("1/0")
    dock._on_enter()
    assert "ZeroDivisionError" in dock.output.toPlainText()


def test_console_history_recall(qtbot):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtCore import QEvent

    dock = ConsoleDock({})
    qtbot.addWidget(dock)
    dock.input.setText("a = 1")
    dock._on_enter()
    dock.input.setText("a + 1")
    dock._on_enter()

    up = QKeyEvent(QEvent.KeyPress, Qt.Key_Up, Qt.NoModifier)
    dock.input.keyPressEvent(up)
    assert dock.input.text() == "a + 1"
    dock.input.keyPressEvent(up)
    assert dock.input.text() == "a = 1"


def test_shell_console_toggle_creates_dock_with_live_library(qtbot):
    window = PrismMainWindow()
    qtbot.addWidget(window)
    qtbot.wait(20)

    assert window._console_dock is None
    window.console_action.setChecked(True)
    assert window._console_dock is not None

    dock = window._console_dock
    dock.input.setText("len(library.all())")
    dock._on_enter()
    assert "0" in dock.output.toPlainText().splitlines()[-1]
