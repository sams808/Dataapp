"""Tests for qt_custom_import.py — the parser/column-override import dialog
(Tk 'Custom import' parity)."""
from __future__ import annotations

import numpy as np

from core.qt_custom_import import AUTO, CustomImportDialog


def _three_column_file(tmp_path):
    """A file whose FIRST two columns are not the interesting pair — the
    exact situation Custom Import exists for."""
    p = tmp_path / "threecol.txt"
    x = np.linspace(100, 200, 40)
    junk = np.zeros(40)
    signal = np.sin(x / 10) + 2
    np.savetxt(p, np.column_stack([junk, x, signal]))
    return str(p)


def test_dialog_parses_and_lists_columns(qtbot, tmp_path):
    path = _three_column_file(tmp_path)
    dlg = CustomImportDialog(None, path)
    qtbot.addWidget(dlg)
    qtbot.wait(20)
    assert dlg._df is not None
    assert dlg.x_combo.count() == 3
    assert "parsed as" in dlg.detected_label.text()


def test_column_override_produces_correct_spectrum(qtbot, tmp_path):
    path = _three_column_file(tmp_path)
    dlg = CustomImportDialog(None, path)
    qtbot.addWidget(dlg)
    qtbot.wait(20)

    cols = [dlg.x_combo.itemText(i) for i in range(dlg.x_combo.count())]
    dlg.x_combo.setCurrentText(cols[1])  # the real x lives in column 2
    dlg.y_combo.setCurrentText(cols[2])  # and the signal in column 3
    qtbot.wait(20)  # let the preview's deferred canvas.draw_idle() complete
    dlg._on_import()
    qtbot.wait(20)

    sp = dlg.spectrum
    assert sp is not None
    assert sp.x.min() >= 100.0 and sp.x.max() <= 200.0
    assert sp.meta["custom_import"]["x_col"] == cols[1]
    assert sp.meta["custom_import"]["y_col"] == cols[2]


def test_parser_override_reparses(qtbot, tmp_path):
    path = _three_column_file(tmp_path)
    dlg = CustomImportDialog(None, path)
    qtbot.addWidget(dlg)
    qtbot.wait(20)
    auto_parser = dlg._meta.get("selected_parser")

    dlg.parser_combo.setCurrentText("generic_xy")
    qtbot.wait(20)
    assert dlg._meta.get("selected_parser") == "generic_xy"
    assert dlg.parser_combo.currentText() != AUTO or auto_parser == "generic_xy"


def test_bad_selection_blocks_import(qtbot, tmp_path):
    p = tmp_path / "tiny.txt"
    p.write_text("1 2\n3 4\n")
    dlg = CustomImportDialog(None, str(p))
    qtbot.addWidget(dlg)
    qtbot.wait(20)
    # Same column for X and Y is fine numerically; simulate an unusable
    # state instead: parser producing nothing plottable.
    dlg._df = None
    dlg._on_import()  # warning dialog neutralized by conftest fixture
    assert dlg.spectrum is None
