"""Tests for the Library management tools (rename/duplicate/reorder/
delete-with-undo) and the Combine/scale dialog — Tk-parity features the
first Qt pass dropped, restored on user request."""
from __future__ import annotations

import numpy as np

from core.qt_models import Spectrum
from core.qt_shell import CombineDialog, PrismMainWindow


def _spectrum(title, value=1.0, n=50):
    x = np.linspace(0, 100, n)
    return Spectrum(id=Spectrum.new_id(), title=title, path="", kind="raman_xy",
                    x=x, y=np.full(n, float(value)))


def _window_with(qtbot, titles_values):
    window = PrismMainWindow()
    qtbot.addWidget(window)
    for title, value in titles_values:
        window.library.add(_spectrum(title, value))
    window.library_page._refresh_table()
    qtbot.wait(20)
    return window


def test_rename_updates_library(qtbot, monkeypatch):
    window = _window_with(qtbot, [("old_name", 1.0)])
    page = window.library_page
    page.table.selectRow(0)
    monkeypatch.setattr("PySide6.QtWidgets.QInputDialog.getText",
                        staticmethod(lambda *a, **k: ("new_name", True)))
    page._rename_selected()
    assert window.library.all()[0].title == "new_name"
    assert page.table.item(0, 0).text() == "new_name"


def test_duplicate_creates_derived_copy(qtbot):
    window = _window_with(qtbot, [("orig", 5.0)])
    page = window.library_page
    page.table.selectRow(0)
    page._duplicate_selected()
    items = window.library.all()
    assert len(items) == 2
    assert items[1].title == "orig_copy"
    assert items[1].status == "derived"
    assert items[1].id != items[0].id
    items[1].y[0] = 999.0
    assert items[0].y[0] == 5.0  # deep copy, not a view


def test_move_up_down_reorders(qtbot):
    window = _window_with(qtbot, [("a", 1), ("b", 2), ("c", 3)])
    page = window.library_page
    page.table.selectRow(2)
    page._move_selected(-1)
    assert [s.title for s in window.library.all()] == ["a", "c", "b"]
    page._move_selected(+1)
    assert [s.title for s in window.library.all()] == ["a", "b", "c"]


def test_delete_and_undo_restores_order(qtbot):
    window = _window_with(qtbot, [("a", 1), ("b", 2), ("c", 3)])
    page = window.library_page
    page.table.selectRow(1)
    page._delete_selected()
    assert [s.title for s in window.library.all()] == ["a", "c"]
    assert page.undo_btn.isEnabled()

    page._undo_delete()
    assert [s.title for s in window.library.all()] == ["a", "b", "c"]
    assert not page.undo_btn.isEnabled()


def test_export_selected_single_writes_two_column_txt(qtbot, tmp_path, monkeypatch):
    window = _window_with(qtbot, [("expo", 2.5)])
    page = window.library_page
    page.table.selectRow(0)

    out = tmp_path / "expo.txt"
    monkeypatch.setattr("core.qt_library.QFileDialog.getSaveFileName", staticmethod(lambda *a, **k: (str(out), "")))
    page._export_selected_txt()

    assert out.exists()
    data = np.loadtxt(out)
    assert data.shape == (50, 2)
    assert np.allclose(data[:, 1], 2.5)


def test_export_selected_multi_writes_into_folder(qtbot, tmp_path, monkeypatch):
    window = _window_with(qtbot, [("one", 1.0), ("two", 2.0)])
    page = window.library_page
    page.table.selectAll()

    monkeypatch.setattr("core.qt_library.QFileDialog.getExistingDirectory", staticmethod(lambda *a, **k: str(tmp_path)))
    page._export_selected_txt()

    assert (tmp_path / "one.txt").exists()
    assert (tmp_path / "two.txt").exists()
    assert np.allclose(np.loadtxt(tmp_path / "two.txt")[:, 1], 2.0)


def test_clear_all_is_undoable(qtbot):
    window = _window_with(qtbot, [("a", 1), ("b", 2), ("c", 3)])
    page = window.library_page
    page.clear_all()  # confirmation auto-answered Yes by conftest fixture
    assert len(window.library) == 0
    assert page.undo_btn.isEnabled()

    page._undo_delete()
    assert [s.title for s in window.library.all()] == ["a", "b", "c"]


def test_rename_is_undoable(qtbot, monkeypatch):
    window = _window_with(qtbot, [("old_name", 1.0)])
    page = window.library_page
    page.table.selectRow(0)
    monkeypatch.setattr("PySide6.QtWidgets.QInputDialog.getText",
                        staticmethod(lambda *a, **k: ("new_name", True)))
    page._rename_selected()
    assert window.library.all()[0].title == "new_name"
    assert page.undo_btn.isEnabled()

    page._undo()
    assert window.library.all()[0].title == "old_name"
    assert not page.undo_btn.isEnabled()


def test_duplicate_is_undoable(qtbot):
    window = _window_with(qtbot, [("orig", 5.0)])
    page = window.library_page
    page.table.selectRow(0)
    page._duplicate_selected()
    assert len(window.library) == 2

    page._undo()
    assert [s.title for s in window.library.all()] == ["orig"]


def test_undo_add_tolerates_already_deleted_spectrum(qtbot):
    window = _window_with(qtbot, [("orig", 5.0)])
    page = window.library_page
    page.table.selectRow(0)
    page._duplicate_selected()
    copy_id = window.library.all()[1].id
    window.library.remove(copy_id)  # user deletes the copy by other means
    page._undo()  # must not raise; nothing left to remove
    assert [s.title for s in window.library.all()] == ["orig"]


def test_accepted_identification_is_undoable(qtbot):
    window = _window_with(qtbot, [("query", 1.0)])
    page = window.library_page
    sp = window.library.all()[0]
    # Simulate what the shell's on_accept callback records, then the
    # workspace's own meta write.
    page.push_undo(("ident", sp.id, None))
    sp.meta["rruff_match"] = {"mineral": "Quartz", "rruff_id": "R040031"}

    page._undo()
    assert "rruff_match" not in sp.meta

    # A re-identification restores the PREVIOUS match on undo, not nothing.
    old = {"mineral": "Calcite", "rruff_id": "R040070"}
    sp.meta["rruff_match"] = dict(old)
    page.push_undo(("ident", sp.id, dict(old)))
    sp.meta["rruff_match"] = {"mineral": "Quartz", "rruff_id": "R040031"}
    page._undo()
    assert sp.meta["rruff_match"] == old


def test_applied_baseline_is_undoable_through_shell(qtbot):
    window = _window_with(qtbot, [("raw", 1.0)])
    page = window.library_page
    bl_page = window.baseline_page
    bl_page.set_spectra([s.id for s in window.library.all()])
    bl_page.file_list.item(0).setSelected(True)
    bl_page.method_combo.setCurrentText("poly")
    bl_page.param_edits[0].setText("1")
    bl_page.roi_edit.setText("0-100")
    bl_page.apply_selected()
    qtbot.wait(20)
    assert [s.title for s in window.library.all()] == ["raw", "raw_bl"]
    assert page.undo_btn.isEnabled()

    page._undo()
    assert [s.title for s in window.library.all()] == ["raw"]


def test_mixed_undo_stack_unwinds_in_reverse_order(qtbot):
    window = _window_with(qtbot, [("a", 1), ("b", 2)])
    page = window.library_page
    page.table.selectRow(0)
    page._delete_selected()  # a gone
    page.table.selectRow(0)
    page._duplicate_selected()  # b_copy added
    assert [s.title for s in window.library.all()] == ["b", "b_copy"]

    page._undo()  # undoes the duplicate first
    assert [s.title for s in window.library.all()] == ["b"]
    page._undo()  # then the delete
    assert [s.title for s in window.library.all()] == ["a", "b"]


def test_combine_dialog_sum(qtbot):
    a = _spectrum("a", 2.0)
    b = _spectrum("b", 3.0)
    dlg = CombineDialog(None, [a, b])
    qtbot.addWidget(dlg)
    dlg.op_combo.setCurrentText("Sum")
    dlg._on_create()
    result = dlg.result_spectrum
    assert result is not None
    assert np.allclose(result.y, 5.0)
    assert result.status == "derived"
    assert result.meta["sources"] == ["a", "b"]


def test_combine_dialog_subtract_with_weights(qtbot):
    a = _spectrum("a", 10.0)
    b = _spectrum("b", 4.0)
    dlg = CombineDialog(None, [a, b])
    qtbot.addWidget(dlg)
    dlg.op_combo.setCurrentText("Subtract (1st − rest)")
    dlg.weights_edit.setText("1, 0.5")
    dlg._on_create()
    assert np.allclose(dlg.result_spectrum.y, 8.0)


def test_combine_dialog_scale_single(qtbot):
    a = _spectrum("a", 3.0)
    dlg = CombineDialog(None, [a])
    qtbot.addWidget(dlg)
    dlg.factor_edit.setText("2")
    dlg.offset_edit.setText("1")
    dlg._on_create()
    assert np.allclose(dlg.result_spectrum.y, 7.0)
    assert dlg.result_spectrum.title == "a_scaled"
