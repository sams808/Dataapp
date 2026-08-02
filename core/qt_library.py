"""
core/qt_library.py — internal implementation detail of qt_shell.py: the
Library nav page (import, browse, rename/duplicate/reorder/delete-with-
undo, export, Combine/scale) and its CombineDialog. Split out of
qt_shell.py (the app shell itself) since these are a self-contained
feature, not shell wiring — qt_shell.py re-imports both names so every
existing `from core.qt_shell import LibraryPage`/`CombineDialog` call
site needs no changes.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget,
)

import core.io_universal as io_universal
from core.qt_models import Spectrum, SpectrumLibrary
from core.qt_widgets import PlotWidget

logger = logging.getLogger("prism")


def _load_spectrum_from_path(path: str) -> Spectrum:
    """Generic import via io_universal's parser registry, picking X/Y from
    each parser's canonical_map (every parser sets canonical_map["X"]/["Y"]
    as a fallback pair even when it can't infer richer canonical keys)."""
    df, meta = io_universal.load_any(path, return_meta=True)
    canon = meta.get("canonical_map", {}) or {}
    x_col = canon.get("X") or df.columns[0]
    y_col = canon.get("Y") or df.columns[1]
    x = df[x_col].astype(float).to_numpy()
    y = df[y_col].astype(float).to_numpy()
    order = np.argsort(x, kind="mergesort")
    return Spectrum(
        id=Spectrum.new_id(),
        title=Path(path).stem,
        path=str(path),
        kind=meta.get("selected_parser", "generic_xy"),
        x=x[order], y=y[order],
        df=df, meta=meta, status="imported",
    )


class CombineDialog(QDialog):
    """Sum / average / weighted-subtract multiple spectra, or scale one —
    the generalized successor of the old Tk SpectralSumWindow. The result
    is added to the Library as a derived spectrum."""

    def __init__(self, parent, spectra: list):
        super().__init__(parent)
        self.setWindowTitle("Combine / scale spectra")
        self.spectra = spectra
        self.result_spectrum: Optional[Spectrum] = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Inputs: " + ", ".join(s.title for s in spectra)))

        from PySide6.QtWidgets import QComboBox
        op_row = QHBoxLayout()
        op_row.addWidget(QLabel("Operation"))
        self.op_combo = QComboBox()
        ops = ["Scale (single spectrum)"] if len(spectra) == 1 else [
            "Sum", "Average", "Subtract (1st − rest)",
        ]
        self.op_combo.addItems(ops)
        op_row.addWidget(self.op_combo, 1)
        layout.addLayout(op_row)

        self.weights_edit = QLineEdit()
        self.weights_edit.setPlaceholderText("optional weights, comma-separated (e.g. 1, 0.5)")
        self.factor_edit = QLineEdit("1.0")
        self.offset_edit = QLineEdit("0.0")
        if len(spectra) == 1:
            scale_row = QHBoxLayout()
            scale_row.addWidget(QLabel("Factor"))
            scale_row.addWidget(self.factor_edit)
            scale_row.addWidget(QLabel("Offset"))
            scale_row.addWidget(self.offset_edit)
            layout.addLayout(scale_row)
        else:
            layout.addWidget(self.weights_edit)
            self.normalize_check = QCheckBox("Area-normalize each spectrum first (area → 100)")
            layout.addWidget(self.normalize_check)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Result name"))
        self.name_edit = QLineEdit(self._suggest_name())
        name_row.addWidget(self.name_edit, 1)
        layout.addLayout(name_row)

        buttons = QHBoxLayout()
        ok_btn = QPushButton("Create")
        ok_btn.setObjectName("Primary")
        ok_btn.clicked.connect(self._on_create)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        buttons.addStretch(1)
        buttons.addWidget(cancel_btn)
        buttons.addWidget(ok_btn)
        layout.addLayout(buttons)

    def _suggest_name(self) -> str:
        if len(self.spectra) == 1:
            return f"{self.spectra[0].title}_scaled"
        return f"{self.spectra[0].title}_combined{len(self.spectra)}"

    def _on_create(self) -> None:
        import core.spectrum_math as sm
        try:
            if len(self.spectra) == 1:
                sp = self.spectra[0]
                factor = float(self.factor_edit.text() or "1")
                offset = float(self.offset_edit.text() or "0")
                x, y = sm.scale_spectrum(sp.x, sp.y, factor=factor, offset=offset)
                op_desc = f"scale×{factor:g}+{offset:g}"
            else:
                op_ui = self.op_combo.currentText()
                op = {"Sum": "sum", "Average": "average"}.get(op_ui, "subtract")
                weights = None
                wtext = self.weights_edit.text().strip()
                if wtext:
                    weights = [float(w) for w in wtext.split(",")]
                x, y = sm.combine_spectra(
                    [(s.x, s.y) for s in self.spectra], op=op, weights=weights,
                    normalize_first=self.normalize_check.isChecked(),
                )
                op_desc = op
        except (ValueError, TypeError) as exc:
            QMessageBox.critical(self, "Combine error", str(exc))
            return

        title = self.name_edit.text().strip() or self._suggest_name()
        self.result_spectrum = Spectrum(
            id=Spectrum.new_id(), title=title, path="", kind=self.spectra[0].kind,
            x=x, y=y, df=None,
            meta={"derived": op_desc, "sources": [s.title for s in self.spectra]},
            status="derived",
        )
        self.accept()


class LibraryPage(QWidget):
    """Data hub: import, browse, and plot imported spectra."""

    def __init__(self, library: SpectrumLibrary, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.library = library

        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        left = QWidget()
        left.setObjectName("Card")
        left.setFixedWidth(320)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(12, 12, 12, 12)

        title = QLabel("Imported files")
        title.setObjectName("SectionTitle")
        left_layout.addWidget(title)

        note = QLabel("Bring data in, then select a row to preview it.")
        note.setObjectName("SectionNote")
        note.setWordWrap(True)
        left_layout.addWidget(note)

        import_btn = QPushButton("Import files…")
        import_btn.setObjectName("Primary")
        import_btn.clicked.connect(self._on_import_clicked)
        left_layout.addWidget(import_btn)

        custom_import_btn = QPushButton("Custom import…")
        custom_import_btn.setToolTip("Pick the parser and X/Y columns manually — for files the auto-detection guesses wrong on.")
        custom_import_btn.clicked.connect(self._on_custom_import_clicked)
        left_layout.addWidget(custom_import_btn)

        combine_btn = QPushButton("Combine / scale selected…")
        combine_btn.clicked.connect(self._on_combine_clicked)
        left_layout.addWidget(combine_btn)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Title", "Kind"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_context_menu)
        left_layout.addWidget(self.table, 1)

        self.undo_btn = QPushButton("Undo")
        self.undo_btn.setToolTip("Undo the last library change: delete, rename, duplicate, combine result, applied baseline, or accepted mineral ID.")
        self.undo_btn.setEnabled(False)
        self.undo_btn.clicked.connect(self._undo)
        left_layout.addWidget(self.undo_btn)
        # Typed actions, most recent last:
        #   ("delete", [(position, Spectrum), ...])   undo re-adds at position
        #   ("add", [spectrum_id, ...])               undo removes (derived spectra)
        #   ("rename", spectrum_id, old_title)        undo restores the title
        #   ("ident", spectrum_id, old_match|None)    undo restores meta["rruff_match"]
        self._undo_stack: list = []

        root.addWidget(left)

        right = QWidget()
        right.setObjectName("Card")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 8, 8, 8)
        self.plot = PlotWidget()
        self.plot.clear("Select a spectrum to preview")
        right_layout.addWidget(self.plot)
        root.addWidget(right, 1)

    def _on_import_clicked(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import files", "",
            "Data files (*.txt *.dat *.csv *.xy *.asc);;All files (*.*)",
        )
        if not paths:
            return
        added = 0
        errors = []
        for path in paths:
            try:
                spectrum = _load_spectrum_from_path(path)
                self.library.add(spectrum)
                added += 1
            except Exception as exc:
                errors.append(f"{os.path.basename(path)}: {exc}")
                logger.warning("Import failed for %s", path, exc_info=True)
        if added:
            self._refresh_table()
        if errors:
            QMessageBox.warning(self, "Import", "Some files could not be imported:\n" + "\n".join(errors))

    def _refresh_table(self) -> None:
        items = self.library.all()
        self.table.setRowCount(len(items))
        for row, spectrum in enumerate(items):
            title_item = QTableWidgetItem(spectrum.title)
            title_item.setData(Qt.UserRole, spectrum.id)
            self.table.setItem(row, 0, title_item)
            self.table.setItem(row, 1, QTableWidgetItem(spectrum.kind))

    # ------------------------------------------------------------------
    # Library management: rename / duplicate / reorder / delete-with-undo
    # (parity with the old Tk app's list management, which the first Qt
    # pass dropped) plus Combine/scale (the old SpectralSumWindow,
    # generalized).
    # ------------------------------------------------------------------
    def _selected_spectra(self) -> list:
        rows = sorted({i.row() for i in self.table.selectionModel().selectedRows()})
        out = []
        for row in rows:
            item = self.table.item(row, 0)
            sp = self.library.get(item.data(Qt.UserRole)) if item else None
            if sp is not None:
                out.append(sp)
        return out

    def _on_context_menu(self, pos) -> None:
        from PySide6.QtWidgets import QMenu
        if self.table.itemAt(pos) is None:
            return
        selected = self._selected_spectra()
        menu = QMenu(self)
        if len(selected) == 1:
            menu.addAction("Rename…", self._rename_selected)
            menu.addAction("Duplicate", self._duplicate_selected)
            menu.addSeparator()
            menu.addAction("Move up", lambda: self._move_selected(-1))
            menu.addAction("Move down", lambda: self._move_selected(+1))
            menu.addSeparator()
        if len(selected) >= 2:
            menu.addAction("Combine / scale…", self._on_combine_clicked)
            menu.addSeparator()
        menu.addAction(f"Export {len(selected)} item(s) as text…", self._export_selected_txt)
        menu.addSeparator()
        menu.addAction(f"Delete {len(selected)} item(s)", self._delete_selected)
        menu.exec(self.table.viewport().mapToGlobal(pos))

    def _export_selected_txt(self) -> None:
        """Write each selected spectrum as a two-column tab-separated .txt —
        the way derived spectra (baseline-subtracted, combined, …) get back
        OUT of the app for use elsewhere (Origin, notebooks, colleagues)."""
        selected = self._selected_spectra()
        if not selected:
            return
        if len(selected) == 1:
            sp = selected[0]
            path, _ = QFileDialog.getSaveFileName(self, "Export spectrum as…", f"{sp.title}.txt", "Text (*.txt);;CSV (*.csv)")
            if not path:
                return
            targets = [(sp, path)]
        else:
            folder = QFileDialog.getExistingDirectory(self, "Export selected spectra into folder…")
            if not folder:
                return
            targets = [(sp, os.path.join(folder, f"{sp.title}.txt")) for sp in selected]

        written, errors = 0, []
        for sp, path in targets:
            try:
                sep = "," if path.lower().endswith(".csv") else "\t"
                data = np.column_stack([np.asarray(sp.x, float), np.asarray(sp.y, float)])
                np.savetxt(path, data, delimiter=sep, header=f"{sp.title} (exported from PRISM)", comments="# ")
                written += 1
            except OSError as exc:
                errors.append(f"{sp.title}: {exc}")
        msg = f"Exported {written} file(s)."
        if errors:
            msg += "\nFailed: " + "; ".join(errors)
            QMessageBox.warning(self, "Export", msg)

    def clear_all(self) -> None:
        """Clear the whole library (the old app's 'Clear imports') — done
        through the same delete path, so it's undoable."""
        if len(self.library) == 0:
            return
        resp = QMessageBox.question(self, "Clear imports", f"Remove all {len(self.library)} spectra from the library? (Undo is available.)")
        if resp != QMessageBox.Yes:
            return
        batch = [(i, sp) for i, sp in enumerate(self.library.all())]
        for _, sp in batch:
            self.library.remove(sp.id)
        self.push_undo(("delete", batch))
        self._refresh_table()

    def push_undo(self, action: tuple) -> None:
        """Record an undoable library action (see _undo_stack's format).
        Also the entry point for other workspaces' undoable effects on the
        library (applied baselines, accepted mineral IDs), via the shell."""
        self._undo_stack.append(action)
        self.undo_btn.setEnabled(True)

    def _rename_selected(self) -> None:
        selected = self._selected_spectra()
        if len(selected) != 1:
            return
        from PySide6.QtWidgets import QInputDialog
        sp = selected[0]
        new_title, ok = QInputDialog.getText(self, "Rename", "New name:", text=sp.title)
        if ok and new_title.strip() and new_title.strip() != sp.title:
            self.push_undo(("rename", sp.id, sp.title))
            sp.title = new_title.strip()
            self._refresh_table()

    def _duplicate_selected(self) -> None:
        selected = self._selected_spectra()
        if len(selected) != 1:
            return
        sp = selected[0]
        copy_sp = Spectrum(
            id=Spectrum.new_id(), title=f"{sp.title}_copy", path=sp.path, kind=sp.kind,
            x=np.array(sp.x, float).copy(), y=np.array(sp.y, float).copy(),
            df=sp.df, meta=dict(sp.meta), status="derived",
        )
        self.library.add(copy_sp)
        self.push_undo(("add", [copy_sp.id]))
        self._refresh_table()

    def _move_selected(self, delta: int) -> None:
        selected = self._selected_spectra()
        if len(selected) != 1:
            return
        order = [s.id for s in self.library.all()]
        i = order.index(selected[0].id)
        j = i + delta
        if not (0 <= j < len(order)):
            return
        order[i], order[j] = order[j], order[i]
        self.library.reorder(order)
        self._refresh_table()
        self.table.selectRow(j)

    def _delete_selected(self) -> None:
        selected = self._selected_spectra()
        if not selected:
            return
        order = [s.id for s in self.library.all()]
        batch = [(order.index(sp.id), sp) for sp in selected]
        for _, sp in batch:
            self.library.remove(sp.id)
        self.push_undo(("delete", batch))
        self._refresh_table()

    def _undo(self) -> None:
        if not self._undo_stack:
            return
        action = self._undo_stack.pop()
        kind = action[0]
        if kind == "delete":
            batch = action[1]
            for position, sp in sorted(batch, key=lambda t: t[0]):
                self.library.add(sp)
            # Restore the original ordering as closely as possible.
            order = [s.id for s in self.library.all()]
            for position, sp in sorted(batch, key=lambda t: t[0]):
                order.remove(sp.id)
                order.insert(min(position, len(order)), sp.id)
            self.library.reorder(order)
        elif kind == "add":
            for sid in action[1]:
                if self.library.get(sid) is not None:
                    self.library.remove(sid)
        elif kind == "rename":
            sp = self.library.get(action[1])
            if sp is not None:
                sp.title = action[2]
        elif kind == "ident":
            sp = self.library.get(action[1])
            if sp is not None:
                old = action[2]
                if isinstance(old, dict) and "rruff_match" in old and "rruff_matches" in old:
                    # multi-phase envelope (iterative accept): restore both keys
                    for key in ("rruff_match", "rruff_matches"):
                        if old.get(key) is None:
                            sp.meta.pop(key, None)
                        else:
                            sp.meta[key] = old[key]
                elif old is None:
                    sp.meta.pop("rruff_match", None)
                    sp.meta.pop("rruff_matches", None)
                else:  # legacy single-match record
                    sp.meta["rruff_match"] = old
        elif kind == "xrd_ident":
            sp = self.library.get(action[1])
            if sp is not None:
                old = action[2] or {}
                for key in ("xrd_match", "xrd_matches"):
                    if old.get(key) is None:
                        sp.meta.pop(key, None)
                    else:
                        sp.meta[key] = old[key]
        self.undo_btn.setEnabled(bool(self._undo_stack))
        self._refresh_table()

    # Kept as an alias: the File-menu wiring and older tests used this name
    # when deletion was the only undoable action.
    _undo_delete = _undo

    def _on_custom_import_clicked(self) -> None:
        from core.qt_custom_import import CustomImportDialog
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Custom import", "", "All files (*.*)",
        )
        added = 0
        for path in paths:
            dlg = CustomImportDialog(self, path)
            if dlg.exec() and dlg.spectrum is not None:
                self.library.add(dlg.spectrum)
                added += 1
        if added:
            self._refresh_table()

    def _on_combine_clicked(self) -> None:
        selected = self._selected_spectra()
        if not selected:
            QMessageBox.information(self, "Combine", "Select one spectrum (to scale) or several (to sum/average/subtract).")
            return
        dlg = CombineDialog(self, selected)
        if dlg.exec():
            result = dlg.result_spectrum
            if result is not None:
                self.library.add(result)
                self.push_undo(("add", [result.id]))
                self._refresh_table()
                self.table.selectRow(self.table.rowCount() - 1)

    def _on_selection_changed(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        item = self.table.item(rows[0].row(), 0)
        spectrum_id = item.data(Qt.UserRole)
        spectrum = self.library.get(spectrum_id)
        if spectrum is None:
            return
        self.plot.ax.clear()
        self.plot.ax.plot(spectrum.x, spectrum.y, color="#3c6e71", lw=1.2)
        self.plot.ax.set_title(spectrum.title)
        self.plot.ax.grid(alpha=0.25)
        self.plot.figure.tight_layout()
        self.plot.canvas.draw_idle()
