"""
xas/_qt_xas_objects.py — internal implementation detail of qt_xas.py:
XasWorkspace's import/object-list management (object table, per-tab
selector lists, import ZIP/CSV/.prj, rename/duplicate/delete/export
context menu). Mixed into XasWorkspace, not meant to be used standalone.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog, QListWidget, QListWidgetItem, QMenu, QMessageBox,
    QTableWidgetItem,
)

from xas.xas_science import (
    Operation,
    Spectrum,
    _classify_kind_from_name,
    _extract_energy_angle_signal,
    _uid,
    edge_text,
    infer_edge_label_from_roi_scaled,
    read_athena_prj,
    read_csv_dataset,
    read_easyxafs_zip,
)


class ObjectListMixin:
    def _spectrum_from_record(self, rec: Dict[str, Any]) -> Spectrum:
        angle, energy, signal, cols = _extract_energy_angle_signal(rec["df"])
        kind = _classify_kind_from_name(rec["name"])
        scan_def = rec.get("scan_def", {}) or {}
        label, e0 = infer_edge_label_from_roi_scaled(energy, signal, scan_def)
        return Spectrum(
            sid=_uid("sp"), name=rec["name"], kind=kind, energy=energy, y=signal,
            angle=angle if np.isfinite(angle).any() else None, units="counts/s", label=label, e0=e0,
            meta={"source": rec.get("source", ""), "columns": cols, "scan_def": scan_def, "metadata": rec.get("metadata", {})},
        )

    def import_zips(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Select EasyXAFS ZIP(s)", "", "ZIP files (*.zip);;All files (*.*)")
        if not paths:
            return
        try:
            n = 0
            for zp in paths:
                for rec in read_easyxafs_zip(zp):
                    sp = self._spectrum_from_record(rec)
                    sp.history.append(Operation("import", {"source": rec.get("source", "")}))
                    self.store.add(sp)
                    n += 1
            self._refresh_all()
            self._set_status(f"Imported {n} dataset(s) from ZIP(s).")
        except Exception as exc:
            QMessageBox.critical(self, "Import ZIP error", str(exc))

    def import_csvs(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Select CSV(s)", "", "CSV files (*.csv);;All files (*.*)")
        if not paths:
            return
        try:
            n = 0
            for p in paths:
                rec = read_csv_dataset(p)
                sp = self._spectrum_from_record(rec)
                sp.history.append(Operation("import", {"source": rec.get("source", "")}))
                self.store.add(sp)
                n += 1
            self._refresh_all()
            self._set_status(f"Imported {n} CSV dataset(s).")
        except Exception as exc:
            QMessageBox.critical(self, "Import CSV error", str(exc))

    def import_prj(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Athena project (.prj)", "", "Athena project (*.prj);;All files (*.*)")
        if not path:
            return
        try:
            specs = read_athena_prj(path)
            for sp in specs:
                self.store.add(sp)
            self._refresh_all()
            self._set_status(f"Imported {len(specs)} group(s) from .prj.")
        except Exception as exc:
            QMessageBox.critical(self, "Import .prj error", str(exc))

    def clear_all(self) -> None:
        self.store.clear()
        self.selected_sid = None
        self._refresh_all()
        self._set_status("Cleared all spectra.")

    def load_initial_spectra(self, specs: List[Spectrum]) -> None:
        for sp in specs:
            self.store.add(sp)
        self._refresh_all()

    # ------------------------------------------------------------------
    def _refresh_all(self) -> None:
        self._refresh_table()
        self._refresh_lists()

    def _refresh_table(self) -> None:
        specs = self.store.all()
        self.table.setRowCount(len(specs))
        for row, sp in enumerate(specs):
            e0_txt = "" if sp.e0 is None or not np.isfinite(sp.e0) else f"{sp.e0:.1f}"
            er_txt = f"{np.nanmin(sp.energy):.1f}–{np.nanmax(sp.energy):.1f}" if sp.energy.size else ""
            name_item = QTableWidgetItem(sp.name)
            name_item.setData(Qt.UserRole, sp.sid)
            self.table.setItem(row, 0, name_item)
            self.table.setItem(row, 1, QTableWidgetItem(sp.kind))
            self.table.setItem(row, 2, QTableWidgetItem(edge_text(sp.label)))
            self.table.setItem(row, 3, QTableWidgetItem(e0_txt))
            self.table.setItem(row, 4, QTableWidgetItem(er_txt))
        self.table.resizeColumnsToContents()

    def _refresh_lists(self) -> None:
        all_names = [s.name for s in self.store.all()]
        i0_names = [s.name for s in self.store.all() if s.kind in ("I0", "fit", "I0_fit")]
        it_names = [s.name for s in self.store.all() if s.kind == "It"]
        mu_names = [s.name for s in self.store.all() if s.kind == "mu"]

        self.mu_i0_combo.blockSignals(True)
        current_i0 = self.mu_i0_combo.currentText()
        self.mu_i0_combo.clear()
        self.mu_i0_combo.addItems(i0_names)
        if current_i0 in i0_names:
            self.mu_i0_combo.setCurrentText(current_i0)
        self.mu_i0_combo.blockSignals(False)

        for combo in (self.sm_target_combo, self.ang_before_combo, self.ang_after_combo):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(all_names)
            if current in all_names:
                combo.setCurrentText(current)
            combo.blockSignals(False)

        self._fill_list(self.mu_it_list, it_names)
        self._fill_list(self.norm_mu_list, mu_names)
        self._fill_list(self.edge_list, all_names)
        self._fill_list(self.analysis_list, all_names)
        self._fill_list(self.lcf_targets_list, all_names)
        self._fill_list(self.lcf_refs_list, all_names)
        self._refresh_lcf_required_list()

    def _refresh_lcf_required_list(self) -> None:
        # "Always include" options are limited to whatever's currently
        # selected as a reference -- keeps it from listing names that
        # can't actually be used as a required reference in this run.
        selected_refs = [item.text() for item in self.lcf_refs_list.selectedItems()]
        self._fill_list(self.lcf_required_list, selected_refs)

    @staticmethod
    def _fill_list(listwidget: QListWidget, names: List[str]) -> None:
        selected = {listwidget.item(i).text() for i in range(listwidget.count()) if listwidget.item(i).isSelected()}
        listwidget.clear()
        for n in names:
            item = QListWidgetItem(n)
            listwidget.addItem(item)
            if n in selected:
                item.setSelected(True)

    def _on_selection_changed(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            self.selected_sid = None
            return
        item = self.table.item(rows[0].row(), 0)
        self.selected_sid = item.data(Qt.UserRole)
        self._plot_selected_preview()

    def _on_table_context_menu(self, pos) -> None:
        item = self.table.itemAt(pos)
        if item is None:
            return
        row = item.row()
        sid = self.table.item(row, 0).data(Qt.UserRole)
        self.table.selectRow(row)
        self.selected_sid = sid

        menu = QMenu(self)
        menu.addAction("Rename…", self.rename_selected)
        menu.addAction("Duplicate", self.duplicate_selected)
        menu.addSeparator()
        menu.addAction("Delete", self.delete_selected)
        menu.addSeparator()
        menu.addAction("Export selected as .dat", self.export_athena_dat)
        menu.exec(self.table.viewport().mapToGlobal(pos))

    def rename_selected(self) -> None:
        if self.selected_sid is None:
            return
        sp = self.store.get(self.selected_sid)
        from PySide6.QtWidgets import QInputDialog
        new_name, ok = QInputDialog.getText(self, "Rename", "New name:", text=sp.name)
        if ok and new_name.strip():
            sp.name = new_name.strip()
            self._refresh_all()

    def duplicate_selected(self) -> None:
        if self.selected_sid is None:
            return
        sp = self.store.get(self.selected_sid)
        sp2 = sp.copy(new_name=f"{sp.name}_copy")
        sp2.history.append(Operation("duplicate", {"from": sp.sid}))
        self.store.add(sp2)
        self._refresh_all()

    def delete_selected(self) -> None:
        if self.selected_sid is None:
            return
        sp = self.store.get(self.selected_sid)
        resp = QMessageBox.question(self, "Delete", f"Delete '{sp.name}'?")
        if resp == QMessageBox.Yes:
            self.store.remove(self.selected_sid)
            self.selected_sid = None
            self._refresh_all()
