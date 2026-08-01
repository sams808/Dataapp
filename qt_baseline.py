"""
qt_baseline.py — baseline subtraction workspace, the Qt successor to the
old Tk BaselineParamWindow workflow (the biggest remaining gap vs the old
notebook-era pipelines). Thin layer over baseline_science.py (rampy).

Per-spectrum settings (method, params, ROI text) persist in a
PerItemSettingsStore keyed by Spectrum.id — the structural fix for the
original dual-dict state-bleed bug the audit found in the Tk version.
"""
from __future__ import annotations

from typing import Dict, List, Optional


from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMessageBox, QPushButton, QSplitter, QVBoxLayout,
    QWidget,
)

from baseline_science import BASELINE_METHODS, BASELINE_PARAM_DEFS, compute_baseline, parse_roi_text
from core.qt_models import Spectrum, SpectrumLibrary
from core.qt_settings_store import PerItemSettingsStore
from core.qt_widgets import PlotWidget


def _default_settings() -> Dict[str, str]:
    return {"method": "arPLS", "roi_text": "", "p0": "", "p1": ""}


class BaselineWorkspace(QWidget):
    def __init__(self, parent: Optional[QWidget] = None, library: Optional[SpectrumLibrary] = None,
                 on_derived_added=None):
        super().__init__(parent)
        self.library = library if library is not None else SpectrumLibrary()
        # Shell-provided callback taking the list of newly created spectrum
        # ids, so batch-applied baselines join the Library's undo stack.
        self.on_derived_added = on_derived_added
        self.settings: PerItemSettingsStore[Dict[str, str]] = PerItemSettingsStore(_default_settings)
        self._current_id: Optional[str] = None
        self._last_preview = None  # (x, y_sub, base)
        self._span_selector = None  # matplotlib SpanSelector while pick mode is active
        self._build_ui()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        splitter = QSplitter()
        root.addWidget(splitter)

        left = QWidget()
        left.setObjectName("Card")
        left.setMaximumWidth(330)
        left_layout = QVBoxLayout(left)

        left_layout.addWidget(QLabel("Spectra (multi-select for batch apply)"))
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.file_list.itemSelectionChanged.connect(self._on_selection_changed)
        left_layout.addWidget(self.file_list, 1)

        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(BASELINE_METHODS)
        self.method_combo.currentTextChanged.connect(self._rebuild_param_fields)
        method_row.addWidget(self.method_combo, 1)
        left_layout.addLayout(method_row)

        self.param_labels = [QLabel(""), QLabel("")]
        self.param_edits = [QLineEdit(), QLineEdit()]
        params_row = QHBoxLayout()
        for lbl, edit in zip(self.param_labels, self.param_edits):
            edit.setMaximumWidth(80)
            params_row.addWidget(lbl)
            params_row.addWidget(edit)
        params_row.addStretch(1)
        left_layout.addLayout(params_row)
        self._rebuild_param_fields(self.method_combo.currentText())

        left_layout.addWidget(QLabel("Baseline regions (x-ranges the baseline\nis fitted through, e.g. 100-400; 1800-2600)"))
        self.roi_edit = QLineEdit()
        self.roi_edit.setPlaceholderText("required for poly/unispline/rubberband")
        left_layout.addWidget(self.roi_edit)

        roi_btn_row = QHBoxLayout()
        self.pick_roi_btn = QPushButton("Add region by dragging")
        self.pick_roi_btn.setCheckable(True)
        self.pick_roi_btn.toggled.connect(self._on_pick_roi_toggled)
        roi_btn_row.addWidget(self.pick_roi_btn)
        clear_roi_btn = QPushButton("Clear")
        clear_roi_btn.setMaximumWidth(60)
        clear_roi_btn.clicked.connect(lambda: self.roi_edit.clear())
        roi_btn_row.addWidget(clear_roi_btn)
        left_layout.addLayout(roi_btn_row)

        preview_btn = QPushButton("Preview")
        preview_btn.clicked.connect(self.preview)
        left_layout.addWidget(preview_btn)

        apply_btn = QPushButton("Apply → new spectrum (_bl)")
        apply_btn.setObjectName("Primary")
        apply_btn.clicked.connect(self.apply_selected)
        left_layout.addWidget(apply_btn)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)
        splitter.addWidget(left)

        right = QWidget()
        right.setObjectName("Card")
        right_layout = QVBoxLayout(right)
        self.plot = PlotWidget(figsize=(7, 5.5))
        right_layout.addWidget(self.plot)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

    def _rebuild_param_fields(self, method: str) -> None:
        defs = BASELINE_PARAM_DEFS.get(method, [])
        for i, (lbl, edit) in enumerate(zip(self.param_labels, self.param_edits)):
            if i < len(defs):
                _, label, default = defs[i]
                lbl.setText(label)
                edit.setText(default)
                lbl.show()
                edit.show()
            else:
                lbl.hide()
                edit.hide()

    # ------------------------------------------------------------------
    def set_spectra(self, spectrum_ids: List[str]) -> None:
        selected = {self.file_list.item(i).data(Qt.UserRole)
                    for i in range(self.file_list.count()) if self.file_list.item(i).isSelected()}
        self.file_list.blockSignals(True)
        self.file_list.clear()
        for sid in spectrum_ids:
            sp = self.library.get(sid)
            if sp is None:
                continue
            item = QListWidgetItem(sp.title)
            item.setData(Qt.UserRole, sid)
            self.file_list.addItem(item)
            if sid in selected:
                item.setSelected(True)
        self.file_list.blockSignals(False)
        # Arriving on the page must show a spectrum immediately (user
        # request): auto-select the first one when nothing is selected yet,
        # and draw the raw curve without waiting for a Preview click.
        if not self.file_list.selectedItems() and self.file_list.count():
            self.file_list.item(0).setSelected(True)  # unblocked -> selection hook runs
        else:
            self.plot.request_redraw(self.render_raw)

    def _selected_spectra(self):
        out = []
        for item in self.file_list.selectedItems():
            sp = self.library.get(item.data(Qt.UserRole))
            if sp is not None:
                out.append(sp)
        return out

    def _on_selection_changed(self) -> None:
        # Persist the outgoing spectrum's settings, restore the incoming one's.
        if self._current_id is not None:
            self._store_settings(self._current_id)
        selected = self._selected_spectra()
        self._current_id = selected[0].id if selected else None
        if self._current_id is not None and self.settings.has(self._current_id):
            self._load_settings(self._current_id)
        self.plot.request_redraw(self.render_raw)

    def render_raw(self) -> None:
        """The immediate raw view (page entry / selection change) — Preview
        then adds the computed baseline on top."""
        selected = self._selected_spectra()
        fig = self.plot.figure
        fig.clf()
        ax = fig.add_subplot(111)
        if selected:
            sp = selected[0]
            ax.plot(sp.x, sp.y, lw=1.0, color="0.3", label="raw")
            ax.set_title(sp.title, fontsize=10)
            ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        ax.set_xlabel("x")
        fig.tight_layout()
        self.plot.canvas.draw_idle()
        if self.pick_roi_btn.isChecked():
            self._attach_span_selector()  # the redraw replaced the axes it was bound to

    def _store_settings(self, sid: str) -> None:
        vals = [e.text() for e in self.param_edits]
        self.settings.set(sid, {
            "method": self.method_combo.currentText(),
            "roi_text": self.roi_edit.text(),
            "p0": vals[0], "p1": vals[1] if len(vals) > 1 else "",
        })

    def _load_settings(self, sid: str) -> None:
        s = self.settings.get(sid)
        self.method_combo.setCurrentText(s.get("method", "arPLS"))
        self.roi_edit.setText(s.get("roi_text", ""))
        for edit, key in zip(self.param_edits, ("p0", "p1")):
            if s.get(key):
                edit.setText(s[key])

    def _current_params(self):
        method = self.method_combo.currentText()
        defs = BASELINE_PARAM_DEFS.get(method, [])
        params = {}
        for i, (name, _label, default) in enumerate(defs):
            text = self.param_edits[i].text().strip() or default
            try:
                value = float(text)
            except ValueError:
                raise ValueError(f"Bad value for {name}: {text!r}")
            if name == "polynomial_order":
                value = int(value)
            params[name] = value
        return method, params

    # ------------------------------------------------------------------
    # Interactive ROI picking: drag a horizontal span on the plot to
    # append it to the regions field — no typing required.
    # ------------------------------------------------------------------
    def _on_pick_roi_toggled(self, active: bool) -> None:
        if not active:
            self._detach_span_selector()
            return
        axes = self.plot.figure.get_axes()
        if not axes:
            # Nothing plotted yet — draw the raw spectrum so there's
            # something to drag on.
            selected = self._selected_spectra()
            if not selected:
                QMessageBox.information(self, "Pick region", "Select a spectrum first.")
                self.pick_roi_btn.setChecked(False)
                return
            sp = selected[0]
            fig = self.plot.figure
            fig.clf()
            ax = fig.add_subplot(111)
            ax.plot(sp.x, sp.y, lw=1.0, color="0.3")
            ax.set_title(f"{sp.title} — drag to add baseline regions", fontsize=10)
            ax.grid(alpha=0.25)
            fig.tight_layout()
            self.plot.canvas.draw_idle()
        self._attach_span_selector()

    def _attach_span_selector(self) -> None:
        from matplotlib.widgets import SpanSelector
        axes = self.plot.figure.get_axes()
        if not axes:
            return
        self._detach_span_selector()
        self._span_selector = SpanSelector(
            axes[0], self._on_span_selected, "horizontal",
            useblit=False, props=dict(alpha=0.25, facecolor="#3c6e71"),
            interactive=False,
        )

    def _detach_span_selector(self) -> None:
        if self._span_selector is not None:
            try:
                self._span_selector.disconnect_events()
            except Exception:
                pass
            self._span_selector = None

    def _on_span_selected(self, lo: float, hi: float) -> None:
        if hi <= lo:
            return
        segment = f"{lo:.4g}-{hi:.4g}"
        existing = self.roi_edit.text().strip()
        self.roi_edit.setText(f"{existing}; {segment}" if existing else segment)
        self.status_label.setText(f"Added baseline region {segment}. Drag again to add more, or un-toggle to stop.")

    # ------------------------------------------------------------------
    def preview(self) -> None:
        selected = self._selected_spectra()
        if not selected:
            QMessageBox.warning(self, "Baseline", "Select a spectrum.")
            return
        sp = selected[0]
        try:
            method, params = self._current_params()
            roi = parse_roi_text(self.roi_edit.text())
            x, y_sub, base = compute_baseline(sp.x, sp.y, method=method, roi=roi, params=params)
        except Exception as exc:
            QMessageBox.critical(self, "Baseline error", str(exc))
            return
        self._store_settings(sp.id)
        self._last_preview = (sp.id, x, y_sub, base)

        self.plot.cancel_pending()  # this direct draw supersedes any queued entry render
        fig = self.plot.figure
        fig.clf()
        ax_top, ax_bottom = fig.subplots(2, 1, sharex=True)
        ax_top.plot(sp.x, sp.y, lw=1.0, color="0.3", label="raw")
        ax_top.plot(x, base, lw=1.4, color="crimson", label=f"baseline ({method})")
        roi_arr = roi if roi is not None else []
        for lo, hi in (roi_arr if roi is not None else []):
            ax_top.axvspan(lo, hi, color="#3c6e71", alpha=0.12)
        ax_top.legend(fontsize=8)
        ax_top.grid(alpha=0.25)
        ax_top.set_title(f"{sp.title}", fontsize=10)
        ax_bottom.plot(x, y_sub, lw=1.0, color="#3c6e71", label="subtracted")
        ax_bottom.axhline(0.0, lw=0.7, color="0.6")
        ax_bottom.legend(fontsize=8)
        ax_bottom.grid(alpha=0.25)
        ax_bottom.set_xlabel("x")
        fig.tight_layout()
        self.plot.canvas.draw_idle()
        if self.pick_roi_btn.isChecked():
            self._attach_span_selector()  # the redraw replaced the axes it was bound to
        self.status_label.setText(f"Previewed {method} baseline on '{sp.title}'.")

    def apply_selected(self) -> None:
        selected = self._selected_spectra()
        if not selected:
            QMessageBox.warning(self, "Baseline", "Select at least one spectrum.")
            return
        try:
            method, params = self._current_params()
            roi = parse_roi_text(self.roi_edit.text())
        except Exception as exc:
            QMessageBox.critical(self, "Baseline error", str(exc))
            return

        created_ids, errors = [], []
        for sp in selected:
            try:
                x, y_sub, base = compute_baseline(sp.x, sp.y, method=method, roi=roi, params=params)
            except Exception as exc:
                errors.append(f"{sp.title}: {exc}")
                continue
            self.settings.set(sp.id, {
                "method": method, "roi_text": self.roi_edit.text(),
                "p0": self.param_edits[0].text(), "p1": self.param_edits[1].text(),
            })
            derived = Spectrum(
                id=Spectrum.new_id(), title=f"{sp.title}_bl", path=sp.path, kind=sp.kind,
                x=x, y=y_sub, df=None,
                meta={"derived": f"baseline_{method}", "baseline_params": dict(params), "source": sp.title},
                status="derived",
            )
            self.library.add(derived)
            created_ids.append(derived.id)

        if created_ids and self.on_derived_added is not None:
            self.on_derived_added(created_ids)
        self.set_spectra([s.id for s in self.library.all()])
        msg = f"Created {len(created_ids)} baseline-subtracted spectrum/spectra (suffix _bl)."
        if errors:
            msg += "\nFailed: " + "; ".join(errors)
        self.status_label.setText(msg)
