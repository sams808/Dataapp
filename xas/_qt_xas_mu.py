"""
xas/_qt_xas_mu.py — internal implementation detail of qt_xas.py:
XasWorkspace's μ(E) Builder tab. Mixed into XasWorkspace, not meant to
be used standalone.
"""
from __future__ import annotations

import numpy as np

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

from core.qt_widgets import PlotWidget, to_float as _to_float
from xas.xas_science import Operation, Spectrum, build_mu, deglitch_3x_noise, deglitch_mu
from ._qt_xas_shared import COLORS, _to_int, for_each_selected_spectrum


class MuTabMixin:
    def _build_mu_tab(self) -> QWidget:
        w = QWidget()
        layout = QHBoxLayout(w)

        ctrl = QWidget()
        ctrl.setMaximumWidth(320)
        ctrl_layout = QVBoxLayout(ctrl)

        ctrl_layout.addWidget(QLabel("I0 selection"))
        self.mu_i0_combo = QComboBox()
        self.mu_i0_combo.currentIndexChanged.connect(self._preview_mu)
        ctrl_layout.addWidget(self.mu_i0_combo)

        ctrl_layout.addWidget(QLabel("It spectra (multi-select)"))
        self.mu_it_list = QListWidget()
        self.mu_it_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.mu_it_list.itemSelectionChanged.connect(self._preview_mu)
        ctrl_layout.addWidget(self.mu_it_list, 1)

        log_row = QHBoxLayout()
        log_row.addWidget(QLabel("log"))
        self.mu_log_combo = QComboBox()
        self.mu_log_combo.addItems(["ln", "log10"])
        self.mu_log_combo.currentTextChanged.connect(self._preview_mu)
        log_row.addWidget(self.mu_log_combo)
        ctrl_layout.addLayout(log_row)

        self.mu_deglitch_check = QCheckBox("Deglitch")
        ctrl_layout.addWidget(self.mu_deglitch_check)
        self.mu_deglitch_method_combo = QComboBox()
        self.mu_deglitch_method_combo.addItems(["noise-calibrated (recommended)", "rolling z-score"])
        self.mu_deglitch_method_combo.currentIndexChanged.connect(self._on_deglitch_method_changed)
        ctrl_layout.addWidget(self.mu_deglitch_method_combo)

        degl_row = QHBoxLayout()
        degl_row.addWidget(QLabel("z"))
        self.mu_deglitch_z_edit = QLineEdit("6.0")
        self.mu_deglitch_z_edit.setMaximumWidth(50)
        degl_row.addWidget(self.mu_deglitch_z_edit)
        degl_row.addWidget(QLabel("window"))
        self.mu_deglitch_window_edit = QLineEdit("21")
        self.mu_deglitch_window_edit.setMaximumWidth(50)
        degl_row.addWidget(self.mu_deglitch_window_edit)
        ctrl_layout.addLayout(degl_row)

        degl_row2 = QHBoxLayout()
        degl_row2.addWidget(QLabel("noise x"))
        self.mu_deglitch_noise_mult_edit = QLineEdit("3.0")
        self.mu_deglitch_noise_mult_edit.setMaximumWidth(50)
        degl_row2.addWidget(self.mu_deglitch_noise_mult_edit)
        degl_row2.addWidget(QLabel("local window"))
        self.mu_deglitch_local_window_edit = QLineEdit("5")
        self.mu_deglitch_local_window_edit.setMaximumWidth(50)
        degl_row2.addWidget(self.mu_deglitch_local_window_edit)
        ctrl_layout.addLayout(degl_row2)
        self._on_deglitch_method_changed()

        preview_btn = QPushButton("Preview μ")
        preview_btn.clicked.connect(self._preview_mu)
        ctrl_layout.addWidget(preview_btn)
        compute_btn = QPushButton("Compute μ → new objects")
        compute_btn.setObjectName("Primary")
        compute_btn.clicked.connect(self.compute_mu_selected)
        ctrl_layout.addWidget(compute_btn)
        ctrl_layout.addStretch(1)
        layout.addWidget(ctrl)

        self.mu_plot = PlotWidget(figsize=(7, 5))
        layout.addWidget(self.mu_plot, 1)
        return w

    def _on_deglitch_method_changed(self) -> None:
        is_noise_method = self.mu_deglitch_method_combo.currentIndex() == 0
        self.mu_deglitch_z_edit.setEnabled(not is_noise_method)
        self.mu_deglitch_window_edit.setEnabled(not is_noise_method)
        self.mu_deglitch_noise_mult_edit.setEnabled(is_noise_method)
        self.mu_deglitch_local_window_edit.setEnabled(is_noise_method)

    def _mu_deglitch_params(self):
        enabled = self.mu_deglitch_check.isChecked()
        method = "noise" if self.mu_deglitch_method_combo.currentIndex() == 0 else "zscore"
        z = _to_float(self.mu_deglitch_z_edit.text(), 6.0)
        window = _to_int(self.mu_deglitch_window_edit.text(), 21)
        noise_multiple = _to_float(self.mu_deglitch_noise_mult_edit.text(), 3.0)
        local_window = _to_int(self.mu_deglitch_local_window_edit.text(), 5)
        return enabled, method, z, window, noise_multiple, local_window

    def _build_mu(self, i0: Spectrum, it: Spectrum) -> np.ndarray:
        mu = build_mu(i0_energy=i0.energy, i0=i0.y, it_energy=it.energy, it=it.y, log_mode=self.mu_log_combo.currentText())
        enabled, method, z, window, noise_multiple, local_window = self._mu_deglitch_params()
        if enabled:
            if method == "noise":
                mu, _flagged = deglitch_3x_noise(mu, local_window=local_window, noise_multiple=noise_multiple)
            else:
                mu = deglitch_mu(mu, z=z, window=window)
        return mu

    def _preview_mu(self) -> None:
        i0_name = self.mu_i0_combo.currentText()
        if not i0_name:
            self.mu_plot.clear("μ preview")
            return
        i0 = self.store.find_by_name(i0_name)
        it_items = self.mu_it_list.selectedItems()
        if i0 is None or not it_items:
            self.mu_plot.clear("μ preview")
            return
        it = self.store.find_by_name(it_items[0].text())
        if it is None:
            return
        try:
            mu = self._build_mu(i0, it)
        except Exception as exc:
            QMessageBox.critical(self, "μ preview error", str(exc))
            return
        ax = self.mu_plot.ax
        ax.clear()
        ax.plot(i0.energy, mu, lw=1.2, color=COLORS[0], label=f"μ from {it.name}")
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("arb.")
        ax.set_title(f"μ preview — I0={i0.name}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        self.mu_plot.figure.tight_layout()
        self.mu_plot.canvas.draw_idle()

    def compute_mu_selected(self) -> None:
        i0_name = self.mu_i0_combo.currentText()
        if not i0_name:
            QMessageBox.warning(self, "μ builder", "Select an I0 spectrum.")
            return
        i0 = self.store.find_by_name(i0_name)
        if i0 is None:
            return
        it_items = self.mu_it_list.selectedItems()
        if not it_items:
            QMessageBox.warning(self, "μ builder", "Select at least one It spectrum.")
            return

        def _process(it):
            mu = self._build_mu(i0, it)
            sp_mu = it.copy(new_name=f"{it.name}_mu", new_kind="mu")
            sp_mu.energy = np.asarray(i0.energy, float)
            sp_mu.y = np.asarray(mu, float)
            sp_mu.label = it.label
            sp_mu.e0 = it.e0
            enabled, method, z, window, noise_multiple, local_window = self._mu_deglitch_params()
            degl_params = {"deglitch": enabled, "deglitch_method": method}
            if enabled:
                degl_params.update({"deglitch_z": z, "deglitch_window": window} if method == "zscore"
                                   else {"deglitch_noise_multiple": noise_multiple, "deglitch_local_window": local_window})
            sp_mu.history.append(Operation("mu_builder", {"I0": i0.name, "It": it.name, "log": self.mu_log_combo.currentText(), **degl_params}))
            self.store.add(sp_mu)
            return sp_mu

        last = for_each_selected_spectrum(self, self.store, it_items, _process, "μ builder error")
        self._refresh_all()
        if last is not None:
            self._set_status(f"Computed μ for {len(it_items)} It spectrum/spectra.")
