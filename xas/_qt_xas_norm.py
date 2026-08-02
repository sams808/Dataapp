"""
xas/_qt_xas_norm.py — internal implementation detail of qt_xas.py:
XasWorkspace's Normalization / EXAFS tab (Larch pre_edge/autobk/xftf).
Mixed into XasWorkspace, not meant to be used standalone.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

from core.qt_widgets import PlotWidget, to_float as _to_float
from xas.xas_science import Operation, larch_exafs_pipeline, larch_normalize
from ._qt_xas_shared import _to_int


class NormTabMixin:
    def _build_norm_tab(self) -> QWidget:
        w = QWidget()
        layout = QHBoxLayout(w)

        ctrl = QWidget()
        ctrl.setMaximumWidth(360)
        ctrl_layout = QVBoxLayout(ctrl)

        e0_row = QHBoxLayout()
        e0_row.addWidget(QLabel("E0 method"))
        self.norm_e0_combo = QComboBox()
        self.norm_e0_combo.addItems(["larch", "deriv", "manual"])
        e0_row.addWidget(self.norm_e0_combo)
        e0_row.addWidget(QLabel("manual"))
        self.norm_e0_manual_edit = QLineEdit()
        self.norm_e0_manual_edit.setMaximumWidth(70)
        e0_row.addWidget(self.norm_e0_manual_edit)
        ctrl_layout.addLayout(e0_row)

        grid_row1 = QHBoxLayout()
        grid_row1.addWidget(QLabel("pre1"))
        self.norm_pre1_edit = QLineEdit("-150")
        grid_row1.addWidget(self.norm_pre1_edit)
        grid_row1.addWidget(QLabel("pre2"))
        self.norm_pre2_edit = QLineEdit("-50")
        grid_row1.addWidget(self.norm_pre2_edit)
        ctrl_layout.addLayout(grid_row1)

        grid_row2 = QHBoxLayout()
        grid_row2.addWidget(QLabel("norm1"))
        self.norm_norm1_edit = QLineEdit("30")
        grid_row2.addWidget(self.norm_norm1_edit)
        grid_row2.addWidget(QLabel("norm2"))
        self.norm_norm2_edit = QLineEdit("150")
        grid_row2.addWidget(self.norm_norm2_edit)
        grid_row2.addWidget(QLabel("nnorm"))
        self.norm_nnorm_edit = QLineEdit("1")
        grid_row2.addWidget(self.norm_nnorm_edit)
        ctrl_layout.addLayout(grid_row2)

        self.norm_smooth_check = QCheckBox("Smooth for E0/derivative only")
        self.norm_smooth_check.setChecked(True)
        ctrl_layout.addWidget(self.norm_smooth_check)

        normalize_btn = QPushButton("Normalize selected μ → new objects")
        normalize_btn.setObjectName("Primary")
        normalize_btn.clicked.connect(self.normalize_selected)
        ctrl_layout.addWidget(normalize_btn)

        ctrl_layout.addWidget(QLabel("EXAFS / FT (Larch autobk + xftf)"))
        ex_row1 = QHBoxLayout()
        ex_row1.addWidget(QLabel("rbkg"))
        self.exafs_rbkg_edit = QLineEdit("1.0")
        ex_row1.addWidget(self.exafs_rbkg_edit)
        ex_row1.addWidget(QLabel("kmin"))
        self.exafs_kmin_edit = QLineEdit("0")
        ex_row1.addWidget(self.exafs_kmin_edit)
        ex_row1.addWidget(QLabel("kmax"))
        self.exafs_kmax_edit = QLineEdit("15")
        ex_row1.addWidget(self.exafs_kmax_edit)
        ctrl_layout.addLayout(ex_row1)

        ex_row2 = QHBoxLayout()
        ex_row2.addWidget(QLabel("dk"))
        self.exafs_dk_edit = QLineEdit("0.1")
        ex_row2.addWidget(self.exafs_dk_edit)
        ex_row2.addWidget(QLabel("k-weight"))
        self.exafs_kweight_edit = QLineEdit("2")
        ex_row2.addWidget(self.exafs_kweight_edit)
        ctrl_layout.addLayout(ex_row2)

        ex_row3 = QHBoxLayout()
        ex_row3.addWidget(QLabel("window"))
        self.exafs_window_combo = QComboBox()
        self.exafs_window_combo.addItems(["hanning", "kaiser", "parzen", "welch", "sine", "gaussian"])
        ex_row3.addWidget(self.exafs_window_combo)
        ex_row3.addWidget(QLabel("rmax_out"))
        self.exafs_rmax_edit = QLineEdit("10")
        ex_row3.addWidget(self.exafs_rmax_edit)
        ctrl_layout.addLayout(ex_row3)

        exafs_btn = QPushButton("Compute χ(k) + FT → new objects")
        exafs_btn.clicked.connect(self.exafs_selected)
        ctrl_layout.addWidget(exafs_btn)

        ctrl_layout.addWidget(QLabel("Select μ spectra"))
        self.norm_mu_list = QListWidget()
        self.norm_mu_list.setSelectionMode(QListWidget.ExtendedSelection)
        ctrl_layout.addWidget(self.norm_mu_list, 1)

        layout.addWidget(ctrl)
        self.norm_plot = PlotWidget(figsize=(7, 5))
        layout.addWidget(self.norm_plot, 1)
        return w

    def _norm_params(self):
        e0_method = self.norm_e0_combo.currentText()
        e0_manual = _to_float(self.norm_e0_manual_edit.text()) if e0_method == "manual" else None
        pre1 = _to_float(self.norm_pre1_edit.text(), -150.0)
        pre2 = _to_float(self.norm_pre2_edit.text(), -50.0)
        norm1 = _to_float(self.norm_norm1_edit.text(), 30.0)
        norm2 = _to_float(self.norm_norm2_edit.text(), 150.0)
        nnorm = _to_int(self.norm_nnorm_edit.text(), 1)
        smooth_for_e0 = ("savitzky-golay", {"window": 11, "poly": 3}) if self.norm_smooth_check.isChecked() else None
        return e0_method, e0_manual, pre1, pre2, norm1, norm2, nnorm, smooth_for_e0

    def normalize_selected(self) -> None:
        items = self.norm_mu_list.selectedItems()
        if not items:
            QMessageBox.warning(self, "Normalization", "Select μ spectra.")
            return
        e0_method, e0_manual, pre1, pre2, norm1, norm2, nnorm, smooth_for_e0 = self._norm_params()

        last = None
        for item in items:
            sp = self.store.find_by_name(item.text())
            if sp is None:
                continue
            try:
                out = larch_normalize(sp.energy, sp.y, e0_method=e0_method, e0_manual=e0_manual, pre1=pre1, pre2=pre2, norm1=norm1, norm2=norm2, nnorm=nnorm, smooth_for_e0=smooth_for_e0)
            except Exception as exc:
                QMessageBox.critical(self, "Normalization error", str(exc))
                continue
            sp_norm = sp.copy(new_name=f"{sp.name}_norm", new_kind="norm")
            sp_norm.y = out["norm"]; sp_norm.e0 = out["e0"]
            sp_norm.history.append(Operation("normalize", {"e0_method": e0_method, "e0": out["e0"]}))
            sp_flat = sp.copy(new_name=f"{sp.name}_flat", new_kind="flat")
            sp_flat.y = out["flat"]; sp_flat.e0 = out["e0"]
            sp_flat.history.append(Operation("normalize_flat", {"e0": out["e0"]}))
            self.store.add(sp_norm); self.store.add(sp_flat)
            last = (sp, out)

        self._refresh_all()
        if last is not None:
            sp, out = last
            ax = self.norm_plot.ax
            ax.clear()
            ax.plot(sp.energy, sp.y, lw=1.1, label="μ(E)")
            ax.plot(sp.energy, out["norm"], lw=1.1, label="norm")
            ax.plot(sp.energy, out["deriv"], lw=1.0, label="dμ/dE", alpha=0.6)
            for val in (out["anchors"].get(k) for k in ("pre1", "pre2", "norm1", "norm2")):
                if val is not None:
                    ax.axvline(float(val), ls="--", lw=1.0, alpha=0.7)
            ax.set_xlabel("Energy (eV)"); ax.set_ylabel("arb.")
            ax.set_title(f"{sp.label} — E0={out['e0']:.2f}")
            ax.legend(fontsize=8); ax.grid(alpha=0.25)
            self.norm_plot.figure.tight_layout()
            self.norm_plot.canvas.draw_idle()

    def exafs_selected(self) -> None:
        items = self.norm_mu_list.selectedItems()
        if not items:
            QMessageBox.warning(self, "EXAFS/FT", "Select μ spectra.")
            return
        e0_method, e0_manual, pre1, pre2, norm1, norm2, nnorm, smooth_for_e0 = self._norm_params()
        rbkg = _to_float(self.exafs_rbkg_edit.text(), 1.0)
        kmin = _to_float(self.exafs_kmin_edit.text(), 0.0)
        kmax = _to_float(self.exafs_kmax_edit.text(), 15.0)
        dk = _to_float(self.exafs_dk_edit.text(), 0.1)
        kweight = _to_int(self.exafs_kweight_edit.text(), 2)
        window = self.exafs_window_combo.currentText()
        rmax_out = _to_float(self.exafs_rmax_edit.text(), 10.0)

        last = None
        for item in items:
            sp = self.store.find_by_name(item.text())
            if sp is None:
                continue
            try:
                out = larch_exafs_pipeline(
                    sp.energy, sp.y, e0_method=e0_method, e0_manual=e0_manual, pre1=pre1, pre2=pre2,
                    norm1=norm1, norm2=norm2, nnorm=nnorm, rbkg=rbkg, kmin=kmin, kmax=kmax, dk=dk,
                    kweight=kweight, window=window, rmax_out=rmax_out, smooth_for_e0=smooth_for_e0,
                )
            except Exception as exc:
                QMessageBox.critical(self, "EXAFS/FT error", str(exc))
                continue

            sp_norm = sp.copy(new_name=f"{sp.name}_norm", new_kind="norm"); sp_norm.y = out["norm"]; sp_norm.e0 = out["e0"]
            sp_norm.history.append(Operation("normalize", {"e0_method": e0_method}))
            sp_flat = sp.copy(new_name=f"{sp.name}_flat", new_kind="flat"); sp_flat.y = out["flat"]; sp_flat.e0 = out["e0"]
            sp_flat.history.append(Operation("normalize_flat", {"e0": out["e0"]}))
            self.store.add(sp_norm); self.store.add(sp_flat)

            sp_chi = sp.copy(new_name=f"{sp.name}_chi", new_kind="chi(k)"); sp_chi.energy = out["k"]; sp_chi.y = out["chi"]; sp_chi.e0 = out["e0"]
            sp_chi.history.append(Operation("autobk", {"rbkg": rbkg, "kmin": kmin, "kmax": kmax, "dk": dk}))
            self.store.add(sp_chi)

            sp_chikw = sp.copy(new_name=f"{sp.name}_chi_k{kweight}", new_kind=f"chi(k)*k^{kweight}")
            sp_chikw.energy = out["k"]; sp_chikw.y = out["chi_kw"]; sp_chikw.e0 = out["e0"]
            sp_chikw.history.append(Operation("kweight", {"kweight": kweight}))
            self.store.add(sp_chikw)

            sp_ft = sp.copy(new_name=f"{sp.name}_FTmag", new_kind="FT|chi|"); sp_ft.energy = out["r"]; sp_ft.y = out["chir_mag"]; sp_ft.e0 = out["e0"]
            sp_ft.history.append(Operation("xftf", {"kmin": kmin, "kmax": kmax, "dk": dk, "kweight": kweight, "window": window, "rmax_out": rmax_out}))
            self.store.add(sp_ft)
            last = (sp, out, kweight)

        self._refresh_all()
        if last is not None:
            sp, out, kweight = last
            ax = self.norm_plot.ax
            ax.clear()
            ax.plot(out["k"], out["chi"], lw=1.1, label="χ(k)")
            ax.plot(out["k"], out["chi_kw"], lw=1.1, label=f"χ(k)*k^{kweight}")
            ax.plot(out["r"], out["chir_mag"], lw=1.1, label="|FT|")
            ax.set_xlabel("k (1/Å) / R (Å)"); ax.set_ylabel("arb.")
            ax.set_title(f"{sp.label} — E0={out['e0']:.2f}")
            ax.legend(fontsize=8); ax.grid(alpha=0.25)
            self.norm_plot.figure.tight_layout()
            self.norm_plot.canvas.draw_idle()
