"""
qt_xas.py — XAS/XANES/EXAFS processing ported to Qt (M11), built on
xas_science.py's SpectrumStore/Spectrum/Operation object model (already
identity-stable via `Spectrum.sid` — no redesign needed here, unlike
qt_models.SpectrumLibrary which replaced main.py's four-parallel-list
anti-pattern from scratch).

XasWorkspace itself is a slim shell: its 8 tabs (Preview, Pre-processing,
μ(E) Builder, Normalization/EXAFS, Analysis, Tools, Export, Sample mass)
plus object-list management each live in their own _qt_xas_*.py mixin
module, mixed together here via multiple inheritance so every method
still shares one `self` (store, selected_sid, widgets) exactly as before
the split — only the file layout changed, not the behavior.

Core slice ported faithfully from xas_processing_v10.py's XASUltimateApp:
object list (import ZIP/CSV/.prj, rename/duplicate/delete/export), Preview,
mu(E) Builder, Normalization + EXAFS/FT (Larch), Tools (edge definer),
Export (Athena .dat/.prj).

Two real bugs found and fixed in xas_science.py while building this (see
its own comments): (1) larch_normalize/larch_exafs_pipeline set pre1/pre2/
norm1/norm2/nnorm as Group attributes but never passed them as explicit
kwargs to pre_edge(), which only reads e0 that way — Larch silently used
its own auto-computed defaults instead for every call until now; (2)
compute_mu()'s deglitch/deglitch_z/deglitch_window parameters were dead
(accepted, never referenced) — exactly the gap the plan asked M11 to
"confirm." Fixed there; exposed here as real deglitch controls in the
mu(E) Builder tab (applied as an optional post-step after build_mu(), since
the interactive builder aligns two possibly-different-grid I0/It spectra
via build_mu()'s own interpolation — compute_mu() assumes a shared grid
already and isn't a drop-in replacement for that).

Athena-inspired additions (new "Analysis" tab, cheap/self-contained ones
only — PCA needs a new scikit-learn dependency better co-scoped with M16;
self-absorption correction is a substantial standalone physics feature):
merge/average repeat scans, difference spectra, linear combination fitting
(NNLS-based, 2+ references).

The Pre-processing tab (smoothing preview/apply, Bragg angle/energy
correction, Mode C interactive click-based feature alignment) and PCA
(Analysis tab, scikit-learn-based) were originally deferred but are both
now implemented below. Still genuinely missing, deliberately not
half-implemented: self-absorption correction (a substantial standalone
physics feature), the separate I0 baseline-"Fit" tab from the old Tk app,
and the CSV Builder export tool.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QSplitter, QTableWidget, QTabWidget,
    QVBoxLayout, QWidget,
)

from xas.xas_science import SpectrumStore
from ._qt_xas_analysis import AnalysisTabMixin
from ._qt_xas_export import ExportTabMixin
from ._qt_xas_mass import MassTabMixin
from ._qt_xas_mu import MuTabMixin
from ._qt_xas_norm import NormTabMixin
from ._qt_xas_objects import ObjectListMixin
from ._qt_xas_preproc import PreprocTabMixin
from ._qt_xas_preview import PreviewTabMixin
from ._qt_xas_tools import ToolsTabMixin


class XasWorkspace(QWidget, PreviewTabMixin, PreprocTabMixin, MuTabMixin, NormTabMixin,
                   AnalysisTabMixin, ToolsTabMixin, ExportTabMixin, MassTabMixin, ObjectListMixin):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.store = SpectrumStore()
        self.selected_sid: Optional[str] = None
        self._build_ui()
        self._refresh_all()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        splitter = QSplitter()
        root.addWidget(splitter)

        left = QWidget()
        left.setObjectName("Card")
        left.setMaximumWidth(340)
        left_layout = QVBoxLayout(left)

        import_row = QHBoxLayout()
        for label, handler in [
            ("ZIP…", self.import_zips), ("CSV…", self.import_csvs),
            (".prj…", self.import_prj), ("Clear", self.clear_all),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(handler)
            import_row.addWidget(btn)
        left_layout.addLayout(import_row)

        left_layout.addWidget(QLabel("Imported spectra (objects)"))
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Name", "Kind", "Edge", "E0", "E range"])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_table_context_menu)
        left_layout.addWidget(self.table, 1)

        self.status_label = QLabel("Ready.")
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_preview_tab(), "Preview")
        self.tabs.addTab(self._build_preproc_tab(), "Pre-processing")
        self.tabs.addTab(self._build_mu_tab(), "μ(E) Builder")
        self.tabs.addTab(self._build_norm_tab(), "Normalization / EXAFS")
        self.tabs.addTab(self._build_analysis_tab(), "Analysis")
        self.tabs.addTab(self._build_tools_tab(), "Tools")
        self.tabs.addTab(self._build_export_tab(), "Export")
        self.tabs.addTab(self._build_mass_tab(), "Sample mass")
        right_layout.addWidget(self.tabs)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

    def _set_status(self, msg: str) -> None:
        self.status_label.setText(msg)
