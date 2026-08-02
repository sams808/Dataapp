"""
qt_shell.py — the Qt application shell (M5).

Not a mechanical port of main.py's layout: that app is a flat wall of
buttons regardless of what's loaded. This organizes by technique/workflow
instead — a left rail for Library / Raman / XAS / DTA workspaces, so (once
M6-M11 fill in the technique pages) a DTA user only ever sees DTA-relevant
tools. For now the technique pages are placeholders; Library is fully
functional (import via io_universal, select, plot) so this milestone is a
real, demoable slice rather than inert scaffolding.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog, QHBoxLayout, QListWidget, QListWidgetItem, QMainWindow,
    QMessageBox, QStackedWidget, QVBoxLayout, QWidget,
)

from dta.qt_dta import DtaWorkspace
from core.qt_library import (
    CombineDialog as CombineDialog,
    LibraryPage,
    _load_spectrum_from_path as _load_spectrum_from_path,
)
from core.qt_models import SpectrumLibrary
from processing.qt_multi_fit import MultiFitWorkspace
from core.qt_settings_store import PerItemSettingsStore
from raman.qt_raman import RamanWorkspace
from processing.qt_single_fit import SingleFitWorkspace
from processing.qt_baseline import BaselineWorkspace
from processing.qt_calc import CalcWorkspace
from figures.qt_figures import FiguresWorkspace
from glass.qt_glass import GlassWorkspace
from saxs.qt_saxs import SaxsWorkspace
from xrd.qt_xrd import XrdIdWorkspace
from xrd.qt_htxrd import HtxrdWorkspace
from raman.qt_rruff import RruffMatchWorkspace
from xas.qt_xas import XasWorkspace

NAV_LIBRARY = "Library"
NAV_RAMAN = "Raman"
NAV_XAS = "XAS"
NAV_DTA = "DTA / Thermal"
NAV_FITTING = "Peak Fitting"
NAV_MULTIFIT = "Multi-Fit"
NAV_RRUFF = "Raman ID"
NAV_HTXRD = "HT-XRD"
NAV_BASELINE = "Baseline"
# NAV_FITTING/NAV_MULTIFIT/NAV_RRUFF/NAV_HTXRD are appended at the end (not
# inserted after Raman) so the DTA page keeps nav row 3 —
# test_qt_dta.py's test_shell_dta_page_picks_up_library_records hardcodes
# setCurrentRow(3), and there's no reason to reorder the rail just to churn
# that index.
NAV_CALC = "Calculations"
NAV_XRD_ID = "XRD ID"
NAV_FIGURES = "Figures"
NAV_SAXS = "SAXS/WAXS"
NAV_GLASS = "Glass"
# Ordered PER TECHNIQUE (user request): Raman block, then XRD, XAS,
# Thermal, SAXS, cross-technique processing, Figures.
NAV_ITEMS = [NAV_LIBRARY, NAV_RAMAN, NAV_RRUFF, NAV_FITTING, NAV_MULTIFIT,
             NAV_BASELINE, NAV_XRD_ID, NAV_HTXRD, NAV_XAS, NAV_DTA,
             NAV_SAXS, NAV_GLASS, NAV_CALC, NAV_FIGURES]

# Activatable modules (user request: a Raman-only or XRD-only user should
# see a simple app). Each module = (accent color, its nav pages); the
# Library is the always-on core. Toggled from the Modules toolbar,
# persisted via QSettings, and the colors mark the nav rail entries.
MODULES = {
    "Raman":      ("#3fa66a", [NAV_RAMAN, NAV_RRUFF]),
    "Fitting":    ("#c9873a", [NAV_FITTING, NAV_MULTIFIT]),
    "XRD":        ("#e0563c", [NAV_XRD_ID, NAV_HTXRD]),
    "XAS":        ("#8b5cf6", [NAV_XAS]),
    "Thermal":    ("#d43f6e", [NAV_DTA]),
    "Processing": ("#3b82f6", [NAV_BASELINE, NAV_CALC]),
    "Figures":    ("#14b8a6", [NAV_FIGURES]),
    "SAXS/WAXS":  ("#b08f26", [NAV_SAXS]),
    "Glass":      ("#5b8dd6", [NAV_GLASS]),
}
CORE_COLOR = "#8a97b5"  # Library


def _nav_color(name: str) -> str:
    for color, pages in MODULES.values():
        if name in pages:
            return color
    return CORE_COLOR
DTA_KINDS = {"ta_sdt", "dta_table"}


class PrismMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        import os
        from core.qt_help import APP_NAME, APP_VERSION, asset_path
        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        icon_path = asset_path("prism_logo.png")
        if os.path.isfile(icon_path):
            from PySide6.QtGui import QIcon
            self.setWindowIcon(QIcon(icon_path))
        self.resize(1280, 820)

        self.library = SpectrumLibrary()
        # Shared by Peak Fitting (M8) and Multi-Fit (M9): a batch write-back
        # must be immediately visible in Peak Fitting and vice versa, so
        # there's exactly one PerItemSettingsStore, not one per workspace.
        self.fit_param_memory = PerItemSettingsStore(list)

        central = QWidget()
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        sidebar = QWidget()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(180)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 12, 0, 0)

        self.nav = QListWidget()
        self.nav.setObjectName("NavList")
        from PySide6.QtGui import QColor, QIcon, QPixmap
        for name in NAV_ITEMS:
            item = QListWidgetItem(name, self.nav)
            square = QPixmap(10, 10)
            square.fill(QColor(_nav_color(name)))
            item.setIcon(QIcon(square))
        self.nav.currentRowChanged.connect(self._on_nav_changed)
        sidebar_layout.addWidget(self.nav)
        outer.addWidget(sidebar)

        self.stack = QStackedWidget()
        self.library_page = LibraryPage(self.library)
        self.stack.addWidget(self.library_page)
        self.raman_page = RamanWorkspace(library=self.library)
        self.stack.addWidget(self.raman_page)
        self.xas_page = XasWorkspace()
        self.stack.addWidget(self.xas_page)
        self.dta_page = DtaWorkspace()
        self.stack.addWidget(self.dta_page)
        self.fitting_page = SingleFitWorkspace(library=self.library, fit_param_memory=self.fit_param_memory)
        self.stack.addWidget(self.fitting_page)
        self.multifit_page = MultiFitWorkspace(library=self.library, fit_param_memory=self.fit_param_memory)
        self.stack.addWidget(self.multifit_page)
        self.rruff_page = RruffMatchWorkspace(
            library=self.library, on_send_cifs=self._on_rruff_send_cifs,
            on_accept=lambda sid, old: self.library_page.push_undo(("ident", sid, old)),
        )
        self.stack.addWidget(self.rruff_page)
        self.htxrd_page = HtxrdWorkspace()
        self.stack.addWidget(self.htxrd_page)
        self.baseline_page = BaselineWorkspace(
            library=self.library,
            on_derived_added=lambda ids: self.library_page.push_undo(("add", list(ids))),
        )
        self.stack.addWidget(self.baseline_page)
        self.calc_page = CalcWorkspace(
            library=self.library,
            on_derived_added=lambda ids: self.library_page.push_undo(("add", list(ids))),
        )
        self.stack.addWidget(self.calc_page)
        self.xrd_id_page = XrdIdWorkspace(
            library=self.library,
            on_accept=lambda sid, old: self.library_page.push_undo(("xrd_ident", sid, old)),
        )
        self.stack.addWidget(self.xrd_id_page)
        self.figures_page = FiguresWorkspace(library=self.library)
        self.stack.addWidget(self.figures_page)
        self.saxs_page = SaxsWorkspace(
            library=self.library,
            on_derived_added=lambda ids: self.library_page.push_undo(("add", list(ids))),
        )
        self.stack.addWidget(self.saxs_page)
        self.glass_page = GlassWorkspace()
        self.stack.addWidget(self.glass_page)
        outer.addWidget(self.stack, 1)

        # Nav row -> page by NAME (the rail is ordered per technique, the
        # stack in construction order — never map the two positionally).
        self._pages_by_nav = {
            NAV_LIBRARY: self.library_page, NAV_RAMAN: self.raman_page,
            NAV_XAS: self.xas_page, NAV_DTA: self.dta_page,
            NAV_FITTING: self.fitting_page, NAV_MULTIFIT: self.multifit_page,
            NAV_RRUFF: self.rruff_page, NAV_HTXRD: self.htxrd_page,
            NAV_BASELINE: self.baseline_page, NAV_CALC: self.calc_page,
            NAV_XRD_ID: self.xrd_id_page, NAV_FIGURES: self.figures_page,
            NAV_SAXS: self.saxs_page, NAV_GLASS: self.glass_page,
        }
        self.nav.setCurrentRow(0)

        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction("Import files…", self.library_page._on_import_clicked, "Ctrl+O")
        file_menu.addAction("Custom import…", self.library_page._on_custom_import_clicked, "Ctrl+I")
        file_menu.addAction("Export selected as text…", self.library_page._export_selected_txt, "Ctrl+E")
        file_menu.addSeparator()
        file_menu.addAction("Open project…", self.open_project, "Ctrl+Shift+O")
        file_menu.addAction("Save project as…", self.save_project, "Ctrl+S")
        file_menu.addSeparator()
        file_menu.addAction("Clear imports…", self.library_page.clear_all)
        file_menu.addAction("Undo", self.library_page._undo, "Ctrl+Z")
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close, "Ctrl+Q")

        view_menu = self.menuBar().addMenu("&View")
        self.dark_mode_action = view_menu.addAction("Dark mode")
        self.dark_mode_action.setCheckable(True)
        self.dark_mode_action.toggled.connect(self._on_dark_mode_toggled)
        self.dark_mode_action.setChecked(True)  # PRISM starts in dark mode

        self.console_action = view_menu.addAction("Python console")
        self.console_action.setCheckable(True)
        self.console_action.toggled.connect(self._on_console_toggled)
        self._console_dock = None  # created lazily on first open

        # --- Modules menu (between View and Help): checkable per-module
        # entries; default = everything off except Raman, so a new user
        # starts with the simplest app.
        from PySide6.QtCore import QSettings
        settings = QSettings("PRISM", "PRISM")
        modules_menu = self.menuBar().addMenu("&Modules")
        self.module_checks: dict = {}
        from PySide6.QtGui import QAction, QColor, QPixmap, QIcon
        for mod_name, (color, _pages) in MODULES.items():
            act = QAction(mod_name, self)
            act.setCheckable(True)
            square = QPixmap(10, 10)
            square.fill(QColor(color))
            act.setIcon(QIcon(square))
            act.setChecked(settings.value(f"modules/{mod_name}", mod_name == "Raman", type=bool))
            act.toggled.connect(lambda on, m=mod_name: self._on_module_toggled(m, on))
            self.module_checks[mod_name] = act
            modules_menu.addAction(act)
        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction("Quick-start guide", self.show_help, "F1")
        from core.qt_help import MODULE_GUIDES
        guides_menu = help_menu.addMenu("Module guides")
        for gname in MODULE_GUIDES:
            guides_menu.addAction(gname, lambda g=gname: self.show_module_guide(g))
        help_menu.addAction("About", self.show_about)
        help_menu.addAction("Credits", self.show_credits)

        self._apply_module_visibility()

        self.statusBar().showMessage("Ready.")

        # Restore window geometry + last-used workspace from the previous
        # session (QSettings, per-user registry on Windows).
        geometry = settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        nav_row = settings.value("nav_row", 0, type=int)
        if 0 <= nav_row < self.nav.count() and not self.nav.isRowHidden(nav_row):
            self.nav.setCurrentRow(nav_row)
            # setCurrentRow doesn't fire currentRowChanged when the row is
            # unchanged, and the per-workspace refresh hook must still run
            # for whatever page we restored into.
            self._on_nav_changed(self.nav.currentRow())

    # ------------------------------------------------------------------
    def _on_module_toggled(self, module: str, enabled: bool) -> None:
        from PySide6.QtCore import QSettings
        QSettings("PRISM", "PRISM").setValue(f"modules/{module}", bool(enabled))
        self._apply_module_visibility()

    def _apply_module_visibility(self) -> None:
        """Hide the nav rows of every disabled module; the Library is
        always-on. If the current page just vanished, fall back to it."""
        hidden_pages = set()
        for mod_name, (color, pages) in MODULES.items():
            if not self.module_checks[mod_name].isChecked():
                hidden_pages.update(pages)
        for row, name in enumerate(NAV_ITEMS):
            self.nav.setRowHidden(row, name in hidden_pages)
        current = self.nav.currentRow()
        if current >= 0 and self.nav.isRowHidden(current):
            self.nav.setCurrentRow(NAV_ITEMS.index(NAV_LIBRARY))

    def show_module_guide(self, name: str) -> None:
        from core.qt_help import MODULE_GUIDES, HelpDialog
        HelpDialog(self, html=MODULE_GUIDES[name], title=f"{name} — guide").exec()

    def show_credits(self) -> None:
        from core.qt_help import CREDITS_HTML, HelpDialog
        HelpDialog(self, html=CREDITS_HTML, title="Credits").exec()

    def closeEvent(self, event) -> None:
        from PySide6.QtCore import QSettings
        settings = QSettings("PRISM", "PRISM")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("nav_row", self.nav.currentRow())
        super().closeEvent(event)

    def show_help(self) -> None:
        from core.qt_help import HelpDialog
        HelpDialog(self).exec()

    def show_about(self) -> None:
        from core.qt_help import ABOUT_HTML, HelpDialog
        HelpDialog(self, html=ABOUT_HTML, title="About PRISM").exec()

    def _on_rruff_send_cifs(self, cif_paths) -> None:
        """RRUFF→CIF handoff target: add the structures to the Raman
        workspace's CIF overlay and switch to it so the result is visible."""
        added = self.raman_page.add_cif_files(list(cif_paths))
        self.nav.setCurrentRow(NAV_ITEMS.index(NAV_RAMAN))
        self.statusBar().showMessage(f"Added {added} CIF(s) to the Raman CIF overlay.")

    def _on_console_toggled(self, visible: bool) -> None:
        if self._console_dock is None:
            import numpy as np
            import pandas as pd
            from core.qt_console import ConsoleDock
            self._console_dock = ConsoleDock({
                "window": self,
                "library": self.library,
                "xas_store": self.xas_page.store,
                "htxrd_series": self.htxrd_page.series,
                "fit_params": self.fit_param_memory,
                "np": np,
                "pd": pd,
            }, parent=self)
            self.addDockWidget(Qt.BottomDockWidgetArea, self._console_dock)
            # Keep the menu checkbox honest when the user closes the dock
            # via its own title-bar X instead of the menu.
            self._console_dock.visibilityChanged.connect(self.console_action.setChecked)
        self._console_dock.setVisible(visible)

    def _on_dark_mode_toggled(self, enabled: bool) -> None:
        """Dark mode restyles the Qt chrome only — matplotlib plot areas
        stay white so what's on screen always matches PNG/SVG/PDF export."""
        from PySide6.QtWidgets import QApplication
        from core.qt_theme import apply_theme
        app = QApplication.instance()
        if app is not None:
            apply_theme(app, dark=enabled)

    # ------------------------------------------------------------------
    # Project persistence (M14): everything in the shared Library plus the
    # shared fit-parameter store, in one .prism file (legacy-extension
    # projects still load). The file format is versioned so workspaces can
    # be added without breaking old projects.
    # ------------------------------------------------------------------
    def save_project(self) -> None:
        import core.project_io as project_io
        path, _ = QFileDialog.getSaveFileName(self, "Save project as…", "", "PRISM project (*.prism);;Legacy project (*.dataapp)")
        if not path:
            return
        if not path.lower().endswith((".prism", ".dataapp")):
            path += ".prism"
        fit_params = {sid: params for sid, params in self.fit_param_memory.items()}
        try:
            cif_overlays = [
                {k: s.get(k) for k in ("path", "label", "plot_label", "visible", "color", "pad")}
                for s in self.raman_page.cif_series
            ]
            baseline_settings = {sid: dict(val) for sid, val in self.baseline_page.settings.items()}
            project_io.save_project(
                path, self.library.all(), fit_params,
                xas_spectra=self.xas_page.store.all(),
                htxrd_patterns=self.htxrd_page.series,
                cif_overlays=cif_overlays,
                baseline_settings=baseline_settings,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save project error", str(exc))
            return
        self.statusBar().showMessage(f"Project saved: {path}")

    def open_project(self) -> None:
        import core.project_io as project_io
        path, _ = QFileDialog.getOpenFileName(self, "Open project", "", "PRISM projects (*.prism *.dataapp);;All files (*.*)")
        if not path:
            return
        if len(self.library) > 0:
            resp = QMessageBox.question(
                self, "Open project",
                "Opening a project replaces the current library contents. Continue?",
            )
            if resp != QMessageBox.Yes:
                return
        try:
            project = project_io.load_project(path)
        except Exception as exc:
            QMessageBox.critical(self, "Open project error", str(exc))
            return

        self.library.clear()
        self.fit_param_memory.clear()
        for sp in project.spectra:
            self.library.add(sp)
        for sid, params in project.fit_params.items():
            self.fit_param_memory.set(sid, params)

        self.xas_page.store.clear()
        for sp in project.xas_spectra:
            self.xas_page.store.add(sp)
        self.xas_page.selected_sid = None
        self.xas_page._refresh_all()

        if project.htxrd_patterns:
            self.htxrd_page.set_series(project.htxrd_patterns)

        if project.cif_overlays:
            self.raman_page.restore_cif_overlays(project.cif_overlays)
        self.baseline_page.settings.clear()
        for sid, val in project.baseline_settings.items():
            self.baseline_page.settings.set(sid, val)

        self.library_page._refresh_table()
        # Re-sync whichever workspace is currently visible.
        self._on_nav_changed(self.nav.currentRow())
        self.statusBar().showMessage(
            f"Project loaded: {len(project.spectra)} spectra, {len(project.xas_spectra)} XAS objects, "
            f"{len(project.htxrd_patterns)} HT-XRD patterns from {path}"
        )

    def _dta_records_from_library(self):
        records = []
        for spectrum in self.library.by_kind(DTA_KINDS):
            records.append({
                "title": spectrum.title,
                "path": spectrum.path,
                "df": spectrum.df,
                "meta": spectrum.meta,
            })
        return records

    def _on_nav_changed(self, row: int) -> None:
        if not (0 <= row < len(NAV_ITEMS)):
            return
        # Resolve by NAME: the rail is ordered per technique while the stack
        # keeps construction order — never map the two positionally.
        page = self._pages_by_nav[NAV_ITEMS[row]]
        self.stack.setCurrentWidget(page)
        if page is self.dta_page:
            self.dta_page.set_records(self._dta_records_from_library())
        elif page is self.multifit_page:
            self.multifit_page.set_spectra([s.id for s in self.library.all()])
            # recipes saved while on another page must appear on entry
            self.multifit_page._refresh_recipe_list()
        elif hasattr(page, "set_spectra"):
            page.set_spectra([s.id for s in self.library.all()])
