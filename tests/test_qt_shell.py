"""Tests for the Qt shell (M5) — window construction, real-file import via
io_universal, and selection-triggered plotting. Uses pytest-qt's qtbot to
manage the QApplication/event loop.
"""
from __future__ import annotations


from core.qt_shell import NAV_ITEMS, PrismMainWindow, _load_spectrum_from_path
from core.qt_widgets import PlotWidget


def test_plot_widget_constructs_and_clears(qtbot):
    widget = PlotWidget()
    qtbot.addWidget(widget)
    widget.clear("Empty")
    qtbot.wait(20)  # let matplotlib's deferred draw_idle() complete before teardown
    assert widget.ax.get_title() == "Empty"


def test_load_spectrum_from_path_dta_example(dta_example_path):
    spectrum = _load_spectrum_from_path(str(dta_example_path))
    assert spectrum.kind == "ta_sdt"
    assert len(spectrum.x) > 100
    assert spectrum.title == "DTA_example"


def test_module_toggle_does_not_duplicate_help_menu(qtbot):
    """Regression (user screenshot: five Help menus): _on_module_toggled
    once contained a copy-pasted Help-menu construction block, so every
    module toggle grew the menu bar."""
    window = PrismMainWindow()
    qtbot.addWidget(window)
    titles_before = [a.text() for a in window.menuBar().actions()]
    assert titles_before.count("&Help") == 1
    window.module_checks["XRD"].setChecked(False)
    window.module_checks["XRD"].setChecked(True)
    window.module_checks["Glass"].setChecked(False)
    titles_after = [a.text() for a in window.menuBar().actions()]
    assert titles_after == titles_before


def test_main_window_constructs_with_nav_and_library_page(qtbot):
    window = PrismMainWindow()
    qtbot.addWidget(window)
    qtbot.wait(20)
    # Tied to NAV_ITEMS rather than a hardcoded count — this assertion broke
    # (correctly, but noisily) at M8, M9, and M12 as each milestone appended
    # a new nav entry. What it actually guards is nav/stack consistency:
    assert window.nav.count() == len(NAV_ITEMS)
    assert window.stack.count() == len(NAV_ITEMS)
    assert window.nav.currentRow() == 0
    assert window.stack.currentWidget() is window.library_page


def test_import_and_select_populates_plot(qtbot, dta_example_path):
    window = PrismMainWindow()
    qtbot.addWidget(window)

    spectrum = _load_spectrum_from_path(str(dta_example_path))
    window.library.add(spectrum)
    window.library_page._refresh_table()

    assert window.library_page.table.rowCount() == 1
    window.library_page.table.selectRow(0)
    qtbot.wait(50)

    assert len(window.library_page.plot.ax.lines) == 1


def test_two_imports_with_same_stem_do_not_collide(qtbot, raman_example_path, tmp_path):
    """Regression guard for the identity bug this data model fixes: import
    the same file twice (simulating two files that would produce the same
    display title) and confirm both survive as distinct library entries."""
    window = PrismMainWindow()
    qtbot.addWidget(window)

    copy_path = tmp_path / raman_example_path.name
    copy_path.write_bytes(raman_example_path.read_bytes())

    s1 = _load_spectrum_from_path(str(raman_example_path))
    s2 = _load_spectrum_from_path(str(copy_path))
    assert s1.title == s2.title  # same stem -> same display title
    assert s1.id != s2.id  # but distinct identity

    window.library.add(s1)
    window.library.add(s2)
    window.library_page._refresh_table()
    qtbot.wait(20)
    assert window.library_page.table.rowCount() == 2


# --------------------------------------------------------------------------
# Module system + branding (PRISM): activatable module checkboxes, colored
# nav, credits.
# --------------------------------------------------------------------------

def test_modules_cover_every_non_library_nav_item():
    from core.qt_shell import MODULES, NAV_ITEMS, NAV_LIBRARY
    in_modules = [p for _color, pages in MODULES.values() for p in pages]
    assert sorted(in_modules) == sorted(n for n in NAV_ITEMS if n != NAV_LIBRARY)
    assert len(in_modules) == len(set(in_modules))  # no page in two modules


def test_module_toggle_hides_nav_rows_and_falls_back_to_library(qtbot):
    from core.qt_shell import PrismMainWindow, NAV_ITEMS
    window = PrismMainWindow()
    qtbot.addWidget(window)
    xas_row = NAV_ITEMS.index("XAS")

    assert not window.nav.isRowHidden(xas_row)  # conftest enables all modules
    window.nav.setCurrentRow(xas_row)
    qtbot.wait(20)

    window.module_checks["XAS"].setChecked(False)
    assert window.nav.isRowHidden(xas_row)
    # current page vanished -> back to the Library
    assert window.nav.currentRow() == NAV_ITEMS.index("Library")
    window.module_checks["XAS"].setChecked(True)
    assert not window.nav.isRowHidden(xas_row)


def test_fresh_install_defaults_to_raman_only(qtbot):
    """User request: a new user starts with only the Raman module on."""
    from PySide6.QtCore import QSettings
    from core.qt_shell import MODULES, PrismMainWindow, NAV_ITEMS
    s = QSettings("PRISM", "PRISM")
    for m in MODULES:  # simulate the fresh-install store (no conftest override)
        s.setValue(f"modules/{m}", m == "Raman")
    window = PrismMainWindow()
    qtbot.addWidget(window)
    assert window.module_checks["Raman"].isChecked()
    assert not window.module_checks["XAS"].isChecked()
    assert window.nav.isRowHidden(NAV_ITEMS.index("XAS"))
    assert not window.nav.isRowHidden(NAV_ITEMS.index("Raman ID"))


def test_module_state_persists_via_qsettings(qtbot):
    from core.qt_shell import PrismMainWindow, NAV_ITEMS
    w1 = PrismMainWindow()
    qtbot.addWidget(w1)
    w1.module_checks["Thermal"].setChecked(False)
    w1.close()

    w2 = PrismMainWindow()  # same (hermetic, in-memory) settings store
    qtbot.addWidget(w2)
    assert not w2.module_checks["Thermal"].isChecked()
    assert w2.nav.isRowHidden(NAV_ITEMS.index("DTA / Thermal"))


def test_window_title_and_credits(qtbot):
    from core.qt_help import APP_NAME, APP_VERSION, CREDITS_HTML
    from core.qt_shell import PrismMainWindow
    window = PrismMainWindow()
    qtbot.addWidget(window)
    assert window.windowTitle() == f"{APP_NAME} {APP_VERSION}"
    assert APP_NAME == "PRISM"
    for needle in ("NOME group", "Department of Energy", "McCloy", "ChatGPT", "Claude"):
        assert needle in CREDITS_HTML, needle
    # the personal credit stays hidden: an HTML comment, not rendered text
    assert "Sam Souda" in CREDITS_HTML
    comment_free = CREDITS_HTML
    while "<!--" in comment_free:
        start = comment_free.index("<!--")
        end = comment_free.index("-->", start) + 3
        comment_free = comment_free[:start] + comment_free[end:]
    assert "Sam Souda" not in comment_free


def test_nav_items_have_module_colors(qtbot):
    from core.qt_shell import PrismMainWindow, NAV_ITEMS
    window = PrismMainWindow()
    qtbot.addWidget(window)
    for row in range(window.nav.count()):
        assert not window.nav.item(row).icon().isNull(), NAV_ITEMS[row]
