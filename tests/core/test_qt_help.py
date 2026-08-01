"""Tests for qt_help.py — the in-app quick-start guide and About dialog."""
from __future__ import annotations

from core.qt_help import ABOUT_HTML, HELP_HTML, HelpDialog
from core.qt_shell import NAV_ITEMS, PrismMainWindow


def test_help_covers_every_workspace():
    """The guide must mention every nav workspace — a new workspace without
    a help section fails here, keeping onboarding in sync with the app."""
    # Nav labels vs help-section wording differ slightly; map explicitly.
    expected_mentions = {
        "Library": "Library",
        "Raman": "Simple Plot",
        "XAS": "XAS",
        "DTA / Thermal": "DTA / Thermal",
        "Peak Fitting": "Peak Fitting",
        "Multi-Fit": "Multi-Fit",
        "Raman ID": "Raman ID",
        "HT-XRD": "HT-XRD",
        "Baseline": "Baseline",
        "Calculations": "Calculations",
        "XRD ID": "XRD ID",
        "Figures": "Figures",
        "SAXS/WAXS": "SAXS/WAXS",
        "Glass": "Glass",
    }
    assert set(expected_mentions) == set(NAV_ITEMS), "nav changed — update the help guide and this map"
    for nav, needle in expected_mentions.items():
        assert needle in HELP_HTML, f"help guide has no section mentioning {nav!r}"


def test_help_carries_the_two_critical_findings():
    # The HWHM width convention and the glass-baseline lambda tip are the
    # two facts most likely to save a group member from a wrong result.
    assert "HWHM" in HELP_HTML
    assert "1e7" in HELP_HTML


def test_about_carries_database_citations():
    assert "Lafuente" in ABOUT_HTML  # RRUFF
    assert "American Mineralogist Crystal Structure" in ABOUT_HTML  # AMCSD


def test_help_dialog_constructs_and_renders(qtbot):
    dlg = HelpDialog(None)
    qtbot.addWidget(dlg)
    assert "quick-start" in dlg.windowTitle().lower()
    assert "Library" in dlg.browser.toPlainText()


def test_shell_help_menu_actions_exist(qtbot):
    window = PrismMainWindow()
    qtbot.addWidget(window)
    qtbot.wait(20)
    menus = [a.text().replace("&", "") for a in window.menuBar().actions()]
    assert "Help" in menus
