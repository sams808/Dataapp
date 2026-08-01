"""Tests for qt_settings_store.py — the shared per-item settings pattern
that should be reused by every Qt tool window instead of re-solved ad hoc
per window (which is exactly how main.py's BaselineParamWindow dual-dict
bug and Simple Plot's multi-select state-bleed bug happened independently).
"""
from __future__ import annotations

from core.qt_settings_store import PerItemSettingsStore


def test_get_creates_default_on_first_access():
    calls = []

    def factory():
        calls.append(1)
        return {"roi_n": 6}

    store = PerItemSettingsStore(factory)
    a = store.get("spec-1")
    assert a == {"roi_n": 6}
    assert len(calls) == 1

    # Second get for the SAME id must not re-invoke the factory.
    b = store.get("spec-1")
    assert b is a
    assert len(calls) == 1


def test_different_ids_get_independent_defaults():
    store = PerItemSettingsStore(lambda: {"roi_n": 6})
    a = store.get("spec-1")
    b = store.get("spec-2")
    a["roi_n"] = 99
    assert b["roi_n"] == 6  # mutating one item's settings must not affect another's


def test_set_overwrites():
    store = PerItemSettingsStore(lambda: {"roi_n": 6})
    store.get("spec-1")
    store.set("spec-1", {"roi_n": 42})
    assert store.get("spec-1") == {"roi_n": 42}


def test_has_and_discard():
    store = PerItemSettingsStore(lambda: {})
    assert not store.has("spec-1")
    store.get("spec-1")
    assert store.has("spec-1")
    store.discard("spec-1")
    assert not store.has("spec-1")


def test_switching_between_items_does_not_lose_or_cross_contaminate_state():
    """Directly mirrors the bug this store fixes: main.py's BaselineParamWindow
    lost/mixed up ROI settings when switching spectra because self.state
    (used to build the UI) and self.spec_states (used on switch) were two
    disconnected dicts. Here there's only one dict, keyed by stable id."""
    store = PerItemSettingsStore(lambda: {"roi_values": [], "type": "poly5"})

    s1 = store.get("spec-1")
    s1["roi_values"] = [[100, 200]]
    s1["type"] = "als"

    s2 = store.get("spec-2")
    assert s2["roi_values"] == []  # fresh default, unaffected by spec-1's edits
    s2["roi_values"] = [[300, 400]]

    # Switch back to spec-1 — its edited state must still be there.
    s1_again = store.get("spec-1")
    assert s1_again["roi_values"] == [[100, 200]]
    assert s1_again["type"] == "als"
