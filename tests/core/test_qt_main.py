"""Tests for qt_main.py's headless CLI mode: `PRISM.exe --build-rruff-cache`
/ `--build-amcsd-cache`, what the shipped Download-*.bat/.ps1 scripts run so
a colleague with only the portable exe (no Python) can build the local
reference databases without ever opening the GUI.

No real network access: rruff_science's download entry points are
monkeypatched throughout.
"""
from __future__ import annotations

import sys

import raman
import qt_main


def _patch_rruff_science(monkeypatch, fake) -> None:
    """`qt_main._cli_build_*` does `import raman.rruff_science as rs` —
    an ALIASED dotted import, which CPython resolves via `IMPORT_FROM`
    (an attribute lookup on the parent `raman` package), not directly via
    `sys.modules`. Patching only `sys.modules["raman.rruff_science"]`
    leaves that attribute lookup finding the REAL module (already cached
    as `raman.rruff_science` from an earlier real import elsewhere in the
    test session) and silently bypassing the fake. Both patches are
    needed: sys.modules so a not-yet-imported real module is never
    executed, and the attribute (raising=False, since a fully isolated
    run may not have that attribute set yet either) so IMPORT_FROM finds
    the fake regardless of what else has already imported this module."""
    monkeypatch.setitem(sys.modules, "raman.rruff_science", fake)
    monkeypatch.setattr(raman, "rruff_science", fake, raising=False)


def test_configure_headless_stdio_replaces_none_streams(monkeypatch):
    """The exact PyInstaller --windowed condition: no console attached, so
    sys.stdout/stderr are None and a bare print() would crash."""
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    qt_main._configure_headless_stdio()
    assert sys.stdout is not None
    print("this must not raise")  # the actual regression this guards against


def test_configure_headless_stdio_leaves_real_streams_alone():
    real_stdout = sys.stdout
    qt_main._configure_headless_stdio()
    assert sys.stdout is real_stdout


def test_cli_build_rruff_cache_success(monkeypatch, tmp_path):
    calls = {}

    class FakeRs:
        @staticmethod
        def download_and_build_rruff_cache(categories=None, log=None):
            calls["categories"] = categories
            log("progress line")
            return 999

    _patch_rruff_science(monkeypatch, FakeRs)
    logs = []
    code = qt_main._cli_build_rruff_cache(["--build-rruff-cache"], log=logs.append)
    assert code == 0
    assert calls["categories"] is None
    assert any("999" in m for m in logs)


def test_cli_build_rruff_cache_parses_categories(monkeypatch):
    calls = {}

    class FakeRs:
        @staticmethod
        def download_and_build_rruff_cache(categories=None, log=None):
            calls["categories"] = categories
            log("ok")
            return 1

    _patch_rruff_science(monkeypatch, FakeRs)
    code = qt_main._cli_build_rruff_cache(
        ["--build-rruff-cache", "--categories", "excellent_oriented", "fair_oriented"], log=lambda m: None,
    )
    assert code == 0
    assert calls["categories"] == ["excellent_oriented", "fair_oriented"]


def test_cli_build_rruff_cache_failure_returns_nonzero(monkeypatch):
    class FakeRs:
        @staticmethod
        def download_and_build_rruff_cache(categories=None, log=None):
            raise RuntimeError("no internet")

    _patch_rruff_science(monkeypatch, FakeRs)
    logs = []
    code = qt_main._cli_build_rruff_cache(["--build-rruff-cache"], log=logs.append)
    assert code == 1
    assert any("FAILED" in m and "no internet" in m for m in logs)


def test_cli_build_amcsd_cache_success(monkeypatch):
    class FakeRs:
        @staticmethod
        def download_and_build_amcsd_cache(log=None):
            log("ok")
            return 42

    _patch_rruff_science(monkeypatch, FakeRs)
    logs = []
    code = qt_main._cli_build_amcsd_cache(["--build-amcsd-cache"], log=logs.append)
    assert code == 0
    assert any("42" in m for m in logs)


def test_cli_build_amcsd_cache_failure_returns_nonzero(monkeypatch):
    class FakeRs:
        @staticmethod
        def download_and_build_amcsd_cache(log=None):
            raise OSError("disk full")

    _patch_rruff_science(monkeypatch, FakeRs)
    logs = []
    code = qt_main._cli_build_amcsd_cache(["--build-amcsd-cache"], log=logs.append)
    assert code == 1
    assert any("FAILED" in m for m in logs)
