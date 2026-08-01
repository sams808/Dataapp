"""Tests for qt_main.py's headless CLI mode: `PRISM.exe --build-rruff-cache`
/ `--build-amcsd-cache`, what the shipped Download-*.bat/.ps1 scripts run so
a colleague with only the portable exe (no Python) can build the local
reference databases without ever opening the GUI.

No real network access: rruff_science's download entry points are
monkeypatched throughout.
"""
from __future__ import annotations

import sys

import qt_main


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

    monkeypatch.setitem(sys.modules, "rruff_science", FakeRs)
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

    monkeypatch.setitem(sys.modules, "rruff_science", FakeRs)
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

    monkeypatch.setitem(sys.modules, "rruff_science", FakeRs)
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

    monkeypatch.setitem(sys.modules, "rruff_science", FakeRs)
    logs = []
    code = qt_main._cli_build_amcsd_cache(["--build-amcsd-cache"], log=logs.append)
    assert code == 0
    assert any("42" in m for m in logs)


def test_cli_build_amcsd_cache_failure_returns_nonzero(monkeypatch):
    class FakeRs:
        @staticmethod
        def download_and_build_amcsd_cache(log=None):
            raise OSError("disk full")

    monkeypatch.setitem(sys.modules, "rruff_science", FakeRs)
    logs = []
    code = qt_main._cli_build_amcsd_cache(["--build-amcsd-cache"], log=logs.append)
    assert code == 1
    assert any("FAILED" in m for m in logs)
