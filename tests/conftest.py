"""Shared pytest fixtures: paths to bundled EXAMPLES data and the archived
real-world dataset. Fixture files are RAW inputs only — never outputs from
the (historically buggy) pipeline. Expected values are derived from
hand-verified/fixed computations or invariant checks, not from snapshotting
old pipeline output.
"""
from __future__ import annotations

from pathlib import Path

import pytest

PRISM_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PRISM_ROOT / "EXAMPLES"


def _find_archive_dir() -> Path:
    """The archived real-data folder (ISG pressure series, PBi0-1 maps) is
    not part of the repo; the repo has lived both next to it and one level
    up (…\\NEW\\PRISM vs the old …\\NEW\\Raman\\<repo>). Take the first
    candidate that exists — archive-dependent tests skip when none does."""
    candidates = [
        PRISM_ROOT.parent / "data_and_notebooks",
        PRISM_ROOT.parent / "Raman" / "data_and_notebooks",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return candidates[0]


ARCHIVE_DIR = _find_archive_dir()


def _find_saxs_physic_based_dir() -> Path:
    """The real, reduced PBi glass-series SAXS profiles (physic_based/) are
    not part of the repo. PRISM_SAXS_PHYSIC_BASED_DIR overrides the default
    location for other machines/CI; SAXS-real-data tests skip when neither
    is present."""
    import os
    override = os.environ.get("PRISM_SAXS_PHYSIC_BASED_DIR")
    if override:
        return Path(override)
    return PRISM_ROOT.parent.parent.parent / "SAXS" / "PBi-sorted" / "physic_based"


SAXS_PHYSIC_BASED_DIR = _find_saxs_physic_based_dir()


@pytest.fixture(scope="session")
def examples_dir() -> Path:
    return EXAMPLES_DIR


@pytest.fixture(scope="session")
def archive_dir() -> Path:
    return ARCHIVE_DIR


@pytest.fixture(scope="session")
def dta_example_path() -> Path:
    return EXAMPLES_DIR / "DTA_example.txt"


@pytest.fixture(scope="session")
def raman_example_path() -> Path:
    return EXAMPLES_DIR / "Raman_example.txt"


@pytest.fixture(scope="session")
def xrd_example_path() -> Path:
    return EXAMPLES_DIR / "XRD_example.xy"


@pytest.fixture(scope="session")
def saxs_example_path() -> Path:
    return EXAMPLES_DIR / "SAXS_example.dat"


@pytest.fixture(scope="session")
def rruff_sample_zip_path() -> Path:
    """A real, small (271K, 17 spectra) RRUFF bulk-download ZIP
    (fair_oriented.zip, fetched 2026-07-15 from
    https://www.rruff.net/zipped_data_files/raman/), checked in as a fixture
    rather than downloaded per-test-run — mirrors EXAMPLES/HTXRD_example.rasx."""
    return EXAMPLES_DIR / "RRUFF_fair_oriented_sample.zip"


@pytest.fixture(scope="session")
def isg_series_paths() -> list[Path]:
    """Real archived Raman series (headerless 2-column, same shape as EXAMPLES)."""
    return sorted(ARCHIVE_DIR.glob("ISG_*gpa.txt"))


@pytest.fixture(scope="session")
def pbi0_map_paths() -> list[Path]:
    """Real archived multi-file Raman map series — RAW map spectra only
    (`...-mapN.txt`). The same folder also holds derived outputs from past
    processing runs (`..._bl.txt` baseline-subtracted, `..._bl_fit....txt`
    fit-parameter tables with ~8 rows) that the old `*-map*.txt` glob
    silently swept in; a fit-parameter table is not a spectrum."""
    import re
    return sorted(
        p for p in (ARCHIVE_DIR / "PBi0-1").glob("*-map*.txt")
        if re.search(r"-map\d+\.txt$", p.name)
    )


def larch_available() -> bool:
    try:
        import larch  # noqa: F401
        return True
    except Exception:
        return False


requires_larch = pytest.mark.skipif(not larch_available(), reason="larch is not installed")


@pytest.fixture(autouse=True, scope="session")
def _synchronous_workers():
    """Run qt_worker background jobs inline during tests, so tests assert
    immediately after triggering an operation instead of polling for a
    worker thread to finish. Production code never flips this switch."""
    try:
        import core.qt_worker as qt_worker
    except Exception:
        yield
        return
    qt_worker.set_synchronous(True)
    yield
    qt_worker.set_synchronous(False)


@pytest.fixture(autouse=True)
def _hermetic_xrd_registry(tmp_path, monkeypatch):
    """The XRD ID workspace reads/writes a per-user database registry
    (~/.raman_cache/xrd_id/databases.json) and would auto-migrate a real
    local database into it — point all registry paths at a temp location so
    tests never touch (or migrate!) the user's real registered databases.
    Works because the registry functions resolve these module attributes at
    call time, not def time."""
    import xrd.xrd_id_science as xid
    monkeypatch.setattr(xid, "XRD_ID_REGISTRY_PATH", str(tmp_path / "xrd_registry.json"))
    monkeypatch.setattr(xid, "XRD_ID_DB_PATH", str(tmp_path / "xrdid_legacy_absent.sq"))
    monkeypatch.setattr(xid, "XRD_ID_IMPORT_DIR", str(tmp_path / "xrd_imported"))
    yield


@pytest.fixture(autouse=True)
def _hermetic_cif_cache(tmp_path, monkeypatch):
    """cif_tools.bragg_peaks_from_cif_generic() caches to the real per-user
    ~/.raman_cache/cif by default (use_cache=True) — point _cache_dir() at a
    temp location so tests never read/write the user's real cache, without
    every call site having to remember use_cache=False."""
    import raman.cif_tools as cif_tools
    cache_dir = tmp_path / "cif_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cif_tools, "_cache_dir", lambda: str(cache_dir))
    yield


@pytest.fixture(autouse=True)
def _hermetic_qsettings(monkeypatch):
    """Tests must not read or write the real per-user QSettings (Windows
    registry): the shell restores window geometry and the last-used nav row
    from there, so a REAL value left by the developer's own app launches
    would leak into tests (it did — a restored nav row broke two shell
    tests until this fixture). Replace QSettings with an in-memory no-op."""
    import sys
    if "PySide6.QtCore" not in sys.modules:
        yield
        return

    store = {}  # in-memory, per-test: hermetic, but persistence testable

    class _FakeQSettings:
        def __init__(self, *args, **kwargs):
            pass

        def value(self, key, default=None, type=None):  # noqa: A002 - Qt's own kwarg name
            if key.startswith("modules/") and key not in store:
                return True  # tests see the full app; the Raman-only default is seeded explicitly where tested
            if key in store:
                v = store[key]
                if type is bool:
                    return v in (True, "true", "True", 1)
                if type is int:
                    try:
                        return int(v)
                    except (TypeError, ValueError):
                        return default
                return v
            return default

        def setValue(self, key, value):
            store[key] = value

    import PySide6.QtCore
    monkeypatch.setattr(PySide6.QtCore, "QSettings", _FakeQSettings)
    yield


@pytest.fixture(autouse=True)
def _flush_pending_qt_draws():
    """Auto-preview-on-entry means many actions queue a render through
    PlotWidget's 120ms debounce; matplotlib's draw_idle() then schedules an
    UNPARENTED QTimer.singleShot(0). A test that ends inside that window
    leaves the callback to fire on a torn-down canvas, poisoning the NEXT
    test with 'Internal C++ object already deleted'. Flush the event loop
    for a beat after each Qt test so every pending debounce + idle draw
    completes while the widgets are still alive."""
    yield
    import sys
    import time
    if "PySide6.QtWidgets" not in sys.modules:
        return
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        return
    deadline = time.time() + 0.25
    while time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)


@pytest.fixture(autouse=True)
def _prevent_blocking_qt_dialogs(monkeypatch):
    """Autouse safety net for Qt tests: an accidental REAL (unmocked)
    QMessageBox.information/warning/critical/question call opens a real
    modal dialog and blocks the event loop indefinitely waiting for a human
    click — fatal during an unattended/autonomous run. This is not
    theoretical: an unmocked success-path QMessageBox.information() in
    test_qt_single_fit.py's export-components test caused a real ~74-minute
    hang while verifying M8 (eventually returned exit 0, presumably via some
    outer supervisor timeout — not something to rely on). Default every
    convenience dialog to an immediate no-op/accept; a test that specifically
    wants to verify a dialog appeared can still monkeypatch over this within
    its own body (a test-local monkeypatch.setattr applied after this
    fixture's setup wins).

    The sys.modules guard (skip when PySide6 isn't already imported) is a
    holdover from when this suite also contained Tk-based tests and
    importing PySide6 process-wide would have corrupted Tk's Tcl
    interpreter on Windows; the Tk tests are gone now, but the guard is
    still harmlessly correct — pure-science test sessions simply never
    need QMessageBox patched.
    """
    import sys
    if "PySide6.QtWidgets" not in sys.modules:
        return
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok))
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
