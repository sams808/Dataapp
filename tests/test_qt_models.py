"""Tests for qt_models.py — the stable-identity Spectrum/SpectrumLibrary
that replaces main.py's four-parallel-list pattern.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.qt_models import Spectrum, SpectrumLibrary


def _make(title="a"):
    return Spectrum(id=Spectrum.new_id(), title=title, path=f"/x/{title}.txt",
                     kind="generic_xy", x=np.array([1.0, 2.0]), y=np.array([1.0, 2.0]))


def test_two_spectra_with_same_title_get_distinct_ids():
    # This is the exact scenario that broke main.py's reorder handler
    # (self.file_paths[self.file_titles.index(title)] picks the wrong file
    # when two imports share a title). Here, nothing is ever looked up by
    # title, so the collision can't cause a misidentification.
    a = _make("duplicate")
    b = _make("duplicate")
    assert a.id != b.id

    lib = SpectrumLibrary()
    lib.add(a)
    lib.add(b)
    assert lib.get(a.id) is a
    assert lib.get(b.id) is b


def test_library_add_get_remove():
    lib = SpectrumLibrary()
    s = _make()
    lib.add(s)
    assert lib.get(s.id) is s
    assert len(lib) == 1
    lib.remove(s.id)
    assert lib.get(s.id) is None
    assert len(lib) == 0


def test_library_add_duplicate_id_raises():
    lib = SpectrumLibrary()
    s = _make()
    lib.add(s)
    with pytest.raises(ValueError):
        lib.add(s)


def test_library_preserves_insertion_order():
    lib = SpectrumLibrary()
    items = [_make(f"s{i}") for i in range(5)]
    for s in items:
        lib.add(s)
    assert [s.id for s in lib.all()] == [s.id for s in items]


def test_library_reorder():
    lib = SpectrumLibrary()
    items = [_make(f"s{i}") for i in range(3)]
    for s in items:
        lib.add(s)
    new_order = [items[2].id, items[0].id, items[1].id]
    lib.reorder(new_order)
    assert [s.id for s in lib.all()] == new_order


def test_library_reorder_rejects_non_permutation():
    lib = SpectrumLibrary()
    s = _make()
    lib.add(s)
    with pytest.raises(ValueError):
        lib.reorder(["not-a-real-id"])


def test_library_by_kind():
    lib = SpectrumLibrary()
    a = _make("a"); a.kind = "raman_xy"
    b = _make("b"); b.kind = "ta_sdt"
    lib.add(a); lib.add(b)
    assert lib.by_kind("raman_xy") == [a]
    assert lib.by_kind({"raman_xy", "ta_sdt"}) == [a, b]


def test_library_clear():
    lib = SpectrumLibrary()
    lib.add(_make())
    lib.clear()
    assert len(lib) == 0
