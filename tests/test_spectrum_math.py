"""Tests for spectrum_math.py — sum/average/subtract/scale."""
from __future__ import annotations

import numpy as np
import pytest

import core.spectrum_math as sm


def test_sum_on_identical_grids():
    x = np.linspace(0, 100, 50)
    grid, y = sm.combine_spectra([(x, np.full(50, 2.0)), (x, np.full(50, 3.0))], op="sum")
    assert np.allclose(y, 5.0)
    assert grid[0] == pytest.approx(0.0) and grid[-1] == pytest.approx(100.0)


def test_average_interpolates_to_overlap():
    a = (np.linspace(0, 100, 60), np.full(60, 4.0))
    b = (np.linspace(20, 120, 40), np.full(40, 8.0))
    grid, y = sm.combine_spectra([a, b], op="average")
    assert grid[0] == pytest.approx(20.0)
    assert grid[-1] == pytest.approx(100.0)
    assert np.allclose(y, 6.0)


def test_subtract_is_first_minus_rest():
    x = np.linspace(0, 10, 30)
    grid, y = sm.combine_spectra([(x, np.full(30, 10.0)), (x, np.full(30, 3.0)), (x, np.full(30, 2.0))], op="subtract")
    assert np.allclose(y, 5.0)


def test_weights_apply_per_spectrum():
    x = np.linspace(0, 10, 30)
    grid, y = sm.combine_spectra(
        [(x, np.full(30, 10.0)), (x, np.full(30, 4.0))],
        op="subtract", weights=[1.0, 0.5],
    )
    assert np.allclose(y, 8.0)  # 10 - 0.5*4


def test_normalize_first_equalizes_areas():
    x = np.linspace(0, 100, 200)
    grid, y = sm.combine_spectra(
        [(x, np.full(200, 2.0)), (x, np.full(200, 20.0))],
        op="subtract", normalize_first=True,
    )
    assert np.allclose(y, 0.0, atol=1e-9)  # both normalized to area 100 -> identical


def test_disjoint_ranges_raise():
    a = (np.linspace(0, 10, 20), np.ones(20))
    b = (np.linspace(50, 60, 20), np.ones(20))
    with pytest.raises(ValueError, match="common x-range"):
        sm.combine_spectra([a, b], op="sum")


def test_validation_errors():
    x = np.linspace(0, 10, 20)
    with pytest.raises(ValueError, match="at least 2"):
        sm.combine_spectra([(x, x)], op="sum")
    with pytest.raises(ValueError, match="Unknown operation"):
        sm.combine_spectra([(x, x), (x, x)], op="divide")
    with pytest.raises(ValueError, match="weights length"):
        sm.combine_spectra([(x, x), (x, x)], op="sum", weights=[1.0])


def test_scale_spectrum():
    x = np.linspace(0, 10, 5)
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    x2, y2 = sm.scale_spectrum(x, y, factor=2.0, offset=1.0)
    assert np.allclose(y2, [3, 5, 7, 9, 11])
    assert np.allclose(x2, x)
