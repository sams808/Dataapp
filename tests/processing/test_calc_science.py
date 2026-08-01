"""Tests for calc_science.py — the Calculations engine (user request:
'EVERY TYPE OF CALCULATION useful for such a data processing app')."""
from __future__ import annotations

import numpy as np
import pytest

import processing.calc_science as cs


def _xy(value=2.0, n=101, lo=0.0, hi=100.0):
    x = np.linspace(lo, hi, n)
    return x, np.full(n, float(value))


def test_multi_ops_add_subtract_multiply_divide_average():
    a, b = _xy(6.0), _xy(2.0)
    grid, y = cs.multi_op([a, b], "add")
    assert np.allclose(y, 8.0)
    _, y = cs.multi_op([a, b], "subtract")
    assert np.allclose(y, 4.0)
    _, y = cs.multi_op([a, b], "multiply")
    assert np.allclose(y, 12.0)
    _, y = cs.multi_op([a, b], "divide")
    assert np.allclose(y, 3.0)
    _, y = cs.multi_op([a, b], "average")
    assert np.allclose(y, 4.0)
    _, y = cs.multi_op([a, b], "weighted_sum", weights=[1.0, 0.5])
    assert np.allclose(y, 7.0)


def test_divide_by_near_zero_gives_nan_not_inf():
    a = _xy(1.0)
    x, y = _xy(0.0)
    _, out = cs.multi_op([a, (x, y)], "divide")
    assert np.all(np.isnan(out))


def test_multi_op_uses_overlap_grid():
    a = (np.linspace(0, 100, 50), np.full(50, 1.0))
    b = (np.linspace(50, 150, 50), np.full(50, 1.0))
    grid, y = cs.multi_op([a, b], "add")
    assert grid[0] >= 50.0 and grid[-1] <= 100.0


def test_modulated_addition_envelopes():
    a, b = _xy(0.0), _xy(10.0)
    # constant: everywhere +k·B
    _, y = cs.modulated_addition(a, b, envelope="constant", k=0.5)
    assert np.allclose(y, 5.0)
    # ramp: 0 before x1, k·B after x2
    grid, y = cs.modulated_addition(a, b, envelope="ramp", k=1.0, x1=40.0, x2=60.0)
    assert y[grid < 40].max() == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(y[grid > 60], 10.0)
    assert 0.0 < y[np.argmin(np.abs(grid - 50.0))] < 10.0
    # gaussian: peaks at the center, decays away
    grid, y = cs.modulated_addition(a, b, envelope="gaussian", k=1.0, center=50.0, width=5.0)
    assert y[np.argmin(np.abs(grid - 50.0))] == pytest.approx(10.0, rel=0.01)
    assert y[0] < 0.1


def test_transforms():
    x = np.linspace(1, 10, 10)
    y = x.copy()
    _, out = cs.transform(x, y, "scale_offset", factor=2.0, offset=1.0)
    assert np.allclose(out, 2 * y + 1)
    _, out = cs.transform(x, y, "normalize_max")
    assert out.max() == pytest.approx(1.0)
    _, out = cs.transform(x, y, "normalize_minmax")
    assert out.min() == pytest.approx(0.0) and out.max() == pytest.approx(1.0)
    _, out = cs.transform(x, y, "log10")
    assert out[9] == pytest.approx(1.0)
    _, out = cs.transform(x, y, "power", power=2.0)
    assert np.allclose(out, y ** 2)
    _, out = cs.transform(x, y, "reciprocal")
    assert np.allclose(out, 1.0 / y)
    x2, out = cs.transform(x, y, "x_shift", offset=5.0)
    assert x2[0] == pytest.approx(6.0)
    x2, _ = cs.transform(x, y, "x_scale", factor=2.0)
    assert x2[-1] == pytest.approx(20.0)


def test_log_of_nonpositive_is_nan():
    x = np.linspace(0, 5, 6)
    y = np.array([-1.0, 0.0, 1.0, 10.0, 100.0, 1000.0])
    _, out = cs.transform(x, y, "log10")
    assert np.isnan(out[0]) and np.isnan(out[1])
    assert out[3] == pytest.approx(1.0)


def test_crop_and_resample():
    x, y = _xy(1.0, n=101)
    cx, cy = cs.crop(x, y, xmin=20.0, xmax=40.0)
    assert cx[0] >= 20.0 and cx[-1] <= 40.0
    rx, ry = cs.resample(x, y, n_points=17)
    assert len(rx) == 17 and np.allclose(ry, 1.0)
    with pytest.raises(ValueError):
        cs.crop(x, y, xmin=40, xmax=20)


def test_smooth_reduces_noise_but_keeps_signal():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 100, 500)
    clean = np.sin(x / 8.0)
    noisy = clean + rng.normal(0, 0.2, x.shape)
    for method in ("savgol", "moving_average", "median"):
        _, out = cs.smooth(x, noisy, method=method, window=15)
        assert np.std(out - clean) < np.std(noisy - clean)


def test_despike_removes_spikes_and_leaves_signal():
    x = np.linspace(0, 100, 300)
    y = np.sin(x / 10.0)
    y_spiked = y.copy()
    y_spiked[[50, 150, 250]] += 40.0
    _, out = cs.despike(x, y_spiked, z=6.0)
    assert np.max(np.abs(out - y)) < 0.5  # spikes gone
    _, untouched = cs.despike(x, y, z=6.0)
    assert np.allclose(untouched, y, atol=0.2)  # clean data mostly untouched


def test_derivative_of_line_and_parabola():
    x = np.linspace(0, 10, 200)
    _, d1 = cs.derivative(x, 3.0 * x, order=1)
    assert np.allclose(d1[10:-10], 3.0, atol=0.01)
    _, d2 = cs.derivative(x, x ** 2, order=2)
    assert np.allclose(d2[10:-10], 2.0, atol=0.05)


def test_cumulative_integral_of_constant():
    x, y = _xy(2.0, n=101, lo=0.0, hi=10.0)
    _, integ = cs.cumulative_integral(x, y)
    assert integ[0] == 0.0
    assert integ[-1] == pytest.approx(20.0)


def test_statistics_report():
    x = np.linspace(0, 10, 101)
    y = np.exp(-0.5 * ((x - 4.0) / 1.0) ** 2)
    stats = cs.statistics_report(x, y)
    assert stats["apex_x"] == pytest.approx(4.0, abs=0.1)
    assert stats["centroid_x"] == pytest.approx(4.0, abs=0.1)
    assert stats["y_max"] == pytest.approx(1.0, abs=0.01)
    assert stats["n_points"] == 101


def test_linear_combination_fit_recovers_known_mixture():
    x = np.linspace(0, 100, 400)
    r1 = np.exp(-0.5 * ((x - 30) / 5.0) ** 2)
    r2 = np.exp(-0.5 * ((x - 70) / 8.0) ** 2)
    target = 0.7 * r1 + 0.3 * r2
    res = cs.linear_combination_fit((x, target), [(x, r1), (x, r2)], ref_names=["A", "B"])
    assert res.coefficients[0] == pytest.approx(0.7, abs=0.01)
    assert res.coefficients[1] == pytest.approx(0.3, abs=0.01)
    assert res.r_squared > 0.999
    assert res.names == ["A", "B"]


def test_lcf_non_negative_constraint():
    x = np.linspace(0, 100, 200)
    r1 = np.sin(x / 10.0) + 2.0
    target = -0.5 * r1  # only representable with a negative coefficient
    res_nn = cs.linear_combination_fit((x, target), [(x, r1)], non_negative=True)
    assert res_nn.coefficients[0] == 0.0  # clamped at zero
    res_free = cs.linear_combination_fit((x, target), [(x, r1)], non_negative=False)
    assert res_free.coefficients[0] == pytest.approx(-0.5, abs=0.01)


def test_registry_is_complete_and_consistent():
    assert len(cs.CALC_OPERATIONS) >= 25  # "every type of calculation"
    for label, spec in cs.CALC_OPERATIONS.items():
        assert spec["group"], label
        assert isinstance(spec["params"], list), label
