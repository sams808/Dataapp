"""Tests for baseline_science.py — rampy-backed baseline subtraction."""
from __future__ import annotations

import numpy as np
import pytest
import rampy as rp

import processing.baseline_science as bs


def _spectrum_with_linear_baseline():
    x = np.linspace(100, 1000, 600)
    baseline = 0.02 * x + 5.0
    peak = rp.gaussian(x, 100.0, 550.0, 15.0)
    return x, baseline + peak, baseline


def test_parse_roi_text_variants():
    roi = bs.parse_roi_text("100-200; 500-600")
    assert roi.shape == (2, 2)
    assert roi[0][0] == 100.0 and roi[1][1] == 600.0

    roi2 = bs.parse_roi_text("100, 200\n500 , 600")
    assert np.allclose(roi, roi2)

    assert bs.parse_roi_text("") is None
    assert bs.parse_roi_text("   ") is None


def test_parse_roi_text_rejects_bad_input():
    with pytest.raises(ValueError, match="two numbers"):
        bs.parse_roi_text("100")
    with pytest.raises(ValueError, match="max must exceed min"):
        bs.parse_roi_text("300-200")


def test_poly_baseline_recovers_linear_background():
    x, y, true_base = _spectrum_with_linear_baseline()
    roi = np.array([[100.0, 450.0], [700.0, 1000.0]])  # peak-free regions
    x_c, y_sub, base = bs.compute_baseline(x, y, method="poly", roi=roi, params={"polynomial_order": 1})
    # Baseline recovered to within a small tolerance in the ROI regions.
    mask = (x_c < 450) | (x_c > 700)
    assert np.abs(base[mask] - true_base[mask]).max() < 0.5
    # Subtracted spectrum's peak survives.
    assert y_sub.max() == pytest.approx(100.0, rel=0.05)


def test_arpls_runs_without_roi():
    x, y, true_base = _spectrum_with_linear_baseline()
    x_c, y_sub, base = bs.compute_baseline(x, y, method="arPLS", params={"lam": 1e5, "ratio": 0.01})
    assert y_sub.shape == x_c.shape
    assert base.shape == x_c.shape
    # Peak region should stay strongly positive after subtraction.
    peak_region = (x_c > 500) & (x_c < 600)
    assert y_sub[peak_region].max() > 50


def test_roi_required_methods_raise_without_roi():
    x, y, _ = _spectrum_with_linear_baseline()
    with pytest.raises(ValueError, match="needs at least one baseline region"):
        bs.compute_baseline(x, y, method="poly", roi=None)


def test_roi_fully_outside_range_raises():
    x, y, _ = _spectrum_with_linear_baseline()
    with pytest.raises(ValueError, match="outside the spectrum"):
        bs.compute_baseline(x, y, method="poly", roi=np.array([[2000.0, 3000.0]]))


def test_unknown_method_raises():
    x, y, _ = _spectrum_with_linear_baseline()
    with pytest.raises(ValueError, match="Unknown baseline method"):
        bs.compute_baseline(x, y, method="magic", roi=np.array([[100.0, 200.0]]))
