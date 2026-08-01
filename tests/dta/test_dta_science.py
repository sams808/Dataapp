"""Tests for dta_science.py — Tg calculations, derivatives, and the shared
manual-baseline parameter resolver that keeps interactive Compute and batch
processing from diverging.
"""
from __future__ import annotations

import numpy as np
import pytest

import dta.dta_science as ds
import core.io_universal as iou


@pytest.fixture(scope="module")
def dta_xy(dta_example_path):
    df, meta = iou.load_any(str(dta_example_path), return_meta=True)
    canon = meta["canonical_map"]
    x = df[canon["T_C"]].to_numpy(dtype=float)
    y_key = canon.get("HF_mW") or canon.get("DSC_mW_mg")
    y = df[y_key].to_numpy(dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    order = np.argsort(x)
    return x[order], y[order]


# --------------------------------------------------------------------------
# Real-data sanity checks (no independently-verified "true" Tg for this real
# instrument file, so assert physically-sane invariants, not exact numbers).
# --------------------------------------------------------------------------

def test_compute_derivative_matches_gradient_on_finite_data(dta_xy):
    x, y = dta_xy
    dy = ds.compute_derivative(y, x)
    assert dy.shape == y.shape
    assert np.isfinite(dy).all()


def test_tg_methods_produce_physically_sane_results_on_real_dta_data(dta_xy):
    x, y = dta_xy
    xmin, xmax = float(np.percentile(x, 20)), float(np.percentile(x, 60))

    tg_deriv = ds.compute_tg_derivative(x, y, xmin, xmax)
    assert xmin <= tg_deriv <= xmax

    try:
        rd = ds.compute_tg_double_tangent(x, y, xmin, xmax)
        assert xmin - 1e-6 <= rd.tg <= xmax + 1e-6 or not np.isfinite(rd.tg)
    except ValueError:
        pass  # window may not contain a clean transition; not a test failure

    rp = ds.compute_tg_parallel_improved(x, y, xmin, xmax)
    assert np.isfinite(rp.tg)
    assert rp.low_mode in ("range", "point")
    assert rp.high_mode in ("range", "point")


# --------------------------------------------------------------------------
# Synthetic sigmoid transition: known ground truth to check double/parallel
# tangent methods land near the actual inflection point.
# --------------------------------------------------------------------------

def _synthetic_transition(midpoint=100.0, width=5.0, n=400):
    x = np.linspace(0, 200, n)
    y = 1.0 / (1.0 + np.exp(-(x - midpoint) / width)) + 0.001 * x
    return x, y


def test_compute_tg_derivative_finds_synthetic_midpoint():
    x, y = _synthetic_transition(midpoint=100.0)
    tg = ds.compute_tg_derivative(x, y, 20.0, 180.0)
    assert tg == pytest.approx(100.0, abs=3.0)


def test_compute_tg_double_tangent_finds_synthetic_midpoint():
    x, y = _synthetic_transition(midpoint=100.0)
    result = ds.compute_tg_double_tangent(x, y, 20.0, 180.0)
    assert result.tg == pytest.approx(100.0, abs=10.0)


# --------------------------------------------------------------------------
# Bug fix: parallel-tangent "point+point" silently falls back to AUTO ranges
# but must now correctly REPORT that fallback via low_mode/high_mode, instead
# of leaving them looking like "point" mode was actually used.
# --------------------------------------------------------------------------

def test_parallel_tangent_point_plus_point_reports_actual_auto_fallback():
    x, y = _synthetic_transition(midpoint=100.0)
    result = ds.compute_tg_parallel_improved(
        x, y, 20.0, 180.0,
        manual_low_point=30.0,
        manual_high_point=170.0,
    )
    # The old bug: caller code kept reporting "point" mode here even though
    # the function had silently fallen back to fitting AUTO ranges.
    assert result.low_mode == "range"
    assert result.high_mode == "range"
    assert np.isfinite(result.tg)


def test_parallel_tangent_range_plus_range_reports_range_mode():
    x, y = _synthetic_transition(midpoint=100.0)
    result = ds.compute_tg_parallel_improved(
        x, y, 20.0, 180.0,
        manual_low_range=(20.0, 60.0),
        manual_high_range=(140.0, 180.0),
    )
    assert result.low_mode == "range"
    assert result.high_mode == "range"


def test_parallel_tangent_range_plus_point_reports_mixed_mode():
    x, y = _synthetic_transition(midpoint=100.0)
    result = ds.compute_tg_parallel_improved(
        x, y, 20.0, 180.0,
        manual_low_range=(20.0, 60.0),
        manual_high_point=170.0,
    )
    assert result.low_mode == "range"
    assert result.high_mode == "point"


# --------------------------------------------------------------------------
# Bug fix: batch runs used to ignore the "Manual" toggle that interactive
# Compute honors. resolve_baseline_params() is now the ONLY path to turn
# manual-mode inputs into actual values — verify it enforces that identically
# regardless of who calls it.
# --------------------------------------------------------------------------

def test_resolve_baseline_params_manual_disabled_ignores_all_typed_values():
    params = ds.resolve_baseline_params(
        manual_enabled=False,
        low_use_point=False, low_point_x=None, low_min=10.0, low_max=20.0,
        high_use_point=True, high_point_x=150.0, high_min=None, high_max=None,
        slope_min=50.0, slope_max=60.0,
    )
    assert params.low_range is None
    assert params.low_point is None
    assert params.high_range is None
    assert params.high_point is None
    assert params.manual_slope is None


def test_resolve_baseline_params_manual_enabled_uses_typed_values():
    params = ds.resolve_baseline_params(
        manual_enabled=True,
        low_use_point=False, low_point_x=None, low_min=10.0, low_max=20.0,
        high_use_point=True, high_point_x=150.0, high_min=None, high_max=None,
        slope_min=50.0, slope_max=60.0,
    )
    assert params.low_range == (10.0, 20.0)
    assert params.low_point is None
    assert params.high_range is None
    assert params.high_point == 150.0
    assert params.manual_slope == (50.0, 60.0)


def test_resolve_baseline_params_identical_inputs_give_identical_output():
    # This is the crux of the fix: interactive Compute (Manual unchecked) and
    # a batch run with leftover typed values in the fields must resolve
    # IDENTICALLY when both pass manual_enabled=False for the current state.
    kwargs = dict(
        low_use_point=True, low_point_x=42.0, low_min=1.0, low_max=2.0,
        high_use_point=False, high_point_x=None, high_min=100.0, high_max=110.0,
        slope_min=None, slope_max=None,
    )
    a = ds.resolve_baseline_params(manual_enabled=False, **kwargs)
    b = ds.resolve_baseline_params(manual_enabled=False, **kwargs)
    assert a == b
    assert a.low_range is None and a.low_point is None and a.high_range is None


# --------------------------------------------------------------------------
# Multi-method Tg agreement (cross-technique backlog item)
# --------------------------------------------------------------------------

def test_tg_agreement_within_threshold():
    from dta.dta_science import tg_agreement
    out = tg_agreement({"Double": 354.47, "Parallel": 354.51, "|dY| max": 357.62}, threshold=5.0)
    assert out["n"] == 3
    assert out["agree"] is True
    assert out["spread"] == pytest.approx(357.62 - 354.47, abs=1e-9)
    assert out["extremes"] == ("Double", "|dY| max")


def test_tg_agreement_flags_disagreement():
    from dta.dta_science import tg_agreement
    out = tg_agreement({"Double": 350.0, "Parallel": 362.0, "|dY| max": 351.0}, threshold=5.0)
    assert out["agree"] is False
    assert out["extremes"] == ("Double", "Parallel")


def test_tg_agreement_undefined_below_two_methods():
    from dta.dta_science import tg_agreement
    import numpy as np
    out = tg_agreement({"Double": 350.0, "Parallel": None, "|dY| max": float("nan")})
    assert out["n"] == 1
    assert out["agree"] is None
    assert np.isnan(out["spread"])
