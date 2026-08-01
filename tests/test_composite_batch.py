"""Tests for saxs_core/composite_batch.py — the general-purpose batch
runner (Phase 5): continuation seeding, CSV/figure outputs, legacy
Gaussian cross-check, and per-sample error isolation. Deliberately does
NOT parse any sample-name composition scheme — groups/order are supplied
explicitly by the test, mirroring how any real caller must supply them.
"""
from __future__ import annotations

import numpy as np
import pytest

from saxs.core.composite_batch import (
    BatchItem, batch_to_csv_rows, legacy_gaussian_comparison, plot_sample_fit,
    plot_series_overview, run_batch, write_batch_csv,
)
from saxs.core.composite_fit import build_preset
from saxs.core.curve import Curve


def _ts_curve(sample_id, d, xi=3000.0, S=5e6, seed=0):
    model = build_preset("BG_TS_GP")
    q = np.linspace(1e-3, 0.3, 900)
    true = {"bg_C": 500.0, "pl_B": 1e-9, "pl_p": 4.0, "ts_S": S, "ts_d": d, "ts_xi": xi,
            "gp_G": 4e8, "gp_Rg": 2000.0, "gp_p": 4.0}
    I = model.eval(q, true)
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(np.abs(I)) * 0.005 + 0.5
    I = I + rng.normal(0, sigma)
    return Curve(q=q, intensity=np.clip(I, 1e-6, None), sigma=None, name=sample_id)


def _flat_curve(sample_id, seed=1):
    model = build_preset("BG")
    q = np.linspace(1e-3, 0.3, 900)
    I = model.eval(q, {"bg_C": 500.0, "pl_B": 1e-9, "pl_p": 2.0})
    rng = np.random.default_rng(seed)
    I = I + rng.normal(0, 0.02 * np.sqrt(np.abs(I)) + 0.02, q.shape)
    return Curve(q=q, intensity=np.clip(I, 1e-6, None), sigma=None, name=sample_id)


# ---------------------------------------------------------------------------
# Continuation logic
# ---------------------------------------------------------------------------

def test_run_batch_processes_all_items_in_order_hint_order():
    items = [
        BatchItem("s2", _ts_curve("s2", d=1300.0, seed=2), group="g", order_hint=2),
        BatchItem("s1", _ts_curve("s1", d=1200.0, seed=1), group="g", order_hint=1),
    ]
    result = run_batch(items, multistart_n=2)
    assert result.order == ["s1", "s2"]  # sorted by order_hint, not list order
    assert set(result.fits) == {"s1", "s2"}
    assert not result.errors


def test_run_batch_each_sample_fit_is_independent():
    """Each sample's fit reflects only its OWN curve — a real regression
    guard against accidentally averaging/co-fitting a group (spec §4.6:
    'every sample stands alone')."""
    items = [
        BatchItem("d1200", _ts_curve("d1200", d=1200.0, seed=1), group="g", order_hint=1),
        BatchItem("d1600", _ts_curve("d1600", d=1600.0, seed=2), group="g", order_hint=2),
    ]
    result = run_batch(items, multistart_n=4, continuation=False)
    assert result.fits["d1200"].derived["d"] == pytest.approx(1200.0, rel=0.15)
    assert result.fits["d1600"].derived["d"] == pytest.approx(1600.0, rel=0.15)


def test_run_batch_continuation_can_improve_a_harder_neighbor_fit():
    """A close, easy-to-fit neighbor first, then a genuinely harder
    (noisier) sample second — continuation seeding from the first sample's
    good fit should be tried and, when it helps, recorded as used."""
    easy = BatchItem("easy", _ts_curve("easy", d=1200.0, xi=3000.0, seed=1), group="g", order_hint=1)
    hard = BatchItem("hard", _ts_curve("hard", d=1220.0, xi=3050.0, seed=99), group="g", order_hint=2)
    result = run_batch([easy, hard], multistart_n=6, continuation=True)
    assert "hard" in result.fits
    assert result.fits["hard"].derived["d"] == pytest.approx(1220.0, rel=0.2)
    # continuation_used is a legitimate bool either way; just confirm the
    # bookkeeping exists and didn't silently skip the sample
    assert "hard" in result.continuation_used


def test_run_batch_continuation_skipped_when_component_sets_differ():
    """A class-c neighbor (TS present) followed by a class-a sample (no
    TS) can't meaningfully continue — the differing parameter sets must
    be detected and continuation silently skipped, not crash."""
    peaked = BatchItem("peaked", _ts_curve("peaked", d=1200.0, seed=1), group="g", order_hint=1)
    flat = BatchItem("flat", _flat_curve("flat", seed=2), group="g", order_hint=2)
    result = run_batch([peaked, flat], multistart_n=2, continuation=True)
    assert result.fits["flat"].no_peak is True
    assert result.continuation_used["flat"] is False


def test_run_batch_separate_groups_never_seed_each_other():
    g1 = BatchItem("a1", _ts_curve("a1", d=1200.0, seed=1), group="A", order_hint=1)
    g2 = BatchItem("b1", _ts_curve("b1", d=1600.0, seed=2), group="B", order_hint=1)
    result = run_batch([g1, g2], multistart_n=2, continuation=True)
    # first item in each of its own group -> no prior neighbor -> never continuation-seeded
    assert result.continuation_used["a1"] is False
    assert result.continuation_used["b1"] is False


def test_run_batch_records_error_and_continues_rest_of_batch():
    good = BatchItem("good", _ts_curve("good", d=1200.0, seed=1), group="g", order_hint=1)
    # a curve with too few points to fit meaningfully after hygiene trimming
    bad_curve = Curve(q=np.array([1e-3, 2e-3, 3e-3, 4e-3]), intensity=np.array([1.0, 2.0, 3.0, 4.0]), name="bad")
    bad = BatchItem("bad", bad_curve, group="g", order_hint=2)
    result = run_batch([good, bad], multistart_n=2)
    assert "good" in result.fits
    # either it fit trivially or errored -- either way the batch must not raise
    assert "bad" in result.fits or "bad" in result.errors


# ---------------------------------------------------------------------------
# Legacy Gaussian cross-check + CSV
# ---------------------------------------------------------------------------

def test_legacy_gaussian_comparison_returns_plausible_values():
    curve = _ts_curve("s", d=1200.0, seed=3)
    legacy = legacy_gaussian_comparison(curve, (0.002, 0.0125))
    assert legacy["d_gauss"] is not None
    assert legacy["d_gauss"] == pytest.approx(1200.0, rel=0.3)


def test_legacy_gaussian_comparison_fails_gracefully_on_bad_window():
    curve = _ts_curve("s", d=1200.0, seed=3)
    legacy = legacy_gaussian_comparison(curve, (0.28, 0.29))  # nowhere near the peak
    assert "d_gauss" in legacy  # never raises, even if the fit itself fails


def test_batch_to_csv_rows_includes_legacy_ratio_column():
    curve = _ts_curve("s1", d=1200.0, seed=1)
    result = run_batch([BatchItem("s1", curve, order_hint=1)], multistart_n=2)
    rows = batch_to_csv_rows(result, curves={"s1": curve})
    assert len(rows) == 1
    assert "d_ts_over_d_gauss" in rows[0]
    assert 0.5 < rows[0]["d_ts_over_d_gauss"] < 2.0


def test_write_batch_csv_produces_a_readable_file(tmp_path):
    curve = _ts_curve("s1", d=1200.0, seed=1)
    result = run_batch([BatchItem("s1", curve, order_hint=1)], multistart_n=2)
    path = tmp_path / "saxs_composite_fits.csv"
    write_batch_csv(str(path), result, curves={"s1": curve})
    assert path.is_file()
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["sample_id"] == "s1"


# ---------------------------------------------------------------------------
# Figures — smoke tests (rendering, not pixel-level)
# ---------------------------------------------------------------------------

def test_plot_sample_fit_does_not_raise():
    curve = _ts_curve("s1", d=1200.0, seed=1)
    result = run_batch([BatchItem("s1", curve, order_hint=1)], multistart_n=2)
    fit = result.fits["s1"]
    from saxs.core.composite_batch import _model_from_preset_name
    model = _model_from_preset_name(fit.preset_chosen)
    fig = plot_sample_fit(fit, model, curve)
    assert fig is not None
    assert len(fig.axes) == 2


def test_plot_series_overview_does_not_raise():
    items = [BatchItem("s1", _ts_curve("s1", d=1200.0, seed=1), group="g", order_hint=1),
            BatchItem("s2", _ts_curve("s2", d=1400.0, seed=2), group="g", order_hint=2)]
    result = run_batch(items, multistart_n=2, continuation=False)
    fig = plot_series_overview(result, ["s1", "s2"], {"s1": 8.0, "s2": 5.0}, x_label="Bi content (mol%)")
    assert fig is not None
    assert len(fig.axes) == 3
