"""Tests for htxrd_science.py (M20) — HTXRD series processing.

Pure-Python (no Qt), runs in the default suite. Uses the real bundled
EXAMPLES/HTXRD_example.rasx (one scan from the user's own P5Bi8-12
room-to-900C series) plus synthetic multi-pattern series built from known
Gaussians for the tracking/flagging logic.
"""
from __future__ import annotations

import numpy as np
import pytest
import rampy as rp

import xrd.htxrd_science as hs
from conftest import EXAMPLES_DIR


HTXRD_EXAMPLE = EXAMPLES_DIR / "HTXRD_example.rasx"


# --------------------------------------------------------------------------
# Filename-template ramp derivation (Jana-style ??? rewriting)
# --------------------------------------------------------------------------

def test_ramp_value_from_template_basic():
    assert hs.ramp_value_from_template("NB-LM01MO_100.XRDML", "NB-LM01MO_???.XRDML") == 100.0
    assert hs.ramp_value_from_template("scan_025.xy", "scan_???.xy") == 25.0


def test_ramp_value_from_template_case_insensitive_and_path_stripped():
    assert hs.ramp_value_from_template(r"C:\data\SCAN_300.XY", "scan_???.xy") == 300.0


def test_ramp_value_from_template_non_matching_returns_none():
    assert hs.ramp_value_from_template("other_100.xy", "scan_???.xy") is None
    assert hs.ramp_value_from_template("scan_abc.xy", "scan_???.xy") is None
    assert hs.ramp_value_from_template("scan_100.xy", "no_wildcards.xy") is None


# --------------------------------------------------------------------------
# Real .rasx loading
# --------------------------------------------------------------------------

def test_load_pattern_real_rasx_gets_temperature_from_metadata():
    pat = hs.load_pattern(str(HTXRD_EXAMPLE))
    assert pat.ramp_source == "metadata"
    assert pat.ramp_value is not None
    assert 0.0 < pat.ramp_value < 1000.0  # a physical temperature for a RT-900C series
    assert len(pat.x) > 100
    assert np.all(np.diff(pat.x) >= 0)  # sorted 2theta


def test_load_series_falls_back_to_index_without_metadata_or_template(tmp_path):
    # Two plain .xy files with no temperature info anywhere.
    x = np.linspace(10, 60, 200)
    for i, name in enumerate(["b.xy", "a.xy"]):
        y = rp.gaussian(x, 100.0, 30.0, 0.5)
        np.savetxt(tmp_path / name, np.column_stack([x, y]))
    series = hs.load_series([str(tmp_path / "b.xy"), str(tmp_path / "a.xy")])
    assert [p.ramp_source for p in series] == ["index", "index"]
    assert [p.ramp_value for p in series] == [0.0, 1.0]


def test_load_series_uses_filename_template(tmp_path):
    x = np.linspace(10, 60, 200)
    y = rp.gaussian(x, 100.0, 30.0, 0.5)
    for t in (300, 100, 200):
        np.savetxt(tmp_path / f"scan_{t}.xy", np.column_stack([x, y]))
    paths = [str(tmp_path / f"scan_{t}.xy") for t in (300, 100, 200)]
    series = hs.load_series(paths, filename_template="scan_???.xy")
    assert [p.ramp_value for p in series] == [100.0, 200.0, 300.0]  # sorted by ramp
    assert all(p.ramp_source == "filename" for p in series)


def test_find_series_files_filters_and_sorts(tmp_path):
    (tmp_path / "b.rasx").write_bytes(b"zip-ish")
    (tmp_path / "a.xy").write_text("1 2")
    (tmp_path / "notes.md").write_text("not a pattern")
    files = hs.find_series_files(str(tmp_path))
    names = [f.split("\\")[-1].split("/")[-1] for f in files]
    assert names == ["a.xy", "b.rasx"]


# --------------------------------------------------------------------------
# Peak tracking on a synthetic temperature series with KNOWN behavior:
# center shifts linearly with T (thermal expansion analog), constant FWHM.
# --------------------------------------------------------------------------

def _synthetic_series(centers_by_temp):
    x = np.linspace(20, 40, 800)
    rng = np.random.default_rng(0)
    series = []
    for temp, center in centers_by_temp:
        y = rp.gaussian(x, 500.0, center, 0.3) + 50.0 + rng.normal(0, 2.0, x.shape)
        series.append(hs.HtxrdPattern(path="", name=f"T{temp}", x=x, y=y, ramp_value=float(temp), ramp_source="metadata"))
    return series


def test_track_peak_recovers_linear_center_shift():
    temps_centers = [(100, 30.00), (200, 29.95), (300, 29.90), (400, 29.85)]
    series = _synthetic_series(temps_centers)
    results = hs.track_peak(series, window_lo=29.0, window_hi=31.0)

    assert len(results) == 4
    assert all(r.error is None for r in results)
    for (temp, true_center), r in zip(temps_centers, results):
        assert r.ramp_value == temp
        assert r.center == pytest.approx(true_center, abs=0.02)
        assert r.fwhm == pytest.approx(0.3, abs=0.05)


def test_track_peak_reports_error_row_for_empty_window():
    series = _synthetic_series([(100, 30.0)])
    results = hs.track_peak(series, window_lo=50.0, window_hi=55.0)  # no data there
    assert len(results) == 1
    assert results[0].error is not None
    assert np.isnan(results[0].center)


def test_track_peak_rejects_inverted_window():
    with pytest.raises(ValueError):
        hs.track_peak([], window_lo=31.0, window_hi=29.0)


# --------------------------------------------------------------------------
# Transition flagging
# --------------------------------------------------------------------------

def test_flag_transition_candidates_flags_anomalous_chi2():
    results = [
        hs.PeakTrackResult("a", 100.0, 30.0, 0.3, 500.0, 100.0, 1.0),
        hs.PeakTrackResult("b", 200.0, 30.0, 0.3, 500.0, 100.0, 1.1),
        hs.PeakTrackResult("c", 300.0, 30.0, 0.3, 500.0, 100.0, 0.9),
        hs.PeakTrackResult("d", 400.0, 30.0, 0.3, 500.0, 100.0, 50.0),  # transition!
        hs.PeakTrackResult("e", 500.0, 30.0, 0.3, 500.0, 100.0, 1.05),
    ]
    flags = hs.flag_transition_candidates(results, z=3.0)
    flagged_temps = [f[0] for f in flags]
    assert 400.0 in flagged_temps
    assert 100.0 not in flagged_temps


def test_flag_transition_candidates_includes_failed_fits():
    results = [
        hs.PeakTrackResult("a", 100.0, 30.0, 0.3, 500.0, 100.0, 1.0),
        hs.PeakTrackResult("b", 200.0, np.nan, np.nan, np.nan, np.nan, np.nan, error="peak vanished"),
    ]
    flags = hs.flag_transition_candidates(results)
    assert any("fit failed" in f[2] for f in flags)
    assert any(f[0] == 200.0 for f in flags)


def test_flag_transition_candidates_quiet_series_returns_no_chi2_flags():
    results = [hs.PeakTrackResult(f"p{i}", 100.0 * i, 30.0, 0.3, 500.0, 100.0, 1.0 + 0.01 * i) for i in range(5)]
    flags = hs.flag_transition_candidates(results, z=3.0)
    assert flags == []


# --------------------------------------------------------------------------
# Tracking rework (user feedback: "not effective with several peaks, or when
# they disappear or appear"): anchored seeding, sequential following,
# absence detection, multi-window runs, and the notebook-ported map helpers.
# --------------------------------------------------------------------------

def _two_peak_series():
    """Strong static peak at 28.6; weaker peak drifting 29.4 -> 29.9.
    Both inside one wide window: the old window-max seeding locked onto the
    strong one no matter what the user wanted."""
    x = np.linspace(27, 31, 1200)
    rng = np.random.default_rng(1)
    series = []
    for i, temp in enumerate((100, 200, 300, 400, 500, 600)):
        drift_center = 29.4 + 0.1 * i
        y = (rp.gaussian(x, 900.0, 28.6, 0.25) + rp.gaussian(x, 250.0, drift_center, 0.20)
             + 40.0 + rng.normal(0, 2.0, x.shape))
        series.append(hs.HtxrdPattern(path="", name=f"T{temp}", x=x, y=y,
                                      ramp_value=float(temp), ramp_source="metadata"))
    return series


def test_track_peak_anchor_tracks_the_intended_weaker_peak():
    series = _two_peak_series()
    results = hs.track_peak(series, window_lo=28.0, window_hi=30.5, initial_center=29.4)
    assert all(r.error is None and r.present for r in results)
    for i, r in enumerate(results):
        assert r.center == pytest.approx(29.4 + 0.1 * i, abs=0.05)


def test_track_peak_without_anchor_seeds_on_strongest():
    series = _two_peak_series()
    results = hs.track_peak(series, window_lo=28.0, window_hi=30.5)
    assert results[0].center == pytest.approx(28.6, abs=0.05)  # the strong static one


def _vanishing_series():
    """Peak present at 100-300, gone at 400-500, back at 600 (e.g. a phase
    that melts and recrystallizes)."""
    x = np.linspace(27, 31, 1000)
    rng = np.random.default_rng(2)
    series = []
    for temp in (100, 200, 300, 400, 500, 600):
        amp = 600.0 if temp not in (400, 500) else 0.0
        y = rp.gaussian(x, amp, 29.0, 0.25) + 40.0 + rng.normal(0, 2.5, x.shape)
        series.append(hs.HtxrdPattern(path="", name=f"T{temp}", x=x, y=y,
                                      ramp_value=float(temp), ramp_source="metadata"))
    return series


def test_track_peak_marks_absent_instead_of_fitting_noise():
    series = _vanishing_series()
    results = hs.track_peak(series, window_lo=28.0, window_hi=30.0, absence_sigma=5.0)
    present = [r.present for r in results]
    assert present == [True, True, True, False, False, True]
    for r in results:
        assert r.error is None  # absence is a result, not a failure
    assert np.isnan(results[3].center)  # no garbage values for the gap
    # after the gap the peak is re-acquired at the right position
    assert results[5].center == pytest.approx(29.0, abs=0.05)


def test_flag_transition_candidates_reports_vanish_and_reappear():
    series = _vanishing_series()
    results = hs.track_peak(series, window_lo=28.0, window_hi=30.0, absence_sigma=5.0)
    flags = hs.flag_transition_candidates(results)
    reasons = " | ".join(f[2] for f in flags)
    assert "vanished at 400" in reasons
    assert "appeared at 600" in reasons


def test_parse_track_windows_with_anchors_and_errors():
    ws = hs.parse_track_windows("28.5-29.5 @ 28.98; 31-32")
    assert ws == [{"lo": 28.5, "hi": 29.5, "center": 28.98}, {"lo": 31.0, "hi": 32.0, "center": None}]
    with pytest.raises(ValueError, match="hi must exceed"):
        hs.parse_track_windows("29-28")
    with pytest.raises(ValueError, match="outside window"):
        hs.parse_track_windows("28-29 @ 31")
    with pytest.raises(ValueError, match="Cannot parse"):
        hs.parse_track_windows("banana")


def test_track_peaks_multi_returns_one_series_per_window():
    series = _two_peak_series()
    out = hs.track_peaks_multi(series, hs.parse_track_windows("28.2-29.0; 29.1-30.2 @ 29.4"))
    assert set(out.keys()) == {"28.2-29", "29.1-30.2 @ 29.4"}
    static = out["28.2-29"]
    drifting = out["29.1-30.2 @ 29.4"]
    assert all(r.center == pytest.approx(28.6, abs=0.05) for r in static)
    assert drifting[-1].center == pytest.approx(29.9, abs=0.05)
    assert all(r.window_label for r in static + drifting)


# --------------------------------------------------------------------------
# Map helpers ported from the user's XRD_HT.ipynb
# --------------------------------------------------------------------------

def test_common_grid_and_intensity_map():
    series = _synthetic_series([(100, 30.0), (200, 29.9)])
    grid = hs.build_common_grid(series, npts=500)
    assert grid[0] >= 20.0 and grid[-1] <= 40.0 and len(grid) == 500
    m = hs.build_intensity_map(series, grid)
    assert m.shape == (2, 500)
    mn = hs.build_intensity_map(series, grid, normalize=True)
    assert np.nanmax(mn) == pytest.approx(1.0)


def test_reference_index_forms():
    series = _synthetic_series([(100, 30.0), (200, 29.9), (300, 29.8)])
    assert hs.reference_index(series, "first") == 0
    assert hs.reference_index(series, 2) == 2
    assert hs.reference_index(series, 195.0) == 1  # nearest temperature
    with pytest.raises(ValueError):
        hs.reference_index(series, 7)


def test_relative_time_axis_from_heating_rate():
    t = hs.compute_relative_time_minutes([100.0, 200.0, 300.0], 10.0)
    assert np.allclose(t, [0.0, 10.0, 20.0])
    assert hs.compute_relative_time_minutes([100.0], None) is None


def test_peak_guides_parse_and_evaluate():
    guides = hs.parse_peak_guides(["{1:21.2; 5:19.6}", "", "{2:30; 4:31}"])
    assert len(guides) == 2
    slices, xpos = hs.evaluate_peak_guide(guides[0], n_slices=5)
    assert np.allclose(xpos, np.linspace(21.2, 19.6, 5))
    slices, xpos = hs.evaluate_peak_guide(guides[1], n_slices=5)
    assert np.isnan(xpos[0]) and np.isnan(xpos[4])
    assert xpos[2] == pytest.approx(30.5)
    with pytest.raises(ValueError, match="Invalid guide"):
        hs.parse_peak_guides(["21.2; 19.6"])


# --------------------------------------------------------------------------
# Guided tracking (pick a start and an end point on the waterfall) and
# automatic tracking setup — the second feedback wave's UX rework.
# --------------------------------------------------------------------------

def test_guide_centers_interpolate_and_clamp():
    centers = hs.guide_centers_for_patterns([(2, 29.0), (4, 29.4)], n_patterns=6)
    assert np.allclose(centers, [29.0, 29.0, 29.2, 29.4, 29.4, 29.4])  # flat beyond anchors
    with pytest.raises(ValueError):
        hs.guide_centers_for_patterns([], 5)


def test_track_peak_guided_follows_picked_drift_next_to_stronger_neighbor():
    series = _two_peak_series()  # weak peak drifts 29.4 -> 29.9 beside a strong 28.6 one
    # two clicks: near the weak peak on the first and last patterns
    results = hs.track_peak_guided(series, [(1, 29.4), (6, 29.9)], half_window=0.3)
    assert all(r.error is None and r.present for r in results)
    for i, r in enumerate(results):
        assert r.center == pytest.approx(29.4 + 0.1 * i, abs=0.05)


def test_track_peak_guided_intermediate_anchor_handles_appearing_peak():
    """One click at an intermediate temperature where the peak exists must
    give a full-series guide: absent before, present after."""
    x = np.linspace(27, 31, 1000)
    rng = np.random.default_rng(3)
    series = []
    for temp in (100, 200, 300, 400, 500, 600):
        amp = 500.0 if temp >= 400 else 0.0  # peak appears at 400
        y = rp.gaussian(x, amp, 29.2, 0.2) + 40.0 + rng.normal(0, 2.5, x.shape)
        series.append(hs.HtxrdPattern(path="", name=f"T{temp}", x=x, y=y,
                                      ramp_value=float(temp), ramp_source="metadata"))
    # single pick on slice 5 (500 degC), where the peak clearly exists
    results = hs.track_peak_guided(series, [(5, 29.2)], half_window=0.3, absence_sigma=5.0)
    assert [r.present for r in results] == [False, False, False, True, True, True]
    assert results[4].center == pytest.approx(29.2, abs=0.03)
    flags = hs.flag_transition_candidates(results)
    assert any("appeared at 400" in f[2] for f in flags)


def test_auto_track_windows_builds_anchored_windows_around_each_peak():
    x = np.linspace(20, 40, 2000)
    rng = np.random.default_rng(4)
    y = (rp.gaussian(x, 900.0, 28.6, 0.15) + rp.gaussian(x, 400.0, 33.2, 0.2)
         + 60.0 + rng.normal(0, 3.0, x.shape))
    pat = hs.HtxrdPattern(path="", name="ref", x=x, y=y, ramp_value=25.0)

    windows = hs.auto_track_windows(pat)
    anchors = [w["center"] for w in windows]
    assert any(abs(a - 28.6) < 0.1 for a in anchors)
    assert any(abs(a - 33.2) < 0.1 for a in anchors)
    for w in windows:
        assert w["lo"] < w["center"] < w["hi"]
        assert (w["hi"] - w["lo"]) < 3.0  # local widths, not pattern-scale ones
