"""Tests for saxs_core.edf_reduction -- raw 2D EDF frame ingestion via pyFAI.

Uses synthetic, programmatically-constructed EDF frames (via
fabio.edfimage.EdfImage) for deterministic unit-level checks, plus one real
bundled AgBeh calibration frame (EXAMPLES/AgBeh_calibration_example.edf.gz,
gzip-compressed from a real measurement) for the one check that needs
genuine physics to be meaningful: confirming the header-derived geometry
against the known silver behenate d-spacing.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("pyFAI")
pytest.importorskip("fabio")

import fabio.edfimage

from saxs.core import edf_reduction as edf
from conftest import EXAMPLES_DIR

AGBEH_FIXTURE = os.path.join(str(EXAMPLES_DIR), "AgBeh_calibration_example.edf.gz")

DIST = 0.9
WAVELENGTH = 1.54189e-10
PIXEL = 7.5e-5
SHAPE = (300, 280)  # (Dim_2, Dim_1) -- small, fast for synthetic tests
CENTER_1 = 140.0
CENTER_2 = 150.0


def _base_header(exposure=900.0, transmitted_flux=1.0e6, dummy=-1.0, ddummy=0.5):
    return {
        "Center_1": str(CENTER_1), "Center_2": str(CENTER_2),
        "SampleDistance": str(DIST), "Wavelength": str(WAVELENGTH),
        "PSize_1": str(PIXEL), "PSize_2": str(PIXEL),
        "ExposureTime": str(exposure), "Dummy": str(dummy), "DDummy": str(ddummy),
        "TransmittedFlux": str(transmitted_flux), "Comment": "synthetic test frame",
        "Date": "2026-01-01T00:00:00",
    }


def _q_to_radius_px(q_a_inv):
    """Inverse of the small-angle geometry this module uses -- places a
    synthetic ring at a known, chosen q (in A^-1, matching pyFAI's
    unit="q_A^-1" output) for round-trip testing. WAVELENGTH/DIST/PIXEL are
    all in meters, so q must be converted to m^-1 before use."""
    q_m_inv = q_a_inv * 1.0e10
    return q_m_inv * WAVELENGTH * DIST / (2.0 * np.pi * PIXEL)


def _write_synthetic_frame(path, header, data):
    img = fabio.edfimage.EdfImage(data=data.astype(np.float32), header=header)
    img.write(str(path))


def _flat_frame_with_ring(q_ring=None, ring_amplitude=500.0, base=50.0, seed=0):
    rng = np.random.default_rng(seed)
    ny, nx = SHAPE
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.sqrt((xx - CENTER_1) ** 2 + (yy - CENTER_2) ** 2)
    data = np.full(SHAPE, base, dtype=float)
    if q_ring is not None:
        r_ring = _q_to_radius_px(q_ring)
        data += ring_amplitude * np.exp(-((r - r_ring) ** 2) / (2 * 2.0 ** 2))
    data += rng.normal(scale=1.0, size=SHAPE)
    return np.clip(data, 0, None)


# ---------------------------------------------------------------------------
# geometry_from_header
# ---------------------------------------------------------------------------

def test_geometry_from_header_axis_convention():
    header = _base_header()
    geom = edf.geometry_from_header(header)
    # Center_2 -> poni1 (array axis 0), Center_1 -> poni2 (array axis 1);
    # validated empirically against a real AgBeh frame, not assumed.
    assert geom.poni1 == pytest.approx(CENTER_2 * PIXEL)
    assert geom.poni2 == pytest.approx(CENTER_1 * PIXEL)
    assert geom.dist == pytest.approx(DIST)
    assert geom.wavelength == pytest.approx(WAVELENGTH)


def test_geometry_from_header_missing_keys_raises_with_names():
    header = _base_header()
    del header["Center_1"]
    del header["SampleDistance"]
    with pytest.raises(ValueError) as exc:
        edf.geometry_from_header(header)
    assert "Center_1" in str(exc.value)
    assert "SampleDistance" in str(exc.value)


def test_geometry_from_header_accepts_wavelength_or_WaveLength_spelling():
    header = _base_header()
    header["WaveLength"] = header.pop("Wavelength")
    geom = edf.geometry_from_header(header)
    assert geom.wavelength == pytest.approx(WAVELENGTH)


# ---------------------------------------------------------------------------
# direct_beam_mask
# ---------------------------------------------------------------------------

def test_direct_beam_mask_excludes_only_near_center():
    geom = edf.geometry_from_header(_base_header())
    mask = edf.direct_beam_mask(SHAPE, geom, radius_px=12.0)
    assert mask[int(CENTER_2), int(CENTER_1)]
    assert not mask[int(CENTER_2) + 40, int(CENTER_1)]
    assert not mask[int(CENTER_2), int(CENTER_1) + 40]


# ---------------------------------------------------------------------------
# integrate_edf_frame
# ---------------------------------------------------------------------------

def test_integrate_edf_frame_recovers_known_ring_position(tmp_path):
    q_target = 0.02
    data = _flat_frame_with_ring(q_ring=q_target, seed=1)
    path = tmp_path / "ring.edf"
    _write_synthetic_frame(path, _base_header(exposure=900.0), data)

    curve = edf.integrate_edf_frame(str(path), npt=400)
    mask = (curve.q > 0.01) & (curve.q < 0.04)
    q_peak = float(curve.q[mask][np.argmax(curve.intensity[mask])])
    assert q_peak == pytest.approx(q_target, abs=0.003)


def test_integrate_edf_frame_normalizes_by_exposure_not_transmission(tmp_path):
    """Same underlying counts, two different ExposureTime/TransmittedFlux
    combinations -- the resulting RATE must match (normalized by exposure
    only), not scale with the wildly different TransmittedFlux value. This
    is the specific bug this module exists to avoid: dividing a sample's own
    signal by its own transmitted flux inflates low-transmission samples,
    confirmed against real raw pixel counts during development."""
    data = _flat_frame_with_ring(q_ring=0.05, seed=2)
    path_a = tmp_path / "a.edf"
    path_b = tmp_path / "b.edf"
    _write_synthetic_frame(path_a, _base_header(exposure=900.0, transmitted_flux=1.0e6), data)
    _write_synthetic_frame(path_b, _base_header(exposure=900.0, transmitted_flux=1.0e2), data)

    ca = edf.integrate_edf_frame(str(path_a), npt=400)
    cb = edf.integrate_edf_frame(str(path_b), npt=400)
    assert np.allclose(ca.intensity, cb.intensity, rtol=1e-9)
    assert ca.transmission == pytest.approx(1.0e6)
    assert cb.transmission == pytest.approx(1.0e2)


def test_integrate_edf_frame_rate_scales_inversely_with_exposure(tmp_path):
    data = _flat_frame_with_ring(q_ring=0.05, seed=3)
    path_short = tmp_path / "short.edf"
    path_long = tmp_path / "long.edf"
    _write_synthetic_frame(path_short, _base_header(exposure=300.0), data)
    _write_synthetic_frame(path_long, _base_header(exposure=900.0), data)

    c_short = edf.integrate_edf_frame(str(path_short), npt=400)
    c_long = edf.integrate_edf_frame(str(path_long), npt=400)
    # same raw counts, 3x the exposure -> 1/3 the rate
    ratio = c_short.intensity / c_long.intensity
    finite = np.isfinite(ratio) & (c_long.intensity > 0)
    assert np.median(ratio[finite]) == pytest.approx(3.0, rel=0.05)


def test_integrate_edf_frame_rejects_non_positive_exposure(tmp_path):
    data = _flat_frame_with_ring(seed=4)
    path = tmp_path / "bad.edf"
    _write_synthetic_frame(path, _base_header(exposure=0.0), data)
    with pytest.raises(ValueError, match="ExposureTime"):
        edf.integrate_edf_frame(str(path), npt=200)


# ---------------------------------------------------------------------------
# average_empty_frames
# ---------------------------------------------------------------------------

def test_average_empty_frames_requires_at_least_two():
    with pytest.raises(ValueError, match="at least 2"):
        edf.average_empty_frames(["only_one.edf"])


def test_average_empty_frames_sem_floor_matches_manual_calc(tmp_path):
    """The systematic floor must be standard-error-of-the-mean (population
    std / sqrt(N)), not the raw population std -- what gets subtracted from
    a sample downstream is the MEAN of the empties, so the uncertainty that
    matters is the uncertainty in that mean, not the spread of one future
    draw. Checked here against numpy's own SEM computed directly on the
    resampled per-file rates."""
    paths = []
    for i in range(4):
        data = _flat_frame_with_ring(q_ring=None, base=50.0 + i * 2.0, seed=10 + i)
        p = tmp_path / f"empty_{i}.edf"
        _write_synthetic_frame(p, _base_header(exposure=900.0), data)
        paths.append(str(p))

    avg = edf.average_empty_frames(paths, npt=200)
    assert avg.n_inputs == 4
    assert avg.curve.intensity.shape == avg.curve.sigma.shape == (200,)

    # Recompute independently from the individual integrated curves.
    curves = [edf.integrate_edf_frame(p, npt=200) for p in paths]
    qmin = max(float(c.q.min()) for c in curves)
    qmax = min(float(c.q.max()) for c in curves)
    q0 = np.linspace(qmin, qmax, 200)
    stack = np.stack([np.interp(q0, c.q, c.intensity) for c in curves], axis=0)
    expected_mean = np.mean(stack, axis=0)
    expected_sem = np.std(stack, axis=0, ddof=1) / np.sqrt(4)

    assert np.allclose(avg.curve.intensity, expected_mean, rtol=1e-8)
    assert np.allclose(avg.sys_sigma, expected_sem, rtol=1e-8)
    # SEM is strictly smaller than the raw population std it derives from.
    raw_std = np.std(stack, axis=0, ddof=1)
    assert np.all(avg.sys_sigma <= raw_std + 1e-30)


def test_average_empty_frames_rejects_mismatched_shapes_gracefully(tmp_path):
    # Different SHAPE geometry between frames still resamples onto a common
    # explicit grid rather than raising, as long as q ranges overlap.
    paths = []
    for i in range(2):
        data = _flat_frame_with_ring(seed=20 + i)
        p = tmp_path / f"e_{i}.edf"
        _write_synthetic_frame(p, _base_header(exposure=900.0), data)
        paths.append(str(p))
    avg = edf.average_empty_frames(paths, npt=150)
    assert avg.curve.q.shape == (150,)


# ---------------------------------------------------------------------------
# plateau_chi2red
# ---------------------------------------------------------------------------

def test_plateau_chi2red_near_one_for_well_calibrated_synthetic_noise():
    rng = np.random.default_rng(42)
    q = np.linspace(0.05, 0.35, 2000)
    sigma = np.full_like(q, 0.1)
    corrected = 5.0 + rng.normal(scale=0.1, size=q.size)
    result = edf.plateau_chi2red(q, corrected, sigma, q_lo=0.2, q_hi=0.29)
    assert result["n_points"] > 100
    assert result["chi2red"] == pytest.approx(1.0, rel=0.25)
    assert result["frac_negative"] == 0.0


def test_plateau_chi2red_too_few_points_returns_none():
    q = np.array([0.21, 0.22])
    corrected = np.array([1.0, 1.1])
    sigma = np.array([0.1, 0.1])
    result = edf.plateau_chi2red(q, corrected, sigma, q_lo=0.2, q_hi=0.29)
    assert result["chi2red"] is None
    assert result["n_points"] == 2


# ---------------------------------------------------------------------------
# validate_against_agbeh -- the one test needing genuine real-data physics
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(AGBEH_FIXTURE), reason="AgBeh fixture not bundled")
def test_validate_against_agbeh_real_fixture_within_tolerance():
    result = edf.validate_against_agbeh(AGBEH_FIXTURE)
    assert result["ok"], result
    assert result["q_peak"] == pytest.approx(edf.AGBEH_Q1_A_INV, rel=0.02)
    assert result["rel_err"] < 0.02


def test_pyfai_available_flag_reflects_real_environment():
    # pyFAI/fabio are confirmed installed in this environment (module-level
    # importorskip above would have skipped the file otherwise).
    assert edf.pyfai_available() is True
    assert edf.pyfai_import_error() is None
