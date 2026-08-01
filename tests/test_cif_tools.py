"""Tests for cif_tools.py — CIF parsing + Bragg peak generation."""
from __future__ import annotations

import math

import pytest

import raman.cif_tools as ct


TRICLINIC_CIF = """\
data_test_triclinic
_cell_length_a    5.000
_cell_length_b    6.000
_cell_length_c    7.000
_cell_angle_alpha 80.0
_cell_angle_beta  95.0
_cell_angle_gamma 100.0
_diffrn_radiation_wavelength 1.5406
"""

CUBIC_CIF = """\
data_test_cubic
_cell_length_a    5.000
_cell_length_b    5.000
_cell_length_c    5.000
_cell_angle_alpha 90.0
_cell_angle_beta  90.0
_cell_angle_gamma 90.0
_diffrn_radiation_wavelength 1.5406
"""


def _write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return str(p)


def test_parse_cif_generic_reads_cell_and_wavelength(tmp_path):
    path = _write(tmp_path, "tri.cif", TRICLINIC_CIF)
    a, b, c, alpha, beta, gamma, wavelength = ct.parse_cif_generic(path)
    assert (a, b, c) == (5.0, 6.0, 7.0)
    assert (alpha, beta, gamma) == (80.0, 95.0, 100.0)
    assert wavelength == pytest.approx(1.5406)


def test_parse_cif_generic_defaults_wavelength_when_absent(tmp_path):
    content = CUBIC_CIF.replace("_diffrn_radiation_wavelength 1.5406\n", "")
    path = _write(tmp_path, "no_wl.cif", content)
    *_, wavelength = ct.parse_cif_generic(path)
    assert wavelength == pytest.approx(1.5406)  # Cu K-alpha default


def test_d_triclinic_matches_cubic_formula_for_orthogonal_cell():
    # For alpha=beta=gamma=90, d_hkl = a / sqrt(h^2+k^2+l^2) (cubic formula).
    a = 5.0
    d = ct._d_triclinic(a, a, a, 90.0, 90.0, 90.0, 1, 1, 1)
    expected = a / math.sqrt(3)
    assert d == pytest.approx(expected, rel=1e-6)


def test_d_triclinic_friedel_pair_has_equal_d():
    # (h,k,l) and (-h,-k,-l) must always give the same |d|, any cell shape.
    d1 = ct._d_triclinic(5.0, 6.0, 7.0, 80.0, 95.0, 100.0, 2, 1, 3)
    d2 = ct._d_triclinic(5.0, 6.0, 7.0, 80.0, 95.0, 100.0, -2, -1, -3)
    assert d1 == pytest.approx(d2, rel=1e-9)


def test_d_triclinic_mixed_sign_differs_from_same_sign_for_nonorthogonal_cell():
    # This is exactly the coverage gap the bug fix addresses: for a
    # non-orthogonal cell, (h,k,l) and (h,k,-l) are NOT equivalent.
    d_same = ct._d_triclinic(5.0, 6.0, 7.0, 80.0, 95.0, 100.0, 1, 1, 2)
    d_mixed = ct._d_triclinic(5.0, 6.0, 7.0, 80.0, 95.0, 100.0, 1, 1, -2)
    assert d_same != pytest.approx(d_mixed, rel=1e-6)


def test_bragg_peaks_no_duplicate_two_theta_values(tmp_path):
    path = _write(tmp_path, "tri2.cif", TRICLINIC_CIF)
    peaks = ct.bragg_peaks_from_cif_generic(path, two_theta_max=80.0, hkl_max=4, use_cache=False)
    two_thetas = [round(tt, 6) for tt, _hkl, _d in peaks]
    assert len(two_thetas) == len(set(two_thetas)), "Friedel-equivalent reflections must be deduped"


def test_bragg_peaks_covers_mixed_sign_reflections_for_triclinic_cell(tmp_path):
    # Reproduce the bug directly: the OLD non-negative-only octant would never
    # find a peak matching a mixed-sign (h,k,-l)-type reflection that has a
    # genuinely different d-spacing than its all-positive counterpart.
    path = _write(tmp_path, "tri3.cif", TRICLINIC_CIF)
    peaks = ct.bragg_peaks_from_cif_generic(path, two_theta_max=90.0, hkl_max=3, use_cache=False)
    d_mixed = ct._d_triclinic(5.0, 6.0, 7.0, 80.0, 95.0, 100.0, 1, 1, -2)
    matched = [d for _tt, _hkl, d in peaks if math.isclose(d, d_mixed, rel_tol=1e-6)]
    assert matched, "mixed-sign reflection's d-spacing must appear in the peak list"


def test_bragg_peaks_cubic_sanity(tmp_path):
    path = _write(tmp_path, "cubic.cif", CUBIC_CIF)
    peaks = ct.bragg_peaks_from_cif_generic(path, two_theta_max=80.0, hkl_max=3, use_cache=False)
    assert len(peaks) > 0
    for tt, hkl, d in peaks:
        assert 0 < tt <= 80.0
        assert d > 0
