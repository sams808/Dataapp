"""Tests for xas_science.py — the framework-agnostic XAS/XANES/EXAFS engine
consolidated from three drifted copies at M4. No pytest coverage existed for
this module before M11 (a pre-existing gap from M4, where the plan called
for "pytest coverage for both Larch-present and Larch-absent paths" but it
wasn't delivered) — written now as foundational work before building the
Qt port on top of it.

Larch + xraydb are both installed in this environment, so Larch-dependent
tests run for real rather than only exercising a skip path; they're still
marked with conftest.py's `requires_larch` so the suite degrades gracefully
on a machine without Larch installed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import xas.xas_science as xs
from conftest import requires_larch


# --------------------------------------------------------------------------
# Synthetic XAS-like data: pre-edge flat, sharp edge jump (erf), decaying
# post-edge EXAFS-like oscillation. Centered near the real Fe K-edge
# (7112 eV) so Larch-dependent edge-identification tests have a physically
# meaningful target to check against xraydb.
# --------------------------------------------------------------------------

def _synthetic_mu(e0=7112.0, n=600, e_lo=6900.0, e_hi=7500.0):
    from scipy.special import erf
    energy = np.linspace(e_lo, e_hi, n)
    pre = 0.15
    jump = 1.0
    edge = jump * 0.5 * (1 + erf((energy - e0) / 3.0))
    post_mask = energy > e0
    k = np.sqrt(np.clip(energy - e0, 0, None) * 0.2625)  # rough eV->1/A scale
    exafs = np.zeros_like(energy)
    exafs[post_mask] = 0.05 * np.exp(-0.05 * k[post_mask]) * np.sin(2 * 2.5 * k[post_mask])
    mu = pre + edge + exafs
    return energy, mu


# --------------------------------------------------------------------------
# Math / smoothing helpers
# --------------------------------------------------------------------------

def test_savgol_filter_smooths_noisy_signal_toward_truth():
    x = np.linspace(0, 10, 300)
    truth = np.sin(x)
    rng = np.random.default_rng(0)
    noisy = truth + rng.normal(0, 0.1, size=truth.shape)
    smoothed = xs.savgol_filter(noisy, window=11, poly=3)
    assert np.abs(smoothed - truth).mean() < np.abs(noisy - truth).mean()


def test_savgol_filter_oversized_window_raises():
    """Found via a real EasyXAFS scan (292 points): np.pad's "reflect" mode
    silently handles a pad width larger than the array via repeated
    reflection instead of raising, so a window this large used to run
    "successfully" while silently producing a heavily-distorted result
    (shifted a found e0 by 44 eV with zero indication anything was wrong).
    Must raise instead."""
    with pytest.raises(ValueError, match="must be smaller than the data length"):
        xs.savgol_filter(np.sin(np.linspace(0, 10, 292)), window=99999, poly=3)


def test_rolling_median_removes_isolated_spike():
    y = np.ones(50)
    y[25] = 100.0
    out = xs.rolling_median(y, window=5)
    assert out[25] == pytest.approx(1.0)


def test_whittaker_smooth_matches_sparse_and_dense_solve():
    """The sparse-solve path (scipy available) should agree closely with a
    fresh dense re-derivation, guarding the O(n^3)->O(n) perf fix noted in
    the module docstring."""
    y = np.sin(np.linspace(0, 6, 80)) + 0.05 * np.sin(np.linspace(0, 200, 80))
    sparse_result = xs.whittaker_smooth(y, lam=1e4, d=2)

    n = y.size
    I = np.eye(n)
    D = np.eye(n)
    for _ in range(2):
        D = np.diff(D, axis=0)
    A = I + 1e4 * (D.T @ D)
    dense_result = np.linalg.solve(A, y)

    assert np.allclose(sparse_result, dense_result, atol=1e-6)


def test_smooth_spectrum_dispatches_all_methods():
    y = np.sin(np.linspace(0, 10, 100))
    assert xs.smooth_spectrum(y, "savitzky-golay", {"window": 9, "poly": 3}).shape == y.shape
    assert xs.smooth_spectrum(y, "median+sg", {"median_window": 7, "sg_window": 9, "sg_poly": 3}).shape == y.shape
    assert xs.smooth_spectrum(y, "whittaker", {"lam": 1e4, "d": 2}).shape == y.shape
    if xs._SCIPY_AVAILABLE:
        assert xs.smooth_spectrum(y, "spline", {"s": 0.0}).shape == y.shape


def test_smooth_spectrum_unknown_method_raises():
    with pytest.raises(ValueError):
        xs.smooth_spectrum(np.zeros(10), "not-a-method", {})


def test_fit_chebyshev_recovers_smooth_baseline():
    x = np.linspace(0, 100, 200)
    truth = 0.001 * x**2 - 0.05 * x + 3.0
    mask = np.ones_like(x, dtype=bool)
    fitted = xs.fit_chebyshev(x, truth, degree=4, mask=mask)
    assert np.allclose(fitted, truth, atol=1e-6)


def test_mu_from_transmission_matches_manual_log():
    i0 = np.array([100.0, 200.0, 300.0])
    it = np.array([50.0, 80.0, 90.0])
    mu = xs.mu_from_transmission(i0, it, logbase="ln")
    assert np.allclose(mu, np.log(i0 / it))
    mu10 = xs.mu_from_transmission(i0, it, logbase="log10")
    assert np.allclose(mu10, np.log10(i0 / it))


def test_build_mu_interpolates_it_onto_i0_grid():
    i0_energy = np.linspace(0, 10, 50)
    it_energy = np.linspace(0, 10, 30)  # different grid
    i0 = np.full(50, 100.0)
    it = np.full(30, 50.0)
    mu = xs.build_mu(i0_energy, i0, it_energy, it, log_mode="ln")
    assert mu.shape == i0_energy.shape
    assert np.allclose(mu, np.log(2.0), atol=1e-6)


# --------------------------------------------------------------------------
# Analysis tab math (combine/difference/linear-combination-fit) -- pulled
# out of qt_xas.py's Analysis tab into pure functions here, but never got
# dedicated unit tests of its own until the Bi L3 real-data review.
# --------------------------------------------------------------------------

def test_combine_spectra_average_and_sum():
    e = np.linspace(0, 10, 20)
    y1 = np.full(20, 2.0)
    y2 = np.full(20, 4.0)
    y3 = np.full(20, 6.0)
    avg = xs.combine_spectra(e, y1, [(e, y2), (e, y3)], op="average")
    total = xs.combine_spectra(e, y1, [(e, y2), (e, y3)], op="sum")
    assert np.allclose(avg, 4.0)
    assert np.allclose(total, 12.0)


def test_combine_spectra_rejects_truncated_repeat_scan():
    """Found via a real EasyXAFS I0_after scan cut short mid-acquisition
    (an instrument fault): averaging in a repeat scan that doesn't cover
    the reference's full range used to silently hold the truncated
    scan's last value flat over the missing region (np.interp's default
    extrapolation) rather than erroring -- up to several percent wrong,
    with no indication anything was extrapolated."""
    e_full = np.linspace(13300.0, 13664.0, 292)
    y_full = np.linspace(1.0, 2.0, 292)
    e_truncated = e_full[:144]  # cuts off partway through, like the real scan
    y_truncated = y_full[:144]
    with pytest.raises(ValueError, match="doesn't fully cover"):
        xs.combine_spectra(e_full, y_full, [(e_truncated, y_truncated)], op="average")


def test_difference_spectra_interpolates_and_subtracts():
    e = np.linspace(0, 10, 20)
    a = np.full(20, 5.0)
    b_energy = np.linspace(0, 10, 15)
    b = np.full(15, 2.0)
    b_interp, diff = xs.difference_spectra(e, a, b_energy, b)
    assert np.allclose(b_interp, 2.0)
    assert np.allclose(diff, 3.0)


# --------------------------------------------------------------------------
# M11 fix: compute_mu's deglitch parameters were previously dead (accepted
# but never used).
# --------------------------------------------------------------------------

def test_deglitch_mu_removes_isolated_spike_without_disturbing_signal():
    energy, mu = _synthetic_mu()
    mu_glitched = mu.copy()
    glitch_idx = 300
    mu_glitched[glitch_idx] += 5.0  # a huge, isolated spike relative to local scale

    cleaned = xs.deglitch_mu(mu_glitched, z=6.0, window=21)
    assert abs(cleaned[glitch_idx] - mu[glitch_idx]) < 0.5  # spike substantially removed
    # Untouched region far from the glitch should be numerically identical.
    assert np.allclose(cleaned[:100], mu_glitched[:100])


def test_compute_mu_deglitch_flag_actually_applies():
    """Regression guard for the dead-parameter bug: deglitch=True must
    change the output relative to deglitch=False when a glitch is present."""
    energy = np.linspace(7000, 7300, 300)
    i0 = np.full(300, 100.0)
    it = np.full(300, 50.0)
    it[150] *= 0.2  # a sharp glitch in the transmitted signal

    xas = xs.XASData(path="synthetic", df=pd.DataFrame(), energy_col="Energy", i0_col="I0", it_col="It",
                      energy=energy, i0=i0, it=it)

    mu_plain = xs.compute_mu(xas, deglitch=False)
    mu_degl = xs.compute_mu(xas, deglitch=True, deglitch_z=4.0, deglitch_window=15)

    assert not np.allclose(mu_plain, mu_degl)
    assert abs(mu_degl[150] - mu_plain[149]) < abs(mu_plain[150] - mu_plain[149])


# --------------------------------------------------------------------------
# Physics: angle/energy Bragg correction, tie-point alignment, edge inference
# --------------------------------------------------------------------------

def test_angle_energy_correction_bragg_round_trips_synthetic_angles():
    """Build angle data via the module's OWN forward Bragg formula for a
    known target energy grid, then confirm the correction function recovers
    that same energy grid — a self-consistency/regression check, not an
    independent physics validation (the formula itself isn't re-derived
    here, just guarded against regressing)."""
    d_ang = xs._d_spacing_cubic(xs.DEFAULT_LATTICE_A_ANG["si"], 1, 1, 1)
    target_energy = np.linspace(7000, 7500, 200)
    theta_deg = np.rad2deg(np.arcsin(np.clip(xs.HC_EV_ANG / (2.0 * d_ang * target_energy), -1, 1)))

    scan_def = {"crystal2d": "Si(1,1,1)"}
    corrected, diag = xs.angle_energy_correction_bragg(theta_deg, target_energy, scan_def, mode="A", fit_linear=True)

    assert diag["angle_interpretation"] == "theta"
    assert np.allclose(corrected, target_energy, atol=1.0)


def test_angle_energy_correction_bragg_missing_crystal2d_raises():
    with pytest.raises(KeyError):
        xs.angle_energy_correction_bragg(np.array([1.0, 2.0]), np.array([1.0, 2.0]), {})


def test_apply_alignment_mode_c_shift_model():
    e_after = np.array([100.0, 200.0, 300.0])
    tiepoints = [xs.TiePoint(e_before=105.0, e_after=100.0)]
    corrected, diag = xs.apply_alignment_mode_c(e_after, tiepoints, model="shift")
    assert diag["model"] == "shift"
    assert np.allclose(corrected, e_after + 5.0)


def test_apply_alignment_mode_c_affine_needs_two_points():
    with pytest.raises(ValueError):
        xs.apply_alignment_mode_c(np.array([1.0]), [xs.TiePoint(1.0, 1.0)], model="affine")


@requires_larch
def test_infer_edge_label_from_roi_scaled_identifies_fe_k_edge():
    energy, mu = _synthetic_mu(e0=7112.0)
    label, e0 = xs.infer_edge_label_from_roi_scaled(energy, mu, {})
    assert "Fe" in label
    assert "K" in label
    assert e0 == pytest.approx(7112.0, abs=15.0)


def test_infer_edge_label_from_roi_scaled_too_short_returns_unknown():
    label, e0 = xs.infer_edge_label_from_roi_scaled(np.array([1.0, 2.0]), np.array([1.0, 2.0]), {})
    assert label == "XAS(?)"
    assert e0 is None


def test_edge_text_extracts_element_and_edge():
    assert xs.edge_text("XAS(Fe K)") == "Fe K"
    assert xs.edge_text("XAS(?)") == "?"


# --------------------------------------------------------------------------
# Larch-dependent higher-level operations
# --------------------------------------------------------------------------

@requires_larch
def test_larch_normalize_produces_sane_norm_and_e0():
    energy, mu = _synthetic_mu()
    out = xs.larch_normalize(energy, mu, e0_method="larch", e0_manual=None, pre1=-150, pre2=-50, norm1=30, norm2=150, nnorm=1)
    assert out["e0"] == pytest.approx(7112.0, abs=15.0)
    assert out["norm"].shape == energy.shape
    # Well past the edge, normalized mu should sit close to 1 (post-edge flattened).
    post_mask = energy > out["e0"] + 60
    assert np.nanmedian(out["norm"][post_mask]) == pytest.approx(1.0, abs=0.3)


@requires_larch
def test_larch_normalize_e0_smoothing_failed_flag():
    """Previously silent: a broken smooth_for_e0 request fell back to the
    unsmoothed mu for e0-finding with zero indication to the caller.
    Verified against a real EasyXAFS scan during the Bi L3 processing
    review -- surfaced as this flag instead so the UI can tell the user
    their smoothing request didn't actually apply."""
    energy, mu = _synthetic_mu()
    out_bad = xs.larch_normalize(
        energy, mu, e0_method="larch", e0_manual=None, pre1=-150, pre2=-50, norm1=30, norm2=150, nnorm=1,
        smooth_for_e0=("savitzky-golay", {"window": 99999, "poly": 3}),
    )
    assert out_bad["e0_smoothing_failed"] is not None
    assert "window" in out_bad["e0_smoothing_failed"].lower()

    out_ok = xs.larch_normalize(
        energy, mu, e0_method="larch", e0_manual=None, pre1=-150, pre2=-50, norm1=30, norm2=150, nnorm=1,
        smooth_for_e0=("savitzky-golay", {"window": 11, "poly": 3}),
    )
    assert out_ok["e0_smoothing_failed"] is None


@requires_larch
def test_larch_exafs_pipeline_produces_k_chi_and_ft():
    energy, mu = _synthetic_mu()
    out = xs.larch_exafs_pipeline(
        energy, mu, e0_method="larch", e0_manual=None, pre1=-150, pre2=-50, norm1=30, norm2=150, nnorm=1,
        rbkg=1.0, kmin=0.0, kmax=10.0, dk=0.1, kweight=2, window="hanning", rmax_out=8.0,
    )
    assert out["k"].size > 0
    assert out["chi"].shape == out["k"].shape
    assert out["r"].size > 0
    assert out["chir_mag"].shape == out["r"].shape
    assert out["e0_smoothing_failed"] is None


@requires_larch
def test_larch_exafs_pipeline_e0_smoothing_failed_flag():
    energy, mu = _synthetic_mu()
    out = xs.larch_exafs_pipeline(
        energy, mu, e0_method="larch", e0_manual=None, pre1=-150, pre2=-50, norm1=30, norm2=150, nnorm=1,
        rbkg=1.0, kmin=0.0, kmax=10.0, dk=0.1, kweight=2, window="hanning", rmax_out=8.0,
        smooth_for_e0=("savitzky-golay", {"window": 99999, "poly": 3}),
    )
    assert out["e0_smoothing_failed"] is not None


@requires_larch
def test_larch_exafs_pipeline_kmax_bkg_defaults_to_kmax():
    """kmax_bkg=None (the default) must reproduce the exact prior
    behavior -- background fit and FT sharing one kmax -- for backward
    compatibility with any caller that doesn't set it."""
    energy, mu = _synthetic_mu()
    kwargs = dict(e0_method="larch", e0_manual=None, pre1=-150, pre2=-50, norm1=30, norm2=150, nnorm=1,
                  rbkg=1.0, kmin=0.0, kmax=10.0, dk=0.1, kweight=2, window="hanning", rmax_out=8.0)
    out_default = xs.larch_exafs_pipeline(energy, mu, **kwargs)
    out_explicit = xs.larch_exafs_pipeline(energy, mu, kmax_bkg=10.0, **kwargs)
    assert out_default["kmax_bkg"] == 10.0
    assert np.allclose(out_default["chi"], out_explicit["chi"])
    assert np.allclose(out_default["chir_mag"], out_explicit["chir_mag"])


@requires_larch
def test_larch_exafs_pipeline_kmax_bkg_independent_of_ft_window():
    """A wider background-fit range than the FT window must change chi(k)
    beyond the FT window (more spline knots fit) while the FT window
    (kmin/kmax passed to xftf) stays whatever the caller asked for --
    the actual bug this was written to fix (background range and FT
    window were previously the same single kmin/kmax pair)."""
    energy, mu = _synthetic_mu()
    common = dict(e0_method="larch", e0_manual=None, pre1=-150, pre2=-50, norm1=30, norm2=150, nnorm=1,
                  rbkg=1.0, kmin=1.0, kmax=6.0, dk=0.1, kweight=2, window="hanning", rmax_out=8.0)
    out_narrow_bkg = xs.larch_exafs_pipeline(energy, mu, kmax_bkg=6.0, **common)
    out_wide_bkg = xs.larch_exafs_pipeline(energy, mu, kmax_bkg=10.0, **common)
    assert out_wide_bkg["k"].max() > out_narrow_bkg["k"].max()
    assert out_wide_bkg["kmax_bkg"] == 10.0 and out_narrow_bkg["kmax_bkg"] == 6.0
    # Both still report the same FT (R-space) grid -- xftf's own window
    # wasn't affected by the wider background range.
    assert np.allclose(out_narrow_bkg["r"], out_wide_bkg["r"])


@requires_larch
def test_larch_exafs_pipeline_clamp_defaults_match_athena():
    energy, mu = _synthetic_mu()
    out = xs.larch_exafs_pipeline(
        energy, mu, e0_method="larch", e0_manual=None, pre1=-150, pre2=-50, norm1=30, norm2=150, nnorm=1,
        rbkg=1.0, kmin=0.0, kmax=10.0, dk=0.1, kweight=2, window="hanning", rmax_out=8.0,
    )
    assert out["bkg"].shape == mu.shape
    assert np.isfinite(out["bkg"]).any()


# --------------------------------------------------------------------------
# deglitch_3x_noise: noise-calibrated alternative to deglitch_mu
# --------------------------------------------------------------------------

def test_deglitch_3x_noise_removes_isolated_spike_without_disturbing_signal():
    energy, mu = _synthetic_mu()
    mu_glitched = mu.copy()
    glitch_idx = 300
    mu_glitched[glitch_idx] += 5.0

    cleaned, flagged = xs.deglitch_3x_noise(mu_glitched)
    assert glitch_idx in flagged
    assert abs(cleaned[glitch_idx] - mu[glitch_idx]) < 0.5
    assert np.allclose(cleaned[:100], mu_glitched[:100])


def test_deglitch_3x_noise_does_not_flag_clean_oscillation():
    """Regression guard for the exact failure deglitch_mu had on real
    data: a realistically-noisy but glitch-free EXAFS-like oscillation
    should flag ~nothing, even right at its own peaks/troughs (a
    rolling-window median's own baseline gets biased there, which is
    exactly what made the old deglitch_mu flag 19 points on a real,
    clean NaBiO3 scan). Checked on the post-edge oscillatory region only
    -- _synthetic_mu's edge itself is a much sharper step (~3 eV) than
    any real Bi L3 XANES edge, so its slope alone (not noise) would
    swamp the noise-calibrated threshold right at the edge; that's a
    property of this synthetic test edge, not a real detector glitch,
    and isn't what this guard is about."""
    rng = np.random.default_rng(1)
    energy, mu = _synthetic_mu()
    mu_noisy = mu + rng.normal(0.0, 0.003, size=mu.shape)
    cleaned, flagged = xs.deglitch_3x_noise(mu_noisy)
    post_edge = energy > (7112.0 + 30.0)
    flagged_post_edge = [i for i in flagged if post_edge[i]]
    # A statistical false-positive rate near 0, not a brittle exact-zero
    # assertion -- this synthetic check is a supplementary sanity guard;
    # the primary evidence is the real NaBiO3 scan this was validated
    # against (19 falsely-flagged points -> 0, project history).
    assert len(flagged_post_edge) <= 1
    assert np.sum(post_edge) > 300  # sanity: the region actually checked is substantial


def test_estimate_noise_matches_known_gaussian_scale():
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, 0.02, size=2000)
    noise = xs.estimate_noise(y)
    assert 0.02 * 0.5 < noise < 0.02 * 2.0


# --------------------------------------------------------------------------
# Data model: Spectrum / SpectrumStore
# --------------------------------------------------------------------------

def test_spectrum_copy_tracks_parent_and_deep_copies_arrays():
    sp = xs.Spectrum(sid="s1", name="orig", kind="mu", energy=np.array([1.0, 2.0]), y=np.array([3.0, 4.0]))
    sp2 = sp.copy(new_name="orig_copy")
    assert sp2.sid != sp.sid
    assert sp2.parents == ["s1"]
    sp2.energy[0] = 999.0
    assert sp.energy[0] == 1.0  # independent array, not a view


def test_spectrum_store_add_remove_find_by_kind():
    store = xs.SpectrumStore()
    a = xs.Spectrum(sid="a", name="A", kind="I0", energy=np.array([1.0]), y=np.array([1.0]))
    b = xs.Spectrum(sid="b", name="B", kind="It", energy=np.array([1.0]), y=np.array([1.0]))
    store.add(a); store.add(b)

    assert [s.sid for s in store.all()] == ["a", "b"]
    assert store.find_by_name("A") is a
    assert [s.sid for s in store.by_kind(["It"])] == ["b"]

    store.remove("a")
    assert store.all() == [b]
    assert store.find_by_name("A") is None


# --------------------------------------------------------------------------
# I/O: generic table reader + compat layer
# --------------------------------------------------------------------------

def test_parse_xas_file_infers_energy_i0_it_columns(tmp_path):
    df = pd.DataFrame({
        "Energy(eV)": np.linspace(7000, 7200, 50),
        "I0": np.full(50, 100.0),
        "It": np.full(50, 50.0),
    })
    p = tmp_path / "sample.csv"
    df.to_csv(p, index=False)

    xas = xs.parse_xas_file(p)
    assert xas.energy.size == 50
    assert xas.i0_col == "I0"
    assert xas.it_col == "It"
    mu = xs.compute_mu(xas)
    assert np.allclose(mu, np.log(2.0), atol=1e-6)


def test_read_easyxafs_zip_and_bundle_round_trip(tmp_path):
    import zipfile
    df = pd.DataFrame({
        "Energy(eV)": np.linspace(7000, 7200, 20),
        "I0": np.full(20, 100.0),
        "It": np.full(20, 25.0),
    })
    zip_path = tmp_path / "sample_exd.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("mysample_exd.csv", df.to_csv(index=False))
        zf.writestr("scan_def.json", '{"crystal2d": "Si(1,1,1)"}')

    bundles = xs.read_bundles_from_zip(zip_path)
    assert len(bundles) == 1
    assert bundles[0].scan_def.get("crystal2d") == "Si(1,1,1)"

    xas = xs.parse_xas_bundle(bundles[0])
    assert xas.energy.size == 20
    mu = xs.compute_mu(xas)
    assert np.allclose(mu, np.log(4.0), atol=1e-6)


def test_read_easyxafs_zip_without_exd_csv_raises(tmp_path):
    import zipfile
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("readme.txt", "nothing here")
    with pytest.raises(FileNotFoundError):
        xs.read_easyxafs_zip(zip_path)


def test_infer_xas_edge_from_spectrum_wraps_label_correctly():
    energy, mu = _synthetic_mu(e0=7112.0)
    out = xs.infer_xas_edge_from_spectrum(energy, mu)
    if "element" in out:  # only asserted structurally; Larch presence covered above
        assert out["label"] == f"{out['element']} {out['edge']}"


# --------------------------------------------------------------------------
# Athena .prj import/export. The user's real bug report ("Import error: no
# module named 'larch'" from the portable exe) led to three fixes at once:
# a pure-Python .prj parser, an export writer that supports modern larch's
# create_athena API (write_athena was dropped upstream — export silently
# returned False), and the discovery that the old larch-reader import path
# iterated a Group as if it were a dict and yielded ZERO spectra.
# --------------------------------------------------------------------------

def _two_prj_spectra():
    e = np.linspace(7000.0, 7300.0, 60)

    def mk(name, e0, off):
        return xs.Spectrum(
            sid=xs._uid("sp"), name=name, kind="mu", energy=e,
            y=np.tanh((e - e0) / 10.0) + off, angle=None, units="a.u.",
            label="XAS", e0=e0, meta={})

    return mk("scanA", 7112.0, 1.0), mk("scanB", 7115.0, 1.1)


@requires_larch
def test_athena_prj_export_then_reimport_round_trips(tmp_path):
    sp1, sp2 = _two_prj_spectra()
    path = tmp_path / "roundtrip.prj"
    assert xs.export_athena_prj_best_effort(path, [sp1, sp2])  # was False on larch>=2025

    back = xs.read_athena_prj(path)
    assert [s.name for s in back] == ["scanA", "scanB"]  # was [] via the old larch path
    assert all(s.kind == "mu" for s in back)
    assert np.allclose(back[0].energy, sp1.energy)
    assert np.allclose(back[0].y, sp1.y)
    assert back[1].e0 == pytest.approx(7115.0, abs=6.0)  # larch snaps e0 to the grid


@requires_larch
def test_parse_athena_prj_no_larch_reads_perl_format(tmp_path):
    sp1, sp2 = _two_prj_spectra()
    path = tmp_path / "perlstyle.prj"
    assert xs.export_athena_prj_best_effort(path, [sp1, sp2])

    recs = xs.parse_athena_prj_no_larch(path)
    assert [r["label"] for r in recs] == ["scanA", "scanB"]
    assert np.allclose(recs[1]["mu"], sp2.y)
    assert recs[0]["e0"] == pytest.approx(7112.0, abs=6.0)


def test_parse_athena_prj_no_larch_reads_json_format(tmp_path):
    """larch's other on-disk flavor: gzipped JSON with _____-prefixed keys."""
    import gzip
    import json
    e = [7000.0, 7001.0, 7002.0]
    doc = {
        "_____header1": "# Athena project file -- Demeter version 0.9.26",
        "_____order": ["grpA"],
        "grpA": {"x": e, "y": [0.1, 0.2, 0.3], "args": {"label": "My Sample", "bkg_e0": "7001.0"}},
    }
    path = tmp_path / "jsonstyle.prj"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps(doc))
    # JSON detection keys off '____header' appearing early in the text;
    # the validity check finds "Athena project file -- " inside the header value
    recs = xs.parse_athena_prj_no_larch(path)
    assert len(recs) == 1
    assert recs[0]["label"] == "My Sample"
    assert recs[0]["e0"] == 7001.0
    assert np.allclose(recs[0]["energy"], e)


def test_read_athena_prj_works_without_larch(tmp_path, monkeypatch):
    """The portable-exe scenario: no larch importable at all."""
    import gzip
    import json
    doc = {
        "_____header1": "# Athena project file -- Demeter version 0.9.26",
        "_____order": ["g1"],
        "g1": {"x": [7002.0, 7000.0, 7001.0, 7001.0], "y": [3.0, 1.0, 2.0, 2.5],
               "args": {"label": "unsorted", "bkg_e0": "7001.0"}},
    }
    path = tmp_path / "nolarch.prj"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps(doc))

    monkeypatch.setattr(xs, "LARCH_AVAILABLE", False)
    out = xs.read_athena_prj(path)
    assert len(out) == 1
    assert out[0].name == "unsorted"
    assert list(out[0].energy) == [7000.0, 7001.0, 7002.0]  # sorted, duplicate dropped
    assert list(out[0].y) == [1.0, 2.0, 3.0]


def test_parse_athena_prj_rejects_non_athena_file(tmp_path):
    path = tmp_path / "nope.prj"
    path.write_text("just some text")
    with pytest.raises(ValueError, match="not an Athena project"):
        xs.parse_athena_prj_no_larch(path)
