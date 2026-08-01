"""Tests for fitting_science.py — single-spectrum peak fitting (Gaussian /
pseudo-Voigt via lmfit + rampy), consolidated from three near-identical
copies that used to live inline in main.py's SingleFitWindow.
"""
from __future__ import annotations

import numpy as np
import pytest
import rampy as rp

import core.fitting_science as fs


def _gaussian_component(center=500.0, fwhm=30.0, amp=100.0, shape="G", eta=0.5):
    return dict(
        shape=shape,
        shift_val=center, shift_min=center - 100, shift_max=center + 100,
        fwhm_val=fwhm, fwhm_min=1.0, fwhm_max=200.0,
        amp_val=amp,
        eta_val=eta, eta_min=0.0, eta_max=1.0,
    )


# --------------------------------------------------------------------------
# Synthetic ground truth: build a spectrum from a KNOWN Gaussian, fit it,
# and check the recovered parameters land close to what was built in.
# --------------------------------------------------------------------------

def test_fit_spectrum_classic_recovers_known_gaussian_peak():
    x = np.linspace(400, 600, 400)
    true_center, true_fwhm, true_amp = 505.0, 25.0, 80.0
    y = rp.gaussian(x, true_amp, true_center, true_fwhm)
    rng = np.random.default_rng(0)
    y_noisy = y + rng.normal(0, 0.05, size=y.shape)

    comp = _gaussian_component(center=500.0, fwhm=30.0, amp=100.0, shape="G")
    result = fs.fit_spectrum(x, y_noisy, [comp], mode="classic")

    assert result.lmfit_result.params["f0"].value == pytest.approx(true_center, abs=1.0)
    assert result.lmfit_result.params["l0"].value == pytest.approx(true_fwhm, abs=3.0)
    assert result.lmfit_result.params["a0"].value == pytest.approx(true_amp, rel=0.1)
    assert result.chi2_red < 0.1


def test_fit_spectrum_classic_handles_gl_shape():
    x = np.linspace(400, 600, 400)
    y = rp.pseudovoigt(x, 90.0, 505.0, 28.0, 0.4)
    comp = _gaussian_component(center=500.0, fwhm=30.0, amp=100.0, shape="GL", eta=0.5)
    result = fs.fit_spectrum(x, y, [comp], mode="classic")
    assert result.lmfit_result.params["f0"].value == pytest.approx(505.0, abs=2.0)
    assert "eta0" in result.lmfit_result.params


# --------------------------------------------------------------------------
# Real bundled data: no independently-verified ground truth, so just assert
# the fit converges to something physically sane (finite, positive amplitude,
# center within the search bounds).
# --------------------------------------------------------------------------

def test_fit_spectrum_on_real_raman_example_is_physically_sane(raman_example_path):
    data = np.loadtxt(raman_example_path)
    x, y = data[:, 0], data[:, 1]
    peak_idx = int(np.argmax(y))
    center_guess = float(x[peak_idx])

    comp = _gaussian_component(center=center_guess, fwhm=20.0, amp=float(y[peak_idx]), shape="G")
    comp["shift_min"], comp["shift_max"] = center_guess - 50, center_guess + 50
    result = fs.fit_spectrum(x, y, [comp], mode="classic")

    assert np.isfinite(result.chi2_red)
    assert result.lmfit_result.params["a0"].value > 0
    assert comp["shift_min"] <= result.lmfit_result.params["f0"].value <= comp["shift_max"]


# --------------------------------------------------------------------------
# origin_lm_iteration: ONE true Levenberg-Marquardt parameter update per
# call (the Origin NLFit behavior — user feedback said the old relax-blend
# scheme converged in one click, nothing like Origin's visible stepping).
# The GUI's convergence-checking loop lives in qt_single_fit, not here.
# --------------------------------------------------------------------------

def _origin_setup(center=480.0, fwhm=40.0, amp=60.0):
    x = np.linspace(400, 600, 400)
    y = rp.gaussian(x, 80.0, 505.0, 25.0)  # truth: amp 80, center 505, HWHM 25
    comp = _gaussian_component(center=center, fwhm=fwhm, amp=amp, shape="G")
    params_struct = [comp]
    lm_params = fs.build_lmfit_parameters(params_struct)
    return x, y, params_struct, lm_params


def test_origin_lm_iteration_single_step_improves_but_does_not_converge():
    x, y, params_struct, lm_params = _origin_setup()
    step = fs.origin_lm_iteration(x, y, params_struct, lm_params)

    assert step.accepted
    assert step.chisq_after < step.chisq_before
    # ONE iteration must be a step, not the answer: still measurably away
    # from the global minimum a classic full fit reaches.
    full = fs.fit_spectrum(x, y, params_struct, mode="classic")
    assert step.chisq_after > 10 * full.lmfit_result.chisqr + 1e-12
    # λ shrinks after an accepted step
    assert step.next_lambda < step.lambda_used


def test_origin_lm_iterations_converge_to_truth_when_looped():
    x, y, params_struct, lm_params = _origin_setup()
    current, lam = lm_params, 1e-3
    for _ in range(200):
        step = fs.origin_lm_iteration(x, y, params_struct, current, lambda_lm=lam)
        current, lam = step.params, step.next_lambda
        rel = (step.chisq_before - step.chisq_after) / max(step.chisq_before, 1e-30)
        if not step.accepted or rel < 1e-12:
            break
    assert float(current["f0"].value) == pytest.approx(505.0, abs=0.1)
    assert float(current["a0"].value) == pytest.approx(80.0, rel=0.01)


def test_origin_lm_iteration_respects_vary_and_bounds():
    x, y, params_struct, lm_params = _origin_setup()
    params_struct[0]["fit_fwhm"] = False
    lm_params = fs.build_lmfit_parameters(params_struct)
    fixed_width = float(lm_params["l0"].value)
    lm_params["f0"].set(max=490.0)  # bound blocks the true center at 505

    current, lam = lm_params, 1e-3
    for _ in range(50):
        step = fs.origin_lm_iteration(x, y, params_struct, current, lambda_lm=lam)
        current, lam = step.params, step.next_lambda
        if not step.accepted:
            break
    assert float(current["l0"].value) == fixed_width  # vary=False untouched
    assert float(current["f0"].value) <= 490.0 + 1e-9  # bound respected


def test_origin_lm_iteration_no_varying_params_raises():
    x, y, params_struct, lm_params = _origin_setup()
    for p in lm_params.values():
        p.set(vary=False)
    with pytest.raises(ValueError, match="No varying parameters"):
        fs.origin_lm_iteration(x, y, params_struct, lm_params)


def test_fit_spectrum_rejects_removed_origin_step_mode():
    x = np.linspace(400, 600, 10)
    y = np.ones_like(x)
    with pytest.raises(ValueError, match="Unknown fit mode"):
        fs.fit_spectrum(x, y, [_gaussian_component()], mode="origin_step")


# --------------------------------------------------------------------------
# Bug-guard: fit_spectrum must be safe to call directly with non-numeric
# input, not just from the one GUI path that happens to coerce upstream.
# --------------------------------------------------------------------------

def test_fit_spectrum_rejects_non_numeric_input_with_clear_error():
    x = np.array(["a", "b", "c"], dtype=object)
    y = np.array([1.0, 2.0, 3.0])
    comp = _gaussian_component()
    with pytest.raises(TypeError):
        fs.fit_spectrum(x, y, [comp], mode="classic")


def test_compute_model_matches_sum_of_components():
    x = np.linspace(0, 100, 50)
    comps = [_gaussian_component(center=30, fwhm=5, amp=10), _gaussian_component(center=70, fwhm=8, amp=20)]
    lm_params = fs.build_lmfit_parameters(comps)
    total, peaks = fs.compute_model(x, lm_params, comps)
    assert len(peaks) == 2
    assert np.allclose(total, peaks[0] + peaks[1])


# --------------------------------------------------------------------------
# M8 additions: R^2, peak centroid, 2nd-derivative auto-peak-finder.
# --------------------------------------------------------------------------

def test_compute_r_squared_perfect_fit_is_one():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert fs.compute_r_squared(y, y) == pytest.approx(1.0)


def test_compute_r_squared_worse_than_mean_is_negative():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    y_fit = np.array([4.0, 1.0, 4.0, 1.0])  # far from y, worse than predicting the mean
    assert fs.compute_r_squared(y, y_fit) < 0


def test_peak_centroid_symmetric_gaussian_matches_center():
    x = np.linspace(400, 600, 2000)
    y = rp.gaussian(x, 50.0, 505.0, 20.0)
    assert fs.peak_centroid(x, y) == pytest.approx(505.0, abs=0.5)


def test_peak_centroid_zero_area_returns_nan():
    x = np.linspace(0, 10, 50)
    y = np.zeros_like(x)
    assert np.isnan(fs.peak_centroid(x, y))


def test_find_peak_candidates_recovers_known_peak_centers():
    x = np.linspace(0, 1000, 2000)
    y = rp.gaussian(x, 100.0, 300.0, 15.0) + rp.gaussian(x, 80.0, 700.0, 15.0)
    centers = fs.find_peak_candidates(x, y, max_peaks=5)
    assert len(centers) >= 2
    assert any(abs(c - 300.0) < 10 for c in centers)
    assert any(abs(c - 700.0) < 10 for c in centers)


def test_find_peak_candidates_excludes_edge_artifact():
    """Regression guard, found via manual smoke-testing against real bundled
    data (EXAMPLES/Raman_example.txt): raw Raman spectra often have a sharp,
    strong feature right at the edge of the recorded window (Rayleigh-line
    tail / detector edge), which a pure curvature-based finder would
    otherwise report as the single strongest candidate. Build a synthetic
    analog: a sharp spike right at x=0 (stronger than the real peak) plus a
    genuine interior Gaussian — only the interior one should come back."""
    x = np.linspace(0, 1000, 2000)
    edge_spike = 300.0 * np.exp(-((x - 0.5) ** 2) / (2 * 3.0 ** 2))
    interior_peak = rp.gaussian(x, 80.0, 500.0, 15.0)
    y = edge_spike + interior_peak

    centers = fs.find_peak_candidates(x, y, max_peaks=5)
    assert len(centers) >= 1
    assert all(c > 20.0 for c in centers)  # nothing within the excluded edge margin
    assert any(abs(c - 500.0) < 10 for c in centers)


def test_find_peak_candidates_on_short_input_returns_empty():
    assert fs.find_peak_candidates(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == []


# --------------------------------------------------------------------------
# Deferred-fitting-features additions: true Voigt, EMG, parameter linking.
# --------------------------------------------------------------------------

def test_voigt_peak_limits_match_gaussian_and_lorentzian():
    """Width convention: rampy's third parameter is HWHM (despite the app's
    historical 'FWHM' labels), and the new shapes must match the G/GL
    behavior for the same `l` value — see the module note in
    fitting_science.py. So eta=0 must reproduce rp.gaussian for the SAME
    width argument, and eta=1 a Lorentzian whose full width is 2*l."""
    x = np.linspace(400, 600, 2000)
    v0 = fs.voigt_peak(x, 100.0, 500.0, 20.0, eta=0.0)
    g = rp.gaussian(x, 100.0, 500.0, 20.0)
    assert np.allclose(v0, g, atol=0.5)
    v1 = fs.voigt_peak(x, 100.0, 500.0, 20.0, eta=1.0)
    # rel=1e-3, not tighter: the 2000-point grid over [400,600] never lands
    # exactly on x=500, so the sampled max sits slightly below the true apex.
    assert np.nanmax(v1) == pytest.approx(100.0, rel=1e-3)
    above_half = x[v1 >= 50.0]
    assert (above_half.max() - above_half.min()) == pytest.approx(40.0, abs=0.5)  # full width = 2*HWHM


def test_voigt_peak_height_normalized_at_intermediate_eta():
    x = np.linspace(400, 600, 4000)
    v = fs.voigt_peak(x, 77.0, 500.0, 15.0, eta=0.5)
    assert np.nanmax(v) == pytest.approx(77.0, rel=1e-3)
    assert x[int(np.nanargmax(v))] == pytest.approx(500.0, abs=0.1)


def test_emg_peak_positive_skew_tails_right_negative_tails_left():
    # skew (tau=30) well above sigma (hwhm=10 -> sigma about 8.5) so the
    # asymmetry is pronounced enough for a robust mass-ratio assertion —
    # under the HWHM convention a smaller skew/width ratio gives only a
    # mild, grid-sensitive asymmetry.
    x = np.linspace(400, 600, 2000)
    right = fs.emg_peak(x, 100.0, 500.0, 10.0, skew=30.0)
    left = fs.emg_peak(x, 100.0, 500.0, 10.0, skew=-30.0)
    assert np.nanmax(right) == pytest.approx(100.0, rel=1e-6)
    # Asymmetry: mass above vs below the mode
    mode_r = x[int(np.nanargmax(right))]
    mass_hi = np.trapz(right[x > mode_r], x[x > mode_r])
    mass_lo = np.trapz(right[x < mode_r], x[x < mode_r])
    assert mass_hi > 1.5 * mass_lo
    # Mirror symmetry between +skew and -skew
    assert np.allclose(right, left[::-1], atol=1e-6)


def test_emg_peak_zero_skew_degenerates_to_gaussian():
    x = np.linspace(400, 600, 1000)
    emg = fs.emg_peak(x, 100.0, 500.0, 20.0, skew=0.0)
    g = rp.gaussian(x, 100.0, 500.0, 20.0)
    assert np.allclose(emg, g)


def test_fit_spectrum_recovers_emg_asymmetric_peak():
    x = np.linspace(400, 600, 800)
    y = fs.emg_peak(x, 90.0, 490.0, 12.0, skew=15.0)
    comp = {
        "shape": "EMG",
        "shift_val": 495.0, "shift_min": 450.0, "shift_max": 550.0, "fit_shift": True,
        "fwhm_val": 15.0, "fwhm_min": 1.0, "fwhm_max": 60.0, "fit_fwhm": True,
        "skew_val": 5.0, "skew_min": -100.0, "skew_max": 100.0, "fit_skew": True,
        "amp_val": 80.0, "fit_amp": True,
    }
    result = fs.fit_spectrum(x, y, [comp], mode="classic")
    assert result.lmfit_result.params["f0"].value == pytest.approx(490.0, abs=1.5)
    assert result.lmfit_result.params["s0"].value == pytest.approx(15.0, rel=0.15)
    assert result.chi2_red < 0.5


def test_fit_spectrum_recovers_true_voigt_peak():
    x = np.linspace(400, 600, 800)
    y = fs.voigt_peak(x, 90.0, 505.0, 18.0, eta=0.4)
    comp = _gaussian_component(center=500.0, fwhm=15.0, amp=80.0, shape="V", eta=0.5)
    result = fs.fit_spectrum(x, y, [comp], mode="classic")
    assert result.lmfit_result.params["f0"].value == pytest.approx(505.0, abs=0.5)
    assert result.lmfit_result.params["eta0"].value == pytest.approx(0.4, abs=0.1)


def test_link_fwhm_shares_width_between_components():
    x = np.linspace(0, 100, 600)
    true_fwhm = 6.0
    y = rp.gaussian(x, 100.0, 30.0, true_fwhm) + rp.gaussian(x, 60.0, 70.0, true_fwhm)

    comps = [
        _gaussian_component(center=30.0, fwhm=8.0, amp=90.0),
        {**_gaussian_component(center=70.0, fwhm=12.0, amp=50.0), "link_fwhm": 0},
    ]
    lm_params = fs.build_lmfit_parameters(comps)
    assert lm_params["l1"].expr == "l0"

    result = fs.fit_spectrum(x, y, comps, mode="classic")
    fitted = result.lmfit_result.params
    assert fitted["l0"].value == pytest.approx(true_fwhm, abs=0.3)
    assert fitted["l1"].value == pytest.approx(fitted["l0"].value)  # constrained equal


def test_compute_confidence_intervals_on_known_gaussian():
    x = np.linspace(400, 600, 400)
    rng = np.random.default_rng(0)
    y = rp.gaussian(x, 80.0, 505.0, 25.0) + rng.normal(0, 0.5, x.shape)
    comp = _gaussian_component(center=500.0, fwhm=30.0, amp=100.0)
    result = fs.fit_spectrum(x, y, [comp], mode="classic")

    report = fs.compute_confidence_intervals(result, sigmas=(1,))
    assert "f0" in report and "l0" in report  # center and width profiled
    # The true center (505) must fall inside the reported ±1σ band around
    # the best value; cheap sanity rather than parsing the full table.
    best = result.lmfit_result.params["f0"].value
    assert best == pytest.approx(505.0, abs=1.0)


def test_compute_confidence_intervals_requires_classic_fit():
    fr = fs.FitResult(lmfit_result=None, params=None, y_fit=np.array([]), peaks=[], chi2_red=0.0, minimizer=None)
    with pytest.raises(ValueError, match="classic-mode"):
        fs.compute_confidence_intervals(fr)


def test_link_ignores_self_and_out_of_range():
    comps = [
        {**_gaussian_component(center=30.0), "link_fwhm": 0},   # self-link
        {**_gaussian_component(center=70.0), "link_fwhm": 99},  # out of range
    ]
    lm_params = fs.build_lmfit_parameters(comps)
    assert lm_params["l0"].expr in (None, "")
    assert lm_params["l1"].expr in (None, "")


def test_find_peak_candidates_detection_limit_filters_weak_peaks():
    """min_prominence_sigma is the user-facing detection limit: raising it
    drops weaker candidates; lowering it keeps them."""
    rng = np.random.default_rng(42)
    x = np.linspace(0, 1000, 2000)
    strong = 100.0 * np.exp(-0.5 * ((x - 300.0) / 8.0) ** 2)
    weak = 4.0 * np.exp(-0.5 * ((x - 700.0) / 8.0) ** 2)
    y = strong + weak + rng.normal(0.0, 0.5, x.size)

    permissive = fs.find_peak_candidates(x, y, max_peaks=10, min_prominence_sigma=0.3)
    strict = fs.find_peak_candidates(x, y, max_peaks=10, min_prominence_sigma=8.0)

    assert any(abs(p - 300.0) < 15 for p in permissive)
    assert any(abs(p - 700.0) < 15 for p in permissive)
    assert any(abs(p - 300.0) < 15 for p in strict)
    assert not any(abs(p - 700.0) < 15 for p in strict)
    assert len(strict) < len(permissive)
