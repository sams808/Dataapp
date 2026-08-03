"""Tests for xas/lcf_batch.py — combinatorial (batch) linear combination
fitting. Independent of Qt; synthetic references + a target built as a
KNOWN exact combination of a subset of them, so the combinatorial search
has a ground-truth answer to be checked against."""
from __future__ import annotations

import numpy as np
import pytest

import xas.lcf_batch as lb


def _synthetic_references(n_refs=4, n_points=200, seed=0):
    """n_refs distinct, smooth, non-collinear curves on a shared energy
    grid -- shaped roughly like XANES/EXAFS spectra (a few Gaussian
    features at different positions/widths per reference) so linear
    combinations of them are realistically distinguishable."""
    rng = np.random.default_rng(seed)
    energy = np.linspace(0.0, 100.0, n_points)
    refs = []
    for i in range(n_refs):
        centers = rng.uniform(10, 90, size=3)
        widths = rng.uniform(3, 10, size=3)
        heights = rng.uniform(0.5, 1.5, size=3)
        y = np.zeros_like(energy)
        for c, w, h in zip(centers, widths, heights):
            y += h * np.exp(-((energy - c) ** 2) / (2 * w ** 2))
        refs.append((f"ref{i}", energy, y))
    return energy, refs


def test_fit_weighted_sum_recovers_known_weights_no_constraint():
    energy, refs = _synthetic_references(n_refs=3)
    A = np.column_stack([y for _, _, y in refs])
    true_weights = np.array([0.3, 0.5, 0.2])
    target = A @ true_weights

    weights, fit_y = lb.fit_weighted_sum(target, A, weight_bounds=(0.0, 1.0))
    assert np.allclose(weights, true_weights, atol=1e-6)
    assert np.allclose(fit_y, target, atol=1e-6)


def test_fit_weighted_sum_respects_upper_bound():
    energy, refs = _synthetic_references(n_refs=2)
    A = np.column_stack([y for _, _, y in refs])
    target = 0.9 * A[:, 0] + 0.1 * A[:, 1]

    weights, _fit_y = lb.fit_weighted_sum(target, A, weight_bounds=(0.0, 0.5))
    assert np.all(weights <= 0.5 + 1e-9)
    assert np.all(weights >= -1e-9)


def test_fit_weighted_sum_sum_to_one_constraint_is_enforced():
    energy, refs = _synthetic_references(n_refs=3)
    A = np.column_stack([y for _, _, y in refs])
    target = 0.4 * A[:, 0] + 0.4 * A[:, 1] + 0.2 * A[:, 2]

    weights, _fit_y = lb.fit_weighted_sum(target, A, weight_bounds=(0.0, 1.0), sum_to_one=True)
    assert abs(np.sum(weights) - 1.0) < 1e-6
    assert np.allclose(weights, [0.4, 0.4, 0.2], atol=1e-3)


def test_score_fit_perfect_fit_gives_r2_one_rms_zero():
    target = np.array([1.0, 2.0, 3.0, 4.0])
    scores = lb.score_fit(target, target.copy(), n_params=1)
    assert scores["r2"] == pytest.approx(1.0)
    assert scores["rms"] == pytest.approx(0.0, abs=1e-12)


def test_combinatorial_lcf_ranks_true_combination_first():
    """The target is an EXACT combination of ref0+ref1 only. Searching
    combinations of size 2-3 across 4 references, the true 2-component
    combination should come out on top by R² (best possible fit, fewer
    parameters) even though 3-component combinations are also tried."""
    energy, refs = _synthetic_references(n_refs=4, seed=1)
    ref_lookup = {name: y for name, _, y in refs}
    target_y = 0.6 * ref_lookup["ref0"] + 0.4 * ref_lookup["ref1"]

    results = lb.combinatorial_lcf(
        "sample_A", energy, target_y, refs,
        min_components=2, max_components=3, sort_by="r2",
    )
    assert len(results) > 0
    best = results[0]
    assert set(best.ref_names) == {"ref0", "ref1"}
    assert best.r2 == pytest.approx(1.0, abs=1e-6)
    # Results must actually be sorted best-first.
    r2_values = [r.r2 for r in results]
    assert r2_values == sorted(r2_values, reverse=True)


def test_combinatorial_lcf_rms_sort_is_ascending_quality():
    energy, refs = _synthetic_references(n_refs=4, seed=2)
    ref_lookup = {name: y for name, _, y in refs}
    target_y = 0.7 * ref_lookup["ref2"] + 0.3 * ref_lookup["ref3"]

    results = lb.combinatorial_lcf(
        "sample_B", energy, target_y, refs,
        min_components=2, max_components=2, sort_by="rms",
    )
    rms_values = [r.rms for r in results]
    assert rms_values == sorted(rms_values)  # ascending: lower RMS is better, first


def test_combinatorial_lcf_required_refs_filters_combinations():
    energy, refs = _synthetic_references(n_refs=4, seed=3)
    results = lb.combinatorial_lcf(
        "sample_C", energy, refs[0][2], refs,
        min_components=2, max_components=3, required_refs=["ref3"],
    )
    assert len(results) > 0
    assert all("ref3" in r.ref_names for r in results)


def test_combinatorial_lcf_combination_count_matches_math():
    """4 references, sizes 2-3 -> C(4,2) + C(4,3) = 6 + 4 = 10 combinations."""
    energy, refs = _synthetic_references(n_refs=4, seed=4)
    results = lb.combinatorial_lcf("sample_D", energy, refs[0][2], refs, min_components=2, max_components=3)
    assert len(results) == 10


def test_combinatorial_lcf_rejects_duplicate_reference_names():
    energy, refs = _synthetic_references(n_refs=2, seed=5)
    dup_refs = refs + [("ref0", refs[0][1], refs[0][2])]
    with pytest.raises(ValueError):
        lb.combinatorial_lcf("sample_E", energy, refs[0][2], dup_refs)


def test_run_batch_lcf_covers_every_target():
    energy, refs = _synthetic_references(n_refs=3, seed=6)
    targets = [("t1", energy, refs[0][2]), ("t2", energy, refs[1][2])]
    params = lb.BatchLCFParams(min_components=2, max_components=2)
    out = lb.run_batch_lcf(targets, refs, params)
    assert set(out.keys()) == {"t1", "t2"}
    assert all(len(v) > 0 for v in out.values())


def test_weight_map_matches_ref_names_and_weights():
    energy, refs = _synthetic_references(n_refs=2, seed=7)
    results = lb.combinatorial_lcf("sample_F", energy, refs[0][2], refs, min_components=2, max_components=2)
    wm = results[0].weight_map()
    assert set(wm.keys()) == set(results[0].ref_names)
    assert all(isinstance(v, float) for v in wm.values())


def test_result_fit_energy_matches_target_energy_when_no_fit_range():
    energy, refs = _synthetic_references(n_refs=2, seed=8)
    results = lb.combinatorial_lcf("s", energy, refs[0][2], refs, min_components=2, max_components=2)
    assert np.allclose(results[0].fit_energy, energy)
    assert results[0].fit_y.shape == energy.shape


# --------------------------------------------------------------------------
# Per-reference weight bounds -- encoding real prior expectations
# (e.g. "ref0 should dominate, ref1 should stay minor")
# --------------------------------------------------------------------------

def test_per_ref_bounds_forces_named_reference_above_its_floor():
    energy, refs = _synthetic_references(n_refs=2, seed=9)
    ref_lookup = {name: y for name, _, y in refs}
    # True mix has ref0 as a MINOR component; force it major via bounds
    # and confirm the fit actually respects the floor (at real fit-quality
    # cost, since it no longer matches the true generating weights).
    target_y = 0.1 * ref_lookup["ref0"] + 0.9 * ref_lookup["ref1"]

    unconstrained = lb.combinatorial_lcf("s", energy, target_y, refs, min_components=2, max_components=2)
    forced = lb.combinatorial_lcf(
        "s", energy, target_y, refs, min_components=2, max_components=2,
        per_ref_bounds={"ref0": (0.6, 1.0)},
    )
    assert unconstrained[0].weight_map()["ref0"] < 0.6
    assert forced[0].weight_map()["ref0"] >= 0.6 - 1e-9
    assert forced[0].r2 <= unconstrained[0].r2 + 1e-9  # constraint can only cost fit quality, never help


def test_per_ref_bounds_caps_named_reference_below_ceiling():
    energy, refs = _synthetic_references(n_refs=2, seed=10)
    ref_lookup = {name: y for name, _, y in refs}
    target_y = 0.5 * ref_lookup["ref0"] + 0.5 * ref_lookup["ref1"]

    results = lb.combinatorial_lcf(
        "s", energy, target_y, refs, min_components=2, max_components=2,
        per_ref_bounds={"ref1": (0.0, 0.2)},
    )
    assert results[0].weight_map()["ref1"] <= 0.2 + 1e-9


def test_per_ref_bounds_leaves_unlisted_references_at_default():
    energy, refs = _synthetic_references(n_refs=3, seed=11)
    ref_lookup = {name: y for name, _, y in refs}
    target_y = 0.3 * ref_lookup["ref0"] + 0.3 * ref_lookup["ref1"] + 0.4 * ref_lookup["ref2"]

    # Only ref0 gets a custom ceiling; ref1/ref2 should still be free to
    # roam the default (0, 1) range.
    results = lb.combinatorial_lcf(
        "s", energy, target_y, refs, min_components=3, max_components=3,
        weight_bounds=(0.0, 1.0), per_ref_bounds={"ref0": (0.0, 0.05)},
    )
    wm = results[0].weight_map()
    assert wm["ref0"] <= 0.05 + 1e-9
    assert wm["ref1"] > 0.05 or wm["ref2"] > 0.05  # at least one absorbed the rest, unconstrained


def test_per_ref_bounds_invalid_range_raises():
    energy, refs = _synthetic_references(n_refs=2, seed=12)
    with pytest.raises(ValueError):
        lb.combinatorial_lcf(
            "s", energy, refs[0][2], refs, min_components=2, max_components=2,
            per_ref_bounds={"ref0": (0.8, 0.2)},
        )


def test_per_ref_bounds_with_sum_to_one():
    energy, refs = _synthetic_references(n_refs=3, seed=13)
    ref_lookup = {name: y for name, _, y in refs}
    target_y = 0.2 * ref_lookup["ref0"] + 0.3 * ref_lookup["ref1"] + 0.5 * ref_lookup["ref2"]

    results = lb.combinatorial_lcf(
        "s", energy, target_y, refs, min_components=3, max_components=3,
        per_ref_bounds={"ref0": (0.5, 1.0)}, sum_to_one=True,
    )
    wm = results[0].weight_map()
    assert wm["ref0"] >= 0.5 - 1e-6
    assert abs(sum(wm.values()) - 1.0) < 1e-6


# --------------------------------------------------------------------------
# align_e0 -- fixes energy-misaligned references being absorbed into
# unphysical weights instead of a real edge-position mismatch
# --------------------------------------------------------------------------

def test_estimate_e0_deriv_finds_known_edge_position():
    from scipy.special import erf
    energy = np.linspace(6900, 7300, 400)
    e0_true = 7112.0
    mu = 0.1 + 0.5 * (1 + erf((energy - e0_true) / 2.0))
    e0_est = lb.estimate_e0_deriv(energy, mu)
    assert abs(e0_est - e0_true) < 2.0  # within a couple grid points of the true edge


def _synthetic_edge_spectrum(e0, energy):
    """A real sigmoid edge (unlike _synthetic_references' multi-Gaussian-
    peak shapes, which have no single well-defined "edge" for
    estimate_e0_deriv to lock onto) -- needed for align_e0 tests
    specifically."""
    from scipy.special import erf
    return 0.1 + 0.5 * (1.0 + erf((energy - e0) / 2.5))


def test_align_e0_recovers_fit_quality_lost_to_misalignment():
    """Two references with DIFFERENT true edge positions; the target is
    an exact mix of them (each still at ITS OWN edge position, as real
    unaligned standards would be) plus a shared reference2 held fixed.
    align_e0=False should fit noticeably worse than align_e0=True, since
    only the aligned fit can actually match a target edge sitting between
    two differently-positioned reference edges."""
    energy = np.linspace(6900.0, 7300.0, 300)
    ref_e0s = {"edgeA": 7112.0, "edgeB": 7118.0}  # 6 eV apart -- a real, if generous, mismatch
    refs = [(name, energy, _synthetic_edge_spectrum(e0, energy)) for name, e0 in ref_e0s.items()]
    ref_lookup = {name: y for name, _, y in refs}

    # A real, single-edge target sitting between the two reference edges
    # -- no reference has the CORRECT edge position, only alignment can
    # actually match its steep rising region well; an unconstrained
    # linear combination of two same-shape-but-shifted edges can distort
    # the fitted transition but can't reproduce the target's true slope
    # location as well as shifting each reference onto it first.
    target_y = _synthetic_edge_spectrum(7115.0, energy)

    unaligned = lb.combinatorial_lcf("s", energy, target_y, refs, min_components=2, max_components=2, align_e0=False)
    aligned = lb.combinatorial_lcf("s", energy, target_y, refs, min_components=2, max_components=2, align_e0=True)
    assert aligned[0].r2 > unaligned[0].r2


def test_align_e0_records_shift_per_reference_in_result():
    energy, refs = _synthetic_references(n_refs=2, seed=15)
    results = lb.combinatorial_lcf("s", energy, refs[0][2], refs, min_components=2, max_components=2, align_e0=True)
    assert set(results[0].e0_shifts_ev.keys()) == set(results[0].ref_names)
    assert all(isinstance(v, float) for v in results[0].e0_shifts_ev.values())


def test_no_align_e0_leaves_e0_shifts_empty():
    energy, refs = _synthetic_references(n_refs=2, seed=16)
    results = lb.combinatorial_lcf("s", energy, refs[0][2], refs, min_components=2, max_components=2, align_e0=False)
    assert results[0].e0_shifts_ev == {}


# --------------------------------------------------------------------------
# fit_range -- restrict fitting/scoring to an energy sub-window
# --------------------------------------------------------------------------

def test_fit_range_restricts_fit_energy_and_scoring():
    energy, refs = _synthetic_references(n_refs=2, seed=17)
    fit_range = (20.0, 80.0)
    results = lb.combinatorial_lcf(
        "s", energy, refs[0][2], refs, min_components=2, max_components=2, fit_range=fit_range,
    )
    assert results[0].fit_energy.min() >= fit_range[0]
    assert results[0].fit_energy.max() <= fit_range[1]
    assert results[0].n_points < energy.size


def test_fit_range_outside_data_raises():
    energy, refs = _synthetic_references(n_refs=2, seed=18)
    with pytest.raises(ValueError):
        lb.combinatorial_lcf(
            "s", energy, refs[0][2], refs, min_components=2, max_components=2, fit_range=(1000.0, 2000.0),
        )


def test_batch_lcf_params_new_fields_have_sensible_defaults():
    params = lb.BatchLCFParams()
    assert params.align_e0 is False
    assert params.fit_range is None
    assert params.per_ref_bounds == {}


def test_run_batch_lcf_passes_through_new_params():
    energy, refs = _synthetic_references(n_refs=2, seed=19)
    ref_lookup = {name: y for name, _, y in refs}
    targets = [("t1", energy, 0.5 * ref_lookup["ref0"] + 0.5 * ref_lookup["ref1"])]
    params = lb.BatchLCFParams(min_components=2, max_components=2, per_ref_bounds={"ref0": (0.3, 1.0)}, align_e0=True)
    out = lb.run_batch_lcf(targets, refs, params)
    assert out["t1"][0].weight_map()["ref0"] >= 0.3 - 1e-9
    assert out["t1"][0].e0_shifts_ev  # align_e0 was on
