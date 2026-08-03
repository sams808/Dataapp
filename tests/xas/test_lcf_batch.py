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
