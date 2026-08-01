"""Tests for cluster_science.py (M16) — spectral series clustering.

Synthetic ground truth: two well-separated spectral families (a peak at
300 vs a peak at 700) must land in two clean clusters regardless of
method. Plus a real-data run over the archived PBi0-1 Raman map series
when it's present on this machine.
"""
from __future__ import annotations

import numpy as np
import pytest
import rampy as rp

import core.cluster_science as cs
from conftest import ARCHIVE_DIR


def _family(center, n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        x = np.linspace(0, 1000, 800)
        amp = 80 + rng.normal(0, 5)
        y = rp.gaussian(x, amp, center + rng.normal(0, 2), 20.0) + rng.normal(0, 0.5, x.shape)
        out.append((x, y))
    return out


def test_build_feature_matrix_shapes_and_grid_overlap():
    a = (np.linspace(0, 100, 50), np.ones(50))
    b = (np.linspace(20, 120, 60), np.ones(60))
    matrix, grid = cs.build_feature_matrix([a, b], n_points=100, normalize=None)
    assert matrix.shape == (2, 100)
    assert grid[0] == pytest.approx(20.0)   # overlap only
    assert grid[-1] == pytest.approx(100.0)


def test_build_feature_matrix_normalize_area():
    a = (np.linspace(0, 100, 200), np.full(200, 2.0))
    b = (np.linspace(0, 100, 200), np.full(200, 8.0))
    matrix, grid = cs.build_feature_matrix([a, b], n_points=200, normalize="area")
    # After area normalization both flat spectra become identical.
    assert np.allclose(matrix[0], matrix[1])


def test_build_feature_matrix_rejects_disjoint_ranges():
    a = (np.linspace(0, 10, 20), np.ones(20))
    b = (np.linspace(50, 60, 20), np.ones(20))
    with pytest.raises(ValueError, match="common x-range"):
        cs.build_feature_matrix([a, b])


@pytest.mark.parametrize("method", ["kmeans", "hierarchical"])
def test_cluster_spectra_separates_two_known_families(method):
    spectra = _family(300.0, 6, seed=1) + _family(700.0, 6, seed=2)
    matrix, _ = cs.build_feature_matrix(spectra, normalize="max")
    result = cs.cluster_spectra(matrix, method=method, n_clusters=2)

    labels = result["labels"]
    assert len(set(labels[:6].tolist())) == 1   # family 1 all together
    assert len(set(labels[6:].tolist())) == 1   # family 2 all together
    assert labels[0] != labels[6]               # and apart from each other
    assert result["silhouette"] is not None and result["silhouette"] > 0.5


def test_cluster_spectra_validates_inputs():
    matrix = np.random.default_rng(0).normal(size=(5, 20))
    with pytest.raises(ValueError, match="at least 2"):
        cs.cluster_spectra(matrix, n_clusters=1)
    with pytest.raises(ValueError, match="smaller than"):
        cs.cluster_spectra(matrix, n_clusters=5)
    with pytest.raises(ValueError, match="Unknown clustering method"):
        cs.cluster_spectra(matrix, method="umap", n_clusters=2)


def test_pca_scores_separate_families_on_first_component():
    spectra = _family(300.0, 6, seed=3) + _family(700.0, 6, seed=4)
    matrix, _ = cs.build_feature_matrix(spectra, normalize="max")
    out = cs.pca_scores(matrix, n_components=2)
    assert out["scores"].shape == (12, 2)
    pc1_a = out["scores"][:6, 0]
    pc1_b = out["scores"][6:, 0]
    assert (pc1_a.max() < pc1_b.min()) or (pc1_b.max() < pc1_a.min())
    assert out["explained_variance_ratio"][0] > 0.5


def test_cluster_means_returns_one_mean_per_label():
    matrix = np.vstack([np.zeros((3, 10)), np.ones((2, 10)) * 4.0])
    labels = np.array([0, 0, 0, 1, 1])
    means = cs.cluster_means(matrix, labels)
    assert set(means) == {0, 1}
    assert np.allclose(means[0], 0.0)
    assert np.allclose(means[1], 4.0)


@pytest.mark.skipif(not (ARCHIVE_DIR / "PBi0-1").is_dir(), reason="archived PBi0-1 map series not present")
def test_clustering_runs_on_real_pbi0_map_series(pbi0_map_paths):
    """Real-data smoke: the archived 56-point Raman map series must cluster
    without errors and produce sane structures (no ground-truth labels
    exist, so only invariants are checked)."""
    import core.io_universal as iu

    paths = pbi0_map_paths[:20]  # subset keeps the test fast
    assert len(paths) >= 10
    spectra = []
    for p in paths:
        df, meta = iu.load_any(str(p), return_meta=True)
        canon = meta.get("canonical_map", {})
        x = df[canon.get("X", df.columns[0])].astype(float).to_numpy()
        y = df[canon.get("Y", df.columns[1])].astype(float).to_numpy()
        spectra.append((x, y))

    matrix, grid = cs.build_feature_matrix(spectra, normalize="area")
    result = cs.cluster_spectra(matrix, method="kmeans", n_clusters=3)
    assert len(result["labels"]) == len(paths)
    assert set(result["labels"].tolist()) <= {0, 1, 2}
    means = cs.cluster_means(matrix, result["labels"])
    for m in means.values():
        assert m.shape == grid.shape
