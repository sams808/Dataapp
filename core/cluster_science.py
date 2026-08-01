"""
cluster_science.py — cluster analysis for spectral series (M16),
framework-agnostic. RamanLab-inspired: group similar spectra in a
multi-point/map series to spot phases or spatial regions.

Pipeline: interpolate every spectrum onto one common grid
(build_feature_matrix) -> KMeans or agglomerative clustering
(cluster_spectra) -> 2-component PCA projection for the overview scatter
(pca_scores) -> per-cluster mean spectra (cluster_means).

scikit-learn is imported lazily inside the functions so the module (and
anything that merely imports it) stays usable without sklearn installed;
the functions themselves raise a clear ImportError when actually called.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


def build_feature_matrix(
    spectra_xy: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    n_points: int = 500,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    normalize: Optional[str] = "area",
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate every (x, y) spectrum onto one shared grid and stack
    into an (n_spectra, n_points) matrix. The grid spans the OVERLAP of
    all spectra (optionally further restricted by x_min/x_max) — comparing
    intensities outside a spectrum's measured range would be extrapolated
    fiction. normalize: "area" (integral -> 1), "max" (peak -> 1), or None.
    """
    if len(spectra_xy) < 2:
        raise ValueError("Need at least 2 spectra to build a feature matrix.")

    lo = max(float(np.nanmin(x)) for x, _ in spectra_xy)
    hi = min(float(np.nanmax(x)) for x, _ in spectra_xy)
    if x_min is not None:
        lo = max(lo, float(x_min))
    if x_max is not None:
        hi = min(hi, float(x_max))
    if hi <= lo:
        raise ValueError(f"Spectra have no common x-range (overlap [{lo}, {hi}]).")

    grid = np.linspace(lo, hi, int(n_points))
    rows = []
    for x, y in spectra_xy:
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        m = np.isfinite(x) & np.isfinite(y)
        order = np.argsort(x[m], kind="mergesort")
        yi = np.interp(grid, x[m][order], y[m][order])
        if normalize == "area":
            area = np.trapz(yi, grid)
            if abs(area) > 1e-30:
                yi = yi / area
        elif normalize == "max":
            peak = np.nanmax(np.abs(yi))
            if peak > 1e-30:
                yi = yi / peak
        rows.append(yi)
    return np.vstack(rows), grid


def cluster_spectra(
    matrix: np.ndarray, *, method: str = "kmeans", n_clusters: int = 3, random_state: int = 0,
) -> Dict[str, Any]:
    """Cluster the rows of the feature matrix. Returns labels plus a
    silhouette score (when computable) as a cluster-quality hint."""
    try:
        from sklearn.cluster import AgglomerativeClustering, KMeans
        from sklearn.metrics import silhouette_score
    except ImportError as exc:
        raise ImportError("Cluster analysis requires scikit-learn (pip install scikit-learn).") from exc

    matrix = np.asarray(matrix, float)
    n_clusters = int(n_clusters)
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2.")
    if n_clusters >= len(matrix):
        raise ValueError(f"n_clusters ({n_clusters}) must be smaller than the number of spectra ({len(matrix)}).")

    method = method.lower().strip()
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = model.fit_predict(matrix)
    elif method in ("hierarchical", "agglomerative"):
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(matrix)
    else:
        raise ValueError(f"Unknown clustering method: {method!r} (expected 'kmeans' or 'hierarchical').")

    silhouette = None
    if len(set(labels.tolist())) > 1:
        try:
            silhouette = float(silhouette_score(matrix, labels))
        except Exception:
            silhouette = None

    return {"labels": labels.astype(int), "silhouette": silhouette, "method": method, "n_clusters": n_clusters}


def pca_scores(matrix: np.ndarray, n_components: int = 2) -> Dict[str, Any]:
    """Project spectra onto their first principal components — the
    RamanLab-style 2D overview scatter, and (per M21) the 'how many
    distinct species are present' diagnostic via explained variance."""
    try:
        from sklearn.decomposition import PCA
    except ImportError as exc:
        raise ImportError("PCA requires scikit-learn (pip install scikit-learn).") from exc

    matrix = np.asarray(matrix, float)
    n_components = min(int(n_components), matrix.shape[0], matrix.shape[1])
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(matrix)
    return {
        "scores": scores,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "components": pca.components_,
    }


def cluster_means(matrix: np.ndarray, labels: np.ndarray) -> Dict[int, np.ndarray]:
    """Mean spectrum per cluster label."""
    matrix = np.asarray(matrix, float)
    labels = np.asarray(labels, int)
    return {int(lbl): matrix[labels == lbl].mean(axis=0) for lbl in sorted(set(labels.tolist()))}
