"""Tests for xas/lcf_report.py — PDF + MD batch LCF report generation.
No PDF-parsing dependency is added to check page content, so these check
what's cheaply and robustly verifiable: files exist, are non-empty, the
right number of PNGs got produced, and the MD has the expected sample
sections/tables."""
from __future__ import annotations

import numpy as np
import pytest

import xas.lcf_batch as lb
import xas.lcf_report as lr


def _synthetic_references(n_refs=3, n_points=150, seed=0):
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


def _make_batch(n_targets=2):
    energy, refs = _synthetic_references(n_refs=3, seed=1)
    ref_lookup = {name: (e, y) for name, e, y in refs}
    targets = []
    target_lookup = {}
    for i in range(n_targets):
        w = np.array([0.5 + 0.1 * i, 0.3, 0.2 - 0.1 * i])
        w = np.clip(w, 0.0, None)
        y = sum(wi * ref_lookup[f"ref{j}"][1] for j, wi in enumerate(w))
        name = f"sample_{i}"
        targets.append((name, energy, y))
        target_lookup[name] = (energy, y)

    params = lb.BatchLCFParams(min_components=2, max_components=3, top_n_report=5)
    results = lb.run_batch_lcf(targets, refs, params)
    return results, target_lookup, ref_lookup, params


def test_build_batch_report_creates_pdf_md_and_pngs(tmp_path):
    results, target_lookup, ref_lookup, params = _make_batch(n_targets=2)

    out = lr.build_batch_report(
        results, target_lookup, ref_lookup,
        sort_by="r2", top_n=params.top_n_report, out_dir=tmp_path, report_name="test_batch_lcf",
    )

    assert out["pdf"].exists() and out["pdf"].stat().st_size > 0
    assert out["md"].exists() and out["md"].stat().st_size > 0
    pngs = list(out["figures_dir"].glob("*.png"))
    assert len(pngs) == 4  # 2 samples x 2 pages each


def test_build_batch_report_md_contains_each_sample_section(tmp_path):
    results, target_lookup, ref_lookup, params = _make_batch(n_targets=2)
    out = lr.build_batch_report(
        results, target_lookup, ref_lookup,
        sort_by="rms", top_n=params.top_n_report, out_dir=tmp_path, report_name="test_batch_lcf2",
    )
    text = out["md"].read_text(encoding="utf-8")
    assert "## sample_0" in text
    assert "## sample_1" in text
    assert "Best combination:" in text
    assert "| rank |" in text


def test_build_batch_report_handles_empty_results_gracefully(tmp_path):
    results = {"empty_sample": []}
    energy = np.linspace(0, 100, 50)
    target_lookup = {"empty_sample": (energy, np.zeros(50))}
    ref_lookup = {}

    out = lr.build_batch_report(
        results, target_lookup, ref_lookup,
        sort_by="r2", top_n=5, out_dir=tmp_path, report_name="test_batch_lcf_empty",
    )
    text = out["md"].read_text(encoding="utf-8")
    assert "No LCF combinations produced" in text
    # PdfPages only creates the file on its first savefig() call -- with
    # zero samples producing any page, no PDF file is written at all,
    # which is fine (nothing to open); the MD still explains why.


def test_build_fit_overlay_figure_returns_figure_with_axes():
    energy, refs = _synthetic_references(n_refs=2, seed=2)
    ref_lookup = {name: (e, y) for name, e, y in refs}
    target_y = 0.5 * refs[0][2] + 0.5 * refs[1][2]
    results = lb.combinatorial_lcf("s", energy, target_y, refs, min_components=2, max_components=2)

    fig = lr.build_fit_overlay_figure(results[0], energy, target_y, ref_lookup)
    assert len(fig.axes) == 2  # main + residual
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_build_stats_page_figure_returns_figure_with_expected_axes():
    energy, refs = _synthetic_references(n_refs=3, seed=3)
    target_y = refs[0][2]
    results = lb.combinatorial_lcf("s", energy, target_y, refs, min_components=2, max_components=3)

    fig = lr.build_stats_page_figure("s", results, sort_by="r2", top_n=5)
    assert len(fig.axes) == 3  # weight bar, rank scatter, table
    import matplotlib.pyplot as plt
    plt.close(fig)
