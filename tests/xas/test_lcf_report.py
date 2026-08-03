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


def test_build_batch_report_pdf_only_writes_no_md_or_images(tmp_path):
    results, target_lookup, ref_lookup, params = _make_batch(n_targets=2)
    out = lr.build_batch_report(
        results, target_lookup, ref_lookup,
        sort_by="r2", top_n=params.top_n_report, out_dir=tmp_path, report_name="pdf_only",
        save_pdf=True, save_md=False, image_formats=(),
    )
    assert set(out.keys()) == {"pdf"}
    assert out["pdf"].exists() and out["pdf"].stat().st_size > 0
    assert not (tmp_path / "pdf_only.md").exists()
    assert not (tmp_path / "pdf_only_figures").exists()


def test_build_batch_report_md_only_writes_no_pdf_or_images(tmp_path):
    results, target_lookup, ref_lookup, params = _make_batch(n_targets=2)
    out = lr.build_batch_report(
        results, target_lookup, ref_lookup,
        sort_by="r2", top_n=params.top_n_report, out_dir=tmp_path, report_name="md_only",
        save_pdf=False, save_md=True, image_formats=(),
    )
    assert set(out.keys()) == {"md"}
    assert out["md"].exists()
    assert not (tmp_path / "md_only.pdf").exists()
    assert not (tmp_path / "md_only_figures").exists()
    text = out["md"].read_text(encoding="utf-8")
    # No images to embed, but the ranking table/stats text is still there.
    assert "![" not in text
    assert "| rank |" in text


def test_build_batch_report_md_with_images_but_no_pdf(tmp_path):
    results, target_lookup, ref_lookup, params = _make_batch(n_targets=1)
    out = lr.build_batch_report(
        results, target_lookup, ref_lookup,
        sort_by="r2", top_n=params.top_n_report, out_dir=tmp_path, report_name="md_with_images",
        save_pdf=False, save_md=True, image_formats=("png",),
    )
    assert set(out.keys()) == {"md", "figures_dir"}
    text = out["md"].read_text(encoding="utf-8")
    assert "![" in text
    assert len(list(out["figures_dir"].glob("*.png"))) == 2


def test_build_batch_report_requires_at_least_one_of_pdf_or_md(tmp_path):
    results, target_lookup, ref_lookup, params = _make_batch(n_targets=1)
    with pytest.raises(ValueError):
        lr.build_batch_report(
            results, target_lookup, ref_lookup,
            sort_by="r2", top_n=params.top_n_report, out_dir=tmp_path, report_name="neither",
            save_pdf=False, save_md=False, image_formats=("png",),
        )


def test_build_batch_report_svg_only():
    results, target_lookup, ref_lookup, params = _make_batch(n_targets=1)
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        out = lr.build_batch_report(
            results, target_lookup, ref_lookup,
            sort_by="r2", top_n=params.top_n_report, out_dir=Path(td), report_name="svg_only",
            save_pdf=False, save_md=True, image_formats=("svg",),
        )
        pngs = list(out["figures_dir"].glob("*.png"))
        svgs = list(out["figures_dir"].glob("*.svg"))
        assert len(pngs) == 0
        assert len(svgs) == 2
        # SVG is not PNG, so it's still what gets embedded (only format available).
        text = out["md"].read_text(encoding="utf-8")
        assert ".svg)" in text


def test_build_batch_report_png_and_svg_both_prefers_png_for_md_embed():
    results, target_lookup, ref_lookup, params = _make_batch(n_targets=1)
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        out = lr.build_batch_report(
            results, target_lookup, ref_lookup,
            sort_by="r2", top_n=params.top_n_report, out_dir=Path(td), report_name="both_formats",
            save_pdf=False, save_md=True, image_formats=("svg", "png"),
        )
        assert len(list(out["figures_dir"].glob("*.png"))) == 2
        assert len(list(out["figures_dir"].glob("*.svg"))) == 2
        text = out["md"].read_text(encoding="utf-8")
        assert ".png)" in text
        assert ".svg)" not in text  # PNG preferred for the embed even though both exist


def test_build_batch_report_rejects_unknown_image_format(tmp_path):
    results, target_lookup, ref_lookup, params = _make_batch(n_targets=1)
    with pytest.raises(ValueError):
        lr.build_batch_report(
            results, target_lookup, ref_lookup,
            sort_by="r2", top_n=params.top_n_report, out_dir=tmp_path, report_name="bad_fmt",
            image_formats=("jpg",),
        )


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


def test_build_fit_overlay_figure_uses_fit_energy_not_full_target_range():
    """When fit_range narrowed the actual fit domain, the plotted fit
    line/residual must span result.fit_energy, not the full target
    range -- and the un-fit region should be visibly shaded."""
    import matplotlib.pyplot as plt
    energy, refs = _synthetic_references(n_refs=2, seed=4)
    ref_lookup = {name: (e, y) for name, e, y in refs}
    target_y = 0.5 * refs[0][2] + 0.5 * refs[1][2]
    fit_range = (20.0, 80.0)
    results = lb.combinatorial_lcf(
        "s", energy, target_y, refs, min_components=2, max_components=2, fit_range=fit_range,
    )

    fig = lr.build_fit_overlay_figure(results[0], energy, target_y, ref_lookup)
    ax_main = fig.axes[0]
    # One of the plotted lines should be the fit, spanning only fit_range.
    fit_line = [ln for ln in ax_main.get_lines() if "LCF fit" in (ln.get_label() or "")][0]
    xdata = fit_line.get_xdata()
    assert xdata.min() >= fit_range[0] - 1e-6
    assert xdata.max() <= fit_range[1] + 1e-6
    # The shaded "fit range" span (axvspan) should be present as a patch
    # since the fit range is narrower than the full target range.
    assert len(ax_main.patches) > 0
    plt.close(fig)


def test_build_fit_overlay_figure_shows_e0_shift_when_aligned():
    import matplotlib.pyplot as plt
    energy, refs = _synthetic_references(n_refs=2, seed=5)
    ref_lookup = {name: (e, y) for name, e, y in refs}
    target_y = 0.5 * refs[0][2] + 0.5 * refs[1][2]
    results = lb.combinatorial_lcf("s", energy, target_y, refs, min_components=2, max_components=2, align_e0=True)

    fig = lr.build_fit_overlay_figure(results[0], energy, target_y, ref_lookup)
    ax_main = fig.axes[0]
    labels = [ln.get_label() for ln in ax_main.get_lines()]
    assert any("e0 shift" in lbl for lbl in labels)
    plt.close(fig)
