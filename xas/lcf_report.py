"""
xas/lcf_report.py — PDF + MD report builder for batch/combinatorial LCF
results (xas/lcf_batch.py). One matplotlib Figure object feeds both the
standalone image file(s) (embedded in the .md) and the corresponding
page in the shared multi-page PDF -- same pattern the standalone Bi L3
xas_pipeline's own reports.py uses (PdfPages + parallel markdown, zero
new dependencies), kept independent of Qt so it's usable from a plain
script too.

Two pages per sample (a single page gets cramped trying to fit the fit
overlay, residual, weight bar chart, rank-quality scatter, AND the
ranking table all at once and still be readable):
  1. Fit overlay -- target vs best combination's fit, each weighted
     reference contribution, residual below. The primary, "which
     combination actually explains this spectrum" plot. If the fit was
     restricted to a sub-range (lcf_batch's `fit_range`), that window is
     shaded on the full spectrum so it's clear what was and wasn't fit.
  2. Stats/ranking -- best-fit weight bar chart, sort-metric-vs-rank
     scatter across every combination tried (a real degeneracy signal:
     a flat curve near rank 1 means several combinations fit about
     equally well, not that the winner is uniquely correct), and the
     top-N ranking table.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .lcf_batch import LCFCombinationResult
from .xas_science import _interp_to_grid

_METRIC_LABELS = {"r2": "R²", "rms": "RMS residual", "reduced_chi_square": "reduced χ²"}
_VALID_IMAGE_FORMATS = ("png", "svg")


def _fmt(v, nd=4) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:.{nd}f}"


def build_fit_overlay_figure(result: LCFCombinationResult, target_energy: np.ndarray, target_y: np.ndarray,
                             ref_lookup: Dict[str, Tuple[np.ndarray, np.ndarray]],
                             x_label: str = "Energy (eV)", y_label: str = "Normalized signal") -> plt.Figure:
    """Page 1: the primary "mainly the best one" plot -- data, best-fit
    combination, each weighted reference contribution, residual below.
    `result.fit_energy`/`result.fit_y` (not `target_energy`/`target_y`)
    are the actual fit domain -- narrower than the full target spectrum
    whenever `fit_range` was used, in which case that window is shaded
    on the full-spectrum data curve so it's visible what was and wasn't
    included in the fit and its R²/RMS/χ²."""
    fig = plt.figure(figsize=(8.5, 11))
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 3, 1.4, 0.1])
    ax_main = fig.add_subplot(gs[0:2, 0])
    ax_resid = fig.add_subplot(gs[2, 0], sharex=ax_main)

    target_energy = np.asarray(target_energy, float)
    fit_lo, fit_hi = float(np.min(result.fit_energy)), float(np.max(result.fit_energy))
    full_lo, full_hi = float(np.min(target_energy)), float(np.max(target_energy))
    if fit_lo > full_lo + 1e-6 or fit_hi < full_hi - 1e-6:
        ax_main.axvspan(fit_lo, fit_hi, color="0.85", zorder=0, label="fit range")

    ax_main.plot(target_energy, target_y, color="black", lw=1.5, label=f"{result.target} (data)", zorder=5)
    ax_main.plot(result.fit_energy, result.fit_y, color="red", lw=1.6, ls="--",
                 label=f"LCF fit (R²={result.r2:.4f})", zorder=4)

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(result.ref_names), 1)))
    for name, w, color in zip(result.ref_names, result.weights, colors):
        ref_e, ref_y = ref_lookup[name]
        shift = result.e0_shifts_ev.get(name, 0.0)
        ref_interp = _interp_to_grid(np.asarray(ref_e, float) + shift, ref_y, result.fit_energy)
        shift_note = f", e0 shift {shift:+.1f} eV" if shift else ""
        ax_main.plot(result.fit_energy, w * ref_interp, color=color, lw=1.1, alpha=0.85,
                     label=f"{name} × {w:.3f}{shift_note}")

    ax_main.set_ylabel(y_label)
    ax_main.set_title(f"{result.target} — best combination: {' + '.join(result.ref_names)}", fontsize=12, loc="left")
    ax_main.legend(fontsize=7.5, loc="best", framealpha=0.9)
    ax_main.grid(alpha=0.25)
    ax_main.tick_params(labelbottom=False)

    # residual = data - fit, both defined on fit_energy (fit_y is already
    # the combination's own prediction there; interpolate the full-
    # resolution target data onto that same grid for the subtraction).
    target_on_fit_grid = _interp_to_grid(target_energy, target_y, result.fit_energy)
    resid = target_on_fit_grid - result.fit_y
    ax_resid.plot(result.fit_energy, resid, color="0.3", lw=1.0)
    ax_resid.axhline(0.0, color="black", lw=0.7, ls=":")
    ax_resid.set_ylabel("residual")
    ax_resid.set_xlabel(x_label)
    ax_resid.grid(alpha=0.25)
    ax_resid.set_xlim(ax_main.get_xlim())

    fig.suptitle(f"Batch LCF — {result.target}", fontsize=10, y=0.98, color="0.4")
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.06, top=0.92, hspace=0.12)
    return fig


def build_stats_page_figure(target_name: str, ranked_results: List[LCFCombinationResult],
                            sort_by: str, top_n: int = 10) -> plt.Figure:
    """Page 2: best-fit weight bar chart, metric-vs-rank scatter across
    every combination tried, and the top-N ranking table."""
    fig = plt.figure(figsize=(8.5, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[3.0, 0.1, 2.0])

    best = ranked_results[0]
    ax_w = fig.add_subplot(gs[0, 0])
    ax_w.bar(best.ref_names, best.weights, color="tab:blue")
    ax_w.set_ylabel("weight")
    ax_w.set_title("Best-fit weights", fontsize=10)
    ax_w.tick_params(axis="x", rotation=30, labelsize=7.5)
    ax_w.grid(alpha=0.25, axis="y")

    ax_rank = fig.add_subplot(gs[0, 1])
    ranks = np.arange(1, len(ranked_results) + 1)
    metric_vals = [getattr(r, sort_by) for r in ranked_results]
    ax_rank.plot(ranks, metric_vals, "o-", ms=3, lw=1.0, color="tab:orange")
    ax_rank.set_xlabel("rank")
    ax_rank.set_ylabel(_METRIC_LABELS.get(sort_by, sort_by))
    ax_rank.set_title(f"{_METRIC_LABELS.get(sort_by, sort_by)} across {len(ranked_results)} combinations", fontsize=9)
    ax_rank.grid(alpha=0.25)

    ax_tbl = fig.add_subplot(gs[2, :])
    ax_tbl.axis("off")
    rows = []
    for i, r in enumerate(ranked_results[:top_n], start=1):
        composition = ", ".join(f"{n}:{w:.3f}" for n, w in zip(r.ref_names, r.weights))
        rows.append([str(i), composition, _fmt(r.r2), _fmt(r.rms, 5), _fmt(r.reduced_chi_square, 5)])
    tbl = ax_tbl.table(cellText=rows, colLabels=["rank", "combination (name:weight)", "R²", "RMS", "reduced χ²"],
                       cellLoc="left", bbox=[0.0, 1.0 - 0.11 * (len(rows) + 1), 1.0, 0.11 * (len(rows) + 1)])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.auto_set_column_width(list(range(5)))
    ax_tbl.set_title(f"Ranking (top {min(top_n, len(ranked_results))} of {len(ranked_results)}, sorted by {_METRIC_LABELS.get(sort_by, sort_by)})",
                     fontsize=10, loc="left")

    fig.suptitle(f"Batch LCF — {target_name} — stats", fontsize=10, y=0.98, color="0.4")
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.05, top=0.90, hspace=0.55, wspace=0.35)
    return fig


def build_batch_report(results_by_target: Dict[str, List[LCFCombinationResult]],
                       target_lookup: Dict[str, Tuple[np.ndarray, np.ndarray]],
                       ref_lookup: Dict[str, Tuple[np.ndarray, np.ndarray]], *,
                       sort_by: str, top_n: int, out_dir: Path, report_name: str = "batch_lcf_report",
                       save_pdf: bool = True, save_md: bool = True,
                       image_formats: Sequence[str] = ("png",),
                       x_label: str = "Energy (eV)", y_label: str = "Normalized signal") -> Dict[str, Path]:
    """Writes any combination of `<report_name>.pdf` (all samples, 2 pages
    each), `<report_name>.md`, and standalone per-page images (in
    `<report_name>_figures/`, one file per requested format in
    `image_formats` -- e.g. `("png", "svg")` writes both) to `out_dir`.
    Each output is independent: PDF-only or MD-only (with or without
    images, in either or both formats) all work. If save_md is True but
    image_formats is empty, the MD still gets its ranking tables/stats
    text, just without embedded images (there's nothing to embed --
    the first format in image_formats, if any, is what gets linked).
    save_pdf and a non-empty image_formats both need the actual figures
    rendered; if save_pdf is False and image_formats is empty, only the
    numeric results are used and no matplotlib figure is ever built
    (saves the work for a text-only MD).
    Returns whichever of {"pdf": path, "md": path, "figures_dir": path}
    were actually written."""
    if not (save_pdf or save_md):
        raise ValueError("At least one of save_pdf or save_md must be True.")
    image_formats = tuple(image_formats)
    bad_formats = set(image_formats) - set(_VALID_IMAGE_FORMATS)
    if bad_formats:
        raise ValueError(f"Unsupported image format(s): {bad_formats}. Supported: {_VALID_IMAGE_FORMATS}")
    save_images = len(image_formats) > 0
    need_figures = save_pdf or save_images

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / f"{report_name}_figures" if save_images else None
    if fig_dir is not None:
        fig_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{report_name}.pdf" if save_pdf else None
    md_path = out_dir / f"{report_name}.md" if save_md else None

    md_lines = None
    if save_md:
        md_lines = [f"# Batch LCF report", "",
                   f"{len(results_by_target)} sample(s), sorted by **{_METRIC_LABELS.get(sort_by, sort_by)}**.", ""]

    pdf = PdfPages(pdf_path) if save_pdf else None
    try:
        for target_name, ranked_results in results_by_target.items():
            if not ranked_results:
                if save_md:
                    md_lines.append(f"## {target_name}\n\nNo LCF combinations produced (check reference selection).\n")
                continue
            target_energy, target_y = target_lookup[target_name]
            best = ranked_results[0]

            paths1: Dict[str, Path] = {}
            paths2: Dict[str, Path] = {}
            if need_figures:
                fig1 = build_fit_overlay_figure(best, target_energy, target_y, ref_lookup, x_label=x_label, y_label=y_label)
                if pdf is not None:
                    fig1.savefig(pdf, format="pdf")
                for fmt in image_formats:
                    p = fig_dir / f"{target_name}_fit.{fmt}"
                    fig1.savefig(p, dpi=200, bbox_inches="tight")
                    paths1[fmt] = p
                plt.close(fig1)

                fig2 = build_stats_page_figure(target_name, ranked_results, sort_by=sort_by, top_n=top_n)
                if pdf is not None:
                    fig2.savefig(pdf, format="pdf")
                for fmt in image_formats:
                    p = fig_dir / f"{target_name}_stats.{fmt}"
                    fig2.savefig(p, dpi=200, bbox_inches="tight")
                    paths2[fmt] = p
                plt.close(fig2)

            if save_md:
                md_lines.append(f"## {target_name}")
                md_lines.append("")
                md_lines.append(f"Best combination: **{' + '.join(best.ref_names)}** "
                                f"(R²={_fmt(best.r2)}, RMS={_fmt(best.rms, 5)}, reduced χ²={_fmt(best.reduced_chi_square, 5)})")
                md_lines.append("")
                # Markdown can only embed one image per figure; PNG is
                # preferred (universally rendered) when both were written.
                embed_fmt = "png" if "png" in paths1 else (next(iter(paths1), None))
                if embed_fmt is not None:
                    md_lines.append(f"![{target_name} fit]({fig_dir.name}/{paths1[embed_fmt].name})")
                    md_lines.append("")
                    md_lines.append(f"![{target_name} stats]({fig_dir.name}/{paths2[embed_fmt].name})")
                    md_lines.append("")
                md_lines.append(f"| rank | combination (name:weight) | R² | RMS | reduced χ² |")
                md_lines.append("|---|---|---|---|---|")
                for i, r in enumerate(ranked_results[:top_n], start=1):
                    composition = ", ".join(f"{n}:{w:.3f}" for n, w in zip(r.ref_names, r.weights))
                    md_lines.append(f"| {i} | {composition} | {_fmt(r.r2)} | {_fmt(r.rms, 5)} | {_fmt(r.reduced_chi_square, 5)} |")
                md_lines.append("")
    finally:
        if pdf is not None:
            pdf.close()

    out: Dict[str, Path] = {}
    if save_md:
        md_path.write_text("\n".join(md_lines), encoding="utf-8")
        out["md"] = md_path
    if save_pdf:
        out["pdf"] = pdf_path
    if save_images:
        out["figures_dir"] = fig_dir
    return out
