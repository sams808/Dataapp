"""Batch reduction pipeline: one row per sample, organized outputs.

Mirrors the historical batch runner's outputs:
  corrected_curves/*.dat, plots/*.png, batch_summary.{csv,xlsx,json}
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List

from .chemistry import CapillaryConfig, SamplePhysicsConfig
from .loader import export_correction_table, export_summary_table, load_curve
from .reduction import CorrectionSettings, correct_sample


@dataclass
class BatchRow:
    sample_path: str
    empty_path: str
    label: str = ""
    sample_cfg: SamplePhysicsConfig = field(default_factory=SamplePhysicsConfig)
    capillary_cfg: CapillaryConfig = field(default_factory=CapillaryConfig)
    settings: CorrectionSettings = field(default_factory=CorrectionSettings)

    def to_json(self) -> Dict[str, object]:
        return {
            "sample_path": self.sample_path,
            "empty_path": self.empty_path,
            "label": self.label,
            "sample_cfg": self.sample_cfg.to_json(),
            "capillary_cfg": asdict(self.capillary_cfg),
            "settings": self.settings.to_json(),
        }

    @classmethod
    def from_json(cls, payload: Dict[str, object]) -> "BatchRow":
        return cls(
            sample_path=str(payload["sample_path"]),
            empty_path=str(payload["empty_path"]),
            label=str(payload.get("label", "")),
            sample_cfg=SamplePhysicsConfig(**_only_fields(SamplePhysicsConfig, payload.get("sample_cfg", {}))),
            capillary_cfg=CapillaryConfig(**_only_fields(CapillaryConfig, payload.get("capillary_cfg", {}))),
            settings=CorrectionSettings(**_only_fields(CorrectionSettings, payload.get("settings", {}))),
        )


def _only_fields(cls, payload: Dict[str, object]) -> Dict[str, object]:
    names = set(getattr(cls, "__dataclass_fields__", {}))
    return {k: v for k, v in dict(payload).items() if k in names}


def run_batch(
    rows: List[BatchRow],
    output_dir: str,
    energy_ev: float,
    make_plots: bool = True,
    progress=None,
) -> List[Dict[str, object]]:
    """Run every row; returns summary dicts (also written to disk)."""
    out = Path(output_dir)
    curves_dir = out / "corrected_curves"
    plots_dir = out / "plots"
    curves_dir.mkdir(parents=True, exist_ok=True)
    if make_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []
    for i, row in enumerate(rows):
        label = row.label or Path(row.sample_path).stem
        entry: Dict[str, object] = {"label": label, "sample": row.sample_path, "empty": row.empty_path}
        try:
            sample = load_curve(row.sample_path, file_role="sample")
            empty = load_curve(row.empty_path, file_role="empty")
            result = correct_sample(sample, empty, row.sample_cfg, row.capillary_cfg, row.settings, energy_ev)
            entry.update(result.summary())
            entry["status"] = "ok"

            curve_path = curves_dir / f"{label}_corrected.dat"
            export_correction_table(
                str(curve_path), result.q, result.sample_aligned, result.empty_scaled,
                result.corrected, result.sigma_corrected,
                metadata={"label": label, **{k: v for k, v in result.summary().items()}},
            )
            entry["corrected_file"] = str(curve_path)

            if make_plots:
                _save_plot(plots_dir / f"{label}.png", result, label)
                entry["plot_file"] = str(plots_dir / f"{label}.png")
        except Exception as exc:
            entry["status"] = f"error: {exc}"
        summary_rows.append(entry)
        if progress is not None:
            progress(i + 1, len(rows), label)

    export_summary_table(str(out / "batch_summary.csv"), summary_rows, fmt="csv")
    export_summary_table(str(out / "batch_summary.xlsx"), summary_rows, fmt="xlsx")
    export_summary_table(str(out / "batch_summary.json"), summary_rows, fmt="json")
    return summary_rows


def _save_plot(path: Path, result, label: str) -> None:
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=130)
    ax.loglog(result.q, result.sample_aligned, label="sample", lw=1.0)
    ax.loglog(result.q, result.empty_scaled, label=f"empty x {result.scale_factor:.4g}", lw=1.0)
    mask = result.corrected > 0
    ax.loglog(result.q[mask], result.corrected[mask], label="corrected", lw=1.4)
    ax.set_xlabel(r"q [$\AA^{-1}$]")
    ax.set_ylabel("I(q) [a.u.]")
    ax.set_title(label)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
