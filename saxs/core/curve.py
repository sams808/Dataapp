"""Central 1D scattering-curve container shared by every perspective.

Field names deliberately match the historical reduction app's DataSet
(q, intensity, sigma, transmission, thickness_mm, file_role) so the ported
reduction physics runs unchanged.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Curve:
    q: np.ndarray
    intensity: np.ndarray
    sigma: Optional[np.ndarray] = None
    name: str = "curve"
    path: Optional[str] = None
    header_lines: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)
    transmission: Optional[float] = None
    thickness_mm: Optional[float] = None
    file_role: str = "sample"  # sample | empty | corrected | other
    provenance: List[Dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.q = np.asarray(self.q, dtype=float)
        self.intensity = np.asarray(self.intensity, dtype=float)
        if self.sigma is not None:
            self.sigma = np.asarray(self.sigma, dtype=float)

    @property
    def npts(self) -> int:
        return int(self.q.size)

    @property
    def qmin(self) -> float:
        return float(self.q.min()) if self.q.size else float("nan")

    @property
    def qmax(self) -> float:
        return float(self.q.max()) if self.q.size else float("nan")

    def record(self, step: str, **details: object) -> None:
        """Append a provenance entry describing a processing step."""
        entry: Dict[str, object] = {"step": step, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        entry.update(details)
        self.provenance.append(entry)

    def copy_with(
        self,
        q: Optional[np.ndarray] = None,
        intensity: Optional[np.ndarray] = None,
        sigma: Optional[np.ndarray] = "unset",
        name: Optional[str] = None,
        file_role: Optional[str] = None,
        step: Optional[str] = None,
        **step_details: object,
    ) -> "Curve":
        """Derive a new Curve, carrying metadata and provenance forward."""
        new_sigma = self.sigma if isinstance(sigma, str) and sigma == "unset" else sigma
        child = Curve(
            q=np.array(self.q if q is None else q, copy=True),
            intensity=np.array(self.intensity if intensity is None else intensity, copy=True),
            sigma=None if new_sigma is None else np.array(new_sigma, copy=True),
            name=name or self.name,
            path=self.path,
            header_lines=list(self.header_lines),
            metadata=dict(self.metadata),
            transmission=self.transmission,
            thickness_mm=self.thickness_mm,
            file_role=file_role or self.file_role,
            provenance=[dict(p) for p in self.provenance],
        )
        if step:
            child.record(step, **step_details)
        return child

    def cropped(self, qmin: float, qmax: float) -> "Curve":
        lo, hi = sorted((float(qmin), float(qmax)))
        mask = (self.q >= lo) & (self.q <= hi)
        return self.copy_with(
            q=self.q[mask],
            intensity=self.intensity[mask],
            sigma=None if self.sigma is None else self.sigma[mask],
            step="crop",
            qmin=lo,
            qmax=hi,
        )

    def positive_mask(self) -> np.ndarray:
        """Mask of points usable on log axes (finite, q>0, I>0)."""
        mask = np.isfinite(self.q) & np.isfinite(self.intensity) & (self.q > 0) & (self.intensity > 0)
        return mask

    def to_json(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "path": self.path,
            "q": self.q.tolist(),
            "intensity": self.intensity.tolist(),
            "sigma": None if self.sigma is None else self.sigma.tolist(),
            "header_lines": list(self.header_lines),
            "metadata": _jsonable(self.metadata),
            "transmission": self.transmission,
            "thickness_mm": self.thickness_mm,
            "file_role": self.file_role,
            "provenance": self.provenance,
        }

    @classmethod
    def from_json(cls, payload: Dict[str, object]) -> "Curve":
        sigma = payload.get("sigma")
        return cls(
            q=np.asarray(payload["q"], dtype=float),
            intensity=np.asarray(payload["intensity"], dtype=float),
            sigma=None if sigma is None else np.asarray(sigma, dtype=float),
            name=str(payload.get("name", "curve")),
            path=payload.get("path"),
            header_lines=list(payload.get("header_lines", [])),
            metadata=dict(payload.get("metadata", {})),
            transmission=payload.get("transmission"),
            thickness_mm=payload.get("thickness_mm"),
            file_role=str(payload.get("file_role", "sample")),
            provenance=list(payload.get("provenance", [])),
        )

    def display_label(self) -> str:
        stem = Path(self.path).stem if self.path else self.name
        return self.name or stem


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj
