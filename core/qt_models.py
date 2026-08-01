"""
qt_models.py — the Spectrum data model with stable identity, replacing
main.py's four-parallel-list pattern (file_paths/file_titles/file_statuses/
xy_by_path kept in sync by hand at every call site).

Every imported item gets a UUID at import time. All cross-references
(selection, per-item settings, fit-parameter memory) key off that id, never
off display title or list position — main.py's reorder handler does
`self.file_paths[self.file_titles.index(title)]`, which silently picks the
wrong file if two imports share a title. That bug class is structurally
impossible here because nothing ever looks anything up by title.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class Spectrum:
    id: str
    title: str
    path: str
    kind: str  # "raman_xy" | "xrd_xy" | "saxs_edf_ascii" | "dta" | "ta_sdt" | "xas" | "generic_xy"
    x: np.ndarray
    y: np.ndarray
    df: Optional[pd.DataFrame] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    status: str = "imported"  # "imported" | "derived" (sum/baseline-subtracted/fit output, etc.)

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex


class SpectrumLibrary:
    """Ordered collection of Spectrum objects, addressed by stable id."""

    def __init__(self) -> None:
        self._order: List[str] = []
        self._items: Dict[str, Spectrum] = {}

    def add(self, spectrum: Spectrum) -> None:
        if spectrum.id in self._items:
            raise ValueError(f"Spectrum id already present: {spectrum.id}")
        self._items[spectrum.id] = spectrum
        self._order.append(spectrum.id)

    def remove(self, item_id: str) -> None:
        self._items.pop(item_id, None)
        self._order = [i for i in self._order if i != item_id]

    def get(self, item_id: str) -> Optional[Spectrum]:
        return self._items.get(item_id)

    def all(self) -> List[Spectrum]:
        return [self._items[i] for i in self._order if i in self._items]

    def by_kind(self, kinds) -> List[Spectrum]:
        kinds = set(kinds) if not isinstance(kinds, str) else {kinds}
        return [s for s in self.all() if s.kind in kinds]

    def reorder(self, new_order: List[str]) -> None:
        if set(new_order) != set(self._order):
            raise ValueError("reorder() must be given a permutation of the current ids.")
        self._order = list(new_order)

    def clear(self) -> None:
        self._items.clear()
        self._order.clear()

    def __len__(self) -> int:
        return len(self._order)

    def __iter__(self):
        return iter(self.all())
