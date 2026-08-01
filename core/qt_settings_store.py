"""
qt_settings_store.py — the generic "settings that persist per item" store.

ui_fit_params.py's fit_param_memory dict (keyed by identity, snapshot/restore
around use) already gets this right today. BaselineParamWindow's dual-dict
bug (self.state/self.memory vs self.spec_states, main.py) and Simple Plot's
multi-select DTA state-bleed bug both happened because that correct pattern
was re-solved ad hoc per window instead of shared. Every Qt tool window that
needs per-spectrum settings (baseline ROIs, fit params, DTA plot options)
should use ONE of these, keyed by Spectrum.id — never by title or position.
"""
from __future__ import annotations

from typing import Callable, Dict, Generic, Iterator, Tuple, TypeVar

T = TypeVar("T")


class PerItemSettingsStore(Generic[T]):
    def __init__(self, default_factory: Callable[[], T]):
        self._default_factory = default_factory
        self._store: Dict[str, T] = {}

    def get(self, item_id: str) -> T:
        if item_id not in self._store:
            self._store[item_id] = self._default_factory()
        return self._store[item_id]

    def set(self, item_id: str, value: T) -> None:
        self._store[item_id] = value

    def has(self, item_id: str) -> bool:
        return item_id in self._store

    def discard(self, item_id: str) -> None:
        self._store.pop(item_id, None)

    def clear(self) -> None:
        self._store.clear()

    def items(self) -> Iterator[Tuple[str, T]]:
        return iter(self._store.items())
