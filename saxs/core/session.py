"""Project session persistence (JSON): loaded curves + perspective settings."""

from __future__ import annotations

import json
from typing import Dict, List

from . import __version__
from .curve import Curve


def save_session(path: str, curves: List[Curve], settings: Dict[str, object]) -> None:
    payload = {
        "app": "pomme",
        "version": __version__,
        "curves": [c.to_json() for c in curves],
        "settings": settings,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)


def load_session(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    curves = [Curve.from_json(c) for c in payload.get("curves", [])]
    settings = payload.get("settings", {})
    return curves, settings
