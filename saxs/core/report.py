"""Collects analysis results across perspectives and exports CSV/JSON/HTML."""

from __future__ import annotations

import dataclasses
import json
import time
from html import escape
from typing import Dict, List

import numpy as np


class ReportStore:
    """Flat list of result entries; each entry is one analysis on one curve."""

    def __init__(self) -> None:
        self.entries: List[Dict[str, object]] = []

    def add(self, curve_name: str, kind: str, result: object, notes: str = "") -> Dict[str, object]:
        payload = _result_to_dict(result)
        entry = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "curve": curve_name,
            "analysis": kind,
            "notes": notes,
            **payload,
        }
        self.entries.append(entry)
        return entry

    def clear(self) -> None:
        self.entries = []

    def to_csv(self, path: str) -> None:
        import pandas as pd
        pd.DataFrame(self.entries).to_csv(path, index=False)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.entries, fh, indent=2, default=_json_default)

    def to_html(self, path: str, title: str = "Analysis report") -> None:
        rows = []
        for entry in self.entries:
            cells = "".join(
                f"<tr><th>{escape(str(k))}</th><td>{escape(_fmt(v))}</td></tr>"
                for k, v in entry.items()
            )
            rows.append(f'<table class="entry">{cells}</table>')
        body = "\n".join(rows) if rows else "<p>No entries.</p>"
        html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{escape(title)}</title>
<style>
body {{ font-family: Segoe UI, Arial, sans-serif; margin: 2em; color: #222; }}
h1 {{ font-size: 1.4em; }}
table.entry {{ border-collapse: collapse; margin-bottom: 1.5em; }}
table.entry th {{ text-align: left; padding: 2px 10px 2px 0; color: #555; font-weight: 600; }}
table.entry td {{ padding: 2px 6px; }}
table.entry tr:nth-child(odd) {{ background: #f5f5f5; }}
</style></head><body>
<h1>{escape(title)}</h1>
<p>Generated {time.strftime("%Y-%m-%d %H:%M:%S")} - {len(self.entries)} entr{'y' if len(self.entries) == 1 else 'ies'}</p>
{body}
</body></html>"""
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)


def _result_to_dict(result: object) -> Dict[str, object]:
    if isinstance(result, dict):
        return {k: _plain(v) for k, v in result.items()}
    if dataclasses.is_dataclass(result) and not isinstance(result, type):
        return {k: _plain(v) for k, v in dataclasses.asdict(result).items()}
    return {"value": _plain(result)}


def _plain(value):
    if isinstance(value, np.ndarray):
        if value.size > 12:
            return f"array[{value.size}]"
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (list, tuple)) and len(value) <= 12:
        return [_plain(v) for v in value]
    return value


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return str(obj)
