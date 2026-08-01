"""
project_io.py — .prism project file save/load (M14), framework-agnostic.

The starkest gap versus every reference tool (Origin .opju, Spectragryph
sessions): until now, closing the app lost every imported spectrum, fit
parameter set, and accepted RRUFF identification. A .prism file is a ZIP:

    manifest.json     {"format": "prism-project", "version": 2, ...}
    spectra.json      ordered list of Library Spectrum records (id/title/
                      path/kind/meta/status) — everything except the arrays
    data.npz          all arrays: library "{id}_x"/"{id}_y", XAS
                      "xas_{sid}_energy"/"xas_{sid}_y"(/"xas_{sid}_angle"),
                      HT-XRD "ht_{i}_x"/"ht_{i}_y"
    fit_params.json   {spectrum_id: params_struct} from the shared
                      PerItemSettingsStore (Peak Fitting / Multi-Fit)
    df/{id}.csv       full imported DataFrame per spectrum, when present
                      (needed by e.g. the DTA workspace's column pickers)
    xas_spectra.json  XAS workspace SpectrumStore records incl. e0/label/
                      units/parents and the full Operation history  (v2)
    htxrd.json        HT-XRD series records (name/ramp value/source/path) (v2)

Version 2 added the XAS and HT-XRD sections; v1 files load fine (the new
sections are simply absent -> empty). Arrays are stored in the project
itself rather than re-read from the original source paths on load — source
files move, network shares disappear; a project must stand alone. The
original path is still kept as provenance metadata.
"""
from __future__ import annotations

import io
import json
import time
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.qt_models import Spectrum

PROJECT_FORMAT = "prism-project"
LEGACY_FORMATS = {"dataapp-project"}  # pre-rename projects still load
PROJECT_VERSION = 3  # v3 added cif_overlays.json + baseline_settings.json


@dataclass
class ProjectData:
    spectra: List[Spectrum] = field(default_factory=list)
    fit_params: Dict[str, list] = field(default_factory=dict)
    xas_spectra: list = field(default_factory=list)      # xas_science.Spectrum
    htxrd_patterns: list = field(default_factory=list)   # htxrd_science.HtxrdPattern
    cif_overlays: list = field(default_factory=list)     # Simple Plot CIF series (paths + display fields; peaks recomputed on load)
    baseline_settings: Dict[str, dict] = field(default_factory=dict)  # per-spectrum Baseline workspace settings


def _json_safe(obj: Any) -> Any:
    """Recursively convert meta dicts to JSON-safe structures (numpy
    scalars/arrays and other exotic values appear in parser metadata)."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def save_project(
    path: str, spectra: List[Spectrum], fit_params: Dict[str, list],
    *, xas_spectra: Optional[list] = None, htxrd_patterns: Optional[list] = None,
    cif_overlays: Optional[list] = None, baseline_settings: Optional[Dict[str, dict]] = None,
) -> None:
    records = []
    arrays: Dict[str, np.ndarray] = {}
    dfs: Dict[str, pd.DataFrame] = {}

    for sp in spectra:
        records.append({
            "id": sp.id,
            "title": sp.title,
            "path": sp.path,
            "kind": sp.kind,
            "meta": _json_safe(sp.meta),
            "status": sp.status,
        })
        arrays[f"{sp.id}_x"] = np.asarray(sp.x, dtype=float)
        arrays[f"{sp.id}_y"] = np.asarray(sp.y, dtype=float)
        if sp.df is not None:
            dfs[sp.id] = sp.df

    xas_records = []
    for sp in (xas_spectra or []):
        xas_records.append({
            "sid": sp.sid,
            "name": sp.name,
            "kind": sp.kind,
            "units": sp.units,
            "label": sp.label,
            "e0": None if sp.e0 is None or not np.isfinite(sp.e0) else float(sp.e0),
            "meta": _json_safe(sp.meta),
            "parents": list(sp.parents),
            "history": [{"name": op.name, "params": _json_safe(op.params), "when": op.when} for op in sp.history],
            "has_angle": sp.angle is not None,
        })
        arrays[f"xas_{sp.sid}_energy"] = np.asarray(sp.energy, dtype=float)
        arrays[f"xas_{sp.sid}_y"] = np.asarray(sp.y, dtype=float)
        if sp.angle is not None:
            arrays[f"xas_{sp.sid}_angle"] = np.asarray(sp.angle, dtype=float)

    ht_records = []
    for i, pat in enumerate(htxrd_patterns or []):
        ht_records.append({
            "index": i,
            "path": pat.path,
            "name": pat.name,
            "ramp_value": pat.ramp_value,
            "ramp_source": pat.ramp_source,
            "meta": _json_safe(pat.meta),
        })
        arrays[f"ht_{i}_x"] = np.asarray(pat.x, dtype=float)
        arrays[f"ht_{i}_y"] = np.asarray(pat.y, dtype=float)

    npz_buf = io.BytesIO()
    np.savez_compressed(npz_buf, **arrays)

    manifest = {
        "format": PROJECT_FORMAT,
        "version": PROJECT_VERSION,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_spectra": len(records),
        "n_xas_spectra": len(xas_records),
        "n_htxrd_patterns": len(ht_records),
    }

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        zf.writestr("spectra.json", json.dumps(records, indent=2))
        zf.writestr("data.npz", npz_buf.getvalue())
        zf.writestr("fit_params.json", json.dumps(_json_safe(fit_params), indent=2))
        if xas_records:
            zf.writestr("xas_spectra.json", json.dumps(xas_records, indent=2))
        if ht_records:
            zf.writestr("htxrd.json", json.dumps(ht_records, indent=2))
        if cif_overlays:
            zf.writestr("cif_overlays.json", json.dumps(_json_safe(cif_overlays), indent=2))
        if baseline_settings:
            zf.writestr("baseline_settings.json", json.dumps(_json_safe(baseline_settings), indent=2))
        for sid, df in dfs.items():
            zf.writestr(f"df/{sid}.csv", df.to_csv(index=False))


def load_project(path: str) -> ProjectData:
    with zipfile.ZipFile(path, "r") as zf:
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        if manifest.get("format") != PROJECT_FORMAT and manifest.get("format") not in LEGACY_FORMATS:
            raise ValueError(f"Not a PRISM project file: {path}")
        if int(manifest.get("version", 0)) > PROJECT_VERSION:
            raise ValueError(
                f"Project was saved by a newer PRISM (project version "
                f"{manifest.get('version')}, this build reads up to {PROJECT_VERSION})."
            )

        names = set(zf.namelist())
        records = json.loads(zf.read("spectra.json").decode("utf-8"))
        with np.load(io.BytesIO(zf.read("data.npz"))) as data:
            arrays = {k: data[k] for k in data.files}
        fit_params = json.loads(zf.read("fit_params.json").decode("utf-8"))

        df_names = {n for n in names if n.startswith("df/") and n.endswith(".csv")}

        spectra: List[Spectrum] = []
        for rec in records:
            sid = rec["id"]
            df = None
            df_name = f"df/{sid}.csv"
            if df_name in df_names:
                try:
                    df = pd.read_csv(io.BytesIO(zf.read(df_name)))
                except Exception:
                    df = None
            spectra.append(Spectrum(
                id=sid,
                title=rec.get("title", sid),
                path=rec.get("path", ""),
                kind=rec.get("kind", "generic_xy"),
                x=arrays.get(f"{sid}_x", np.array([])),
                y=arrays.get(f"{sid}_y", np.array([])),
                df=df,
                meta=rec.get("meta", {}) or {},
                status=rec.get("status", "imported"),
            ))

        xas_spectra = []
        if "xas_spectra.json" in names:
            from xas_science import Operation as XasOperation, Spectrum as XasSpectrum
            for rec in json.loads(zf.read("xas_spectra.json").decode("utf-8")):
                sid = rec["sid"]
                xas_spectra.append(XasSpectrum(
                    sid=sid,
                    name=rec.get("name", sid),
                    kind=rec.get("kind", "mu"),
                    energy=arrays.get(f"xas_{sid}_energy", np.array([])),
                    y=arrays.get(f"xas_{sid}_y", np.array([])),
                    angle=arrays.get(f"xas_{sid}_angle") if rec.get("has_angle") else None,
                    units=rec.get("units", "a.u."),
                    label=rec.get("label", "XAS(Unknown)"),
                    e0=rec.get("e0"),
                    meta=rec.get("meta", {}) or {},
                    parents=list(rec.get("parents", [])),
                    history=[XasOperation(name=op.get("name", "?"), params=op.get("params", {}) or {}, when=op.get("when", 0.0))
                             for op in rec.get("history", [])],
                ))

        htxrd_patterns = []
        if "htxrd.json" in names:
            from xrd.htxrd_science import HtxrdPattern
            for rec in json.loads(zf.read("htxrd.json").decode("utf-8")):
                i = rec["index"]
                htxrd_patterns.append(HtxrdPattern(
                    path=rec.get("path", ""),
                    name=rec.get("name", f"pattern_{i}"),
                    x=arrays.get(f"ht_{i}_x", np.array([])),
                    y=arrays.get(f"ht_{i}_y", np.array([])),
                    ramp_value=rec.get("ramp_value"),
                    ramp_source=rec.get("ramp_source", "none"),
                    meta=rec.get("meta", {}) or {},
                ))

        cif_overlays = []
        if "cif_overlays.json" in names:
            cif_overlays = json.loads(zf.read("cif_overlays.json").decode("utf-8"))
        baseline_settings = {}
        if "baseline_settings.json" in names:
            baseline_settings = json.loads(zf.read("baseline_settings.json").decode("utf-8"))

    return ProjectData(spectra=spectra, fit_params=fit_params,
                       xas_spectra=xas_spectra, htxrd_patterns=htxrd_patterns,
                       cif_overlays=cif_overlays, baseline_settings=baseline_settings)
