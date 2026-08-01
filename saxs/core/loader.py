"""Unified ASCII loader for 1D SAXS/WAXS curves.

Merges the two loaders that evolved separately in the historical tools:
- the reduction app's parser (broad comment prefixes, dominant-column-count
  filtering, transmission/thickness extraction from Xenocs-style headers);
- the plot tool's named-column detection (q / corrected / sigma_corrected ...).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .curve import Curve

COMMENT_PREFIXES = ("#", ";", "//", "%", "!", "'")
DELIM_RE = re.compile(r"[\s,;\t]+")

TRANSMISSION_PATTERNS = [
    re.compile(r"\bTransmittedFlux\b\s*[:=]?\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", re.IGNORECASE),
    re.compile(r"\btrans(?:mission)?\b\s*[:=]?\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", re.IGNORECASE),
    re.compile(r"\btr\b\s*[:=]?\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", re.IGNORECASE),
]
THICKNESS_PATTERNS = [
    re.compile(r"\bthickness\b\s*[:=]?\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*(mm|cm|um|µm)?", re.IGNORECASE),
    re.compile(r"\bpath\s*length\b\s*[:=]?\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*(mm|cm|um|µm)?", re.IGNORECASE),
]
IMPORTANT_HEADER_KEYS = [
    "Date", "ExposureTime", "SampleDistance", "Intensity1", "SampleEnvKind",
    "SampleEnvPN", "WaveLength", "DetectorModel", "FlatField", "TransmittedFlux",
]


def _convert_length_to_mm(value: float, unit: Optional[str]) -> float:
    unit = (unit or "mm").lower()
    if unit == "cm":
        return value * 10.0
    if unit in ("um", "µm"):
        return value / 1000.0
    return value


def parse_header_key_values(header_lines: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw in header_lines:
        line = raw.strip()
        for prefix in COMMENT_PREFIXES:
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
                break
        m = re.match(r"([A-Za-z][A-Za-z0-9_\-]*)\s*(?:[:=]|\s)\s*(.+)", line)
        if not m:
            continue
        key, value = m.group(1).strip(), m.group(2).strip()
        if key and value and key not in out:
            out[key] = value
    return out


def extract_metadata_from_header(header_lines: List[str]) -> Dict[str, object]:
    metadata: Dict[str, object] = {}
    text = "\n".join(header_lines)
    header_map = parse_header_key_values(header_lines)
    metadata["header_fields"] = header_map

    transmission = None
    for pat in TRANSMISSION_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                transmission = float(m.group(1))
            except Exception:
                continue
            break
    if transmission is not None:
        metadata["transmission"] = transmission

    thickness_mm = None
    for pat in THICKNESS_PATTERNS:
        m = pat.search(text)
        if m:
            thickness_mm = _convert_length_to_mm(float(m.group(1)), m.group(2))
            break
    if thickness_mm is not None:
        metadata["thickness_mm"] = thickness_mm

    for key in IMPORTANT_HEADER_KEYS:
        if key in header_map:
            metadata[key] = header_map[key]
    return metadata


def normalize_colname(name: str) -> str:
    return (
        name.strip().lower()
        .replace("(", "").replace(")", "")
        .replace("[", "").replace("]", "")
        .replace("/", "_").replace("-", "_")
    )


def choose_q_i_err_columns(header_tokens: List[str], ncols: int) -> Tuple[int, int, Optional[int]]:
    """Choose q / intensity / error column indices from a named header line."""
    if not header_tokens or len(header_tokens) != ncols:
        if ncols >= 3:
            return 0, 1, 2
        return 0, 1, None

    names = [normalize_colname(x) for x in header_tokens]
    q_candidates = ["q", "q_a^1", "q_a^-1", "q_inv_a", "q_a_1", "q_a__1", "2theta", "two_theta", "tth"]
    i_priority = ["corrected", "iq", "i_q", "intensity", "sample_aligned", "sample", "counts"]
    err_priority = ["sigma_corrected", "error", "sigma", "err", "uncertainty", "di", "dy"]

    q_idx = 0
    for cand in q_candidates:
        if cand in names:
            q_idx = names.index(cand)
            break
    else:
        for j, nm in enumerate(names):
            if nm.startswith("q"):
                q_idx = j
                break

    i_idx = None
    for cand in i_priority:
        if cand in names:
            i_idx = names.index(cand)
            break
    if i_idx is None or i_idx == q_idx:
        i_idx = 1 if ncols >= 2 else 0

    err_idx = None
    for cand in err_priority:
        if cand in names:
            err_idx = names.index(cand)
            break
    if err_idx is None and ncols == 3:
        err_idx = 2
    return q_idx, i_idx, err_idx


def _parse_numeric_block(path: str) -> Tuple[List[str], np.ndarray]:
    header_lines: List[str] = []
    rows: List[List[float]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(COMMENT_PREFIXES):
                header_lines.append(stripped)
                continue
            tokens = [tok for tok in DELIM_RE.split(stripped) if tok]
            numeric: List[float] = []
            bad = False
            for tok in tokens:
                try:
                    numeric.append(float(tok))
                except ValueError:
                    bad = True
                    break
            if bad:
                header_lines.append(stripped)
                continue
            rows.append(numeric)

    if not rows:
        raise ValueError(f"No numeric rows found in {path}")

    counts: Dict[int, int] = {}
    for row in rows:
        counts[len(row)] = counts.get(len(row), 0) + 1
    n_columns = max(counts.items(), key=lambda item: item[1])[0]
    if n_columns < 2:
        raise ValueError(f"Need at least 2 numeric columns in {path}")
    filtered = [row[:n_columns] for row in rows if len(row) >= n_columns]
    return header_lines, np.asarray(filtered, dtype=float)


def _last_header_tokens(header_lines: List[str], ncols: int) -> List[str]:
    """Tokens of the last commented header line — usually the column names."""
    for raw in reversed(header_lines):
        s = raw.strip()
        for prefix in COMMENT_PREFIXES:
            if s.startswith(prefix):
                s = s[len(prefix):].strip()
                break
        if not s:
            continue
        parts = [p for p in DELIM_RE.split(s) if p]
        if len(parts) == ncols:
            return parts
    return []


def load_curve(path: str, file_role: str = "sample") -> Curve:
    """Load one 1D curve from an ASCII file into a Curve."""
    header_lines, arr = _parse_numeric_block(path)
    ncols = arr.shape[1]
    tokens = _last_header_tokens(header_lines, ncols)
    q_idx, i_idx, err_idx = choose_q_i_err_columns(tokens, ncols)

    q = arr[:, q_idx]
    intensity = arr[:, i_idx]
    sigma = arr[:, err_idx] if err_idx is not None and err_idx < ncols else None

    order = np.argsort(q)
    q, intensity = q[order], intensity[order]
    if sigma is not None:
        sigma = sigma[order]

    mask = np.isfinite(q) & np.isfinite(intensity)
    if sigma is not None:
        mask &= np.isfinite(sigma)
    q, intensity = q[mask], intensity[mask]
    if sigma is not None:
        sigma = sigma[mask]

    uniq = np.concatenate(([True], np.diff(q) > 0))
    q, intensity = q[uniq], intensity[uniq]
    if sigma is not None:
        sigma = sigma[uniq]

    metadata = extract_metadata_from_header(header_lines)
    curve = Curve(
        q=q,
        intensity=intensity,
        sigma=sigma,
        name=Path(path).stem,
        path=str(Path(path).resolve()),
        header_lines=header_lines,
        metadata=metadata,
        transmission=metadata.get("transmission"),
        thickness_mm=metadata.get("thickness_mm"),
        file_role=file_role,
    )
    curve.record("load", path=str(path), columns={"q": q_idx, "I": i_idx, "sigma": err_idx})
    return curve


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


def export_curve(path: str, curve: Curve, fmt: str = "dat", metadata: Optional[Dict[str, object]] = None) -> None:
    """Write a single curve as q / I / sigma with commented metadata."""
    df = pd.DataFrame({"q_A^-1": curve.q, "intensity": curve.intensity})
    if curve.sigma is not None:
        df["sigma"] = curve.sigma
    meta = dict(metadata or {})
    meta.setdefault("name", curve.name)
    fmt = fmt.lower()
    if fmt == "csv":
        with open(path, "w", encoding="utf-8") as fh:
            for key, value in meta.items():
                fh.write(f"# {key}: {value}\n")
            df.to_csv(fh, index=False, lineterminator="\n")
        return
    if fmt == "xlsx":
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="curve")
            meta_df = pd.DataFrame({"key": list(meta.keys()), "value": [str(v) for v in meta.values()]})
            meta_df.to_excel(writer, index=False, sheet_name="metadata")
        return
    with open(path, "w", encoding="utf-8") as fh:
        for key, value in meta.items():
            fh.write(f"# {key}: {value}\n")
        fh.write("# " + "\t".join(df.columns) + "\n")
        for _, row in df.iterrows():
            fh.write("\t".join(f"{val:.10g}" for val in row.values) + "\n")


def export_correction_table(
    path: str,
    q: np.ndarray,
    sample: np.ndarray,
    empty_scaled: np.ndarray,
    corrected: np.ndarray,
    sigma: Optional[np.ndarray] = None,
    metadata: Optional[Dict[str, object]] = None,
    fmt: str = "dat",
) -> None:
    df = pd.DataFrame({
        "q_A^-1": q,
        "sample_aligned": sample,
        "empty_scaled": empty_scaled,
        "corrected": corrected,
    })
    if sigma is not None:
        df["sigma_corrected"] = sigma
    fmt = fmt.lower()
    meta = metadata or {}
    if fmt == "csv":
        with open(path, "w", encoding="utf-8") as fh:
            for key, value in meta.items():
                fh.write(f"# {key}: {value}\n")
            df.to_csv(fh, index=False, lineterminator="\n")
        return
    if fmt == "xlsx":
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="corrected_curve")
            if meta:
                meta_df = pd.DataFrame({"key": list(meta.keys()), "value": [str(v) for v in meta.values()]})
                meta_df.to_excel(writer, index=False, sheet_name="metadata")
        return
    with open(path, "w", encoding="utf-8") as fh:
        for key, value in meta.items():
            fh.write(f"# {key}: {value}\n")
        fh.write("# q_A^-1\tsample_aligned\tempty_scaled\tcorrected")
        if sigma is not None:
            fh.write("\tsigma_corrected")
        fh.write("\n")
        for idx in range(len(q)):
            row = [q[idx], sample[idx], empty_scaled[idx], corrected[idx]]
            if sigma is not None:
                row.append(sigma[idx])
            fh.write("\t".join(f"{val:.10g}" for val in row) + "\n")


def export_summary_table(path: str, rows: List[Dict[str, object]], fmt: str = "csv") -> None:
    df = pd.DataFrame(rows)
    fmt = fmt.lower()
    if fmt == "xlsx":
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="summary")
        return
    if fmt == "json":
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2, default=str)
        return
    df.to_csv(path, index=False)
