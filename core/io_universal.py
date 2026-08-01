# io_universal_v40.py
# --------------------------------------------------------------------------------------
# Universal loader (V40): robust sniffers + parsers + registry
# Supports:
#   1) TA SDT Q600 ASCII (DSC-TGA) with SigN + StartOfData
#   2) SAXS EDF ASCII exports (header with '#', then 'q(A-1)  I(q)  Sig(q)')
#   3) Raman XY (Raman shift vs intensity; common ASCII exports)
#   4) XRD XY   (2theta vs intensity; CSV/TSV variants)
#   5) Richer DTA/STA text exports (semicolon/comma/tab; decimal comma aware)
#   6) Generic XY text (spaces/commas/tabs/semicolons; decimal comma; Fortran 'D')
#
# Public API:
#   - load_any(path, *, x_key=None, y_key=None, prefer=None, return_meta=False)
#   - import_xy(path, *, x_key=None, y_key=None, deduplicate=None, ...)
#   - list_columns(path)
#   - register_parser(ParserSpec)
#
# Canonical keys exposed in meta["canonical_map"] for robust references:
#   TA SDT: 't_min', 'T_C', 'm_mg', 'HF_mW', 'dT_C', 'dT_uV', 'flow_mL_min'
#   SAXS  : 'q_A^-1', 'I', 'sigma'
#   Raman : 'shift_cm^-1', 'intensity'
#   XRD   : '2theta_deg', 'intensity'
#   DTA   : 'T_C', 'time_min', 'HF_mW', 'DSC_mW_mg', 'TG_pct', 'DTG_pct_min'
#
from __future__ import annotations

import io
import os
import re
import typing as _t
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

# ==============================
# Low-level numeric utilities
# ==============================
_DEC_COMMA = re.compile(r"(?<=\d),(?=\d)")
_FORTRAN_D = re.compile(r"([+-]?\d+(?:\.\d+)?)[dD]([+-]?\d+)")
_MULTI_SEP  = re.compile(r"[,\t; ]+")
_NUM_LIKE   = re.compile(r"""^[\s]*
    [+-]?                 # sign
    (?:\d+\.?\d*|\.\d+)   # '12', '12.3', '.004'
    (?:[eEdD][+-]?\d+)?   # exponent
    [\s]*$""", re.X)

def _decode_text_autodetect(raw: bytes, extra_encodings: tuple[str, ...] = ()) -> tuple[str, str]:
    """
    Decode bytes with TA/Netzsch-friendly heuristics (mirrors EXAMPLES/plot_dta.py):
    - Prefer UTF-16 when BOM or many NUL bytes are present.
    - Fall back to UTF-8 variants, then latin-1, then UTF-8(ignore).
    """
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace"), "utf-16"

    head = raw[:4000]
    if head.count(b"\x00") > len(head) * 0.1:
        try:
            return raw.decode("utf-16", errors="replace"), "utf-16"
        except Exception:
            pass

    # latin-1 never raises (it maps every byte 0x00-0xFF), so it must stay
    # last or any caller-supplied extra_encodings would be unreachable.
    for enc in ("utf-8-sig", "utf-8", *extra_encodings, "latin-1"):
        try:
            return raw.decode(enc, errors="strict"), enc
        except Exception:
            continue

    return raw.decode("utf-8", errors="ignore"), "binary->utf-8(ignore)"


def _read_text_with_fallbacks(path: str, encodings=("utf-8-sig","utf-8","latin-1")) -> tuple[str,str]:
    raw = Path(path).read_bytes()
    return _decode_text_autodetect(raw, extra_encodings=tuple(encodings))


def _normalize_num_token(tok: str) -> str | None:
    s = tok.strip()
    if not _NUM_LIKE.match(s):
        return None
    s = _DEC_COMMA.sub(".", s)                               # 12,34 -> 12.34
    s = _FORTRAN_D.sub(lambda m: f"{m.group(1)}E{m.group(2)}", s)  # 1.2D+3 -> 1.2E+3
    if s.startswith(".") and (len(s) == 1 or s[1].isdigit()):      # .004 -> 0.004
        s = "0" + s
    return s

def _split_numeric_line(line: str) -> list[str] | None:
    parts = _MULTI_SEP.split(line.strip())
    if len(parts) < 2:
        return None
    t0 = _normalize_num_token(parts[0])
    t1 = _normalize_num_token(parts[1])
    if t0 is None or t1 is None:
        return None
    parts[0], parts[1] = t0, t1
    return parts

def _coerce_2cols_to_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if df.shape[1] < 2:
        raise ValueError("Found data, but fewer than two usable columns.")
    x = df.iloc[:,0].astype(float).to_numpy()
    y = df.iloc[:,1].astype(float).to_numpy()
    if x.size != y.size or x.size < 2:
        raise ValueError("X or Y length is invalid.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("NaN/Inf found in X or Y.")
    return x, y

def _stable_sort_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray,np.ndarray,bool]:
    sorted_now = bool(np.all(np.diff(x) > 0))
    if sorted_now:
        return x, y, True
    order = np.argsort(x, kind="mergesort")
    return x[order], y[order], False

def _deduplicate_xy(x: np.ndarray, y: np.ndarray, how: str, dec: int) -> tuple[np.ndarray, np.ndarray]:
    key = np.round(x, dec)
    uniq, inv = np.unique(key, return_inverse=True)

    if how == "first":
        keep_idx = np.zeros_like(uniq, dtype=int)
        seen = np.zeros_like(uniq, dtype=bool)
        for i,g in enumerate(inv):
            if not seen[g]:
                keep_idx[g] = i
                seen[g] = True
        return uniq, y[keep_idx]
    if how == "last":
        keep_idx = np.zeros_like(uniq, dtype=int)
        seen = np.zeros_like(uniq, dtype=bool)
        for i in range(inv.size-1, -1, -1):
            g = inv[i]
            if not seen[g]:
                keep_idx[g] = i
                seen[g] = True
        return uniq, y[keep_idx]
    if how == "mean":
        sums = np.zeros_like(uniq, dtype=float)
        counts = np.zeros_like(uniq, dtype=int)
        for val, g in zip(y, inv):
            sums[g] += val
            counts[g] += 1
        return uniq, sums / np.maximum(counts, 1)
    raise ValueError("Invalid deduplicate method. Use 'first'|'last'|'mean'.")


# ==============================
# DTA helpers (TA Instruments / Netzsch text exports)
# ==============================
def _read_text_autodetect(path: Path) -> str:
    """
    Auto-detect UTF-16 BOM / UTF-8-ish encodings used by TA/Netzsch exports.
    Mirrors the standalone EXAMPLES/plot_dta.py helper for consistency.
    """
    return _decode_text_autodetect(path.read_bytes())[0]

def parse_ta_sdt_txt(path: Path) -> tuple[dict[str, str], list[str], pd.DataFrame]:
    """
    Parse TA Instruments SDT/Q600-style text export:
    - Header ends at a line containing 'StartOfData' (case-insensitive)
    - Column names are pulled from 'SigN\tName' lines when available
    Returns: (header_dict, column_names, dataframe)
    """
    text = _read_text_autodetect(path)
    lines = text.splitlines()

    start_idx = None
    marker_found = False
    for i, line in enumerate(lines):
        norm = re.sub(r"\s+", "", line).lower()
        if "startofdata" in norm:
            start_idx = i
            marker_found = True
            break

    if start_idx is None:
        for i, line in enumerate(lines):
            s = line.strip()
            if not s:
                continue
            if s[0].isdigit() or s[0] in "+-.":
                start_idx = i
                break

    if start_idx is None:
        raise ValueError("Could not find data start (no StartOfData marker and no numeric data rows).")

    if marker_found:
        header_lines = lines[:start_idx]
        data_lines = lines[start_idx + 1:]
    else:
        # No StartOfData marker: start_idx points at the first numeric row.
        # Header is everything before it (including the column-name line
        # immediately preceding the data, which must NOT be dropped); data
        # starts exactly at start_idx (handles start_idx == 0 correctly too).
        header_lines = lines[:start_idx]
        data_lines = lines[start_idx:]

    header: dict[str, str] = {}
    for line in header_lines:
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            key = parts[0].strip()
            val = "\t".join(parts[1:]).strip()
            header.setdefault(key, val)

    sig_map: dict[int, str] = {}
    sig_re = re.compile(r"^Sig(\d+)\s+(.*)$")
    for line in header_lines:
        m = sig_re.match(line.strip())
        if m:
            idx = int(m.group(1))
            name = m.group(2).strip()
            sig_map[idx] = name

    if sig_map:
        colnames = [sig_map[i] for i in sorted(sig_map.keys())]
    else:
        first_data = next((ln for ln in data_lines if ln.strip()), "")
        ncols = len(re.split(r"\s+", first_data.strip()))
        # Try the last non-empty header line as real column names (common when
        # there's no StartOfData/SigN metadata but a plain header row exists).
        header_row_candidate = next((ln for ln in reversed(header_lines) if ln.strip()), None)
        colnames = None
        if header_row_candidate is not None:
            tokens = re.split(r"\s+", header_row_candidate.strip())
            if len(tokens) == ncols and not any(_NUM_LIKE.match(t) for t in tokens):
                colnames = [t.strip() for t in tokens]
        if colnames is None:
            colnames = [f"col{i+1}" for i in range(ncols)]

    numeric_rows = []
    for ln in data_lines:
        s = ln.strip()
        if not s:
            continue
        if s[0].isdigit() or s[0] in "+-.":
            numeric_rows.append(s)

    if not numeric_rows:
        raise ValueError("No numeric data rows found after header.")

    df = pd.read_csv(
        io.StringIO("\n".join(numeric_rows)),
        sep=r"\s+",
        engine="python",
        names=colnames,
    )

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="all").reset_index(drop=True)

    return header, colnames, df


def _build_dta_canonical(df: pd.DataFrame) -> dict[str, str]:
    """Infer canonical keys (T_C, time_min, HF_mW, etc.) from DTA-style columns."""
    canon: dict[str, str] = {}
    for col in df.columns:
        low = str(col).lower()
        if "temp" in low:
            canon.setdefault("T_C", col)
        if "time" in low:
            canon.setdefault("time_min", col)
        if "heat flow" in low or "heatflow" in low or "dsc" in low:
            canon.setdefault("DSC_mW_mg", col)
            canon.setdefault("HF_mW", col)
        if ("tg" in low or "mass" in low or "weight" in low) and "%" in low:
            canon.setdefault("TG_pct", col)
        if "dtg" in low or "%/min" in low:
            canon.setdefault("DTG_pct_min", col)
        if ("mass" in low or "weight" in low) and "%" not in low:
            canon.setdefault("mass_mg", col)
    return canon


def _pick_dta_xy(df: pd.DataFrame, canon: dict[str, str]) -> tuple[str, str]:
    if df.shape[1] < 2:
        raise ValueError("DTA table must contain at least two columns.")
    if "T_C" in canon and ("DSC_mW_mg" in canon or "HF_mW" in canon):
        return canon.get("T_C"), canon.get("DSC_mW_mg") or canon.get("HF_mW")
    if "time_min" in canon and ("DSC_mW_mg" in canon or "HF_mW" in canon):
        return canon.get("time_min"), canon.get("DSC_mW_mg") or canon.get("HF_mW")
    if "T_C" in canon and "TG_pct" in canon:
        return canon.get("T_C"), canon.get("TG_pct")
    if "T_C" in canon and "mass_mg" in canon:
        return canon.get("T_C"), canon.get("mass_mg")
    return df.columns[0], df.columns[1]

# ==============================
# Parser registry
# ==============================
@dataclass
class ParserSpec:
    name: str
    sniff: _t.Callable[[str, str], bool]                      # (path, head_text) -> bool
    parse: _t.Callable[[str], tuple[pd.DataFrame, dict]]      # path -> (df, meta)
    priority: int = 100

_REGISTRY: list[ParserSpec] = []

def register_parser(spec: ParserSpec) -> None:
    _REGISTRY.append(spec)
    _REGISTRY.sort(key=lambda s: s.priority)

def _head_text(path: str, n_lines: int = 160) -> str:
    text, _ = _read_text_with_fallbacks(path)
    return "\n".join(text.splitlines()[:n_lines])

# ==============================
# Sniffers
# ==============================
def sniff_ta_sdt(path: str, head: str) -> bool:
    if re.search(r"startofdata", head, re.I) and re.search(r"^Sig\d+\s+", head, re.M | re.I):
        return True
    return False

def sniff_saxs_edf_ascii(path: str, head: str) -> bool:
    if re.search(r"^#\s*EDF_DataBlockID", head, re.M) and re.search(r"^\s*q\(A-1\)\s+I\(q\)\s+Sig\(q\)", head, re.M):
        return True
    if re.search(r"^\s*q\s*\(A-?1\)\s+I\(q\)\s+Sig\(q\)", head, re.M):
        return True
    return False

def sniff_generic_xy(path: str, head: str) -> bool:
    lines = head.splitlines()
    numeric_count = 0
    for s in lines:
        s = s.strip()
        if not s or s.startswith("#") or s.lower().startswith(("version","language","run","sig1","sig2","nsig","startofdata")):
            continue
        tokens = _split_numeric_line(s)
        if tokens is not None:
            numeric_count += 1
        if numeric_count >= 5:
            return True
    return False

def sniff_raman(path: str, head: str) -> bool:
    low = head.lower()
    if "raman" in low and ("shift" in low or "cm-1" in low or "cm^" in low):
        return True
    if re.search(r"raman\s*shift", head, re.I):
        return True
    # simple two-column numeric tables with Raman-ish headers
    for ln in head.splitlines()[:20]:
        if re.search(r"raman\s*shift" , ln, re.I) and re.search(r"intensity|counts|a\.u", ln, re.I):
            return True
    return False

def sniff_xrd(path: str, head: str) -> bool:
    low = head.lower()
    if "xrd" in low and ("2theta" in low or "two theta" in low):
        return True
    for ln in head.splitlines()[:30]:
        if re.search(r"2\s*theta|two\s*theta", ln, re.I) and re.search(r"intensity|counts", ln, re.I):
            return True
    return False

def sniff_dta_table(path: str, head: str) -> bool:
    low = head.lower()
    if "netzsch" in low or "ta instruments" in low or "universal analysis" in low:
        return True
    if re.search(r"dsc|heat\s*flow|tg|dtg", low):
        return True
    # Require actual semicolon-delimited *table structure* on the same line as
    # a temperature token (a real DTA header row has multiple such fields),
    # not just a semicolon and the word "temperature" appearing independently
    # anywhere in the header block — that misfired on e.g. a Raman file whose
    # comment header merely says "# Raman spectrum; acquired at room temperature".
    for ln in head.splitlines():
        if ln.count(";") >= 2 and re.search(r"\btemperature\b", ln, re.I):
            return True
    return False

# ==============================
# Parsers
# ==============================
def parse_ta_sdt(path: str) -> tuple[pd.DataFrame, dict]:
    header, signals, df = parse_ta_sdt_txt(Path(path))
    canonical = _build_dta_canonical(df)
    x_col, y_col = _pick_dta_xy(df, canonical)
    canonical.setdefault("X", x_col)
    canonical.setdefault("Y", y_col)

    meta = {
        "parser": "ta_sdt",
        "used_encoding": "auto",
        "signals": signals,
        "canonical_map": canonical,
        "raw_header": header,
        "path": os.path.abspath(path),
    }
    return df, meta

def parse_saxs_edf_ascii(path: str) -> tuple[pd.DataFrame, dict]:
    text, used_enc = _read_text_with_fallbacks(path)
    lines = text.splitlines()

    header = []
    header_columns_line = None
    data_rows: list[str] = []

    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("#"):
            header.append(s); continue
        if header_columns_line is None:
            if re.search(r"q\s*\(A-?1\)", s) and "I(q)" in s:
                header_columns_line = s
                continue
        data_rows.append(s)

    if header_columns_line is None:
        # fallback: treat first non-# non-numeric as header row
        for i, ln in enumerate(lines):
            s = ln.strip()
            if not s or s.startswith("#"): continue
            parts = _MULTI_SEP.split(s)
            if len(parts) >= 2 and (_normalize_num_token(parts[0]) is None or _normalize_num_token(parts[1]) is None):
                header_columns_line = s
                data_rows = [l.strip() for l in lines[i+1:] if l.strip() and not l.strip().startswith("#")]
                break
    if header_columns_line is None:
        raise ValueError("SAXS EDF ASCII: Missing column header line (e.g., 'q(A-1) I(q) Sig(q)').")

    colnames = _MULTI_SEP.split(header_columns_line.strip())
    cleaned_rows = []
    for raw in data_rows:
        parts = _MULTI_SEP.split(raw.strip())
        norm, ok = [], True
        for p in parts:
            p2 = _normalize_num_token(p)
            if p2 is None:
                ok = False; break
            norm.append(p2)
        if ok and len(norm) == len(colnames):
            cleaned_rows.append(" ".join(norm))
    if not cleaned_rows:
        raise ValueError("SAXS EDF ASCII: no numeric data rows found.")

    df = pd.read_csv(io.StringIO("\n".join(cleaned_rows)), sep=r"\s+", engine="python", header=None)
    df.columns = colnames

    canonical = {}
    i_count = 0
    for col in df.columns:
        low = col.lower()
        if low.startswith("q"):
            canonical.setdefault("q_A^-1", col)
        elif "sig" in low:
            canonical.setdefault("sigma", col)
        else:
            i_count += 1
            canonical["I" if i_count == 1 else f"I{i_count}"] = col

    canonical.setdefault("X", canonical.get("q_A^-1"))
    canonical.setdefault("Y", canonical.get("I"))

    meta = {
        "parser": "saxs_edf_ascii",
        "used_encoding": used_enc,
        "raw_header": "\n".join(header),
        "canonical_map": canonical,
        "path": os.path.abspath(path),
    }
    return df, meta

def parse_generic_xy(path: str) -> tuple[pd.DataFrame, dict]:
    text, used_enc = _read_text_with_fallbacks(path)
    raw_lines = text.splitlines()

    # gather candidate lines (text vs data)
    data_candidates = []
    for s in raw_lines:
        st = s.strip()
        if not st:
            continue
        if st.startswith(("#",";","//","%")):
            continue
        toks = _split_numeric_line(st)
        if toks is not None:
            data_candidates.append(("data", st, toks))
        else:
            data_candidates.append(("text", st, None))

    # first streak of >= 5 numeric lines allowing intermittent text
    dense_window = 7
    min_dense = 5
    start_idx = None
    for i in range(len(data_candidates)):
        window = data_candidates[i:i+dense_window]
        data_positions = [j for j,(k,_,_) in enumerate(window) if k == "data"]
        if len(data_positions) >= min_dense:
            start_idx = i + data_positions[0]
            break

    if start_idx is None:
        # fallback: accept if >= 2 data lines exist anywhere
        numeric_total = sum(1 for k,_,_ in data_candidates if k == "data")
        if numeric_total < 2:
            raise ValueError("Generic XY: No numeric XY data detected in this file.")
        for i,(k, st, toks) in enumerate(data_candidates):
            if k == "data":
                start_idx = i; break

    header_row = None
    if start_idx is not None and start_idx > 0:
        prev_kind, prev_st, _ = data_candidates[start_idx-1]
        if prev_kind == "text":
            header_row = prev_st

    cleaned_rows = []
    for kind, st, toks in data_candidates[start_idx:]:
        if kind != "data":
            continue
        parts = _MULTI_SEP.split(st.strip())
        norm = []
        for p in parts:
            p2 = _normalize_num_token(p)
            if p2 is None:
                break
            norm.append(p2)
        if len(norm) >= 2:
            cleaned_rows.append(" ".join(norm))
    if not cleaned_rows:
        raise ValueError("Generic XY: Could not build a numeric table from detected data rows.")

    df = pd.read_csv(io.StringIO("\n".join(cleaned_rows)), sep=r"\s+", engine="python", header=None)

    if header_row is not None:
        head_tokens = _MULTI_SEP.split(header_row.strip())
        if len(head_tokens) >= df.shape[1] and all(_normalize_num_token(t) is None for t in head_tokens[:2]):
            df.columns = head_tokens[:df.shape[1]]
        else:
            df.columns = [f"col{i+1}" for i in range(df.shape[1])]
    else:
        df.columns = [f"col{i+1}" for i in range(df.shape[1])]

    meta = {
        "parser": "generic_xy",
        "used_encoding": used_enc,
        "canonical_map": {"X": df.columns[0], "Y": df.columns[1]},
        "path": os.path.abspath(path),
    }
    return df, meta

def _read_table_flexible(path: str) -> tuple[pd.DataFrame, str, str | None]:
    text, enc = _read_text_with_fallbacks(path)
    decimal = "," if len(re.findall(r"\d+,\d", text[:2000])) > len(re.findall(r"\d+\.\d", text[:2000])) else "."

    # Decide header=None vs header=0 BEFORE reading, so headerless numeric
    # files never lose their first data row to a phantom pandas header.
    first_line = next((ln for ln in text.splitlines() if ln.strip()), "")
    first_is_numeric = _split_numeric_line(first_line) is not None

    for sep in [";", "\t", ",", " ", None]:
        try:
            df = pd.read_csv(path, sep=sep, decimal=decimal, engine="python",
                              header=None if first_is_numeric else 0)
            if df.shape[1] >= 2:
                return df, enc, sep
        except Exception:
            continue
    raise ValueError("Could not read table with flexible settings.")

def parse_raman(path: str) -> tuple[pd.DataFrame, dict]:
    df, enc, sep = _read_table_flexible(path)
    df = df.dropna(how="all").dropna(how="all", axis=1)
    if df.columns.size < 2:
        raise ValueError("Raman parser needs at least two columns.")

    # detect header row as text if first row non-numeric
    first_row = df.iloc[0]
    if not pd.to_numeric(first_row, errors="coerce").notna().all():
        df.columns = first_row
        df = df.iloc[1:]
    df.columns = [str(c).strip() for c in df.columns]
    shift_col, intensity_col = None, None
    for col in df.columns:
        low = col.lower()
        if shift_col is None and ("raman" in low or "shift" in low or "cm" in low):
            shift_col = col
        elif intensity_col is None and re.search(r"intensity|counts|a\.u", low):
            intensity_col = col
    if shift_col is None or intensity_col is None:
        shift_col, intensity_col = df.columns[0], df.columns[1]

    canonical = {
        "shift_cm^-1": shift_col,
        "intensity": intensity_col,
        "X": shift_col,
        "Y": intensity_col,
    }
    meta = {
        "parser": "raman_xy",
        "used_encoding": enc,
        "sep": sep,
        "canonical_map": canonical,
        "path": os.path.abspath(path),
    }
    return df.reset_index(drop=True), meta

def parse_xrd(path: str) -> tuple[pd.DataFrame, dict]:
    df, enc, sep = _read_table_flexible(path)
    df = df.dropna(how="all").dropna(how="all", axis=1)
    if df.columns.size < 2:
        raise ValueError("XRD parser needs at least two columns.")

    first_row = df.iloc[0]
    if not pd.to_numeric(first_row, errors="coerce").notna().all():
        df.columns = first_row
        df = df.iloc[1:]
    df.columns = [str(c).strip() for c in df.columns]
    two_theta, intensity = None, None
    d_spacing = None
    for col in df.columns:
        low = col.lower()
        if two_theta is None and re.search(r"2\s*theta|two\s*theta|2θ", low):
            two_theta = col
        elif intensity is None and re.search(r"intensity|counts", low):
            intensity = col
        elif d_spacing is None and re.search(r"\bd\b|spacing", low):
            d_spacing = col
    if two_theta is None or intensity is None:
        two_theta, intensity = df.columns[0], df.columns[1]

    canonical = {
        "2theta_deg": two_theta,
        "intensity": intensity,
        "X": two_theta,
        "Y": intensity,
    }
    if d_spacing:
        canonical["d_A"] = d_spacing
    meta = {
        "parser": "xrd_xy",
        "used_encoding": enc,
        "sep": sep,
        "canonical_map": canonical,
        "path": os.path.abspath(path),
    }
    return df.reset_index(drop=True), meta

def sniff_rasx(path: str, head: str) -> bool:
    """Rigaku SmartLab .rasx is a ZIP container, not text — detect by extension
    (the generic _head_text() sniff can't see inside a ZIP's binary layout)."""
    return path.lower().endswith(".rasx")


def parse_rasx(path: str) -> tuple[pd.DataFrame, dict]:
    """Rigaku SmartLab .rasx: a ZIP container. Data*/Profile*.txt holds
    headerless tab-separated (2theta, intensity, count) rows;
    Data*/MesurementConditions*.xml holds axis metadata, including the "Temp"
    axis's Position/EndPosition (deg. C) — the scan's temperature range for
    in-situ/high-temperature XRD, where each pattern is collected while the
    sample continues heating (a real range, not a single instantaneous value).
    """
    import zipfile
    from xml.etree import ElementTree as ET

    with zipfile.ZipFile(path, "r") as zf:
        names = zf.namelist()
        profile_names = sorted(n for n in names if re.match(r"Data\d+/Profile\d+\.txt$", n))
        if not profile_names:
            raise ValueError(f"No Data*/Profile*.txt found inside .rasx: {path}")
        profile_name = profile_names[0]
        raw = zf.read(profile_name).decode("utf-8-sig", errors="replace")

        temp_start = temp_end = None
        cond_name = re.sub(r"Profile(\d+)\.txt$", r"MesurementConditions\1.xml", profile_name)
        if cond_name in names:
            try:
                root_xml = ET.fromstring(zf.read(cond_name))
                for axis in root_xml.iter("Axis"):
                    if axis.get("Name") == "Temp":
                        temp_start = _clean_float_or_none(axis.get("Position"))
                        temp_end = _clean_float_or_none(axis.get("EndPosition")) or temp_start
                        break
            except Exception:
                pass

    data_lines = [ln for ln in raw.splitlines() if ln.strip()]
    if not data_lines:
        raise ValueError(f"Empty profile data in .rasx: {path}")
    rows = [ln.split("\t") for ln in data_lines]
    ncols = len(rows[0])
    colnames = ["2theta_deg", "intensity"] + [f"col{i+1}" for i in range(2, ncols)]
    df = pd.DataFrame(rows, columns=colnames[:ncols])
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="all").reset_index(drop=True)

    canonical = {"2theta_deg": "2theta_deg", "intensity": "intensity", "X": "2theta_deg", "Y": "intensity"}
    meta = {
        "parser": "rasx",
        "canonical_map": canonical,
        "path": os.path.abspath(path),
        "temp_start_C": temp_start,
        "temp_end_C": temp_end,
    }
    return df, meta


def _clean_float_or_none(val) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ==============================
# JCAMP-DX (M15) — the de facto interchange standard across UV/IR/Raman/
# NMR/MS instrument software; one reader unlocks broad compatibility.
# Supports ##XYDATA=(X++(Y..Y)) with AFFN/PAC and compressed ASDF
# (SQZ/DIF/DUP) encodings, plus ##XYPOINTS/##PEAK TABLE=(XY..XY) pairs.
# Compound/multi-block files: only the first data block is read.
# ==============================

_JCAMP_SQZ = {c: v for v, c in enumerate("@ABCDEFGHI")}          # absolute, positive first digit
_JCAMP_SQZ.update({c: -(v + 1) for v, c in enumerate("abcdefghi")})   # absolute, negative
_JCAMP_DIF = {c: v for v, c in enumerate("%JKLMNOPQR")}          # difference, positive
_JCAMP_DIF.update({c: -(v + 1) for v, c in enumerate("jklmnopqr")})   # difference, negative
_JCAMP_DUP = {c: v + 1 for v, c in enumerate("STUVWXYZ")}        # duplicate count 1-8
_JCAMP_DUP["s"] = 9


def _jcamp_tokenize_line(line: str) -> list[tuple[str, str]]:
    """Split one ASDF data line into (kind, numeric_text) tokens, where kind
    is 'num' (AFFN/PAC absolute), 'sqz' (absolute), 'dif' (difference), or
    'dup' (repeat count). The SQZ/DIF/DUP letter encodes the sign and first
    digit; remaining digits follow it."""
    tokens: list[tuple[str, str]] = []
    i, n = 0, len(line)
    while i < n:
        c = line[i]
        if c in " \t,;":
            i += 1
            continue
        if c in _JCAMP_SQZ or c in _JCAMP_DIF or c in _JCAMP_DUP:
            j = i + 1
            while j < n and (line[j].isdigit() or line[j] == "."):
                j += 1
            rest = line[i + 1:j]
            if c in _JCAMP_SQZ:
                v = _JCAMP_SQZ[c]
                text = ("-" if v < 0 else "") + str(abs(v)) + rest
                tokens.append(("sqz", text))
            elif c in _JCAMP_DIF:
                v = _JCAMP_DIF[c]
                text = ("-" if v < 0 else "") + str(abs(v)) + rest
                tokens.append(("dif", text))
            else:
                tokens.append(("dup", str(_JCAMP_DUP[c]) + rest))
            i = j
            continue
        if c.isdigit() or c in "+-.":
            j = i + 1 if c in "+-" else i
            k = j
            while k < n and (line[k].isdigit() or line[k] in ".eE" or (line[k] in "+-" and line[k - 1] in "eE")):
                k += 1
            tokens.append(("num", line[i:k]))
            i = k
            continue
        if c == "?":  # JCAMP's explicit missing-value marker
            tokens.append(("num", "nan"))
            i += 1
            continue
        i += 1  # unknown character: skip
    return tokens


def _jcamp_decode_xydata(data_lines: list[str]) -> list[float]:
    """Decode ##XYDATA=(X++(Y..Y)) lines into the flat Y sequence. The
    first token per line is the line's X value (dropped — X is rebuilt from
    FIRSTX/LASTX/NPOINTS); in DIF mode the FIRST Y of the next line is a
    check value duplicating the previous line's last Y and must be dropped
    (per the JCAMP-DX 4.24 spec's Y-value check convention)."""
    ys: list[float] = []
    last_was_dif = False
    for line in data_lines:
        tokens = _jcamp_tokenize_line(line)
        if not tokens:
            continue
        tokens = tokens[1:]  # drop the leading X token
        first_y_of_line = True
        for kind, text in tokens:
            if kind == "dup":
                count = int(float(text))
                if not ys:
                    continue
                for _ in range(count - 1):
                    if last_was_dif and len(ys) >= 2:
                        ys.append(ys[-1] + (ys[-1] - ys[-2]))
                    else:
                        ys.append(ys[-1])
                first_y_of_line = False
                continue
            value = float(text)
            if kind == "dif":
                if not ys:
                    ys.append(value)
                else:
                    ys.append(ys[-1] + value)
                last_was_dif = True
            else:  # 'num' or 'sqz': absolute value
                if first_y_of_line and last_was_dif and ys:
                    # DIF-mode line-start check value: should equal the
                    # running Y; drop it rather than duplicating the point.
                    first_y_of_line = False
                    last_was_dif = False
                    continue
                ys.append(value)
                last_was_dif = False
            first_y_of_line = False
    return ys


def sniff_jcamp(path: str, head: str) -> bool:
    return bool(re.search(r"^\s*##\s*JCAMP-?DX\s*=", head, re.M | re.I))


def parse_jcamp(path: str) -> tuple[pd.DataFrame, dict]:
    text, used_encoding = _read_text_with_fallbacks(path)
    lines = text.splitlines()

    header: dict[str, str] = {}
    data_mode: str | None = None
    data_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("##"):
            if data_mode is not None:
                break  # first data block only; next LDR ends it
            key, _, value = stripped[2:].partition("=")
            key_norm = re.sub(r"[\s_-]", "", key).upper()
            value = value.split("$$", 1)[0].strip()  # strip inline comments
            header[key_norm] = value
            if key_norm == "XYDATA":
                data_mode = "xydata"
            elif key_norm in ("XYPOINTS", "PEAKTABLE"):
                data_mode = "xypoints"
            continue
        if data_mode is not None and stripped:
            data_lines.append(stripped.split("$$", 1)[0])

    if data_mode is None:
        raise ValueError(f"No ##XYDATA/##XYPOINTS/##PEAK TABLE block found in JCAMP file: {path}")

    xfactor = _clean_float_or_none(header.get("XFACTOR")) or 1.0
    yfactor = _clean_float_or_none(header.get("YFACTOR")) or 1.0

    if data_mode == "xydata":
        ys_raw = _jcamp_decode_xydata(data_lines)
        firstx = _clean_float_or_none(header.get("FIRSTX"))
        lastx = _clean_float_or_none(header.get("LASTX"))
        npoints = _clean_float_or_none(header.get("NPOINTS"))
        n = len(ys_raw)
        if npoints is not None and int(npoints) != n:
            # Trust the decoded sequence but record the discrepancy.
            header["_NPOINTS_MISMATCH"] = f"declared {int(npoints)}, decoded {n}"
        if firstx is not None and lastx is not None and n > 1:
            x = np.linspace(firstx, lastx, n)
        else:
            deltax = _clean_float_or_none(header.get("DELTAX")) or 1.0
            x0 = firstx if firstx is not None else 0.0
            x = x0 + deltax * np.arange(n, dtype=float)
        y = np.asarray(ys_raw, dtype=float) * yfactor
    else:
        xs, ys = [], []
        for line in data_lines:
            for pair in re.split(r"[;]", line):
                nums = re.findall(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", pair)
                if len(nums) >= 2:
                    xs.append(float(nums[0]) * xfactor)
                    ys.append(float(nums[1]) * yfactor)
        x = np.asarray(xs, dtype=float)
        y = np.asarray(ys, dtype=float)

    if len(x) < 2:
        raise ValueError(f"JCAMP file decoded to fewer than 2 points: {path}")

    df = pd.DataFrame({"x": x, "y": y})
    data_type = header.get("DATATYPE", "")
    canonical = {"X": "x", "Y": "y"}
    meta = {
        "parser": "jcamp",
        "used_encoding": used_encoding,
        "canonical_map": canonical,
        "path": os.path.abspath(path),
        "jcamp_title": header.get("TITLE", ""),
        "jcamp_data_type": data_type,
        "jcamp_xunits": header.get("XUNITS", ""),
        "jcamp_yunits": header.get("YUNITS", ""),
    }
    return df, meta


def parse_dta_table(path: str) -> tuple[pd.DataFrame, dict]:
    header, signals, df = parse_ta_sdt_txt(Path(path))
    canonical = _build_dta_canonical(df)
    x_col, y_col = _pick_dta_xy(df, canonical)
    canonical.setdefault("X", x_col)
    canonical.setdefault("Y", y_col)

    meta = {
        "parser": "dta_table",
        "used_encoding": "auto",
        "canonical_map": canonical,
        "path": os.path.abspath(path),
        "raw_header": header,
        "signals": signals,
    }
    return df, meta

# ==============================
# Register parsers (priority)
# ==============================
register_parser(ParserSpec("rasx", sniff_rasx, parse_rasx, priority=5))
register_parser(ParserSpec("jcamp", sniff_jcamp, parse_jcamp, priority=8))
register_parser(ParserSpec("ta_sdt", sniff_ta_sdt, parse_ta_sdt, priority=10))
register_parser(ParserSpec("saxs_edf_ascii", sniff_saxs_edf_ascii, parse_saxs_edf_ascii, priority=20))
register_parser(ParserSpec("dta_table", sniff_dta_table, parse_dta_table, priority=30))
register_parser(ParserSpec("raman_xy", sniff_raman, parse_raman, priority=40))
register_parser(ParserSpec("xrd_xy", sniff_xrd, parse_xrd, priority=50))
register_parser(ParserSpec("generic_xy", sniff_generic_xy, parse_generic_xy, priority=90))

# ==============================
# Public API
# ==============================
def available_parsers() -> list[str]:
    """Registered parser names in priority order — for UI parser-override
    dropdowns (the Custom Import dialog)."""
    return [spec.name for spec in _REGISTRY]


def load_any(path: str, *, x_key: str | None = None, y_key: str | None = None,
             prefer: str | None = None, return_meta: bool = False):
    """
    Load any supported file. If x_key & y_key are provided, return (x, y[, meta]) arrays
    for backward-compat, otherwise return (DataFrame[, meta]).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    head = _head_text(path, n_lines=160)

    chosen = None
    autodetect_failed = False
    if prefer:
        chosen = next((spec for spec in _REGISTRY if spec.name == prefer), None)
        if chosen is None:
            raise ValueError(f"Unknown parser requested via 'prefer': {prefer}")
    if chosen is None:
        for spec in _REGISTRY:
            try:
                if spec.sniff(path, head):
                    chosen = spec; break
            except Exception:
                continue
    if chosen is None:
        chosen = next((s for s in _REGISTRY if s.name == "generic_xy"), None)
        autodetect_failed = True
        if chosen is None:
            raise RuntimeError("No parsers registered.")

    df, meta = chosen.parse(path)
    meta["selected_parser"] = chosen.name
    meta["autodetect_failed"] = autodetect_failed

    if x_key is not None and y_key is not None:
        if x_key not in df.columns or y_key not in df.columns:
            raise KeyError(f"x_key/y_key not found. Available columns: {list(df.columns)}")
        x = df[x_key].astype(float).to_numpy()
        y = df[y_key].astype(float).to_numpy()
        x, y, _ = _stable_sort_xy(x, y)
        return (x, y, meta) if return_meta else (x, y)

    return (df, meta) if return_meta else df

def import_xy(path: str, *, x_key: str | None = None, y_key: str | None = None,
              deduplicate: str | None = None, dedup_round_decimals: int = 12,
              force_ascending: bool = True) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Backward-compatible helper: always returns (x, y, meta).
    If no x_key/y_key are given, the first two numeric columns are used.
    """
    df, meta = load_any(path, return_meta=True)

    if x_key is None or y_key is None:
        x, y = _coerce_2cols_to_xy(df)
        if force_ascending:
            x, y, _ = _stable_sort_xy(x, y)
        if deduplicate is not None:
            x, y = _deduplicate_xy(x, y, deduplicate, dedup_round_decimals)
        return x, y, meta

    if x_key not in df.columns or y_key not in df.columns:
        raise KeyError(f"x_key/y_key not found. Available: {list(df.columns)}")
    x = df[x_key].astype(float).to_numpy()
    y = df[y_key].astype(float).to_numpy()
    if force_ascending:
        x, y, _ = _stable_sort_xy(x, y)
    if deduplicate is not None:
        x, y = _deduplicate_xy(x, y, deduplicate, dedup_round_decimals)
    return x, y, meta

def list_columns(path: str) -> list[str]:
    df, _ = load_any(path, return_meta=True)
    return list(df.columns)
