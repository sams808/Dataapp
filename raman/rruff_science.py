"""
rruff_science.py — RRUFF mineral Raman database ingest (M10), framework-
agnostic (no Qt/Tkinter dependency — the parallel-track milestone the plan
calls out as startable independent of the Qt rewrite).

Format confirmed directly against the real, current rruff.net bulk download
(https://www.rruff.net/about/download-data/ ->
https://www.rruff.net/zipped_data_files/raman/, 2026-07-15) rather than
assumed from memory — the site was reportedly updated, so this is a fresh
spike, not carried over from an old format:

Filename (double-underscore-delimited, extension .txt):
    {Mineral}__{RRUFF_ID}__Raman__{wavelength_nm}__{orientation_deg|''}__
        {polarization|''}__{Raman_Data_Processed|Raman_Data_RAW}__{hash}.txt
Fields are positional but several are legitimately empty for unoriented
samples (e.g. "Abramovite__R070037__Raman__785______Raman_Data_RAW__<hash>.txt"
has two empty fields between wavelength and data-kind).

In-file content: a JCAMP-like ##KEY=VALUE metadata header (keys vary per
sample — e.g. unoriented/unconfirmed samples may lack ##ORIENTATION,
##MEASURED CHEMISTRY, ##PIN_ID entirely — so the header is parsed as an
open-ended dict, not a fixed schema), then a blank line, then comma-
separated "raman_shift, intensity" data rows.

Multiple laser excitation wavelengths are used across the database (514,
532, 633, 780, 785 nm and others seen during ingest) — per the user's own
explicit requirement, this module never auto-labels a match; it only
prepares data for an assisted matching UI (M12) that always shows the
candidate's excitation wavelength and requires explicit user confirmation.

Citation (confirmed from rruff.net/about/): Lafuente, B., Downs, R. T.,
Yang, H., & Stone, N. (2015). The power of databases: the RRUFF project.
Highlights in Mineralogical Crystallography, 1-30. RRUFF also asks that
per-sample external contributors (the ##OWNER/##SOURCE header fields) be
acknowledged "when applicable" — surfaced to the user via RRUFF_CITATION /
RRUFF_ATTRIBUTION_NOTE below (M12's UI should display these, not just this
module quietly knowing them).
"""
from __future__ import annotations

import json
import os
import zipfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

RRUFF_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".raman_cache", "rruff")
RRUFF_INDEX_PATH = os.path.join(RRUFF_CACHE_DIR, "index.json")
RRUFF_RAW_DIR = os.path.join(RRUFF_CACHE_DIR, "raw")


def _safe_print(msg: str) -> None:
    """The default `log` callable everywhere in this module. A PyInstaller
    --windowed build has NO console attached, so sys.stdout/stderr are None
    and a bare print() raises AttributeError deep inside a background
    download/ingest — this swallows that instead of crashing the whole
    operation over a cosmetic progress line."""
    try:
        print(msg)
    except Exception:
        pass

RRUFF_CITATION = (
    "Lafuente, B., Downs, R. T., Yang, H., & Stone, N. (2015). "
    "The power of databases: the RRUFF project. "
    "Highlights in Mineralogical Crystallography, 1-30."
)
RRUFF_ATTRIBUTION_NOTE = (
    "Please also acknowledge the individual sample owner/source shown for "
    "each RRUFF match, when applicable (see that sample's OWNER/SOURCE "
    "fields), per RRUFF's stated attribution request."
)


@dataclass
class RruffSpectrum:
    mineral: str
    rruff_id: str
    wavelength_nm: Optional[float]
    orientation_deg: Optional[str]
    polarization: Optional[str]
    data_kind: str  # "Processed" | "RAW" | ""
    scan_type: str = "Raman"  # "Raman" (high-res category ZIPs) | "Broad_Scan" (LR-Raman.zip)
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    ideal_chemistry: Optional[str] = None
    locality: Optional[str] = None
    owner: Optional[str] = None
    source: Optional[str] = None
    url: Optional[str] = None
    orientation_text: Optional[str] = None
    cell_parameters: Optional[str] = None
    header: Dict[str, str] = field(default_factory=dict)
    source_filename: str = ""


def parse_rruff_filename(filename: str) -> Dict[str, Any]:
    """Split a RRUFF bulk-download filename into its positional fields.
    Returns {} if the filename doesn't match the expected double-underscore
    schema (e.g. a directory entry or an unrelated file some archives
    occasionally bundle) rather than raising — callers should skip those.

    Field 3 is the scan type: "Raman" in the 8 quality/orientation category
    ZIPs, "Broad_Scan" in LR-Raman.zip — confirmed by downloading and
    inspecting the real LR-Raman.zip (9,941 files, ~227MB): "LR" turns out
    to be low-resolution broad-range survey scans of the same samples,
    complementary to (not duplicating) the high-resolution scans. Same
    ##-header in-file format; the scan type is preserved in the returned
    dict so matching UIs can distinguish them.
    """
    stem = filename[:-4] if filename.lower().endswith(".txt") else filename
    parts = stem.split("__")
    if len(parts) < 7:
        return {}
    mineral, rruff_id, scan_type = parts[0], parts[1], parts[2]
    if scan_type not in ("Raman", "Broad_Scan"):
        return {}
    wavelength_txt, orientation_deg, polarization = parts[3], parts[4], parts[5]
    data_kind_field = parts[6]
    data_kind = data_kind_field.replace("Raman_Data_", "") if data_kind_field.startswith("Raman_Data_") else data_kind_field
    try:
        wavelength_nm = float(wavelength_txt) if wavelength_txt else None
    except ValueError:
        wavelength_nm = None
    return {
        "mineral": mineral,
        "rruff_id": rruff_id,
        "scan_type": scan_type,
        "wavelength_nm": wavelength_nm,
        "orientation_deg": orientation_deg or None,
        "polarization": polarization or None,
        "data_kind": data_kind,
    }


def parse_rruff_txt(text: str, source_filename: str = "") -> RruffSpectrum:
    """Parse one RRUFF .txt file's content (## header + comma-separated
    x,y data) into a RruffSpectrum. Header keys are kept generically in
    `.header`; a handful of commonly-present ones are also promoted to
    named fields for convenience, but nothing is assumed mandatory —
    confirmed directly against real files that omit ORIENTATION,
    MEASURED CHEMISTRY, and PIN_ID entirely for some samples."""
    header: Dict[str, str] = {}
    xs: List[float] = []
    ys: List[float] = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("##"):
            key, _, value = line[2:].partition("=")
            header[key.strip()] = value.strip()
            continue
        if "," in line:
            left, _, right = line.partition(",")
            try:
                xs.append(float(left.strip()))
                ys.append(float(right.strip()))
            except ValueError:
                continue

    filename_fields = parse_rruff_filename(source_filename) if source_filename else {}
    # Header first, filename as fallback only — confirmed necessary on the
    # real database, not just theoretical: Gysinite-(Ce) R250121's filename
    # says wavelength "53" (a genuine truncated-digit typo in that one
    # filename) while its own ##RAMAN WAVELENGTH header correctly says 532.
    # The structured header is curated metadata; the filename is a derived,
    # manually-typeable string, so it's the less trustworthy of the two —
    # same direction already used for mineral/rruff_id below.
    wavelength_nm = None
    if "RAMAN WAVELENGTH" in header:
        try:
            wavelength_nm = float(header["RAMAN WAVELENGTH"])
        except ValueError:
            wavelength_nm = None
    if wavelength_nm is None:
        wavelength_nm = filename_fields.get("wavelength_nm")

    data_kind = filename_fields.get("data_kind", "")
    if not data_kind and "FILETYPE" in header:
        data_kind = header["FILETYPE"].replace("Raman ", "")

    return RruffSpectrum(
        mineral=header.get("NAMES") or filename_fields.get("mineral", ""),
        rruff_id=header.get("RRUFFID") or filename_fields.get("rruff_id", ""),
        wavelength_nm=wavelength_nm,
        orientation_deg=filename_fields.get("orientation_deg"),
        polarization=filename_fields.get("polarization"),
        data_kind=data_kind,
        scan_type=filename_fields.get("scan_type", "Raman"),
        x=np.asarray(xs, dtype=float),
        y=np.asarray(ys, dtype=float),
        ideal_chemistry=header.get("IDEAL CHEMISTRY"),
        locality=header.get("LOCALITY"),
        owner=header.get("OWNER"),
        source=header.get("SOURCE"),
        url=header.get("URL"),
        orientation_text=header.get("ORIENTATION"),
        cell_parameters=header.get("CELL PARAMETERS"),
        header=header,
        source_filename=source_filename,
    )


def ingest_zip(
    zip_path: str, *, raw_dir: str = RRUFF_RAW_DIR, category: str = "",
    max_peaks: int = 15, progress_every: int = 500, log: Callable[[str], None] = _safe_print,
) -> List[Dict[str, Any]]:
    """Extract every .txt in `zip_path`, parse it, pre-extract peak
    candidates (fitting_science.find_peak_candidates — the same utility
    M8's "Auto-find peaks" uses, per the plan's reuse principle), and
    return one lightweight index record per spectrum. Only "Processed"
    spectra get peak-extracted (RAW files are kept for reference/overlay
    but a baseline-uncorrected RAW curve isn't a meaningful peak-finder
    input) — both kinds are indexed either way.

    Raw text is written under `raw_dir` for later full-spectrum overlay;
    the returned records are what actually goes into the searchable index.
    """
    from core.fitting_science import find_peak_candidates

    os.makedirs(raw_dir, exist_ok=True)
    records: List[Dict[str, Any]] = []

    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".txt")]
        for i, name in enumerate(names):
            if progress_every and i and i % progress_every == 0:
                log(f"  ...{i}/{len(names)} in {os.path.basename(zip_path)}")
            try:
                raw_bytes = zf.read(name)
                text = raw_bytes.decode("utf-8", errors="replace")
            except (OSError, zipfile.BadZipFile):
                continue

            spectrum = parse_rruff_txt(text, source_filename=name)
            if len(spectrum.x) < 5 or not spectrum.mineral:
                continue

            out_path = os.path.join(raw_dir, name)
            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(text)
            except OSError:
                out_path = ""

            peaks: List[float] = []
            if spectrum.data_kind.lower().startswith("process"):
                try:
                    peaks = find_peak_candidates(spectrum.x, spectrum.y, max_peaks=max_peaks)
                except Exception:
                    peaks = []

            records.append({
                "mineral": spectrum.mineral,
                "rruff_id": spectrum.rruff_id,
                "wavelength_nm": spectrum.wavelength_nm,
                "orientation_deg": spectrum.orientation_deg,
                "polarization": spectrum.polarization,
                "data_kind": spectrum.data_kind,
                "scan_type": spectrum.scan_type,
                "ideal_chemistry": spectrum.ideal_chemistry,
                "locality": spectrum.locality,
                "owner": spectrum.owner,
                "source": spectrum.source,
                "url": spectrum.url,
                "peaks": peaks,
                "raw_path": out_path,
                "category": category,
                "x_min": float(spectrum.x.min()) if len(spectrum.x) else None,
                "x_max": float(spectrum.x.max()) if len(spectrum.x) else None,
            })

    return records


def build_index(zip_paths_with_categories: List[tuple], *, cache_dir: str = RRUFF_CACHE_DIR,
                log: Callable[[str], None] = _safe_print) -> int:
    """Ingest a list of (zip_path, category_label) pairs and write one
    consolidated index.json under cache_dir. Returns the total record
    count. Existing raw files/index are overwritten by a full rebuild —
    this is meant to be re-run wholesale when refreshing the database, not
    incrementally patched."""
    os.makedirs(cache_dir, exist_ok=True)
    raw_dir = os.path.join(cache_dir, "raw")
    all_records: List[Dict[str, Any]] = []
    for zip_path, category in zip_paths_with_categories:
        log(f"Ingesting {zip_path} (category={category})...")
        records = ingest_zip(zip_path, raw_dir=raw_dir, category=category, log=log)
        log(f"  -> {len(records)} spectra")
        all_records.extend(records)

    index_path = os.path.join(cache_dir, "index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f)
    return len(all_records)


def load_index(cache_dir: str = RRUFF_CACHE_DIR) -> List[Dict[str, Any]]:
    index_path = os.path.join(cache_dir, "index.json")
    if not os.path.isfile(index_path):
        return []
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def index_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Small diagnostic summary — mineral/wavelength coverage — useful both
    for a one-off sanity check after ingest and for a future "database
    info" panel in the match-assist UI (M12)."""
    minerals = {r["mineral"] for r in records if r.get("mineral")}
    wavelengths = sorted({r["wavelength_nm"] for r in records if r.get("wavelength_nm")})
    return {
        "n_spectra": len(records),
        "n_minerals": len(minerals),
        "wavelengths_nm": wavelengths,
    }


# =============================================================================
# No-Python-needed setup: download the category ZIPs straight from rruff.net
# and build the index in one call. This is what lets a colleague running
# only the portable PRISM.exe (no Python install) get the Raman ID database
# — either by clicking "Download RRUFF database..." in the workspace, or via
# `PRISM.exe --build-rruff-cache` (what the shipped .bat/.ps1 scripts run).
# stdlib-only (urllib) so it works from the frozen exe with no new dependency.
# =============================================================================

RRUFF_DOWNLOAD_BASE = "https://www.rruff.net/zipped_data_files/raman/"
# The 7 non-empty quality x orientation category ZIPs (poor_oriented.zip does
# not exist on the server — confirmed against the real site, see the module
# docstring). "lr_broad_scan" (LR-Raman.zip, ~227MB of low-resolution survey
# scans) is available but NOT downloaded by default — opt in explicitly, it
# roughly doubles the download for scans that duplicate the same minerals at
# lower resolution.
RRUFF_CATEGORIES: Tuple[str, ...] = (
    "excellent_oriented", "excellent_unoriented",
    "fair_oriented", "fair_unoriented",
    "poor_unoriented",
    "unrated_oriented", "unrated_unoriented",
)
RRUFF_BROAD_SCAN_CATEGORY = "lr_broad_scan"


def _category_url(category: str) -> str:
    if category == RRUFF_BROAD_SCAN_CATEGORY:
        return RRUFF_DOWNLOAD_BASE + "LR-Raman.zip"
    return f"{RRUFF_DOWNLOAD_BASE}{category}.zip"


def _download_file(url: str, dest: str, *, log: Callable[[str], None] = _safe_print,
                   chunk_size: int = 1 << 20) -> None:
    """Stream a URL to disk (urllib only — no extra dependency, and it works
    from the frozen exe). Skips re-downloading a file that's already there
    and non-empty, so an interrupted run can simply be re-launched; writes
    to a .part sidecar first so a download that dies mid-stream is never
    mistaken for a complete one on the next run."""
    import urllib.request
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        log(f"  already downloaded: {os.path.basename(dest)}")
        return
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "PRISM/1 (github.com/sams808/PRISM)"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = getattr(resp, "length", None) or 0
        read = 0
        with open(tmp, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                read += len(chunk)
                pct = f" ({100 * read / total:.0f}%)" if total else ""
                log(f"  {os.path.basename(dest)}: {read / 1e6:.0f} MB{pct}")
    os.replace(tmp, dest)


def download_rruff_zips(
    target_dir: str, *, categories: Optional[Sequence[str]] = None,
    log: Callable[[str], None] = _safe_print,
) -> List[Tuple[str, str]]:
    """Download the RRUFF category ZIPs into target_dir (kept there — a
    re-run reuses whatever already downloaded successfully). Returns
    [(zip_path, category), ...] for build_index(), skipping any category
    whose download failed (logged, not raised — a flaky connection
    shouldn't lose the categories that DID succeed)."""
    os.makedirs(target_dir, exist_ok=True)
    cats = list(categories) if categories else list(RRUFF_CATEGORIES)
    out: List[Tuple[str, str]] = []
    for cat in cats:
        dest = os.path.join(target_dir, f"{cat}.zip" if cat != RRUFF_BROAD_SCAN_CATEGORY else "LR-Raman.zip")
        log(f"Downloading {os.path.basename(dest)}...")
        try:
            _download_file(_category_url(cat), dest, log=log)
        except Exception as exc:
            log(f"  FAILED ({cat}): {exc}")
            continue
        out.append((dest, cat))
    return out


def download_and_build_rruff_cache(
    *, target_dir: Optional[str] = None, cache_dir: str = RRUFF_CACHE_DIR,
    categories: Optional[Sequence[str]] = None, log: Callable[[str], None] = _safe_print,
) -> int:
    """One call, no Python needed: download the RRUFF category ZIPs (kept
    under cache_dir/downloads so a re-run doesn't re-download) and build
    the local search index. Returns the number of spectra ingested; raises
    if every download failed (nothing to index)."""
    target_dir = target_dir or os.path.join(cache_dir, "downloads")
    zips = download_rruff_zips(target_dir, categories=categories, log=log)
    if not zips:
        raise RuntimeError(
            "No RRUFF category ZIP could be downloaded — check the internet "
            "connection (or a firewall blocking rruff.net) and try again."
        )
    log(f"Building the search index from {len(zips)} ZIP(s)...")
    return build_index(zips, cache_dir=cache_dir, log=log)


# =============================================================================
# Match-assist scoring (M12) — ranks candidates, NEVER auto-labels. Per the
# user's explicit original requirement: RRUFF spectra span many laser
# excitation wavelengths, which affects relative peak intensities/fluorescence
# even though Raman shift (cm^-1) is nominally wavelength-independent — so
# every candidate must show its own wavelength, and the user always makes
# the final call. This module only ranks; qt_rruff.py's UI is what actually
# requires an explicit user accept/reject before anything gets applied.
# =============================================================================

def score_match(query_peaks: List[float], candidate_peaks: List[float], tolerance: float = 10.0) -> Dict[str, float]:
    """How many of the query's peaks have a candidate peak within
    `tolerance` cm^-1. A simple, explainable first-pass metric (peak-count
    overlap), not a claim of definitive identification — deliberately easy
    for a user to sanity-check by eye against the overlay this feeds."""
    if not query_peaks:
        return {"matched": 0, "fraction": 0.0}
    matched = sum(1 for qp in query_peaks if any(abs(qp - cp) <= tolerance for cp in candidate_peaks))
    return {"matched": matched, "fraction": matched / len(query_peaks)}


def filter_rruff_index(
    index: List[Dict[str, Any]], *,
    wavelength_nm: Optional[float] = None, wavelength_tol: float = 2.0,
    oriented: Optional[bool] = None,
    scan_type: Optional[str] = None,
    quality: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Restrict the searchable index before ranking (user request: choose
    the laser wavelength, orientation, etc. instead of matching everything).
    Every argument is an optional constraint; None means "any".

    - wavelength_nm: keep records within wavelength_tol of the requested λ
      (the default ±2 nm groups the corpus's near-duplicates like 532/532.6,
      632.8/633; records with no recorded λ are excluded when filtering).
    - oriented: True keeps only records with an orientation angle, False
      only those without one.
    - scan_type: exact match ("Raman" high-res or "Broad_Scan" LR survey).
    - quality: prefix match on the ingest category ("excellent", "fair",
      "poor", "unrated" — categories look like "excellent_oriented").
    """
    out = []
    for rec in index:
        if wavelength_nm is not None:
            wl = rec.get("wavelength_nm")
            if wl is None or abs(float(wl) - float(wavelength_nm)) > wavelength_tol:
                continue
        if oriented is not None:
            has_orientation = rec.get("orientation_deg") not in (None, "")
            if has_orientation != oriented:
                continue
        if scan_type is not None and rec.get("scan_type") != scan_type:
            continue
        if quality is not None and not str(rec.get("category", "")).startswith(quality):
            continue
        out.append(rec)
    return out


def rank_rruff_matches(
    query_peaks: List[float], index: List[Dict[str, Any]], *, tolerance: float = 10.0, top_n: int = 25,
) -> List[Dict[str, Any]]:
    """Rank indexed RRUFF spectra by peak-overlap against query_peaks.
    Returns a plain ranked list (each record + matched/fraction fields),
    best match first — never a single "the" answer. Records with zero
    matched peaks, or with no pre-extracted peaks at all (RAW spectra, per
    ingest_zip's Processed-only peak extraction), are excluded."""
    scored = []
    for rec in index:
        candidate_peaks = rec.get("peaks") or []
        if not candidate_peaks:
            continue
        result = score_match(query_peaks, candidate_peaks, tolerance=tolerance)
        if result["matched"] == 0:
            continue
        scored.append({**rec, "matched_peaks": result["matched"], "match_fraction": result["fraction"]})
    scored.sort(key=lambda r: (r["matched_peaks"], r["match_fraction"]), reverse=True)
    return scored[:top_n]


# =============================================================================
# AMCSD CIF companion database (RRUFF->CIF overlay handoff, backlog item).
# rruff.net also distributes the American Mineralogist Crystal Structure
# Database as per-record CIF files named "{Mineral}__{AMCSD_ID}.cif"
# (https://www.rruff.net/AMS/zipped_files/cif.zip, ~21.7k files) — enough to
# hand a matched mineral's predicted XRD pattern straight to the CIF
# overlay, closing the loop from Raman identification to structural
# verification.
# =============================================================================

AMCSD_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".raman_cache", "amcsd")


def _normalize_mineral(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


AMCSD_DOWNLOAD_URL = "https://www.rruff.net/AMS/zipped_files/cif.zip"


def download_and_build_amcsd_cache(
    *, target_dir: Optional[str] = None, cache_dir: str = AMCSD_CACHE_DIR,
    log: Callable[[str], None] = _safe_print,
) -> int:
    """The AMCSD counterpart of download_and_build_rruff_cache(): downloads
    the ~66MB AMCSD structures ZIP (no Python needed) and ingests it, so the
    "Overlay candidate's XRD (CIF)" button works without anyone having run a
    Python one-liner first."""
    target_dir = target_dir or os.path.join(cache_dir, "downloads")
    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, "cif.zip")
    log("Downloading AMCSD structures (cif.zip)...")
    _download_file(AMCSD_DOWNLOAD_URL, zip_path, log=log)
    log("Indexing CIFs...")
    return ingest_amcsd_cif_zip(zip_path, cache_dir=cache_dir, log=log)


def ingest_amcsd_cif_zip(zip_path: str, *, cache_dir: str = AMCSD_CACHE_DIR, progress_every: int = 2000,
                         log: Callable[[str], None] = _safe_print) -> int:
    """Extract every .cif from the AMCSD bulk ZIP into cache_dir/cif/ and
    build cif_index.json mapping normalized mineral name -> list of cif
    filenames. Returns the number of CIFs indexed."""
    cif_dir = os.path.join(cache_dir, "cif")
    os.makedirs(cif_dir, exist_ok=True)
    index: Dict[str, List[str]] = {}

    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".cif")]
        for i, name in enumerate(names):
            if progress_every and i and i % progress_every == 0:
                log(f"  ...{i}/{len(names)} CIFs")
            base = os.path.basename(name)
            mineral = base.split("__", 1)[0]
            key = _normalize_mineral(mineral)
            if not key:
                continue
            out_path = os.path.join(cif_dir, base)
            try:
                with open(out_path, "wb") as f:
                    f.write(zf.read(name))
            except OSError:
                continue
            index.setdefault(key, []).append(base)

    with open(os.path.join(cache_dir, "cif_index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f)
    return sum(len(v) for v in index.values())


def find_cifs_for_mineral(mineral: str, *, cache_dir: str = AMCSD_CACHE_DIR, max_results: int = 3) -> List[str]:
    """Absolute paths of cached AMCSD CIFs for a mineral name
    (case/punctuation-insensitive). Empty list when the cache hasn't been
    built or the mineral has no AMCSD structure."""
    index_path = os.path.join(cache_dir, "cif_index.json")
    if not os.path.isfile(index_path):
        return []
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    names = index.get(_normalize_mineral(mineral), [])
    cif_dir = os.path.join(cache_dir, "cif")
    out = [os.path.join(cif_dir, n) for n in names[:max_results]]
    return [p for p in out if os.path.isfile(p)]


# =============================================================================
# Shareable single-file cache (user request: hand colleagues ONE file).
# =============================================================================

def pack_rruff_database(cache_dir: str = RRUFF_CACHE_DIR, out_path: Optional[str] = None) -> str:
    """Pack the whole RRUFF cache (index.json + raw spectra) into ONE
    SQLite file a colleague can import with unpack_rruff_database().
    RRUFF data is redistributable with attribution (Lafuente et al. 2015)."""
    import json
    import sqlite3
    out_path = out_path or os.path.join(cache_dir, "rruff_pack.sq")
    index_path = os.path.join(cache_dir, "index.json")
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    if os.path.exists(out_path):
        os.remove(out_path)
    con = sqlite3.connect(out_path)
    con.executescript("CREATE TABLE meta (key TEXT, value TEXT);"
                      "CREATE TABLE raw (filename TEXT PRIMARY KEY, content TEXT);"
                      "CREATE TABLE idx (json TEXT);")
    con.execute("INSERT INTO meta VALUES ('citation', ?)", (RRUFF_CITATION,))
    con.execute("INSERT INTO idx VALUES (?)", (json.dumps(index),))
    for rec in index:
        p = rec.get("raw_path") or ""
        if p and os.path.isfile(p):
            with open(p, encoding="utf-8", errors="replace") as f:
                con.execute("INSERT OR IGNORE INTO raw VALUES (?, ?)", (os.path.basename(p), f.read()))
    con.commit()
    con.close()
    return out_path


def unpack_rruff_database(sq_path: str, cache_dir: str = RRUFF_CACHE_DIR) -> int:
    """Restore a pack_rruff_database() file into a local cache; raw_path
    entries are rewritten to this machine's cache location."""
    import json
    import sqlite3
    raw_dir = os.path.join(cache_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    con = sqlite3.connect(f"file:{sq_path}?mode=ro", uri=True)
    index = json.loads(con.execute("SELECT json FROM idx").fetchone()[0])
    n = 0
    for filename, content in con.execute("SELECT filename, content FROM raw"):
        with open(os.path.join(raw_dir, filename), "w", encoding="utf-8") as f:
            f.write(content)
        n += 1
    con.close()
    for rec in index:
        if rec.get("raw_path"):
            rec["raw_path"] = os.path.join(raw_dir, os.path.basename(rec["raw_path"]))
    with open(os.path.join(cache_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f)
    return n
