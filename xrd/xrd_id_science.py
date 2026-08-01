"""
xrd_id_science.py — XRD phase identification (framework-agnostic): a
QualX-style search-match engine over the user's OWN reference databases.

PRISM ships no reference data. Users download whichever card database they
have the rights to use — any QualX-format .sq works — and register it in
the XRD ID workspace ("Add database…" / "Add folder…"). Registered
databases live in a small JSON registry (~/.raman_cache/xrd_id/
databases.json); any number can be enabled at once and every search probes
all enabled ones, so results can mix several databases exactly like QualX.

QualX .sq format (reverse-engineered): an `id` table with one row per
card — name / mineralname / chemical_formula / spacegroup / quality / rir
and the reference pattern as comma-separated text in `dvalue` (d-spacings,
Å) and `intensita` (intensities) — plus a `chemical` table (card id →
element) and `infodb` (provenance). Registering such a file converts it
ONCE into PRISM's indexed format below (stored under ~/.raman_cache/
xrd_id/imported/); a .sq that is already in PRISM's format (e.g. handed
over by a colleague) registers in place, no copy.

PRISM indexed .sq schema:
  cards(card_id, source, source_code, name, mineral, formula, spacegroup,
        quality, rir, nd, d, i)          — lines capped at the strongest 60
  elements(card_id, element)             — for chemistry filters
  strong_lines(card_id, d, i)            — top-3 lines, d-indexed, the
                                           search-match prefilter
  sources(tag, date, ncard, origin)      — provenance of each merged input

Search-match: query peaks (2θ, I) at wavelength λ → d-spacings; candidate
cards are prefiltered by strong-line coincidence, then scored by the
geometric mean of intensity-weighted coverage in both directions (how much
of the card's pattern the measurement explains, and vice versa) — never an
automatic answer, always a ranked list, same philosophy as the RRUFF
match-assist.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

XRD_ID_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".raman_cache", "xrd_id")
XRD_ID_DB_PATH = os.path.join(XRD_ID_CACHE_DIR, "xrdid.sq")  # legacy single-db location
XRD_ID_REGISTRY_PATH = os.path.join(XRD_ID_CACHE_DIR, "databases.json")
XRD_ID_IMPORT_DIR = os.path.join(XRD_ID_CACHE_DIR, "imported")

CU_KA1 = 1.5406  # Å — the default lab wavelength

MAX_LINES_PER_CARD = 60  # strongest lines kept; plenty for identification
N_STRONG = 3             # lines per card in the prefilter index


# =============================================================================
# d-spacing / 2θ conversions
# =============================================================================

def two_theta_to_d(two_theta_deg: np.ndarray, wavelength: float = CU_KA1) -> np.ndarray:
    tt = np.asarray(two_theta_deg, float)
    s = np.sin(np.radians(tt / 2.0))
    return np.where(s > 1e-9, wavelength / (2.0 * np.where(s > 1e-9, s, 1.0)), np.inf)


def d_to_two_theta(d: np.ndarray, wavelength: float = CU_KA1) -> np.ndarray:
    d = np.asarray(d, float)
    arg = wavelength / (2.0 * np.where(d > 1e-9, d, np.inf))
    out = np.full_like(d, np.nan)
    ok = (arg > 0) & (arg <= 1.0)
    out[ok] = 2.0 * np.degrees(np.arcsin(arg[ok]))
    return out


# =============================================================================
# Database build (one-time merge of the QualX-format sources)
# =============================================================================

def _parse_lines_text(d_text, i_text) -> Tuple[np.ndarray, np.ndarray]:
    """The dvalue/intensita columns are comma-separated float text (some
    databases declare them BLOB but store text all the same)."""
    def _floats(t):
        if t is None:
            return np.array([])
        if isinstance(t, bytes):
            t = t.decode("ascii", errors="replace")
        vals = []
        for tok in str(t).split(","):
            tok = tok.strip()
            if tok:
                try:
                    vals.append(float(tok))
                except ValueError:
                    pass
        return np.array(vals, float)

    d = _floats(d_text)
    i = _floats(i_text)
    n = min(len(d), len(i))
    return d[:n], i[:n]


def _safe_float(v) -> Optional[float]:
    """Some databases store empty numerics as runs of spaces — not None, not ''."""
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


def _clean_name(name: Optional[str]) -> str:
    """Some databases carry $-prefixed typesetting codes in names ('$GB-...')."""
    s = (name or "").strip()
    s = re.sub(r"\$[A-Z0-9]{1,3}[- ]?", "", s)
    return s.strip()


def build_xrd_database(
    sources: Sequence[Tuple[str, str]], *, out_path: str = XRD_ID_DB_PATH,
    min_lines: int = 3, progress_every: int = 50000, log=print,
) -> Dict[str, int]:
    """Convert (and optionally merge) QualX-format .sq files into one
    PRISM-indexed database. sources: [(path, tag), ...] — the tag labels
    every card's provenance in results. Cards keep their original id as
    source_code. Cards with fewer than min_lines usable lines are skipped
    (nothing to match against). Returns {tag: n_cards_ingested}."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        os.remove(out_path)
    out = sqlite3.connect(out_path)
    out.executescript("""
        CREATE TABLE cards (
            card_id INTEGER PRIMARY KEY,
            source TEXT NOT NULL, source_code TEXT NOT NULL,
            name TEXT, mineral TEXT, formula TEXT, spacegroup TEXT,
            quality TEXT, rir REAL, nd INTEGER, d TEXT, i TEXT
        );
        CREATE TABLE elements (card_id INTEGER NOT NULL, element TEXT NOT NULL);
        CREATE TABLE strong_lines (card_id INTEGER NOT NULL, d REAL NOT NULL, i REAL NOT NULL);
        CREATE TABLE sources (tag TEXT, date TEXT, ncard INTEGER, origin TEXT);
    """)

    counts: Dict[str, int] = {}
    for path, tag in sources:
        src = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            for row in src.execute("SELECT * FROM infodb"):
                out.execute("INSERT INTO sources VALUES (?,?,?,?)", (tag, str(row[1]), int(row[2]), str(row[-1])))
        except sqlite3.Error:
            out.execute("INSERT INTO sources VALUES (?,?,?,?)", (tag, "", 0, ""))

        n_in = 0
        cur = src.execute(
            "SELECT id, name, mineralname, chemical_formula, spacegroup, quality, rir, dvalue, intensita FROM id"
        )
        rows_cards, rows_strong = [], []
        for k, (cid, name, mineral, formula, sg, quality, rir, d_text, i_text) in enumerate(cur):
            if progress_every and k and k % progress_every == 0:
                log(f"  [{tag}] scanned {k} cards, kept {n_in}…")
            d, i = _parse_lines_text(d_text, i_text)
            if len(d) < min_lines:
                continue
            imax = float(i.max()) if len(i) else 0.0
            if imax <= 0:
                continue
            i = i / imax * 100.0
            order = np.argsort(i)[::-1][:MAX_LINES_PER_CARD]
            d, i = d[order], i[order]
            rows_cards.append((
                tag, str(cid), _clean_name(name), _clean_name(mineral),
                (formula or "").strip().strip('"'), (sg or "").strip(),
                (quality or "").strip(), _safe_float(rir),
                len(d), ",".join(f"{v:.5f}" for v in d), ",".join(f"{v:.2f}" for v in i),
            ))
            n_in += 1
        out.executemany(
            "INSERT INTO cards (source, source_code, name, mineral, formula, spacegroup, quality, rir, nd, d, i) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows_cards)
        # strong lines need the new card_id: fetch back in insert order
        first_id = out.execute("SELECT MAX(card_id) FROM cards").fetchone()[0] - len(rows_cards) + 1
        for offset, rec in enumerate(rows_cards):
            d = [float(v) for v in rec[9].split(",")[:N_STRONG]]
            i = [float(v) for v in rec[10].split(",")[:N_STRONG]]
            for dv, iv in zip(d, i):
                rows_strong.append((first_id + offset, dv, iv))
        out.executemany("INSERT INTO strong_lines VALUES (?,?,?)", rows_strong)

        # elements (chemistry filter): only for cards we kept
        code_to_id = {rec[1]: first_id + off for off, rec in enumerate(rows_cards)}
        rows_el = []
        try:
            for cid, el in src.execute("SELECT id, chemical_element FROM chemical"):
                card_id = code_to_id.get(str(cid))
                if card_id is not None and el:
                    rows_el.append((card_id, str(el).strip().capitalize()))
        except sqlite3.Error:
            pass
        out.executemany("INSERT INTO elements VALUES (?,?)", rows_el)
        out.commit()
        counts[tag] = n_in
        log(f"[{tag}] ingested {n_in} cards ({len(rows_el)} element links)")
        src.close()

    log("indexing…")
    out.executescript("""
        CREATE INDEX idx_strong_d ON strong_lines(d);
        CREATE INDEX idx_elements ON elements(element, card_id);
        CREATE INDEX idx_cards_mineral ON cards(mineral);
        CREATE INDEX idx_cards_formula ON cards(formula);
    """)
    out.commit()
    out.close()
    return counts


def database_summary(db_path: str = XRD_ID_DB_PATH) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(db_path):
        return None
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    by_source = dict(con.execute("SELECT source, COUNT(*) FROM cards GROUP BY source"))
    total = sum(by_source.values())
    qualities = sorted({(q or "").strip().upper()
                        for (q,) in con.execute("SELECT DISTINCT quality FROM cards")
                        if (q or "").strip()})
    con.close()
    return {"total_cards": total, "by_source": by_source, "qualities": qualities, "path": db_path}


# =============================================================================
# Database registry — use whatever databases you want, several at once
# =============================================================================

def sniff_sq_format(path: str) -> Optional[str]:
    """'prism' (PRISM's indexed schema), 'qualx' (original QualX schema),
    or None (not a recognizable card database)."""
    if not os.path.isfile(path):
        return None
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        tables = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
        if {"cards", "strong_lines"} <= tables:
            con.close()
            return "prism"
        if "id" in tables:
            cols = {r[1] for r in con.execute("PRAGMA table_info(id)")}
            con.close()
            if {"dvalue", "intensita"} <= cols:
                return "qualx"
            return None
        con.close()
    except sqlite3.Error:
        return None
    return None


def load_registry(registry_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """The registered databases: [{'name', 'path', 'enabled', 'origin'?}, …].
    First call migrates a pre-registry unified database (the old fixed
    ~/.raman_cache/xrd_id/xrdid.sq location) into the registry so existing
    setups keep working untouched."""
    registry_path = registry_path or XRD_ID_REGISTRY_PATH
    entries: List[Dict[str, Any]] = []
    if os.path.isfile(registry_path):
        try:
            with open(registry_path, encoding="utf-8") as f:
                entries = list(json.load(f).get("databases", []))
        except (OSError, ValueError):
            entries = []
    if not any(os.path.normcase(e.get("path", "")) == os.path.normcase(XRD_ID_DB_PATH)
               for e in entries) and os.path.isfile(XRD_ID_DB_PATH) \
            and os.path.normcase(registry_path) == os.path.normcase(XRD_ID_REGISTRY_PATH):
        entries.insert(0, {"name": "Local database", "path": XRD_ID_DB_PATH, "enabled": True})
        save_registry(entries, registry_path)
    return entries


def save_registry(entries: Sequence[Dict[str, Any]], registry_path: Optional[str] = None) -> None:
    registry_path = registry_path or XRD_ID_REGISTRY_PATH
    os.makedirs(os.path.dirname(registry_path), exist_ok=True)
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "databases": list(entries)}, f, indent=1)


def _unique_name(base: str, entries: Sequence[Dict[str, Any]]) -> str:
    taken = {e.get("name") for e in entries}
    name, k = base, 2
    while name in taken:
        name = f"{base} ({k})"
        k += 1
    return name


def register_database(path: str, name: Optional[str] = None, *,
                      registry_path: Optional[str] = None,
                      import_dir: Optional[str] = None,
                      log=print) -> Dict[str, Any]:
    """Register a card database for searching. A PRISM-format .sq is
    registered in place (no copy — network drives are fine); a QualX-format
    .sq is converted ONCE into the indexed format under import_dir, then
    the converted file is registered (with the original recorded as
    'origin'). Re-registering an already-registered path is a no-op.
    Raises ValueError when the file isn't a recognizable card database."""
    registry_path = registry_path or XRD_ID_REGISTRY_PATH
    import_dir = import_dir or XRD_ID_IMPORT_DIR
    path = os.path.abspath(path)
    fmt = sniff_sq_format(path)
    if fmt is None:
        raise ValueError(
            f"{os.path.basename(path)} is not a recognizable card database "
            "(expected a QualX-format or PRISM-format .sq SQLite file).")
    entries = load_registry(registry_path)
    for e in entries:
        if os.path.normcase(e.get("path", "")) == os.path.normcase(path) or \
                os.path.normcase(e.get("origin") or "") == os.path.normcase(path):
            return e
    name = _unique_name((name or os.path.splitext(os.path.basename(path))[0]).strip(), entries)
    if fmt == "qualx":
        os.makedirs(import_dir, exist_ok=True)
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", name) or "db"
        out_path = os.path.join(import_dir, f"{slug}.sq")
        log(f"Importing '{name}' (one-time indexing — large databases take a few minutes)…")
        counts = build_xrd_database([(path, name)], out_path=out_path, log=log)
        log(f"Imported {sum(counts.values())} cards from '{name}'.")
        entry = {"name": name, "path": out_path, "enabled": True, "origin": path}
    else:
        entry = {"name": name, "path": path, "enabled": True}
    entries.append(entry)
    save_registry(entries, registry_path)
    return entry


def register_folder(folder: str, *, registry_path: Optional[str] = None,
                    import_dir: Optional[str] = None, log=print) -> List[Dict[str, Any]]:
    """Register every recognizable .sq in a folder (non-recursive first,
    then one level of subfolders — the layout database downloads unpack to).
    Unrecognizable files are skipped with a log line."""
    added: List[Dict[str, Any]] = []
    candidates: List[str] = []
    for root in [folder] + sorted(
            os.path.join(folder, d) for d in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, d))):
        candidates += sorted(
            os.path.join(root, f) for f in os.listdir(root)
            if f.lower().endswith(".sq") and os.path.isfile(os.path.join(root, f)))
    for p in candidates:
        try:
            added.append(register_database(p, registry_path=registry_path,
                                           import_dir=import_dir, log=log))
        except ValueError as exc:
            log(f"Skipped {os.path.basename(p)}: {exc}")
    return added


def unregister_database(name: str, *, registry_path: Optional[str] = None) -> None:
    """Remove a database from the registry (the .sq file itself is never
    deleted — it may be the user's only copy)."""
    entries = [e for e in load_registry(registry_path) if e.get("name") != name]
    save_registry(entries, registry_path)


def set_database_enabled(name: str, enabled: bool, *,
                         registry_path: Optional[str] = None) -> None:
    entries = load_registry(registry_path)
    for e in entries:
        if e.get("name") == name:
            e["enabled"] = bool(enabled)
    save_registry(entries, registry_path)


def enabled_database_paths(registry_path: Optional[str] = None) -> List[str]:
    return [e["path"] for e in load_registry(registry_path)
            if e.get("enabled", True) and os.path.isfile(e.get("path", ""))]


# =============================================================================
# Crystal-system inference (filters): the databases store the Hermann-
# Mauguin symbol as text, so the system is derived from the symbol's
# rotational content — standard H-M reading order (cubic before trigonal
# because m-3m contains a 3; hexagonal before tetragonal is irrelevant as
# 6 and 4 never co-occur as principal axes).
# =============================================================================

CRYSTAL_SYSTEMS = ["cubic", "hexagonal", "trigonal", "tetragonal",
                   "orthorhombic", "monoclinic", "triclinic"]


def crystal_system(spacegroup: str) -> str:
    """Best-effort crystal system from an H-M space-group string; ''
    when the string is empty/unreadable. Decision tree: R lattice ->
    trigonal; primary 6 -> hexagonal; a 3 in the PRIMARY position ->
    trigonal, a 3 anywhere later (the body diagonal) -> cubic (covers
    23, m-3, 432, -43m, m-3m and screw variants like P4132); primary
    4 -> tetragonal; else count 2-fold/mirror directions."""
    s = (spacegroup or "").strip()
    if not s:
        return ""
    body = re.sub(r"\s+", "", s)
    rest = body[1:] if body[:1].upper() in "PABCIFR" and len(body) > 1 else body
    seq = re.sub(r"[-/_:]", "", rest)  # symbol sequence without decorations
    if not seq:
        return ""
    if body[:1].upper() == "R":
        return "trigonal"
    if seq[0] == "6":
        return "hexagonal"
    if "3" in seq:
        return "trigonal" if seq[0] == "3" else "cubic"
    if seq[0] == "4":
        return "tetragonal"
    n_dirs = len(re.findall(r"[2mcdnabe]", seq))
    if n_dirs >= 3:
        return "orthorhombic"
    if seq in ("1", "-1") or rest in ("1", "-1"):
        return "triclinic"
    if n_dirs >= 1:
        return "monoclinic"
    return ""


# =============================================================================
# Search-match
# =============================================================================

@dataclass
class XrdMatch:
    card_id: int
    source: str
    source_code: str
    name: str
    mineral: str
    formula: str
    spacegroup: str
    quality: str
    rir: Optional[float]
    fom: float               # 0..1 figure of merit (geometric-mean coverage)
    cov_card: float          # how much of the card's pattern is in the data
    cov_query: float         # how much of the data the card explains
    n_matched: int
    db: str = ""             # which registered database the card came from
    matched_pairs: List[Tuple[float, float]] = field(default_factory=list)  # (query 2θ, card 2θ)
    d: np.ndarray = field(default_factory=lambda: np.array([]))
    i: np.ndarray = field(default_factory=lambda: np.array([]))


def _dedup_key(source_code: str, name: str, mineral: str, formula: str,
               spacegroup: str, quality: str) -> Tuple[str, str, str, str]:
    """Cards identical in code + composition + symmetry + quality are the
    same reference record, however many databases carry it (user request:
    e.g. the same COD entry present in both an inorganic subset and the
    full COD shows up twice otherwise)."""
    phase = (mineral or name or formula or "").strip().lower()
    return (str(source_code).strip(), phase,
            re.sub(r"\s+", "", (spacegroup or "")).lower(), (quality or "").strip().upper())


def search_match(
    query_two_theta: Sequence[float], query_intensity: Optional[Sequence[float]] = None, *,
    wavelength: float = CU_KA1, tol_two_theta: float = 0.2,
    elements_all: Sequence[str] = (), elements_none: Sequence[str] = (),
    sources: Sequence[str] = (), top_n: int = 40,
    two_theta_range: Optional[Tuple[float, float]] = None,
    max_candidates: int = 20000, db_path: str = XRD_ID_DB_PATH,
    db_paths: Optional[Sequence[str]] = None,
    qualities: Sequence[str] = (), crystal_systems: Sequence[str] = (),
    spacegroup_contains: str = "",
) -> List[XrdMatch]:
    """Rank database cards against a measured peak list.

    query_two_theta/intensity: the measured peaks (equal intensities are
    assumed when none given). tol_two_theta: match window in °2θ.
    elements_all: card must contain ALL of these; elements_none: none.
    sources: restrict to these source tags. two_theta_range: only card
    lines inside the measured range count against the card's coverage.
    db_paths: probe SEVERAL registered databases at once (QualX-style) —
    results are merged, DEDUPLICATED (same code+phase+spacegroup+quality
    kept once, best FoM wins), and re-ranked; each match's .db says which
    file it came from. When db_paths is None, the single db_path is probed.
    qualities / crystal_systems / spacegroup_contains: optional card
    filters (quality codes; systems from CRYSTAL_SYSTEMS; substring of
    the H-M symbol)."""
    paths = [str(p) for p in db_paths] if db_paths is not None else [db_path]
    results: List[XrdMatch] = []
    for p in paths:
        results.extend(_search_match_one(
            query_two_theta, query_intensity, wavelength=wavelength,
            tol_two_theta=tol_two_theta, elements_all=elements_all,
            elements_none=elements_none, sources=sources, top_n=top_n,
            two_theta_range=two_theta_range, max_candidates=max_candidates,
            db_path=p, qualities=qualities, crystal_systems=crystal_systems,
            spacegroup_contains=spacegroup_contains))
    results.sort(key=lambda r: (r.fom, r.n_matched), reverse=True)
    seen: set = set()
    unique: List[XrdMatch] = []
    for r in results:
        key = _dedup_key(r.source_code, r.name, r.mineral, r.formula, r.spacegroup, r.quality)
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique[:top_n]


def _search_match_one(
    query_two_theta: Sequence[float], query_intensity: Optional[Sequence[float]] = None, *,
    wavelength: float = CU_KA1, tol_two_theta: float = 0.2,
    elements_all: Sequence[str] = (), elements_none: Sequence[str] = (),
    sources: Sequence[str] = (), top_n: int = 40,
    two_theta_range: Optional[Tuple[float, float]] = None,
    max_candidates: int = 20000, db_path: str = XRD_ID_DB_PATH,
    qualities: Sequence[str] = (), crystal_systems: Sequence[str] = (),
    spacegroup_contains: str = "",
) -> List[XrdMatch]:
    if not os.path.isfile(db_path):
        raise FileNotFoundError(
            f"No XRD database at {db_path}. Register one in the XRD ID "
            "workspace (Add database…) or build one with build_xrd_database().")
    q_tt = np.asarray(list(query_two_theta), float)
    if len(q_tt) == 0:
        return []
    q_i = (np.asarray(list(query_intensity), float) if query_intensity is not None
           else np.ones_like(q_tt))
    q_i = np.where(np.isfinite(q_i) & (q_i > 0), q_i, 1.0)
    q_d = two_theta_to_d(q_tt, wavelength)

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    # ---- candidate prefilter: cards whose strong lines coincide with the
    # query's strongest peaks (union, ranked by hit count). With a flat
    # intensity list (hand-typed peaks, no spectrum) "strongest 5" would be
    # an arbitrary subset — probe every peak instead so no phase whose
    # lines happen to sit late in the list is silently missed.
    n_probe = len(q_tt) if float(q_i.max()) == float(q_i.min()) else 5
    strongest = np.argsort(q_i)[::-1][:n_probe]
    hits: Dict[int, int] = {}
    for qi in strongest:
        tt = q_tt[qi]
        # Δd from Δ2θ: |dd| = d·cot(θ)·Δθ(rad)/1 … evaluated at this line
        theta = np.radians(tt / 2.0)
        dd = abs(q_d[qi] / np.tan(theta) * np.radians(tol_two_theta / 2.0)) if theta > 0 else 0.01
        dd = max(dd, 1e-4)
        for (cid,) in con.execute("SELECT card_id FROM strong_lines WHERE d BETWEEN ? AND ?",
                                  (q_d[qi] - dd, q_d[qi] + dd)):
            hits[cid] = hits.get(cid, 0) + 1
    if not hits:
        con.close()
        return []
    candidates = sorted(hits, key=lambda c: hits[c], reverse=True)[:max_candidates]

    # ---- chemistry / source filters
    if elements_all:
        keep = None
        for el in elements_all:
            got = {cid for (cid,) in con.execute(
                "SELECT card_id FROM elements WHERE element = ?", (el.strip().capitalize(),))}
            keep = got if keep is None else (keep & got)
        candidates = [c for c in candidates if c in (keep or set())]
    if elements_none:
        drop = set()
        for el in elements_none:
            drop |= {cid for (cid,) in con.execute(
                "SELECT card_id FROM elements WHERE element = ?", (el.strip().capitalize(),))}
        candidates = [c for c in candidates if c not in drop]
    if not candidates:
        con.close()
        return []

    # ---- score
    marks = ",".join("?" * len(candidates))
    src_filter = ""
    params: List[Any] = list(candidates)
    if sources:
        src_filter = f" AND source IN ({','.join('?' * len(sources))})"
        params += list(sources)
    rows = con.execute(
        f"SELECT card_id, source, source_code, name, mineral, formula, spacegroup, quality, rir, d, i "
        f"FROM cards WHERE card_id IN ({marks}){src_filter}", params).fetchall()
    con.close()

    db_label = os.path.splitext(os.path.basename(db_path))[0]
    quality_set = {q.strip().upper() for q in qualities if q.strip()}
    system_set = {c.strip().lower() for c in crystal_systems if c.strip()}
    sg_needle = (spacegroup_contains or "").strip().lower()
    results: List[XrdMatch] = []
    for (cid, source, code, name, mineral, formula, sg, quality, rir, d_text, i_text) in rows:
        if quality_set and (quality or "").strip().upper() not in quality_set:
            continue
        if system_set and crystal_system(sg or "") not in system_set:
            continue
        if sg_needle and sg_needle not in (sg or "").lower():
            continue
        c_d = np.array([float(v) for v in d_text.split(",")], float)
        c_i = np.array([float(v) for v in i_text.split(",")], float)
        c_tt = d_to_two_theta(c_d, wavelength)
        ok = np.isfinite(c_tt)
        if two_theta_range is not None:
            ok &= (c_tt >= two_theta_range[0]) & (c_tt <= two_theta_range[1])
        c_tt, c_i_in = c_tt[ok], c_i[ok]
        if len(c_tt) == 0:
            continue

        # greedy matching, card's strongest lines first (they're pre-sorted)
        matched_pairs: List[Tuple[float, float]] = []
        used_q = np.zeros(len(q_tt), bool)
        matched_ci, matched_qi = 0.0, 0.0
        for tt_c, i_c in zip(c_tt, c_i_in):
            diffs = np.abs(q_tt - tt_c)
            diffs[used_q] = np.inf
            j = int(np.argmin(diffs))
            if diffs[j] <= tol_two_theta:
                used_q[j] = True
                matched_pairs.append((float(q_tt[j]), float(tt_c)))
                matched_ci += float(i_c)
                matched_qi += float(q_i[j])
        if not matched_pairs:
            continue
        cov_card = matched_ci / float(c_i_in.sum()) if c_i_in.sum() > 0 else 0.0
        cov_query = matched_qi / float(q_i.sum()) if q_i.sum() > 0 else 0.0
        fom = float(np.sqrt(cov_card * cov_query))
        results.append(XrdMatch(
            card_id=cid, source=source, source_code=code, name=name or "", mineral=mineral or "",
            formula=formula or "", spacegroup=sg or "", quality=quality or "", rir=rir,
            fom=fom, cov_card=cov_card, cov_query=cov_query, n_matched=len(matched_pairs),
            db=db_label, matched_pairs=matched_pairs, d=c_d, i=c_i,
        ))
    results.sort(key=lambda r: (r.fom, r.n_matched), reverse=True)
    return results[:top_n]


def _rows_to_cards(rows, db_label: str = "") -> List[Dict[str, Any]]:
    out = []
    for (cid, source, code, name, mineral, formula, sg, quality, d_text, i_text) in rows:
        out.append({
            "card_id": cid, "source": source, "source_code": code, "name": name,
            "mineral": mineral, "formula": formula, "spacegroup": sg, "quality": quality,
            "db": db_label,
            "d": np.array([float(v) for v in d_text.split(",")], float),
            "i": np.array([float(v) for v in i_text.split(",")], float),
        })
    return out


def find_cards_by_elements(query: str, *, mode: str = "exact", limit: int = 100,
                           db_path: str = XRD_ID_DB_PATH,
                           db_paths: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Element-set card lookup (user report: text-searching 'TiO2' missed
    cards whose formula is written 'O2 Ti'). The query is parsed as a
    chemical formula; cards are matched on their ELEMENT SET, so formula
    spelling/order/stoichiometry writing don't matter.
    mode='exact'  — card contains exactly these elements and no others
    mode='contains' — card contains at least these elements."""
    import xraydb
    if db_paths is not None:
        out: List[Dict[str, Any]] = []
        for p in db_paths:
            out += find_cards_by_elements(query, mode=mode, limit=limit - len(out), db_path=p)
            if len(out) >= limit:
                break
        return out
    if not os.path.isfile(db_path):
        return []
    try:
        elements = sorted({el.capitalize() for el in xraydb.chemparse(query.strip())})
    except (ValueError, KeyError):
        return []  # not a chemical formula — the text search handles it
    if not elements:
        return []
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    marks = ",".join("?" * len(elements))
    if mode == "exact":
        ids = [r[0] for r in con.execute(
            f"SELECT card_id FROM elements GROUP BY card_id "
            f"HAVING SUM(CASE WHEN element IN ({marks}) THEN 1 ELSE 0 END) = COUNT(DISTINCT element) "
            f"AND COUNT(DISTINCT element) = ?", (*elements, len(elements))).fetchall()]
    else:
        keep = None
        for el in elements:
            got = {r[0] for r in con.execute("SELECT card_id FROM elements WHERE element = ?", (el,))}
            keep = got if keep is None else (keep & got)
        ids = sorted(keep or set())
    ids = ids[:limit]
    if not ids:
        con.close()
        return []
    rows = con.execute(
        f"SELECT card_id, source, source_code, name, mineral, formula, spacegroup, quality, d, i "
        f"FROM cards WHERE card_id IN ({','.join('?' * len(ids))})", ids).fetchall()
    con.close()
    return _rows_to_cards(rows, os.path.splitext(os.path.basename(db_path))[0])


def find_cards_by_text(text: str, *, limit: int = 50, db_path: str = XRD_ID_DB_PATH,
                       db_paths: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Name/mineral/formula substring lookup — the Raman↔XRD bridge uses
    this to pull reference patterns for an already-identified mineral."""
    if db_paths is not None:
        out: List[Dict[str, Any]] = []
        for p in db_paths:
            out += find_cards_by_text(text, limit=limit - len(out), db_path=p)
            if len(out) >= limit:
                break
        return out
    if not os.path.isfile(db_path):
        return []
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    like = f"%{text.strip()}%"
    rows = con.execute(
        "SELECT card_id, source, source_code, name, mineral, formula, spacegroup, quality, d, i "
        "FROM cards WHERE mineral LIKE ? OR name LIKE ? OR formula LIKE ? LIMIT ?",
        (like, like, like, int(limit))).fetchall()
    con.close()
    return _rows_to_cards(rows, os.path.splitext(os.path.basename(db_path))[0])
