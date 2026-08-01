# cif_tools.py
# --------------------------------------------------------------------------------------
# Generic, dependency-light CIF utilities (parsing + Bragg peak generation + caching)
# --------------------------------------------------------------------------------------
from __future__ import annotations

import os, json, math, hashlib
from typing import List, Optional

# ------------------------------- parsing helpers -------------------------------

def _clean_number(val: Optional[str]) -> Optional[float]:
    if val is None:
        return None
    v = val.strip()
    v = v.split("#", 1)[0].strip()           # remove trailing comments
    # drop uncertainty like "3.456(12)"
    i1 = v.find("(")
    if i1 != -1:
        v = v[:i1].strip()
    v = v.replace("?", "").strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None

def parse_cif_generic(path: str):
    """
    Parse a reasonably standard CIF and return:
      (a, b, c, alpha, beta, gamma, wavelength)
    Falls back to Cu Kα (1.5406 Å) when wavelength absent.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    tags = {}
    loops = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i].strip()
        low = line.lower()

        # loop_
        if low.startswith("loop_"):
            loop_tags, loop_rows = [], []
            i += 1
            while i < n and lines[i].lstrip().startswith("_"):
                loop_tags.append(lines[i].strip())
                i += 1
            while i < n and lines[i].strip() and not lines[i].lstrip().startswith("_") and not lines[i].lower().startswith("loop_"):
                loop_rows.append(lines[i].strip())
                i += 1
            loops.append((loop_tags, loop_rows))
            continue

        # _tag value
        if low.startswith("_"):
            parts = line.split(None, 1)
            tag = parts[0].lower()
            val = parts[1].strip() if len(parts) > 1 else ""
            tags[tag] = val
            i += 1
            continue

        i += 1

    a     = _clean_number(tags.get("_cell_length_a"))
    b     = _clean_number(tags.get("_cell_length_b"))
    c     = _clean_number(tags.get("_cell_length_c"))
    alpha = _clean_number(tags.get("_cell_angle_alpha"))
    beta  = _clean_number(tags.get("_cell_angle_beta"))
    gamma = _clean_number(tags.get("_cell_angle_gamma"))

    wavelength = _clean_number(tags.get("_diffrn_radiation_wavelength"))
    if wavelength is None:
        for loop_tags, loop_rows in loops:
            low_tags = [t.lower() for t in loop_tags]
            if "_diffrn_radiation_wavelength" in low_tags:
                idx = low_tags.index("_diffrn_radiation_wavelength")
                if loop_rows:
                    first = loop_rows[0].split()
                    if len(first) > idx:
                        wavelength = _clean_number(first[idx])
                break
    if wavelength is None:
        wavelength = 1.5406  # Cu Kα default (Å)

    if None in (a, b, c, alpha, beta, gamma):
        raise ValueError("Missing lattice parameters in CIF.")

    return a, b, c, alpha, beta, gamma, wavelength

# --------------------------- geometry + Bragg peaks ---------------------------

def _d_triclinic(a, b, c, alpha, beta, gamma, h, k, l) -> Optional[float]:
    ar = math.radians(alpha)
    br = math.radians(beta)
    gr = math.radians(gamma)
    ca, cb, cg = math.cos(ar), math.cos(br), math.cos(gr)

    # direct metric tensor G
    G = [
        [a*a,     a*b*cg,  a*c*cb],
        [a*b*cg,  b*b,     b*c*ca],
        [a*c*cb,  b*c*ca,  c*c   ],
    ]
    detG = (
        G[0][0]*(G[1][1]*G[2][2] - G[1][2]*G[2][1])
        - G[0][1]*(G[1][0]*G[2][2] - G[1][2]*G[2][0])
        + G[0][2]*(G[1][0]*G[2][1] - G[1][1]*G[2][0])
    )
    if abs(detG) < 1e-18:
        return None

    invG = [[0.0]*3 for _ in range(3)]
    invG[0][0] =  (G[1][1]*G[2][2] - G[1][2]*G[2][1]) / detG
    invG[0][1] = -(G[0][1]*G[2][2] - G[0][2]*G[2][1]) / detG
    invG[0][2] =  (G[0][1]*G[1][2] - G[0][2]*G[1][1]) / detG
    invG[1][0] = -(G[1][0]*G[2][2] - G[1][2]*G[2][0]) / detG
    invG[1][1] =  (G[0][0]*G[2][2] - G[0][2]*G[2][0]) / detG
    invG[1][2] = -(G[0][0]*G[1][2] - G[0][2]*G[1][0]) / detG
    invG[2][0] =  (G[1][0]*G[2][1] - G[1][1]*G[2][0]) / detG
    invG[2][1] = -(G[0][0]*G[2][1] - G[0][1]*G[2][0]) / detG
    invG[2][2] =  (G[0][0]*G[1][1] - G[0][1]*G[1][0]) / detG

    v = [h, k, l]
    one_over_d2 = (
        v[0]*(invG[0][0]*v[0] + invG[0][1]*v[1] + invG[0][2]*v[2]) +
        v[1]*(invG[1][0]*v[0] + invG[1][1]*v[1] + invG[1][2]*v[2]) +
        v[2]*(invG[2][0]*v[0] + invG[2][1]*v[1] + invG[2][2]*v[2])
    )
    if one_over_d2 <= 0:
        return None
    return 1.0 / math.sqrt(one_over_d2)

def _cache_dir() -> str:
    d = os.path.join(os.path.expanduser("~"), ".raman_cache", "cif")
    os.makedirs(d, exist_ok=True); return d

def _cache_key(path: str) -> str:
    st = os.stat(path)
    sig = f"{path}|{st.st_size}|{int(st.st_mtime)}"
    return hashlib.sha1(sig.encode("utf-8")).hexdigest()

def load_cached_peaks(path: str):
    key = _cache_key(path)
    fp = os.path.join(_cache_dir(), f"{key}.json")
    if not os.path.isfile(fp):
        return None
    try:
        with open(fp, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj.get("peaks")  # [[two_theta, h, k, l, d], ...]
    except Exception:
        return None

def save_cached_peaks(path: str, peaks) -> None:
    key = _cache_key(path)
    fp = os.path.join(_cache_dir(), f"{key}.json")
    data = {"peaks": [[tt, h, k, l, d] for (tt, (h,k,l), d) in peaks]}
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(data, f)

def bragg_peaks_from_cif_generic(path: str, two_theta_max: float = 80.0, hkl_max: int = 6, use_cache: bool=True):
    """
    Return a sorted list of peaks: [(2theta_deg, (h,k,l), d_A), ...]
    """
    if use_cache:
        cached = load_cached_peaks(path)
        if cached:
            return [(tt, (h,k,l), d) for (tt, h, k, l, d) in cached]

    a, b, c, alpha, beta, gamma, wavelength = parse_cif_generic(path)
    # Full +/-hkl_max range for every index: for non-orthogonal cells (nonzero
    # cos(alpha)/cos(beta)/cos(gamma) cross-terms in _d_triclinic), mixed-sign
    # combinations like (h,k,-l) give different d-spacings than (h,k,l) — the
    # old non-negative-only octant silently missed those reflections for
    # triclinic/monoclinic structures. (h,k,l) and (-h,-k,-l) always share the
    # same |d| (Friedel pairs), so dedupe on rounded d instead of restricting
    # the search range, which would risk reintroducing the same coverage gap.
    seen_d: dict[float, tuple] = {}
    for h in range(-hkl_max, hkl_max + 1):
        for k in range(-hkl_max, hkl_max + 1):
            for l in range(-hkl_max, hkl_max + 1):
                if h == k == l == 0:
                    continue
                d = _d_triclinic(a, b, c, alpha, beta, gamma, h, k, l)
                if not d:
                    continue
                arg = wavelength / (2.0 * d)
                if arg > 1.0:
                    continue
                theta = math.degrees(math.asin(arg))
                tt = 2.0 * theta
                if tt > two_theta_max:
                    continue
                key = round(d, 6)
                if key not in seen_d:
                    seen_d[key] = (tt, (h, k, l), d)
    out = list(seen_d.values())
    out.sort(key=lambda t: t[0])
    if use_cache:
        try:
            save_cached_peaks(path, out)
        except Exception:
            pass
    return out

# -------------------------- misc convenience utils --------------------------

def list_cif_files_case_insensitive(folder: str) -> List[str]:
    """Return sorted list of all *.cif files in folder (case-insensitive)."""
    paths = []
    try:
        for entry in os.scandir(folder):
            if entry.is_file() and entry.name.lower().endswith(".cif"):
                paths.append(entry.path)
    except Exception:
        pass
    return sorted(paths)
