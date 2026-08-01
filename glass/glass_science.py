"""
glass_science.py — glass property calculations (framework-agnostic).

Optical basicity Λ:
    Λ = Σ(x_i · n_O,i · Λ_i) / Σ(x_i · n_O,i)
with x_i the molar fraction of oxide i and n_O,i its oxygens per formula
unit (oxygen-weighted mixing, Duffy & Ingram). The per-oxide Λ values are
the RECOMMENDED values (Λrec) of Table B.1 in Rodriguez & McCloy,
PNNL-20184 / EMSP-RPT-003 (2011) — "Optical basicity and nepheline
crystallization in high alumina glasses" — which compiles and reconciles
Duffy & Ingram (1976), Duffy (2002-2006), Dimitrov & Sakka (1996),
Lebouteiller & Courtine (1998), Lenglet (2004), and Mills (1995).

Machine-learning property prediction is delegated to GlassPy's GlassNet
(Cassar, Ceramics International 49 (2023) 36013 — https://glasspy.readthedocs.io),
lazily imported. Further property data sources for manual comparison:
SciGlass (now free, https://github.com/epam/SciGlass), INTERGLAD
(newglass.jp), and glassproperties.com (SciGlass-derived calculators).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Sequence, Tuple

if TYPE_CHECKING:
    import pandas as pd


OPTICAL_BASICITY_SOURCE = (
    "Λrec values: Rodriguez & McCloy, PNNL-20184/EMSP-RPT-003 (2011), Table B.1 "
    "(compiling Duffy & Ingram 1976; Duffy 2002-2006; Dimitrov & Sakka 1996; and others)."
)

# Oxide -> recommended optical basicity Λrec (PNNL-20184 Table B.1)
OPTICAL_BASICITY: Dict[str, float] = {
    "Ag2O": 0.91, "Al2O3": 0.61, "As2O3": 1.01, "As2O5": 0.40, "Au2O3": 1.13,
    "B2O3": 0.40, "BaO": 1.33, "BeO": 0.375, "Bi2O3": 1.19, "CO2": 0.33,
    "CaO": 1.00, "CdO": 0.95, "CeO2": 1.01, "Ce2O3": 1.18, "Cl2O7": 0.27,
    "CoO": 0.98, "Co2O3": 0.96, "Cr2O3": 0.80, "Cs2O": 1.52, "Cu2O": 1.36,
    "CuO": 1.10, "Dy2O3": 1.08, "Eu2O3": 0.95, "Fe2O3": 0.80, "FeO": 0.93,
    "Ga2O3": 0.76, "Gd2O3": 1.18, "GeO2": 0.61, "H2O": 0.40, "HfO2": 0.77,
    "HgO": 1.25, "Ho2O3": 1.04, "In2O3": 1.06, "IrO2": 0.85, "K2O": 1.32,
    "La2O3": 1.18, "Li2O": 0.84, "Lu2O3": 0.97, "MgO": 0.95, "MnO": 0.95,
    "MoO3": 1.07, "N2O5": 0.27, "Na2O": 1.11, "Nb2O5": 1.05, "Nd2O3": 1.19,
    "NiO": 0.92, "P2O5": 0.40, "PbO": 1.18, "PbO2": 1.22, "PdO": 1.19,
    "SO3": 0.33, "Sb2O3": 1.18, "Sc2O3": 0.90, "SeO2": 0.90, "SiO2": 0.48,
    "Sm2O3": 1.14, "SnO2": 0.85, "SrO": 1.08, "Ta2O5": 0.94, "Tb2O3": 0.99,
    "TeO2": 0.93, "ThO2": 0.97, "TiO": 1.30, "TiO2": 0.91, "Tl2O": 1.49,
    "Tl2O3": 1.21, "Tm2O3": 1.00, "UO3": 1.04, "UO2": 0.97, "U3O8": 0.99,
    "V2O5": 1.04, "WO3": 1.05, "Y2O3": 1.00, "Yb2O3": 0.95, "ZnO": 0.80,
    "ZrO2": 0.85,
    # actinides / rare entries from the same table
    "Ac2O3": 1.06, "Am2O3": 1.05, "Bk2O3": 1.05, "Cf2O3": 1.05, "Cm2O3": 1.05,
    "NpO2": 1.01, "PaO2": 1.02, "Tc2O7": 0.86,
}


def _oxide_oxygen_and_mass(formula: str) -> Tuple[float, float]:
    import xraydb
    counts = xraydb.chemparse(formula)
    n_o = float(counts.get("O", 0.0))
    mw = sum(xraydb.atomic_mass(el) * n for el, n in counts.items())
    return n_o, mw


def optical_basicity(components: Sequence[Tuple[str, float]], basis: str = "mol") -> Dict[str, float]:
    """Λ of an oxide mixture. components: [(oxide_formula, fraction)] in
    molar ('mol') or weight ('wt') proportions (normalized internally).
    Returns {'basicity': Λ, 'oxygen_total': Σx·nO} plus per-oxide Λ used.
    Unknown oxides raise with the list of known ones."""
    if not components:
        raise ValueError("Empty composition.")
    num = 0.0
    den = 0.0
    used = {}
    for formula, frac in components:
        formula = formula.strip()
        if formula not in OPTICAL_BASICITY:
            raise ValueError(
                f"No recommended optical basicity for {formula!r}. Known oxides: "
                + ", ".join(sorted(OPTICAL_BASICITY)))
        n_o, mw = _oxide_oxygen_and_mass(formula)
        if n_o <= 0:
            raise ValueError(f"{formula} contains no oxygen.")
        x = float(frac) / mw if basis == "wt" else float(frac)
        lam = OPTICAL_BASICITY[formula]
        num += x * n_o * lam
        den += x * n_o
        used[formula] = lam
    if den <= 0:
        raise ValueError("Composition has zero oxygen content.")
    return {"basicity": num / den, "oxygen_total": den, "per_oxide": used}


def glassnet_available() -> bool:
    # find_spec: "is it installed" without importing glasspy (which pulls in
    # PyTorch — tens of seconds cold). The real import happens on first
    # "GlassNet predict" click.
    import importlib.util
    try:
        return importlib.util.find_spec("glasspy") is not None
    except Exception:
        return False


_GLASSNET_MODEL = None


def glassnet_predict(compositions: "pd.DataFrame"):
    """Predict glass properties with GlassPy's GlassNet (lazy singleton —
    the model load takes seconds). compositions: DataFrame whose columns
    are compound formulas (e.g. SiO2, Na2O) in MOLAR fractions or percent
    (GlassNet normalizes). Returns the full property DataFrame.
    Cite: Cassar (2023), Ceramics International 49, 36013 (GlassNet);
    training data: SciGlass."""
    global _GLASSNET_MODEL
    from glasspy.predict import GlassNet
    if _GLASSNET_MODEL is None:
        _GLASSNET_MODEL = GlassNet()
    return _GLASSNET_MODEL.predict(compositions)


def parse_composition_table(text: str) -> "pd.DataFrame":
    """Parse a pasted composition table: first row = oxide names (any of
    comma/semicolon/tab/space separated), following rows = values; an
    optional first column of sample names is detected non-numerically."""
    import pandas as pd
    rows = [r.strip() for r in text.strip().splitlines() if r.strip()]
    if len(rows) < 2:
        raise ValueError("Paste a header row of oxides plus at least one composition row.")

    def split(row):
        for sep in ("\t", ";", ","):
            if sep in row:
                return [t.strip() for t in row.split(sep) if t.strip()]
        return row.split()

    header = split(rows[0])
    data = [split(r) for r in rows[1:]]
    names = []
    values = []
    has_names = any(not _is_number(d[0]) for d in data if d)
    for k, d in enumerate(data):
        if has_names:
            names.append(d[0])
            d = d[1:]
        else:
            names.append(f"sample{k + 1}")
        values.append([float(v) for v in d])
    cols = header[1:] if has_names and len(header) == len(data[0]) else header
    df = pd.DataFrame(values, columns=cols[:len(values[0])], index=names)
    return df


def _is_number(tok: str) -> bool:
    try:
        float(tok)
        return True
    except (TypeError, ValueError):
        return False
