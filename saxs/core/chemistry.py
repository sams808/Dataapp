from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import math
import re

import xraydb


CLASSICAL_ELECTRON_RADIUS_CM = 2.8179403262e-13
AVOGADRO = 6.02214076e23
HC_KEV_ANG = 12.398419843320026


CAPILLARY_PRESETS = {
    "Custom": {"formula": "SiO2", "density": 2.20, "wall_thickness_mm": 0.01, "outer_diameter_mm": 1.00},
    "Fused silica": {"formula": "SiO2", "density": 2.20, "wall_thickness_mm": 0.01, "outer_diameter_mm": 1.00},
    "Borosilicate": {"formula": "81(SiO2)13(B2O3)4(Na2O)2(Al2O3)", "density": 2.23, "wall_thickness_mm": 0.01, "outer_diameter_mm": 1.00, "composition_mode": "mixture"},
    "Quartz": {"formula": "SiO2", "density": 2.65, "wall_thickness_mm": 0.01, "outer_diameter_mm": 1.00},
    "(CS)Special Glass": {"formula": "SiO2", "density": 2.20, "wall_thickness_mm": 0.01, "outer_diameter_mm": 1.00, "manual_mu_cm_inv": {"CuK": 110.8, "MoK": 11.95}},
    "(CS)Boron-rich": {"formula": "SiO2", "density": 2.20, "wall_thickness_mm": 0.01, "outer_diameter_mm": 1.00, "manual_mu_cm_inv": {"CuK": 71.0, "MoK": 7.35}},
    "(CS)Quartz": {"formula": "SiO2", "density": 2.65, "wall_thickness_mm": 0.01, "outer_diameter_mm": 1.00, "manual_mu_cm_inv": {"CuK": 75.8, "MoK": 8.20}},
}


DEFAULT_OXIDE_MOLAR_VOLUMES_CM3_MOL = {
    "Al2O3": 40.4,
    "B2O3": 28.25,
    "BaO": 22.0,
    "BeO": 7.8,
    "Bi2O3": 45.0,
    "CaO": 14.4,
    "CdO": 17.6,
    "CoO": 14.5,
    "Cs2O": 47.0,
    "FeO": 16.5,
    "Ga2O3": 42.5,
    "HfO2": 27.5,
    "K2O": 34.1,
    "Li2O": 11.0,
    "La2O3": 40.0,
    "MgO": 12.5,
    "MnO": 17.2,
    "Na2O": 20.2,
    "Nb2O5": 56.0,
    "NiO": 13.0,
    "P2O5": 61.7,
    "PbO": 22.25,
    "Rb2O": 43.0,
    "Sb2O3": 47.0,
    "Sc2O3": 28.0,
    "SiO2": 26.675,
    "SnO2": 28.8,
    "SrO": 17.5,
    "Ta2O3": 52.0,
    "ThO2": 31.7,
    "TiO2": 20.75,
    "Tl2O": 63.0,
    "Y2O3": 35.0,
    "ZnO": 14.5,
    "ZrO2": 23.0,
}


@dataclass
class CompositionResult:
    mode: str
    raw_input: str
    basis: str
    formula_text: str
    element_counts: Dict[str, float]
    mass_fractions: Dict[str, float]
    molecular_weight_g_mol: float

    def to_json(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class AbsorptionResult:
    energy_ev: float
    wavelength_ang: float
    density_g_cm3: float
    mu_linear_cm_inv: float
    mu_linear_mm_inv: float
    mu_mass_cm2_g: float
    attenuation_length_mm: float
    transmission_for_1mm: float
    electron_density_cm3: float
    scattering_length_density_cm2: float
    delta: float
    beta: float

    def to_json(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class CapillaryConfig:
    preset: str = "Fused silica"
    composition_mode: str = "formula"
    composition_text: str = "SiO2"
    mixture_basis: str = "molar"
    density_g_cm3: float = 2.20
    wall_thickness_mm: float = 0.01
    outer_diameter_mm: float = 1.00
    manual_mu_linear_mm_inv: Optional[float] = None

    @property
    def inner_diameter_mm(self) -> float:
        return max(0.0, float(self.outer_diameter_mm) - 2.0 * float(self.wall_thickness_mm))

    def to_json(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["inner_diameter_mm"] = self.inner_diameter_mm
        return payload


@dataclass
class SamplePhysicsConfig:
    composition_mode: str = "formula"
    composition_text: str = "Si47Al19Bi10Na48B29P20O208"
    mixture_basis: str = "molar"
    density_g_cm3: float = 2.8
    packing_fraction: float = 1.0
    manual_transmission: Optional[float] = None
    manual_mu_linear_mm_inv: Optional[float] = None
    notes: str = ""

    def effective_density(self) -> float:
        return max(0.0, float(self.density_g_cm3) * float(self.packing_fraction))

    def to_json(self) -> Dict[str, object]:
        return asdict(self)


_TOKEN_SPLIT_RE = re.compile(r"\s*(?:\+|;|\n|\r|\|)\s*")
_CLEAN_TOKEN_RE = re.compile(r"\b(?:mol%|mole%|wt%|weight%|mass%|vol%)\b", re.IGNORECASE)


def sanitize_formula_text(text: str) -> str:
    clean = text.strip()
    clean = clean.replace("·", ".")
    clean = clean.replace("•", ".")
    clean = re.sub(r"\s+", "", clean)
    return clean


def _molar_mass_from_counts(counts: Dict[str, float]) -> float:
    total = 0.0
    for symbol, count in counts.items():
        total += xraydb.atomic_mass(symbol) * count
    return total


def _formula_to_mass_fractions(formula: str) -> Tuple[Dict[str, float], Dict[str, float], float]:
    counts = xraydb.chemparse(formula)
    if not counts:
        raise ValueError(f"Could not parse formula: {formula}")
    mw = _molar_mass_from_counts(counts)
    fractions: Dict[str, float] = {}
    for symbol, count in counts.items():
        fractions[symbol] = xraydb.atomic_mass(symbol) * count / mw
    return counts, fractions, mw


def parse_mixture_expression(text: str) -> List[Tuple[str, float]]:
    clean = _CLEAN_TOKEN_RE.sub("", text.strip())
    if not clean:
        raise ValueError("Mixture expression is empty.")

    compact = re.sub(r"\s+", "", clean)
    parsed: List[Tuple[str, float]] = []

    # Preferred syntax for this app: 35(SiO2)25(Al2O3)50(Na2O)
    paren_tokens = re.findall(r"([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\(([^()]+)\)", compact)
    if paren_tokens:
        rebuilt = "".join(f"{val}({formula})" for val, formula in paren_tokens)
        if rebuilt == compact:
            for value_txt, formula_txt in paren_tokens:
                parsed.append((sanitize_formula_text(formula_txt), float(value_txt)))
            return parsed

    parts = [p.strip() for p in _TOKEN_SPLIT_RE.split(clean) if p.strip()]
    if not parts:
        raise ValueError("No mixture tokens found.")

    for part in parts:
        # Accept: '47 SiO2', 'SiO2 47', 'SiO2=47', 'SiO2:47', or '47(SiO2)'
        m = re.match(r"^([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\(([^()]+)\)$", part)
        if m:
            value = float(m.group(1))
            formula = sanitize_formula_text(m.group(2))
            parsed.append((formula, value))
            continue

        m = re.match(r"^([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\s*([A-Za-z][A-Za-z0-9()\.\[\]]+)$", part)
        if m:
            value = float(m.group(1))
            formula = sanitize_formula_text(m.group(2))
            parsed.append((formula, value))
            continue

        m = re.match(r"^([A-Za-z][A-Za-z0-9()\.\[\]]+)\s*[:=]\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)$", part)
        if m:
            formula = sanitize_formula_text(m.group(1))
            value = float(m.group(2))
            parsed.append((formula, value))
            continue

        m = re.match(r"^([A-Za-z][A-Za-z0-9()\.\[\]]+)\s+([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)$", part)
        if m:
            formula = sanitize_formula_text(m.group(1))
            value = float(m.group(2))
            parsed.append((formula, value))
            continue

        raise ValueError(
            "Could not parse mixture token. Use patterns like '35(SiO2)25(Al2O3)', '47 SiO2', 'SiO2=47', or 'SiO2:47'. "
            f"Problematic token: {part!r}"
        )

    if not parsed:
        raise ValueError("No mixture components parsed.")
    return parsed


def _format_stoich(value: float, decimals: int = 6) -> str:
    v = float(value)
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    txt = f"{v:.{decimals}f}".rstrip("0").rstrip(".")
    return txt if txt else "0"


def _parse_formula_preserve_order(formula: str) -> List[Tuple[str, float]]:
    formula = sanitize_formula_text(formula)
    tokens = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)", formula)
    if not tokens:
        raise ValueError(f"Could not parse formula: {formula}")
    parsed: List[Tuple[str, float]] = []
    for sym, count_txt in tokens:
        parsed.append((sym, float(count_txt) if count_txt else 1.0))
    return parsed


def _canonical_oxide_map() -> Dict[str, Tuple[str, float, float]]:
    mapping: Dict[str, Tuple[str, float, float]] = {}
    for oxide in DEFAULT_OXIDE_MOLAR_VOLUMES_CM3_MOL.keys():
        counts = xraydb.chemparse(oxide)
        if not counts or "O" not in counts:
            continue
        cations = [(sym, cnt) for sym, cnt in counts.items() if sym != "O"]
        if len(cations) != 1:
            continue
        cat, cat_n = cations[0]
        oxy_n = float(counts.get("O", 0.0))
        if cat not in mapping:
            mapping[cat] = (oxide, float(cat_n), oxy_n)
    return mapping


def mixture_to_formula_text(text: str, basis: str = "molar") -> str:
    if (basis or "molar").strip().lower() != "molar":
        raise ValueError("Formula conversion from mixture expects molar basis.")
    items = parse_mixture_expression(text)
    counts: Dict[str, float] = {}
    cation_order: List[str] = []
    for oxide, amount in items:
        parsed = xraydb.chemparse(oxide)
        if not parsed:
            raise ValueError(f"Could not parse oxide: {oxide}")
        for sym, n in parsed.items():
            counts[sym] = counts.get(sym, 0.0) + float(n) * float(amount)
            if sym != "O" and sym not in cation_order:
                cation_order.append(sym)
    order = cation_order + (["O"] if "O" in counts else [])
    parts = []
    for sym in order:
        val = counts.get(sym)
        if val is None:
            continue
        parts.append(f"{sym}{_format_stoich(val)}")
    return "".join(parts) if parts else "SiO2"


def formula_to_mixture_text(text: str) -> str:
    formula = sanitize_formula_text(text)
    ordered = _parse_formula_preserve_order(formula)
    counts = xraydb.chemparse(formula)
    if not counts:
        raise ValueError(f"Could not parse formula: {formula}")
    oxide_map = _canonical_oxide_map()
    cation_order = [sym for sym, _ in ordered if sym != "O"]
    tokens: List[str] = []
    oxygen_expected = 0.0
    oxygen_count = float(counts.get("O", 0.0))
    for sym in cation_order:
        if sym not in oxide_map:
            raise ValueError(f"No canonical oxide found for element {sym}. Add an oxide for it to DEFAULT_OXIDE_MOLAR_VOLUMES_CM3_MOL.")
        oxide, cat_n, oxy_n = oxide_map[sym]
        amount = float(counts.get(sym, 0.0)) / float(cat_n)
        oxygen_expected += amount * oxy_n
        tokens.append(f"{_format_stoich(amount)}({oxide})")
    if abs(oxygen_expected - oxygen_count) > max(1e-6, 1e-3 * max(oxygen_count, 1.0)):
        raise ValueError(
            f"Formula oxygen count is not consistent with canonical oxide mapping: O={oxygen_count:g} in formula vs O={oxygen_expected:g} implied by oxides."
        )
    return "".join(tokens) if tokens else "100(SiO2)"


def convert_composition_text(text: str, from_mode: str, to_mode: str, from_basis: str = "molar") -> str:
    raw = (text or "").strip()
    if not raw:
        return "SiO2" if to_mode == "formula" else "100(SiO2)"
    from_mode = (from_mode or "formula").strip().lower()
    to_mode = (to_mode or "formula").strip().lower()
    if from_mode == to_mode:
        return sanitize_formula_text(raw) if to_mode == "formula" else raw
    try:
        if from_mode == "mixture" and to_mode == "formula":
            return mixture_to_formula_text(raw, basis=from_basis)
        if from_mode == "formula" and to_mode == "mixture":
            return formula_to_mixture_text(raw)
        comp = parse_composition(raw, mode=from_mode, basis=from_basis)
        return comp.formula_text if to_mode == "formula" else raw
    except Exception:
        return sanitize_formula_text(raw) if to_mode == "formula" else raw

def parse_composition(text: str, mode: str = "formula", basis: str = "molar") -> CompositionResult:
    mode = (mode or "formula").strip().lower()
    basis = (basis or "molar").strip().lower()
    raw = text.strip()
    if not raw:
        raise ValueError("Composition cannot be empty.")

    if mode == "formula":
        formula = sanitize_formula_text(raw)
        counts, mass_fractions, mw = _formula_to_mass_fractions(formula)
        return CompositionResult(
            mode=mode,
            raw_input=raw,
            basis="formula",
            formula_text=formula,
            element_counts=counts,
            mass_fractions=mass_fractions,
            molecular_weight_g_mol=mw,
        )

    if mode != "mixture":
        raise ValueError(f"Unknown composition mode: {mode}")

    items = parse_mixture_expression(raw)
    if basis not in {"molar", "mass"}:
        raise ValueError("Mixture basis must be 'molar' or 'mass'.")

    mixture_counts: Dict[str, float] = {}
    element_mass: Dict[str, float] = {}
    formula_parts: List[str] = []

    if basis == "molar":
        total_moles_mass = 0.0
        for formula, amount in items:
            counts, _, mw = _formula_to_mass_fractions(formula)
            total_moles_mass += amount * mw
            formula_parts.append(f"{formula}:{amount}")
            for symbol, count in counts.items():
                mixture_counts[symbol] = mixture_counts.get(symbol, 0.0) + count * amount
                element_mass[symbol] = element_mass.get(symbol, 0.0) + xraydb.atomic_mass(symbol) * count * amount
        total_mass = total_moles_mass
    else:
        total_mass = 0.0
        for formula, amount in items:
            counts, fractions, _ = _formula_to_mass_fractions(formula)
            formula_parts.append(f"{formula}:{amount}")
            total_mass += amount
            for symbol, frac in fractions.items():
                element_mass[symbol] = element_mass.get(symbol, 0.0) + frac * amount
            for symbol, count in counts.items():
                mixture_counts[symbol] = mixture_counts.get(symbol, 0.0) + count * amount

    if total_mass <= 0:
        raise ValueError("Mixture total must be > 0.")

    mass_fractions = {symbol: mass / total_mass for symbol, mass in element_mass.items()}

    # pseudo molecular weight, helpful only as summary. For mixtures it is normalized to 1 mole-equivalent mass if basis=molar.
    mw = 0.0
    for symbol, frac in mass_fractions.items():
        mw += frac * xraydb.atomic_mass(symbol)

    return CompositionResult(
        mode=mode,
        raw_input=raw,
        basis=basis,
        formula_text=" + ".join(formula_parts),
        element_counts=mixture_counts,
        mass_fractions=mass_fractions,
        molecular_weight_g_mol=mw,
    )


def effective_formula_for_xraydb(comp: CompositionResult) -> str:
    """
    Return the most appropriate string for xraydb.material_mu.

    For true formulas, we return the formula as-is.
    For mixtures, xraydb cannot ingest arbitrary weighted mixtures directly,
    so callers should use mass_fractions with elemental mu values.
    """
    if comp.mode == "formula":
        return comp.formula_text
    raise ValueError("Mixture compositions must be evaluated via elemental mass-fraction mixing.")


def mixture_mu_mass_from_fractions(mass_fractions: Dict[str, float], energy_ev: float, kind: str = "total") -> float:
    total = 0.0
    for symbol, frac in mass_fractions.items():
        total += float(frac) * float(xraydb.mu_elam(symbol, energy_ev, kind=kind))
    return total


def _electron_density_from_composition(comp: CompositionResult, density_g_cm3: float) -> float:
    # electrons / cm^3 = density * N_A * sum_i(w_i Z_i / A_i)
    weighted = 0.0
    for symbol, mass_fraction in comp.mass_fractions.items():
        z = xraydb.atomic_number(symbol)
        a = xraydb.atomic_mass(symbol)
        weighted += mass_fraction * z / a
    return density_g_cm3 * AVOGADRO * weighted


def calculate_absorption(
    composition_text: str,
    density_g_cm3: float,
    energy_ev: Optional[float] = None,
    wavelength_ang: Optional[float] = None,
    mode: str = "formula",
    basis: str = "molar",
    kind: str = "total",
) -> Tuple[CompositionResult, AbsorptionResult]:
    if energy_ev is None and wavelength_ang is None:
        wavelength_ang = 1.5418
    if energy_ev is None:
        energy_ev = HC_KEV_ANG * 1000.0 / float(wavelength_ang)
    if wavelength_ang is None:
        wavelength_ang = HC_KEV_ANG * 1000.0 / float(energy_ev)

    comp = parse_composition(composition_text, mode=mode, basis=basis)
    if density_g_cm3 <= 0:
        raise ValueError("Density must be > 0.")

    if comp.mode == "formula":
        mu_linear_cm_inv = float(xraydb.material_mu(comp.formula_text, energy_ev, density=float(density_g_cm3), kind=kind))
    else:
        mu_mass = mixture_mu_mass_from_fractions(comp.mass_fractions, energy_ev, kind=kind)
        mu_linear_cm_inv = float(mu_mass * density_g_cm3)

    mu_mass_cm2_g = mu_linear_cm_inv / density_g_cm3
    attenuation_length_mm = 10.0 / mu_linear_cm_inv if mu_linear_cm_inv > 0 else math.inf
    transmission_for_1mm = math.exp(-(mu_linear_cm_inv / 10.0) * 1.0)
    electron_density = _electron_density_from_composition(comp, density_g_cm3)
    scattering_length_density_cm2 = CLASSICAL_ELECTRON_RADIUS_CM * electron_density

    try:
        if comp.mode == "formula":
            delta, beta, _ = xraydb.xray_delta_beta(comp.formula_text, density_g_cm3, energy_ev)
        else:
            # Approximate mixture as a weighted sum of elemental delta/beta contributions.
            delta = 0.0
            beta = 0.0
            for symbol, frac in comp.mass_fractions.items():
                # Convert each element weight fraction into partial density.
                partial_density = density_g_cm3 * frac
                d_i, b_i, _ = xraydb.xray_delta_beta(symbol, partial_density, energy_ev)
                delta += d_i
                beta += b_i
    except Exception:
        delta = float("nan")
        beta = float("nan")

    absres = AbsorptionResult(
        energy_ev=float(energy_ev),
        wavelength_ang=float(wavelength_ang),
        density_g_cm3=float(density_g_cm3),
        mu_linear_cm_inv=float(mu_linear_cm_inv),
        mu_linear_mm_inv=float(mu_linear_cm_inv) / 10.0,
        mu_mass_cm2_g=float(mu_mass_cm2_g),
        attenuation_length_mm=float(attenuation_length_mm),
        transmission_for_1mm=float(transmission_for_1mm),
        electron_density_cm3=float(electron_density),
        scattering_length_density_cm2=float(scattering_length_density_cm2),
        delta=float(delta),
        beta=float(beta),
    )
    return comp, absres



def density_from_molar_volumes(composition_text: str, mode: str = "mixture") -> float:
    """Estimate density in g/cm^3 from a molar-mixture composition using built-in oxide molar volumes."""
    if (mode or "mixture").strip().lower() != "mixture":
        raise ValueError("Density from molar volumes currently works only in mixture mode.")

    items = parse_mixture_expression(composition_text)
    if not items:
        raise ValueError("Mixture expression is empty.")

    total = sum(max(0.0, float(amount)) for _, amount in items)
    if total <= 0:
        raise ValueError("Mixture amounts sum to zero.")

    total_mass = 0.0
    total_volume = 0.0
    missing = []
    for formula, amount in items:
        frac = float(amount) / total
        clean_formula = sanitize_formula_text(formula)
        counts = xraydb.chemparse(clean_formula)
        if not counts:
            raise ValueError(f"Could not parse oxide formula for density calculation: {clean_formula}")
        mw = _molar_mass_from_counts(counts)
        molar_vol = DEFAULT_OXIDE_MOLAR_VOLUMES_CM3_MOL.get(clean_formula)
        if molar_vol is None:
            missing.append(clean_formula)
            continue
        total_mass += frac * float(mw)
        total_volume += frac * float(molar_vol)

    if missing:
        missing_txt = ", ".join(sorted(set(missing)))
        raise ValueError(f"Missing default molar volumes for: {missing_txt}")
    if total_volume <= 0:
        raise ValueError("Computed total molar volume is not positive.")
    return total_mass / total_volume


def energy_from_wavelength_ang(wavelength_ang: float) -> float:
    return HC_KEV_ANG * 1000.0 / float(wavelength_ang)


def wavelength_from_energy_ev(energy_ev: float) -> float:
    return HC_KEV_ANG * 1000.0 / float(energy_ev)


def theoretical_empty_transmission(capillary_mu_mm_inv: float, wall_thickness_mm: float) -> float:
    total_wall = 2.0 * max(0.0, wall_thickness_mm)
    return math.exp(-max(0.0, capillary_mu_mm_inv) * total_wall)


def theoretical_sample_transmission(
    capillary_mu_mm_inv: float,
    wall_thickness_mm: float,
    sample_mu_mm_inv: float,
    inner_diameter_mm: float,
) -> float:
    return math.exp(-max(0.0, capillary_mu_mm_inv) * 2.0 * max(0.0, wall_thickness_mm) - max(0.0, sample_mu_mm_inv) * max(0.0, inner_diameter_mm))
