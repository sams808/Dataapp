"""
xas_mass.py — sample-mass calculator for XAS (the Hephaestus 'formula'
tool, rebuilt for how this lab actually works: oxide compositions in mol%
or wt%, not just single formulas — see the Bi L3 easyXAFS spreadsheets).

Physics (xraydb mass-attenuation, cm²/g):
  compound μ/ρ(E) = Σ_elements w_i · (μ/ρ)_i(E)
  For a pellet of area A: total absorption μ·t = (μ/ρ) · m/A
  → mass for a target μt: m = target · A / (μ/ρ)
  Edge step Δ(μ/ρ) = μ/ρ(E₀+offset) − μ/ρ(E₀−offset), offset user-selected
  (default 3 eV, matching Hephaestus' own "a few eV above the edge"
  convention; μ/ρ barely moves between +3 eV and +50 eV once you're
  clearly past the jump — a large mass mismatch against another program
  is far more likely a tabulated-edge-energy disagreement than this
  choice of offset. See tests/test_saxs_and_mass.py for the worked
  SeO2/Se-K cross-check that established this).
Hephaestus' rules of thumb: total μt ≈ 2.5 above the edge and edge step
Δμt ≈ 1 for transmission.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import xraydb


def parse_components(text: str) -> List[Tuple[str, float]]:
    """Composition text → [(formula, fraction), ...]. One component per
    line (or ';'-separated): 'SiO2 58.8' — a lone formula gets fraction 1.
    Fractions are normalized later, so mol% / wt% / mole fractions all
    work as-is."""
    out: List[Tuple[str, float]] = []
    for raw in text.replace(";", "\n").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        parts = raw.split()
        formula = parts[0]
        xraydb.chemparse(formula)  # raises on nonsense early
        frac = float(parts[1]) if len(parts) > 1 else 1.0
        if frac < 0:
            raise ValueError(f"Negative fraction for {formula}")
        out.append((formula, frac))
    if not out:
        raise ValueError("Empty composition.")
    return out


def element_mass_fractions(components: List[Tuple[str, float]], basis: str = "mol") -> Dict[str, float]:
    """Element → mass fraction for a mixture of components given in molar
    ('mol') or weight ('wt') proportions."""
    elem_mass: Dict[str, float] = {}
    for formula, frac in components:
        counts = xraydb.chemparse(formula)
        mw = sum(xraydb.atomic_mass(el) * n for el, n in counts.items())
        comp_mass = frac * mw if basis == "mol" else frac
        for el, n in counts.items():
            elem_mass[el] = elem_mass.get(el, 0.0) + comp_mass * (xraydb.atomic_mass(el) * n / mw)
    total = sum(elem_mass.values())
    if total <= 0:
        raise ValueError("Composition has zero total mass.")
    return {el: m / total for el, m in elem_mass.items()}


def compound_mu_rho(mass_fractions: Dict[str, float], energy_ev: float) -> float:
    """Mass attenuation μ/ρ of the mixture at energy_ev, in cm²/g."""
    return float(sum(w * xraydb.mu_elam(el, energy_ev) for el, w in mass_fractions.items()))


@dataclass
class MassReport:
    edge_energy_ev: float
    edge_offset_ev: float        # the "eV above/below edge" used below
    mu_rho_above: float          # cm²/g at E0+edge_offset_ev
    edge_step_mu_rho: float      # Δ(μ/ρ) across the edge, cm²/g
    absorber_fraction: float     # mass fraction of the absorbing element
    pellet_area_cm2: float
    mass_mut_1_mg: float         # mass for total μt = 1
    mass_mut_25_mg: float        # mass for total μt = 2.5 (Hephaestus target)
    mass_step_1_mg: float        # mass for edge step Δμt = 1
    target_mut: float            # the user's own target absorption length, in μt (t / (1/μ))
    mass_target_mut_mg: float    # mass for that target
    unit_absorption_length_um: float | None  # 1/μ for the pure compound (needs density)

    def text(self, element: str, edge: str, diameter_mm: float) -> str:
        lines = [
            f"{element} {edge} edge: E₀ = {self.edge_energy_ev:.1f} eV",
            f"μ/ρ (E₀+{self.edge_offset_ev:g} eV) = {self.mu_rho_above:.2f} cm²/g   edge step Δ(μ/ρ) = {self.edge_step_mu_rho:.2f} cm²/g",
            f"{element} mass fraction in the sample: {100 * self.absorber_fraction:.2f} %",
            f"Pellet ⌀ {diameter_mm:g} mm (A = {self.pellet_area_cm2:.3f} cm²):",
            f"  total μt = 1.0  →  {self.mass_mut_1_mg:.1f} mg",
            f"  total μt = 2.5  →  {self.mass_mut_25_mg:.1f} mg   (Hephaestus transmission target)",
        ]
        # Only add a separate line when the user's target differs from the
        # two reference values already shown above — otherwise it's a
        # redundant duplicate of a line that's already there.
        if abs(self.target_mut - 1.0) > 1e-9 and abs(self.target_mut - 2.5) > 1e-9:
            lines.append(f"  total μt = {self.target_mut:g}  →  {self.mass_target_mut_mg:.1f} mg   (your target)")
        lines.append(f"  edge step Δμt = 1.0  →  {self.mass_step_1_mg:.1f} mg")
        if self.edge_step_mu_rho > 0 and self.mass_step_1_mg > self.mass_mut_25_mg * 2:
            lines.append("  ⚠ dilute sample: reaching Δμt=1 exceeds total μt≈5 — consider fluorescence mode.")
        if self.unit_absorption_length_um:
            lines.append(f"Bulk absorption length at E₀+{self.edge_offset_ev:g} eV: {self.unit_absorption_length_um:.1f} µm")
        return "\n".join(lines)


def sample_mass_report(
    composition_text: str, element: str, edge: str = "K", *,
    basis: str = "mol", pellet_diameter_mm: float = 13.0,
    density_g_cm3: float | None = None, target_mut: float = 2.5,
    edge_offset_ev: float = 3.0,
) -> MassReport:
    """target_mut: the sample thickness YOU want, expressed in absorption
    lengths (μt = t / (1/μ)) — e.g. 2.5 for Hephaestus' transmission rule
    of thumb, lower for a thinner/more dilute sample, higher for a thicker
    one. Defaults to 2.5, which is also shown unconditionally below as the
    Hephaestus reference; set your own to get the mass for THAT thickness
    instead of just the two fixed reference points.

    edge_offset_ev: how far above (and, for the edge-step Δ, below) E0 to
    evaluate μ/ρ, in eV. Defaults to 3, matching Hephaestus' own "a few eV
    above the edge" convention. Note this rarely explains a large mismatch
    against another program's number by itself — μ/ρ changes only ~1%
    between +3 eV and +50 eV once you're cleanly past the absorption jump.
    A big discrepancy is far more likely a disagreement between the two
    programs' TABULATED edge energies (xraydb's E0 vs. another database's),
    which shifts which side of the jump each "+offset" point actually lands
    on."""
    if target_mut <= 0:
        raise ValueError("Target absorption length (μt) must be positive.")
    if edge_offset_ev <= 0:
        raise ValueError("Edge offset (eV) must be positive.")
    components = parse_components(composition_text)
    fractions = element_mass_fractions(components, basis=basis)
    if element not in fractions:
        raise ValueError(f"{element} is not in the composition ({', '.join(sorted(fractions))}).")
    e0 = float(xraydb.xray_edge(element, edge).energy)
    mu_above = compound_mu_rho(fractions, e0 + edge_offset_ev)
    mu_below = compound_mu_rho(fractions, e0 - edge_offset_ev)
    step = mu_above - mu_below
    area = np.pi * (pellet_diameter_mm / 10.0 / 2.0) ** 2  # cm²

    def mass_mg(target, mu):
        return float(target * area / mu * 1000.0) if mu > 0 else float("inf")

    return MassReport(
        edge_energy_ev=e0, edge_offset_ev=float(edge_offset_ev),
        mu_rho_above=mu_above, edge_step_mu_rho=step,
        absorber_fraction=fractions[element], pellet_area_cm2=float(area),
        mass_mut_1_mg=mass_mg(1.0, mu_above), mass_mut_25_mg=mass_mg(2.5, mu_above),
        mass_step_1_mg=mass_mg(1.0, step) if step > 0 else float("inf"),
        target_mut=float(target_mut), mass_target_mut_mg=mass_mg(target_mut, mu_above),
        unit_absorption_length_um=(1e4 / (mu_above * density_g_cm3)) if density_g_cm3 else None,
    )
