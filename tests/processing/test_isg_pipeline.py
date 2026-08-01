"""End-to-end validation on the real archived ISG pressure series (0.5-2
GPa compressed borosilicate glass): raw spectrum -> arPLS baseline ->
windowed 4-Gaussian band fit — the full new pipeline on real research data.

The archived *_fit_report.txt files were deliberately NOT used as golden
values: they predate every bug fix in this rebuild and are internally
inconsistent (their Area column doesn't match amp x width for pure
Gaussians), exactly the "don't freeze buggy output as truth" case the
rewrite plan warned about. Physical invariants are asserted instead.

Findings pinned here (from the interactive validation session):
  - arPLS lambda matters enormously for BROAD glass bands: at 1e5-1e6 the
    baseline eats the band itself (R^2 as low as 0.21); at 1e7 the
    automatic baseline beats the archived hand-drawn one (R^2 0.978 vs
    0.957 on the 1 GPa sample).
  - The main Si-O band red-shifts monotonically with pressure
    (~1069 -> ~965 cm^-1 across 0.5 -> 2 GPa) — the expected
    densification signature.
"""
from __future__ import annotations

import glob
import re

import numpy as np
import pytest

from conftest import ARCHIVE_DIR

pytestmark = pytest.mark.skipif(
    not any(ARCHIVE_DIR.glob("ISG_*gpa.txt")),
    reason="archived ISG pressure series not present on this machine",
)


def _series_files():
    files = glob.glob(str(ARCHIVE_DIR / "ISG_*gpa.txt"))
    # Raw files only (not *_bl_sub etc.)
    files = [f for f in files if re.search(r"ISG_\dp?\d?gpa\.txt$", f)]
    return sorted(files, key=lambda p: float(re.search(r"ISG_(\dp?\d?)gpa", p).group(1).replace("p", ".")))


def _fit_main_band(path):
    from processing.baseline_science import compute_baseline
    from core.fitting_science import compute_r_squared, fit_spectrum

    d = np.loadtxt(path)
    xb, ysub, _ = compute_baseline(d[:, 0], d[:, 1], method="arPLS", params={"lam": 1e7, "ratio": 0.01})
    m = (xb >= 800) & (xb <= 1300)
    x, y = xb[m], ysub[m]
    comps = [{"shape": "G", "shift_val": c, "shift_min": c - 40, "shift_max": c + 40, "fit_shift": True,
              "fwhm_val": 50.0, "fwhm_min": 20.0, "fwhm_max": 90.0, "fit_fwhm": True,
              "amp_val": float(y.max()) / 2, "fit_amp": True} for c in (925.0, 990.0, 1090.0, 1175.0)]
    fr = fit_spectrum(x, y, comps, mode="classic")
    r2 = compute_r_squared(y, fr.y_fit)
    amps = [fr.lmfit_result.params[f"a{i}"].value for i in range(4)]
    k = int(np.argmax(amps))
    return r2, float(fr.lmfit_result.params[f"f{k}"].value)


def test_isg_series_full_pipeline_physical_invariants():
    files = _series_files()
    assert len(files) >= 3

    pressures, r2s, centers = [], [], []
    for f in files:
        pressure = float(re.search(r"ISG_(\dp?\d?)gpa", f).group(1).replace("p", "."))
        r2, center = _fit_main_band(f)
        pressures.append(pressure)
        r2s.append(r2)
        centers.append(center)

    # Every fit is decent; the low-pressure samples (least band evolution
    # away from the 4-band model) are excellent.
    assert all(r2 > 0.85 for r2 in r2s), f"R2s: {r2s}"
    assert r2s[0] > 0.95

    # Densification red-shift: the main band center decreases
    # monotonically with pressure.
    assert all(c2 < c1 for c1, c2 in zip(centers, centers[1:])), f"centers: {centers}"
    assert centers[0] - centers[-1] > 50  # ~1069 -> ~965 cm^-1 overall
