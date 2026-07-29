"""Raw 2D EDF (ESRF Data Format) frame ingestion via pyFAI.

Built against a Xenocs Xeuss Pro: a BEAMSTOPLESS SAXS/WAXS/GISAXS/USAXS
instrument. There is no physical beamstop -- the direct beam is deliberately
co-recorded on the same detector frame as the sample scattering, giving
extended low-q access and letting the instrument's own software (XSACT Pro)
derive transmission directly from that recorded direct-beam spot rather than
from a separate diode. This explains why ``TransmittedFlux``/``Intensity1``/
``pilai0``/``pilct0``/``pilroi0`` all read the same quantity in every frame
checked, and why ``Monitor``/``mon`` are always 0 (no separate monitor
hardware is used in this design).

For instruments whose control software already writes a calibrated beam
center / sample distance / wavelength into each frame's own header (true
here), so no per-batch geometry refinement is required -- callers who want
extra rigor can still validate the embedded geometry against an AgBeh
calibration frame (see ``validate_against_agbeh``).

Produces a properly-normalized :class:`~saxs_core.curve.Curve` that feeds
into the *existing* :mod:`saxs_core.reduction` empty-subtraction pipeline
unchanged -- this module only replaces how a 1D curve is obtained from a raw
frame, not how two 1D curves get combined.

Geometry axis convention (validated against a real AgBeh calibration frame:
integrated first-order ring at q=0.1077 vs the literature 0.10763 A^-1 for
silver behenate, second order at 0.2152 vs 0.2153 -- not assumed, checked):

    poni1 = Center_2 * pixel_size   (array axis 0, "Dim_2" in the EDF header)
    poni2 = Center_1 * pixel_size   (array axis 1, "Dim_1" in the EDF header)

Normalization convention: historically-produced 1D reductions of this data
divided a sample's own integrated signal by that sample's own transmitted
flux, which inflates strongly-absorbing samples -- confirmed directly
against raw pixel counts: a highly-absorbing sample shows LOWER raw counts
than a weakly-absorbing one (as physically expected), the opposite of what
those historical 1D files showed. So intensities here are normalized by
exposure time only (a genuine count rate, unaffected by absorption);
``curve.transmission`` carries the frame's own TransmittedFlux for later use
-- ``saxs_core.reduction.correct_sample``'s existing "transmission" scale
mode already applies that ratio only to the empty side of a subtraction,
which is the physically correct place for it. This normalization fix is
independent of the beamstopless-specific caveat below and holds across the
whole accessible q-range.

KNOWN LIMITATION, disclosed rather than papered over: the region immediately
around the direct beam is excluded from integration here (see
``direct_beam_mask``), not because it is a "shadowed" artifact region (there
is no beamstop) but because repeat measurements of the same nominal empty
capillary disagree by up to ~100x there even after correcting for sub-pixel
beam-center registration between frames -- most plausibly some HDR/
attenuation handling specific to this beamstopless design's high dynamic
range capability, applied by XSACT Pro before/while writing these frames,
that is not documented in what's available here. Rather than guess at an
unknown vendor procedure, that region is masked out entirely; scattering
below roughly q=0.0095 A^-1 (r<28 px at this geometry) is NOT currently
recoverable through this module. Samples whose real structural feature (e.g.
a Teubner-Strey correlation peak) sits below that q are not usable through
this path without first learning XSACT Pro's own direct-beam handling.

pyFAI and fabio are optional at import time (matching the sasmodels/bumps
pattern in ``saxs_core.modelfit``) so the rest of the suite works without
them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .curve import Curve

try:
    import fabio
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator

    HAVE_PYFAI = True
    _IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # pragma: no cover - depends on environment
    HAVE_PYFAI = False
    _IMPORT_ERROR = str(exc)


def pyfai_available() -> bool:
    return HAVE_PYFAI


def pyfai_import_error() -> Optional[str]:
    return _IMPORT_ERROR


def _require_pyfai() -> None:
    if not HAVE_PYFAI:
        raise ImportError(
            "pyFAI/fabio are required for EDF ingestion but are not "
            f"importable ({_IMPORT_ERROR})"
        )


_REQUIRED_GEOMETRY_KEYS = (
    "Center_1", "Center_2", "SampleDistance", "PSize_1", "PSize_2",
)

# Literature silver behenate d-spacing (Huang et al. 1993), for
# validate_against_agbeh.
AGBEH_D_SPACING_A = 58.38
AGBEH_Q1_A_INV = 2.0 * np.pi / AGBEH_D_SPACING_A

# Pixels within this radius of the beam center are excluded before
# integration. This is the direct beam itself (not a beamstop shadow -- this
# instrument has no beamstop), plus a surrounding transition zone: repeat
# empty-capillary measurements disagree by up to ~100x there even after
# correcting for sub-pixel beam-center registration between frames, most
# plausibly from HDR/attenuation handling specific to this beamstopless
# design that isn't documented in what's available here (see the module
# docstring's KNOWN LIMITATION). r=28px was the smallest radius, checked
# against 4 real samples (P0Bi0, P0Bi8-13, P5Bi8-12, P1Bi1-13), that
# eliminated negative post-subtraction points entirely. Scattering below the
# resulting q_min (~0.0095 A^-1 at this geometry) is not currently usable
# through this module.
DEFAULT_DIRECT_BEAM_MASK_RADIUS_PX = 28.0


@dataclass
class EdfGeometry:
    dist: float
    poni1: float
    poni2: float
    pixel1: float
    pixel2: float
    wavelength: float


def geometry_from_header(header: Dict[str, str]) -> EdfGeometry:
    """Build integrator geometry from an EDF header's own recorded beam
    center / distance / wavelength. Raises ValueError with the specific
    missing keys rather than silently defaulting -- a wrong geometry would
    be a silent, hard-to-detect calibration error.
    """
    missing = [k for k in _REQUIRED_GEOMETRY_KEYS if k not in header]
    wavelength_key = "Wavelength" if "Wavelength" in header else "WaveLength"
    if wavelength_key not in header:
        missing.append("Wavelength/WaveLength")
    if missing:
        raise ValueError(f"EDF header missing required geometry keys: {missing}")

    psize1 = float(header["PSize_1"])
    psize2 = float(header["PSize_2"])
    c1 = float(header["Center_1"])
    c2 = float(header["Center_2"])
    return EdfGeometry(
        dist=float(header["SampleDistance"]),
        poni1=c2 * psize2,
        poni2=c1 * psize1,
        pixel1=psize1,
        pixel2=psize2,
        wavelength=float(header[wavelength_key]),
    )


def build_integrator(geometry: EdfGeometry) -> "AzimuthalIntegrator":
    _require_pyfai()
    return AzimuthalIntegrator(
        dist=geometry.dist, poni1=geometry.poni1, poni2=geometry.poni2,
        pixel1=geometry.pixel1, pixel2=geometry.pixel2,
        wavelength=geometry.wavelength,
    )


def read_edf(path: str) -> Tuple[np.ndarray, Dict[str, str]]:
    """Read one raw 2D EDF frame. Read-only: never writes or modifies the
    source file."""
    _require_pyfai()
    img = fabio.open(str(path))
    return img.data.astype(float), dict(img.header)


def direct_beam_mask(
    shape: Tuple[int, int], geometry: EdfGeometry, radius_px: float = DEFAULT_DIRECT_BEAM_MASK_RADIUS_PX,
) -> np.ndarray:
    """Boolean mask (True = excluded) for the direct beam and its
    surrounding not-reliably-interpretable transition zone (see the module
    docstring's KNOWN LIMITATION) around the beam center."""
    ny, nx = shape
    c1 = geometry.poni2 / geometry.pixel2
    c2 = geometry.poni1 / geometry.pixel1
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.sqrt((xx - c1) ** 2 + (yy - c2) ** 2)
    return r < radius_px


def integrate_edf_frame(
    path: str,
    npt: int = 1000,
    q_range: Optional[Tuple[float, float]] = None,
    direct_beam_mask_radius_px: float = DEFAULT_DIRECT_BEAM_MASK_RADIUS_PX,
    file_role: str = "sample",
) -> Curve:
    """Azimuthally integrate one raw EDF frame into a Curve.

    Intensity is a count RATE (integrated counts / ExposureTime) -- see the
    module docstring for why this deliberately does not divide by the
    frame's own transmitted flux.
    """
    _require_pyfai()
    data, header = read_edf(path)
    geometry = geometry_from_header(header)
    ai = build_integrator(geometry)

    dummy = float(header.get("Dummy", -1.0))
    delta_dummy = float(header.get("DDummy", 0.5))
    exposure = float(header["ExposureTime"])
    if not np.isfinite(exposure) or exposure <= 0:
        raise ValueError(f"{path}: non-positive or invalid ExposureTime ({exposure})")

    mask = direct_beam_mask(data.shape, geometry, radius_px=direct_beam_mask_radius_px)

    q, intensity, sigma = ai.integrate1d(
        data, npt, unit="q_A^-1", dummy=dummy, delta_dummy=delta_dummy,
        error_model="poisson", correctSolidAngle=True, mask=mask,
        radial_range=q_range,
    )

    rate = intensity / exposure
    rate_sigma = sigma / exposure

    transmission = None
    for key in ("TransmittedFlux", "Intensity1"):
        if key in header:
            try:
                transmission = float(header[key])
                break
            except ValueError:
                continue

    metadata = {
        "Date": header.get("Date"),
        "Comment": header.get("Comment"),
        "ExposureTime": exposure,
        "SampleDistance": geometry.dist,
        "Wavelength": geometry.wavelength,
        "TransmittedFlux": transmission,
        "source_path": str(path),
    }

    curve = Curve(
        q=q, intensity=rate, sigma=rate_sigma,
        name=Path(path).stem, path=str(Path(path).resolve()),
        header_lines=[f"# {k}: {v}" for k, v in header.items()],
        metadata=metadata,
        transmission=transmission,
        file_role=file_role,
    )
    curve.record(
        "integrate_edf", npt=npt, dummy=dummy, delta_dummy=delta_dummy,
        exposure_time=exposure, direct_beam_mask_radius_px=direct_beam_mask_radius_px,
    )
    return curve


@dataclass
class AveragedEmpty:
    curve: Curve
    stat_sigma: np.ndarray
    sys_sigma: np.ndarray
    n_inputs: int
    sources: List[str]


def average_empty_frames(
    paths: Sequence[str],
    npt: int = 1000,
    direct_beam_mask_radius_px: float = DEFAULT_DIRECT_BEAM_MASK_RADIUS_PX,
) -> AveragedEmpty:
    """Average >=2 independently-measured empty-capillary frames into one
    canonical empty curve on a common q grid.

    Returns both the propagated STATISTICAL sigma (quadrature/N of each
    empty's own per-point Poisson sigma) and the point-wise SYSTEMATIC floor.

    The systematic floor is the standard ERROR of the mean across the N
    empties (population standard deviation / sqrt(N)), not the raw
    population standard deviation. What downstream code actually uses is
    mean(empty) subtracted from each sample, so the relevant uncertainty is
    the uncertainty *in that estimated mean* -- standard error propagation,
    not the spread of a single future draw. It still captures real
    run-to-run instrument variation (thermal drift, small beam fluctuations)
    that no single frame's own counting statistics could reveal; it just
    reports it on the same "uncertainty of the mean" footing as the
    statistical term above, rather than conflating the two.
    """
    if len(paths) < 2:
        raise ValueError("Need at least 2 empty frames to estimate a systematic floor")

    curves = [
        integrate_edf_frame(p, npt=npt, direct_beam_mask_radius_px=direct_beam_mask_radius_px, file_role="empty")
        for p in paths
    ]
    # pyFAI's own bin centers shift slightly between frames whose beam
    # center differs even by ~1 px (confirmed empirically -- a fixed
    # radial_range does not guarantee identical output q values), so
    # resample every curve onto one explicit shared grid rather than
    # assuming the raw integrator outputs align bin-for-bin.
    qmin = max(float(c.q.min()) for c in curves)
    qmax = min(float(c.q.max()) for c in curves)
    q0 = np.linspace(qmin, qmax, npt)
    resampled_I = [np.interp(q0, c.q, c.intensity) for c in curves]
    resampled_sigma = [np.interp(q0, c.q, c.sigma) for c in curves]

    stack = np.stack(resampled_I, axis=0)
    sigma_stack = np.stack(resampled_sigma, axis=0)

    rate_mean = np.mean(stack, axis=0)
    stat_sigma = np.sqrt(np.sum(sigma_stack ** 2, axis=0)) / len(curves)
    sys_sigma = np.std(stack, axis=0, ddof=1) / np.sqrt(len(curves))
    combined_sigma = np.sqrt(stat_sigma ** 2 + sys_sigma ** 2)

    merged = Curve(
        q=q0, intensity=rate_mean, sigma=combined_sigma,
        name="empty_average", file_role="empty",
        metadata={"n_inputs": len(curves), "sources": [c.path for c in curves]},
    )
    merged.record("average_empty", n_inputs=len(curves), sources=[c.name for c in curves])
    return AveragedEmpty(
        curve=merged, stat_sigma=stat_sigma, sys_sigma=sys_sigma,
        n_inputs=len(curves), sources=[c.path for c in curves],
    )


def plateau_chi2red(
    q: np.ndarray, corrected: np.ndarray, sigma: np.ndarray,
    q_lo: float = 0.2, q_hi: float = 0.29,
) -> Dict[str, object]:
    """QC gate: reduced chi-square of a corrected curve about its own mean
    in a flat, featureless high-q plateau, using the propagated sigma
    AS-IS (no rescaling). This is the ticket's headline acceptance check --
    a well-calibrated uncertainty should give chi2red near 1 here without
    any post-hoc fudge factor.
    """
    mask = (q > q_lo) & (q < q_hi) & np.isfinite(corrected) & np.isfinite(sigma) & (sigma > 0)
    n = int(np.sum(mask))
    if n < 3:
        return {"n_points": n, "chi2red": None, "mean": None}
    vals = corrected[mask]
    sigs = sigma[mask]
    mean_val = float(np.mean(vals))
    chi2 = float(np.sum(((vals - mean_val) / sigs) ** 2))
    dof = n - 1
    frac_negative = float(np.mean(vals < 0))
    return {
        "n_points": n,
        "chi2red": chi2 / dof,
        "mean": mean_val,
        "frac_negative": frac_negative,
    }


def validate_against_agbeh(
    path: str, npt: int = 1500, q_lo: float = 0.08, q_hi: float = 0.5,
    tol_rel: float = 0.01,
) -> Dict[str, object]:
    """Sanity-check an AgBeh calibration frame's header-derived geometry by
    checking the integrated first-order ring lands near q=0.10763 A^-1.

    Returns a dict with the found peak q, the relative error, and whether it
    passed `tol_rel`. Does not raise on a failed check -- callers decide how
    to react (this is a diagnostic, not an assertion).
    """
    curve = integrate_edf_frame(path, npt=npt, file_role="calibrant")
    mask = (curve.q > q_lo) & (curve.q < q_hi)
    if not np.any(mask):
        return {"ok": False, "reason": "no data in expected q range", "q_peak": None}
    q_sub = curve.q[mask]
    i_sub = curve.intensity[mask]
    q_peak = float(q_sub[np.argmax(i_sub)])
    rel_err = abs(q_peak - AGBEH_Q1_A_INV) / AGBEH_Q1_A_INV
    return {
        "ok": rel_err <= tol_rel,
        "q_peak": q_peak,
        "q_expected": AGBEH_Q1_A_INV,
        "rel_err": rel_err,
    }
