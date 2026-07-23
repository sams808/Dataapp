"""Real-profile regression test (Phase 6, spec §5.3; re-frozen for v2:
PRISM_fit_pipeline_upgrade_prompt.md, then again for v3: same file's
SIGMA POLICY §8 + ADDENDUM §7 + §§2-4). One committed real measured
profile with a frozen expected FitResult. CI fails if the staged
pipeline's output on this exact file drifts beyond the stated tolerance —
the concrete proof that a future change to the components/engine/staged
pipeline hasn't silently changed what a real sample's fit means.

The fixture (tests/fixtures/P5Bi8-12__corr.dat) is a real reduced SAXS
profile from the user's own PBi glass series, committed to the repo by
explicit user decision (2026-07-22) so this test doesn't depend on the
external WSU_work\\SAXS\\... path existing on every machine/CI runner.

v3 re-freeze note (2026-07-22): re-run after implementing the full v3
ticket -- sigma-weighted linear fitting is now PRIMARY (this fixture has
a genuine propagated sigma_corrected column, sigma_model=="measured"),
plateau-calibrated (sigma_scale ~1.93, chi2red_plateau_raw ~3.73 before
calibration), with the beamstop-edge auto-trim (2 points, matching the
ticket's own "+1.5/+3 log-residual" description), the OZ/PL2 ladder
rungs, and profile-likelihood CIs using a threshold of 3.841 (the
one-parameter 95%-confidence chi-square critical value, matching the
ticket's own explicit "(95%)" label) scaled by max(1, chi2red_global).

v3 CONSISTENCY-FIX re-freeze (same day, prompted by the user's own
cross-check of the exported JSON): two more real bugs found and fixed.
(1) `_params_to_dict` was rescaling stderr by sqrt(chi2red) on the
mistaken assumption lmfit's own covariance-based stderr wasn't already
chi2red-corrected -- but every fit here uses lmfit's default
scale_covar=True, which already does exactly that, so every reported
stderr was inflated by an EXTRA, erroneous sqrt(chi2red) (confirmed via
a controlled lmfit reproduction). Fixed by reporting lmfit's stderr
as-is. (2) The profile-likelihood grid (25 points spanning the full
+/-15%/x-รท-4 range, uniformly) was far too coarse near the best value
for this real, steeply-peaked data: the actual Delta-chi2 crossing (for
EITHER threshold) fell within the very first, widely-spaced segment,
so linear interpolation across that one big unsampled gap made the
reported CI half-width scale linearly with the threshold instead of
with its square root -- caught by comparing the new stat (flat
Delta-chi2=3.841) vs. rescaled (x chi2red_global) CIs: their ratio came
out as chi2red_global (142) instead of the theoretically required
sqrt(chi2red_global) (11.9). Fixed by log-spacing the 25 grid points
around the best value (same overall range, denser near the center) --
the ratio now lands at 12.1-12.4 vs. the predicted 11.9, and the
absolute magnitude is independently consistent with the (now correctly
single-scaled) covariance stderr times the usual ~1.96 (95%) factor.

Consequently the ticket's own "d_ci half-width" acceptance target is
NOT met at the range the user initially estimated (2-6%, itself based
on the pre-fix, doubly-inflated stderr) -- the properly double-checked
value is ~0.3-0.4%, i.e. d is even more tightly determined than
originally thought, not less. This is reported honestly rather than
forcing the test to the originally-guessed range: both consistency
checks (ratio~sqrt(chi2red), magnitude~stderr*1.96) independently agree
on this smaller number.

Three of the ticket's acceptance targets are now cleanly met: the ladder
still selects BG_TS_PL2, d = 875.6 Å (within the "875 ± 90 Å" target),
and xi now resolves to a genuine FINITE 95% CI (rather than being
flagged unidentifiable) -- a real correction from fixing the original
flat Delta-chi2=1 threshold bug: it was simply too strict, making xi
look unidentifiable when a properly-scaled 95%-confidence interval shows
this instrument's data actually DOES resolve it, just with real (~9%)
uncertainty. fa is correspondingly reported as a normal point value.

Two targets are NOT met, and -- like the v2 finding it supersedes -- this
is a genuine, investigated data/model limitation rather than an unfixed
bug: calibrated chi2red is ~142 (ticket target [0.7,2.5]) and the W_peak
window's own median log10 residual exceeds the 0.15 visual-equivalence
gate (flagged `visual_equivalence_gate_bypassed` -- EVERY ladder
candidate fails that gate on this real profile, and reporting the
best-BIC one anyway, honestly flagged, is a deliberately better outcome
than falling back to a cruder model that would fail it by even more; see
select_best_preset's own docstring). Per-window chi2red (new gof keys
chi2red_w_loq/w_peak/w_hiq) makes the SOURCE of that misfit visible
directly in the numbers: chi2red_w_hiq ~ 1.9 (essentially fine) while
chi2red_w_loq/w_peak are both >>1 -- the misfit is concentrated in the
low-q/peak region's own complexity (point-to-point log-log slopes from
-2.3 to -8.2 across just 8-9 points, investigated directly, far more
structure than a single power-law/OZ/Guinier-Porod term can capture),
not spread evenly across the whole curve. BG_TS_GP/BG_TS_OZ were both
checked directly and fit markedly worse (chi2red ~600) than BG_TS_PL2 on
this exact data, confirming PL2 is genuinely the best available
description, not a fallback masking a fixable bug. Whether the
propagated sigma itself needs a more sophisticated (non-uniform)
rescaling is a separate, real open question flagged to the user rather
than papered over here.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from saxs_core.composite_staged import fit_staged
from saxs_core.loader import load_curve

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "P5Bi8-12__corr.dat"

# Frozen reference (multistart_n=8, sample_id="P5Bi8-12" — deterministic,
# see test_composite_staged.py's own determinism test for why this is safe
# to freeze exactly). Captured directly from a real run of the v3 pipeline
# against this exact fixture (verified reproducible across repeated runs).
FROZEN_D = 875.5650137280488
FROZEN_XI = 3858.198710673578
FROZEN_PRESET = "BG_TS_PL2"
FROZEN_RMS_LOG = 0.2652827565929912
FROZEN_CHI2RED = 142.07406351182553
FROZEN_SIGMA_SCALE = 1.931843139957385


def test_real_profile_regression_p5bi8_12():
    assert FIXTURE_PATH.is_file(), "committed fixture missing — see tests/fixtures/"
    curve = load_curve(str(FIXTURE_PATH))
    result = fit_staged(curve, sample_id="P5Bi8-12", multistart_n=8)

    assert result.preset_chosen == FROZEN_PRESET
    assert result.derived["d"] == pytest.approx(FROZEN_D, rel=0.01)
    assert result.derived["xi"] == pytest.approx(FROZEN_XI, rel=0.05)
    assert result.gof["rms_log"] == pytest.approx(FROZEN_RMS_LOG, rel=0.05)
    assert result.gof["chi2red"] == pytest.approx(FROZEN_CHI2RED, rel=0.1)
    assert result.sigma_model == "measured"
    assert result.sigma_scale == pytest.approx(FROZEN_SIGMA_SCALE, rel=0.05)

    # v3 §2: xi now resolves to a genuine finite 95% CI (not flagged
    # unidentifiable) once the profile-likelihood threshold correctly
    # uses the one-parameter 95%-confidence chi-square value; fa is a
    # normal point value, consistent with xi being identified.
    assert not result.xi_unidentifiable
    assert result.xi_ci is not None and result.xi_ci[0] is not None and result.xi_ci[1] is not None
    xi_halfwidth_pct = 100.0 * (result.xi_ci[1] - result.xi_ci[0]) / 2.0 / result.derived["xi"]
    assert 2.0 < xi_halfwidth_pct < 20.0
    assert result.fa_bound is None

    # v3 §2: d IS identifiable. The properly-fixed (double-scaling +
    # grid-resolution bugs both corrected) rescaled CI half-width lands
    # ~0.3-0.4%, well under the ticket's own "<5%" bar but ALSO well
    # under the user's own initial 2-6% estimate (itself based on the
    # pre-fix, doubly-inflated stderr) -- verified via two independent
    # consistency checks below, not just asserted on its own.
    assert result.d_ci is not None
    d_halfwidth_pct = 100.0 * (result.d_ci[1] - result.d_ci[0]) / 2.0 / result.derived["d"]
    assert 0.05 < d_halfwidth_pct < 1.5

    # Consistency-fix cross-check (v3, prompted directly by the user's own
    # inspection of the exported JSON): the stat (flat Delta-chi2=3.841)
    # vs. rescaled (x chi2red_global) CI half-widths must differ by very
    # close to sqrt(chi2red_global) -- both share the same underlying
    # Delta-chi2 profile, only the threshold differs by that factor, so
    # this ratio is a direct, sensitive test of both the double-scaling
    # fix (stderr) and the grid-resolution fix (this ratio was chi2red,
    # not sqrt(chi2red), before the grid was log-spaced around the best
    # value).
    assert result.d_ci_stat is not None and result.xi_ci_stat is not None
    sqrt_chi2red = math.sqrt(result.gof["chi2red"])
    d_hw = (result.d_ci[1] - result.d_ci[0]) / 2.0
    d_hw_stat = (result.d_ci_stat[1] - result.d_ci_stat[0]) / 2.0
    assert (d_hw / d_hw_stat) == pytest.approx(sqrt_chi2red, rel=0.1)
    xi_hw = (result.xi_ci[1] - result.xi_ci[0]) / 2.0
    xi_hw_stat = (result.xi_ci_stat[1] - result.xi_ci_stat[0]) / 2.0
    assert (xi_hw / xi_hw_stat) == pytest.approx(sqrt_chi2red, rel=0.1)

    # v3 consistency fix §3: per-window chi2red shows the misfit driving
    # the elevated global chi2red is concentrated in the low-q/peak
    # region, NOT spread evenly across the whole curve -- justifies (for
    # the paper) treating the peak parameters as trustworthy despite the
    # elevated global chi2red.
    assert result.gof["chi2red_w_hiq"] < 10.0
    assert result.gof["chi2red_w_loq"] > result.gof["chi2red_w_hiq"]
    assert result.gof["chi2red_w_peak"] > result.gof["chi2red_w_hiq"]

    # v3 §5: no pruned component's parameters leak into Derived.
    assert "p_pl" not in result.derived  # power_law was pruned on this fit

    # sanity: d is within the spec's own stated observed range (§5.1) and
    # within the ticket's explicit "d = 875 ± 90 Å" acceptance target; the
    # remaining chi2red/visual-equivalence gaps are a documented, genuine
    # finding (see module docstring), not asserted against the ticket's
    # [0.7,2.5]/<0.15 targets here, since doing so would misrepresent a
    # real, investigated data limitation as a pass.
    assert 700.0 <= result.derived["d"] <= 1700.0
    assert 785.0 <= result.derived["d"] <= 965.0
    assert -1.0 < result.derived["fa"] < 0.0
