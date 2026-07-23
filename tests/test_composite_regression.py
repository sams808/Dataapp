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
ticket's own explicit "(95%)" label -- NOT a flat Delta-chi2=1, a bug
caught only once synthetic set C exercised the identifiable case)
scaled by max(1, chi2red_min) (see compute_ts_profile_likelihood_cis's
own docstring for why the overall chi2red still needs accounting for).

Three of the ticket's acceptance targets are now cleanly met: the ladder
still selects BG_TS_PL2, d = 875.6 Å (within the "875 ± 90 Å" target,
CI half-width ~0.16% -- comfortably under the "<5%" bar), and xi now
resolves to a genuine FINITE 95% CI (3622-4211 Å, ~7.6% relative
half-width) rather than being flagged unidentifiable -- a real
correction from fixing the threshold bug above: the earlier flat
Delta-chi2=1 threshold was simply too strict, making xi look
unidentifiable when a properly-scaled 95%-confidence interval shows this
instrument's data actually DOES resolve it, just with real (~8%)
uncertainty. fa is correspondingly reported as a normal point value.

Two targets are NOT met, and -- like the v2 finding it supersedes -- this
is a genuine, investigated data/model limitation rather than an unfixed
bug: calibrated chi2red is ~142 (ticket target [0.7,2.5]) and the W_peak
window's own median log10 residual exceeds the 0.15 visual-equivalence
gate (flagged `visual_equivalence_gate_bypassed` -- EVERY ladder
candidate fails that gate on this real profile, and reporting the
best-BIC one anyway, honestly flagged, is a deliberately better outcome
than falling back to a cruder model that would fail it by even more; see
select_best_preset's own docstring). Investigated directly: the raw low-q
data (just past the beamstop-trimmed points) shows point-to-point log-log
slopes from -2.3 to -8.2 across only 8-9 points -- far more structure
than a single power-law/OZ/Guinier-Porod term can capture, and BG_TS_GP/
BG_TS_OZ were both checked directly and fit markedly worse (chi2red ~600)
than BG_TS_PL2 on this exact data, confirming PL2 is genuinely the best
available description, not a fallback masking a fixable bug. Whether the
propagated sigma itself needs a more sophisticated (non-uniform) rescaling
is a separate, real open question flagged to the user rather than papered
over here.
"""
from __future__ import annotations

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
    assert xi_halfwidth_pct < 20.0
    assert result.fa_bound is None

    # v3 §2: d IS identifiable, with a tight CI comfortably clearing the
    # ticket's own "<5% half-width" acceptance bar.
    assert result.d_ci is not None
    d_halfwidth_pct = 100.0 * (result.d_ci[1] - result.d_ci[0]) / 2.0 / result.derived["d"]
    assert d_halfwidth_pct < 5.0

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
