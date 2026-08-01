"""Tests for saxs_core/composite_fit.py — the lmfit-backed composition
engine (Phase 2 of the composite-fitting roadmap): presets, prefixing,
eval/eval_components/derived, and one converging end-to-end fit proving
the lmfit wiring (bounds/vary/prefix/expr) actually works.
"""
from __future__ import annotations

import numpy as np
import pytest

from saxs.core.composite_fit import (
    PRESETS, CompositeModel, build_composite, build_preset, gaussian_smear,
)
from saxs.core.composite_models import COMPONENTS


# ---------------------------------------------------------------------------
# Preset registry
# ---------------------------------------------------------------------------

def test_presets_match_spec_component_sets():
    assert PRESETS["BG"] == ["flat_background", "power_law"]
    assert PRESETS["BG_DAB"] == ["flat_background", "power_law", "dab"]
    assert PRESETS["BG_TS"] == ["flat_background", "power_law", "teubner_strey"]
    assert PRESETS["BG_TS_GP"] == ["flat_background", "power_law", "teubner_strey", "guinier_porod"]
    assert PRESETS["BG_BP"] == ["flat_background", "power_law", "broad_peak"]


def test_build_preset_unknown_name_raises():
    with pytest.raises(KeyError, match="Unknown preset"):
        build_preset("NOT_A_PRESET")


@pytest.mark.parametrize("preset", list(PRESETS))
def test_build_preset_produces_expected_prefixes(preset):
    model = build_preset(preset)
    prefixes = [p for p, _ in model.components]
    assert len(prefixes) == len(PRESETS[preset])
    assert len(set(prefixes)) == len(prefixes)  # no collisions


# ---------------------------------------------------------------------------
# build_composite: general entry point, arbitrary component lists
# ---------------------------------------------------------------------------

def test_build_composite_unknown_component_raises():
    with pytest.raises(KeyError, match="Unknown component"):
        build_composite(["flat_background", "not_a_real_component"])


def test_build_composite_numbers_repeated_component_types():
    model = build_composite(["flat_background", "teubner_strey", "teubner_strey"])
    prefixes = [p for p, _ in model.components]
    assert prefixes == ["bg_", "ts_", "ts2_"]


def test_composite_model_rejects_empty_and_duplicate_prefixes():
    with pytest.raises(ValueError, match="at least one component"):
        CompositeModel([])
    dup = [("bg_", COMPONENTS["flat_background"]()), ("bg_", COMPONENTS["power_law"]())]
    with pytest.raises(ValueError, match="Duplicate"):
        CompositeModel(dup)


# ---------------------------------------------------------------------------
# eval / eval_components / derived — spec §5.2 "composite eval == sum of parts"
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("preset", list(PRESETS))
def test_eval_equals_sum_of_eval_components(preset):
    model = build_preset(preset)
    q = np.linspace(1e-3, 0.3, 300)
    params = model.to_lmfit_parameters()
    total = model.eval(q, params)
    parts = model.eval_components(q, params)
    np.testing.assert_allclose(total, sum(parts.values()), rtol=1e-12)


def test_eval_accepts_plain_dict_or_lmfit_parameters_identically():
    model = build_preset("BG_TS")
    q = np.linspace(1e-3, 0.3, 200)
    params = model.to_lmfit_parameters()
    plain = {name: params[name].value for name in model.param_names()}
    np.testing.assert_allclose(model.eval(q, params), model.eval(q, plain), rtol=1e-12)


def test_derived_is_keyed_like_eval_components_and_carries_ts_fa():
    model = build_preset("BG_TS")
    params = model.to_lmfit_parameters(seed_values={"ts_d": 1200.0, "ts_xi": 3000.0})
    derived = model.derived(params)
    assert set(derived) == {"bg", "pl", "ts"}
    assert "fa" in derived["ts"] and -1.0 < derived["ts"]["fa"] < 0.0


def test_seed_returns_prefixed_keys_for_every_component():
    model = build_preset("BG_TS_GP")
    q = np.linspace(1e-4, 0.3, 1000)
    I = model.eval(q, model.to_lmfit_parameters(seed_values={"ts_d": 1200.0, "ts_xi": 3000.0}))
    seeded = model.seed(q, I)
    assert set(seeded) == set(model.param_names())


# ---------------------------------------------------------------------------
# One converging end-to-end fit — proves the lmfit wiring itself
# ---------------------------------------------------------------------------

def test_bg_fit_recovers_known_background_and_power_law():
    # q kept to a moderate (~1.3 decade) dynamic range with counting-
    # statistics weighting, so the additive background stays identifiable
    # against the power-law tail (an unweighted fit spanning many decades
    # makes a small additive constant statistically invisible — that's
    # real SAXS statistics, not a bug; the spec's own reason to default to
    # sigma weighting, §3).
    model = build_preset("BG")
    q = np.linspace(0.1, 1.0, 300)  # extends far enough to show a genuine background plateau
    true = {"bg_C": 5.0, "pl_B": 0.4, "pl_p": 3.0}
    rng = np.random.default_rng(0)
    I_clean = model.eval(q, true)
    sigma = np.sqrt(np.abs(I_clean)) + 0.5
    I = I_clean + rng.normal(0, sigma)
    result = model.fit(q, I, sigma=sigma,
                      params=model.to_lmfit_parameters(seed_values={"pl_B": 1.0, "pl_p": 2.5}))
    assert result.success
    assert result.params["pl_p"].value == pytest.approx(3.0, rel=0.1)
    assert result.params["bg_C"].value == pytest.approx(5.0, rel=0.3)


def test_ts_fit_recovers_known_d_and_xi():
    # q restricted to a window straddling q_max comfortably (spec's own
    # Stage 2 fits TS on W_peak ∪ W_hiq, never power_law's raw q^-p form
    # all the way down to the real data's q~1e-4, where it legitimately
    # diverges to enormous values — realistic staged windowing is Phase 3;
    # this Phase 2 test just proves the composite/lmfit wiring converges).
    model = build_preset("BG_TS")
    q = np.linspace(0.001, 0.02, 600)
    true = {"bg_C": 1.0, "pl_B": 1e-11, "pl_p": 4.0, "ts_S": 40.0, "ts_d": 1200.0, "ts_xi": 3000.0}
    I = model.eval(q, true)
    seeded = model.to_lmfit_parameters(seed_values={"ts_S": 30.0, "ts_d": 1000.0, "ts_xi": 2000.0, "pl_B": 1e-11, "pl_p": 4.0})
    model.fix(seeded, "pl_p", 4.0)
    model.fix(seeded, "pl_B", 1e-11)
    result = model.fit(q, I, params=seeded)
    assert result.success
    assert result.params["ts_d"].value == pytest.approx(1200.0, rel=0.02)
    assert result.params["ts_xi"].value == pytest.approx(3000.0, rel=0.1)


# ---------------------------------------------------------------------------
# Constraints/expressions (spec §2.1: fix pl_p, tie two components' xi)
# ---------------------------------------------------------------------------

def test_fix_freezes_a_parameter_through_a_fit():
    model = build_preset("BG_TS")
    params = model.to_lmfit_parameters()
    model.fix(params, "pl_p", 4.0)
    assert params["pl_p"].vary is False
    assert params["pl_p"].value == pytest.approx(4.0)


def test_set_expr_ties_two_components_xi_together():
    model = build_composite(["flat_background", "teubner_strey", "teubner_strey"])
    q = np.linspace(1e-4, 0.03, 600)
    true = {"bg_C": 1.0, "ts_S": 30.0, "ts_d": 1000.0, "ts_xi": 2500.0,
            "ts2_S": 15.0, "ts2_d": 1800.0, "ts2_xi": 2500.0}
    I = model.eval(q, true)
    params = model.to_lmfit_parameters(seed_values={
        "ts_S": 25.0, "ts_d": 950.0, "ts_xi": 2400.0,
        "ts2_S": 12.0, "ts2_d": 1750.0, "ts2_xi": 2400.0,
    })
    model.set_expr(params, "ts2_xi", "ts_xi")
    result = model.fit(q, I, params=params)
    assert result.success
    assert result.params["ts2_xi"].value == pytest.approx(result.params["ts_xi"].value, rel=1e-9)


# ---------------------------------------------------------------------------
# Optional smearing utility — off by default, standalone
# ---------------------------------------------------------------------------

def test_gaussian_smear_is_a_noop_at_zero_sigma():
    q = np.linspace(1e-3, 0.3, 200)
    I = np.sin(q * 50.0) + 2.0
    np.testing.assert_allclose(gaussian_smear(q, I, 0.0), I)


def test_gaussian_smear_blurs_a_sharp_feature():
    q = np.linspace(0, 1.0, 1000)
    I = (np.abs(q - 0.5) < 0.01).astype(float)  # a narrow spike
    smeared = gaussian_smear(q, I, sigma_q=0.05)
    assert smeared.max() < I.max()  # peak height reduced by blurring
    assert smeared.sum() == pytest.approx(I.sum(), rel=0.05)  # area roughly conserved
