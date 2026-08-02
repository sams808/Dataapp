"""Tests for the .rasx (Rigaku SmartLab) parser added to io_universal.py.

.rasx is a ZIP container: Data*/Profile*.txt holds headerless tab-separated
(2theta, intensity, count) rows, and Data*/MesurementConditions*.xml holds
axis metadata including the "Temp" axis's Position/EndPosition — needed for
high-temperature XRD series where each pattern is collected while the sample
continues heating (a real temperature range per scan, not a single value).

Fixture: EXAMPLES/HTXRD_example.rasx, one real scan copied from an actual
Rigaku HTXRD series (P5Bi8-12, room temp to 900 C), covering 19-23 C.
"""
from __future__ import annotations

import numpy as np
import pytest

import core.io_universal as iou
from conftest import EXAMPLES_DIR

RASX_PATH = EXAMPLES_DIR / "HTXRD_example.rasx"


def test_sniff_rasx_detects_by_extension_only():
    assert iou.sniff_rasx(str(RASX_PATH), "") is True
    assert iou.sniff_rasx("not_a_rasx.txt", "") is False


def test_parse_rasx_extracts_profile_and_temperature_range():
    df, meta = iou.parse_rasx(str(RASX_PATH))
    assert meta["parser"] == "rasx"
    assert meta["temp_start_C"] == pytest.approx(19.0)
    assert meta["temp_end_C"] == pytest.approx(23.0)

    assert len(df) > 1000
    two_theta = df[meta["canonical_map"]["2theta_deg"]].to_numpy(dtype=float)
    intensity = df[meta["canonical_map"]["intensity"]].to_numpy(dtype=float)
    assert np.isfinite(two_theta).all()
    assert np.isfinite(intensity).all()
    assert np.all(np.diff(two_theta) > 0)  # 2theta should be monotonically increasing
    assert intensity.min() >= 0


def test_load_any_auto_detects_rasx_ahead_of_other_parsers():
    df, meta = iou.load_any(str(RASX_PATH), return_meta=True)
    assert meta["selected_parser"] == "rasx"
    assert meta["autodetect_failed"] is False
    canon = meta["canonical_map"]
    assert canon.get("X") and canon.get("Y")
