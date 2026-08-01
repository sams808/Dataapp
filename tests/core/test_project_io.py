"""Tests for project_io.py (M14) — .prism project save/load."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import core.project_io as project_io
from core.qt_models import Spectrum


def _spectrum(title="s1", with_df=True, with_match=False):
    x = np.linspace(100, 200, 50)
    y = np.sin(x / 10)
    df = pd.DataFrame({"col_x": x, "col_y": y, "extra": y * 2}) if with_df else None
    meta = {"selected_parser": "raman_xy", "canonical_map": {"X": "col_x", "Y": "col_y"}}
    if with_match:
        meta["rruff_match"] = {"mineral": "Quartz", "rruff_id": "R040031", "wavelength_nm": 532.0}
    return Spectrum(id=Spectrum.new_id(), title=title, path="C:/somewhere/original.txt",
                    kind="raman_xy", x=x, y=y, df=df, meta=meta)


def test_save_load_round_trip_preserves_everything(tmp_path):
    s1 = _spectrum("alpha", with_df=True, with_match=True)
    s2 = _spectrum("beta", with_df=False)
    fit_params = {s1.id: [{"shape": "G", "shift_val": 150.0, "fit_shift": True}]}

    path = tmp_path / "session.prism"
    project_io.save_project(str(path), [s1, s2], fit_params)
    project = project_io.load_project(str(path))
    spectra, loaded_params = project.spectra, project.fit_params

    assert [sp.title for sp in spectra] == ["alpha", "beta"]
    assert spectra[0].id == s1.id  # identity survives the round trip
    assert np.allclose(spectra[0].x, s1.x)
    assert np.allclose(spectra[0].y, s1.y)
    assert spectra[0].kind == "raman_xy"
    assert spectra[0].path == "C:/somewhere/original.txt"
    assert spectra[0].meta["rruff_match"]["mineral"] == "Quartz"

    assert spectra[0].df is not None
    assert list(spectra[0].df.columns) == ["col_x", "col_y", "extra"]
    assert spectra[1].df is None

    assert loaded_params[s1.id][0]["shift_val"] == 150.0


def test_meta_with_numpy_values_serializes(tmp_path):
    sp = _spectrum("np_meta", with_df=False)
    sp.meta["temp_start_C"] = np.float64(21.5)
    sp.meta["indices"] = np.array([1, 2, 3])

    path = tmp_path / "np.prism"
    project_io.save_project(str(path), [sp], {})
    spectra = project_io.load_project(str(path)).spectra
    assert spectra[0].meta["temp_start_C"] == 21.5
    assert spectra[0].meta["indices"] == [1, 2, 3]


def test_load_rejects_non_project_zip(tmp_path):
    import zipfile
    bad = tmp_path / "notproject.prism"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("manifest.json", '{"format": "something-else"}')
    with pytest.raises(ValueError, match="Not a PRISM project"):
        project_io.load_project(str(bad))


def test_load_rejects_newer_version(tmp_path):
    import zipfile
    newer = tmp_path / "future.prism"
    with zipfile.ZipFile(newer, "w") as zf:
        zf.writestr("manifest.json", '{"format": "prism-project", "version": 99}')
    with pytest.raises(ValueError, match="newer"):
        project_io.load_project(str(newer))


def test_empty_project_round_trips(tmp_path):
    path = tmp_path / "empty.prism"
    project_io.save_project(str(path), [], {})
    project = project_io.load_project(str(path))
    assert project.spectra == []
    assert project.fit_params == {}
    assert project.xas_spectra == []
    assert project.htxrd_patterns == []


def test_xas_state_round_trips(tmp_path):
    """Project v2: XAS SpectrumStore contents survive, including e0/label/
    history/angle arrays."""
    from xas.xas_science import Operation, Spectrum as XasSpectrum, _uid

    energy = np.linspace(7000, 7300, 100)
    sp = XasSpectrum(
        sid=_uid("sp"), name="scan1_mu", kind="mu", energy=energy,
        y=np.log(np.linspace(2, 3, 100)), angle=np.linspace(10, 12, 100),
        units="μ(E)", label="XAS(Fe K)", e0=7112.0,
        meta={"source": "beamline.zip"},
        history=[Operation("import", {"source": "beamline.zip"}), Operation("mu_builder", {"log": "ln"})],
    )

    path = tmp_path / "xas.prism"
    project_io.save_project(str(path), [], {}, xas_spectra=[sp])
    project = project_io.load_project(str(path))

    assert len(project.xas_spectra) == 1
    loaded = project.xas_spectra[0]
    assert loaded.sid == sp.sid
    assert loaded.name == "scan1_mu"
    assert loaded.kind == "mu"
    assert loaded.label == "XAS(Fe K)"
    assert loaded.e0 == 7112.0
    assert np.allclose(loaded.energy, energy)
    assert np.allclose(loaded.angle, sp.angle)
    assert [op.name for op in loaded.history] == ["import", "mu_builder"]
    assert loaded.history[1].params["log"] == "ln"


def test_bundled_demo_project_loads():
    """EXAMPLES/demo_project.prism is the onboarding demo (File > Open
    project) — it must always load with the shipped format version. It was
    saved under the app's former name, so this also covers loading a
    legacy-format manifest."""
    from conftest import EXAMPLES_DIR

    demo = EXAMPLES_DIR / "demo_project.prism"
    assert demo.is_file()
    project = project_io.load_project(str(demo))
    assert len(project.spectra) == 3
    kinds = {sp.kind for sp in project.spectra}
    assert "ta_sdt" in kinds  # the DTA example came through typed
    assert len(project.fit_params) == 1  # the seeded Raman fit model


def test_htxrd_state_round_trips(tmp_path):
    """Project v2: an HT-XRD series survives with ramp values/sources and
    arrays — independent of whether the original .rasx files still exist."""
    from xrd.htxrd_science import HtxrdPattern

    x = np.linspace(20, 50, 200)
    patterns = [
        HtxrdPattern(path="C:/gone/scan1.rasx", name="scan1", x=x, y=np.sin(x), ramp_value=21.0, ramp_source="metadata"),
        HtxrdPattern(path="C:/gone/scan2.rasx", name="scan2", x=x, y=np.cos(x), ramp_value=100.0, ramp_source="metadata"),
    ]

    path = tmp_path / "ht.prism"
    project_io.save_project(str(path), [], {}, htxrd_patterns=patterns)
    project = project_io.load_project(str(path))

    assert len(project.htxrd_patterns) == 2
    assert [p.ramp_value for p in project.htxrd_patterns] == [21.0, 100.0]
    assert project.htxrd_patterns[0].ramp_source == "metadata"
    assert np.allclose(project.htxrd_patterns[1].y, np.cos(x))
