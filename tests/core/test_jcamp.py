"""Tests for the JCAMP-DX parser (M15) in io_universal.py.

Covers plain AFFN ##XYDATA, compressed ASDF (SQZ/DIF/DUP), ##XYPOINTS
pairs, YFACTOR scaling, and parser-registry integration.
"""
from __future__ import annotations

import numpy as np
import pytest

import core.io_universal as iu


def _write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return str(p)


AFFN_FILE = """\
##TITLE=Test spectrum (AFFN)
##JCAMP-DX=4.24
##DATA TYPE=RAMAN SPECTRUM
##XUNITS=1/CM
##YUNITS=ARBITRARY UNITS
##XFACTOR=1.0
##YFACTOR=1.0
##FIRSTX=100.0
##LASTX=104.0
##NPOINTS=5
##XYDATA=(X++(Y..Y))
100.0 10.0 20.0 30.0
103.0 40.0 50.0
##END=
"""


def test_jcamp_affn_xydata(tmp_path):
    path = _write(tmp_path, "affn.jdx", AFFN_FILE)
    df, meta = iu.parse_jcamp(path)
    assert meta["parser"] == "jcamp"
    assert meta["jcamp_title"] == "Test spectrum (AFFN)"
    assert meta["jcamp_data_type"] == "RAMAN SPECTRUM"
    assert np.allclose(df["x"], [100, 101, 102, 103, 104])
    assert np.allclose(df["y"], [10, 20, 30, 40, 50])


def test_jcamp_yfactor_scales_y(tmp_path):
    content = AFFN_FILE.replace("##YFACTOR=1.0", "##YFACTOR=0.5")
    path = _write(tmp_path, "yfac.jdx", content)
    df, _ = iu.parse_jcamp(path)
    assert np.allclose(df["y"], [5, 10, 15, 20, 25])


SQZ_FILE = """\
##TITLE=SQZ compressed
##JCAMP-DX=4.24
##FIRSTX=0.0
##LASTX=4.0
##NPOINTS=5
##YFACTOR=1
##XYDATA=(X++(Y..Y))
0 A0B0C0D0E0
##END=
"""


def test_jcamp_sqz_compression(tmp_path):
    # SQZ: A=1,B=2,... so A0=10, B0=20, C0=30, D0=40, E0=50
    path = _write(tmp_path, "sqz.jdx", SQZ_FILE)
    df, _ = iu.parse_jcamp(path)
    assert np.allclose(df["y"], [10, 20, 30, 40, 50])
    assert np.allclose(df["x"], [0, 1, 2, 3, 4])


DIF_FILE = """\
##TITLE=DIF compressed
##JCAMP-DX=4.24
##FIRSTX=0.0
##LASTX=3.0
##NPOINTS=4
##YFACTOR=1
##XYDATA=(X++(Y..Y))
0 A0JJj
##END=
"""


def test_jcamp_dif_compression(tmp_path):
    # A0=10 absolute, then J=+1 -> 11, J=+1 -> 12, j=-1 -> 11
    path = _write(tmp_path, "dif.jdx", DIF_FILE)
    df, _ = iu.parse_jcamp(path)
    assert np.allclose(df["y"], [10, 11, 12, 11])


DIF_CHECKVALUE_FILE = """\
##TITLE=DIF with line-start check values
##JCAMP-DX=4.24
##FIRSTX=0.0
##LASTX=5.0
##NPOINTS=6
##YFACTOR=1
##XYDATA=(X++(Y..Y))
0 A0JJ
3 B2JJ
##END=
"""


def test_jcamp_dif_line_start_check_value_dropped(tmp_path):
    """Per the spec, in DIF mode each new line starts (after X) with an
    absolute Y that duplicates the previous line's final Y as a data-
    integrity check — it must be dropped, not duplicated. Line 1: 10,11,12;
    line 2 starts with check value B2=12 (dropped), then +1,+1 -> 13,14.
    Hmm — that yields 5 points for NPOINTS=6; use values so the sequence
    is 10,11,12 | check 12, +1->13, +1->14, and declared NPOINTS=5."""
    content = DIF_CHECKVALUE_FILE.replace("##NPOINTS=6", "##NPOINTS=5").replace("##LASTX=5.0", "##LASTX=4.0")
    path = _write(tmp_path, "difcheck.jdx", content)
    df, _ = iu.parse_jcamp(path)
    assert np.allclose(df["y"], [10, 11, 12, 13, 14])
    assert len(df) == 5


DUP_FILE = """\
##TITLE=DUP compressed
##JCAMP-DX=4.24
##FIRSTX=0.0
##LASTX=4.0
##NPOINTS=5
##YFACTOR=1
##XYDATA=(X++(Y..Y))
0 A0T B0U
##END=
"""


def test_jcamp_dup_compression(tmp_path):
    # A0=10, T(=2): 10 occurs twice total; B0=20, U(=3): 20 occurs 3 times.
    path = _write(tmp_path, "dup.jdx", DUP_FILE)
    df, _ = iu.parse_jcamp(path)
    assert np.allclose(df["y"], [10, 10, 20, 20, 20])


XYPOINTS_FILE = """\
##TITLE=XY pairs
##JCAMP-DX=4.24
##XFACTOR=1
##YFACTOR=2
##XYPOINTS=(XY..XY)
100.0, 1.0; 200.0, 2.0
300.0, 3.0
##END=
"""


def test_jcamp_xypoints_pairs_with_factors(tmp_path):
    path = _write(tmp_path, "pairs.jdx", XYPOINTS_FILE)
    df, _ = iu.parse_jcamp(path)
    assert np.allclose(df["x"], [100, 200, 300])
    assert np.allclose(df["y"], [2, 4, 6])  # YFACTOR=2


def test_jcamp_registry_integration(tmp_path):
    """load_any must route a .jdx file to the jcamp parser via the sniffer."""
    path = _write(tmp_path, "auto.jdx", AFFN_FILE)
    df, meta = iu.load_any(path, return_meta=True)
    assert meta["selected_parser"] == "jcamp"
    canon = meta["canonical_map"]
    assert np.allclose(df[canon["X"]], [100, 101, 102, 103, 104])


def test_jcamp_no_data_block_raises(tmp_path):
    path = _write(tmp_path, "empty.jdx", "##TITLE=x\n##JCAMP-DX=4.24\n##END=\n")
    with pytest.raises(ValueError, match="No ##XYDATA"):
        iu.parse_jcamp(path)


def test_jcamp_sniffer_rejects_non_jcamp(tmp_path):
    path = _write(tmp_path, "plain.txt", "100 1\n200 2\n")
    assert not iu.sniff_jcamp(path, "100 1\n200 2\n")
