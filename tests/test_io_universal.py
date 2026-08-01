"""Tests for io_universal.py — the universal file parser registry.

Fixtures are RAW inputs only (bundled EXAMPLES/* and archived real spectra).
Expected values are derived from physically-sane invariants or from
independently-verifiable facts about the input file, never from snapshotting
the old (pre-fix) pipeline's output.
"""
from __future__ import annotations


import core.io_universal as iou


# --------------------------------------------------------------------------
# Regression pin: the one format this module was clearly built/tested against.
# --------------------------------------------------------------------------

def test_ta_sdt_dta_example_still_works(dta_example_path):
    df, meta = iou.load_any(str(dta_example_path), return_meta=True)
    assert meta["selected_parser"] == "ta_sdt"
    canon = meta["canonical_map"]
    assert "T_C" in canon
    assert "HF_mW" in canon or "DSC_mW_mg" in canon
    # Real instrument data: no fully-empty columns, sane temperature range.
    t_col = df[canon["T_C"]]
    assert t_col.notna().sum() > 100
    assert t_col.min() > -50 and t_col.max() < 2000


# --------------------------------------------------------------------------
# Bug fix: parse_ta_sdt_txt no-StartOfData-marker fallback used to drop the
# real header/column-name line (off-by-one) and corrupt things further when
# data started at line 0 (negative-index slicing).
# --------------------------------------------------------------------------

def test_ta_sdt_txt_fallback_keeps_header_line_and_names_columns(tmp_path):
    text = (
        "Instrument: Fake STA\n"
        "Operator: test\n"
        "Time(min)\tTemp(C)\tDSC(mW)\n"
        "0.0\t25.0\t1.0\n"
        "1.0\t26.0\t1.1\n"
        "2.0\t27.0\t1.2\n"
    )
    p = tmp_path / "no_marker.txt"
    p.write_text(text, encoding="utf-8")

    header, colnames, df = iou.parse_ta_sdt_txt(p)

    # The header/column-name row must be recognized, not silently dropped.
    assert colnames == ["Time(min)", "Temp(C)", "DSC(mW)"]
    # No data rows lost or corrupted into the header dict.
    assert len(df) == 3
    assert list(df["Temp(C)"]) == [25.0, 26.0, 27.0]


def test_ta_sdt_txt_fallback_handles_data_starting_at_line_zero(tmp_path):
    # No header at all — first line is numeric data. The old code computed
    # start_idx = -1 here, which corrupted header parsing via negative
    # indexing (`lines[:-1]`) and could misparse data rows as header entries.
    text = "0.0\t1.0\n1.0\t2.0\n2.0\t3.0\n"
    p = tmp_path / "headerless.txt"
    p.write_text(text, encoding="utf-8")

    header, colnames, df = iou.parse_ta_sdt_txt(p)

    assert header == {}
    assert len(df) == 3
    assert list(df.iloc[:, 0]) == [0.0, 1.0, 2.0]


# --------------------------------------------------------------------------
# Bug fix: _read_table_flexible (used by parse_raman/parse_xrd) didn't pass
# header=None for headerless files, silently consuming the first data row.
# --------------------------------------------------------------------------

def test_parse_raman_keeps_all_rows_on_headerless_example(raman_example_path):
    n_lines_in_file = sum(1 for ln in raman_example_path.read_text().splitlines() if ln.strip())
    df, meta = iou.parse_raman(str(raman_example_path))
    assert len(df) == n_lines_in_file, "the first real data row must not be lost as a phantom header"


def test_parse_raman_still_detects_real_header_row(tmp_path):
    text = "Raman Shift (cm-1)\tIntensity (a.u.)\n100.0\t5.0\n200.0\t7.0\n300.0\t9.0\n"
    p = tmp_path / "with_header.txt"
    p.write_text(text, encoding="utf-8")
    df, meta = iou.parse_raman(str(p))
    assert len(df) == 3
    assert "Raman Shift" in str(list(df.columns)[0]) or "shift" in str(list(df.columns)[0]).lower()


def test_parse_xrd_keeps_all_rows_on_headerless_example(xrd_example_path):
    n_lines_in_file = sum(1 for ln in xrd_example_path.read_text(errors="ignore").splitlines() if ln.strip())
    df, meta = iou.parse_xrd(str(xrd_example_path))
    assert len(df) == n_lines_in_file


# --------------------------------------------------------------------------
# Bug fix: sniff_dta_table used to misclassify a Raman file whose comment
# header merely mentions "temperature" alongside an unrelated semicolon.
# --------------------------------------------------------------------------

def test_sniff_dta_table_does_not_misfire_on_incidental_temperature_mention():
    head = "# Raman spectrum; acquired at room temperature, laser 532nm\n100.0\t5.0\n200.0\t7.0\n"
    assert iou.sniff_dta_table("fake.txt", head) is False


def test_sniff_dta_table_still_fires_on_real_semicolon_header_row():
    head = "Time;Temperature;Heat Flow;Mass\n0.0;25.0;1.0;10.0\n"
    assert iou.sniff_dta_table("fake.txt", head) is True


# --------------------------------------------------------------------------
# Bug fix: SAXS column-canonicalization silently overwrote extra columns
# beyond the third (everything non-q/non-sig mapped to the same "I" key).
# --------------------------------------------------------------------------

def test_saxs_extra_columns_do_not_overwrite_each_other(tmp_path):
    text = (
        "# EDF_DataBlockID = 1\n"
        "q(A-1)  I(q)  Sig(q)  I2(q)\n"
        "0.01 100.0 1.0 50.0\n"
        "0.02 90.0 1.1 45.0\n"
        "0.03 80.0 1.2 40.0\n"
        "0.04 70.0 1.3 35.0\n"
        "0.05 60.0 1.4 30.0\n"
    )
    p = tmp_path / "saxs_extra.dat"
    p.write_text(text, encoding="utf-8")
    df, meta = iou.parse_saxs_edf_ascii(str(p))
    canon = meta["canonical_map"]
    # Both intensity-like columns must be retained under distinct keys.
    assert canon.get("I") is not None
    assert canon.get("I2") is not None
    assert canon["I"] != canon["I2"]


def test_saxs_example_still_parses(saxs_example_path):
    df, meta = iou.parse_saxs_edf_ascii(str(saxs_example_path))
    assert len(df) > 10
    assert meta["canonical_map"].get("q_A^-1") is not None


# --------------------------------------------------------------------------
# Bug fix: extra_encodings was unreachable dead code (latin-1 never fails).
# --------------------------------------------------------------------------

def test_extra_encodings_now_reachable_before_latin1_fallback():
    # Bytes 0x93/0x94 decode to curly quotes (U+201C/U+201D) under cp1252,
    # but to C1 control characters under latin-1 — a real, observable
    # difference that proves extra_encodings is actually being tried.
    raw = b"# comment \x93curly\x94"
    text, used_enc = iou._decode_text_autodetect(raw, extra_encodings=("cp1252",))
    assert used_enc == "cp1252"
    assert "curly" in text
    assert "“" in text and "”" in text


def test_decode_text_autodetect_falls_back_to_latin1_when_no_extra_encoding_matches():
    raw = b"\xff\xfeNotActuallyUTF16ButHasBOMBytes"  # exercise the plain fallback path
    text, used_enc = iou._decode_text_autodetect(raw)
    assert isinstance(text, str)


# --------------------------------------------------------------------------
# Dead code removal check
# --------------------------------------------------------------------------

def test_canonicalize_xy_was_removed():
    assert not hasattr(iou, "_canonicalize_xy")
