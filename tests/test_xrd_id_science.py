"""Tests for xrd_id_science.py — the QualX-style XRD phase-identification
engine. Uses tiny synthetic QualX-format source databases (same schema as
real downloadable card databases), never real multi-GB ones."""
from __future__ import annotations

import sqlite3

import numpy as np
import pytest

import xrd.xrd_id_science as xid


def _make_source_sq(path, cards):
    """cards: [(id, name, mineral, formula, sg, quality, d_list, i_list, elements)]"""
    con = sqlite3.connect(str(path))
    con.executescript("""
        CREATE TABLE id (id int, name varchar, mineralname varchar, chemical_formula varchar,
                         spacegroup varchar, quality varchar, rir double, nrec int,
                         dvalue blob, intensita blob, nd int);
        CREATE TABLE chemical (id int, chemical_element varchar);
        CREATE TABLE infodb (id varchar, date varchar, ncard int, type varchar, source varchar);
    """)
    con.execute("INSERT INTO infodb VALUES ('t', '2026-01-01', ?, 'TEST', 'unit-test')", (len(cards),))
    for cid, name, mineral, formula, sg, quality, d, i, elements in cards:
        con.execute("INSERT INTO id VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (cid, name, mineral, formula, sg, quality, 1.0, 0,
                     ",".join(f"{v:.6f}" for v in d), ",".join(f"{v:.4f}" for v in i), len(d)))
        for el in elements:
            con.execute("INSERT INTO chemical VALUES (?,?)", (cid, el))
    con.commit()
    con.close()


QUARTZ_D = [3.342, 4.257, 1.8179, 2.457, 2.282, 1.5418, 1.3718]
QUARTZ_I = [100.0, 22.0, 14.0, 8.0, 8.0, 9.0, 8.0]
CALCITE_D = [3.035, 2.285, 2.095, 1.913, 1.875, 3.86]
CALCITE_I = [100.0, 18.0, 18.0, 17.0, 17.0, 12.0]


@pytest.fixture()
def unified_db(tmp_path):
    src1 = tmp_path / "src1.sq"
    _make_source_sq(src1, [
        (1010, "Quartz low", "Quartz", "Si O2", "P 32 2 1", "A", QUARTZ_D, QUARTZ_I, ["Si", "O"]),
        (1020, "Calcite", "Calcite", "C Ca O3", "R -3 c", "A", CALCITE_D, CALCITE_I, ["C", "Ca", "O"]),
        (1030, "no-lines junk", "", "", "", "C", [3.1], [100.0], []),  # < min_lines -> skipped
    ])
    src2 = tmp_path / "src2.sq"
    _make_source_sq(src2, [
        (77, " $GB-Quartz synthetic ", "", "Si O2", "P 32 2 1", "B", QUARTZ_D, QUARTZ_I, ["Si", "O"]),
    ])
    out = tmp_path / "unified.sq"
    counts = xid.build_xrd_database([(str(src1), "SRC1"), (str(src2), "SRC2")],
                                    out_path=str(out), log=lambda *a: None)
    assert counts == {"SRC1": 2, "SRC2": 1}
    return str(out)


def test_two_theta_d_round_trip():
    tt = np.array([20.0, 26.64, 60.0])
    d = xid.two_theta_to_d(tt)
    back = xid.d_to_two_theta(d)
    assert np.allclose(back, tt)
    # quartz's main line: d=3.342 Å -> 2θ ≈ 26.65° at Cu Kα
    assert xid.d_to_two_theta(np.array([3.342]))[0] == pytest.approx(26.65, abs=0.05)


def test_build_keeps_sources_codes_and_cleans_typeset_names(unified_db):
    con = sqlite3.connect(unified_db)
    rows = con.execute("SELECT source, source_code, name FROM cards ORDER BY card_id").fetchall()
    con.close()
    assert ("SRC1", "1010", "Quartz low") in rows
    assert ("SRC2", "77", "Quartz synthetic") in rows  # $GB- code stripped, original id kept
    summary = xid.database_summary(unified_db)
    assert summary["total_cards"] == 3
    assert summary["by_source"] == {"SRC1": 2, "SRC2": 1}


def test_search_match_identifies_quartz_from_measured_peaks(unified_db):
    # simulated measured pattern: quartz's lines at Cu Kα with small offsets
    tt = xid.d_to_two_theta(np.array(QUARTZ_D)) + 0.03
    results = xid.search_match(tt, QUARTZ_I, tol_two_theta=0.2, db_path=unified_db)
    assert results
    assert results[0].mineral == "Quartz" or "Quartz" in results[0].name
    assert results[0].fom > 0.9
    assert results[0].n_matched >= 6
    # Calcite scores far lower (only coincidental overlaps)
    calcite = [r for r in results if r.mineral == "Calcite"]
    assert not calcite or calcite[0].fom < 0.5
    # both quartz cards (both sources) are found
    quartz_sources = {r.source for r in results if "Quartz" in (r.mineral + r.name)}
    assert quartz_sources == {"SRC1", "SRC2"}


def test_search_match_element_filters(unified_db):
    tt = xid.d_to_two_theta(np.array(QUARTZ_D))
    with_ca = xid.search_match(tt, QUARTZ_I, elements_all=["Ca"], db_path=unified_db)
    assert all(r.mineral == "Calcite" for r in with_ca)
    no_si = xid.search_match(tt, QUARTZ_I, elements_none=["Si"], db_path=unified_db)
    assert all("Quartz" not in (r.mineral + r.name) for r in no_si)


def test_search_match_source_filter(unified_db):
    tt = xid.d_to_two_theta(np.array(QUARTZ_D))
    only2 = xid.search_match(tt, QUARTZ_I, sources=["SRC2"], db_path=unified_db)
    assert only2 and all(r.source == "SRC2" for r in only2)


def test_search_match_range_limits_card_coverage(unified_db):
    # only the strongest quartz line measured, in a narrow range: coverage of
    # the card must be computed against lines INSIDE that range only
    tt_main = float(xid.d_to_two_theta(np.array([3.342]))[0])
    results = xid.search_match([tt_main], [100.0], two_theta_range=(25.0, 28.0),
                               db_path=unified_db)
    quartz = next(r for r in results if "Quartz" in (r.mineral + r.name))
    assert quartz.cov_card == pytest.approx(1.0, abs=0.01)  # the only in-range line is matched


def test_search_match_missing_db_raises_helpfully(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_xrd_database"):
        xid.search_match([26.6], db_path=str(tmp_path / "absent.sq"))


def test_find_cards_by_text(unified_db):
    hits = xid.find_cards_by_text("quartz", db_path=unified_db)
    assert len(hits) == 2
    assert all(len(h["d"]) == len(QUARTZ_D) for h in hits)
    hits = xid.find_cards_by_text("Ca O3", db_path=unified_db)
    assert len(hits) == 1 and hits[0]["mineral"] == "Calcite"


# ---------------------------------------------------------------------------
# Database registry + multi-database probing
# ---------------------------------------------------------------------------

def test_sniff_sq_format(tmp_path, unified_db):
    src = tmp_path / "qualx_src.sq"
    _make_source_sq(src, [(1, "X", "", "Si O2", "", "A", QUARTZ_D, QUARTZ_I, ["Si", "O"])])
    assert xid.sniff_sq_format(str(src)) == "qualx"
    assert xid.sniff_sq_format(unified_db) == "prism"
    junk = tmp_path / "not_a_db.sq"
    junk.write_text("hello, not sqlite")
    assert xid.sniff_sq_format(str(junk)) is None
    assert xid.sniff_sq_format(str(tmp_path / "absent.sq")) is None


def test_register_qualx_database_converts_once(tmp_path):
    reg = str(tmp_path / "reg.json")
    imp = str(tmp_path / "imported")
    src = tmp_path / "MyCOD.sq"
    _make_source_sq(src, [(1010, "Quartz low", "Quartz", "Si O2", "P 32 2 1", "A",
                           QUARTZ_D, QUARTZ_I, ["Si", "O"])])
    entry = xid.register_database(str(src), registry_path=reg, import_dir=imp, log=lambda *a: None)
    assert entry["name"] == "MyCOD"
    assert entry["origin"] == str(src)
    assert xid.sniff_sq_format(entry["path"]) == "prism"
    # provenance tag inside the converted file = the registry name
    assert xid.database_summary(entry["path"])["by_source"] == {"MyCOD": 1}
    # re-registering the ORIGINAL path is a no-op, not a duplicate
    xid.register_database(str(src), registry_path=reg, import_dir=imp, log=lambda *a: None)
    assert len(xid.load_registry(reg)) == 1


def test_register_prism_database_in_place_no_copy(tmp_path, unified_db):
    import os
    reg = str(tmp_path / "reg2.json")
    entry = xid.register_database(unified_db, registry_path=reg, log=lambda *a: None)
    assert entry["path"] == os.path.abspath(unified_db)
    assert "origin" not in entry


def test_register_rejects_non_database(tmp_path):
    bad = tmp_path / "junk.sq"
    bad.write_text("definitely not sqlite")
    with pytest.raises(ValueError, match="not a recognizable"):
        xid.register_database(str(bad), registry_path=str(tmp_path / "r.json"))


def test_enable_disable_and_unregister(tmp_path, unified_db):
    import os
    reg = str(tmp_path / "reg3.json")
    xid.register_database(unified_db, name="DB A", registry_path=reg, log=lambda *a: None)
    assert xid.enabled_database_paths(reg) == [os.path.abspath(unified_db)]
    xid.set_database_enabled("DB A", False, registry_path=reg)
    assert xid.enabled_database_paths(reg) == []
    xid.unregister_database("DB A", registry_path=reg)
    assert xid.load_registry(reg) == []


def test_load_registry_migrates_legacy_unified_db(tmp_path, monkeypatch, unified_db):
    """A pre-registry setup (only the old fixed-location unified .sq) must
    keep working: first load_registry() call registers it automatically."""
    reg = tmp_path / "migrated_registry.json"
    monkeypatch.setattr(xid, "XRD_ID_REGISTRY_PATH", str(reg))
    monkeypatch.setattr(xid, "XRD_ID_DB_PATH", unified_db)
    entries = xid.load_registry()
    assert len(entries) == 1
    assert entries[0]["path"] == unified_db
    assert entries[0]["enabled"] is True
    assert reg.is_file()  # persisted, so the migration happens exactly once


def test_search_match_across_multiple_databases(tmp_path):
    """Quartz lives only in DB1, calcite only in DB2 — one search over both
    must find both, each hit tagged with the database it came from."""
    src1, src2 = tmp_path / "s1.sq", tmp_path / "s2.sq"
    _make_source_sq(src1, [(1, "Quartz", "Quartz", "Si O2", "", "A", QUARTZ_D, QUARTZ_I, ["Si", "O"])])
    _make_source_sq(src2, [(2, "Calcite", "Calcite", "C Ca O3", "", "A", CALCITE_D, CALCITE_I, ["C", "Ca", "O"])])
    db1, db2 = tmp_path / "db_quartz.sq", tmp_path / "db_calcite.sq"
    xid.build_xrd_database([(str(src1), "DB1")], out_path=str(db1), log=lambda *a: None)
    xid.build_xrd_database([(str(src2), "DB2")], out_path=str(db2), log=lambda *a: None)

    tt = np.concatenate([xid.d_to_two_theta(np.array(QUARTZ_D)),
                         xid.d_to_two_theta(np.array(CALCITE_D))])
    ii = np.concatenate([QUARTZ_I, CALCITE_I])
    results = xid.search_match(tt, ii, db_paths=[str(db1), str(db2)])
    minerals = {r.mineral for r in results}
    assert {"Quartz", "Calcite"} <= minerals
    assert {r.db for r in results} == {"db_quartz", "db_calcite"}


def test_find_cards_by_text_across_multiple_databases(tmp_path):
    src1, src2 = tmp_path / "t1.sq", tmp_path / "t2.sq"
    _make_source_sq(src1, [(1, "Quartz", "Quartz", "Si O2", "", "A", QUARTZ_D, QUARTZ_I, ["Si", "O"])])
    _make_source_sq(src2, [(2, "Quartz high", "Quartz", "Si O2", "", "B", QUARTZ_D, QUARTZ_I, ["Si", "O"])])
    db1, db2 = tmp_path / "ta.sq", tmp_path / "tb.sq"
    xid.build_xrd_database([(str(src1), "A")], out_path=str(db1), log=lambda *a: None)
    xid.build_xrd_database([(str(src2), "B")], out_path=str(db2), log=lambda *a: None)
    hits = xid.find_cards_by_text("quartz", db_paths=[str(db1), str(db2)])
    assert len(hits) == 2
    assert {h["db"] for h in hits} == {"ta", "tb"}


def test_crystal_system_inference():
    cases = {
        "Fm-3m": "cubic", "Im-3m": "cubic", "P 41 32": "cubic", "Ia-3d": "cubic",
        "P 63/m m c": "hexagonal", "P 6222": "hexagonal",
        "R -3 c": "trigonal", "P 32 2 1": "trigonal", "P -3 m 1": "trigonal",
        "I 41/a m d": "tetragonal", "P 42/m n m": "tetragonal",
        "P n m a": "orthorhombic", "C m c m": "orthorhombic", "P 21 21 21": "orthorhombic",
        "P 21/c": "monoclinic", "C 2/m": "monoclinic", "P 2": "monoclinic",
        "P -1": "triclinic", "P 1": "triclinic",
        "": "",
    }
    for sg, expected in cases.items():
        assert xid.crystal_system(sg) == expected, f"{sg!r} -> {xid.crystal_system(sg)!r}, expected {expected!r}"


def test_search_match_quality_system_and_spacegroup_filters(unified_db):
    tt = xid.d_to_two_theta(np.array(QUARTZ_D))
    # quartz exists as quality A (SRC1) and quality B (SRC2), both P 32 2 1
    only_b = xid.search_match(tt, QUARTZ_I, qualities=["B"], db_path=unified_db)
    assert only_b and all(r.quality == "B" for r in only_b)
    # nothing in the fixture DB is cubic
    assert xid.search_match(tt, QUARTZ_I, crystal_systems=["cubic"], db_path=unified_db) == []
    sg_hits = xid.search_match(tt, QUARTZ_I, spacegroup_contains="32 2 1", db_path=unified_db)
    assert sg_hits and all("32 2 1" in r.spacegroup for r in sg_hits)


def test_search_match_dedups_identical_cards_across_databases(tmp_path):
    """The same card (code + phase + space group + quality) carried by two
    registered databases must appear once, not twice (user request)."""
    src = tmp_path / "same.sq"
    _make_source_sq(src, [(1010, "Quartz low", "Quartz", "Si O2", "P 32 2 1", "A",
                           QUARTZ_D, QUARTZ_I, ["Si", "O"])])
    db1, db2 = tmp_path / "d1.sq", tmp_path / "d2.sq"
    xid.build_xrd_database([(str(src), "TAG1")], out_path=str(db1), log=lambda *a: None)
    xid.build_xrd_database([(str(src), "TAG2")], out_path=str(db2), log=lambda *a: None)
    tt = xid.d_to_two_theta(np.array(QUARTZ_D))
    res = xid.search_match(tt, QUARTZ_I, db_paths=[str(db1), str(db2)])
    assert len([r for r in res if r.mineral == "Quartz"]) == 1


def test_database_summary_lists_qualities(unified_db):
    # kept cards: two quality-A (SRC1) + one quality-B (SRC2); the junk
    # quality-C card is skipped at build time (< min_lines)
    assert xid.database_summary(unified_db)["qualities"] == ["A", "B"]


def test_find_cards_by_elements_matches_regardless_of_formula_order(tmp_path):
    """User report: text search for 'TiO2' missed cards written 'O2 Ti'."""
    src = tmp_path / "src_ti.sq"
    _make_source_sq(src, [
        (1, "Anatase", "Anatase", "O2 Ti", "I 41/a m d", "A", [3.52, 2.38, 1.89], [100.0, 20.0, 30.0], ["Ti", "O"]),
        (2, "Rutile", "Rutile", "Ti O2", "P 42/m n m", "A", [3.25, 2.49, 1.69], [100.0, 50.0, 60.0], ["Ti", "O"]),
        (3, "LiTi oxide", "", "Li0.5 O2 Ti", "", "B", [4.0, 2.0, 1.5], [100.0, 40.0, 20.0], ["Li", "Ti", "O"]),
    ])
    db = tmp_path / "uni_ti.sq"
    xid.build_xrd_database([(str(src), "T")], out_path=str(db), log=lambda *a: None)

    exact = xid.find_cards_by_elements("TiO2", mode="exact", db_path=str(db))
    assert sorted(h["mineral"] or h["formula"] for h in exact) == ["Anatase", "Rutile"]
    contains = xid.find_cards_by_elements("TiO2", mode="contains", db_path=str(db))
    assert len(contains) == 3  # the Li-bearing card included
    assert xid.find_cards_by_elements("Zr2O", mode="exact", db_path=str(db)) == []
