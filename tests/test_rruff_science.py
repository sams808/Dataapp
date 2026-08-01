"""Tests for rruff_science.py (M10) — RRUFF mineral Raman database ingest.

Format verified directly against the real, current rruff.net bulk download
(2026-07-15) rather than assumed — see rruff_science.py's module docstring.
Uses EXAMPLES/RRUFF_fair_oriented_sample.zip, a real (not synthetic) small
RRUFF ZIP checked in as a fixture.
"""
from __future__ import annotations

import json

import pytest

import raman.rruff_science as rs


# --------------------------------------------------------------------------
# Filename parsing
# --------------------------------------------------------------------------

def test_parse_rruff_filename_oriented_with_polarization():
    fields = rs.parse_rruff_filename(
        "Annite__R060211-3__Raman__514__0-000__depolarized__Raman_Data_Processed__989cc43b1a7a123bfdaaa6d30516.txt"
    )
    assert fields["mineral"] == "Annite"
    assert fields["rruff_id"] == "R060211-3"
    assert fields["wavelength_nm"] == 514.0
    assert fields["orientation_deg"] == "0-000"
    assert fields["polarization"] == "depolarized"
    assert fields["data_kind"] == "Processed"


def test_parse_rruff_filename_unoriented_has_empty_middle_fields():
    fields = rs.parse_rruff_filename(
        "Abramovite__R070037__Raman__785______Raman_Data_RAW__b8534873d9d9446834252f023c0e.txt"
    )
    assert fields["mineral"] == "Abramovite"
    assert fields["rruff_id"] == "R070037"
    assert fields["wavelength_nm"] == 785.0
    assert fields["orientation_deg"] is None
    assert fields["polarization"] is None
    assert fields["data_kind"] == "RAW"


def test_parse_rruff_filename_non_matching_returns_empty_dict():
    assert rs.parse_rruff_filename("not_a_rruff_file.txt") == {}
    assert rs.parse_rruff_filename("Weird__Format__NotRaman__x.txt") == {}


def test_parse_rruff_filename_broad_scan_from_lr_raman_zip():
    """LR-Raman.zip (confirmed by downloading/inspecting the real 227MB
    archive: 9,941 low-resolution broad-range survey scans) uses
    'Broad_Scan' as the scan-type field where the category ZIPs say
    'Raman' — same positional schema otherwise."""
    fields = rs.parse_rruff_filename(
        "Abramovite__R070037__Broad_Scan__532__0__unoriented__Raman_Data_Processed__db2de3a53da39189f1a8dfee39f1.txt"
    )
    assert fields["mineral"] == "Abramovite"
    assert fields["scan_type"] == "Broad_Scan"
    assert fields["wavelength_nm"] == 532.0
    assert fields["data_kind"] == "Processed"

    high_res = rs.parse_rruff_filename(
        "Quartz__R040031__Raman__532__0-000____Raman_Data_Processed__xyz.txt"
    )
    assert high_res["scan_type"] == "Raman"


# --------------------------------------------------------------------------
# Header + data parsing
# --------------------------------------------------------------------------

_REAL_TEXT = """\
##NAMES=Corundum
##RRUFFID=R060020
##IDEAL CHEMISTRY=Al_2_O_3_
##LOCALITY=Yogo Gulch, Montana, USA
##OWNER=RRUFF
##SOURCE=American Museum of Natural History
##ORIENTATION=Laser parallel to c* (0 0 1)
##CELL PARAMETERS=a: 4.76274 b: 4.76274 c: 13.0016 alpha: 90 beta: 90 gamma: 120 volume: 255.412 crystal system: hexagonal
##FILETYPE=Raman Processed
##RAMAN WAVELENGTH=514

179.5050, 136.8613
180.7290, 44.83008
181.9520, 48.55469
"""

_MINIMAL_TEXT = """\
##NAMES=Abramovite
##RRUFFID=R070037
##FILETYPE=Raman RAW
##RAMAN WAVELENGTH=785

137.2198, 1591.066
137.7019, 1623.034
"""


def test_parse_rruff_txt_extracts_full_header_and_data():
    spectrum = rs.parse_rruff_txt(_REAL_TEXT, source_filename="Corundum__R060020__Raman__514__0-000____Raman_Data_Processed__abc.txt")
    assert spectrum.mineral == "Corundum"
    assert spectrum.rruff_id == "R060020"
    assert spectrum.wavelength_nm == 514.0
    assert spectrum.ideal_chemistry == "Al_2_O_3_"
    assert spectrum.locality == "Yogo Gulch, Montana, USA"
    assert spectrum.orientation_text.startswith("Laser parallel")
    assert spectrum.cell_parameters is not None
    assert len(spectrum.x) == 3
    assert spectrum.x[0] == pytest.approx(179.5050)
    assert spectrum.y[0] == pytest.approx(136.8613)


def test_parse_rruff_txt_handles_missing_optional_fields_gracefully():
    """Real unoriented/unconfirmed samples can omit ORIENTATION, MEASURED
    CHEMISTRY, PIN_ID entirely — must not raise or require them."""
    spectrum = rs.parse_rruff_txt(_MINIMAL_TEXT, source_filename="Abramovite__R070037__Raman__785______Raman_Data_RAW__b85.txt")
    assert spectrum.mineral == "Abramovite"
    assert spectrum.orientation_text is None
    assert spectrum.locality is None
    assert spectrum.wavelength_nm == 785.0
    assert len(spectrum.x) == 2


def test_parse_rruff_txt_falls_back_to_filename_when_header_sparse():
    """If NAMES/RRUFFID were ever absent from the header, the filename's
    own fields should still let mineral/rruff_id/wavelength resolve."""
    text = "##FILETYPE=Raman Processed\n\n100.0, 1.0\n200.0, 2.0\n"
    spectrum = rs.parse_rruff_txt(text, source_filename="Quartz__R040031__Raman__532__0-000____Raman_Data_Processed__xyz.txt")
    assert spectrum.mineral == "Quartz"
    assert spectrum.rruff_id == "R040031"
    assert spectrum.wavelength_nm == 532.0


def test_parse_rruff_txt_prefers_header_wavelength_over_mismatched_filename():
    """Regression guard for a real discrepancy found during the full-corpus
    ingest: Gysinite-(Ce) R250121's filename says wavelength "53" (a
    genuine truncated-digit typo in that one filename on rruff.net) while
    its own ##RAMAN WAVELENGTH header correctly says 532. The header must
    win — it's curated structured metadata, the filename is a derived,
    manually-typeable string."""
    text = "##NAMES=Gysinite-(Ce)\n##RRUFFID=R250121\n##RAMAN WAVELENGTH=532\n\n100.0, 1.0\n"
    spectrum = rs.parse_rruff_txt(text, source_filename="Gysinite-(Ce)__R250121__Raman__53______Raman_Data_Processed__abc.txt")
    assert spectrum.wavelength_nm == 532.0


# --------------------------------------------------------------------------
# Real-ZIP ingest (EXAMPLES/RRUFF_fair_oriented_sample.zip, 25 real spectra)
# --------------------------------------------------------------------------

def test_ingest_zip_on_real_sample_produces_sane_records(rruff_sample_zip_path, tmp_path):
    records = rs.ingest_zip(str(rruff_sample_zip_path), raw_dir=str(tmp_path / "raw"), category="fair_oriented")
    assert len(records) == 25

    minerals = {r["mineral"] for r in records}
    assert "Corundum" in minerals
    assert "Annite" in minerals

    for r in records:
        assert r["wavelength_nm"] == 514.0  # this sample ZIP is all 514nm
        assert r["category"] == "fair_oriented"
        assert r["x_min"] is not None and r["x_max"] is not None
        assert r["x_min"] < r["x_max"]

    processed = [r for r in records if r["data_kind"].lower().startswith("process")]
    assert len(processed) > 0
    assert all(isinstance(r["peaks"], list) for r in processed)
    assert any(len(r["peaks"]) > 0 for r in processed)


def test_ingest_zip_writes_raw_text_files_for_later_overlay(rruff_sample_zip_path, tmp_path):
    raw_dir = tmp_path / "raw"
    records = rs.ingest_zip(str(rruff_sample_zip_path), raw_dir=str(raw_dir), category="fair_oriented")
    for r in records:
        assert r["raw_path"]
        with open(r["raw_path"], encoding="utf-8") as f:
            content = f.read()
        assert "##NAMES=" in content


def test_build_index_and_load_index_round_trip(rruff_sample_zip_path, tmp_path):
    cache_dir = tmp_path / "rruff_cache"
    count = rs.build_index([(str(rruff_sample_zip_path), "fair_oriented")], cache_dir=str(cache_dir))
    assert count == 25

    loaded = rs.load_index(cache_dir=str(cache_dir))
    assert len(loaded) == 25
    assert (cache_dir / "index.json").is_file()

    with open(cache_dir / "index.json", encoding="utf-8") as f:
        raw_json = json.load(f)
    assert raw_json == loaded


def test_load_index_missing_cache_returns_empty_list(tmp_path):
    assert rs.load_index(cache_dir=str(tmp_path / "does_not_exist")) == []


def test_index_summary_reports_mineral_and_wavelength_coverage(rruff_sample_zip_path, tmp_path):
    records = rs.ingest_zip(str(rruff_sample_zip_path), raw_dir=str(tmp_path / "raw"), category="fair_oriented")
    summary = rs.index_summary(records)
    assert summary["n_spectra"] == 25
    assert summary["n_minerals"] >= 4  # Annite, Corundum, Scholzite, Spinel
    assert summary["wavelengths_nm"] == [514.0]


def test_citation_constants_are_nonempty_strings():
    assert "RRUFF" in rs.RRUFF_CITATION
    assert "Lafuente" in rs.RRUFF_CITATION
    assert len(rs.RRUFF_ATTRIBUTION_NOTE) > 0


# --------------------------------------------------------------------------
# M12 match-assist scoring
# --------------------------------------------------------------------------

def test_score_match_counts_peaks_within_tolerance():
    query = [100.0, 500.0, 1000.0]
    candidate = [102.0, 501.0, 2000.0]  # 2 of 3 within tolerance=10
    result = rs.score_match(query, candidate, tolerance=10.0)
    assert result["matched"] == 2
    assert result["fraction"] == pytest.approx(2 / 3)


def test_score_match_empty_query_returns_zero():
    assert rs.score_match([], [100.0], tolerance=10.0) == {"matched": 0, "fraction": 0.0}


def test_score_match_no_overlap_returns_zero():
    result = rs.score_match([100.0], [500.0, 600.0], tolerance=5.0)
    assert result["matched"] == 0
    assert result["fraction"] == 0.0


def test_rank_rruff_matches_orders_best_first_and_excludes_zero_matches():
    index = [
        {"mineral": "Best", "peaks": [100.0, 200.0, 300.0]},
        {"mineral": "Partial", "peaks": [100.0, 999.0, 999.0]},
        {"mineral": "NoMatch", "peaks": [700.0, 800.0]},
        {"mineral": "NoPeaks", "peaks": []},
    ]
    query = [100.0, 200.0, 300.0]
    ranked = rs.rank_rruff_matches(query, index, tolerance=5.0, top_n=10)

    names = [r["mineral"] for r in ranked]
    assert names[0] == "Best"
    assert "NoMatch" not in names
    assert "NoPeaks" not in names
    assert ranked[0]["matched_peaks"] == 3
    assert ranked[0]["match_fraction"] == pytest.approx(1.0)


def test_rank_rruff_matches_respects_top_n():
    index = [{"mineral": f"M{i}", "peaks": [100.0]} for i in range(50)]
    ranked = rs.rank_rruff_matches([100.0], index, tolerance=5.0, top_n=5)
    assert len(ranked) == 5


def test_rank_rruff_matches_preserves_original_record_fields():
    index = [{"mineral": "Quartz", "rruff_id": "R040031", "wavelength_nm": 532.0, "peaks": [464.0]}]
    ranked = rs.rank_rruff_matches([464.0], index, tolerance=5.0)
    assert ranked[0]["rruff_id"] == "R040031"
    assert ranked[0]["wavelength_nm"] == 532.0


# --------------------------------------------------------------------------
# AMCSD CIF companion cache (RRUFF -> CIF overlay handoff)
# --------------------------------------------------------------------------

_MINIMAL_CIF = """\
data_global
_chemical_name_mineral 'Quartz'
_cell_length_a 4.913
_cell_length_b 4.913
_cell_length_c 5.405
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 120
"""


def test_ingest_amcsd_cif_zip_and_lookup(tmp_path):
    import zipfile as _zip
    zip_path = tmp_path / "cif.zip"
    with _zip.ZipFile(zip_path, "w") as zf:
        zf.writestr("Quartz__0000789.cif", _MINIMAL_CIF)
        zf.writestr("Quartz__0000790.cif", _MINIMAL_CIF)
        zf.writestr("Gysinite-(Ce)__0001234.cif", _MINIMAL_CIF.replace("Quartz", "Gysinite-(Ce)"))

    cache = tmp_path / "amcsd_cache"
    n = rs.ingest_amcsd_cif_zip(str(zip_path), cache_dir=str(cache))
    assert n == 3

    quartz = rs.find_cifs_for_mineral("quartz", cache_dir=str(cache))
    assert len(quartz) == 2
    assert all(p.endswith(".cif") for p in quartz)

    # Punctuation/case-insensitive lookup for hyphenated/suffixed minerals.
    gys = rs.find_cifs_for_mineral("GYSINITE-(CE)", cache_dir=str(cache))
    assert len(gys) == 1

    assert rs.find_cifs_for_mineral("unobtainium", cache_dir=str(cache)) == []


def test_find_cifs_for_mineral_missing_cache_returns_empty(tmp_path):
    assert rs.find_cifs_for_mineral("quartz", cache_dir=str(tmp_path / "nope")) == []


# --------------------------------------------------------------------------
# No-Python-needed download (colleagues on the portable exe, no network in
# tests: urllib.request.urlopen is monkeypatched throughout).
# --------------------------------------------------------------------------

class _FakeResponse:
    """Enough of urllib's response object for _download_file: length,
    chunked read(), and context-manager protocol."""

    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0
        self.length = len(data)

    def read(self, n):
        chunk = self._data[self._pos:self._pos + n]
        self._pos += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_safe_print_never_raises(monkeypatch):
    """The exact PyInstaller --windowed failure mode: sys.stdout is None,
    so a bare print() raises AttributeError deep inside a background job."""
    monkeypatch.setattr("builtins.print", lambda *a, **k: (_ for _ in ()).throw(AttributeError("no stdout")))
    rs._safe_print("this must not raise")  # no exception = pass


def test_download_file_writes_and_skips_when_already_present(tmp_path, monkeypatch):
    import urllib.request
    calls = []

    def fake_urlopen(req, timeout=60):
        calls.append(req.full_url)
        return _FakeResponse(b"hello world")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    dest = tmp_path / "x.zip"
    logs = []
    rs._download_file("http://example.test/x.zip", str(dest), log=logs.append)
    assert dest.read_bytes() == b"hello world"
    assert not (tmp_path / "x.zip.part").exists()  # .part renamed away on success
    assert len(calls) == 1
    assert any("100%" in m for m in logs)

    # already downloaded -> no second network call
    rs._download_file("http://example.test/x.zip", str(dest), log=logs.append)
    assert len(calls) == 1


def test_download_file_leaves_no_dest_on_failure(tmp_path, monkeypatch):
    import urllib.request

    def raising_urlopen(req, timeout=60):
        raise OSError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", raising_urlopen)
    dest = tmp_path / "y.zip"
    with pytest.raises(OSError):
        rs._download_file("http://example.test/y.zip", str(dest))
    assert not dest.exists()  # never renamed from .part -> a retry won't think it's done


def test_category_url_maps_broad_scan_specially():
    assert rs._category_url("excellent_oriented") == rs.RRUFF_DOWNLOAD_BASE + "excellent_oriented.zip"
    assert rs._category_url(rs.RRUFF_BROAD_SCAN_CATEGORY) == rs.RRUFF_DOWNLOAD_BASE + "LR-Raman.zip"


def test_default_categories_are_the_seven_non_empty_ones():
    assert len(rs.RRUFF_CATEGORIES) == 7
    assert "poor_oriented" not in rs.RRUFF_CATEGORIES  # confirmed not to exist on the server
    assert rs.RRUFF_BROAD_SCAN_CATEGORY not in rs.RRUFF_CATEGORIES  # opt-in only


def test_download_rruff_zips_continues_past_one_failure(tmp_path, monkeypatch):
    calls = []

    def fake_download(url, dest, log=print, **kw):
        calls.append(url)
        if "fair_oriented" in url:
            raise OSError("simulated network failure")
        with open(dest, "wb") as f:
            f.write(b"zip-bytes")

    monkeypatch.setattr(rs, "_download_file", fake_download)
    logs = []
    out = rs.download_rruff_zips(str(tmp_path), categories=["excellent_oriented", "fair_oriented"], log=logs.append)
    assert [cat for _, cat in out] == ["excellent_oriented"]
    assert any("FAILED" in m for m in logs)
    assert len(calls) == 2  # both attempted despite the failure


def test_download_and_build_rruff_cache_orchestrates_download_then_build(tmp_path, monkeypatch):
    calls = {}

    def fake_download_zips(target_dir, categories=None, log=print):
        calls["target_dir"] = target_dir
        calls["categories"] = categories
        return [(str(tmp_path / "a.zip"), "excellent_oriented")]

    def fake_build_index(zips, cache_dir=rs.RRUFF_CACHE_DIR, log=print):
        calls["zips"] = zips
        calls["cache_dir"] = cache_dir
        return 123

    monkeypatch.setattr(rs, "download_rruff_zips", fake_download_zips)
    monkeypatch.setattr(rs, "build_index", fake_build_index)
    n = rs.download_and_build_rruff_cache(cache_dir=str(tmp_path / "cache"), categories=["excellent_oriented"])
    assert n == 123
    assert calls["categories"] == ["excellent_oriented"]
    assert calls["cache_dir"] == str(tmp_path / "cache")
    assert calls["zips"] == [(str(tmp_path / "a.zip"), "excellent_oriented")]


def test_download_and_build_rruff_cache_raises_when_every_download_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "download_rruff_zips", lambda *a, **k: [])
    with pytest.raises(RuntimeError, match="internet connection"):
        rs.download_and_build_rruff_cache(cache_dir=str(tmp_path))


def test_download_and_build_amcsd_cache_orchestrates_download_then_ingest(tmp_path, monkeypatch):
    calls = {}

    def fake_download_file(url, dest, log=print):
        calls["url"] = url
        calls["dest"] = dest
        with open(dest, "wb") as f:
            f.write(b"zip-bytes")

    def fake_ingest(zip_path, cache_dir=rs.AMCSD_CACHE_DIR, log=print):
        calls["zip_path"] = zip_path
        calls["cache_dir"] = cache_dir
        return 456

    monkeypatch.setattr(rs, "_download_file", fake_download_file)
    monkeypatch.setattr(rs, "ingest_amcsd_cif_zip", fake_ingest)
    n = rs.download_and_build_amcsd_cache(cache_dir=str(tmp_path / "amcsd"))
    assert n == 456
    assert calls["url"] == rs.AMCSD_DOWNLOAD_URL
    assert calls["cache_dir"] == str(tmp_path / "amcsd")


def test_ingested_amcsd_cif_feeds_bragg_generation(tmp_path):
    """End of the handoff chain: an ingested AMCSD CIF must parse through
    cif_tools and yield Bragg peaks."""
    import zipfile as _zip
    from raman.cif_tools import bragg_peaks_from_cif_generic

    zip_path = tmp_path / "cif.zip"
    with _zip.ZipFile(zip_path, "w") as zf:
        zf.writestr("Quartz__0000789.cif", _MINIMAL_CIF)
    cache = tmp_path / "amcsd_cache"
    rs.ingest_amcsd_cif_zip(str(zip_path), cache_dir=str(cache))

    path = rs.find_cifs_for_mineral("Quartz", cache_dir=str(cache))[0]
    peaks = bragg_peaks_from_cif_generic(path, two_theta_max=80.0, hkl_max=4, use_cache=False)
    assert len(peaks) > 5
    two_thetas = [tt for tt, _hkl, _d in peaks]
    assert any(25 < tt < 28 for tt in two_thetas)  # quartz (101) ~26.6 deg (Cu Ka)


def test_filter_rruff_index_by_each_criterion():
    from raman.rruff_science import filter_rruff_index
    index = [
        {"mineral": "A", "wavelength_nm": 532.0, "orientation_deg": "90", "scan_type": "Raman", "category": "excellent_oriented"},
        {"mineral": "B", "wavelength_nm": 532.6, "orientation_deg": None, "scan_type": "Raman", "category": "fair_unoriented"},
        {"mineral": "C", "wavelength_nm": 785.0, "orientation_deg": "", "scan_type": "Broad_Scan", "category": "lr_broad_scan"},
        {"mineral": "D", "wavelength_nm": None, "orientation_deg": None, "scan_type": "Raman", "category": "unrated_unoriented"},
    ]
    # wavelength with the default ±2 nm tolerance groups 532 and 532.6;
    # records with no recorded λ are excluded when a λ filter is active
    assert [r["mineral"] for r in filter_rruff_index(index, wavelength_nm=532.0)] == ["A", "B"]
    assert [r["mineral"] for r in filter_rruff_index(index, wavelength_nm=785.0)] == ["C"]
    # orientation: empty string counts as unoriented
    assert [r["mineral"] for r in filter_rruff_index(index, oriented=True)] == ["A"]
    assert [r["mineral"] for r in filter_rruff_index(index, oriented=False)] == ["B", "C", "D"]
    # scan type exact, quality by prefix
    assert [r["mineral"] for r in filter_rruff_index(index, scan_type="Broad_Scan")] == ["C"]
    assert [r["mineral"] for r in filter_rruff_index(index, quality="excellent")] == ["A"]
    # no constraints = everything; combined constraints intersect
    assert len(filter_rruff_index(index)) == 4
    assert [r["mineral"] for r in filter_rruff_index(index, wavelength_nm=532.0, oriented=False)] == ["B"]


def test_pack_and_unpack_rruff_database_round_trip(tmp_path):
    """User request: one shareable file for the whole RRUFF cache."""
    import json
    from raman.rruff_science import pack_rruff_database, unpack_rruff_database
    src_cache = tmp_path / "cache_a"
    raw = src_cache / "raw"
    raw.mkdir(parents=True)
    (raw / "Quartz__R1__Raman__532____Processed__x.txt").write_text("##NAMES=Quartz\n100.0, 5.0\n")
    index = [{"mineral": "Quartz", "rruff_id": "R1", "peaks": [464.0],
              "raw_path": str(raw / "Quartz__R1__Raman__532____Processed__x.txt")}]
    (src_cache / "index.json").write_text(json.dumps(index))

    pack = pack_rruff_database(str(src_cache), str(tmp_path / "share.sq"))
    dest_cache = tmp_path / "cache_b"
    n = unpack_rruff_database(pack, str(dest_cache))
    assert n == 1
    restored = json.loads((dest_cache / "index.json").read_text())
    assert restored[0]["mineral"] == "Quartz"
    assert str(dest_cache) in restored[0]["raw_path"]  # rewritten to the new machine
    assert (dest_cache / "raw" / "Quartz__R1__Raman__532____Processed__x.txt").exists()
