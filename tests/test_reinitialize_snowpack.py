"""
tests/test_reinitialize_snowpack.py — Tests for read_sno / scour_sno / write_sno.

Run with:  pytest tests/test_reinitialize_snowpack.py -v

Key risk: scour_sno does partial-layer trimming and recomputes header fields
(nSnowLayerData, HS_Last, ErosionLevel, ProfileDate).  A corrupt .sno
propagates silently through the rest of the season.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "avachain"))

from reinitialize_snowpack import read_sno, scour_sno, write_sno, sanitize_sno_dir


# ---------------------------------------------------------------------------
# Synthetic .sno factory
# ---------------------------------------------------------------------------

def _make_sno(tmp_path: Path,
              layers: list[dict],
              name: str = "cluster_0001_cluster_0001.sno") -> Path:
    """
    Write a minimal SNOWPACK .sno file.

    Each layer dict must contain 'Layer_Thick' and 'Density' at minimum.
    Layers are written bottom-to-top (order preserved).
    """
    n = len(layers)
    hs = sum(float(l["Layer_Thick"]) for l in layers)
    p = tmp_path / name

    header = textwrap.dedent(f"""\
        SMET 1.1 ASCII
        [HEADER]
        ProfileDate    = 2020-01-15T12:00
        nSnowLayerData = {n}
        HS_Last        = {hs:.6f}
        ErosionLevel   = {max(0, n - 1)}
        fields         = Layer_Thick Density
        [DATA]
    """)
    data_lines = ""
    for l in layers:
        thick = l["Layer_Thick"]
        dens = l.get("Density", "200.0")
        data_lines += f"     {thick}     {dens}\n"

    p.write_text(header + data_lines)
    return p


def _parse_header_value(sno_data: dict, key: str):
    """Return the string value for a header key."""
    return sno_data["header"].get(key)


# ---------------------------------------------------------------------------
# read_sno / write_sno round-trip
# ---------------------------------------------------------------------------

class TestReadWriteRoundTrip:
    def test_layer_count_preserved(self, tmp_path):
        layers = [
            {"Layer_Thick": "0.100000", "Density": "200.0"},
            {"Layer_Thick": "0.200000", "Density": "250.0"},
            {"Layer_Thick": "0.300000", "Density": "300.0"},
        ]
        p = _make_sno(tmp_path, layers)
        data = read_sno(str(p))
        assert len(data["layers"]) == 3

    def test_layer_thickness_values_read_correctly(self, tmp_path):
        layers = [
            {"Layer_Thick": "0.150000"},
            {"Layer_Thick": "0.250000"},
        ]
        p = _make_sno(tmp_path, layers)
        data = read_sno(str(p))
        thicknesses = [float(l["Layer_Thick"]) for l in data["layers"]]
        assert thicknesses == pytest.approx([0.15, 0.25])

    def test_fields_parsed(self, tmp_path):
        p = _make_sno(tmp_path, [{"Layer_Thick": "0.1"}])
        data = read_sno(str(p))
        assert "Layer_Thick" in data["fields"]

    def test_header_dict_populated(self, tmp_path):
        p = _make_sno(tmp_path, [{"Layer_Thick": "0.1"}])
        data = read_sno(str(p))
        assert data["header"]["nSnowLayerData"] == "1"

    def test_write_then_read_identical(self, tmp_path):
        layers = [
            {"Layer_Thick": "0.100000", "Density": "200.0"},
            {"Layer_Thick": "0.300000", "Density": "300.0"},
        ]
        p = _make_sno(tmp_path, layers)
        original = read_sno(str(p))

        out_path = tmp_path / "written.sno"
        write_sno(original, str(out_path))
        reread = read_sno(str(out_path))

        assert len(reread["layers"]) == len(original["layers"])
        for i in range(len(original["layers"])):
            assert (float(reread["layers"][i]["Layer_Thick"]) ==
                    pytest.approx(float(original["layers"][i]["Layer_Thick"])))


# ---------------------------------------------------------------------------
# scour_sno — whole layer removal
# ---------------------------------------------------------------------------

class TestScourSnoWholeLayer:
    def test_removes_top_layer_exactly(self, tmp_path):
        layers = [
            {"Layer_Thick": "0.300000"},
            {"Layer_Thick": "0.200000"},
            {"Layer_Thick": "0.100000"},   # top layer
        ]
        p = _make_sno(tmp_path, layers)
        data = read_sno(str(p))
        result = scour_sno(data, scour_depth_m=0.1, event_timestamp="2020-01-18T12:00")
        assert len(result["layers"]) == 2
        remaining = [float(l["Layer_Thick"]) for l in result["layers"]]
        assert remaining == pytest.approx([0.3, 0.2])

    def test_removes_multiple_whole_layers(self, tmp_path):
        layers = [
            {"Layer_Thick": "0.400000"},
            {"Layer_Thick": "0.100000"},
            {"Layer_Thick": "0.200000"},
            {"Layer_Thick": "0.300000"},   # top
        ]
        p = _make_sno(tmp_path, layers)
        data = read_sno(str(p))
        # Scour 0.5m: removes 0.3 + 0.2 = 0.5
        result = scour_sno(data, scour_depth_m=0.5, event_timestamp="2020-01-18T12:00")
        assert len(result["layers"]) == 2
        remaining = [float(l["Layer_Thick"]) for l in result["layers"]]
        assert remaining == pytest.approx([0.4, 0.1])

    def test_hs_last_updated_in_header(self, tmp_path):
        layers = [
            {"Layer_Thick": "0.300000"},
            {"Layer_Thick": "0.200000"},
        ]
        p = _make_sno(tmp_path, layers)
        data = read_sno(str(p))
        result = scour_sno(data, scour_depth_m=0.2, event_timestamp="2020-01-18T12:00")
        new_hs = float(_parse_header_value(result, "HS_Last"))
        assert new_hs == pytest.approx(0.3, abs=1e-5)

    def test_nsnowlayerdata_updated_in_header(self, tmp_path):
        layers = [
            {"Layer_Thick": "0.300000"},
            {"Layer_Thick": "0.200000"},
            {"Layer_Thick": "0.100000"},
        ]
        p = _make_sno(tmp_path, layers)
        data = read_sno(str(p))
        result = scour_sno(data, scour_depth_m=0.1, event_timestamp="2020-01-18T12:00")
        assert _parse_header_value(result, "nSnowLayerData") == "2"

    def test_profile_date_updated(self, tmp_path):
        p = _make_sno(tmp_path, [{"Layer_Thick": "0.300000"},
                                  {"Layer_Thick": "0.200000"}])
        data = read_sno(str(p))
        result = scour_sno(data, scour_depth_m=0.1, event_timestamp="2026-01-18T14:00")
        assert _parse_header_value(result, "ProfileDate") == "2026-01-18T14:00"

    def test_erosion_level_updated(self, tmp_path):
        layers = [{"Layer_Thick": "0.200000"},
                  {"Layer_Thick": "0.200000"},
                  {"Layer_Thick": "0.200000"}]
        p = _make_sno(tmp_path, layers)
        data = read_sno(str(p))
        result = scour_sno(data, scour_depth_m=0.2, event_timestamp="2020-01-18T12:00")
        # 2 layers remaining → ErosionLevel = max(0, 2-1) = 1
        assert _parse_header_value(result, "ErosionLevel") == "1"

    def test_scour_stats_present(self, tmp_path):
        p = _make_sno(tmp_path, [{"Layer_Thick": "0.300000"},
                                  {"Layer_Thick": "0.200000"}])
        data = read_sno(str(p))
        result = scour_sno(data, scour_depth_m=0.2, event_timestamp="2020-01-18T12:00")
        assert "scour_stats" in result
        assert result["scour_stats"]["n_layers_removed"] == 1

    def test_original_not_mutated(self, tmp_path):
        layers = [{"Layer_Thick": "0.300000"}, {"Layer_Thick": "0.200000"}]
        p = _make_sno(tmp_path, layers)
        data = read_sno(str(p))
        original_count = len(data["layers"])
        scour_sno(data, scour_depth_m=0.2, event_timestamp="2020-01-18T12:00")
        assert len(data["layers"]) == original_count  # deep copy; original untouched


# ---------------------------------------------------------------------------
# scour_sno — partial layer split
# ---------------------------------------------------------------------------

class TestScourSnoPartialLayer:
    def test_partial_layer_thickness_reduced(self, tmp_path):
        layers = [
            {"Layer_Thick": "0.300000"},
            {"Layer_Thick": "0.200000"},   # top; scour 0.15 into this
        ]
        p = _make_sno(tmp_path, layers)
        data = read_sno(str(p))
        result = scour_sno(data, scour_depth_m=0.15, event_timestamp="2020-01-18T12:00")
        assert len(result["layers"]) == 2
        top = float(result["layers"][-1]["Layer_Thick"])
        assert top == pytest.approx(0.05, abs=1e-6)

    def test_partial_scour_hs_correct(self, tmp_path):
        layers = [
            {"Layer_Thick": "0.300000"},
            {"Layer_Thick": "0.200000"},
        ]
        p = _make_sno(tmp_path, layers)
        data = read_sno(str(p))
        result = scour_sno(data, scour_depth_m=0.15, event_timestamp="2020-01-18T12:00")
        new_hs = float(_parse_header_value(result, "HS_Last"))
        # 0.3 + (0.2 - 0.15) = 0.35
        assert new_hs == pytest.approx(0.35, abs=1e-5)


# ---------------------------------------------------------------------------
# scour_sno — edge cases
# ---------------------------------------------------------------------------

class TestScourSnoEdgeCases:
    def test_scour_deeper_than_all_layers_removes_all(self, tmp_path):
        layers = [
            {"Layer_Thick": "0.100000"},
            {"Layer_Thick": "0.200000"},
        ]
        p = _make_sno(tmp_path, layers)
        data = read_sno(str(p))
        result = scour_sno(data, scour_depth_m=10.0, event_timestamp="2020-01-18T12:00")
        assert len(result["layers"]) == 0
        assert float(_parse_header_value(result, "HS_Last")) == pytest.approx(0.0)
        assert _parse_header_value(result, "nSnowLayerData") == "0"

    def test_empty_sno_scour_noop(self, tmp_path):
        p = _make_sno(tmp_path, [])
        data = read_sno(str(p))
        result = scour_sno(data, scour_depth_m=0.5, event_timestamp="2020-01-18T12:00")
        assert len(result["layers"]) == 0

    def test_zero_scour_depth_removes_nothing(self, tmp_path):
        layers = [{"Layer_Thick": "0.300000"}, {"Layer_Thick": "0.200000"}]
        p = _make_sno(tmp_path, layers)
        data = read_sno(str(p))
        result = scour_sno(data, scour_depth_m=0.0, event_timestamp="2020-01-18T12:00")
        assert len(result["layers"]) == 2


# ---------------------------------------------------------------------------
# sanitize_sno_dir — zero-thickness layer removal
# ---------------------------------------------------------------------------

class TestSanitizeSnoDir:
    def test_removes_zero_thickness_layers(self, tmp_path):
        layers = [
            {"Layer_Thick": "0.000000"},   # zero — should be removed
            {"Layer_Thick": "0.200000"},
            {"Layer_Thick": "0.300000"},
        ]
        _make_sno(tmp_path, layers)
        n_fixed = sanitize_sno_dir(tmp_path)
        assert n_fixed == 1
        data = read_sno(str(tmp_path / "cluster_0001_cluster_0001.sno"))
        assert len(data["layers"]) == 2

    def test_clean_file_not_modified(self, tmp_path):
        layers = [
            {"Layer_Thick": "0.100000"},
            {"Layer_Thick": "0.200000"},
        ]
        p = _make_sno(tmp_path, layers)
        original_mtime = p.stat().st_mtime
        sanitize_sno_dir(tmp_path)
        assert p.stat().st_mtime == original_mtime  # file untouched

    def test_returns_count_of_files_modified(self, tmp_path):
        # Two files: one dirty, one clean
        _make_sno(tmp_path, [{"Layer_Thick": "0.000000"}, {"Layer_Thick": "0.2"}],
                  name="cluster_0001_cluster_0001.sno")
        _make_sno(tmp_path, [{"Layer_Thick": "0.1"}, {"Layer_Thick": "0.2"}],
                  name="cluster_0002_cluster_0002.sno")
        n = sanitize_sno_dir(tmp_path)
        assert n == 1

    def test_dry_run_does_not_write(self, tmp_path):
        layers = [{"Layer_Thick": "0.000000"}, {"Layer_Thick": "0.2"}]
        p = _make_sno(tmp_path, layers)
        original_text = p.read_text()
        sanitize_sno_dir(tmp_path, dry_run=True)
        assert p.read_text() == original_text

    def test_hs_last_updated_after_sanitize(self, tmp_path):
        layers = [
            {"Layer_Thick": "0.000000"},
            {"Layer_Thick": "0.200000"},
            {"Layer_Thick": "0.300000"},
        ]
        _make_sno(tmp_path, layers)
        sanitize_sno_dir(tmp_path)
        data = read_sno(str(tmp_path / "cluster_0001_cluster_0001.sno"))
        assert float(data["header"]["HS_Last"]) == pytest.approx(0.5, abs=1e-5)
