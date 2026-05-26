import numpy as np
import pytest

from emsaplibraries.indicators import (
    CustomAtom,
    ProteinMetricsResult,
    acid_base_stability_estimator,
    calculate_pka_by_sasa,
    calculate_protein_metrics,
    calculate_protein_pka_sasa,
    calculate_q_sasa,
    interpolate_potential,
    parse_dx,
    parse_pqr,
    parse_propka_pka,
)


def test_parse_pqr_reads_coords_charges_and_radii(tmp_path):
    pqr = tmp_path / "sample.pqr"
    pqr.write_text(
        "ATOM      1  N   GLY A   1       1.000   2.000   3.000 -0.3000 1.5000\n"
        "HETATM    2  O   HOH A   2       4.000   5.000   6.000 -0.8000 1.4000\n",
        encoding="utf-8",
    )

    coords, charges, radii = parse_pqr(pqr)

    np.testing.assert_allclose(coords, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_allclose(charges, [-0.3, -0.8])
    np.testing.assert_allclose(radii, [1.5, 1.4])


def test_parse_dx_reads_grid_metadata_and_values(tmp_path):
    dx = tmp_path / "sample.dx"
    dx.write_text(
        "object 1 class gridpositions counts 2 2 2\n"
        "origin 0.0 0.0 0.0\n"
        "delta 1.0 0.0 0.0\n"
        "delta 0.0 1.0 0.0\n"
        "delta 0.0 0.0 1.0\n"
        "object 3 class array type double rank 0 items 8 data follows\n"
        "0 1 2\n"
        "3 4 5\n"
        "6 7\n",
        encoding="utf-8",
    )

    grid, origin, spacing = parse_dx(dx)

    assert grid.shape == (2, 2, 2)
    np.testing.assert_allclose(origin, [0.0, 0.0, 0.0])
    assert spacing == (1.0, 1.0, 1.0)
    np.testing.assert_allclose(grid.ravel(), np.arange(8))


def test_interpolate_potential_on_unit_grid():
    grid = np.arange(8, dtype=float).reshape((2, 2, 2))
    atom = CustomAtom("X", "RES", "A", "1", np.array([0.5, 0.5, 0.5]), 0.0)

    assert (
        interpolate_potential(atom, grid, np.array([0.0, 0.0, 0.0]), (1.0, 1.0, 1.0))
        == 3.5
    )


def test_calculate_q_sasa_smoke(tmp_path):
    pytest.importorskip("freesasa")
    pqr = tmp_path / "sample.pqr"
    pqr.write_text(
        "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.0000 1.5000\n"
        "ATOM      2  C   GLY A   1       4.000   0.000   0.000 -1.0000 1.5000\n",
        encoding="utf-8",
    )

    q_sasa, numerator, total_sasa = calculate_q_sasa(pqr)

    assert isinstance(q_sasa, float)
    assert isinstance(numerator, float)
    assert total_sasa > 0


def test_parse_propka_pka_reads_valid_rows_only(tmp_path):
    pka = tmp_path / "protein.pka"
    pka.write_text(
        "SUMMARY OF THIS PREDICTION\n"
        "ASP 25 A 3.90 0.00 0.00\n"
        "GLU 101 B not-a-number\n"
        "header row that should be ignored\n"
        "LYS 5 A 10.50\n"
        "CYS 7 AB 8.30\n",
        encoding="utf-8",
    )

    assert parse_propka_pka(pka) == {
        "ASP25A": 3.9,
        "LYS5A": 10.5,
    }


def test_pka_metrics_require_pka_file_and_use_parsed_values(monkeypatch, tmp_path):
    pka = tmp_path / "protein.pka"
    pka.write_text("ASP 1 A 4.90\nLYS 2 A 10.50\n", encoding="utf-8")
    monkeypatch.setattr(
        "emsaplibraries.indicators.residue_sasa_from_pqr",
        lambda pdb, pqr: {"ASP1A": 55.0, "LYS2A": 200.0},
    )

    result = calculate_pka_by_sasa("protein.pdb", "protein.pqr", pka)

    assert result == {
        "ASP1A": (4.9, 0.5, 2.45),
        "LYS2A": (10.5, 1.0, 10.5),
    }
    assert calculate_protein_pka_sasa("protein.pdb", "protein.pqr", pka) == 6.475
    assert acid_base_stability_estimator("protein.pdb", "protein.pqr", pka) > 0.0


def test_indicators_module_does_not_import_subprocess():
    import emsaplibraries.indicators as indicators

    assert not hasattr(indicators, "subprocess")


def test_calculate_protein_metrics_without_pka_file(monkeypatch):
    monkeypatch.setattr(
        "emsaplibraries.indicators.calculate_p_sasa",
        lambda pqr, pdb, dx: (1.0, 10.0, 20.0),
    )
    monkeypatch.setattr(
        "emsaplibraries.indicators.calculate_q_sasa",
        lambda pqr: (2.0, 30.0, 40.0),
    )
    monkeypatch.setattr(
        "emsaplibraries.indicators.calculate_residue_exposed_charge",
        lambda pqr, pdb: {"percent_exposed_charge": 3.0, "per_residue": []},
    )
    monkeypatch.setattr("emsaplibraries.indicators.calculate_see", lambda pqr, dx: 4.0)
    monkeypatch.setattr(
        "emsaplibraries.indicators.calculate_hse",
        lambda pdb, pqr: (5.0, 0.0, 0.0),
    )

    result = calculate_protein_metrics("protein.pdb", "protein.pqr", "protein.dx")

    assert isinstance(result, ProteinMetricsResult)
    assert result.protein_name == "protein"
    assert result.p_sasa == 1.0
    assert result.q_sasa == 2.0
    assert result.ecpi_percent == 3.0
    assert result.see == 4.0
    assert result.hse == 5.0
    assert result.pka_sasa is None
    assert result.abse is None
    assert result.p_sasa_numerator == 10.0
    assert result.p_sasa_denominator == 20.0
    assert result.q_sasa_numerator == 30.0
    assert result.q_sasa_total_sasa == 40.0
    assert result.as_dict()["pka_file"] is None
    assert result.as_dict()["pdb_file"] == "protein.pdb"


def test_calculate_protein_metrics_with_pka_file(monkeypatch):
    pka_calls = []

    monkeypatch.setattr(
        "emsaplibraries.indicators.calculate_p_sasa",
        lambda pqr, pdb, dx: (1.0, 10.0, 20.0),
    )
    monkeypatch.setattr(
        "emsaplibraries.indicators.calculate_q_sasa",
        lambda pqr: (2.0, 30.0, 40.0),
    )
    monkeypatch.setattr(
        "emsaplibraries.indicators.calculate_residue_exposed_charge",
        lambda pqr, pdb: {"percent_exposed_charge": 3.0},
    )
    monkeypatch.setattr("emsaplibraries.indicators.calculate_see", lambda pqr, dx: 4.0)
    monkeypatch.setattr(
        "emsaplibraries.indicators.calculate_hse",
        lambda pdb, pqr: (5.0, 0.0, 0.0),
    )

    def fake_pka_sasa(pdb, pqr, pka):
        pka_calls.append((pdb, pqr, pka))
        return 6.0

    monkeypatch.setattr(
        "emsaplibraries.indicators.calculate_protein_pka_sasa",
        fake_pka_sasa,
    )
    monkeypatch.setattr(
        "emsaplibraries.indicators.acid_base_stability_estimator",
        lambda pdb, pqr, pka: 7.0,
    )

    result = calculate_protein_metrics(
        "protein.pdb", "protein.pqr", "protein.dx", "protein.pka"
    )

    assert result.pka_sasa == 6.0
    assert result.abse == 7.0
    assert result.as_dict()["pka_file"] == "protein.pka"
    assert pka_calls == [(result.pdb_file, result.pqr_file, result.pka_file)]
