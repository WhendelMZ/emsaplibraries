import numpy as np
import pytest

from emsaplibraries.indicators import (
    CustomAtom,
    calculate_q_sasa,
    interpolate_potential,
    parse_dx,
    parse_pqr,
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

    assert interpolate_potential(atom, grid, np.array([0.0, 0.0, 0.0]), (1.0, 1.0, 1.0)) == 3.5


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
