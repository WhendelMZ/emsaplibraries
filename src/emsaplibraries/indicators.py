"""Protein electrostatic and surface indicator calculations."""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from Bio.PDB.PDBParser import PDBParser

from ._runtime import require_executable, require_module


@dataclass
class CustomAtom:
    """Simple atom container used for electrostatic calculations."""

    name: str
    residue_name: str
    chain_id: str
    residue_number: str
    coord: np.ndarray
    charge: float
    potential: float = 0.0


def _freesasa():
    return require_module("freesasa", "pip install freesasa")


def _pdb_parser(quiet: bool = True):
    bio_pdb = require_module("Bio.PDB", "pip install biopython")
    return bio_pdb.PDBParser(QUIET=quiet)


def extract_epi(pdb_filename: str) -> str:
    """Extract an EPI_ISL identifier from a filename, or return ``N/A``."""
    match = re.search(r"EPI_ISL_\d+", str(pdb_filename))
    return match.group(0) if match else "N/A"


def parse_dx(
    dx_file: str | Path,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Parse APBS/OpenDX potential grid data from a DX file."""
    origin = None
    grid_size = None
    spacing = [1.0, 1.0, 1.0]
    potentials: list[float] = []

    with open(dx_file, encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if not parts or parts[0].startswith("#"):
                continue

            if parts[0] == "origin":
                origin = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "object" and len(parts) > 2 and parts[1] == "1":
                grid_size = np.array([int(parts[-3]), int(parts[-2]), int(parts[-1])])
            elif parts[0] == "delta":
                delta_vals = [float(x) for x in parts[1:4]]
                for index, value in enumerate(delta_vals):
                    if value != 0.0:
                        spacing[index] = abs(value)
            else:
                try:
                    potentials.extend(float(value) for value in parts)
                except ValueError:
                    continue

    if grid_size is None or origin is None:
        raise ValueError("Failed to read DX grid parameters.")

    expected = int(np.prod(grid_size))
    if len(potentials) != expected:
        raise ValueError(
            f"DX potential count ({len(potentials)}) does not match grid size "
            f"({expected})."
        )

    potential_grid = np.array(potentials).reshape(grid_size)
    return potential_grid, origin, tuple(spacing)


def interpolate_potential(
    atom: CustomAtom,
    potential_grid: np.ndarray,
    origin: np.ndarray,
    spacing: tuple[float, float, float],
) -> float:
    """Perform trilinear interpolation of electrostatic potential at an atom."""
    spacing_array = np.array(spacing, dtype=float)
    x, y, z = (np.array(atom.coord, dtype=float) - origin) / spacing_array

    x0, y0, z0 = np.floor([x, y, z]).astype(int)
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1

    max_x, max_y, max_z = np.array(potential_grid.shape) - 1
    x0, x1 = np.clip([x0, x1], 0, max_x)
    y0, y1 = np.clip([y0, y1], 0, max_y)
    z0, z1 = np.clip([z0, z1], 0, max_z)

    xd, yd, zd = x - x0, y - y0, z - z0

    c000 = potential_grid[x0, y0, z0]
    c100 = potential_grid[x1, y0, z0]
    c010 = potential_grid[x0, y1, z0]
    c110 = potential_grid[x1, y1, z0]
    c001 = potential_grid[x0, y0, z1]
    c101 = potential_grid[x1, y0, z1]
    c011 = potential_grid[x0, y1, z1]
    c111 = potential_grid[x1, y1, z1]

    c00 = c000 * (1 - xd) + c100 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c11 = c011 * (1 - xd) + c111 * xd

    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd
    return float(c0 * (1 - zd) + c1 * zd)


def parse_pqr(pqr_file: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse atomic coordinates, charges, and radii from a PQR file."""
    coords: list[list[float]] = []
    charges: list[float] = []
    radii: list[float] = []

    with open(pqr_file, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")):
                parts = line.split()
                if len(parts) < 7:
                    raise ValueError(f"Malformed PQR atom line: {line.rstrip()}")

                charge = float(parts[-2])
                radius = float(parts[-1])
                x, y, z = map(float, parts[-5:-2])

                coords.append([x, y, z])
                charges.append(charge)
                radii.append(radius)

    return np.array(coords), np.array(charges), np.array(radii)


def _pdb_atoms(pdb_file: str | Path) -> list[CustomAtom]:
    structure = _pdb_parser().get_structure("protein", str(pdb_file))
    atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atoms.append(
                        CustomAtom(
                            atom.get_name(),
                            residue.resname,
                            chain.id,
                            str(residue.id[1]),
                            atom.get_coord(),
                            charge=0.0,
                        )
                    )
    return atoms


def _sasa_from_coords(
    coords: np.ndarray,
    radii: np.ndarray,
    atoms: list[CustomAtom] | None = None,
) -> tuple[np.ndarray, float]:
    freesasa = _freesasa()
    structure = freesasa.Structure()

    if atoms is None:
        for index, (x, y, z) in enumerate(coords):
            structure.addAtom(
                "X", "RES", str(index + 1), "A", float(x), float(y), float(z)
            )
    else:
        for atom in atoms:
            structure.addAtom(
                atom.name,
                atom.residue_name,
                atom.chain_id,
                atom.residue_number,
                float(atom.coord[0]),
                float(atom.coord[1]),
                float(atom.coord[2]),
            )

    structure.setRadii(radii)
    params = freesasa.Parameters({"algorithm": freesasa.ShrakeRupley})
    result = freesasa.calc(structure, params)
    sasa_atoms = np.array([result.atomArea(i) for i in range(structure.nAtoms())])
    return sasa_atoms, float(sasa_atoms.sum())


def calculate_p_sasa(pqr_file: str | Path, pdb_file: str | Path, dx_file: str | Path):
    """Calculate P_SASA as ``sum(phi_i * SASA_i) / total_SASA``."""
    _, _, radii_pqr = parse_pqr(pqr_file)
    atom_list = _pdb_atoms(pdb_file)

    if len(atom_list) != len(radii_pqr):
        raise ValueError("Mismatch between PDB and PQR atom counts.")

    potential_grid, origin, spacing = parse_dx(dx_file)
    potentials = [
        interpolate_potential(atom, potential_grid, origin, spacing)
        for atom in atom_list
    ]
    sasa, total_sasa = _sasa_from_coords(
        np.array([atom.coord for atom in atom_list]), radii_pqr, atom_list
    )

    numerator = float(np.sum(np.array(potentials) * sasa))
    denominator = float(total_sasa)
    p_sasa = numerator / denominator if denominator > 0 else 0.0
    return p_sasa, numerator, denominator


def calculate_sasa_from_pqr(pqr_file: str | Path):
    """Compute atomic and total SASA from PQR coordinates and radii."""
    coords, charges, radii = parse_pqr(pqr_file)
    sasa_atoms, total_sasa = _sasa_from_coords(coords, radii)
    return sasa_atoms, total_sasa, charges


def calculate_q_sasa(pqr_file: str | Path):
    """Compute Q_SASA as ``sum(q_i * SASA_i) / total_SASA``."""
    sasa_atoms, total_sasa, charges = calculate_sasa_from_pqr(pqr_file)
    numerator = float(np.sum(charges * sasa_atoms))
    q_sasa = numerator / total_sasa if total_sasa > 0 else 0.0
    return q_sasa, numerator, total_sasa


def atoms_outside_grid_coords(coords, origin, spacing, grid_shape) -> list[int]:
    """Identify atom indices outside DX grid boundaries."""
    mins = np.array(origin)
    maxs = mins + np.array(grid_shape) * np.array(spacing)
    return [
        index
        for index, coord in enumerate(coords)
        if np.any(np.array(coord) < mins) or np.any(np.array(coord) > maxs)
    ]


def calculate_see(
    pqr_file: str | Path,
    dx_file: str | Path,
    use_pdb_for_coords: bool = False,
    pdb_file: str | Path | None = None,
    debug: bool = False,
) -> float:
    """
    Compute Surface Electrostatic Exposure.
    ``sum(q_i * phi_i * SASA_i) / sum(SASA_i)``
    """
    coords_pqr, charges_pqr, radii_pqr = parse_pqr(pqr_file)
    atom_count = len(coords_pqr)
    if atom_count == 0:
        raise ValueError("parse_pqr returned zero atoms.")

    potential_grid, origin, spacing = parse_dx(dx_file)

    if use_pdb_for_coords:
        if pdb_file is None:
            raise ValueError("pdb_file must be provided when use_pdb_for_coords=True.")
        atoms_pdb = _pdb_atoms(pdb_file)
        coords_for_sasa = np.array([atom.coord for atom in atoms_pdb])
        if len(coords_for_sasa) != atom_count:
            raise ValueError(
                f"PDB atom count ({len(coords_for_sasa)}) != PQR atom count "
                f"({atom_count})."
            )
    else:
        coords_for_sasa = np.array(coords_pqr)

    sasa, _ = _sasa_from_coords(coords_for_sasa, radii_pqr)

    if debug:
        zero_frac = np.mean(sasa == 0.0)
        warnings.warn(
            f"Fraction of zero SASA atoms: {zero_frac:.3f}",
            RuntimeWarning,
            stacklevel=2,
        )

    outside_atoms = atoms_outside_grid_coords(
        coords_for_sasa, origin, spacing, potential_grid.shape
    )
    if outside_atoms:
        warnings.warn(
            f"{len(outside_atoms)} atoms are outside the DX grid.",
            RuntimeWarning,
            stacklevel=2,
        )

    atoms = [
        CustomAtom(
            "X",
            "RES",
            "A",
            str(index + 1),
            np.array(coords_pqr[index]),
            float(charges_pqr[index]),
        )
        for index in range(atom_count)
    ]
    potentials = np.array(
        [interpolate_potential(atom, potential_grid, origin, spacing) for atom in atoms]
    )

    n = min(len(sasa), len(potentials), len(charges_pqr))
    numerator = float(np.sum(np.array(charges_pqr[:n]) * potentials[:n] * sasa[:n]))
    denominator = float(np.sum(sasa[:n]))
    return numerator / denominator if denominator > 0 else 0.0


def calculate_surface_potential_fraction(
    pqr_file: str | Path,
    dx_file: str | Path,
    threshold: float = 1.0,
) -> float:
    """Return percentage of surface atoms above an electrostatic threshold."""
    coords, charges, _ = parse_pqr(pqr_file)
    sasa_atoms, _, _ = calculate_sasa_from_pqr(pqr_file)
    potential_grid, origin, spacing = parse_dx(dx_file)

    atoms = [
        CustomAtom("X", "RES", "A", str(index + 1), coords[index], charges[index])
        for index in range(len(coords))
    ]
    potentials = np.array(
        [interpolate_potential(atom, potential_grid, origin, spacing) for atom in atoms]
    )

    surface_mask = sasa_atoms > 0
    if surface_mask.sum() == 0:
        return 0.0

    surface_potentials = potentials[surface_mask]
    above_threshold = np.sum(surface_potentials > threshold)
    return float((above_threshold / len(surface_potentials)) * 100)


def calculate_residue_exposed_charge(
    pqr_file: str | Path, pdb_file: str | Path
) -> dict:
    """Compute residue-level exposed charge using ECPi logic."""

    _, charges, radii = parse_pqr(pqr_file)

    sasa_atoms, _, charges_chk = calculate_sasa_from_pqr(pqr_file)

    if charges_chk is not None and len(charges_chk) == len(charges):
        charges = charges_chk

    structure = _pdb_parser().get_structure("protein", str(pdb_file))

    pdb_atoms = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":
                    continue

                for atom in residue:
                    pdb_atoms.append(
                        (
                            residue.resname,
                            residue.id[1],
                            chain.id,
                            atom.name,
                        )
                    )

    if len(pdb_atoms) != len(charges):
        raise ValueError(
            f"PDB atoms ({len(pdb_atoms)}) != PQR atoms ({len(charges)}). "
            "Files are inconsistent."
        )

    residues = defaultdict(
        lambda: {
            "atom_indices": [],
            "abs_charge": 0.0,
            "sasa": 0.0,
            "max_sasa": 0.0,
            "exposed_charge": 0.0,
        }
    )

    total_abs_charge = 0.0
    total_exposed_charge = 0.0

    for index, (resname, resnum, chain, _atom_name) in enumerate(pdb_atoms):
        key = f"{resname}{resnum}{chain}"

        qi = abs(float(charges[index]))

        sasa_i = float(sasa_atoms[index])

        radius_i = float(radii[index])

        sasa_max_i = 4.0 * math.pi * (radius_i**2)

        frac_exp = sasa_i / sasa_max_i if sasa_max_i > 0 else 0.0

        frac_exp = min(1.0, frac_exp)

        exposed_q = qi * frac_exp

        residues[key]["atom_indices"].append(index)
        residues[key]["abs_charge"] += qi
        residues[key]["sasa"] += sasa_i
        residues[key]["max_sasa"] += sasa_max_i
        residues[key]["exposed_charge"] += exposed_q

        total_abs_charge += qi
        total_exposed_charge += exposed_q

    per_residue = []

    for key, info in residues.items():
        exposure_fraction = (
            info["exposed_charge"] / info["abs_charge"]
            if info["abs_charge"] > 1e-12
            else 0.0
        )

        per_residue.append(
            {
                "residue": key,
                "abs_charge": info["abs_charge"],
                "sasa": info["sasa"],
                "max_sasa": info["max_sasa"],
                "exposure_fraction": exposure_fraction,
                "exposed_charge": info["exposed_charge"],
            }
        )

    percent_exposed_charge = (
        (total_exposed_charge / total_abs_charge) * 100.0
        if total_abs_charge > 1e-12
        else 0.0
    )

    percent_exposed_charge = min(percent_exposed_charge, 100.0)

    return {
        "total_abs_charge": total_abs_charge,
        "total_exposed_charge": total_exposed_charge,
        "percent_exposed_charge": percent_exposed_charge,
        "per_residue": per_residue,
    }


def calculate_hse(
    pdb_file: str | Path, pqr_file: str | Path
) -> tuple[float, float, float]:
    """
    Computes HSE (Hydrophobic Surface Exposure).

    Keeps compatibility with the previous function signature.

    Returns:
    --------
    hse_value : float
        Final normalized HSE value.

    hydrophobic_exposure : float
        Hydrophobic contribution term:
            Σ(KD * SASA)

    electrostatic_exposure : float
        Placeholder value (not used in this metric).
    """

    kd_scale = {
        "ALA": 1.8,
        "ARG": -4.5,
        "ASN": -3.5,
        "ASP": -3.5,
        "CYS": 2.5,
        "GLN": -3.5,
        "GLU": -3.5,
        "GLY": -0.4,
        "HIS": -3.2,
        "ILE": 4.5,
        "LEU": 3.8,
        "LYS": -3.9,
        "MET": 1.9,
        "PHE": 2.8,
        "PRO": -1.6,
        "SER": -0.8,
        "THR": -0.7,
        "TRP": -0.9,
        "TYR": -1.3,
        "VAL": 4.2,
    }

    # Load structure
    structure = PDBParser(QUIET=True).get_structure("protein", pdb_file)

    # Compute atomic SASA
    sasa_atoms, total_sasa, charges = calculate_sasa_from_pqr(pqr_file)

    residue_sasa = {}

    atom_index = 0

    for model in structure:
        for chain in model:
            for residue in chain:
                residue_id = (chain.id, residue.id[1], residue.resname)

                residue_sasa.setdefault(residue_id, 0.0)

                for _atom in residue:
                    residue_sasa[residue_id] += sasa_atoms[atom_index]

                    atom_index += 1

    hydrophobic_exposure = 0.0

    total_surface_area = 0.0

    for (_chain_id, _residue_number, residue_name), sasa in residue_sasa.items():
        kd_value = kd_scale.get(residue_name, 0.0)

        hydrophobic_exposure += kd_value * sasa

        total_surface_area += sasa

    hse_value = (
        hydrophobic_exposure / total_surface_area if total_surface_area > 0 else 0.0
    )

    # Not used in this metric
    electrostatic_exposure = 0.0

    return (hse_value, hydrophobic_exposure, electrostatic_exposure)


pka_ref = {
    "ASP": 3.9,
    "GLU": 4.3,
    "HIS": 6.0,
    "CYS": 8.5,
    "TYR": 10.1,
    "LYS": 10.5,
    "ARG": 12.5,
}


def residue_sasa_from_pqr(pdb_file: str | Path, pqr_file: str | Path) -> dict:
    # 1. SASA per atom (correct PQR radii)
    sasa_atoms, _, _ = calculate_sasa_from_pqr(pqr_file)

    # 2. Read structure from PDB to map atoms → residues
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_file)

    residue_sasa = {}
    atom_index = 0

    for model in structure:
        for chain in model:
            for residue in chain:
                resname = residue.resname
                resnum = residue.id[1]
                chain_id = chain.id

                key = f"{resname}{resnum}{chain_id}"

                area = 0.0
                for _atom in residue:
                    area += sasa_atoms[atom_index]
                    atom_index += 1

                residue_sasa[key] = area

    return residue_sasa


def run_propka(pdb_file):
    pdb_base = os.path.basename(pdb_file)
    out = os.path.splitext(pdb_base)[0] + ".pka"
    out_file = os.path.join("/content", out)

    subprocess.run(["propka3", pdb_file], check=True)

    results = {}
    with open(out_file) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue

            resname = parts[0]
            resnum = parts[1]
            chain = parts[2]

            try:
                pka_val = float(parts[3])
            except (TypeError, ValueError):
                continue

            key = f"{resname}{resnum}{chain}"
            results[key] = pka_val

    return results


SASA_MAX = {
    "ASP": 110.0,
    "GLU": 140.0,
    "HIS": 160.0,
    "CYS": 135.0,
    "TYR": 230.0,
    "LYS": 200.0,
    "ARG": 225.0,
}


def calculate_pka_by_sasa(pdb_file: str | Path, pqr_file: str | Path) -> dict:
    """Calculate pKa values weighted by solvent exposure."""

    pkaI = run_propka(pdb_file)
    res_sasa = residue_sasa_from_pqr(pdb_file, pqr_file)

    results = {}

    for key, pka_val in pkaI.items():
        m = re.match(r"([A-Z]{3})(\d+)([A-Z])", key)
        if not m:
            continue

        resname = m.group(1)

        if resname not in SASA_MAX:
            continue
        if key not in res_sasa:
            continue

        sasa_res = res_sasa[key]
        frac_exp = min(1.0, sasa_res / SASA_MAX[resname])

        pka_sasa = frac_exp * pka_val

        results[key] = (pka_val, frac_exp, pka_sasa)

    return results


def calculate_protein_pka_sasa(pdb_file, pqr_file):
    residues = calculate_pka_by_sasa(pdb_file, pqr_file)

    if not residues:
        return 0.0

    return sum(v[2] for v in residues.values()) / len(residues)


R = 8.134  # J/mol/K
T = 298.15  # K


def acid_base_stability_estimator(pdb_file: str | Path, pqr_file: str | Path) -> float:
    """
    Computes the electrostatic stability estimator (ΔG)
    using PROPKA-derived pKa values and SASA exposure.

    The electrostatic free energy contribution is calculated as:

        ΔG = 2.303 * R * T * (pKa_eff - pKa_ref)

    Only residues with sufficient solvent exposure
    (relative exposure ≥ 0.20) are considered.

    Returns
    -------
    float
        Mean electrostatic contribution in kJ/mol.
    """

    reference_pka = {
        "ASP": 3.9,
        "GLU": 4.3,
        "HIS": 6.0,
        "CYS": 8.3,
        "TYR": 10.1,
        "LYS": 10.5,
        "ARG": 12.5,
    }

    # Compute SASA-weighted pKa values
    pka_results = calculate_pka_by_sasa(pdb_file, pqr_file)

    if not pka_results:
        return 0.0

    delta_g_values = []

    for residue_key, (effective_pka, exposure_fraction, _) in pka_results.items():
        match = re.match(r"([A-Z]{3})\d+[A-Z]", residue_key)

        if not match:
            continue

        residue_name = match.group(1)

        if residue_name not in reference_pka:
            continue

        # ΔpKa
        delta_pka = effective_pka - reference_pka[residue_name]

        # Electrostatic free energy (J/mol)
        delta_g = 2.303 * R * T * delta_pka

        # Consider only solvent-exposed residues
        if exposure_fraction >= 0.20:
            delta_g_values.append(delta_g)

    if not delta_g_values:
        return 0.0

    # Convert to kJ/mol
    return (sum(delta_g_values) / len(delta_g_values)) / 1000.0


def process_single_protein(
    pdb_file: str | Path,
    aux_output_dir: str | Path,
    bbox_min,
    bbox_max,
):
    """Run the available PDB2PQR/APBS indicator pipeline for one protein.

    Returns the legacy tuple shape. Metrics not implemented in this package version
    are returned as ``N/A`` rather than relying on undefined helper functions.
    """
    from .electrostatics import find_dx_file, generate_apbs_in_fixed

    require_executable("pdb2pqr", "Install PDB2PQR and ensure 'pdb2pqr' is on PATH.")
    require_executable("apbs", "Install APBS and ensure 'apbs' is on PATH.")

    pdb_path = Path(pdb_file)
    pdb_name = pdb_path.stem
    pqr_file = Path(f"{pdb_name}.pqr")

    subprocess.run(
        ["pdb2pqr", "--ff=PARSE", "--with-ph=7", str(pdb_path), str(pqr_file)],
        check=True,
    )
    in_path = generate_apbs_in_fixed(
        pdb_path, pdb_name, bbox_min, bbox_max, resolution=0.75
    )
    log_path = Path(f"{pdb_name}.out").resolve()
    with open(log_path, "w", encoding="utf-8") as log_handle:
        subprocess.run(
            ["apbs", in_path], stdout=log_handle, stderr=subprocess.STDOUT, check=True
        )

    dx_path = find_dx_file(pdb_name)

    p_sasa, _, _ = calculate_p_sasa(pqr_file, pdb_path, dx_path)
    q_sasa, _, _ = calculate_q_sasa(pqr_file)
    ecpi_data = calculate_residue_exposed_charge(pqr_file, pdb_path)
    see_val = calculate_see(pqr_file, dx_path)
    pkaI_val = calculate_protein_pka_sasa(pdb_file, pqr_file)
    hse_val, _, _ = calculate_hse(pdb_file, pqr_file)
    abse_val = acid_base_stability_estimator(pdb_file, pqr_file)

    # --- Print summary ---
    print(f"✅ Protein: {pdb_name}")
    print(f" Solvent-Accessible Surface Potential - P_SASA (kBT/e): {p_sasa:.2f}")
    print(f" Solvent-Accessible Surface Charge - Q_SASA (e): {q_sasa:.2f}")
    print(f" Exposed Charge % Index : {ecpi_data['percent_exposed_charge']:.2f}")
    print(f" Surface Electrostatic Energy - SEE (kBT): {see_val:.2f}")
    print(f" Hydrophobic Surface Exposure - HSE (dimensionless): {hse_val:.2f}")
    print(
        f" pKa Index of Ionizable Residue Groups - pKaI (dimensionless): {pkaI_val:.2f}"
    )
    print(f" Acid-Base Stability Estimator - ABSE (kJ/mol): {abse_val:.2f}")

    aux_dir = Path(aux_output_dir)
    aux_dir.mkdir(parents=True, exist_ok=True)
    for candidate in [
        f"{pdb_name}.in",
        f"{pdb_name}.out",
        f"{pdb_name}.pka",
        f"{pdb_name}.pqr",
        str(dx_path),
        f"{pdb_name}-input.p",
    ]:
        candidate_path = Path(candidate)
        if candidate_path.exists():
            shutil.move(str(candidate_path), aux_dir / candidate_path.name)

    return (
        pdb_name,
        f"{p_sasa:.2f}",
        f"{q_sasa:.2f}",
        f"{ecpi_data['percent_exposed_charge']:.2f}",
        f"{see_val:.2f}",
        f"{hse_val:.2f}",
        f"{pkaI_val:.2f}",
        f"{abse_val:.2f}",
        # f"{surface_potential_percent:.2f}",
    )
