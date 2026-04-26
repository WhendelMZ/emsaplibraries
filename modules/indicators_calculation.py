import re
import numpy as np
import freesasa
import math
from Bio.PDB import PDBParser
from collections import defaultdict

class CustomAtom:
    """
    Simple atom container used for electrostatic calculations.
    """
    def __init__(self, name, residue_name, chain_id, residue_number, coord, charge):
        self.name = name
        self.residue_name = residue_name
        self.chain_id = chain_id
        self.residue_number = residue_number
        self.coord = coord
        self.charge = charge
        self.potential = 0.0  # Used in electrostatic calculations


def extract_epi(pdb_filename):
    """
    Extract the EPI_ISL identifier from a PDB filename.

    Parameters:
        pdb_filename (str): Name of the PDB file

    Returns:
        str: Extracted EPI_ISL identifier or "N/A" if not found
    """
    match = re.search(r"EPI_ISL_\d+", pdb_filename)
    return match.group(0) if match else "N/A"


def parse_dx(dx_file):
    """
    Parse a .dx file and extract electrostatic potential grid data.

    Extracts:
        - potential_grid: 3D numpy array of potentials
        - origin: grid origin (x, y, z)
        - spacing: grid spacing (dx, dy, dz)

    Parameters:
        dx_file (str): Path to DX file

    Returns:
        tuple:
            potential_grid (np.ndarray)
            origin (np.ndarray)
            spacing (tuple)
    """
    with open(dx_file, 'r') as f:
        lines = f.readlines()

    origin = None
    grid_size = None
    spacing = [1.0, 1.0, 1.0]  # default spacing
    potentials = []

    for line in lines:
        parts = line.split()

        if len(parts) == 0 or parts[0].startswith("#"):
            continue

        if parts[0] == "origin":
            origin = np.array([float(parts[1]), float(parts[2]), float(parts[3])])

        elif parts[0] == "object" and parts[1] == "1":
            grid_size = np.array([int(parts[-3]), int(parts[-2]), int(parts[-1])])

        elif parts[0] == "delta":
            delta_vals = [float(x) for x in parts[1:4]]
            for i in range(3):
                if delta_vals[i] != 0.0:
                    spacing[i] = abs(delta_vals[i])

        else:
            try:
                potentials.extend([float(x) for x in parts])
            except ValueError:
                continue

    if grid_size is None or origin is None:
        raise ValueError("Failed to read DX grid parameters")

    potential_grid = np.array(potentials).reshape(grid_size)

    return potential_grid, origin, tuple(spacing)


def interpolate_potential(atom, potential_grid, origin, spacing):
    """
    Perform trilinear interpolation of electrostatic potential at atom position.

    Parameters:
        atom (CustomAtom): Atom with coordinates
        potential_grid (np.ndarray): 3D potential grid
        origin (np.ndarray): Grid origin
        spacing (tuple): Grid spacing (dx, dy, dz)

    Returns:
        float: Interpolated potential value
    """
    spacing = np.array(spacing, dtype=float)

    idx_f = (atom.coord - origin) / spacing
    x, y, z = idx_f

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

    c = c0 * (1 - zd) + c1 * zd

    return float(c)


def parse_pqr(pqr_file):
    """
    Parse a PQR file.

    Extracts:
        - atomic coordinates
        - atomic charges
        - atomic radii

    Parameters:
        pqr_file (str): Path to PQR file

    Returns:
        tuple:
            coords (np.ndarray)
            charges (np.ndarray)
            radii (np.ndarray)
    """
    coords = []
    charges = []
    radii = []

    with open(pqr_file, 'r') as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                parts = line.split()

                charge = float(parts[-2])
                radius = float(parts[-1])
                x, y, z = map(float, parts[-5:-2])

                coords.append([x, y, z])
                charges.append(charge)
                radii.append(radius)

    return np.array(coords), np.array(charges), np.array(radii)


def calculate_p_sasa(pqr_file, pdb_file, dx_file):
    """
    Calculate P_SASA descriptor.

    P_SASA is defined as:
        sum(phi_i * SASA_i) / total_SASA

    where:
        phi_i = electrostatic potential at atom i
        SASA_i = solvent accessible surface area of atom i

    Parameters:
        pqr_file (str): Path to PQR file (radii + charges)
        pdb_file (str): Path to PDB file (structure)
        dx_file (str): Path to DX file (electrostatic potential)

    Returns:
        tuple:
            p_sasa (float)
            numerator (float)
            denominator (float)
    """

    #Load PQR data
    coords_pqr, charges_pqr, radii_pqr = parse_pqr(pqr_file)

    #Load PDB structure
    structure = PDBParser().get_structure("protein", pdb_file)
    atom_list = []

    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom_list.append(
                        CustomAtom(
                            atom.get_name(),
                            residue.resname,
                            chain.id,
                            str(residue.id[1]),
                            atom.get_coord(),
                            charge=0.0
                        )
                    )

    if len(atom_list) != len(radii_pqr):
        raise ValueError("Mismatch between PDB and PQR atom counts")

    #Load DX grid
    potential_grid, origin, spacing = parse_dx(dx_file)

    #Interpolate potentials
    potentials = [
        interpolate_potential(atom, potential_grid, origin, spacing)
        for atom in atom_list
    ]

    #Compute SASA
    fs = freesasa.Structure()

    for atom in atom_list:
        fs.addAtom(
            atom.name,
            atom.residue_name,
            atom.chain_id,
            atom.residue_number,
            atom.coord[0],
            atom.coord[1],
            atom.coord[2]
        )

    fs.setRadii(radii_pqr)
    params = freesasa.Parameters({'algorithm': freesasa.ShrakeRupley})
    result = freesasa.calc(fs, params)

    sasa = np.array([result.atomArea(i) for i in range(fs.nAtoms())])
    total_sasa = sasa.sum()

    #Compute P_SASA
    numerator = np.sum(np.array(potentials) * sasa)
    denominator = total_sasa

    p_sasa = numerator / denominator if denominator > 0 else 0.0

    return p_sasa, numerator, denominator

import numpy as np
import freesasa
from Bio.PDB import PDBParser


def calculate_sasa_from_pqr(pqr_file):
    """
    Compute solvent-accessible surface area (SASA) from a PQR file.

    Uses the Shrake–Rupley algorithm with atomic radii provided in the PQR file.

    Parameters:
        pqr_file (str): Path to the PQR file

    Returns:
        tuple:
            sasa_atoms (np.ndarray): SASA per atom
            total_sasa (float): Total SASA
            charges (np.ndarray): Atomic charges
    """
    coords, charges, radii = parse_pqr(pqr_file)

    structure = freesasa.Structure()

    # Add atoms (without radii initially)
    for i, coord in enumerate(coords):
        structure.addAtom(
            "X", "RES", str(i + 1), "A",
            coord[0], coord[1], coord[2]
        )

    # Apply radii from PQR
    structure.setRadii(radii)

    # Run Shrake–Rupley
    params = freesasa.Parameters({'algorithm': freesasa.ShrakeRupley})
    result = freesasa.calc(structure, params)

    sasa_atoms = np.array([result.atomArea(i) for i in range(structure.nAtoms())])
    total_sasa = sasa_atoms.sum()

    return sasa_atoms, total_sasa, charges


def calculate_q_sasa(pqr_file):
    """
    Compute Q_SASA descriptor.

    Defined as:
        Q_SASA = Σ(q_i * SASA_i) / total_SASA

    Parameters:
        pqr_file (str): Path to the PQR file

    Returns:
        tuple:
            q_sasa (float): Final Q_SASA value
            numerator (float): Σ(q_i * SASA_i)
            total_sasa (float): Total SASA
    """
    sasa_atoms, total_sasa, charges = calculate_sasa_from_pqr(pqr_file)

    numerator = np.sum(charges * sasa_atoms)
    q_sasa = numerator / total_sasa if total_sasa > 0 else 0.0

    return q_sasa, numerator, total_sasa


def atoms_outside_grid_coords(coords, origin, spacing, grid_shape):
    """
    Identify atoms located outside the DX grid boundaries.

    Parameters:
        coords (array-like): Atomic coordinates (N, 3)
        origin (np.ndarray): Grid origin [ox, oy, oz]
        spacing (tuple): Grid spacing (dx, dy, dz)
        grid_shape (tuple): Grid dimensions (nx, ny, nz)

    Returns:
        list: Indices of atoms outside the grid
    """
    mins = origin
    maxs = origin + np.array(grid_shape) * np.array(spacing)

    outside_indices = [
        i for i, c in enumerate(coords)
        if np.any(np.array(c) < mins) or np.any(np.array(c) > maxs)
    ]

    return outside_indices


def calculate_see(pqr_file, dx_file, use_pdb_for_coords=False, pdb_file=None, debug=False):
    """
    Compute Surface Electrostatic Exposure (SEE).

    Defined as:
        SEE = Σ(q_i * φ_i * SASA_i) / Σ(SASA_i)

    Where:
        q_i   = atomic charge
        φ_i   = electrostatic potential
        SASA_i = solvent accessible surface area

    Features:
        - Uses PQR coordinates by default
        - Uses atomic radii from PQR
        - Supports optional PDB-based coordinates for SASA
        - Performs grid boundary diagnostics

    Parameters:
        pqr_file (str): Path to PQR file
        dx_file (str): Path to DX file (electrostatic potential)
        use_pdb_for_coords (bool): Whether to use PDB coordinates for SASA
        pdb_file (str): Required if use_pdb_for_coords=True
        debug (bool): Enable debug output

    Returns:
        float: SEE value
    """

    #Load PQR data
    coords_pqr, charges_pqr, radii_pqr = parse_pqr(pqr_file)
    N = len(coords_pqr)

    if N == 0:
        raise ValueError("parse_pqr returned zero atoms.")

    #Load DX grid
    potential_grid, origin, spacing = parse_dx(dx_file)
    grid_shape = potential_grid.shape

    #Select coordinates for SASA
    if use_pdb_for_coords:
        if pdb_file is None:
            raise ValueError("pdb_file must be provided when use_pdb_for_coords=True")

        structure = PDBParser(QUIET=True).get_structure("prot", pdb_file)

        atoms_pdb = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        atoms_pdb.append(
                            CustomAtom(
                                atom.get_name(),
                                residue.resname,
                                chain.id,
                                str(residue.id[1]),
                                atom.get_coord(),
                                charge=0.0
                            )
                        )

        coords_for_sasa = np.array([a.coord for a in atoms_pdb])

        if len(coords_for_sasa) != N:
            raise ValueError(
                f"PDB atom count ({len(coords_for_sasa)}) "
                f"!= PQR atom count ({N})"
            )
    else:
        coords_for_sasa = np.array(coords_pqr)

    #Compute SASA
    fs = freesasa.Structure()

    for i, (x, y, z) in enumerate(coords_for_sasa):
        fs.addAtom("X", "RES", str(i + 1), "A", float(x), float(y), float(z))

    fs.setRadii(radii_pqr)

    params = freesasa.Parameters({'algorithm': freesasa.ShrakeRupley})
    result = freesasa.calc(fs, params)

    sasa = np.array([result.atomArea(i) for i in range(fs.nAtoms())])

    #Diagnostics 
    if debug:
        zero_frac = np.mean(sasa == 0.0)
        print(f"Fraction of zero SASA atoms: {zero_frac:.3f}")

    outside_atoms = atoms_outside_grid_coords(coords_for_sasa, origin, spacing, grid_shape)

    if outside_atoms:
        print(f"WARNING: {len(outside_atoms)} atoms outside DX grid")

    #Build atom objects for interpolation
    atoms = [
        CustomAtom("X", "RES", "A", str(i + 1),
                   np.array(coords_pqr[i], dtype=float),
                   float(charges_pqr[i]))
        for i in range(N)
    ]

    #Interpolate potentials
    potentials = np.array([
        interpolate_potential(atom, potential_grid, origin, spacing)
        for atom in atoms
    ])

    #Ensure consistent vector lengths
    n = min(len(sasa), len(potentials), len(charges_pqr))

    sasa = sasa[:n]
    potentials = potentials[:n]
    charges = np.array(charges_pqr[:n])

    #Compute SEE
    numerator = np.sum(charges * potentials * sasa)
    denominator = np.sum(sasa)

    see = numerator / denominator if denominator > 0 else 0.0

    return see

def calculate_surface_potential_fraction(pqr_file, dx_file, threshold=1.0):
    """
    Calculates the percentage of surface atoms whose electrostatic potential
    is above a given threshold (default = 1 kT/e).

    Parameters:
        pqr_file (str): path to the PQR file
        dx_file (str): path to the DX electrostatic potential file
        threshold (float): potential threshold

    Returns:
        float: percentage of surface atoms above threshold
    """

    coords, charges, radii = parse_pqr(pqr_file)

    # Atomic SASA
    sasa_atoms, total_sasa, _ = calculate_sasa_from_pqr(pqr_file)

    # Load potential grid
    potential_grid, origin, spacing = parse_dx(dx_file)

    # Build atom objects
    atoms = [
        CustomAtom("X", "RES", "A", str(i+1), coords[i], charges[i])
        for i in range(len(coords))
    ]

    # Interpolate potentials
    potentials = np.array([
        interpolate_potential(atom, potential_grid, origin, spacing)
        for atom in atoms
    ])

    # Select surface atoms
    surface_mask = sasa_atoms > 0

    if surface_mask.sum() == 0:
        return 0.0

    surface_potentials = potentials[surface_mask]

    # Count atoms above threshold
    above_threshold = np.sum(surface_potentials > threshold)

    percentage = (above_threshold / len(surface_potentials)) * 100

    return percentage

def calculate_residue_exposed_charge(pqr_file, pdb_file):
    """
    Computes residue-level exposed charge using:

    - Atomic SASA from PQR (Shrake–Rupley)
    - Charges from PQR
    - Geometric normalization using 4πr² (maximum SASA)

    Returns:
        dict:
            total_charge (float)
            total_exposed_charge (float)
            percent_exposed_charge (float, 0–100)
            per_residue (list of dicts)
    """

    # --- 1) Load PQR data ---
    coords, charges, radii = parse_pqr(pqr_file)

    # --- 2) Atomic SASA ---
    sasa_atoms, total_sasa, charges_chk = calculate_sasa_from_pqr(pqr_file)
    if charges_chk is not None and len(charges_chk) == len(charges):
        charges = charges_chk

    N = len(charges)

    # --- 3) Map PDB ↔ PQR ---
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    pdb_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                for atom in residue:
                    pdb_atoms.append((residue.resname, residue.id[1], chain.id))

    if len(pdb_atoms) != N:
        raise ValueError(
            f"PDB atoms ({len(pdb_atoms)}) ≠ PQR atoms ({N}). Files are inconsistent."
        )

    # --- 4) Aggregate per residue ---
    residues = defaultdict(lambda: {
        'atom_indices': [],
        'net_charge': 0.0,
        'sasa': 0.0,
        'max_sasa': 0.0
    })

    for i, (resname, resnum, chain) in enumerate(pdb_atoms):
        key = f"{resname}{resnum}{chain}"

        residues[key]['atom_indices'].append(i)
        residues[key]['net_charge'] += float(charges[i])
        residues[key]['sasa'] += float(sasa_atoms[i])
        residues[key]['max_sasa'] += 4 * math.pi * (float(radii[i]) ** 2)

    # --- 5) Compute exposed charge ---
    per_residue = []
    total_charge = 0.0
    total_exposed_charge = 0.0

    for key, info in residues.items():
        net_q = info['net_charge']
        sasa_res = info['sasa']
        max_sasa_res = info['max_sasa']

        exposure_fraction = sasa_res / max_sasa_res if max_sasa_res > 0 else 0.0
        exposure_fraction = min(1.0, exposure_fraction)

        exposed_q = net_q * exposure_fraction

        per_residue.append({
            "residue": key,
            "net_charge": net_q,
            "sasa": sasa_res,
            "max_sasa": max_sasa_res,
            "exposure_fraction": exposure_fraction,
            "exposed_charge": exposed_q
        })

        total_charge += net_q
        total_exposed_charge += exposed_q

    # --- 6) Normalize percentage ---
    total_abs_charge = np.sum(np.abs(charges))
    percent_exposed_charge = (
        (total_exposed_charge / total_abs_charge) * 100
        if total_abs_charge > 1e-12 else 0.0
    )
    percent_exposed_charge = min(percent_exposed_charge, 100.0)

    return {
        "total_charge": total_charge,
        "total_exposed_charge": total_exposed_charge,
        "percent_exposed_charge": percent_exposed_charge,
        "per_residue": per_residue
    }

import os
import subprocess
import shutil

def process_single_protein(pdb_file, aux_output_dir):
    """
    Runs the full electrostatic + SASA + pKa pipeline for a single protein.

    Steps:
        - Generate PQR (PDB2PQR)
        - Run APBS
        - Compute descriptors:
            P_SASA, Q_SASA, ECPi, SEE, HSE, pKaI, ESE, SE

    Returns:
        tuple: results formatted for CSV
    """

    try:
        pdb_basename = os.path.basename(pdb_file)
        pdb_name = os.path.splitext(pdb_basename)[0]

        # --- Generate PQR ---
        pqr_file = f"{pdb_name}.pqr"
        subprocess.run(
            ['pdb2pqr', '--ff=PARSE', '--with-ph=7', pdb_file, pqr_file],
            check=True
        )

        # --- Run APBS ---
        in_path = generate_apbs_in_fixed(
            pdb_file, pdb_name, bbox_min, bbox_max, resolution=0.75
        )

        log_path = os.path.abspath(f"{pdb_name}.out")
        with open(log_path, "w") as logf:
            subprocess.run(
                ['apbs', in_path],
                stdout=logf,
                stderr=subprocess.STDOUT,
                check=True
            )

        # --- Solvation energy ---
        solvation_energy = parse_apbs_energy(log_path)

        # --- Locate DX file ---
        dx_candidates = [
            f"{pdb_name}.pqr-PE0.dx",
            f"{pdb_name}.dx",
            f"{pdb_name}.dx-PE0.dx"
        ]

        dx_path = next(
            (os.path.abspath(f) for f in dx_candidates if os.path.isfile(f)),
            None
        )

        if dx_path is None:
            raise FileNotFoundError(f"DX file not found for {pdb_name}")

        # --- Metrics ---
        p_sasa, _, _ = calculate_p_sasa(pqr_file, pdb_file, dx_path)
        q_sasa, _, _ = calculate_q_sasa(pqr_file)

        ecpi_data = calculate_residue_exposed_charge(pqr_file, pdb_file)
        percent_exposed_charge = ecpi_data["percent_exposed_charge"]

        epi_val = calculate_epi(pqr_file, dx_path)

        hse_val, _, _ = calculate_hse(pdb_file, pqr_file)

        pkaI_val = calculate_protein_pka_sasa(pdb_file, pqr_file)

        ese_val = calculate_electrostatic_stability_estimator(pdb_file, pqr_file)

        surface_potential_percent = calculate_surface_potential_fraction(
            pqr_file, dx_path
        )

        # --- Print summary ---
        print(f"✅ Protein: {pdb_name}")
        print(f" P_SASA: {p_sasa:.2f}")
        print(f" Q_SASA: {q_sasa:.2f}")
        print(f" ECP (%): {percent_exposed_charge:.2f}")
        print(f" SEE: {epi_val:.2f}")
        print(f" HSE: {hse_val:.2f}")
        print(f" pKaI: {pkaI_val:.2f}")
        print(f" ESE: {ese_val:.2f}")
        print(f" SE: {solvation_energy:.2f}" if solvation_energy else " SE: N/A")
        print(f" Surface >1 kT/e: {surface_potential_percent:.2f}")

        # --- Move auxiliary files ---
        os.makedirs(aux_output_dir, exist_ok=True)

        files_to_move = [
            f"{pdb_name}.in",
            f"{pdb_name}.out",
            f"{pdb_name}.pka",
            f"{pdb_name}.pqr",
            dx_path,
            f"{pdb_name}-input.p"
        ]

        for f in files_to_move:
            if os.path.exists(f):
                try:
                    shutil.move(f, os.path.join(aux_output_dir, os.path.basename(f)))
                except Exception as e:
                    print(f"⚠️ Could not move {f}: {e}")

        return (
            pdb_name,
            f"{p_sasa:.4f}",
            f"{q_sasa:.4f}",
            f"{percent_exposed_charge:.4f}",
            f"{epi_val:.4f}",
            f"{hse_val:.4f}",
            f"{pkaI_val:.4f}",
            f"{ese_val:.4f}",
            f"{solvation_energy:.4f}" if solvation_energy else "N/A",
            f"{surface_potential_percent:.4f}"
        )

    except Exception as e:
        print(f"⚠️ Error processing {pdb_file}: {e}")
        return (os.path.basename(pdb_file), "ERROR", None, None, None, None, None, None, None)
    