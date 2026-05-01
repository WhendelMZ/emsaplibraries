"""Helpers for structure alignment and APBS/PDB2PQR electrostatic runs."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np

from ._runtime import require_executable, require_module


def replace_rU_with_r(file_path: str | Path) -> None:
    """Replace deprecated ``rU`` file mode with ``r`` in a file."""
    path = Path(file_path)
    content = path.read_text(encoding="utf-8")
    path.write_text(content.replace("rU", "r"), encoding="utf-8")


def patch_pdb2pqr_legacy_files(paths: list[str | Path]) -> None:
    """Patch legacy PDB2PQR Python files explicitly; never runs at import time."""
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_dir():
            continue
        for file_path in path.rglob("*.py"):
            replace_rU_with_r(file_path)


def align_proteins_to_reference(
    reference_pdb: str | Path,
    input_dir: str | Path,
    output_dir: str | Path,
) -> None:
    """Align PDB structures to a reference using C-alpha atoms."""
    bio_pdb = require_module("Bio.PDB", "pip install biopython")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    parser = bio_pdb.PDBParser(QUIET=True)
    io = bio_pdb.PDBIO()
    sup = bio_pdb.Superimposer()

    ref_structure = parser.get_structure("ref", str(reference_pdb))
    ref_atoms = [atom for atom in ref_structure.get_atoms() if atom.get_id() == "CA"]
    if not ref_atoms:
        raise ValueError("Reference structure contains no CA atoms.")

    for pdb_file in Path(input_dir).glob("*.pdb"):
        structure = parser.get_structure(pdb_file.name, str(pdb_file))
        atoms = [atom for atom in structure.get_atoms() if atom.get_id() == "CA"]
        if not atoms:
            continue

        min_len = min(len(ref_atoms), len(atoms))
        sup.set_atoms(ref_atoms[:min_len], atoms[:min_len])
        sup.apply(structure.get_atoms())
        io.set_structure(structure)
        io.save(str(output / pdb_file.name))


def generate_apbs_in_fixed(
    pdb_file: str | Path,
    out_basename: str | Path,
    bbox_min,
    bbox_max,
    resolution: float = 0.75,
) -> str:
    """Generate an APBS input file using a fixed global bounding box."""
    bbox_min = np.array(bbox_min)
    bbox_max = np.array(bbox_max)
    bbox_size = bbox_max - bbox_min
    dime = np.ceil(bbox_size / resolution).astype(int)

    def adjust_dime(values):
        adjusted = []
        for value in values:
            n = 3
            while (2**n + 1) < value:
                n += 1
            adjusted.append(2**n + 1)
        return np.array(adjusted)

    dime = adjust_dime(dime)
    center = (bbox_min + bbox_max) / 2.0
    basename = str(out_basename)
    in_file = f"{basename}.in"

    with open(in_file, "w", encoding="utf-8") as handle:
        handle.write(f"read\n  mol pqr {basename}.pqr\nend\n")
        handle.write("elec\n")
        handle.write("  mg-auto\n")
        handle.write(f"  dime {dime[0]} {dime[1]} {dime[2]}\n")
        handle.write(
            f"  fglen {bbox_size[0]:.3f} {bbox_size[1]:.3f} {bbox_size[2]:.3f}\n"
        )
        handle.write(
            f"  cglen {bbox_size[0]:.3f} {bbox_size[1]:.3f} {bbox_size[2]:.3f}\n"
        )
        handle.write(f"  fgcent {center[0]:.3f} {center[1]:.3f} {center[2]:.3f}\n")
        handle.write(f"  cgcent {center[0]:.3f} {center[1]:.3f} {center[2]:.3f}\n")
        handle.write("  mol 1\n")
        handle.write("  npbe\n")
        handle.write("  bcfl sdh\n")
        handle.write("  pdie 2.0\n")
        handle.write("  sdie 78.54\n")
        handle.write("  srfm smol\n")
        handle.write("  chgm spl2\n")
        handle.write("  sdens 10.0\n")
        handle.write("  srad 1.4\n")
        handle.write("  swin 0.3\n")
        handle.write("  temp 298.15\n")
        handle.write("  calcenergy total\n")
        handle.write("  calcforce no\n")
        handle.write("write\n")
        handle.write(f"  pot dx {basename}.dx\n")
        handle.write("end\n")
        handle.write("print elecEnergy 1 end\n")
        handle.write("quit\n")

    return in_file


def run_pdb2pqr(pdb_file: str | Path, pdb_name: str) -> str:
    """Run PDB2PQR and return the generated PQR path."""
    require_executable("pdb2pqr", "Install PDB2PQR and ensure 'pdb2pqr' is on PATH.")
    pqr_file = f"{pdb_name}.pqr"
    subprocess.run(
        ["pdb2pqr", "--ff=PARSE", "--with-ph=7", str(pdb_file), pqr_file],
        check=True,
    )
    return pqr_file


def run_apbs(pdb_file: str | Path, pdb_name: str, bbox_min, bbox_max) -> str:
    """Run APBS and return the output log path."""
    require_executable("apbs", "Install APBS and ensure 'apbs' is on PATH.")
    in_path = generate_apbs_in_fixed(pdb_file, pdb_name, bbox_min, bbox_max)
    log_path = os.path.abspath(f"{pdb_name}.out")

    with open(log_path, "w", encoding="utf-8") as log_handle:
        subprocess.run(
            ["apbs", in_path], stdout=log_handle, stderr=subprocess.STDOUT, check=True
        )

    return log_path


def find_dx_file(pdb_name: str) -> str:
    """Locate the electrostatic potential DX file generated by APBS."""
    candidates = [
        f"{pdb_name}.pqr-PE0.dx",
        f"{pdb_name}.dx",
        f"{pdb_name}.dx-PE0.dx",
    ]
    dx_path = next(
        (os.path.abspath(path) for path in candidates if os.path.isfile(path)), None
    )
    if dx_path is None:
        raise FileNotFoundError(f"DX file not found for {pdb_name}.")
    return dx_path
