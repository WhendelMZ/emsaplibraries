"""Optional wrappers for third-party command-line tools."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from ._runtime import require_executable
from .electrostatics import generate_apbs_in_fixed


def run_pdb2pqr(pdb_file: str | Path, pdb_name: str) -> str:
    """Run PDB2PQR and return the generated PQR path."""
    pdb2pqr = require_executable(
        "pdb2pqr", "Install PDB2PQR and ensure 'pdb2pqr' is on PATH."
    )
    pqr_file = f"{pdb_name}.pqr"
    subprocess.run(
        [pdb2pqr, "--ff=PARSE", "--with-ph=7", str(pdb_file), pqr_file],
        check=True,
    )
    return pqr_file


def run_apbs(pdb_file: str | Path, pdb_name: str, bbox_min, bbox_max) -> str:
    """Run APBS and return the output log path."""
    apbs = require_executable("apbs", "Install APBS and ensure 'apbs' is on PATH.")
    in_path = generate_apbs_in_fixed(pdb_file, pdb_name, bbox_min, bbox_max)
    log_path = os.path.abspath(f"{pdb_name}.out")

    with open(log_path, "w", encoding="utf-8") as log_handle:
        subprocess.run(
            [apbs, in_path], stdout=log_handle, stderr=subprocess.STDOUT, check=True
        )

    return log_path


def run_mafft(input_fasta: str | Path, output_fasta: str | Path) -> None:
    """Perform multiple sequence alignment using the MAFFT executable."""
    mafft = require_executable("mafft", "Install MAFFT and ensure 'mafft' is on PATH.")
    result = subprocess.run(
        [mafft, str(input_fasta)],
        check=True,
        capture_output=True,
        text=True,
    )
    Path(output_fasta).write_text(result.stdout, encoding="utf-8")


def run_propka(pdb_file: str | Path) -> Path:
    """Run PROPKA and return the generated .pka file path."""
    propka = require_executable(
        "propka3", "Install PROPKA and ensure 'propka3' is on PATH."
    )
    pdb_path = Path(pdb_file)
    pka_path = Path(f"{pdb_path.stem}.pka")

    subprocess.run([propka, str(pdb_path)], check=True)

    if not pka_path.exists():
        raise FileNotFoundError(f"PROPKA output file not found: {pka_path}")
    return pka_path
