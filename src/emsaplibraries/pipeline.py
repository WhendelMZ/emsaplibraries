"""Pipeline-style orchestration helpers for whole-protein processing."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ._runtime import require_executable
from .electrostatics import find_dx_file, generate_apbs_in_fixed
from .indicators import (
    acid_base_stability_estimator,
    calculate_hse,
    calculate_p_sasa,
    calculate_protein_pka_sasa,
    calculate_q_sasa,
    calculate_residue_exposed_charge,
    calculate_see,
)


@dataclass
class ProteinPipelineResult:
    """Metrics and generated artifacts for one processed protein."""

    protein_name: str
    p_sasa: float
    q_sasa: float
    ecpi_percent: float
    see: float
    hse: float
    pka_sasa: float
    abse: float
    pdb_file: Path
    pqr_file: Path
    dx_file: Path
    apbs_input_file: Path
    apbs_log_file: Path
    pka_file: Path | None = None

    def as_legacy_tuple(self) -> tuple[str, str, str, str, str, str, str, str]:
        """Return the legacy tuple shape with metrics formatted to two decimals."""
        return (
            self.protein_name,
            f"{self.p_sasa:.2f}",
            f"{self.q_sasa:.2f}",
            f"{self.ecpi_percent:.2f}",
            f"{self.see:.2f}",
            f"{self.hse:.2f}",
            f"{self.pka_sasa:.2f}",
            f"{self.abse:.2f}",
        )


def _print_summary(result: ProteinPipelineResult) -> None:
    print(f"Protein: {result.protein_name}")
    print(
        " Solvent-Accessible Surface Potential - P_SASA (kBT/e): "
        f"{result.p_sasa:.2f}"
    )
    print(
        " Solvent-Accessible Surface Charge - Q_SASA (e): "
        f"{result.q_sasa:.2f}"
    )
    print(f" Exposed Charge % Index : {result.ecpi_percent:.2f}")
    print(f" Surface Electrostatic Energy - SEE (kBT): {result.see:.2f}")
    print(f" Hydrophobic Surface Exposure - HSE (dimensionless): {result.hse:.2f}")
    print(
        " pKa Index of Ionizable Residue Groups - pKaI (dimensionless): "
        f"{result.pka_sasa:.2f}"
    )
    print(f" Acid-Base Stability Estimator - ABSE (kJ/mol): {result.abse:.2f}")


def _move_if_exists(path: str | Path, output_dir: Path) -> Path | None:
    source = Path(path)
    if not source.exists():
        return None

    destination = output_dir / source.name
    if source.resolve() == destination.resolve():
        return destination

    shutil.move(str(source), destination)
    return destination


def process_single_protein(
    pdb_file: str | Path,
    aux_output_dir: str | Path,
    bbox_min,
    bbox_max,
    *,
    verbose: bool = False,
) -> ProteinPipelineResult:
    """Run the available PDB2PQR/APBS indicator pipeline for one protein."""
    require_executable("pdb2pqr", "Install PDB2PQR and ensure 'pdb2pqr' is on PATH.")
    require_executable("apbs", "Install APBS and ensure 'apbs' is on PATH.")

    pdb_path = Path(pdb_file)
    pdb_name = pdb_path.stem
    pqr_file = Path(f"{pdb_name}.pqr")

    subprocess.run(
        ["pdb2pqr", "--ff=PARSE", "--with-ph=7", str(pdb_path), str(pqr_file)],
        check=True,
    )
    in_path = Path(
        generate_apbs_in_fixed(pdb_path, pdb_name, bbox_min, bbox_max, resolution=0.75)
    )
    log_path = Path(f"{pdb_name}.out").resolve()
    with open(log_path, "w", encoding="utf-8") as log_handle:
        subprocess.run(
            ["apbs", str(in_path)],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=True,
        )

    dx_path = Path(find_dx_file(pdb_name))

    p_sasa, _, _ = calculate_p_sasa(pqr_file, pdb_path, dx_path)
    q_sasa, _, _ = calculate_q_sasa(pqr_file)
    ecpi_data = calculate_residue_exposed_charge(pqr_file, pdb_path)
    see_val = calculate_see(pqr_file, dx_path)
    pka_sasa_val = calculate_protein_pka_sasa(pdb_file, pqr_file)
    hse_val, _, _ = calculate_hse(pdb_file, pqr_file)
    abse_val = acid_base_stability_estimator(pdb_file, pqr_file)

    aux_dir = Path(aux_output_dir)
    aux_dir.mkdir(parents=True, exist_ok=True)

    moved_in = _move_if_exists(in_path, aux_dir) or in_path
    moved_out = _move_if_exists(log_path, aux_dir) or log_path
    moved_pka = _move_if_exists(f"{pdb_name}.pka", aux_dir)
    moved_pqr = _move_if_exists(pqr_file, aux_dir) or pqr_file
    moved_dx = _move_if_exists(dx_path, aux_dir) or dx_path
    _move_if_exists(f"{pdb_name}-input.p", aux_dir)

    result = ProteinPipelineResult(
        protein_name=pdb_name,
        p_sasa=float(p_sasa),
        q_sasa=float(q_sasa),
        ecpi_percent=float(ecpi_data["percent_exposed_charge"]),
        see=float(see_val),
        hse=float(hse_val),
        pka_sasa=float(pka_sasa_val),
        abse=float(abse_val),
        pdb_file=pdb_path,
        pqr_file=moved_pqr,
        dx_file=moved_dx,
        apbs_input_file=moved_in,
        apbs_log_file=moved_out,
        pka_file=moved_pka,
    )

    if verbose:
        _print_summary(result)

    return result
