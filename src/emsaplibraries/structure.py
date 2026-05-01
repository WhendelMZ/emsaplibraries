"""Protein structure archive extraction, CIF/PDB conversion, and relaxation."""

from __future__ import annotations

from pathlib import Path
import os
import warnings
import zipfile

from ._runtime import require_module


def build_mapping_from_manifest(manifest_data) -> dict[str, str]:
    """Build a sequence ID to label mapping from manifest data."""
    mapping = {}
    for entries in manifest_data.values():
        for entry in entries:
            mapping[entry["id"].lower()] = entry["label"]
    return mapping


def clean_name(name: str) -> str:
    """Sanitize names for filenames."""
    for ch in [" ", "|", ";", ",", "/", "\\", "(", ")", "[", "]", "{", "}", ":", "=", "+"]:
        name = name.replace(ch, "_")
    return name


def make_unique_path(path: Path) -> Path:
    """Ensure a file path is unique by adding a numeric suffix if needed."""
    if not path.exists():
        return path
    base = path.stem
    suffix = 1
    while True:
        new_path = path.with_name(f"{base}_{suffix}{path.suffix}")
        if not new_path.exists():
            return new_path
        suffix += 1


def extract_seq_id(name: str) -> str:
    """Extract normalized sequence ID from AlphaFold-style filenames."""
    normalized = name.lower()
    for token in [
        "fold_",
        "_model_0",
        "_summary",
        "summary",
        "_confidences_0",
        "confidences_0",
        "_confidence_0",
        "confidence_0",
    ]:
        normalized = normalized.replace(token, "")
    return normalized


def extract_cif_and_confidence_files(
    zip_dir: str | Path,
    cif_dir: str | Path,
    qm_dir: str | Path,
    mapping: dict[str, str],
) -> tuple[int, int]:
    """Extract CIF and confidence files from ZIP archives using a mapping dict."""
    zip_path = Path(zip_dir)
    cif_path = Path(cif_dir)
    qm_path = Path(qm_dir)
    cif_path.mkdir(parents=True, exist_ok=True)
    qm_path.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        raise FileNotFoundError(f"Folder not found: {zip_path}")

    extracted_cif = 0
    extracted_qm = 0
    seen_confidences = set()

    for archive_path in zip_path.glob("*.zip"):
        try:
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                for file_info in zip_ref.infolist():
                    if file_info.is_dir():
                        continue

                    inner_name = Path(file_info.filename).name
                    ext = Path(inner_name).suffix
                    seq_id = extract_seq_id(Path(inner_name).stem)
                    dest_filename = f"{clean_name(mapping.get(seq_id, seq_id))}{ext}"

                    if "model_0" in inner_name.lower():
                        dest_path = make_unique_path(cif_path / dest_filename)
                        with zip_ref.open(file_info) as src, open(dest_path, "wb") as dst:
                            dst.write(src.read())
                        extracted_cif += 1
                    elif any(k in inner_name.lower() for k in ["confidence_0", "confidences_0"]):
                        if seq_id not in seen_confidences:
                            dest_path = make_unique_path(qm_path / dest_filename)
                            with zip_ref.open(file_info) as src, open(dest_path, "wb") as dst:
                                dst.write(src.read())
                            extracted_qm += 1
                            seen_confidences.add(seq_id)
        except zipfile.BadZipFile:
            warnings.warn(f"Invalid ZIP file: {archive_path.name}", RuntimeWarning)

    return extracted_cif, extracted_qm


def cif_to_pdb(cif_folder: str | Path, pdb_folder: str | Path, mapping: dict[str, str]) -> list[Path]:
    """Convert CIF files to PDB format using a mapping dict."""
    bio_pdb = require_module("Bio.PDB", "pip install biopython")
    output = Path(pdb_folder)
    output.mkdir(parents=True, exist_ok=True)
    parser = bio_pdb.MMCIFParser(QUIET=True)
    io = bio_pdb.PDBIO()
    written = []

    for cif_path in Path(cif_folder).glob("*.cif"):
        seq_id = extract_seq_id(cif_path.stem.lower())
        cleaned_name = clean_name(mapping.get(seq_id, seq_id))
        pdb_path = make_unique_path(output / f"{cleaned_name}.pdb")
        structure = parser.get_structure(cleaned_name, str(cif_path))
        io.set_structure(structure)
        io.save(str(pdb_path))
        written.append(pdb_path)

    return written


def _import_pyrosetta():
    try:
        import pyrosetta
        from pyrosetta import MoveMap, pose_from_pdb
        from pyrosetta.rosetta.core.id import AtomID
        from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory
        from pyrosetta.rosetta.core.scoring.constraints import CoordinateConstraint
        from pyrosetta.rosetta.core.scoring.func import HarmonicFunc
        from pyrosetta.rosetta.numeric import xyzVector_double_t
        from pyrosetta.rosetta.protocols.relax import FastRelax
    except ImportError as exc:
        raise ImportError(
            "PyRosetta is required for relax_pdbs. Install PyRosetta according to "
            "the official licensing and installation instructions, or use "
            "pyrosetta-installer where appropriate."
        ) from exc

    if not pyrosetta.rosetta.basic.was_init_called():
        pyrosetta.init("-mute all")

    return {
        "pyrosetta": pyrosetta,
        "MoveMap": MoveMap,
        "pose_from_pdb": pose_from_pdb,
        "AtomID": AtomID,
        "ScoreFunctionFactory": ScoreFunctionFactory,
        "CoordinateConstraint": CoordinateConstraint,
        "HarmonicFunc": HarmonicFunc,
        "xyzVector_double_t": xyzVector_double_t,
        "FastRelax": FastRelax,
    }


def relax_pdbs(input_folder: str | Path, output_folder: str | Path) -> None:
    """Relax PDB sidechains with PyRosetta while constraining C-alpha atoms."""
    pr = _import_pyrosetta()
    output = Path(output_folder)
    output.mkdir(parents=True, exist_ok=True)
    coord_dev = 0.5
    weight = 1.0

    def add_ca_constraints(pose):
        func = pr["HarmonicFunc"](0.0, coord_dev)
        for index in range(1, pose.total_residue() + 1):
            if not pose.residue(index).has("CA"):
                continue
            atom_id = pr["AtomID"](pose.residue(index).atom_index("CA"), index)
            xyz = pose.residue(index).xyz("CA")
            xyz_vec = pr["xyzVector_double_t"](xyz.x, xyz.y, xyz.z)
            constraint = pr["CoordinateConstraint"](atom_id, pr["AtomID"](1, 1), xyz_vec, func)
            pose.add_constraint(constraint)

    for filename in os.listdir(input_folder):
        if not filename.endswith(".pdb"):
            continue
        pose = pr["pose_from_pdb"](str(Path(input_folder) / filename))
        add_ca_constraints(pose)

        movemap = pr["MoveMap"]()
        movemap.set_bb(False)
        movemap.set_chi(True)

        scorefxn = pr["ScoreFunctionFactory"].create_score_function("ref2015")
        scorefxn.set_weight(
            pr["pyrosetta"].rosetta.core.scoring.coordinate_constraint,
            weight,
        )

        relax = pr["FastRelax"]()
        relax.set_scorefxn(scorefxn)
        relax.set_movemap(movemap)
        relax.max_iter(1)
        relax.apply(pose)
        pose.dump_pdb(str(output / filename))
