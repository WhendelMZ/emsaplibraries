from pathlib import Path
import zipfile
import os
import glob
from Bio.PDB import MMCIFParser, PDBIO

def build_mapping_from_manifest(manifest_data):
    """
    Build mapping (seq_id -> name) from manifest_data.
    Here we use label as the output name (you can customize).
    """
    mapping = {}

    for entries in manifest_data.values():
        for entry in entries:
            seq_id = entry["id"].lower()
            label = entry["label"]

            mapping[seq_id] = label  # or customize this

    return mapping

def clean_name(name: str) -> str:
    """Sanitize filenames by replacing unsafe characters."""
    for ch in [" ", "|", ";", ",", "/", "\\", "(", ")", "[", "]", "{", "}", ":", "=", "+"]:
        name = name.replace(ch, "_")
    return name


def make_unique_path(path: Path) -> Path:
    """Ensure file path is unique by adding suffix if needed."""
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
    """Extract normalized sequence ID from filename."""
    name = name.lower()
    for token in [
        "fold_", "_model_0", "_summary", "summary",
        "_confidences_0", "confidences_0",
        "_confidence_0", "confidence_0"
    ]:
        name = name.replace(token, "")
    return name

def extract_cif_and_confidence_files(zip_dir, cif_dir, qm_dir, mapping):
    """
    Extract CIF and confidence files from ZIP archives using a mapping dict.
    """

    zip_dir = Path(zip_dir)
    cif_dir = Path(cif_dir)
    qm_dir = Path(qm_dir)

    cif_dir.mkdir(parents=True, exist_ok=True)
    qm_dir.mkdir(parents=True, exist_ok=True)

    if not zip_dir.exists():
        raise FileNotFoundError(f"❌ Folder not found: {zip_dir}")

    extracted_cif = 0
    extracted_qm = 0
    seen_confidences = set()

    for zip_path in zip_dir.glob("*.zip"):
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for file_info in zip_ref.infolist():

                    if file_info.is_dir():
                        continue

                    inner_name = Path(file_info.filename).name
                    ext = Path(inner_name).suffix

                    seq_id = extract_seq_id(Path(inner_name).stem)
                    mapped_name = mapping.get(seq_id, seq_id)
                    cleaned_name = clean_name(mapped_name)

                    dest_filename = f"{cleaned_name}{ext}"

                    # CIF extraction
                    if "model_0" in inner_name.lower():
                        dest_path = make_unique_path(cif_dir / dest_filename)

                        with zip_ref.open(file_info) as src, open(dest_path, "wb") as dst:
                            dst.write(src.read())

                        extracted_cif += 1

                    # Confidence extraction
                    elif any(k in inner_name.lower() for k in ["confidence_0", "confidences_0"]):
                        if seq_id not in seen_confidences:
                            dest_path = make_unique_path(qm_dir / dest_filename)

                            with zip_ref.open(file_info) as src, open(dest_path, "wb") as dst:
                                dst.write(src.read())

                            extracted_qm += 1
                            seen_confidences.add(seq_id)

        except zipfile.BadZipFile:
            print(f"❌ Invalid ZIP file: {zip_path.name}")
        except Exception as e:
            print(f"❌ Error processing {zip_path.name}: {e}")

def cif_to_pdb(cif_folder, pdb_folder, mapping):
    """
    Convert CIF files to PDB format using mapping dict.
    """

    os.makedirs(pdb_folder, exist_ok=True)

    parser = MMCIFParser(QUIET=True)
    io = PDBIO()

    for cif_path in Path(cif_folder).glob("*.cif"):

        cif_name = cif_path.stem.lower()
        seq_id = extract_seq_id(cif_name)

        mapped_name = mapping.get(seq_id, seq_id)
        cleaned_name = clean_name(mapped_name)

        pdb_path = make_unique_path(Path(pdb_folder) / f"{cleaned_name}.pdb")

        try:
            structure = parser.get_structure(cleaned_name, str(cif_path))
            io.set_structure(structure)
            io.save(str(pdb_path))

        except Exception as e:
            print(f"❌ Error converting {cif_name}.cif: {e}")

def cif_to_pdb(cif_folder, pdb_folder, mapping):
    """
    Convert CIF files to PDB format using mapping dict.
    """

    os.makedirs(pdb_folder, exist_ok=True)

    parser = MMCIFParser(QUIET=True)
    io = PDBIO()

    for cif_path in Path(cif_folder).glob("*.cif"):

        cif_name = cif_path.stem.lower()
        seq_id = extract_seq_id(cif_name)

        mapped_name = mapping.get(seq_id, seq_id)
        cleaned_name = clean_name(mapped_name)

        pdb_path = make_unique_path(Path(pdb_folder) / f"{cleaned_name}.pdb")

        try:
            structure = parser.get_structure(cleaned_name, str(cif_path))
            io.set_structure(structure)
            io.save(str(pdb_path))

        except Exception as e:
            print(f"❌ Error converting {cif_name}.cif: {e}")

def relax_pdbs(input_folder: str, output_folder: str):
    # Fixed parameters
    coord_dev = 0.5  # harmonic constraint standard deviation
    weight = 1.0     # weight of coordinate constraints

    os.makedirs(output_folder, exist_ok=True)

    def add_ca_constraints(pose: Pose):
        """Add harmonic coordinate constraints to all C-alpha atoms."""
        func = HarmonicFunc(0.0, coord_dev)
        for i in range(1, pose.total_residue() + 1):
            if not pose.residue(i).has("CA"):
                continue
            atom_id = AtomID(pose.residue(i).atom_index("CA"), i)
            xyz = pose.residue(i).xyz("CA")
            xyz_vec = xyzVector_double_t(xyz.x, xyz.y, xyz.z)
            constraint = CoordinateConstraint(atom_id, AtomID(1, 1), xyz_vec, func)
            pose.add_constraint(constraint)

    # Loop over all PDB files
    for filename in os.listdir(input_folder):
        if filename.endswith(".pdb"):
            pdb_path = os.path.join(input_folder, filename)
            pose = pose_from_pdb(pdb_path)

            # Apply C-alpha constraints
            add_ca_constraints(pose)

            # Setup MoveMap
            movemap = MoveMap()
            movemap.set_bb(False)  # backbone
            movemap.set_chi(True)  # sidechains

            # Setup FastRelax
            scorefxn = ScoreFunctionFactory.create_score_function("ref2015")
            scorefxn.set_weight(pyrosetta.rosetta.core.scoring.coordinate_constraint, weight)

            relax = FastRelax()
            relax.set_scorefxn(scorefxn)
            relax.set_movemap(movemap)
            relax.max_iter(1)  # 1 cycle

            # Apply relaxation
            relax.apply(pose)

            # Save relaxed PDB
            output_path = os.path.join(output_folder, filename)
            pose.dump_pdb(output_path)
            
