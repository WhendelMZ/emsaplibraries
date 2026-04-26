import os
import re
import math
import shutil
import subprocess
import numpy as np

# Function to replace 'rU' with 'r' in a file
def replace_rU_with_r(file_path):

    """
    Replace deprecated 'rU' file mode with 'r' in a Python file.

    This function reads the content of a file, replaces all occurrences
    of the deprecated universal newline mode 'rU' with 'r', and writes
    the modified content back to the same file.

    Parameters:
        file_path (str): Path to the Python file to be modified.

    Notes:
        - This is useful for fixing legacy Python 2 code compatibility issues.
        - The function overwrites the original file.

    Raises:
        Exception: If the file cannot be accessed or modified.
    """
    try:
        # Read file content
        with open(file_path, 'r') as file:
            content = file.read()

        # Replace 'rU' with 'r'
        modified_content = content.replace("rU", "r")

        # Write changes back to the same file
        with open(file_path, 'w') as file:
            file.write(modified_content)

        # Optional: print confirmation
        # print(f"Replacements applied to file: {file_path}")

    except Exception as e:
        print(f"Error accessing file: {e}")


# Possible pdb2pqr installation paths
paths = [
    "/usr/share/doc/pdb2pqr",
    "/usr/share/apbs/tools/conversion/param/pdb2pqr",
    "/usr/share/pdb2pqr",
    "/usr/lib/pdb2pqr",
    "/usr/bin/pdb2pqr"
]


# Traverse directories and modify Python files
for path in paths:
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".py"):  # Only process Python files
                    file_path = os.path.join(root, file)
                    replace_rU_with_r(file_path)

def align_proteins_to_reference(reference_pdb, input_dir, output_dir):
    """
    Align all protein structures in a directory to a reference structure
    using C-alpha (CA) atoms.

    The aligned structures are saved in the specified output directory.

    Parameters:
        reference_pdb (str): Path to the reference PDB file.
        input_dir (str): Directory containing PDB files to be aligned.
        output_dir (str): Directory where aligned PDB files will be saved.

    Raises:
        ValueError: If the reference structure contains no CA atoms.
    """
    os.makedirs(output_dir, exist_ok=True)

    parser = PDBParser(QUIET=True)
    io = PDBIO()

    # Load reference structure
    ref_structure = parser.get_structure("ref", reference_pdb)
    ref_atoms = [atom for atom in ref_structure.get_atoms() if atom.get_id() == "CA"]

    if len(ref_atoms) == 0:
        raise ValueError("Reference structure contains no CA atoms.")

    sup = Superimposer()

    # Loop through all PDB files in the directory
    for pdb_file in os.listdir(input_dir):
        if not pdb_file.endswith(".pdb"):
            continue

        pdb_path = os.path.join(input_dir, pdb_file)

        try:
            structure = parser.get_structure(pdb_file, pdb_path)
            atoms = [atom for atom in structure.get_atoms() if atom.get_id() == "CA"]

            if len(atoms) == 0:
                continue

            # Align using minimum number of atoms
            min_len = min(len(ref_atoms), len(atoms))
            sup.set_atoms(ref_atoms[:min_len], atoms[:min_len])
            sup.apply(structure.get_atoms())

            # Save aligned PDB
            io.set_structure(structure)
            aligned_path = os.path.join(output_dir, pdb_file)
            io.save(aligned_path)

        except Exception as e:
            print(f"⚠️ Error processing {pdb_file}: {e}")

def generate_apbs_in_fixed(pdb_file, out_basename, bbox_min, bbox_max, resolution=0.75):
    """
    Generate an APBS input (.in) file using a fixed global bounding box.

    This approach ensures that all electrostatic calculations are performed
    within the same spatial grid, which is critical for comparative analysis
    across multiple aligned protein structures.

    Parameters:
        pdb_file (str): Path to the PDB file.
        out_basename (str): Base name for output files (.dx, .out, etc.).
        bbox_min (array-like): Minimum bounding box coordinates [x_min, y_min, z_min].
        bbox_max (array-like): Maximum bounding box coordinates [x_max, y_max, z_max].
        resolution (float): Grid spacing in Å (default = 0.75).

    Returns:
        str: Path to the generated APBS input file.
    """
    import numpy as np

    # --- Compute bounding box size ---
    bbox_min = np.array(bbox_min)
    bbox_max = np.array(bbox_max)
    bbox_size = bbox_max - bbox_min

    # --- Compute grid dimensions (DIME) ---
    dime = np.ceil(bbox_size / resolution).astype(int)

    # --- Adjust to APBS format (2^n + 1) ---
    def adjust_dime(dime):
        adjusted = []
        for d in dime:
            n = 3
            while (2**n + 1) < d:
                n += 1
            adjusted.append(2**n + 1)
        return np.array(adjusted)

    dime = adjust_dime(dime)

    # --- Compute box center ---
    center = (bbox_min + bbox_max) / 2.0

    # --- Output file path ---
    in_file = f"{out_basename}.in"

    # --- Compute actual grid spacing (for reference/debug) ---
    grid_spacing = bbox_size / (dime - 1)

    # --- Write APBS input file ---
    with open(in_file, "w") as f:
        f.write(f"read\n  mol pqr {out_basename}.pqr\nend\n")
        f.write("elec\n")
        f.write("  mg-auto\n")
        f.write(f"  dime {dime[0]} {dime[1]} {dime[2]}\n")
        f.write(f"  fglen {bbox_size[0]:.3f} {bbox_size[1]:.3f} {bbox_size[2]:.3f}\n")
        f.write(f"  cglen {bbox_size[0]:.3f} {bbox_size[1]:.3f} {bbox_size[2]:.3f}\n")
        f.write(f"  fgcent {center[0]:.3f} {center[1]:.3f} {center[2]:.3f}\n")
        f.write(f"  cgcent {center[0]:.3f} {center[1]:.3f} {center[2]:.3f}\n")
        f.write("  mol 1\n")
        f.write("  npbe\n")
        f.write("  bcfl sdh\n")
        f.write("  pdie 2.0\n")
        f.write("  sdie 78.54\n")
        f.write("  srfm smol\n")
        f.write("  chgm spl2\n")
        f.write("  sdens 10.0\n")
        f.write("  srad 1.4\n")
        f.write("  swin 0.3\n")
        f.write("  temp 298.15\n")
        f.write("  calcenergy total\n")
        f.write("  calcforce no\n")
        f.write("write\n")
        f.write(f"  pot dx {out_basename}.dx\n")
        f.write("end\n")
        f.write("print elecEnergy 1 end\n")
        f.write("quit\n")

    return in_file

def run_pdb2pqr(pdb_file, pdb_name):
    """
    Run PDB2PQR to generate a PQR file from a PDB structure.

    This function calls the external PDB2PQR tool using the PARSE force field
    and a fixed pH of 7.0. The output file is named based on the provided
    pdb_name.

    Parameters:
        pdb_file (str): Path to the input PDB file.
        pdb_name (str): Base name used to generate the output PQR filename.

    Returns:
        str: Path to the generated PQR file.

    Raises:
        subprocess.CalledProcessError: If PDB2PQR execution fails.
    """
    pqr_file = f"{pdb_name}.pqr"

    subprocess.run(
        ['pdb2pqr', '--ff=PARSE', '--with-ph=7', pdb_file, pqr_file],
        check=True
    )

    return pqr_file

def run_apbs(pdb_file, pdb_name, bbox_min, bbox_max):
    """
    Run APBS electrostatics calculation for a given structure.

    This function generates an APBS input file using a fixed bounding box,
    executes APBS, and writes the output log to disk.

    Parameters:
        pdb_file (str): Path to the input PDB file.
        pdb_name (str): Base name used for output files.
        bbox_min (array-like): Minimum coordinates of the bounding box [x, y, z].
        bbox_max (array-like): Maximum coordinates of the bounding box [x, y, z].

    Returns:
        str: Path to the APBS output log file (.out).

    Raises:
        subprocess.CalledProcessError: If APBS execution fails.
    """
    # Generate APBS input file (.in)
    in_path = generate_apbs_in_fixed(pdb_file, pdb_name, bbox_min, bbox_max)

    # Define output log path
    log_path = os.path.abspath(f"{pdb_name}.out")

    # Execute APBS and capture output
    with open(log_path, "w") as logf:
        subprocess.run(
            ['apbs', in_path],
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=True
        )

    return log_path

def find_dx_file(pdb_name):
    """
    Locate the electrostatic potential DX file generated by APBS.

    APBS/PDB2PQR may generate DX files with different naming conventions.
    This function searches for the most common patterns and returns the first match.

    Parameters:
        pdb_name (str): Base name used for the simulation files.

    Returns:
        str: Absolute path to the DX file.

    Raises:
        FileNotFoundError: If no DX file is found among expected candidates.
    """
    candidates = [
        f"{pdb_name}.pqr-PE0.dx",
        f"{pdb_name}.dx",
        f"{pdb_name}.dx-PE0.dx"
    ]

    dx_path = next(
        (os.path.abspath(f) for f in candidates if os.path.isfile(f)),
        None
    )

    if dx_path is None:
        raise FileNotFoundError(f"DX file not found for {pdb_name}")

    return dx_path

