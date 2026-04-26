import csv
import tempfile
from collections import defaultdict
from Bio import SeqIO, AlignIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align.Applications import MafftCommandline

def load_manifest(manifest_path):
    """
    Reads a manifest CSV file with columns:
        id, fasta_file_path, label

    Returns a dictionary structured as:

    {
        fasta_file_path_1: [
            {"id": id1, "label": label1},
            {"id": id2, "label": label2}
        ],
        fasta_file_path_2: [
            {"id": id3, "label": label3}
        ]
    }
    """

    manifest_data = defaultdict(list)

    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            manifest_data[row["fasta_file_path"]].append({
                "id": row["id"],
                "label": row["label"]
            })

    return manifest_data


def build_combined_fasta(fasta_ref, manifest_data, output_path):
    """
    Create a combined FASTA file containing the reference sequence and selected user sequences.

    Parameters:
        fasta_ref (str): Path to reference FASTA file.
        manifest_data (dict): Dictionary mapping FASTA files to sequence IDs.
        output_path (str): Path to save the combined FASTA file.
    """
    # Load reference sequence(s)
    records = list(SeqIO.parse(fasta_ref, "fasta"))

    # Iterate through user-provided FASTA files
    for fasta_path, entries in manifest_data.items():
        # Extract only the IDs of interest
        ids = {entry["id"] for entry in entries}

        # Parse sequences and filter by ID
        for record in SeqIO.parse(fasta_path, "fasta"):
            if record.id in ids:
                records.append(record)

    # Write all selected sequences to a single FASTA file
    SeqIO.write(records, output_path, "fasta")


def run_mafft(input_fasta, output_fasta):
    """
    Perform multiple sequence alignment using MAFFT.

    Parameters:
        input_fasta (str): Path to input FASTA file.
        output_fasta (str): Path to save aligned FASTA file.
    """
    # Initialize MAFFT command line
    mafft_cline = MafftCommandline(input=input_fasta)

    # Execute MAFFT and capture output
    stdout, stderr = mafft_cline()

    # Save alignment result to file
    with open(output_fasta, "w") as handle:
        handle.write(stdout)


def load_alignment(aligned_fasta):
    """
    Load a multiple sequence alignment from a FASTA file.

    Parameters:
        aligned_fasta (str): Path to aligned FASTA file.

    Returns:
        MultipleSeqAlignment: Biopython alignment object.
    """
    return AlignIO.read(aligned_fasta, "fasta")


def get_reference_id(fasta_ref):
    """
    Retrieve the ID of the reference sequence.

    Parameters:
        fasta_ref (str): Path to reference FASTA file.

    Returns:
        str: Reference sequence ID.
    """
    return next(SeqIO.parse(fasta_ref, "fasta")).id


def extract_aligned_reference(alignment, ref_id):
    """
    Extract the aligned reference sequence from the alignment.

    Parameters:
        alignment (MultipleSeqAlignment): Alignment object.
        ref_id (str): Reference sequence ID.

    Returns:
        str: Aligned reference sequence as a string.

    Raises:
        ValueError: If the reference sequence is not found.
    """
    # Search for the reference sequence in the alignment
    for record in alignment:
        if record.id == ref_id:
            return str(record.seq)

    # Raise error if not found
    raise ValueError("Reference sequence not found in alignment!")


def get_trim_positions(aligned_ref_seq):
    """
    Determine the start and end positions of the reference sequence excluding gaps.

    Parameters:
        aligned_ref_seq (str): Aligned reference sequence.

    Returns:
        tuple: (start, end) positions defining the ungapped region.
    """
    # Identify positions that are not gaps
    non_gap_positions = [i for i, aa in enumerate(aligned_ref_seq) if aa != "-"]

    # First and last valid positions define trimming boundaries
    start = non_gap_positions[0]
    end = non_gap_positions[-1] + 1

    return start, end


def trim_alignment(alignment, ref_id, start, end):
    """
    Trim all sequences in the alignment based on reference boundaries.

    Parameters:
        alignment (MultipleSeqAlignment): Alignment object.
        ref_id (str): Reference sequence ID.
        start (int): Start position for trimming.
        end (int): End position for trimming.

    Returns:
        list: List of trimmed SeqRecord objects.
    """
    trimmed = []

    # Iterate through all sequences in the alignment
    for record in alignment:
        # Skip the reference sequence
        if record.id == ref_id:
            continue

        # Extract the segment corresponding to the reference region
        segment = str(record.seq[start:end])

        # Create a new SeqRecord preserving metadata
        trimmed.append(
            SeqRecord(Seq(segment), id=record.id, description=record.description)
        )

    return trimmed


def save_fasta(records, output_path):
    """
    Save a list of sequences to a FASTA file.

    Parameters:
        records (list): List of SeqRecord objects.
        output_path (str): Path to output FASTA file.
    """
    SeqIO.write(records, output_path, "fasta")

def clear_ambiguous_amino_acids(manifest_csv, fasta_input, fasta_output):

    """
    Cleans protein sequences in a FASTA file by removing ambiguous amino acids (B, J, O, U, X, Z) from sequences whose IDs are listed in a manifest CSV file.

    Parameters:
        manifest_csv (str): Path to the CSV file containing sequence IDs (column name: 'id').
        fasta_input (str): Path to the input FASTA file.
        fasta_output (str): Path where the cleaned FASTA file will be saved.

    """

    invalid_amino_acids = {'B', 'J', 'O', 'U', 'X', 'Z'}

    # Read IDs from the manifest
    manifest_ids = set()
    with open(manifest_csv, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            manifest_ids.add(row['id'])

    cleaned_records = []

    for record in SeqIO.parse(fasta_input, "fasta"):
        seq_id = record.id
        sequence_str = str(record.seq).upper()

        if seq_id in manifest_ids:
            cleaned_sequence = "".join(
                aa for aa in sequence_str if aa not in invalid_amino_acids
            )

            if len(cleaned_sequence) != len(sequence_str):
                removed = len(sequence_str) - len(cleaned_sequence)

            record.seq = Seq(cleaned_sequence)

        cleaned_records.append(record)

    SeqIO.write(cleaned_records, fasta_output, "fasta")