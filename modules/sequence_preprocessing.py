import csv
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

def align_sequences( 
    fasta_ref_protein,       # protein reference
    manifest_data,           # user-provided sequences 
    trim_seq                 # output file with trimmed sequences
):
    """
    This function detects whether the input sequences are DNA or protein,
    selects the appropriate protein reference, aligns all sequences using MAFFT, identifies the ungapped region of the
    reference in the alignment, and trims all user sequences to match that
    region while preserving gaps.
    """
    if isinstance(manifest_data, str):
     manifest_data = load_manifest(manifest_data)

    # Select the appropriate reference based on sequence type
    fasta_ref = fasta_ref_protein        # use protein reference

    # Create combined FASTA for alignment
    import tempfile

    to_align = tempfile.NamedTemporaryFile(delete=False).name
    aligned = tempfile.NamedTemporaryFile(delete=False).name

    # join reference + user sequences into a single FASTA
    records = list(SeqIO.parse(fasta_ref, "fasta"))

    for fasta_path, entries in manifest_data.items():
        ids = {entry["id"] for entry in entries}  # extrai só os IDs
        for record in SeqIO.parse(fasta_path, "fasta"):
            if record.id in ids:
                records.append(record)

    SeqIO.write(records, to_align, "fasta")  # write combined FASTA

    # Run MAFFT to generate alignment
    mafft_cline = MafftCommandline(input=to_align)  # default MAFFT command
    stdout, stderr = mafft_cline()                  # execute MAFFT and capture output

    with open(aligned, "w") as handle:
        handle.write(stdout)                        # save alignment to disk

    alignment = AlignIO.read(aligned, "fasta")      # load alignment into memory

    # Find the reference sequence inside the alignment
    ref_id = next(SeqIO.parse(fasta_ref, "fasta")).id   # ID of the original reference

    aligned_ref_seq = None
    for reg in alignment:                               # traverse aligned sequences
        if reg.id == ref_id:                            # identify reference inside alignment
            aligned_ref_seq = str(reg.seq)
            break

    if aligned_ref_seq is None:
        raise ValueError("Reference sequence not found in alignment!")  # if missing, raise error

    # Determine beginning and end of the reference (ignoring gaps)
    non_gap_positions = [i for i, aa in enumerate(aligned_ref_seq) if aa != "-"]
    start_pos = non_gap_positions[0]
    end_pos = non_gap_positions[-1] + 1

    # positions in the aligned protein
    start_pos_protein = start_pos
    end_pos_protein = end_pos

    # Trim each sequence based on the reference interval
    trimmed_seqs = []
    for reg in alignment:
        if reg.id == ref_id:                       # skip reference in output
            continue

        seq_segment = str(reg.seq[start_pos:end_pos])  # cut segment while preserving gaps
        new_record = SeqRecord(Seq(seq_segment), id=reg.id, description=reg.description)
        trimmed_seqs.append(new_record)                # add trimmed sequence

    SeqIO.write(trimmed_seqs, trim_seq, "fasta")       # save trimmed sequences

    return start_pos_protein, end_pos_protein          # return informative positions

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