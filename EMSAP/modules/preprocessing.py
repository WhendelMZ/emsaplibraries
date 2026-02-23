import csv
from Bio import SeqIO
from Bio.Seq import Seq

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