"""Sequence manifest, FASTA, and MAFFT preprocessing helpers."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from ._runtime import require_executable, require_module


def _bio_modules():
    seq_io = require_module("Bio.SeqIO", "pip install biopython")
    align_io = require_module("Bio.AlignIO", "pip install biopython")
    seq = require_module("Bio.Seq", "pip install biopython")
    seq_record = require_module("Bio.SeqRecord", "pip install biopython")
    return seq_io, align_io, seq.Seq, seq_record.SeqRecord


def load_manifest(manifest_path: str | Path):
    """Read a manifest CSV with columns ``id``, ``fasta_file_path``, and ``label``."""
    manifest_data = defaultdict(list)
    with open(manifest_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            manifest_data[row["fasta_file_path"]].append(
                {"id": row["id"], "label": row["label"]}
            )
    return manifest_data


def build_combined_fasta(
    fasta_ref: str | Path, manifest_data, output_path: str | Path
) -> None:
    """Create a combined FASTA containing the reference and selected sequences."""
    seq_io, _, _, _ = _bio_modules()
    records = list(seq_io.parse(str(fasta_ref), "fasta"))
    for fasta_path, entries in manifest_data.items():
        ids = {entry["id"] for entry in entries}
        for record in seq_io.parse(str(fasta_path), "fasta"):
            if record.id in ids:
                records.append(record)
    seq_io.write(records, str(output_path), "fasta")


def run_mafft(input_fasta: str | Path, output_fasta: str | Path) -> None:
    """Perform multiple sequence alignment using the MAFFT executable."""
    mafft = require_executable("mafft", "Install MAFFT and ensure 'mafft' is on PATH.")
    import subprocess

    result = subprocess.run(
        [mafft, str(input_fasta)],
        check=True,
        capture_output=True,
        text=True,
    )
    Path(output_fasta).write_text(result.stdout, encoding="utf-8")


def load_alignment(aligned_fasta: str | Path):
    """Load a FASTA multiple sequence alignment."""
    _, align_io, _, _ = _bio_modules()
    return align_io.read(str(aligned_fasta), "fasta")


def get_reference_id(fasta_ref: str | Path) -> str:
    """Return the ID of the first sequence in a reference FASTA."""
    seq_io, _, _, _ = _bio_modules()
    return next(seq_io.parse(str(fasta_ref), "fasta")).id


def extract_aligned_reference(alignment, ref_id: str) -> str:
    """Extract the aligned reference sequence from an alignment."""
    for record in alignment:
        if record.id == ref_id:
            return str(record.seq)
    raise ValueError("Reference sequence not found in alignment.")


def get_trim_positions(aligned_ref_seq: str) -> tuple[int, int]:
    """Return start/end positions of the ungapped reference region."""
    non_gap_positions = [index for index, aa in enumerate(aligned_ref_seq) if aa != "-"]
    if not non_gap_positions:
        raise ValueError("Aligned reference sequence contains only gaps.")
    return non_gap_positions[0], non_gap_positions[-1] + 1


def trim_alignment(alignment, ref_id: str, start: int, end: int):
    """Trim non-reference sequences using reference alignment boundaries."""
    _, _, seq_cls, seq_record_cls = _bio_modules()
    trimmed = []
    for record in alignment:
        if record.id == ref_id:
            continue
        segment = str(record.seq[start:end])
        trimmed.append(
            seq_record_cls(
                seq_cls(segment), id=record.id, description=record.description
            )
        )
    return trimmed


def save_fasta(records, output_path: str | Path) -> None:
    """Save SeqRecord objects to a FASTA file."""
    seq_io, _, _, _ = _bio_modules()
    seq_io.write(records, str(output_path), "fasta")


def clear_ambiguous_amino_acids(
    manifest_csv: str | Path,
    fasta_input: str | Path,
    fasta_output: str | Path,
) -> None:
    """Remove ambiguous amino acids from manifest-listed protein sequences."""
    seq_io, _, seq_cls, _ = _bio_modules()
    invalid_amino_acids = {"B", "J", "O", "U", "X", "Z"}
    manifest_ids = set()

    with open(manifest_csv, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            manifest_ids.add(row["id"])

    cleaned_records = []
    for record in seq_io.parse(str(fasta_input), "fasta"):
        if record.id in manifest_ids:
            sequence_str = str(record.seq).upper()
            record.seq = seq_cls(
                "".join(aa for aa in sequence_str if aa not in invalid_amino_acids)
            )
        cleaned_records.append(record)

    seq_io.write(cleaned_records, str(fasta_output), "fasta")
