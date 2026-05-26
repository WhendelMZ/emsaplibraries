# emsaplibraries

`emsaplibraries` is an import-first Python package for protein sequence preprocessing, structure handling, electrostatic simulation helpers, and protein indicator calculations.

The main public API is artifact-based: core functions calculate metrics from existing PDB, PQR, DX, and optional PKA files rather than owning a full workflow engine. The preferred entry point for scripts and Nextflow tasks is exported from the package root:

```python
from emsaplibraries import calculate_protein_metrics

metrics = calculate_protein_metrics(
    pdb_file="protein.pdb",
    pqr_file="protein.pqr",
    dx_file="protein.dx",
    pka_file="protein.pka",
)

print(metrics.as_dict())
```

## Installation

For local development:

```bash
pip install -e ".[dev]"
```

For a GitHub install:

```bash
pip install "git+https://github.com/WhendelMZ/emsaplibraries.git"
```

## Python Dependencies

The package metadata installs the normal Python dependencies needed by the public modules:

- `numpy`
- `biopython`
- `freesasa`
- `pyrosetta-installer`

PyRosetta itself has licensing and distribution requirements. Install it according to the official PyRosetta instructions for your environment. If `relax_pdbs` is called without PyRosetta importable, the function raises an actionable `ImportError`.

## External Tool Wrappers

Optional convenience wrappers for command-line tools live in `emsaplibraries.external`. The package imports without these tools, and checks for them only when wrapper or pipeline functions are called.

- `external.run_mafft` requires `mafft`
- `external.run_pdb2pqr` and `pipeline.process_single_protein` require `pdb2pqr`
- `external.run_apbs` and `pipeline.process_single_protein` require `apbs`
- `pipeline.process_single_protein` requires `propka3`

```python
from emsaplibraries.external import run_apbs, run_mafft, run_pdb2pqr
```

The package root does not expose workflow orchestration. Import pipeline helpers explicitly from `emsaplibraries.pipeline`.

## Indicator Helpers

```python
from emsaplibraries import (
    calculate_p_sasa,
    calculate_q_sasa,
    calculate_residue_exposed_charge,
    calculate_see,
)

p_sasa, p_num, p_den = calculate_p_sasa("protein.pqr", "protein.pdb", "protein.dx")
q_sasa, q_num, total_sasa = calculate_q_sasa("protein.pqr")
see = calculate_see("protein.pqr", "protein.dx")
charge = calculate_residue_exposed_charge("protein.pqr", "protein.pdb")
```

These lower-level helpers are useful when a script only needs one metric or needs supporting values such as numerator and denominator terms.

Input expectations:

- PQR files must contain ATOM/HETATM rows with coordinates, charge, and radius in the final columns.
- DX files must contain APBS/OpenDX-style grid metadata and potential values.
- PDB files are used for atom and residue metadata when an indicator needs residue-level or PDB-coordinate mapping.
- pKa and ABSE functions require a precomputed PROPKA `.pka` file.

## Pipeline Helpers

Whole-protein orchestration helpers live in the pipeline module:

```python
from emsaplibraries.pipeline import process_single_protein

result = process_single_protein(
    "protein.pdb",
    "auxiliary-output",
    bbox_min=[0, 0, 0],
    bbox_max=[100, 100, 100],
)

print(result.p_sasa)
print(result.as_legacy_tuple())
```

`process_single_protein` still runs PDB2PQR, APBS, and PROPKA internally for convenience, calculates the available metrics, moves intermediate artifacts into the auxiliary output directory, and returns a typed result object.

## Public Modules

```python
import emsaplibraries.indicators
import emsaplibraries.pipeline
import emsaplibraries.external
import emsaplibraries.electrostatics
import emsaplibraries.preprocessing
import emsaplibraries.structure
```

`emsaplibraries.electrostatics` contains APBS input and DX artifact helpers; command execution wrappers live in `emsaplibraries.external`.

The v1 API keeps the current tuple and dictionary return shapes for indicator functions.
