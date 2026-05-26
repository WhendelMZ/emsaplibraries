# emsaplibraries

`emsaplibraries` is an import-first Python package for protein sequence preprocessing, structure handling, electrostatic simulation helpers, and protein indicator calculations.

The main public API is the indicator module. Common indicator functions are also exported from the package root:

```python
from emsaplibraries import calculate_q_sasa, calculate_p_sasa, calculate_see
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

## External Tool Requirements

Some public functions call external command-line tools. The package imports without these tools, and checks for them only when tool-dependent functions are called.

- `run_mafft` requires `mafft`
- `run_pdb2pqr` and `process_single_protein` require `pdb2pqr`
- `run_apbs` and `process_single_protein` require `apbs`

## Indicator Examples

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

Input expectations:

- PQR files must contain ATOM/HETATM rows with coordinates, charge, and radius in the final columns.
- DX files must contain APBS/OpenDX-style grid metadata and potential values.
- PDB files are used for atom and residue metadata when an indicator needs residue-level or PDB-coordinate mapping.

## Public Modules

```python
import emsaplibraries.indicators
import emsaplibraries.electrostatics
import emsaplibraries.preprocessing
import emsaplibraries.structure
```

The v1 API keeps the current tuple and dictionary return shapes for indicator functions.
