"""Public package API for emsaplibraries."""

import warnings

from .indicators import (
    CustomAtom,
    atoms_outside_grid_coords,
    calculate_p_sasa,
    calculate_q_sasa,
    calculate_residue_exposed_charge,
    calculate_sasa_from_pqr,
    calculate_see,
    calculate_surface_potential_fraction,
    extract_epi,
    interpolate_potential,
    parse_dx,
    parse_pqr,
    parse_propka_pka,
)
from .pipeline import ProteinPipelineResult


def process_single_protein(*args, **kwargs):
    """Deprecated compatibility wrapper for the pipeline helper."""
    warnings.warn(
        "emsaplibraries.process_single_protein is deprecated; import "
        "process_single_protein from emsaplibraries.pipeline instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from .pipeline import process_single_protein as _process_single_protein

    return _process_single_protein(*args, **kwargs)

__all__ = [
    "CustomAtom",
    "ProteinPipelineResult",
    "atoms_outside_grid_coords",
    "calculate_p_sasa",
    "calculate_q_sasa",
    "calculate_residue_exposed_charge",
    "calculate_sasa_from_pqr",
    "calculate_see",
    "calculate_surface_potential_fraction",
    "extract_epi",
    "interpolate_potential",
    "parse_dx",
    "parse_pqr",
    "parse_propka_pka",
    "process_single_protein",
]

__version__ = "0.1.0"
