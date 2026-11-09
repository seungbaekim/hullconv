"""Materials Project entries of a chemical system.

The published ``hull_data`` files already contain the MP formation energies
used in the paper (database release v2025.09.25), so this module is needed
only to build the table for another system or database release. The API key
is read from the environment variable ``MP_API_KEY``.
"""

from __future__ import annotations

import os

THERMO_TYPES = ["GGA_GGA+U"]
API_KEY_ENV = "MP_API_KEY"


def api_key() -> str:
    """Return the MP API key from ``MP_API_KEY``."""
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise RuntimeError(f"Set the environment variable {API_KEY_ENV}.")
    return key


def fetch_entries(elements: list[str]) -> list:
    """MP2020-corrected GGA/GGA+U entries of the chemical system ``elements``.

    Entries are requested uncorrected and corrected here once, so that the
    correction is the same as for the generated structures.
    """
    from mp_api.client import MPRester
    from pymatgen.entries.compatibility import MaterialsProject2020Compatibility

    with MPRester(api_key()) as mpr:
        entries = mpr.get_entries_in_chemsys(
            elements=elements,
            additional_criteria={"thermo_types": THERMO_TYPES},
            compatible_only=False)
    return MaterialsProject2020Compatibility().process_entries(entries)
