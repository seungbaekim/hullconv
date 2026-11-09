"""Locate and read the public data files.

The data directory is given on the command line or by the environment variable
``HULLCONV_DATA`` (default ``./data``). Files are found by name anywhere below
it, so the layout of the downloaded archive does not matter.

``<TAG>_hull_data.csv`` holds one row per structure:

=====================================  =========================================
column                                 meaning
=====================================  =========================================
source                                 ``MP`` or ``generated``
mp_id                                  MP identifier (empty for generated rows)
gen_index                              generation index (empty for MP rows)
formula                                chemical formula of the cell
n_atoms                                number of atoms in the cell
formation_energy_eV_per_atom           MP2020-corrected formation energy
e_above_hull_mp_only_eV_per_atom       Ehull against the MP-only hull
e_above_hull_updated_eV_per_atom       Ehull against the updated hull
=====================================  =========================================
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATA_ENV = "HULLCONV_DATA"
DEFAULT_DATA_DIR = "data"

SOURCE = "source"
MP_ID = "mp_id"
GEN_INDEX = "gen_index"
FORMULA = "formula"
N_ATOMS = "n_atoms"
EF = "formation_energy_eV_per_atom"
EHULL_MP_ONLY = "e_above_hull_mp_only_eV_per_atom"
EHULL_UPDATED = "e_above_hull_updated_eV_per_atom"
REQUIRED_COLUMNS = (SOURCE, MP_ID, GEN_INDEX, FORMULA, N_ATOMS, EF)

SOURCE_MP = "MP"
SOURCE_GENERATED = "generated"

TERNARY_SYSTEMS = ("LMO", "LCO", "LNO", "LZO", "LHO", "LRO")
QUATERNARY_SYSTEMS = ("LNMO", "LNRO")


@dataclass
class HullData:
    """MP and generated rows of one ``<TAG>_hull_data.csv`` file.

    ``gen`` is sorted by ``gen_index``; its row order is the generation order
    used by every ordering-dependent calculation.
    """

    tag: str
    mp: pd.DataFrame
    gen: pd.DataFrame

    @property
    def elements(self) -> list[str]:
        """Elements of the chemical system, from the MP formulas."""
        from pymatgen.core import Composition

        els = set()
        for f in self.mp[FORMULA]:
            els.update(str(e) for e in Composition(f).elements)
        return sorted(els)


def data_dir(path: str | None = None) -> Path:
    """Return the data directory from ``path``, ``HULLCONV_DATA`` or ``./data``."""
    root = Path(path or os.environ.get(DATA_ENV, DEFAULT_DATA_DIR))
    if not root.is_dir():
        raise FileNotFoundError(
            f"Data directory '{root}' not found; pass --data or set {DATA_ENV}."
        )
    return root


def find_file(root: Path, name: str) -> Path:
    """Return the single file called ``name`` below ``root``."""
    hits = sorted(p for p in root.rglob(name) if p.is_file())
    if not hits:
        raise FileNotFoundError(f"'{name}' not found below '{root}'.")
    if len(hits) > 1:
        listed = ", ".join(str(h) for h in hits)
        raise FileExistsError(f"'{name}' is ambiguous below '{root}': {listed}")
    return hits[0]


def load_hull_data(tag: str, root: Path) -> HullData:
    """Read ``<tag>_hull_data.csv`` and split it into MP and generated rows."""
    path = find_file(root, f"{tag}_hull_data.csv")
    df = pd.read_csv(path, dtype={MP_ID: str}, float_precision="round_trip")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    if df[EF].isna().any():
        raise ValueError(f"{path}: formation energy is missing in some rows")
    unknown = set(df[SOURCE]) - {SOURCE_MP, SOURCE_GENERATED}
    if unknown:
        raise ValueError(f"{path}: unknown source values {sorted(unknown)}")
    mp = df[df[SOURCE] == SOURCE_MP].reset_index(drop=True)
    gen = df[df[SOURCE] == SOURCE_GENERATED].copy()
    gen[GEN_INDEX] = gen[GEN_INDEX].astype(int)
    gen = gen.sort_values(GEN_INDEX, kind="stable").reset_index(drop=True)
    return HullData(tag=tag, mp=mp, gen=gen)


def read_structures(tag: str, root: Path, kind: str = "relaxed") -> list:
    """Read ``<tag>_relaxed_PET-OAM-XL.extxyz`` or ``<tag>_generated.extxyz``.

    Returns ASE ``Atoms`` objects sorted by their ``gen_index`` info key.
    """
    from ase.io import read

    names = {"relaxed": f"{tag}_relaxed_PET-OAM-XL.extxyz",
             "generated": f"{tag}_generated.extxyz"}
    if kind not in names:
        raise ValueError(f"kind must be one of {sorted(names)}")
    frames = read(str(find_file(root, names[kind])), index=":")
    return sorted(frames, key=lambda a: int(a.info[GEN_INDEX]))


def output_dir(path: str) -> Path:
    """Create and return the output directory."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out
