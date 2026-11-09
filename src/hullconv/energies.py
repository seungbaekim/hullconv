"""MP2020-corrected energies of MLIP-relaxed structures.

An MLIP gives only a structure and a total energy. To apply the MP2020
compatibility correction, the entry is given the calculation parameters that a
VASP run with the pymatgen ``MPRelaxSet`` would have (POTCAR symbols, Hubbard U
values, run type). The ``VasprunLike`` class is adapted from MatterGen
(Microsoft Corporation, MIT License), which follows the ``Vasprun`` class of
pymatgen.
"""

from __future__ import annotations

import re
from functools import cached_property

import numpy as np
from pymatgen.analysis.structure_analyzer import oxide_type
from pymatgen.core import Composition, Structure
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.vasp.sets import MPRelaxSet

# Hubbard U values below this (eV) are treated as zero.
HUBBARD_TOL = 1e-8


class VasprunLike:
    """Stand-in for a VASP run built from a structure and its total energy (eV)."""

    def __init__(self, structure: Structure, energy: float,
                 potcar_functional: str = "PBE") -> None:
        self.structure = structure
        self.energy = energy
        self.potcar_functional = potcar_functional

    @cached_property
    def mp_set(self) -> MPRelaxSet:
        """MPRelaxSet of the structure.

        A fixed KSPACING avoids the automatic k-point generation, which needs
        a symmetry analysis that can fail for distorted structures.
        """
        return MPRelaxSet(self.structure, user_incar_settings={"KSPACING": 0.5},
                          user_kpoints_settings=None)

    @property
    def potcar_symbols(self) -> list[str]:
        """POTCAR symbols, e.g. ``PBE Li_sv``."""
        functional = self.potcar_functional.upper()
        return [f"{functional} {s}" for s in self.mp_set.potcar_symbols]

    @property
    def hubbards(self) -> dict[str, float]:
        """Hubbard U - J (eV) per element, empty for a GGA run."""
        incar = self.mp_set.incar
        if not incar.get("LDAU", False):
            return {}
        symbols = [re.split(r"_", s.split()[1])[0] for s in self.potcar_symbols]
        us = incar.get("LDAUU", [])
        js = incar.get("LDAUJ", [])
        if len(js) != len(us):
            js = [0] * len(us)
        if len(us) == len(symbols):
            return {el: u - j for el, u, j in zip(symbols, us, js)}
        if sum(us) == 0 and sum(js) == 0:
            return {}
        raise ValueError("Hubbard U values do not match the POTCAR symbols.")

    @property
    def is_hubbard(self) -> bool:
        """True for a GGA+U run."""
        return bool(self.hubbards) and sum(self.hubbards.values()) > HUBBARD_TOL

    def computed_entry(self) -> ComputedStructureEntry:
        """Uncorrected entry carrying the parameters the MP2020 scheme reads."""
        params = {"is_hubbard": self.is_hubbard, "hubbards": self.hubbards,
                  "run_type": "GGA+U" if self.is_hubbard else "GGA",
                  "potcar_symbols": self.potcar_symbols}
        data = {"oxide_type": oxide_type(self.structure),
                "aspherical": self.mp_set.incar.get("LASPH", False)}
        return ComputedStructureEntry(self.structure, self.energy,
                                      parameters=params, data=data)


def corrected_entry(structure: Structure, energy: float) -> ComputedStructureEntry:
    """MP2020-corrected entry of a structure with total energy ``energy`` (eV)."""
    entry = VasprunLike(structure, energy).computed_entry()
    out = MaterialsProject2020Compatibility().process_entry(entry)
    if out is None:
        raise ValueError(f"MP2020 correction rejected {structure.composition}.")
    return out


def frame_energy(atoms) -> float:
    """Total energy (eV) of a frame read from extxyz.

    ASE moves an ``energy`` info key into a single-point calculator on reading,
    so both places are checked.
    """
    if "energy" in atoms.info:
        return float(atoms.info["energy"])
    if atoms.calc is not None:
        return float(atoms.get_potential_energy())
    raise KeyError("Frame has no energy.")


def corrected_entries_from_atoms(frames: list) -> list[ComputedStructureEntry]:
    """Corrected entries of relaxed ASE frames with total energies (eV)."""
    from pymatgen.io.ase import AseAtomsAdaptor

    entries = []
    for atoms in frames:
        structure = AseAtomsAdaptor.get_structure(atoms)
        entry = corrected_entry(structure, frame_energy(atoms))
        entry.entry_id = f"gen-{int(atoms.info['gen_index'])}"
        entries.append(entry)
    return entries


def elemental_references(compositions: list[Composition], energies: np.ndarray,
                         formation_energies: np.ndarray) -> dict[str, float]:
    """Solve for the elemental reference energies (eV/atom) of a hull table.

    For each structure, ``E = sum_el n_el * mu_el + N * E_f``. Given corrected
    total energies ``energies`` (eV per cell) and formation energies (eV/atom),
    the references ``mu`` follow by least squares. This places new structures
    on the formation-energy scale of a published ``hull_data`` file without
    the MP database.
    """
    elements = sorted({str(el) for c in compositions for el in c.elements})
    a = np.array([[c[el] for el in elements] for c in compositions], dtype=float)
    n = a.sum(axis=1)
    b = np.asarray(energies, dtype=float) - n * np.asarray(formation_energies)
    mu, *_ = np.linalg.lstsq(a, b, rcond=None)
    return dict(zip(elements, mu))


def formation_energy_per_atom(composition: Composition, energy: float,
                              references: dict[str, float]) -> float:
    """Formation energy (eV/atom) from a total energy (eV) and references."""
    ref = sum(composition[el] * references[str(el)] for el in composition.elements)
    return (energy - ref) / composition.num_atoms
