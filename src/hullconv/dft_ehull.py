"""DFT energy above the hull against competing phases calculated alike.

Each candidate and each competing phase is a VASP calculation (``vasprun.xml``)
with the same settings. Energies are MP2020-corrected. The hull energy at the
candidate composition is the lowest energy of any combination of the competing
phases with that composition, found by linear programming over atom fractions;
the candidate's Ehull is its energy minus this hull energy (negative below the
hull). The phases with non-zero fractions are its decomposition products.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pymatgen.core import Composition

# Atom fractions below this are dropped from the decomposition.
FRACTION_TOL = 1e-6


@dataclass
class Phase:
    """A relaxed calculation: name, composition and corrected energy (eV/atom)."""

    name: str
    composition: Composition
    energy_per_atom: float
    uncorrected_energy_per_atom: float


def read_phase(folder: Path, name: str | None = None) -> Phase:
    """Read ``vasprun.xml`` in ``folder`` and apply the MP2020 correction."""
    from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
    from pymatgen.io.vasp.outputs import Vasprun

    path = Path(folder) / "vasprun.xml"
    vr = Vasprun(str(path), parse_dos=False, parse_eigen=False,
                 parse_potcar_file=False)
    if not vr.converged_ionic:
        raise RuntimeError(f"Ionic relaxation not converged: {path}")
    entry = vr.get_computed_entry(inc_structure=True)
    raw = entry.energy_per_atom
    out = MaterialsProject2020Compatibility().process_entries([entry], clean=True)
    if not out:
        raise RuntimeError(f"MP2020 correction rejected {path}")
    return Phase(name or Path(folder).name, entry.composition,
                 out[0].energy_per_atom, raw)


def hull_energy(composition: Composition,
                phases: list[Phase]) -> tuple[float, dict[str, float]]:
    """Lowest energy (eV/atom) of a mixture of ``phases`` at ``composition``.

    Returns the energy and the atom fraction of each phase in the mixture.
    """
    from scipy.optimize import linprog

    elements = sorted({str(el) for p in phases for el in p.composition.elements}
                      | {str(el) for el in composition.elements})
    a_eq = np.array([[p.composition.get_atomic_fraction(el) for p in phases]
                     for el in elements])
    b_eq = np.array([composition.get_atomic_fraction(el) for el in elements])
    cost = np.array([p.energy_per_atom for p in phases])
    res = linprog(cost, A_eq=a_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    if not res.success:
        raise ValueError(f"{composition.reduced_formula} cannot be formed from "
                         f"{[p.name for p in phases]}")
    decomposition = {p.name: float(x) for p, x in zip(phases, res.x)
                     if x > FRACTION_TOL}
    return float(res.fun), decomposition


def e_above_hull(candidate: Phase,
                 competing: list[Phase]) -> tuple[float, dict[str, float]]:
    """Signed Ehull (eV/atom) of ``candidate`` and its decomposition products."""
    e_ref, decomposition = hull_energy(candidate.composition, competing)
    return candidate.energy_per_atom - e_ref, decomposition


def format_decomposition(decomposition: dict[str, float],
                         phases: list[Phase]) -> str:
    """'0.667 Li2HfO3 + 0.333 HfO2' style text with reduced formulas."""
    formula = {p.name: p.composition.reduced_formula for p in phases}
    parts = sorted(decomposition.items(), key=lambda kv: -kv[1])
    return " + ".join(f"{x:.3f} {formula[name]}" for name, x in parts)
