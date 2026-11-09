"""Relax generated structures with PET-OAM-XL.

Cell parameters and atomic positions are relaxed together with FIRE and a
Frechet cell filter, without symmetry constraints, to a maximum force of
0.01 eV/Angstrom. A structure not converged after 1,000 steps is relaxed for up
to 4,000 further steps with a freshly initialized FIRE optimizer, starting from
its last geometry (5,000 steps in total).

Input: an extxyz file of generated structures with ``gen_index`` in ``info``
(otherwise the frame number is used). Output: an extxyz file whose frames
carry ``gen_index``, ``formula``, ``energy`` (eV), ``energy_per_atom``
(eV/atom), ``fmax`` (eV/Angstrom), ``converged`` and ``n_steps``.

    python generation/relax_pet.py LRO_generated.extxyz LRO_relaxed_PET-OAM-XL.extxyz
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from ase.filters import FrechetCellFilter
from ase.io import read, write
from ase.optimize import FIRE
from upet.calculator import UPETCalculator

FMAX = 0.01          # eV/Angstrom
FIRST_STEPS = 1000
EXTRA_STEPS = 4000
MODEL, MODEL_VERSION = "pet-oam-xl", "1.0.0"


def relax(atoms, calc) -> tuple[bool, int]:
    """Relax in place; return convergence and the total number of steps."""
    atoms.calc = calc
    opt = FIRE(FrechetCellFilter(atoms), logfile=None)
    opt.run(fmax=FMAX, steps=FIRST_STEPS)
    converged, n_steps = bool(opt.converged()), int(opt.nsteps)
    if not converged:
        opt = FIRE(FrechetCellFilter(atoms), logfile=None)
        opt.run(fmax=FMAX, steps=EXTRA_STEPS)
        converged, n_steps = bool(opt.converged()), n_steps + int(opt.nsteps)
    return converged, n_steps


def relax_frame(atoms, index: int, calc):
    """Relax one frame and attach the output metadata."""
    gen_index = int(atoms.info.get("gen_index", index))
    converged, n_steps = relax(atoms, calc)
    energy = float(atoms.get_potential_energy())
    fmax = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
    atoms.info = {"gen_index": gen_index, "formula": atoms.get_chemical_formula(),
                  "energy": energy, "energy_per_atom": energy / len(atoms),
                  "fmax": fmax, "converged": converged, "n_steps": n_steps}
    return atoms


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("input", help="extxyz file of generated structures")
    p.add_argument("output", help="extxyz file of relaxed structures")
    p.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    args = p.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    calc = UPETCalculator(model=MODEL, version=MODEL_VERSION, device=device)
    frames = read(args.input, index=":")
    relaxed = [relax_frame(a, i, calc) for i, a in enumerate(frames)]
    for atoms in relaxed:
        atoms.calc = None
    write(args.output, relaxed, format="extxyz")


if __name__ == "__main__":
    main()
