"""Sample coverage of the generated structures.

A species is a distinct combination of reduced formula and space-group number,
the latter assigned by spglib (through pymatgen) at ``symprec = 0.1`` Angstrom.
Sample coverage is Turing's estimator ``C = 1 - f1 / N``, where ``f1`` is the
number of species seen exactly once among ``N`` structures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

SYMPREC = 0.1          # Angstrom
N_ORDERINGS = 50
STEP = 100
SEED_BASE = 42
UNKNOWN_SPACEGROUP = "Unknown"


def species_label(atoms) -> str:
    """``<reduced formula>_<space-group number>`` of an ASE ``Atoms`` object.

    Structures for which spglib finds no space group are labelled
    ``<reduced formula>_Unknown``.
    """
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    structure = AseAtomsAdaptor.get_structure(atoms)
    formula = structure.composition.reduced_formula
    try:
        number = str(SpacegroupAnalyzer(structure, symprec=SYMPREC)
                     .get_space_group_number())
    except (ValueError, TypeError, RuntimeError):
        number = UNKNOWN_SPACEGROUP
    return f"{formula}_{number}"


def species_labels(frames: list, n_jobs: int = -1) -> list[str]:
    """Species label of every frame, in input order."""
    return Parallel(n_jobs=n_jobs, batch_size=256)(
        delayed(species_label)(a) for a in frames)


def sample_coverage(labels: np.ndarray) -> float:
    """``1 - f1 / N`` of a sample of species labels."""
    labels = np.asarray(labels)
    if len(labels) == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    return 1.0 - np.sum(counts == 1) / len(labels)


def coverage_curve(labels: np.ndarray, seed: int, steps: list[int]) -> np.ndarray:
    """Coverage at each sample size along one random ordering of ``labels``."""
    shuffled = np.random.default_rng(seed).permutation(np.asarray(labels))
    return np.array([sample_coverage(shuffled[:n]) for n in steps])


def coverage_rarefaction(labels: list[str], n_orderings: int = N_ORDERINGS,
                         step: int = STEP, seed_base: int = SEED_BASE) -> pd.DataFrame:
    """Coverage curves of ``n_orderings`` orderings, one row per sample size.

    Ordering ``i`` uses ``default_rng(seed_base + i)``. Returns the mean,
    standard deviation (ddof = 0) and 2.5th/97.5th percentiles.
    """
    n_total = len(labels)
    steps = list(range(step, n_total + 1, step))
    if not steps or steps[-1] != n_total:
        steps.append(n_total)
    curves = np.array([coverage_curve(np.asarray(labels), seed_base + i, steps)
                       for i in range(n_orderings)])
    lo, hi = np.percentile(curves, [2.5, 97.5], axis=0)
    return pd.DataFrame({"n_gen": steps, "coverage_mean": curves.mean(axis=0),
                         "coverage_std": curves.std(axis=0),
                         "coverage_p2.5": lo, "coverage_p97.5": hi})


def all_elements_mask(labels: list[str], n_elements: int) -> np.ndarray:
    """True for species whose formula contains ``n_elements`` elements."""
    from pymatgen.core import Composition

    return np.array([len(Composition(s.rsplit("_", 1)[0]).elements) == n_elements
                     for s in labels])
