"""Convex hulls, stable candidates and hull-shift metrics.

Every structure enters the hull as a ``PDEntry`` whose energy is its
MP2020-corrected formation energy per atom times its number of atoms. Formation
and total energies differ by a linear function of composition, so a hull built
from formation energies gives the same energy above the hull (Ehull) as one
built from total energies.

Two hulls are used. The MP-only hull is built from the MP structures alone;
the updated hull from the MP structures and the generated structures added so
far. A stable candidate is a generated structure on the updated hull, counted
once per reduced formula. The hull shift of MP structure i is
``Ehull_i(updated) - Ehull_i(MP-only)``; TCS is the sum of its absolute value
over all MP structures and MCS the mean (TCS / N_MP). Energies are in eV/atom
unless the name ends in ``_meV``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition

from hullconv.data import EF, FORMULA, GEN_INDEX, N_ATOMS, HullData

# Numerical tolerance for "on the hull" (eV/atom).
STABLE_TOL = 1e-6


def make_entries(df: pd.DataFrame) -> list[PDEntry]:
    """Build ``PDEntry`` objects from formula, atom count and formation energy.

    The formula is scaled to ``n_atoms`` so that reduced and full formulas give
    the same entry.
    """
    entries = []
    for formula, n_atoms, ef in zip(df[FORMULA], df[N_ATOMS], df[EF]):
        comp = Composition(formula)
        comp = comp * (float(n_atoms) / comp.num_atoms)
        entries.append(PDEntry(comp, float(ef) * float(n_atoms)))
    return entries


def reduced_formulas(df: pd.DataFrame) -> np.ndarray:
    """Reduced formula of every row."""
    return np.array([Composition(f).reduced_formula for f in df[FORMULA]],
                    dtype=object)


def e_above_hull(pdg: PhaseDiagram, entries: list[PDEntry],
                 allow_negative: bool = True) -> np.ndarray:
    """Ehull of each entry (eV/atom); negative below the hull if allowed."""
    return np.array([pdg.get_e_above_hull(e, allow_negative=allow_negative)
                     for e in entries])


def hull_vertex_formulas(pdg: PhaseDiagram, entries: list[PDEntry],
                         formulas: np.ndarray, tol: float = STABLE_TOL) -> set[str]:
    """Reduced formulas of the entries that lie on the hull of ``pdg``.

    An entry within ``tol`` of the hull must sit at the composition of a hull
    vertex, so only those compositions are compared.
    """
    vertex = {v.composition.reduced_formula: v.energy_per_atom
              for v in pdg.stable_entries}
    return {f for e, f in zip(entries, formulas)
            if f in vertex and e.energy_per_atom <= vertex[f] + tol}


def lowest_per_formula(indices: np.ndarray, formulas: np.ndarray,
                       e_per_atom: np.ndarray) -> np.ndarray:
    """Keep, for each reduced formula, the index with the lowest energy.

    Only the lowest-energy structure of a composition can lie on or move the
    hull, so this reduction leaves the hull and the candidate set unchanged.
    """
    best: dict[str, int] = {}
    for j in indices:
        f = formulas[j]
        if f not in best or e_per_atom[j] < e_per_atom[best[f]]:
            best[f] = int(j)
    return np.array(sorted(best.values()), dtype=int)


@dataclass
class HullModel:
    """MP entries, generated entries and the MP-only hull of one system.

    ``relevant`` holds the generated structures on or below the MP-only hull.
    Adding structures only lowers the hull, so no other generated structure
    can become a stable candidate or change the Ehull of an MP structure.
    Formulas listed in ``exclude`` are removed from ``relevant``.
    """

    data: HullData
    exclude: tuple[str, ...] = ()
    vertex_tol: float = STABLE_TOL
    mp_entries: list[PDEntry] = field(init=False)
    gen_entries: list[PDEntry] = field(init=False)
    formulas: np.ndarray = field(init=False)
    e_per_atom: np.ndarray = field(init=False)
    base_mp_ehull: np.ndarray = field(init=False)
    gen_ehull_mp_only: np.ndarray = field(init=False)
    relevant: np.ndarray = field(init=False)
    _cache: dict = field(init=False, default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.mp_entries = make_entries(self.data.mp)
        self.gen_entries = make_entries(self.data.gen)
        self.formulas = reduced_formulas(self.data.gen)
        self.e_per_atom = np.array([e.energy_per_atom for e in self.gen_entries])
        base = PhaseDiagram(self.mp_entries)
        self.base_mp_ehull = e_above_hull(base, self.mp_entries, allow_negative=False)
        self.gen_ehull_mp_only = e_above_hull(base, self.gen_entries)
        keep = self.gen_ehull_mp_only <= STABLE_TOL
        if self.exclude:
            keep &= ~np.isin(self.formulas, list(self.exclude))
        self.relevant = np.flatnonzero(keep)

    @property
    def n_mp(self) -> int:
        """Number of MP structures."""
        return len(self.mp_entries)

    def state(self, present: np.ndarray) -> tuple[int, float, frozenset[str]]:
        """Hull of MP structures plus the generated structures ``present``.

        Returns the number of stable candidates, TCS (eV/atom) and the
        candidate formulas. ``present`` may hold any generated indices; only
        the relevant ones are used.
        """
        present = np.intersect1d(np.asarray(present, dtype=int), self.relevant)
        key = tuple(lowest_per_formula(present, self.formulas, self.e_per_atom))
        if key in self._cache:
            return self._cache[key]
        if not key:
            value = (0, 0.0, frozenset())
        else:
            cur = [self.gen_entries[j] for j in key]
            pdg = PhaseDiagram(self.mp_entries + cur)
            mp_ehull = e_above_hull(pdg, self.mp_entries, allow_negative=False)
            tcs = float(np.sum(np.abs(mp_ehull - self.base_mp_ehull)))
            stable = hull_vertex_formulas(pdg, cur, self.formulas[list(key)],
                                          self.vertex_tol)
            value = (len(stable), tcs, frozenset(stable))
        self._cache[key] = value
        return value


def endpoint_metrics(model: HullModel, n_gen: int | None = None) -> dict:
    """Stable candidates, TCS and MCS after the first ``n_gen`` structures.

    ``n_gen`` defaults to all generated structures. TCS and MCS are returned
    in meV/atom.
    """
    n = len(model.gen_entries) if n_gen is None else n_gen
    n_cand, tcs, stable = model.state(np.arange(n))
    return {"system": model.data.tag, "n_mp": model.n_mp, "n_gen": n,
            "stable_candidates": n_cand, "TCS_meV": 1000.0 * tcs,
            "MCS_meV": 1000.0 * tcs / model.n_mp,
            "candidates": ";".join(sorted(stable))}


def updated_hull_ehull(model: HullModel) -> np.ndarray:
    """Ehull of every generated structure against the updated hull (eV/atom).

    The updated hull here contains all generated structures.
    """
    present = lowest_per_formula(model.relevant, model.formulas, model.e_per_atom)
    pdg = PhaseDiagram(model.mp_entries + [model.gen_entries[j] for j in present])
    return e_above_hull(pdg, model.gen_entries, allow_negative=False)


def candidate_table(model: HullModel) -> pd.DataFrame:
    """One row per stable candidate on the hull of all generated structures.

    Gives the lowest-energy structure of the formula, its Ehull against the
    MP-only hull (eV/atom), the number of generated copies of the formula and
    whether the composition is absent from the MP.
    """
    _, _, stable = model.state(np.arange(len(model.gen_entries)))
    mp_formulas = set(reduced_formulas(model.data.mp))
    rows = []
    for f in sorted(stable):
        idx = np.flatnonzero(model.formulas == f)
        best = idx[np.argmin(model.e_per_atom[idx])]
        rows.append({"formula": f,
                     "gen_index": int(model.data.gen[GEN_INDEX].iloc[best]),
                     "ehull_mp_only_meV": 1000.0 * model.gen_ehull_mp_only[best],
                     "copies": len(idx),
                     "novel_composition": f not in mp_formulas})
    return pd.DataFrame(rows)
