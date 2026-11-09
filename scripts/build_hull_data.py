"""Build ``<TAG>_hull_data.csv`` from relaxed structures and MP entries.

PET-OAM-XL total energies are MP2020-corrected (``hullconv.energies``), the
MP GGA/GGA+U entries of the system are downloaded and corrected the same way
(``hullconv.mp_data``, needs ``MP_API_KEY``), and formation energies are taken
on the phase diagram of all entries. The MP data returned by the API are those
of its current database release; the published files were built from release
v2025.09.25.

    python scripts/build_hull_data.py --tag LRO --elements Li Ru O \
        --relaxed LRO_relaxed_PET-OAM-XL.extxyz --out results
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from ase.io import read
from pymatgen.analysis.phase_diagram import PhaseDiagram

from hullconv import data as d
from hullconv.energies import corrected_entries_from_atoms
from hullconv.mp_data import fetch_entries


def rows(entries: list, pd_all: PhaseDiagram, pd_mp: PhaseDiagram,
         source: str) -> pd.DataFrame:
    """Hull-table rows of MP or generated entries."""
    out = []
    for e in entries:
        is_mp = source == d.SOURCE_MP
        out.append({
            d.SOURCE: source,
            d.MP_ID: str(e.entry_id) if is_mp else "",
            d.GEN_INDEX: "" if is_mp else str(e.entry_id).split("-", 1)[1],
            d.FORMULA: e.composition.formula.replace(" ", ""),
            d.N_ATOMS: int(e.composition.num_atoms),
            d.EF: pd_all.get_form_energy_per_atom(e),
            d.EHULL_MP_ONLY: pd_mp.get_e_above_hull(e, allow_negative=True),
            d.EHULL_UPDATED: np.nan if is_mp else pd_all.get_e_above_hull(e),
        })
    return pd.DataFrame(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tag", required=True)
    p.add_argument("--elements", nargs="+", required=True)
    p.add_argument("--relaxed", required=True, help="relaxed extxyz file")
    p.add_argument("--out", default="results")
    args = p.parse_args()
    frames = sorted(read(args.relaxed, index=":"),
                    key=lambda a: int(a.info[d.GEN_INDEX]))
    gen = corrected_entries_from_atoms(frames)
    mp = fetch_entries(args.elements)
    pd_mp, pd_all = PhaseDiagram(mp), PhaseDiagram(mp + gen)
    table = pd.concat([rows(mp, pd_all, pd_mp, d.SOURCE_MP),
                       rows(gen, pd_all, pd_mp, d.SOURCE_GENERATED)])
    out = d.output_dir(args.out) / f"{args.tag}_hull_data.csv"
    table.to_csv(out, index=False)
    print(f"{out}: {len(mp)} MP and {len(gen)} generated rows")


if __name__ == "__main__":
    main()
