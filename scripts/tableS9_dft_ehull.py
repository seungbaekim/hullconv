"""Table S9: DFT energy above the MP-only hull of the validated candidates.

Reads ``dft_summary.csv`` from the data directory. Each row is one VASP
calculation in a folder of the same name next to it (with ``vasprun.xml``):

* ``calculation``: folder name
* ``system``: system tag; a competing phase shared by several systems lists
  them separated by ``;``
* ``role``: ``candidate`` or ``competing``

Each candidate's Ehull is taken against the competing phases of its system,
all MP2020-corrected. Writes ``tableS9.csv``.

    python scripts/tableS9_dft_ehull.py --data data --out results
"""

from __future__ import annotations

import pandas as pd

from hullconv import dft_ehull
from hullconv.cli import parser, resolve
from hullconv.data import find_file

ROLE_CANDIDATE = "candidate"
ROLE_COMPETING = "competing"


def main() -> None:
    p = parser(__doc__.splitlines()[0],
               systems=("LMO", "LCO", "LRO", "LHO", "LZO", "LIO"))
    args = p.parse_args()
    root, out = resolve(args)
    summary_path = find_file(root, "dft_summary.csv")
    summary = pd.read_csv(summary_path)
    base = summary_path.parent
    summary["systems"] = summary["system"].astype(str).str.split(";")
    phases = {c: dft_ehull.read_phase(base / c, c) for c in summary["calculation"]}
    rows = []
    for tag in args.systems:
        mine = summary[summary["systems"].apply(lambda s: tag in s)]
        competing = [phases[c] for c in mine.loc[mine.role == ROLE_COMPETING,
                                                 "calculation"]]
        for c in mine.loc[mine.role == ROLE_CANDIDATE, "calculation"]:
            cand = phases[c]
            ehull, decomp = dft_ehull.e_above_hull(cand, competing)
            rows.append({"system": tag, "calculation": c,
                         "formula": cand.composition.reduced_formula,
                         "energy_corrected_eV_per_atom": cand.energy_per_atom,
                         "ehull_dft_meV": 1000.0 * ehull,
                         "decomposition": dft_ehull.format_decomposition(
                             decomp, competing)})
    table = pd.DataFrame(rows)
    table.to_csv(out / "tableS9.csv", index=False)
    print(table.drop(columns="energy_corrected_eV_per_atom")
          .round(1).to_string(index=False))


if __name__ == "__main__":
    main()
