"""Species labels, sample coverage curves and Table 1.

For each system, assigns a species (reduced formula + space group, spglib at
0.1 Angstrom) to every relaxed structure and writes ``<TAG>_species.csv``.
The coverage curve over 50 random orderings (``default_rng(42 + i)``, every
100 structures) goes to ``<TAG>_coverage_summary.csv``. ``table1.csv`` gives,
at all generated structures: number of MP structures, fraction containing all
elements, median Ehull and fraction with Ehull <= 0.03 eV/atom against the
MP-only hull, and coverage over all structures and over those containing all
elements.

    python scripts/coverage_table1.py --data data --out results
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hullconv import coverage
from hullconv.cli import parser, resolve
from hullconv.data import GEN_INDEX, load_hull_data, read_structures
from hullconv.hull import HullModel

EHULL_TARGET = 0.03  # eV/atom, Ehull used to condition the generation


def species_table(tag: str, root, out, n_jobs: int) -> pd.DataFrame:
    """Species label per generated structure, cached in ``<TAG>_species.csv``."""
    path = out / f"{tag}_species.csv"
    if path.exists():
        return pd.read_csv(path)
    frames = read_structures(tag, root, kind="relaxed")
    df = pd.DataFrame({GEN_INDEX: [int(a.info[GEN_INDEX]) for a in frames],
                       "species": coverage.species_labels(frames, n_jobs=n_jobs)})
    df.to_csv(path, index=False)
    return df


def table1_row(tag: str, model: HullModel, species: pd.DataFrame) -> dict:
    """Table 1 values of one system (percentages)."""
    n_el = len(model.data.elements)
    labels = species["species"].tolist()
    full = coverage.all_elements_mask(labels, n_el)
    ehull = model.gen_ehull_mp_only
    return {"system": tag, "n_mp": model.n_mp, "n_gen": len(labels),
            "all_elements_pct": 100.0 * full.mean(),
            "median_ehull_mp_only_eV": float(np.median(ehull)),
            "ehull_le_0.03_pct": 100.0 * float(np.mean(ehull <= EHULL_TARGET)),
            "coverage_pct": 100.0 * coverage.sample_coverage(np.array(labels)),
            "coverage_all_elements_pct":
                100.0 * coverage.sample_coverage(np.array(labels)[full])}


def main() -> None:
    p = parser(__doc__.splitlines()[0])
    args = p.parse_args()
    root, out = resolve(args)
    rows = []
    for tag in args.systems:
        model = HullModel(load_hull_data(tag, root))
        species = species_table(tag, root, out, args.n_jobs)
        if not (species[GEN_INDEX].to_numpy()
                == model.data.gen[GEN_INDEX].to_numpy()).all():
            raise ValueError(f"{tag}: structures and hull_data rows differ")
        curve = coverage.coverage_rarefaction(species["species"].tolist())
        curve.to_csv(out / f"{tag}_coverage_summary.csv", index=False)
        rows.append(table1_row(tag, model, species))
    table = pd.DataFrame(rows)
    table.to_csv(out / "table1.csv", index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
