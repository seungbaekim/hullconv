"""Table 2: stable candidates, TCS and MCS at 10,000 generated structures.

Writes ``table2.csv`` (one row per system) and ``table2_candidates.csv`` (one
row per stable candidate). With ``--classify``, candidates at compositions
present in the MP are split into reproduced structures and novel polymorphs by
matching against the MP structures, which requires ``MP_API_KEY``.

    python scripts/table2_hull_metrics.py --data data --out results
"""

from __future__ import annotations

import pandas as pd

from hullconv.cli import parser, resolve
from hullconv.data import load_hull_data, read_structures
from hullconv.hull import HullModel, candidate_table, endpoint_metrics

# StructureMatcher tolerances for a candidate that reproduces an MP structure;
# volumes are scaled before matching (pymatgen default).
MATCH_LTOL, MATCH_STOL, MATCH_ANGLE_TOL = 0.2, 0.3, 5.0


def classify(cands: pd.DataFrame, tag: str, elements: list[str], root) -> list[str]:
    """Relation of each candidate to the MP: reproduced, polymorph or composition."""
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.io.ase import AseAtomsAdaptor

    from hullconv.mp_data import fetch_entries

    mp_entries = fetch_entries(elements)
    frames = {int(a.info["gen_index"]): a for a in read_structures(tag, root)}
    matcher = StructureMatcher(ltol=MATCH_LTOL, stol=MATCH_STOL,
                               angle_tol=MATCH_ANGLE_TOL)
    labels = []
    for row in cands.itertuples(index=False):
        if row.novel_composition:
            labels.append("novel composition")
            continue
        gen = AseAtomsAdaptor.get_structure(frames[row.gen_index])
        same = [e.structure for e in mp_entries
                if e.composition.reduced_formula == row.formula]
        hit = any(matcher.fit(gen, s) for s in same)
        labels.append("reproduced" if hit else "novel polymorph")
    return labels


def main() -> None:
    p = parser(__doc__.splitlines()[0])
    p.add_argument("--classify", action="store_true",
                   help="match candidates to MP structures (needs MP_API_KEY)")
    args = p.parse_args()
    root, out = resolve(args)
    rows, cand_rows = [], []
    for tag in args.systems:
        data = load_hull_data(tag, root)
        model = HullModel(data)
        rows.append(endpoint_metrics(model))
        cands = candidate_table(model)
        cands.insert(0, "system", tag)
        if args.classify:
            cands["relation_to_MP"] = classify(cands, tag, data.elements, root)
        cand_rows.append(cands)
    table = pd.DataFrame(rows)
    cand_df = pd.concat(cand_rows, ignore_index=True)
    table["novel_compositions"] = [
        int(cand_df[(cand_df.system == t)].novel_composition.sum())
        for t in table.system]
    table.to_csv(out / "table2.csv", index=False)
    cand_df.to_csv(out / "table2_candidates.csv", index=False)
    print(table.drop(columns="candidates").to_string(index=False))


if __name__ == "__main__":
    main()
