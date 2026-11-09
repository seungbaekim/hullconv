"""Table S3: stable candidates of LRO in two independently generated pools.

For each pool, the hull is rebuilt from the MP structures and all generated
structures of the pool. For every composition that is a stable candidate in
either pool, the table gives the Ehull (meV/atom) of the lowest-energy
generated structure of that composition against the MP-only hull, and the
number of generated copies; the last row gives TCS (meV/atom).

    python scripts/tableS3_independent_pool.py --data data --out results
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hullconv.cli import parser, resolve
from hullconv.data import load_hull_data
from hullconv.hull import HullModel, endpoint_metrics


def pool_columns(model: HullModel, formulas: list[str], label: str) -> pd.DataFrame:
    """Lowest MP-only Ehull (meV/atom) and copy count per composition."""
    rows = []
    for f in formulas:
        idx = np.flatnonzero(model.formulas == f)
        best = 1000.0 * model.gen_ehull_mp_only[idx].min() if len(idx) else np.nan
        rows.append({"formula": f, f"{label}_ehull_meV": best,
                     f"{label}_copies": len(idx)})
    return pd.DataFrame(rows).set_index("formula")


def main() -> None:
    p = parser(__doc__.splitlines()[0], systems=("LRO", "LRO_independent"))
    args = p.parse_args()
    root, out = resolve(args)
    models = {tag: HullModel(load_hull_data(tag, root))
              for tag in args.systems}
    ends = {tag: endpoint_metrics(m) for tag, m in models.items()}
    formulas = sorted({f for e in ends.values() for f in e["candidates"].split(";")
                       if f})
    table = pd.concat([pool_columns(m, formulas, tag) for tag, m in models.items()],
                      axis=1)
    order = table[[c for c in table.columns if c.endswith("_ehull_meV")]].min(axis=1)
    table = table.loc[order.sort_values().index]
    tcs = {f"{tag}_ehull_meV": e["TCS_meV"] for tag, e in ends.items()}
    table.loc["TCS"] = pd.Series(tcs)
    table.to_csv(out / "tableS3.csv")
    print(table.round(1).to_string())


if __name__ == "__main__":
    main()
