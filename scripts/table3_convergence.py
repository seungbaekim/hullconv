"""Table 3: number of generated structures at which each criterion is met.

Reads ``<TAG>_rarefaction_summary.csv`` and ``<TAG>_coverage_summary.csv``
from the results directory (written by ``rarefaction_curves.py`` and
``coverage_table1.py``) and needs TorchSISSO, but not pymatgen.

* TCS and stable candidates: first checkpoint (every 50 structures) at which
  the mean curve reaches 90% of its value at the last structure.
* Coverage: first sample size (every 100 structures) at which the mean
  coverage reaches 0.90.
* SISSO: the candidate curve is fitted up to N = 2,000, 3,000, ..., and each
  fit is evaluated at the last structure; the table gives the N from which
  the endpoint residual stays below 0.5.
* N_min is the largest of the four.

Writes ``table3.csv`` and ``sisso_endpoint_residuals.csv``.

    python scripts/table3_convergence.py --out results
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from hullconv import sisso_fit
from hullconv.data import TERNARY_SYSTEMS, output_dir
from hullconv.rarefaction import first_reach

FRACTION = 0.9
MIN_COVERAGE = 0.90
FIRST_CUTOFF = 2000
CUTOFF_INTERVAL = 1000


def coverage_reach(curve: pd.DataFrame, level: float = MIN_COVERAGE) -> float:
    """First sample size at which the mean coverage reaches ``level``."""
    hit = curve[curve["coverage_mean"] >= level]
    return float(hit["n_gen"].iloc[0]) if len(hit) else float("nan")


def system_row(tag: str, results) -> tuple[dict, pd.DataFrame]:
    """Table 3 row and SISSO residuals of one system."""
    rare = pd.read_csv(results / f"{tag}_rarefaction_summary.csv")
    cov = pd.read_csv(results / f"{tag}_coverage_summary.csv")
    n = rare["n_gen"].to_numpy()
    y = rare["stable_candidates_mean"].to_numpy()
    n_end = int(n[-1])
    cutoffs = list(range(FIRST_CUTOFF, n_end + 1, CUTOFF_INTERVAL))
    res = sisso_fit.endpoint_residuals(n, y, cutoffs, n_end)
    res.insert(0, "system", tag)
    row = {"system": tag,
           "TCS_90pct": first_reach(n, rare["TCS_meV_mean"].to_numpy(), FRACTION),
           "candidates_90pct": first_reach(n, y, FRACTION),
           "coverage_90pct": coverage_reach(cov),
           "sisso_residual_below_0.5": sisso_fit.stays_below_from(res)}
    row["N_min"] = float(np.nanmax(list(row.values())[1:]))
    return row, res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="results", help="results directory")
    p.add_argument("--systems", nargs="+", default=list(TERNARY_SYSTEMS))
    args = p.parse_args()
    results = output_dir(args.out)
    rows, residuals = [], []
    for tag in args.systems:
        row, res = system_row(tag, results)
        rows.append(row)
        residuals.append(res)
    table = pd.DataFrame(rows)
    table.to_csv(results / "table3.csv", index=False)
    pd.concat(residuals).to_csv(results / "sisso_endpoint_residuals.csv",
                                index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
