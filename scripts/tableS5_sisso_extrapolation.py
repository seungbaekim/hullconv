"""Table S5: SISSO fits of the full candidate curves and the next candidate.

Fits the mean stable-candidate curve of each system up to its last structure
and reports the equation, R^2 and the number of additional structures after
which the fitted count first exceeds the observed count by one (``inf`` when
the fit does not reach it). Reads ``<TAG>_rarefaction_summary.csv`` from the
results directory; needs TorchSISSO, not pymatgen. Writes ``tableS5.csv``.

    python scripts/tableS5_sisso_extrapolation.py --out results
    python scripts/tableS5_sisso_extrapolation.py --systems LRO_15000
"""

from __future__ import annotations

import argparse

import pandas as pd

from hullconv import sisso_fit
from hullconv.data import TERNARY_SYSTEMS, output_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="results", help="results directory")
    p.add_argument("--systems", nargs="+",
                   default=list(TERNARY_SYSTEMS) + ["LRO_15000"])
    args = p.parse_args()
    results = output_dir(args.out)
    rows = []
    for tag in args.systems:
        rare = pd.read_csv(results / f"{tag}_rarefaction_summary.csv")
        n = rare["n_gen"].to_numpy()
        y = rare["stable_candidates_mean"].to_numpy()
        fit = sisso_fit.fit_curve(n, y)
        extra = sisso_fit.additional_to_next_candidate(fit, int(n[-1]), float(y[-1]))
        rows.append({"system": tag, "n_end": int(n[-1]), "equation": fit.equation,
                     "r2": fit.r2, "additional_structures_to_next": extra})
    table = pd.DataFrame(rows)
    table.to_csv(results / "tableS5.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
