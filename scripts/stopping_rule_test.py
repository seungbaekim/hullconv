"""Stopping rule applied to the structures in the order they were generated.

Stage ``curves`` (needs pymatgen) rebuilds, at every check N, the rarefaction
curve of the first N structures after removing the stable candidates that the
DFT step rules out, and writes ``<TAG>_stopping_rule_curves.csv``. Stage
``signals`` (needs TorchSISSO) evaluates the three conditions from these
curves and the species labels of ``coverage_table1.py``, and writes
``<TAG>_stopping_rule_signals.csv`` and ``stopping_rule.csv``.

    python scripts/stopping_rule_test.py --stage curves --data data --out results
    python scripts/stopping_rule_test.py --stage signals --out results
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hullconv import stopping_rule as rule
from hullconv.cli import parser, resolve
from hullconv.data import TERNARY_SYSTEMS

# Stable candidates removed by the DFT step: structures that reproduce an MP
# structure and candidates that DFT places above the MP-only hull.
DFT_STEP_EXCLUDE = {
    "LMO": ("Li2O", "Li7Mn2O"),
    "LCO": ("Li2O", "Li4Co2O", "Li9Co2O"),
    "LNO": ("Li2O",),
    "LRO": ("Li2O", "Li2RuO3"),
    "LHO": ("Li2O",),
    "LZO": ("Li2O", "Zr3O", "Zr2O"),
    "LIO": ("Li2O",),
}


def run_curves(tag: str, root, out, args) -> None:
    """Within-N rarefaction curves at every check."""
    from hullconv.data import load_hull_data
    from hullconv.hull import HullModel

    if args.exclude is None:
        excluded = DFT_STEP_EXCLUDE.get(tag, ())
    else:
        excluded = tuple(args.exclude)
    model = HullModel(load_hull_data(tag, root),
                      exclude=excluded)
    checks = rule.default_checks(len(model.gen_entries), args.first_check)
    curves = rule.rule_curves(model, checks)
    curves.insert(1, "excluded", ";".join(excluded))
    curves.to_csv(out / f"{tag}_stopping_rule_curves.csv", index=False)


def run_signals(tag: str, out) -> dict:
    """Conditions at every check and the stopping point."""
    curves = pd.read_csv(out / f"{tag}_stopping_rule_curves.csv")
    labels = pd.read_csv(out / f"{tag}_species.csv")["species"].tolist()
    signals = rule.rule_signals(curves, labels)
    signals.to_csv(out / f"{tag}_stopping_rule_signals.csv", index=False)
    stop = rule.stopping_point(signals)
    at = signals[signals["N"] == stop] if np.isfinite(stop) else signals.tail(1)
    return {"system": tag, "stopped_at": stop,
            "TCS_meV": float(at["TCS_meV"].iloc[0]),
            "stable_candidates": float(at["stable_candidates"].iloc[0])}


def main() -> None:
    p = parser(__doc__.splitlines()[0], systems=TERNARY_SYSTEMS + ("LIO",))
    p.add_argument("--stage", choices=["curves", "signals"], required=True)
    p.add_argument("--first-check", type=int, default=rule.FIRST_CHECK,
                   help="first N at which the conditions are evaluated")
    p.add_argument("--exclude", nargs="*", default=None,
                   help="formulas removed by the DFT step (overrides defaults; "
                        "give the option without formulas to remove none)")
    args = p.parse_args()
    if args.stage == "curves":
        root, out = resolve(args)
        for tag in args.systems:
            run_curves(tag, root, out, args)
            print(f"{tag}: curves written")
        return
    from hullconv.data import output_dir

    out = output_dir(args.out)
    table = pd.DataFrame([run_signals(tag, out) for tag in args.systems])
    table.to_csv(out / "stopping_rule.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
