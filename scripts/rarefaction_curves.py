"""Rarefaction curves of the stable-candidate count, TCS and MCS (Figure 2).

For each system, the generated structures are added in 50 random orderings
(``default_rng(i)``, i = 0..49) and the updated hull is rebuilt every 50
structures. Writes ``<TAG>_rarefaction_orderings.csv`` (every ordering) and
``<TAG>_rarefaction_summary.csv`` (mean, standard deviation, 2.5th and 97.5th
percentiles per checkpoint).

    python scripts/rarefaction_curves.py --data data --out results
    python scripts/rarefaction_curves.py --systems LRO_15000 LNMO LNRO
"""

from __future__ import annotations

from hullconv import rarefaction
from hullconv.cli import parser, resolve
from hullconv.data import load_hull_data
from hullconv.hull import HullModel


def main() -> None:
    p = parser(__doc__.splitlines()[0])
    p.add_argument("--orderings", type=int, default=rarefaction.N_ORDERINGS)
    p.add_argument("--step", type=int, default=rarefaction.STEP)
    args = p.parse_args()
    root, out = resolve(args)
    for tag in args.systems:
        model = HullModel(load_hull_data(tag, root))
        curves = rarefaction.rarefaction_curves(
            model, n_orderings=args.orderings, step=args.step, n_jobs=args.n_jobs)
        summary = rarefaction.summarize(curves)
        curves.to_csv(out / f"{tag}_rarefaction_orderings.csv", index=False)
        summary.to_csv(out / f"{tag}_rarefaction_summary.csv", index=False)
        last = summary.iloc[-1]
        print(f"{tag}: N_gen = {int(last.n_gen)}, "
              f"candidates = {last.stable_candidates_mean:.2f}, "
              f"TCS = {last.TCS_meV_mean:.3f} meV/atom")


if __name__ == "__main__":
    main()
