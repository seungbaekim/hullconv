"""Holdout test (Figure 3): Group A systems with their MP structures reduced to 70.

The 100 subsamples are read from ``<TAG>_holdout_subsamples.csv`` (columns
``subsample`` and ``mp_id``) when the file is present in the data directory;
otherwise they are drawn with ``default_rng(42)``. Writes
``<TAG>_holdout_endpoints.csv`` (candidates, TCS and MCS of every subsample
with all generated structures), ``holdout_summary.csv`` (mean, standard
deviation, 2.5th and 97.5th percentiles over subsamples) and, with
``--curves``, ``<TAG>_holdout_curves.csv`` (16 orderings per subsample, every
250 structures).

    python scripts/holdout_test.py --data data --out results
"""

from __future__ import annotations

import pandas as pd
from joblib import Parallel, delayed

from hullconv import holdout
from hullconv.cli import parser, resolve
from hullconv.data import MP_ID, find_file, load_hull_data


def subsamples(tag: str, root, data) -> dict[int, list[str]]:
    """Published subsamples if available, else a fresh draw."""
    try:
        return holdout.load_subsamples(find_file(root, f"{tag}_holdout_subsamples.csv"))
    except FileNotFoundError:
        draws = holdout.draw_subsamples(list(data.mp[MP_ID]),
                                        holdout.PROTECTED_MP_IDS[tag])
        return {t + 1: ids for t, ids in enumerate(draws)}


def endpoint(data, t: int, ids: list[str]) -> dict:
    """Endpoint metrics of subsample ``t``."""
    return {"subsample": t, **holdout.subsample_endpoint(
        holdout.subsample_model(data, ids))}


def curves(data, t: int, ids: list[str]) -> pd.DataFrame:
    """Ordering-averaged curves of subsample ``t``."""
    return holdout.subsample_curves(holdout.subsample_model(data, ids), t)


def main() -> None:
    p = parser(__doc__.splitlines()[0], systems=tuple(holdout.PROTECTED_MP_IDS))
    p.add_argument("--curves", action="store_true",
                   help="also compute the curves (slower)")
    args = p.parse_args()
    root, out = resolve(args)
    summary = []
    for tag in args.systems:
        data = load_hull_data(tag, root)
        subs = subsamples(tag, root, data)
        ends = pd.DataFrame(Parallel(n_jobs=args.n_jobs)(
            delayed(endpoint)(data, t, ids) for t, ids in sorted(subs.items())))
        ends.to_csv(out / f"{tag}_holdout_endpoints.csv", index=False)
        stats = holdout.summarize_subsamples(ends)
        stats.insert(0, "system", tag)
        summary.append(stats)
        if args.curves:
            parts = Parallel(n_jobs=args.n_jobs)(
                delayed(curves)(data, t, ids) for t, ids in sorted(subs.items()))
            cur = pd.concat(parts, ignore_index=True)
            cur.to_csv(out / f"{tag}_holdout_curves.csv", index=False)
            holdout.summarize_subsamples(cur, by="n_gen").to_csv(
                out / f"{tag}_holdout_curves_summary.csv", index=False)
    table = pd.concat(summary, ignore_index=True)
    table.to_csv(out / "holdout_summary.csv", index=False)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
