"""Permutation rarefaction of the hull metrics.

The generated structures are added one at a time in a random order, and every
``step`` structures the updated hull is rebuilt to record the number of stable
candidates and TCS. The procedure is repeated for ``n_orderings`` random
orderings (permutations without replacement). Ordering ``i`` is
``numpy.random.default_rng(seed_base + i).permutation(n_gen)`` applied to the
generated structures in ``gen_index`` order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

if TYPE_CHECKING:
    from hullconv.hull import HullModel

N_ORDERINGS = 50
STEP = 50
SEED_BASE = 0
PERCENTILES = (2.5, 97.5)
# Absolute slack when testing whether a mean curve has reached a fraction of
# its final value; the candidate count can sit exactly on 90%.
REACH_TOL = 1e-9


def checkpoints(n_gen: int, step: int = STEP) -> np.ndarray:
    """Checkpoints step, 2*step, ..., ending with ``n_gen``."""
    n = np.arange(step, n_gen + 1, step)
    if len(n) == 0 or n[-1] != n_gen:
        n = np.append(n, n_gen)
    return n


def ordering_ranks(seed: int, n_gen: int) -> np.ndarray:
    """Position of each generated structure in ordering ``seed``."""
    perm = np.random.default_rng(seed).permutation(n_gen)
    rank = np.empty(n_gen, dtype=int)
    rank[perm] = np.arange(n_gen)
    return rank


def curve_for_ranks(model: HullModel, rank: np.ndarray,
                    steps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Candidate count and TCS (meV/atom) at each checkpoint of one ordering.

    ``rank`` gives the position of every generated structure; a structure is
    present at checkpoint n when its position is below n.
    """
    rel = model.relevant
    rel_rank = rank[rel]
    n_cand = np.zeros(len(steps), dtype=int)
    tcs = np.zeros(len(steps))
    for k, n in enumerate(steps):
        c, t, _ = model.state(rel[rel_rank < n])
        n_cand[k], tcs[k] = c, 1000.0 * t
    return n_cand, tcs


def _one_ordering(model: HullModel, seed: int, steps: np.ndarray) -> pd.DataFrame:
    n_cand, tcs = curve_for_ranks(model, ordering_ranks(seed, len(model.gen_entries)),
                                  steps)
    return pd.DataFrame({"ordering": seed, "n_gen": steps,
                         "stable_candidates": n_cand, "TCS_meV": tcs,
                         "MCS_meV": tcs / model.n_mp})


def rarefaction_curves(model: HullModel, n_orderings: int = N_ORDERINGS,
                       step: int = STEP, seed_base: int = SEED_BASE,
                       n_jobs: int = -1) -> pd.DataFrame:
    """Curves of all orderings, one row per (ordering, checkpoint)."""
    steps = checkpoints(len(model.gen_entries), step)
    parts = Parallel(n_jobs=n_jobs)(
        delayed(_one_ordering)(model, seed_base + i, steps)
        for i in range(n_orderings))
    return pd.concat(parts, ignore_index=True)


def summarize(curves: pd.DataFrame, value_cols: list[str] | None = None,
              by: str = "n_gen") -> pd.DataFrame:
    """Mean, standard deviation and 2.5th/97.5th percentiles per checkpoint."""
    cols = value_cols or ["stable_candidates", "TCS_meV", "MCS_meV"]
    g = curves.groupby(by)[cols]
    out = pd.concat({"mean": g.mean(), "std": g.std(),
                     "p2.5": g.quantile(PERCENTILES[0] / 100),
                     "p97.5": g.quantile(PERCENTILES[1] / 100)}, axis=1)
    out.columns = [f"{col}_{stat}" for stat, col in out.columns]
    return out.reset_index()


def first_reach(n: np.ndarray, y: np.ndarray, fraction: float = 0.9) -> float:
    """First checkpoint at which ``y`` reaches ``fraction`` of its final value."""
    target = fraction * y[-1]
    hit = np.flatnonzero(np.asarray(y) >= target - REACH_TOL)
    return float(n[hit[0]]) if len(hit) else float("nan")
