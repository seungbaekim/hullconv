"""Holdout test: withhold MP structures and add the same generated structures.

For a system with many MP structures, the MP set is reduced at random to 70
structures, always keeping the elemental phases (Li metal, the transition metal
and O2). The reduction is repeated for 100 subsamples. For each subsample, the
generated structures are added in 16 random orderings and the hull is rebuilt
every 250 structures; the MP-only hull and TCS refer to the 70 retained MP
structures. Results are averaged over the orderings of a subsample and then
summarized over the subsamples.

In each subsample a stable candidate is a generated structure that is a vertex
of the updated hull.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from hullconv.data import MP_ID, HullData
from hullconv.hull import HullModel
from hullconv.rarefaction import checkpoints, curve_for_ranks

TARGET_N_MP = 70
N_SUBSAMPLES = 100
N_ORDERINGS = 16
STEP = 250
SUBSAMPLE_SEED = 42
# Orderings of subsample t use default_rng(t * ORDERING_SEED_FACTOR).
ORDERING_SEED_FACTOR = 1000

# MP structures kept in every subsample: Li (mp-1018134), the transition
# metal, and O2 (mp-12957).
PROTECTED_MP_IDS = {
    "LMO": ("mp-1018134", "mp-35", "mp-12957"),
    "LCO": ("mp-1018134", "mp-54", "mp-12957"),
    "LNO": ("mp-1018134", "mp-23", "mp-12957"),
}


def base_mp_id(mp_id: str) -> str:
    """'mp-1960-GGA' or 'mp-1960' -> 'mp-1960'."""
    return "-".join(str(mp_id).split("-")[:2])


def draw_subsamples(mp_ids: list[str], protected: tuple[str, ...],
                    n_target: int = TARGET_N_MP, n_subsamples: int = N_SUBSAMPLES,
                    seed: int = SUBSAMPLE_SEED) -> list[list[str]]:
    """Random MP subsets of size ``n_target`` that always keep ``protected``.

    The draw depends on the order of ``mp_ids``; to reproduce the published
    subsamples, read them with ``load_subsamples`` instead.
    """
    keep = [m for m in mp_ids if base_mp_id(m) in protected]
    if len(keep) != len(protected):
        raise ValueError(f"Protected MP structures not found: {protected}")
    removable = [m for m in mp_ids if m not in keep]
    rng = np.random.default_rng(seed)
    n_draw = n_target - len(keep)
    return [keep + list(rng.choice(removable, size=n_draw, replace=False))
            for _ in range(n_subsamples)]


def load_subsamples(path: str) -> dict[int, list[str]]:
    """Read a subsample file with columns ``subsample`` and ``mp_id``."""
    df = pd.read_csv(path, dtype={MP_ID: str})
    return {int(t): list(g[MP_ID]) for t, g in df.groupby("subsample")}


def subsample_model(data: HullData, kept_ids: list[str]) -> HullModel:
    """Hull model with only the MP structures in ``kept_ids``."""
    kept = {base_mp_id(m) for m in kept_ids}
    mask = data.mp[MP_ID].map(base_mp_id).isin(kept)
    if mask.sum() != len(kept):
        raise ValueError(f"{len(kept)} MP ids requested, {mask.sum()} found.")
    sub = replace(data, mp=data.mp[mask].reset_index(drop=True))
    return HullModel(sub, vertex_tol=0.0)


def subsample_endpoint(model: HullModel) -> dict:
    """Candidate count, TCS and MCS (meV/atom) with all generated structures."""
    n_cand, tcs, stable = model.state(np.arange(len(model.gen_entries)))
    return {"stable_candidates": n_cand, "TCS_meV": 1000.0 * tcs,
            "MCS_meV": 1000.0 * tcs / model.n_mp,
            "candidates": ";".join(sorted(stable))}


def subsample_curves(model: HullModel, subsample: int,
                     n_orderings: int = N_ORDERINGS,
                     step: int = STEP) -> pd.DataFrame:
    """Curves of one subsample, averaged over its orderings."""
    n_gen = len(model.gen_entries)
    steps = checkpoints(n_gen, step)
    rng = np.random.default_rng(subsample * ORDERING_SEED_FACTOR)
    cand, tcs = [], []
    for _ in range(n_orderings):
        perm = rng.permutation(n_gen)
        rank = np.empty(n_gen, dtype=int)
        rank[perm] = np.arange(n_gen)
        c, t = curve_for_ranks(model, rank, steps)
        cand.append(c)
        tcs.append(t)
    tcs_mean = np.mean(tcs, axis=0)
    return pd.DataFrame({"subsample": subsample, "n_gen": steps,
                         "stable_candidates": np.mean(cand, axis=0),
                         "TCS_meV": tcs_mean, "MCS_meV": tcs_mean / model.n_mp})


def summarize_subsamples(df: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Mean, standard deviation and 2.5th/97.5th percentiles over subsamples."""
    cols = ["stable_candidates", "TCS_meV", "MCS_meV"]
    g = df.groupby(by)[cols] if by else df[cols]
    stats = {"mean": g.mean(), "std": g.std(), "p2.5": g.quantile(0.025),
             "p97.5": g.quantile(0.975)}
    if by is None:
        return pd.DataFrame(stats).T.reset_index(names="statistic")
    out = pd.concat(stats, axis=1)
    out.columns = [f"{col}_{stat}" for stat, col in out.columns]
    return out.reset_index()
