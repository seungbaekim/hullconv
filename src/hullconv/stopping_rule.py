"""Stopping rule for generation, judged from the structures generated so far.

The rule is checked at N = 8,000 structures and after every further 1,000.
At each check, only the first N structures (in generation order) are used:

1. Sample coverage of the N structures is at least 0.90.
2. On the rarefaction curve of the N structures, TCS rises by at most
   1 meV/atom over the second half (from N/2 to N).
3. The stable-candidate count at N exceeds the value predicted at N by a SISSO
   fit to the first half of the candidate curve (n <= N/2) by at most 0.5.

Generation stops at the first check where all three hold. Before the rule is
applied, stable candidates that DFT places above the MP-only hull and those
that reproduce an MP structure are removed (``HullModel.exclude``).

The rarefaction curve at a check is the mean over ``N_ORDERINGS`` random
orderings of the N structures; ordering ``r`` is
``default_rng(SEED_BASE + r).permutation(N)``, sampled every 50 structures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hullconv import sisso_fit
from hullconv.coverage import sample_coverage

FIRST_CHECK = 8000
CHECK_INTERVAL = 1000
MIN_COVERAGE = 0.90
MAX_TCS_RISE_MEV = 1.0      # meV/atom over the second half of the curve
MAX_CANDIDATE_EXCESS = 0.5  # same tolerance as the SISSO endpoint residual
N_ORDERINGS = 20
SEED_BASE = 10_000
STEP = 50


def _join(values: np.ndarray, fmt: str) -> str:
    return ";".join(format(v, fmt) for v in values)


def _split(text: str) -> np.ndarray:
    return np.array([float(v) for v in text.split(";")])


def within_n_curve(model, n: int, n_orderings: int = N_ORDERINGS,
                   seed_base: int = SEED_BASE,
                   step: int = STEP) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean candidate count and TCS (meV/atom) over orderings of the first n structures.

    ``model`` is a ``hullconv.hull.HullModel``. Returns checkpoints, mean
    candidate count and mean TCS.
    """
    from hullconv.rarefaction import curve_for_ranks

    n_gen = len(model.gen_entries)
    steps = np.arange(step, n + 1, step)
    cand, tcs = [], []
    for r in range(n_orderings):
        rank = np.full(n_gen, n_gen)
        rank[:n] = np.random.default_rng(seed_base + r).permutation(n)
        c, t = curve_for_ranks(model, rank, steps)
        cand.append(c)
        tcs.append(t)
    return steps, np.mean(cand, axis=0), np.mean(tcs, axis=0)


def rule_curves(model, checks: list[int], **kwargs) -> pd.DataFrame:
    """Within-N curves at each check, stored as ';'-joined strings."""
    rows = []
    for n in checks:
        steps, cand, tcs = within_n_curve(model, n, **kwargs)
        rows.append({"system": model.data.tag, "N": n,
                     "candidate_curve": _join(cand, ".3f"),
                     "tcs_curve_meV": _join(tcs, ".4f")})
    return pd.DataFrame(rows)


def rule_signals(curves: pd.DataFrame, labels: list[str],
                 step: int = STEP) -> pd.DataFrame:
    """Coverage, TCS rise and SISSO excess at each check.

    ``labels`` are the species labels in generation order. Needs TorchSISSO.
    """
    rows = []
    for row in curves.itertuples(index=False):
        n = int(row.N)
        cand, tcs = _split(row.candidate_curve), _split(row.tcs_curve_meV)
        steps = np.arange(step, n + 1, step)
        half = steps <= n / 2
        fit = sisso_fit.fit_curve(steps[half], cand[half])
        predicted = float(fit.predict(n)[0])
        rows.append({"system": row.system, "N": n,
                     "coverage": sample_coverage(np.asarray(labels[:n])),
                     "TCS_meV": tcs[-1],
                     "TCS_rise_second_half_meV": tcs[-1] - tcs[half][-1],
                     "stable_candidates": cand[-1],
                     "sisso_first_half_prediction": predicted,
                     "candidate_excess": cand[-1] - predicted,
                     "sisso_equation": fit.equation})
    out = pd.DataFrame(rows)
    out["coverage_ok"] = out["coverage"] >= MIN_COVERAGE
    out["tcs_ok"] = out["TCS_rise_second_half_meV"] <= MAX_TCS_RISE_MEV
    out["sisso_ok"] = out["candidate_excess"] <= MAX_CANDIDATE_EXCESS
    out["stop"] = out["coverage_ok"] & out["tcs_ok"] & out["sisso_ok"]
    return out


def stopping_point(signals: pd.DataFrame,
                   first_check: int = FIRST_CHECK) -> float:
    """First check at or after ``first_check`` where the rule stops, else NaN."""
    s = signals[signals["N"] >= first_check].sort_values("N")
    hit = s[s["stop"]]
    return float(hit["N"].iloc[0]) if len(hit) else float("nan")


def default_checks(n_gen: int, first_check: int = FIRST_CHECK,
                   interval: int = CHECK_INTERVAL) -> list[int]:
    """Check points ``first_check, first_check + interval, ...`` up to n_gen."""
    return list(range(first_check, n_gen + 1, interval))
