"""Closed-form fits of the stable-candidate curve with SISSO (TorchSISSO).

The mean number of stable candidates over the orderings is fitted against the
number of generated structures, ``sample_size``, with the operator set
{+, -, *, /, exp, ln, sqrt}, ``n_expansion = 3``, ``n_term = 3`` and ``k = 30``.
The fitted equation is used to extrapolate the curve beyond the generated
range.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass

import numpy as np
import pandas as pd

OPERATORS = ["+", "-", "*", "/", "exp", "ln", "sqrt"]
N_EXPANSION = 3
N_TERM = 3
K = 30
TARGET = "stable_discovery_mean"
FEATURE = "sample_size"
# Range below which a curve is treated as flat and not fitted.
FLAT_RANGE = 1e-9
# Upper bound of the root search for the next candidate (structures).
SEARCH_LIMIT = 1_000_000_000
# Endpoint residual below which a fit reproduces the observed integer count.
RESIDUAL_TOL = 0.5


@dataclass
class SissoFit:
    """Fitted equation (empty for a flat curve) and training statistics."""

    equation: str
    rmse: float
    r2: float
    last_value: float

    def predict(self, x: np.ndarray | float) -> np.ndarray:
        """Evaluate the fit; a flat curve predicts its last value."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        if not self.equation:
            return np.full_like(x, self.last_value)
        return evaluate(self.equation, x)


def evaluate(equation: str, x: np.ndarray) -> np.ndarray:
    """Evaluate a TorchSISSO equation string at ``sample_size = x``."""
    expr = (equation.replace("ln", "np.log").replace("exp", "np.exp")
            .replace("sqrt", "np.sqrt"))
    with np.errstate(all="ignore"):
        value = eval(expr, {"np": np, "__builtins__": {}},  # noqa: S307
                     {FEATURE: np.asarray(x, dtype=float)})
    return np.broadcast_to(np.asarray(value, dtype=float), np.shape(x)).copy()


def fit_curve(n: np.ndarray, y: np.ndarray) -> SissoFit:
    """Fit ``y`` (mean candidate count) against ``n`` (structures)."""
    y = np.asarray(y, dtype=float)
    if np.ptp(y) < FLAT_RANGE:
        return SissoFit("", 0.0, 1.0, float(y[-1]))
    from TorchSisso import SissoModel

    data = pd.DataFrame({TARGET: y, FEATURE: np.asarray(n, dtype=float)})
    data = data.astype(np.float32)
    with contextlib.redirect_stdout(io.StringIO()):
        model = SissoModel(data, OPERATORS, n_expansion=N_EXPANSION,
                           n_term=N_TERM, k=K)
        rmse, equation, r2, _ = model.fit()
    return SissoFit(str(equation), float(rmse), float(r2), float(y[-1]))


def endpoint_residuals(n: np.ndarray, y: np.ndarray, cutoffs: list[int],
                       n_end: int) -> pd.DataFrame:
    """Fit the curve up to each cutoff and compare the prediction at ``n_end``.

    The residual is predicted minus observed count at ``n_end``.
    """
    n, y = np.asarray(n), np.asarray(y, dtype=float)
    y_end = float(y[n == n_end][0])
    rows = []
    for cut in cutoffs:
        keep = n <= cut
        fit = fit_curve(n[keep], y[keep])
        pred = float(fit.predict(n_end)[0])
        rows.append({"cutoff": cut, "equation": fit.equation, "r2": fit.r2,
                     "predicted": pred, "observed": y_end,
                     "residual": pred - y_end})
    return pd.DataFrame(rows)


def stays_below_from(residuals: pd.DataFrame, tol: float = RESIDUAL_TOL) -> float:
    """First cutoff from which ``|residual| < tol`` holds at every later cutoff."""
    ok = (residuals["residual"].abs() < tol).to_numpy()
    cutoffs = residuals["cutoff"].to_numpy()
    for k in range(len(ok)):
        if ok[k:].all():
            return float(cutoffs[k])
    return float("nan")


def additional_to_next_candidate(fit: SissoFit, n_end: int,
                                 observed_end: float) -> float:
    """Structures after ``n_end`` until the fit first exceeds the observed count by one.

    The target is ``floor(observed_end) + 1``. Returns ``inf`` when the fit
    undershoots the observed count, levels off, or does not reach the target
    within ``SEARCH_LIMIT`` structures.
    """
    from scipy.optimize import brentq

    def f(x: float) -> float:
        return float(fit.predict(x)[0])

    y_end = f(n_end)
    if not fit.equation or y_end < observed_end - RESIDUAL_TOL:
        return float("inf")
    # A fit that falls after the endpoint predicts no further candidate.
    if f(3 * n_end) < y_end - 0.1:
        return float("inf")
    target = np.floor(observed_end) + 1.0
    # No sign change in [n_end, SEARCH_LIMIT]: the fit never crosses the target
    # after the endpoint.
    if f(SEARCH_LIMIT) < target or f(n_end) >= target:
        return float("inf")
    root = brentq(lambda x: f(x) - target, n_end, SEARCH_LIMIT)
    return float(np.ceil(root - n_end))
