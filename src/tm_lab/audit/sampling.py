"""Sampling maths for above-/below-the-line (ATL/BTL) testing."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import beta, norm


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Exact two-sided binomial confidence interval for a proportion."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError("k must be between 0 and n")
    a = 1 - conf
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def zero_failure_sample_size(tolerable_rate: float, conf: float = 0.95) -> int:
    """Smallest n such that finding 0 exceptions proves rate <= tolerable at ``conf``."""
    if not 0 < tolerable_rate < 1:
        raise ValueError("tolerable_rate must be in (0, 1)")
    return math.ceil(math.log(1 - conf) / math.log(1 - tolerable_rate))


def detection_sample_size(true_rate: float, conf: float = 0.95) -> int:
    """Sample size giving ``conf`` probability of seeing at least one exception."""
    return zero_failure_sample_size(true_rate, conf)


def btl_estimate(population: int, n: int, k: int, conf: float = 0.95) -> dict:
    """Extrapolate k exceptions found in a simple random sample of n below-the-line items.

    When the whole population was reviewed (n >= population) the result is exact.
    """
    if n >= population:
        return {"rate": k / population, "lo": k / population, "hi": k / population,
                "est": k, "est_lo": k, "est_hi": k, "n": population, "k": k}  # fmt: skip
    lo, hi = clopper_pearson(k, n, conf)
    return {
        "rate": k / n, "lo": lo, "hi": hi,
        "est": population * k / n,
        "est_lo": max(float(k), population * lo),  # cannot be below what was actually found
        "est_hi": population * hi,
        "n": n, "k": k,
    }  # fmt: skip


def stratified_estimate(strata: list[dict], conf: float = 0.95) -> dict:
    """Stratified total from ``[{'pop': N_h, 'n': n_h, 'k': k_h}, ...]`` (normal approximation).

    Variance uses a (k + 0.5) / (n + 1) shrunk proportion so empty strata do not claim
    zero uncertainty, and the finite-population correction (1 - n_h / N_h).
    """
    total = var = 0.0
    found = 0
    for s in strata:
        pop, n, k = s["pop"], s["n"], s["k"]
        total += pop * k / n
        found += k
        p = (k + 0.5) / (n + 1)
        var += pop**2 * (1 - n / pop) * p * (1 - p) / n
    z = float(norm.ppf(1 - (1 - conf) / 2))
    se = math.sqrt(var)
    pop_all = sum(s["pop"] for s in strata)
    return {"est": total, "se": se, "est_lo": max(float(found), total - z * se),
            "est_hi": min(float(pop_all), total + z * se), "found": found}  # fmt: skip


def draw_sample(ids, n: int, seed: int) -> list:
    """Simple random sample without replacement."""
    ids = list(ids)
    rng = np.random.default_rng(seed)
    n = min(n, len(ids))
    return [ids[i] for i in rng.choice(len(ids), size=n, replace=False)]


def stratified_draw(df: pd.DataFrame, stratum_col: str, alloc: dict, seed: int) -> pd.DataFrame:
    """Draw ``alloc[stratum]`` rows from each stratum without replacement."""
    parts = []
    for i, (name, n) in enumerate(alloc.items()):
        grp = df[df[stratum_col] == name]
        parts.append(grp.sample(n=min(n, len(grp)), random_state=seed + i))
    return pd.concat(parts)
