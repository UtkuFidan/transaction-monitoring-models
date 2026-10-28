"""Evaluation design: an early and a late time window, and entity-disjoint train/val/test groups.

Models and rule thresholds are fitted on the early window using the train and validation
entities. They are scored on the late window using only entities the fitting never saw, so the
test set is out-of-time and out-of-entity at once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import labels_of

GROUPS = ("train", "val", "test")


@dataclass(frozen=True)
class Design:
    cut: pd.Timestamp | None  # first timestamp of the late window; None = one window
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)
    seed: int = 42


def median_cut(txns: pd.DataFrame, frac: float = 0.5) -> pd.Timestamp:
    """Timestamp that puts ``frac`` of the transactions in the early window."""
    return pd.Timestamp(txns["ts"].quantile(frac))


def split_windows(txns: pd.DataFrame, cut: pd.Timestamp | None):
    if cut is None:
        return txns, txns
    return txns[txns["ts"] < cut].reset_index(drop=True), txns[txns["ts"] >= cut].reset_index(
        drop=True
    )


def assign_groups(
    entities: pd.Series, fractions: tuple[float, float, float] = (0.6, 0.2, 0.2), seed: int = 42
) -> pd.Series:
    """Deterministic hash assignment, stable across windows, so an entity keeps its group."""
    if abs(sum(fractions) - 1) > 1e-9:
        raise ValueError("fractions must sum to 1")
    h = pd.util.hash_pandas_object(entities.astype(str) + f"|{seed}", index=False).to_numpy()
    u = (h % np.uint64(1_000_003)).astype(float) / 1_000_003
    edges = np.cumsum(fractions)
    idx = np.searchsorted(edges, u, side="right").clip(max=len(GROUPS) - 1)
    return pd.Series(np.array(GROUPS)[idx], index=entities.index, name="group")


def window_labels(txns: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Labels valid for one window: re-derived from transaction flags when the data carry them."""
    if "laundering" in txns:
        return labels_of(txns)
    return labels
