"""Shared helpers for scenario modules."""

from __future__ import annotations

import json

import pandas as pd

ALERT_COLS = ["scenario_id", "entity", "period_start", "period_end", "facts"]


def to_alerts(df: pd.DataFrame, scenario_id: str, fact_cols: list[str]) -> pd.DataFrame:
    """Turn a result frame (entity, period_start, period_end + fact columns) into alerts."""
    if df.empty:
        return pd.DataFrame(columns=ALERT_COLS)
    out = pd.DataFrame(
        {
            "scenario_id": scenario_id,
            "entity": df["entity"].to_numpy(),
            "period_start": pd.to_datetime(df["period_start"]).to_numpy(),
            "period_end": pd.to_datetime(df["period_end"]).to_numpy(),
        }
    )
    out["facts"] = [
        json.dumps({c: _py(r[c]) for c in fact_cols}, default=str) for r in df.to_dict("records")
    ]
    return out[ALERT_COLS]


def _py(v):
    return v.item() if hasattr(v, "item") else v
