"""Rule-based detection scenarios, one module per typology."""

from __future__ import annotations

import copy

import pandas as pd

from . import dormant, fan, high_risk, passthrough, roundtrip, structuring
from ._common import ALERT_COLS

MODULES = [structuring, passthrough, fan, roundtrip, dormant, high_risk]
CONFIG_KEY = {
    "TM-01-STRUCT": "structuring",
    "TM-02-PASSTHRU": "passthrough",
    "TM-03-FAN": "fan",
    "TM-04-ROUNDTRIP": "roundtrip",
    "TM-05-DORMANT": "dormant",
    "TM-06-HIGHRISK": "high_risk",
}
COVERS = {m.SCENARIO_ID: m.TYPOLOGIES for m in MODULES}


def run_scenario(con, cfg: dict, scenario_id: str) -> pd.DataFrame:
    mod = next(m for m in MODULES if m.SCENARIO_ID == scenario_id)
    return mod.detect(con, cfg["scenarios"][CONFIG_KEY[scenario_id]], cfg["reporting_threshold"])


def run_all(con, cfg: dict) -> pd.DataFrame:
    frames = [run_scenario(con, cfg, m.SCENARIO_ID) for m in MODULES]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=ALERT_COLS)
    return pd.concat(frames, ignore_index=True)


def with_override(cfg: dict, scenario_id: str, key: str, value) -> dict:
    new = copy.deepcopy(cfg)
    new["scenarios"][CONFIG_KEY[scenario_id]][key] = value
    return new


def with_params(cfg: dict, scenario_id: str, params: dict) -> dict:
    """Copy of ``cfg`` with several parameters of one scenario replaced."""
    new = copy.deepcopy(cfg)
    new["scenarios"][CONFIG_KEY[scenario_id]].update(params)
    return new
