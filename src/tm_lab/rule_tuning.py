"""Threshold tuning for the rule scenarios.

Each scenario is tuned on its own: maximise the suspicious entities it catches on the tuning
entities, subject to an alert budget (a share of the scenario's alert volume at the current
thresholds). The search is Optuna TPE over a few parameters per scenario; the reporting line
and the round-trip length stay as configured. Results are evaluated on separate entities in a
later time window by the caller.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable

import duckdb
import pandas as pd

from . import scenarios as S
from .audit.analysis import own_entities

# name -> (low, high, kind); kind is "int", "float" or "log" (log-uniform float)
SPACE: dict[str, dict[str, tuple]] = {
    "TM-01-STRUCT": {"lower_band_frac": (0.1, 0.95, "float"), "window_days": (1, 30, "int"),
                     "min_count": (2, 10, "int"), "min_total": (2000, 60000, "log")},
    "TM-02-PASSTHRU": {"min_in_amount": (500, 30000, "log"), "window_hours": (6, 96, "int"),
                       "min_forward_ratio": (0.5, 1.0, "float"), "min_events": (1, 6, "int")},
    "TM-03-FAN": {"window_days": (1, 30, "int"), "min_counterparties": (4, 40, "int"),
                  "min_total": (500, 60000, "log"), "min_onward_ratio": (0.2, 1.0, "float")},
    "TM-04-ROUNDTRIP": {"max_days": (1, 30, "int"), "amount_tolerance": (0.01, 0.3, "float"),
                        "min_amount": (500, 10000, "log")},
    "TM-05-DORMANT": {"dormant_days": (30, 180, "int"), "burst_days": (3, 30, "int"),
                      "min_txn": (2, 15, "int"), "min_total": (1000, 60000, "log")},
    "TM-06-HIGHRISK": {"window_days": (7, 90, "int"), "min_total": (1000, 100000, "log"),
                       "min_count": (1, 6, "int")},
}  # fmt: skip


def _suggest(trial, space: dict) -> dict:
    out = {}
    for k, (lo, hi, kind) in space.items():
        if kind == "int":
            out[k] = trial.suggest_int(k, lo, hi)
        elif kind == "log":
            out[k] = round(trial.suggest_float(k, lo, hi, log=True), 2)
        else:
            out[k] = round(trial.suggest_float(k, lo, hi), 3)
    return out


def tune_scenario(
    con,
    cfg: dict,
    module,
    pool: set,
    labels: pd.DataFrame,
    budget_frac: float,
    seconds: float,
    seed: int = 42,
    log: Callable[[str], None] | None = None,
    max_trial_seconds: float = 40.0,
) -> dict:
    """Tune one scenario on the entities in ``pool``. Returns params and tuning-split counts."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sid = module.SCENARIO_ID
    target = own_entities(labels, module) & pool
    base = {k: cfg["scenarios"][S.CONFIG_KEY[sid]][k] for k in SPACE[sid]}
    base_ents = set(S.run_scenario(con, cfg, sid)["entity"]) & pool
    budget = max(math.ceil(budget_frac * len(base_ents)), 5)
    cache: dict[tuple, tuple[int, int]] = {}

    def evaluate(params: dict) -> tuple[int, int]:
        key = tuple(sorted(params.items()))
        if key not in cache:
            timer = threading.Timer(max_trial_seconds, con.interrupt)
            timer.start()
            try:
                ents = (
                    set(S.run_scenario(con, S.with_params(cfg, sid, params), sid)["entity"]) & pool
                )
                cache[key] = (len(ents), len(ents & target))
            except (
                duckdb.InterruptException
            ):  # too slow to be a practical rule: treat as over budget
                cache[key] = (10**9, 0)
            finally:
                timer.cancel()
        return cache[key]

    def objective(trial) -> float:
        n, hits = evaluate(_suggest(trial, SPACE[sid]))
        return hits * min(1.0, budget / n) if n else 0.0  # over budget: scaled down

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.enqueue_trial(base)  # the current thresholds are always a candidate
    t0 = time.time()
    study.optimize(objective, timeout=seconds)
    feasible = [(hits, -n, dict(k)) for k, (n, hits) in cache.items() if 0 < n <= budget]
    if feasible:
        hits, neg_n, params = max(feasible, key=lambda t: (t[0], t[1]))
        n = -neg_n
    else:  # nothing fits the budget: keep current thresholds
        params, (n, hits) = base, evaluate(base)
    if log:
        log(f"{sid}: {len(cache)} configs in {time.time() - t0:.0f}s, budget {budget}, "
            f"tuning hits {hits} of {len(target)} with {n} entities (current {len(base_ents)})")  # fmt: skip
    return {
        "scenario": sid, "budget_frac": budget_frac, "budget": budget, "params": params,
        "current_params": base, "tuning": {"entities": n, "hits": hits, "targets": len(target),
                                           "current_entities": len(base_ents)},
        "n_configs": len(cache),
    }  # fmt: skip


def apply(cfg: dict, tuned: list[dict]) -> dict:
    out = cfg
    for t in tuned:
        out = S.with_params(out, t["scenario"], t["params"])
    return out
