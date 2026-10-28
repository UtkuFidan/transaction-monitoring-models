"""TM-05 Dormant account: long silence followed by a sudden burst of value."""

from __future__ import annotations

from ._common import to_alerts

SCENARIO_ID = "TM-05-DORMANT"
TYPOLOGIES = ["dormant"]


def detect(con, cfg: dict, threshold: float):
    q = f"""
    WITH l AS (
      SELECT entity, ts, amount, lag(ts) OVER (PARTITION BY entity ORDER BY ts) AS prev_ts
      FROM ledger
    ), wake AS (
      SELECT entity, ts AS wake_ts, epoch(ts - prev_ts) / 86400.0 AS silent_days
      FROM l WHERE prev_ts IS NOT NULL
        AND ts - prev_ts >= INTERVAL '{cfg["dormant_days"]} days'
    )
    SELECT w.entity, w.wake_ts AS period_start,
           w.wake_ts + INTERVAL '{cfg["burst_days"]} days' AS period_end,
           round(w.silent_days, 0) AS silent_days, count(*) AS n_txn,
           round(sum(l.amount), 2) AS burst_total
    FROM wake w JOIN l ON l.entity = w.entity AND l.ts >= w.wake_ts
         AND l.ts <= w.wake_ts + INTERVAL '{cfg["burst_days"]} days'
    GROUP BY w.entity, w.wake_ts, w.silent_days
    HAVING count(*) >= {cfg["min_txn"]} AND sum(l.amount) >= {cfg["min_total"]}
    """
    return to_alerts(con.execute(q).df(), SCENARIO_ID, ["silent_days", "n_txn", "burst_total"])
