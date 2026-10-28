"""TM-02 Rapid pass-through: credits forwarded almost in full within hours."""

from __future__ import annotations

from ._common import to_alerts

SCENARIO_ID = "TM-02-PASSTHRU"
TYPOLOGIES = ["passthrough"]


def detect(con, cfg: dict, threshold: float):
    q = f"""
    WITH i AS (SELECT txn_id, entity, ts, cp, amount FROM ledger
               WHERE direction = 'in' AND amount >= {cfg["min_in_amount"]}),
    o AS (SELECT entity, ts, cp, amount FROM ledger WHERE direction = 'out'),
    ev AS (
      SELECT i.entity, i.ts, i.amount AS credit, sum(o.amount) AS forwarded,
             min(epoch(o.ts - i.ts)) / 3600.0 AS first_lag_h
      FROM i JOIN o ON o.entity = i.entity AND o.cp <> i.cp
        AND o.ts >= i.ts AND o.ts <= i.ts + INTERVAL '{cfg["window_hours"]} hours'
      GROUP BY i.entity, i.txn_id, i.ts, i.amount
      HAVING sum(o.amount) >= {cfg["min_forward_ratio"]} * i.amount
    ), w AS (
      SELECT *, count(*) OVER win AS n_events, sum(credit) OVER win AS total_in,
             min(ts) OVER win AS w_start
      FROM ev
      WINDOW win AS (PARTITION BY entity ORDER BY ts
                     RANGE BETWEEN INTERVAL '{cfg["lookback_days"]} days' PRECEDING AND CURRENT ROW)
    )
    SELECT entity, min(w_start) AS period_start, max(ts) AS period_end,
           max(n_events) AS n_events, round(max(total_in), 2) AS total_in,
           round(min(first_lag_h), 1) AS fastest_forward_h
    FROM w WHERE n_events >= {cfg["min_events"]}
    GROUP BY entity, date_trunc('month', ts)
    """
    return to_alerts(
        con.execute(q).df(), SCENARIO_ID, ["n_events", "total_in", "fastest_forward_h"]
    )
