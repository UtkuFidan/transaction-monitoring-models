"""TM-03 Fan-in / fan-out: many counterparties within a short window, funds move on."""

from __future__ import annotations

import pandas as pd

from ._common import to_alerts

SCENARIO_ID = "TM-03-FAN"
TYPOLOGIES = ["fan_in", "fan_out"]


def _query(cfg: dict, side: str) -> str:
    w = cfg["window_days"]
    other = "out" if side == "in" else "in"
    # fan-in: the money leaves in the window after the first credit;
    # fan-out: the money arrived in the window before the first debit.
    lo, hi = (
        ("a.ts", f"a.ts + INTERVAL '{2 * w} days'")
        if side == "in"
        else (f"a.ts - INTERVAL '{w} days'", f"a.ts + INTERVAL '{w} days'")
    )
    return f"""
    WITH l AS (SELECT entity, ts, cp, amount, direction FROM ledger),
    a AS (
      SELECT entity, ts, amount,
             len(list_distinct(list(cp) OVER win)) AS n_cp, sum(amount) OVER win AS total
      FROM l WHERE direction = '{side}' AND entity IN (  -- prefilter keeps big ledgers cheap
        SELECT entity FROM ledger WHERE direction = '{side}'
        GROUP BY entity HAVING count(DISTINCT cp) >= {cfg["min_counterparties"]}
          AND count(*) <= {cfg.get("max_entity_txns") or 10**12})
      WINDOW win AS (PARTITION BY entity ORDER BY ts
                     RANGE BETWEEN CURRENT ROW AND INTERVAL '{w} days' FOLLOWING)
    ), trig AS (
      SELECT * FROM a WHERE n_cp >= {cfg["min_counterparties"]} AND total >= {cfg["min_total"]}
    )
    SELECT a.entity, '{side}' AS direction, a.ts AS period_start,
           a.ts + INTERVAL '{w} days' AS period_end, a.n_cp AS n_counterparties,
           round(a.total, 2) AS total, round(coalesce(sum(o.amount), 0), 2) AS counter_flow
    FROM trig a LEFT JOIN l o ON o.entity = a.entity AND o.direction = '{other}'
         AND o.ts >= {lo} AND o.ts <= {hi}
    GROUP BY a.entity, a.ts, a.n_cp, a.total
    HAVING coalesce(sum(o.amount), 0) >= {cfg["min_onward_ratio"]} * a.total
    """


def detect(con, cfg: dict, threshold: float):
    frames = [con.execute(_query(cfg, s)).df() for s in ("in", "out")]
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        return to_alerts(df, SCENARIO_ID, [])
    df["month"] = pd.to_datetime(df["period_start"]).dt.to_period("M")
    df = df.sort_values("n_counterparties", ascending=False).drop_duplicates(
        ["entity", "direction", "month"]
    )
    return to_alerts(df, SCENARIO_ID, ["direction", "n_counterparties", "total", "counter_flow"])
