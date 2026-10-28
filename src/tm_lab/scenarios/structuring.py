"""TM-01 Structuring: repeated cash deposits just under the reporting threshold."""

from __future__ import annotations

from ._common import to_alerts

SCENARIO_ID = "TM-01-STRUCT"
TYPOLOGIES = ["structuring"]


def detect(con, cfg: dict, threshold: float):
    lo = cfg["lower_band_frac"] * threshold
    q = f"""
    WITH d AS (
      SELECT entity, ts, amount FROM ledger
      WHERE direction = 'in' AND channel = 'cash' AND amount >= {lo} AND amount < {threshold}
    ), w AS (
      SELECT entity, ts,
             count(*) OVER win AS n_deposits, sum(amount) OVER win AS total,
             min(ts) OVER win AS w_start
      FROM d
      WINDOW win AS (PARTITION BY entity ORDER BY ts
                     RANGE BETWEEN INTERVAL '{cfg["window_days"]} days' PRECEDING AND CURRENT ROW)
    )
    SELECT entity, min(w_start) AS period_start, max(ts) AS period_end,
           max(n_deposits) AS n_deposits, round(max(total), 2) AS total_deposited,
           {lo} AS band_low, {threshold} AS band_high
    FROM w WHERE n_deposits >= {cfg["min_count"]} AND total >= {cfg["min_total"]}
    GROUP BY entity, date_trunc('month', ts)
    """
    return to_alerts(
        con.execute(q).df(), SCENARIO_ID, ["n_deposits", "total_deposited", "band_low", "band_high"]
    )
