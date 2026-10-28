"""TM-06 High-risk jurisdiction: material value to/from listed countries."""

from __future__ import annotations

from ._common import to_alerts

SCENARIO_ID = "TM-06-HIGHRISK"
TYPOLOGIES = ["high_risk"]


def detect(con, cfg: dict, threshold: float):
    countries = ", ".join(f"'{c}'" for c in cfg["countries"])
    q = f"""
    WITH h AS (
      SELECT entity, ts, amount, cp_country FROM ledger WHERE cp_country IN ({countries})
    ), w AS (
      SELECT entity, ts, count(*) OVER win AS n_txn, sum(amount) OVER win AS total,
             len(list_distinct(list(cp_country) OVER win)) AS n_countries,
             min(ts) OVER win AS w_start
      FROM h
      WINDOW win AS (PARTITION BY entity ORDER BY ts
                     RANGE BETWEEN INTERVAL '{cfg["window_days"]} days' PRECEDING AND CURRENT ROW)
    )
    SELECT entity, min(w_start) AS period_start, max(ts) AS period_end,
           max(n_txn) AS n_txn, round(max(total), 2) AS total_hr_value,
           max(n_countries) AS n_countries
    FROM w WHERE n_txn >= {cfg["min_count"]} AND total >= {cfg["min_total"]}
    GROUP BY entity, date_trunc('month', ts)
    """
    return to_alerts(con.execute(q).df(), SCENARIO_ID, ["n_txn", "total_hr_value", "n_countries"])
