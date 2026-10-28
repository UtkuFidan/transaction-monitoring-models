"""TM-04 Round-tripping: funds return to the origin through 2-4 hops within days."""

from __future__ import annotations

import pandas as pd

from ._common import to_alerts

SCENARIO_ID = "TM-04-ROUNDTRIP"
TYPOLOGIES = ["roundtrip"]


def detect(con, cfg: dict, threshold: float):
    d, tol, mx = cfg["max_days"], cfg["amount_tolerance"], cfg["max_length"]
    con.execute(
        f"""CREATE OR REPLACE TEMP TABLE e AS
        SELECT src, dst, ts, amount FROM txns
        WHERE src_is_entity AND dst_is_entity AND src <> dst AND amount >= {cfg["min_amount"]}"""
    )

    def hop(prev: str, cur: str, nxt_dst: str | None) -> str:
        cond = (
            f"{cur}.src = {prev}.dst AND {cur}.ts >= {prev}.ts "
            f"AND {cur}.ts <= a.ts + INTERVAL '{d} days' "
            f"AND {cur}.amount BETWEEN a.amount * {1 - tol} AND a.amount * {1 + tol}"
        )
        return cond

    sels = []
    # cycles of length 2..max_length; the closing hop must land on the first node
    for length in range(2, mx + 1):
        names = ["a"] + [chr(ord("b") + i) for i in range(length - 1)]
        joins = ""
        for p, c in zip(names[:-1], names[1:], strict=True):
            joins += f" JOIN e {c} ON {hop(p, c, None)}"
            if c != names[-1]:
                joins += f" AND {c}.dst <> a.src"
        last = names[-1]
        members = ", ".join(f"{n}.src" for n in names)
        sels.append(
            f"SELECT a.ts AS t0, {last}.ts AS t1, a.amount AS amt, {length} AS length, "
            f"list_value({members}) AS members FROM e a{joins} WHERE {last}.dst = a.src"
        )
    df = con.execute(" UNION ALL ".join(sels)).df()
    if df.empty:
        return to_alerts(df, SCENARIO_ID, [])
    rows = []
    for r in df.itertuples():
        path = " > ".join(r.members) + " > " + r.members[0]
        for m in r.members:
            rows.append((m, r.t0, r.t1, r.length, round(float(r.amt), 2), path))
    big = pd.DataFrame(
        rows, columns=["entity", "period_start", "period_end", "cycle_length", "first_hop", "cycle"]
    )
    big["month"] = big["period_start"].dt.to_period("M")
    big = big.sort_values("period_start").drop_duplicates(["entity", "month"])
    return to_alerts(big, SCENARIO_ID, ["cycle_length", "first_hop", "cycle"])
