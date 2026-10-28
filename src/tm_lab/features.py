"""Entity-level features: velocity, amounts, diversity, flow, cross-border and graph."""

from __future__ import annotations

from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

BASE = [
    "n_txn", "n_in", "n_out", "total_in", "total_out", "mean_amt", "max_amt", "std_amt",
    "active_days", "txn_per_active_day", "max_7d_count", "max_7d_amount", "max_gap_days",
    "uniq_cp", "uniq_cp_in", "uniq_cp_out", "cp_diversity", "in_out_ratio", "round_share",
    "cross_border_share", "high_risk_share", "cash_share", "near_threshold_cash",
    "fast_forward_ratio", "fan_in_peak", "fan_out_peak",
]  # fmt: skip
BURST = ["burstiness", "gap_cv", "max_hour_count", "max_day_share", "n_channels", "cash_out_share"]
GRAPH = ["entity_degree", "pagerank", "in_cycle", "n_cycles"]
GRAPH2 = ["paths_in_2hop", "paths_out_2hop", "reciprocal_links", "triangles"]
PROPAGATION = ["cp_alert_share", "cp_alert_amount_share", "risk_prop_2hop"]
FEATURES = BASE + BURST + GRAPH + GRAPH2 + PROPAGATION
GROUPS = {
    "base": BASE + GRAPH,
    "+ burstiness": BASE + GRAPH + BURST,
    "+ 2-hop structure": BASE + GRAPH + BURST + GRAPH2,
    "+ risk propagation": FEATURES,
}
TRIANGLE_DEGREE_CAP = 300  # nodes above this are left out of the triangle count (quadratic blow-up)


def _sql_features(con, cfg: dict) -> pd.DataFrame:
    # Hub accounts (aggregators, processors) make the pass-through self-join and the rolling
    # windows quadratic, so entities above the cap are left out of the feature table.
    cap = cfg["scenarios"]["fan"].get("max_entity_txns")
    hubs = (
        f"WHERE entity NOT IN (SELECT entity FROM ledger GROUP BY entity HAVING count(*) > {cap})"
        if cap
        else ""
    )
    con.execute(f"CREATE OR REPLACE TEMP VIEW feat_ledger AS SELECT * FROM ledger {hubs}")
    T = cfg["reporting_threshold"]
    lo = cfg["scenarios"]["structuring"]["lower_band_frac"] * T
    hr = ", ".join(f"'{c}'" for c in cfg["scenarios"]["high_risk"]["countries"])
    w = cfg["scenarios"]["fan"]["window_days"]
    fwd_h = cfg["scenarios"]["passthrough"]["window_hours"]
    return con.execute(
        f"""
    WITH base AS (
      SELECT entity, count(*) AS n_txn,
        sum((direction = 'in')::INT) AS n_in, sum((direction = 'out')::INT) AS n_out,
        sum(CASE WHEN direction = 'in' THEN amount ELSE 0 END) AS total_in,
        sum(CASE WHEN direction = 'out' THEN amount ELSE 0 END) AS total_out,
        avg(amount) AS mean_amt, max(amount) AS max_amt,
        coalesce(stddev_samp(amount), 0) AS std_amt,
        count(DISTINCT CAST(ts AS DATE)) AS active_days, count(DISTINCT cp) AS uniq_cp,
        count(DISTINCT CASE WHEN direction = 'in' THEN cp END) AS uniq_cp_in,
        count(DISTINCT CASE WHEN direction = 'out' THEN cp END) AS uniq_cp_out,
        avg((amount >= 500 AND amount % 500 = 0)::INT) AS round_share,
        avg(is_cross_border::INT) AS cross_border_share,
        sum(CASE WHEN cp_country IN ({hr}) THEN amount ELSE 0 END) / sum(amount) AS high_risk_share,
        avg((channel = 'cash')::INT) AS cash_share, count(DISTINCT channel) AS n_channels,
        coalesce(avg(CASE WHEN direction = 'out' THEN (channel = 'cash')::INT END), 0)
          AS cash_out_share,
        sum((channel = 'cash' AND direction = 'in' AND amount >= {lo} AND amount < {T})::INT)
          AS near_threshold_cash
      FROM feat_ledger GROUP BY entity
    ),
    roll AS (
      SELECT entity, ts, amount, direction, cp,
        count(*) OVER r7 AS c7, sum(amount) OVER r7 AS s7,
        len(list_distinct(list(CASE WHEN direction = 'in' THEN cp END) OVER r7f)) AS fi,
        len(list_distinct(list(CASE WHEN direction = 'out' THEN cp END) OVER r7f)) AS fo,
        epoch(ts - lag(ts) OVER o) / 86400.0 AS gap
      FROM feat_ledger
      WINDOW o AS (PARTITION BY entity ORDER BY ts),
        r7 AS (PARTITION BY entity ORDER BY ts RANGE BETWEEN INTERVAL '{w} days' PRECEDING AND CURRENT ROW),
        r7f AS (PARTITION BY entity ORDER BY ts RANGE BETWEEN CURRENT ROW AND INTERVAL '{w} days' FOLLOWING)
    ), rollagg AS (
      SELECT entity, max(c7) AS max_7d_count, max(s7) AS max_7d_amount, coalesce(max(gap), 0) AS max_gap_days,
             max(fi) AS fan_in_peak, max(fo) AS fan_out_peak,
             coalesce(avg(gap), 0) AS gap_mean, coalesce(stddev_pop(gap), 0) AS gap_std
      FROM roll GROUP BY entity
    ), hourly AS (
      SELECT entity, max(h) AS max_hour_count, max(h)::DOUBLE / sum(h) AS hour_share
      FROM (SELECT entity, count(*) AS h FROM feat_ledger GROUP BY entity, date_trunc('hour', ts))
      GROUP BY entity
    ), daily AS (
      SELECT entity, max(d)::DOUBLE / sum(d) AS max_day_share
      FROM (SELECT entity, count(*) AS d FROM feat_ledger GROUP BY entity, CAST(ts AS DATE))
      GROUP BY entity
    ), fwd AS (
      SELECT i.entity, sum(least(1.0, coalesce(f.fwd, 0) / i.amount) * i.amount) / sum(i.amount)
             AS fast_forward_ratio
      FROM (SELECT * FROM feat_ledger WHERE direction = 'in' AND amount >= 1000) i
      LEFT JOIN (
        SELECT i2.txn_id, sum(o.amount) AS fwd
        FROM (SELECT * FROM feat_ledger WHERE direction = 'in' AND amount >= 1000) i2
        JOIN (SELECT * FROM feat_ledger WHERE direction = 'out') o
          ON o.entity = i2.entity AND o.cp <> i2.cp AND o.ts >= i2.ts
         AND o.ts <= i2.ts + INTERVAL '{fwd_h} hours'
        GROUP BY i2.txn_id
      ) f ON f.txn_id = i.txn_id
      GROUP BY i.entity
    )
    SELECT b.*, r.max_7d_count, r.max_7d_amount, r.max_gap_days, r.fan_in_peak, r.fan_out_peak,
           r.gap_mean, r.gap_std, h.max_hour_count, d.max_day_share,
           coalesce(f.fast_forward_ratio, 0) AS fast_forward_ratio
    FROM base b JOIN rollagg r USING (entity) JOIN hourly h USING (entity)
         JOIN daily d USING (entity) LEFT JOIN fwd f USING (entity)
    """
    ).df()


def _edges(txns: pd.DataFrame) -> pd.DataFrame:
    e = txns[txns["src_is_entity"] & txns["dst_is_entity"] & (txns["src"] != txns["dst"])]
    return e.groupby(["src", "dst"], as_index=False)["amount"].sum()


def _sparse(agg: pd.DataFrame, nodes: pd.Index) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Binary and amount-weighted adjacency (row = sender) over ``nodes``."""
    i = nodes.get_indexer(agg["src"])
    j = nodes.get_indexer(agg["dst"])
    n = len(nodes)
    w = sp.csr_matrix((agg["amount"].to_numpy(float), (i, j)), shape=(n, n))
    a = w.copy()
    a.data[:] = 1.0
    return a, w


def _row_normalise(m: sp.csr_matrix) -> sp.csr_matrix:
    d = np.asarray(m.sum(axis=1)).ravel()
    inv = np.divide(1.0, d, out=np.zeros_like(d), where=d > 0)
    return sp.diags(inv) @ m


def graph_features(
    txns: pd.DataFrame,
    entities: list[str],
    min_amount: float,
    max_len: int,
    alerted: set[str] | None = None,
) -> pd.DataFrame:
    """Structural and propagated graph features for ``entities``.

    Degree, PageRank and bounded short-cycle membership (networkx); 2-hop path counts, reciprocal
    links and directed triangles (sparse algebra); and, when ``alerted`` is given, how much
    rule-alert risk sits among an entity's counterparties (label-free propagation).
    """
    agg = _edges(txns)
    g = nx.DiGraph()
    g.add_nodes_from(entities)
    g.add_weighted_edges_from(agg[["src", "dst", "amount"]].itertuples(index=False))
    pr = nx.pagerank(g, weight="weight")
    e = txns[txns["src_is_entity"] & txns["dst_is_entity"] & (txns["src"] != txns["dst"])]
    big = e[e["amount"] >= min_amount].groupby(["src", "dst"]).size().reset_index()
    h = nx.DiGraph()
    h.add_edges_from(big[["src", "dst"]].itertuples(index=False))
    n_cycles: Counter[str] = Counter()
    for comp in nx.strongly_connected_components(h):
        if len(comp) < 2:
            continue
        for cyc in nx.simple_cycles(h.subgraph(comp), length_bound=max_len):
            n_cycles.update(cyc)

    nodes = pd.Index(list(g.nodes))
    a, w = _sparse(agg, nodes)
    indeg = np.asarray(a.sum(axis=0)).ravel()
    outdeg = np.asarray(a.sum(axis=1)).ravel()
    deg_ok = (indeg + outdeg) <= TRIANGLE_DEGREE_CAP
    keep = sp.diags(deg_ok.astype(float))
    core = keep @ a @ keep
    tri = np.asarray(((core @ core).multiply(core.T)).sum(axis=1)).ravel()
    recip = np.asarray(a.multiply(a.T).sum(axis=1)).ravel()
    sym = (a + a.T).tocsr()
    sym.data[:] = 1.0
    wsym = (w + w.T).tocsr()
    alert_vec = nodes.isin(alerted).astype(float) if alerted else np.zeros(len(nodes))
    p_cp = _row_normalise(sym)
    p_amt = _row_normalise(wsym)
    p1 = p_cp @ alert_vec
    ix = nodes.get_indexer(entities)

    out = pd.DataFrame({"entity": entities})
    out["entity_degree"] = (indeg + outdeg)[ix]
    out["pagerank"] = [pr[n] * len(entities) for n in entities]
    out["n_cycles"] = [n_cycles[n] for n in entities]
    out["in_cycle"] = (out["n_cycles"] > 0).astype(int)
    out["paths_in_2hop"] = (a.T @ indeg)[ix]
    out["paths_out_2hop"] = (a @ outdeg)[ix]
    out["reciprocal_links"] = recip[ix]
    out["triangles"] = tri[ix]
    out["cp_alert_share"] = p1[ix]
    out["cp_alert_amount_share"] = (p_amt @ alert_vec)[ix]
    out["risk_prop_2hop"] = (p_cp @ p1)[ix]
    return out


def build_features(
    con, txns: pd.DataFrame, cfg: dict, alerts: pd.DataFrame | None = None
) -> pd.DataFrame:
    """One row per entity; hubs above ``fan.max_entity_txns`` (if set) are excluded.

    ``alerts`` (rule output for the same period) feeds the risk-propagation features; it never
    carries labels.
    """
    f = _sql_features(con, cfg)
    ents = sorted(f["entity"])
    f["txn_per_active_day"] = f["n_txn"] / f["active_days"].clip(lower=1)
    f["cp_diversity"] = f["uniq_cp"] / f["n_txn"].clip(lower=1)
    f["in_out_ratio"] = np.log1p(f["total_in"]) - np.log1p(f["total_out"])
    f["burstiness"] = (f["gap_std"] - f["gap_mean"]) / (f["gap_std"] + f["gap_mean"]).replace(
        0, np.nan
    )
    f["gap_cv"] = f["gap_std"] / f["gap_mean"].replace(0, np.nan)
    rt = cfg["scenarios"]["roundtrip"]
    alerted = set(alerts["entity"]) if alerts is not None and len(alerts) else None
    g = graph_features(txns, ents, rt["min_amount"], rt["max_length"], alerted)
    out = pd.DataFrame({"entity": ents}).merge(f, on="entity", how="left").merge(g, on="entity")
    out[FEATURES] = out[FEATURES].fillna(0.0).astype(float)
    return out[["entity", *FEATURES]]
