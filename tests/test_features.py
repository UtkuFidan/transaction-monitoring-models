import numpy as np

from tm_lab.data import connect
from tm_lab.features import FEATURES, build_features, graph_features

from .conftest import make_txns

ROWS = [
    ("2025-05-01 10:00", "CASH", "A1", 9500, "cash"),
    ("2025-05-01 12:00", "CASH", "A1", 9000, "cash"),
    ("2025-05-02 09:00", "A1", "X1", 1000, "wire", "H1"),
    ("2025-05-03 09:00", "A1", "X2", 500, "wire"),
    ("2025-05-04 09:00", "X3", "A1", 1234.5, "wire"),
]


def test_entity_features(cfg):
    t = make_txns(ROWS)
    f = build_features(connect(t), t, cfg).set_index("entity").loc["A1"]
    assert f["n_txn"] == 5 and f["n_in"] == 3 and f["n_out"] == 2
    assert f["total_in"] == 9500 + 9000 + 1234.5 and f["total_out"] == 1500
    assert f["uniq_cp"] == 4
    assert np.isclose(f["round_share"], 4 / 5)  # 1000 and 500 are round, 9500 too (multiple of 500)
    assert np.isclose(f["cash_share"], 2 / 5)
    assert f["near_threshold_cash"] == 2
    assert np.isclose(f["cross_border_share"], 1 / 5)
    assert np.isclose(f["high_risk_share"], 1000 / sum(r[3] for r in ROWS))
    assert np.isclose(f["in_out_ratio"], np.log1p(19734.5) - np.log1p(1500))
    assert f["max_7d_count"] == 5


def test_feature_columns_complete(cfg):
    t = make_txns(ROWS)
    out = build_features(connect(t), t, cfg)
    assert list(out.columns) == ["entity", *FEATURES]
    assert not out.isna().any().any()


def test_graph_cycle_membership():
    t = make_txns([("2025-05-01", "A1", "A2", 5000), ("2025-05-02", "A2", "A3", 5000),
                   ("2025-05-03", "A3", "A1", 5000), ("2025-05-03", "A3", "A4", 5000)])  # fmt: skip
    g = graph_features(t, ["A1", "A2", "A3", "A4"], 2000, 4).set_index("entity")
    assert list(g["in_cycle"]) == [1, 1, 1, 0]
    assert g.loc["A3", "entity_degree"] == 3
    small = graph_features(t, ["A1", "A2", "A3", "A4"], 6000, 4).set_index("entity")
    assert small["in_cycle"].sum() == 0  # edges below the amount floor are ignored


def test_hub_accounts_are_excluded_when_capped(cfg):
    rows = [(f"2025-05-01 {h:02d}:00", f"A{h}", "A99", 100) for h in range(1, 7)]
    t = make_txns(rows)
    capped = {
        **cfg,
        "scenarios": {**cfg["scenarios"], "fan": {**cfg["scenarios"]["fan"], "max_entity_txns": 3}},
    }
    assert "A99" in set(build_features(connect(t), t, cfg)["entity"])
    assert "A99" not in set(build_features(connect(t), t, capped)["entity"])


def test_burstiness_and_hourly_peaks(cfg):
    # three transfers within one minute, then two quiet weeks, then one more
    rows = [(f"2025-05-01 10:00:{s:02d}", "A1", f"X{s}", 100) for s in (0, 20, 40)]
    rows += [("2025-05-15 10:00:00", "A1", "X9", 100)]
    t = make_txns(rows)
    f = build_features(connect(t), t, cfg).set_index("entity").loc["A1"]
    assert f["max_hour_count"] == 3
    assert np.isclose(f["max_day_share"], 3 / 4)
    assert f["burstiness"] > 0.1  # gaps are very uneven
    regular = make_txns([(f"2025-05-{d:02d} 10:00", "A2", f"X{d}", 100) for d in range(1, 9)])
    g = build_features(connect(regular), regular, cfg).set_index("entity").loc["A2"]
    assert np.isclose(g["burstiness"], -1.0) and np.isclose(g["gap_cv"], 0.0)  # metronome


def test_two_hop_reciprocal_and_triangle_features():
    t = make_txns([
        ("2025-05-01", "A1", "A2", 100), ("2025-05-01", "A3", "A2", 100),   # two senders into A2
        ("2025-05-02", "A4", "A1", 100), ("2025-05-02", "A5", "A1", 100),  # two senders into A1
        ("2025-05-03", "A2", "A6", 100),                                    # A2 moves on
        ("2025-05-03", "A6", "A2", 100),                                    # ... and A6 pays back
        ("2025-05-04", "A7", "A8", 100), ("2025-05-04", "A8", "A9", 100), ("2025-05-04", "A9", "A7", 100),
    ])  # fmt: skip
    ents = [f"A{i}" for i in range(1, 10)]
    g = graph_features(t, ents, 1e9, 3).set_index("entity")
    assert g.loc["A2", "paths_in_2hop"] == 3  # in-neighbours A1 (2 senders), A3 (0), A6 (1: A2)
    assert g.loc["A4", "paths_out_2hop"] == 1  # A4 -> A1 -> A2
    assert g.loc["A2", "reciprocal_links"] == 1 and g.loc["A6", "reciprocal_links"] == 1
    assert g.loc["A7", "triangles"] == 1 and g.loc["A2", "triangles"] == 0


def test_risk_propagation_uses_counterparty_alerts_only():
    t = make_txns([("2025-05-01", "A1", "A2", 100), ("2025-05-01", "A3", "A2", 300),
                   ("2025-05-02", "A2", "A4", 100)])  # fmt: skip
    ents = ["A1", "A2", "A3", "A4"]
    g = graph_features(t, ents, 1e9, 3, alerted={"A1"}).set_index("entity")
    assert np.isclose(g.loc["A2", "cp_alert_share"], 1 / 3)  # one of three counterparties alerted
    assert np.isclose(g.loc["A2", "cp_alert_amount_share"], 100 / 500)
    assert g.loc["A1", "cp_alert_share"] == 0  # an entity's own alert does not count
    assert g.loc["A4", "risk_prop_2hop"] > 0  # A4 -> A2 -> A1: two hops from the alert
    none = graph_features(t, ents, 1e9, 3).set_index("entity")
    assert (none[["cp_alert_share", "risk_prop_2hop"]] == 0).all().all()
