import numpy as np
import pandas as pd
import pytest

from tm_lab import model as M
from tm_lab import rule_tuning as RT
from tm_lab import saml, scenarios
from tm_lab.audit.analysis import has_typologies
from tm_lab.data import connect, labels_of
from tm_lab.experiment import run_experiment
from tm_lab.generator import generate
from tm_lab.windows import Design, assign_groups, median_cut, split_windows, window_labels


def test_assign_groups_is_deterministic_and_close_to_fractions():
    ents = pd.Series([f"E{i}" for i in range(20000)])
    a = assign_groups(ents, (0.6, 0.2, 0.2), 1)
    b = assign_groups(ents, (0.6, 0.2, 0.2), 1)
    assert a.equals(b) and not a.equals(assign_groups(ents, (0.6, 0.2, 0.2), 2))
    share = a.value_counts(normalize=True)
    assert abs(share["train"] - 0.6) < 0.02 and abs(share["test"] - 0.2) < 0.02
    with pytest.raises(ValueError):
        assign_groups(ents, (0.5, 0.2, 0.2))


def test_windows_partition_transactions_and_relabel():
    t = pd.DataFrame({
        "ts": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-10", "2025-01-11"]),
        "src": ["A1", "A2", "A3", "A1"], "dst": ["A2", "A3", "A1", "A4"],
        "laundering": [0, 1, 1, 0],
    })  # fmt: skip
    cut = pd.Timestamp("2025-01-05")
    early, late = split_windows(t, cut)
    assert len(early) + len(late) == len(t) and early.ts.max() < cut <= late.ts.min()
    assert set(window_labels(early, None)["entity"]) == {"A2", "A3"}
    assert set(window_labels(late, None)["entity"]) == {"A3", "A1"}
    assert median_cut(t) == pd.Timestamp("2025-01-06")
    one, same = split_windows(t, None)
    assert len(one) == len(same) == 4


def test_saml_adapter_maps_schema_and_labels(tmp_path):
    csv = tmp_path / "SAML-D.csv"
    csv.write_text(
        "Time,Date,Sender_account,Receiver_account,Amount,Payment_currency,Received_currency,"
        "Sender_bank_location,Receiver_bank_location,Payment_type,Is_laundering,Laundering_type\n"
        "10:35:19,2022-10-07,1,2,1459.15,UK pounds,UK pounds,UK,UK,Cash Deposit,0,Normal_Cash_Deposits\n"
        "11:00:00,2022-10-07,2,3,9000,UK pounds,Dirham,UK,UAE,Cross-border,1,Layered_Fan_In\n"
        "12:00:00,2022-10-08,3,4,500,Euro,Euro,Spain,Spain,ACH,0,Normal_Fan_Out\n"
    )
    txns, accounts, labels = saml.convert(csv)
    assert list(txns["channel"]) == ["cash", "cross-border", "ach"]
    assert list(txns["is_cross_border"]) == [False, True, False]
    assert txns["ts"].iloc[1] == pd.Timestamp("2022-10-07 11:00:00")
    assert len(accounts) == 4
    assert set(labels["entity"]) == {"2", "3"} and set(labels["typology"]) == {"Layered_Fan_In"}
    assert not has_typologies(labels)
    assert set(labels_of(txns)["entity"]) == {"2", "3"}
    sub, _, _ = saml.convert(csv, account_frac=0.5, seed=1)
    assert len(sub) <= len(txns)


def test_bootstrap_interval_contains_point_estimate():
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=2000) < 0.05).astype(int)
    good = y + rng.normal(0, 0.6, 2000)
    rules = (rng.uniform(size=2000) < 0.2).astype(int) | y
    out = M.bootstrap(y, {"m": good}, {"r": rules}, n_boot=100, seed=1)
    lo, hi = out["pr_auc_ci"]["m"]
    assert lo < M.evaluate_scores(y, good)["pr_auc"] < hi
    rlo, rhi = out["reduction_ci"]["m|r"]
    assert rlo < rhi


def test_blend_weights_prefer_the_informative_score():
    rng = np.random.default_rng(3)
    y = (rng.uniform(size=3000) < 0.1).astype(int)
    gbm = y * 2.0 + rng.normal(size=3000)
    noise = rng.normal(size=3000)
    w, ap = M.tune_blend(y, gbm, noise, np.zeros(3000))
    assert w[0] > w[1] and abs(sum(w) - 1) < 1e-9 and ap > 0.3


def test_downsample_keeps_every_positive():
    x = np.arange(1000).reshape(-1, 1).astype(float)
    y = np.zeros(1000, dtype=int)
    y[::50] = 1
    xs, ys = M.downsample_negatives(x, y, 200, 0)
    assert len(ys) == 200 and ys.sum() == y.sum()


def test_rule_tuning_never_exceeds_budget_and_is_applied(small, cfg):
    labels = small["labels"]
    pool = set(small["con"].execute("SELECT DISTINCT entity FROM ledger").df()["entity"])
    module = next(m for m in scenarios.MODULES if m.SCENARIO_ID == "TM-01-STRUCT")
    res = RT.tune_scenario(small["con"], cfg, module, pool, labels, 1.0, seconds=5)
    assert res["tuning"]["entities"] <= res["budget"]
    assert res["tuning"]["hits"] >= 0
    tuned = RT.apply(cfg, [res])
    assert tuned["scenarios"]["structuring"]["min_count"] == res["params"]["min_count"]
    assert cfg["scenarios"]["structuring"]["min_count"] == 4  # the input config is untouched


def test_experiment_end_to_end_on_a_small_set(cfg):
    txns, _, labels = generate(n_accounts=500, days=200, seed=11)
    res = run_experiment(
        txns, labels, cfg, Design(cut=None, fractions=(0.5, 0.2, 0.3)),
        tune_seconds=3, rule_seconds=2, n_boot=20, log=lambda *_: None,
    )  # fmt: skip
    m, scores = res["metrics"], res["scores"]
    pops = m["design"]["populations"]
    assert pops["test"]["entities"] == len(scores) == m["n_entities"]
    assert 0 < m["pr_auc"]["gradient_boosting"] <= 1
    assert m["pr_auc"]["gradient_boosting"] > m["pr_auc"]["rules_binary"]
    assert set(m["tuning"]) == {"lightgbm", "xgboost", "catboost"}
    assert m["selected_model"] in {"lightgbm", "xgboost", "catboost", "blend"}
    assert m["rules"]["untuned"]["recall"] >= 0
    assert (scores["reason_codes"].str.len() > 0).any()
    # test entities are disjoint from the fitting entities
    groups = assign_groups(scores["entity"], (0.5, 0.2, 0.3), 42)
    assert (groups == "test").all()
    # replaying stored parameters skips the search and reproduces the model scores
    again = run_experiment(
        txns, labels, cfg, Design(cut=None, fractions=(0.5, 0.2, 0.3)), tuned=res["tuned"],
        n_boot=20, log=lambda *_: None,
    )  # fmt: skip
    assert np.allclose(again["scores"]["gbm_score"], scores["gbm_score"])


def test_connect_keeps_extra_columns_out_of_ledger():
    t = pd.DataFrame({
        "txn_id": [1], "ts": [pd.Timestamp("2025-01-01")], "src": ["A1"], "dst": ["A2"],
        "amount": [10.0], "channel": ["wire"], "src_country": ["DM"], "dst_country": ["DM"],
        "is_cross_border": [False], "src_is_entity": [True], "dst_is_entity": [True],
        "laundering": [1],
    })  # fmt: skip
    con = connect(t)
    assert "laundering" not in con.execute("SELECT * FROM ledger").df().columns
