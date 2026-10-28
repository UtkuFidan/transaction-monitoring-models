import json

import pytest

from tm_lab import scenarios
from tm_lab.data import connect
from tm_lab.scenarios._common import ALERT_COLS

from .conftest import make_txns


@pytest.mark.parametrize("module", scenarios.MODULES, ids=lambda m: m.SCENARIO_ID)
def test_scenario_detects_injected_typology(module, small, small_alerts):
    """Obvious (non-evasive) injected cases of the scenario's typology must be found."""
    lab = small["labels"]
    own = lab[lab["typology"].isin(module.TYPOLOGIES) & ~lab["evasive"]]
    assert len(own) >= 3
    hit = set(small_alerts.loc[small_alerts["scenario_id"] == module.SCENARIO_ID, "entity"])
    recall = len(hit & set(own["entity"])) / len(own)
    assert recall >= 0.8, f"{module.SCENARIO_ID} recall {recall:.2f}"


def test_alert_schema_and_facts(small_alerts):
    assert list(small_alerts.columns) == ALERT_COLS
    assert (small_alerts["period_end"] >= small_alerts["period_start"]).all()
    for f in small_alerts["facts"].head(50):
        assert isinstance(json.loads(f), dict)


def test_evasive_variants_sit_below_the_line(small, small_alerts):
    lab = small["labels"]
    ev = lab[lab["evasive"]]
    obv = lab[~lab["evasive"]]
    hit = set(small_alerts["entity"])
    assert len(set(ev["entity"]) & hit) / len(ev) < len(set(obv["entity"]) & hit) / len(obv)


def test_structuring_count_and_band_boundaries(cfg):
    dep = lambda d, a: (f"2025-05-0{d} 10:00", "CASH", "A1", a, "cash")  # noqa: E731
    three = make_txns([dep(1, 9000), dep(2, 9100), dep(3, 9200)])
    four = make_txns([dep(1, 9000), dep(2, 9100), dep(3, 9200), dep(4, 9300)])
    below_band = make_txns([dep(1, 7900), dep(2, 7900), dep(3, 7900), dep(4, 7900)])
    at_line = make_txns([dep(1, 10000), dep(2, 10000), dep(3, 10000), dep(4, 10000)])
    run = lambda t: scenarios.run_scenario(connect(t), cfg, "TM-01-STRUCT")  # noqa: E731
    assert run(three).empty
    assert run(below_band).empty
    assert run(at_line).empty  # at the reporting line is not "just under" it
    a = run(four)
    assert list(a["entity"]) == ["A1"]
    assert json.loads(a["facts"].iloc[0])["n_deposits"] == 4


def test_passthrough_requires_speed_and_ratio(cfg):
    def case(lag_h, ratio):
        return make_txns([
            ("2025-05-01 08:00", "X1", "A1", 10000), (f"2025-05-0{1 + lag_h // 24} 12:00", "A1", "X2", 10000 * ratio),
            ("2025-05-05 08:00", "X3", "A1", 10000), (f"2025-05-0{5 + lag_h // 24} 12:00", "A1", "X4", 10000 * ratio),
        ])  # fmt: skip

    run = lambda t: scenarios.run_scenario(connect(t), cfg, "TM-02-PASSTHRU")  # noqa: E731
    assert len(run(case(0, 0.98))) == 1
    assert run(case(0, 0.5)).empty
    assert run(case(72, 0.98)).empty


def test_roundtrip_needs_closing_loop(cfg):
    loop = make_txns([("2025-05-01 10:00", "A1", "A2", 20000), ("2025-05-02 10:00", "A2", "A3", 19800),
                      ("2025-05-03 10:00", "A3", "A1", 19600)])  # fmt: skip
    open_chain = make_txns([("2025-05-01 10:00", "A1", "A2", 20000), ("2025-05-02 10:00", "A2", "A3", 19800),
                            ("2025-05-03 10:00", "A3", "A4", 19600)])  # fmt: skip
    assert set(scenarios.run_scenario(connect(loop), cfg, "TM-04-ROUNDTRIP")["entity"]) == {
        "A1",
        "A2",
        "A3",
    }
    assert scenarios.run_scenario(connect(open_chain), cfg, "TM-04-ROUNDTRIP").empty
