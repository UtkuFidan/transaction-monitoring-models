import math

import pandas as pd
import pytest

from tm_lab import scenarios
from tm_lab.audit import sampling as s
from tm_lab.audit.analysis import has_typologies, own_entities


def test_clopper_pearson_zero_exceptions_matches_closed_form():
    lo, hi = s.clopper_pearson(0, 100, 0.95)
    assert lo == 0.0
    assert hi == pytest.approx(1 - 0.025 ** (1 / 100), rel=1e-9)


def test_clopper_pearson_known_value():
    lo, hi = s.clopper_pearson(10, 100, 0.95)  # textbook: 0.0490 to 0.1762
    assert (lo, hi) == (pytest.approx(0.0490, abs=5e-4), pytest.approx(0.1762, abs=5e-4))


def test_clopper_pearson_bounds_and_validation():
    assert s.clopper_pearson(5, 5)[1] == 1.0
    with pytest.raises(ValueError):
        s.clopper_pearson(6, 5)


def test_zero_failure_sample_size():
    assert s.zero_failure_sample_size(0.05, 0.95) == 59
    assert s.zero_failure_sample_size(0.01, 0.95) == 299
    assert s.zero_failure_sample_size(0.01, 0.99) == math.ceil(math.log(0.01) / math.log(0.99))


def test_btl_estimate_extrapolation():
    e = s.btl_estimate(population=1000, n=100, k=2)
    assert e["rate"] == 0.02 and e["est"] == 20
    assert e["est_lo"] <= 20 <= e["est_hi"]
    assert e["est_lo"] >= 2  # never below what was actually found


def test_btl_estimate_full_review_is_exact():
    e = s.btl_estimate(population=50, n=50, k=3)
    assert e["est"] == e["est_lo"] == e["est_hi"] == 3


def test_stratified_estimate_arithmetic():
    r = s.stratified_estimate([{"pop": 100, "n": 20, "k": 4}, {"pop": 900, "n": 90, "k": 1}])
    assert r["est"] == pytest.approx(100 * 0.2 + 900 * (1 / 90))
    assert r["found"] == 5 and r["est_lo"] <= r["est"] <= r["est_hi"]


def test_stratified_full_census_has_no_variance():
    r = s.stratified_estimate([{"pop": 10, "n": 10, "k": 3}])
    assert r["se"] == 0 and r["est"] == 3


def test_draw_sample_is_seeded_and_unique():
    ids = list(range(100))
    a, b = s.draw_sample(ids, 20, seed=1), s.draw_sample(ids, 20, seed=1)
    assert a == b and len(set(a)) == 20
    assert s.draw_sample(ids, 20, seed=2) != a
    assert len(s.draw_sample(ids, 500, seed=1)) == 100


def test_untyped_labels_count_every_labelled_entity_as_own():
    labels = pd.DataFrame({"entity": ["A1", "A2"], "typology": ["laundering"] * 2})
    typed = pd.DataFrame({"entity": ["A1", "A2"], "typology": ["structuring", "dormant"]})
    assert not has_typologies(labels) and has_typologies(typed)
    assert own_entities(labels, scenarios.structuring) == {"A1", "A2"}
    assert own_entities(typed, scenarios.structuring) == {"A1"}
