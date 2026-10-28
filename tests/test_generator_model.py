import numpy as np

from tm_lab.generator import generate
from tm_lab.model import top_k_for_recall


def test_generator_is_deterministic_and_labelled():
    a = generate(500, 200, seed=3)
    b = generate(500, 200, seed=3)
    assert a[0].equals(b[0]) and a[2].equals(b[2])
    assert not generate(500, 200, seed=4)[0].equals(a[0])
    labels = a[2]
    assert set(labels["typology"]) == {
        "structuring",
        "passthrough",
        "fan_in",
        "fan_out",
        "roundtrip",
        "dormant",
        "high_risk",
    }
    assert labels["entity"].is_unique and labels["evasive"].any()
    assert (a[0]["amount"] > 0).all() and a[0]["ts"].is_monotonic_increasing


def test_top_k_for_recall():
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    y = np.array([1, 0, 1, 0, 1])
    assert top_k_for_recall(score, y, 0.5) == 3  # need 2 of 3 positives -> rank 3
    assert top_k_for_recall(score, y, 1.0) == 5
    assert top_k_for_recall(score, y, 0.3) == 1
