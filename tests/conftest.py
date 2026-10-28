import pandas as pd
import pytest

from tm_lab import scenarios
from tm_lab.data import TXN_COLS, connect, default_config
from tm_lab.generator import generate


@pytest.fixture(scope="session")
def cfg():
    return default_config()


@pytest.fixture(scope="session")
def small():
    txns, accounts, labels = generate(n_accounts=600, days=240, seed=7)
    con = connect(txns, accounts)
    return {"txns": txns, "labels": labels, "con": con}


@pytest.fixture(scope="session")
def small_alerts(small, cfg):
    return scenarios.run_all(small["con"], cfg)


def make_txns(rows):
    """rows: (ts, src, dst, amount[, channel[, dst_country]]) -> frame in the ledger schema."""
    recs = []
    for i, r in enumerate(rows, 1):
        ts, src, dst, amt = r[:4]
        channel = r[4] if len(r) > 4 else "wire"
        dc = r[5] if len(r) > 5 else "DM"
        recs.append({
            "txn_id": i, "ts": pd.Timestamp(ts), "src": src, "dst": dst, "amount": float(amt),
            "channel": channel, "src_country": "DM", "dst_country": dc,
            "is_cross_border": dc != "DM", "src_is_entity": src.startswith("A"),
            "dst_is_entity": dst.startswith("A"),
        })  # fmt: skip
    return pd.DataFrame(recs)[TXN_COLS]
