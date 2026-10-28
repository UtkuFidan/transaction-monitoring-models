"""Loading, saving and the DuckDB ledger shared by scenarios and features."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import yaml

TXN_COLS = [
    "txn_id", "ts", "src", "dst", "amount", "channel", "src_country", "dst_country",
    "is_cross_border", "src_is_entity", "dst_is_entity",
]  # fmt: skip


def default_config() -> dict:
    return yaml.safe_load((Path(__file__).parent / "thresholds.yaml").read_text())


def load_config(path: str | Path | None = None) -> dict:
    return default_config() if path is None else yaml.safe_load(Path(path).read_text())


def save_dataset(
    out: Path, txns: pd.DataFrame, accounts: pd.DataFrame, labels: pd.DataFrame
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    txns.to_parquet(out / "transactions.parquet", compression="zstd", index=False)
    accounts.to_parquet(out / "accounts.parquet", index=False)
    labels.to_parquet(out / "labels.parquet", index=False)


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p = Path(path)
    return (
        pd.read_parquet(p / "transactions.parquet"),
        pd.read_parquet(p / "accounts.parquet"),
        pd.read_parquet(p / "labels.parquet"),
    )


def connect(txns: pd.DataFrame, accounts: pd.DataFrame | None = None) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with ``txns`` and a per-entity ``ledger`` (one row per side)."""
    con = duckdb.connect()
    con.register("txns_df", txns[TXN_COLS])
    con.execute("CREATE TABLE txns AS SELECT * FROM txns_df")
    con.execute(
        """
        CREATE TABLE ledger AS
        SELECT txn_id, ts, src AS entity, dst AS cp, 'out' AS direction, amount, channel,
               dst_country AS cp_country, is_cross_border FROM txns WHERE src_is_entity
        UNION ALL
        SELECT txn_id, ts, dst, src, 'in', amount, channel,
               src_country, is_cross_border FROM txns WHERE dst_is_entity
        """
    )
    if accounts is not None:
        con.register("accounts_df", accounts)
        con.execute("CREATE TABLE accounts AS SELECT * FROM accounts_df")
    return con


def entities_of(txns: pd.DataFrame) -> pd.DataFrame:
    """Account table (every account that appears on either side)."""
    ents = pd.unique(pd.concat([txns["src"], txns["dst"]]))
    return pd.DataFrame({"entity": ents, "segment": "unknown", "country": None})


def labels_of(txns: pd.DataFrame) -> pd.DataFrame:
    """Entity labels from the transaction-level flags: one row per (entity, typology)."""
    flagged = txns[txns["laundering"] == 1]
    typ = flagged["laundering_type"] if "laundering_type" in flagged else "laundering"
    both = pd.concat(
        [
            pd.DataFrame({"entity": flagged["src"], "typology": typ}),
            pd.DataFrame({"entity": flagged["dst"], "typology": typ}),
        ]
    )
    out = both.drop_duplicates().reset_index(drop=True)
    out["evasive"] = False
    out["group"] = ""
    return out
