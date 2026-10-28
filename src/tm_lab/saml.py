"""Adapter for SAML-D (Oztas et al., 2023), the second public AML simulation used here.

Kaggle: ``berkanoztas/synthetic-transaction-monitoring-dataset-aml`` (CC BY-NC-SA 4.0, about
9.5M rows, 28 typologies of which 11 normal and 17 laundering). The archive needs a Kaggle login,
so this module only converts a CSV that was downloaded by hand into ``--raw``.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from .data import entities_of, labels_of

MEMBER = "SAML-D.csv"


def convert(csv: Path, account_frac: float = 1.0, seed: int = 42):
    """Map SAML-D rows to the lab schema.

    Both account columns are entities; the bank-location columns become the country fields, and
    ``Cash Deposit`` / ``Cash Withdrawal`` become the ``cash`` channel. An account is labelled
    with every laundering typology of the flagged transactions it touches (one row per pair).
    ``account_frac < 1`` keeps a random share of accounts and only transactions between kept
    accounts; the default keeps everything.
    """
    con = duckdb.connect()
    con.execute(
        f"""
        CREATE TABLE raw AS SELECT
          CAST("Date" AS DATE) + CAST("Time" AS TIME) AS ts,
          CAST(Sender_account AS VARCHAR) AS src, CAST(Receiver_account AS VARCHAR) AS dst,
          Amount AS amount, Payment_type, Payment_currency, Received_currency,
          Sender_bank_location AS src_country, Receiver_bank_location AS dst_country,
          CAST(Is_laundering AS TINYINT) AS laundering, Laundering_type
        FROM read_csv('{csv}', header=true)
        """
    )
    if account_frac < 1.0:
        con.execute(
            f"""
            CREATE TABLE keep AS
            SELECT a FROM (SELECT src AS a FROM raw UNION SELECT dst FROM raw)
            WHERE hash(a || '{seed}') % 10000 < {int(account_frac * 10000)}
            """
        )
        con.execute(
            "DELETE FROM raw WHERE src NOT IN (SELECT a FROM keep) OR dst NOT IN (SELECT a FROM keep)"
        )
    txns = con.execute(
        """
        SELECT row_number() OVER (ORDER BY ts) AS txn_id, ts, src, dst, amount,
               CASE WHEN Payment_type LIKE 'Cash%' THEN 'cash' ELSE lower(replace(Payment_type, ' ', '_')) END
                 AS channel,
               src_country, dst_country, src_country <> dst_country AS is_cross_border,
               TRUE AS src_is_entity, TRUE AS dst_is_entity, laundering,
               CASE WHEN laundering = 1 THEN Laundering_type END AS laundering_type
        FROM raw ORDER BY ts
        """
    ).df()
    return txns, entities_of(txns), labels_of(txns)
