"""Optional adapter for the public IBM AMLworld dataset (HI-Small variant).

The Kaggle archive ``ealtman2019/ibm-transactions-for-anti-money-laundering-aml`` can be
fetched without a login through Kaggle's public download endpoint.  Because the archive is
large, only the requested member is pulled with HTTP range requests (``remotezip``).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import duckdb
import pandas as pd

URL = "https://www.kaggle.com/api/v1/datasets/download/ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
MEMBER = "HI-Small_Trans.csv"


def fetch(out: Path, member: str = MEMBER) -> Path:
    """Download one member of the Kaggle archive into ``out`` (needs the ``ibm`` extra)."""
    from remotezip import RemoteZip

    out.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL) as r:  # redirects to a signed storage URL
        final = r.geturl()
    with RemoteZip(final) as z:
        z.extract(member, out)
    return out / member


def convert(csv: Path, days: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Map IBM AMLworld rows to the lab schema.

    An account is labelled suspicious if it touches a transaction flagged as laundering. The
    transaction frame keeps that flag in a ``laundering`` column so labels can be re-derived
    for any time window.
    ``days`` keeps only the first N days of the file; labels are derived from that window.
    """
    con = duckdb.connect()
    con.execute(
        f"""
        CREATE TABLE raw AS SELECT * FROM read_csv('{csv}', header=false, skip=1,
          names=['ts','from_bank','src_acct','to_bank','dst_acct','amt_recv','cur_recv',
                 'amt_paid','cur_paid','fmt','laundering'],
          types={{'ts':'TIMESTAMP','from_bank':'VARCHAR','src_acct':'VARCHAR','to_bank':'VARCHAR',
                 'dst_acct':'VARCHAR','amt_recv':'DOUBLE','cur_recv':'VARCHAR','amt_paid':'DOUBLE',
                 'cur_paid':'VARCHAR','fmt':'VARCHAR','laundering':'INTEGER'}},
          timestampformat='%Y/%m/%d %H:%M')
        """
    )
    if days is not None:
        con.execute(
            f"DELETE FROM raw WHERE ts >= (SELECT min(ts) FROM raw) + INTERVAL '{int(days)} days'"
        )
    txns = con.execute(
        """
        SELECT row_number() OVER (ORDER BY ts) AS txn_id, ts,
               from_bank || '_' || src_acct AS src, to_bank || '_' || dst_acct AS dst,
               amt_paid AS amount, lower(fmt) AS channel,
               CAST(NULL AS VARCHAR) AS src_country, CAST(NULL AS VARCHAR) AS dst_country,
               cur_paid <> cur_recv AS is_cross_border,
               TRUE AS src_is_entity, TRUE AS dst_is_entity,
               CAST(laundering AS TINYINT) AS laundering
        FROM raw ORDER BY ts
        """
    ).df()
    sus = con.execute(
        """
        SELECT DISTINCT e AS entity FROM (
          SELECT from_bank || '_' || src_acct AS e FROM raw WHERE laundering = 1
          UNION ALL SELECT to_bank || '_' || dst_acct FROM raw WHERE laundering = 1)
        """
    ).df()
    sus["typology"] = "laundering"
    sus["evasive"] = False
    sus["group"] = ""
    ents = pd.unique(pd.concat([txns["src"], txns["dst"]]))
    accounts = pd.DataFrame({"entity": ents, "segment": "unknown", "country": None})
    return txns, accounts, sus
