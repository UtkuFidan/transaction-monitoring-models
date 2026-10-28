"""Seeded synthetic transaction generator with labelled typologies.

Entities are accounts ``A000001...``; counterparties outside the institution are
``X00001...`` (or the pseudo-counterparty ``CASH``).  Every injected typology
comes in an *obvious* and an *evasive* variant; the evasive one is built to sit
just below the default scenario thresholds, which is what makes the below-the-line
audit interesting.  Country codes are placeholders: ``DM`` domestic, ``F1..F6``
foreign, ``H1..H3`` high-risk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HIGH_RISK = ["H1", "H2", "H3"]
FOREIGN = [f"F{i}" for i in range(1, 7)]
DAY = 86400


class _Buf:
    def __init__(self) -> None:
        self.cols: dict[str, list] = {
            k: [] for k in ("ts", "src", "dst", "amount", "channel", "src_c", "dst_c")
        }

    def add(self, ts, src, dst, amount, channel, src_c="DM", dst_c="DM") -> None:
        for k, v in zip(self.cols, (ts, src, dst, amount, channel, src_c, dst_c), strict=True):
            self.cols[k].append(v)

    def add_many(self, ts, src, dst, amount, channel, src_c, dst_c) -> None:
        for k, v in zip(self.cols, (ts, src, dst, amount, channel, src_c, dst_c), strict=True):
            self.cols[k].extend(list(v) if not isinstance(v, str) else [v] * len(ts))


def generate(
    n_accounts: int = 2500, days: int = 270, seed: int = 42, start: str = "2025-04-01"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (transactions, accounts, labels)."""
    rng = np.random.default_rng(seed)
    t0 = int(pd.Timestamp(start).timestamp())
    n_ext = n_accounts * 2
    ext_ids = np.array([f"X{i:05d}" for i in range(n_ext)])
    ext_country = rng.choice(
        ["DM", *FOREIGN, *HIGH_RISK], size=n_ext, p=[0.90] + [0.095 / 6] * 6 + [0.005 / 3] * 3
    )
    country_of = dict(zip(ext_ids, ext_country, strict=True))
    dom_ext = ext_ids[ext_country == "DM"]
    for_ext = ext_ids[np.isin(ext_country, FOREIGN)]
    hr_ext = ext_ids[np.isin(ext_country, HIGH_RISK)]
    acct_ids = np.array([f"A{i:06d}" for i in range(1, n_accounts + 1)])

    # assign roles on a shuffled list so specials never collide
    pool = list(rng.permutation(acct_ids))

    def take(n: int) -> list[str]:
        out = [pool.pop() for _ in range(min(n, len(pool)))]
        return out

    s = n_accounts / 2500
    k = lambda x: max(1, round(x * s))  # noqa: E731
    merchants, processors = take(k(60)), take(k(10))
    cash_biz, windfall = take(k(50)), take(k(6))
    hr_remit, trade_hr = take(k(35)), take(k(5))
    occasional = take(k(120))
    inj = {
        "structuring": take(k(20)),
        "passthrough": take(k(18)),
        "fan_in": take(k(12)),
        "fan_out": take(k(12)),
        "dormant": take(k(14)),
        "high_risk": take(k(18)),
    }
    n_rings = k(6)
    rings = [take(int(rng.integers(3, 5))) for _ in range(n_rings)]
    segment = {a: "retail" for a in acct_ids}
    for lst, seg in (
        (merchants, "merchant"),
        (processors, "processor"),
        (cash_biz, "cash_business"),
    ):
        for a in lst:
            segment[a] = seg

    mult = {a: (float(rng.uniform(4, 12)) if rng.random() < 0.22 else 1.0) for a in acct_ids}
    buf = _Buf()
    peer_pool = np.array([a for a in acct_ids if a not in set(inj["dormant"])])
    # seasonality: weekday shape x Nov/Dec uplift
    dates = pd.to_datetime(t0 + np.arange(days) * DAY, unit="s")
    season = np.where(dates.day_of_week == 5, 1.15, np.where(dates.day_of_week == 6, 0.8, 1.0))
    season = season * np.where(dates.month == 11, 1.25, np.where(dates.month == 12, 1.5, 1.0))

    def daily(rate: float, lo: int = 0, hi: int | None = None) -> np.ndarray:
        hi = days if hi is None else hi
        cnt = np.zeros(days, dtype=int)
        cnt[lo:hi] = rng.poisson(rate * season[lo:hi])
        return np.repeat(np.arange(days), cnt)

    def stamp(day_idx: np.ndarray, h0: float = 7, h1: float = 22) -> np.ndarray:
        return (t0 + day_idx * DAY + rng.uniform(h0, h1, len(day_idx)) * 3600).astype(np.int64)

    def T0(day: float) -> int:
        return int(t0 + day * DAY + rng.uniform(8, 18) * 3600)

    def background(a: str, lo: int = 0, hi: int | None = None, scale: float = 1.0) -> None:
        hi = days if hi is None else hi
        seg = segment[a]
        mu = mult[a]
        rate = 0.12 * scale * (1.0 + 0.2 * mu)
        # purchases
        d = daily(rate, lo, hi)
        amt = np.minimum(rng.lognormal(np.log(40 * mu), 1.0, len(d)), 3000 * mu).round(2)
        m = rng.choice(dom_ext, len(d))
        buf.add_many(stamp(d), [a] * len(d), m, amt, "card", "DM", "DM")
        # P2P with other internal accounts
        d = daily(0.05 * scale, lo, hi)
        if len(d):
            amt = np.minimum(rng.lognormal(np.log(150 * mu), 1.0, len(d)), 5000 * mu).round(2)
            peer = rng.choice(peer_pool, len(d))
            peer = np.where(peer == a, acct_ids[0] if a != acct_ids[0] else acct_ids[1], peer)
            out = rng.random(len(d)) < 0.5
            for t, p, v, o in zip(stamp(d), peer, amt, out, strict=True):
                buf.add(t, a if o else p, p if o else a, v, "p2p")
        # salary + rent monthly
        if seg == "retail":
            employer = rng.choice(dom_ext)
            base = rng.normal(2800, 900) * (mu if mu > 1 else 1.0)
            rent = rng.choice([0, 0, 600, 800, 1000, 1200]) * mu
            for mo in range(lo // 30, min(hi, days) // 30 + 1):
                dd = mo * 30 + int(rng.integers(24, 28))
                if lo <= dd < hi and dd < days and base > 800:
                    amt = (
                        round(base + rng.normal(0, 40), -1)
                        if rng.random() < 0.6
                        else round(base + rng.normal(0, 40), 2)
                    )
                    buf.add(
                        t0 + dd * DAY + int(rng.integers(6, 10)) * 3600, employer, a, amt, "ach"
                    )
                dd = mo * 30 + int(rng.integers(1, 4))
                if rent and lo <= dd < hi and dd < days:
                    buf.add(t0 + dd * DAY + 9 * 3600, a, rng.choice(dom_ext), rent, "ach")

    for a in acct_ids:
        if a in inj["dormant"]:
            background(a, 0, int(rng.integers(25, 40)))
        elif a in occasional:
            lo = int(rng.integers(0, days - 40))
            background(a, lo, lo + 30, scale=0.5)
        else:
            background(a)

    # hard negatives: legitimate behaviour that looks suspicious
    for a in cash_biz:  # cash-intensive shops: frequent cash, sometimes just under the line
        d = daily(1.5)
        amt = np.minimum(rng.lognormal(np.log(900), 0.9, len(d)), 7900).round(0)
        near = rng.random(len(d)) < 0.08
        amt = np.where(near, rng.uniform(8200, 9900, len(d)).round(-1), amt)
        buf.add_many(stamp(d), ["CASH"] * len(d), [a] * len(d), amt, "cash", "DM", "DM")
    for a in merchants:  # many small payers, weekly supplier payment
        rate = float(rng.uniform(0.4, 3.5))
        d = daily(rate)
        amt = rng.lognormal(np.log(400), 0.6, len(d)).round(2)
        payers = rng.choice(dom_ext, len(d))
        buf.add_many(stamp(d), payers, [a] * len(d), amt, "p2p", "DM", "DM")
        for wk in range(days // 7):
            buf.add(
                t0 + (wk * 7 + 4) * DAY + 11 * 3600,
                a,
                rng.choice(dom_ext),
                round(rate * 7 * 400 * rng.uniform(0.5, 0.9), 2),
                "ach",
            )
    for a in processors:  # payment processors: forward almost everything within hours
        for _ in range(int(rng.integers(40, 70))):
            day = rng.integers(0, days - 3)
            amt = round(float(rng.uniform(4000, 20000)), 2)
            t = t0 + int(day) * DAY + int(rng.integers(8, 18)) * 3600
            buf.add(t, rng.choice(dom_ext), a, amt, "wire")
            buf.add(
                t + int(rng.integers(2, 30)) * 3600,
                a,
                rng.choice(dom_ext),
                round(amt * rng.uniform(0.97, 0.995), 2),
                "wire",
            )
    for a in windfall:  # legitimate one-off large credit (inheritance, property sale)
        day = int(rng.integers(min(120, days // 2), days - 20))
        for i in range(int(rng.integers(5, 9))):
            buf.add(
                t0 + (day + i) * DAY + 10 * 3600,
                rng.choice(dom_ext),
                a,
                round(float(rng.uniform(3500, 9000)), 2),
                "wire",
            )
    for a in acct_ids:  # ordinary remittances (small, regular, some to high-risk places)
        r = rng.random()
        if a in hr_remit or r < 0.04:
            tgt = hr_ext if a in hr_remit else for_ext
            if len(tgt) == 0:
                continue
            cp = rng.choice(tgt)
            for mo in range(days // 30):
                buf.add(
                    t0 + (mo * 30 + int(rng.integers(5, 25))) * DAY + 12 * 3600,
                    a,
                    cp,
                    round(float(rng.uniform(150, 900)), -1),
                    "wire",
                    "DM",
                    country_of[cp],
                )
    for a in trade_hr:  # legitimate importer paying a high-risk-country supplier
        cp = rng.choice(hr_ext)
        for _ in range(int(rng.integers(2, 4))):
            buf.add(
                t0 + int(rng.integers(10, days - 5)) * DAY + 14 * 3600,
                a,
                cp,
                round(float(rng.uniform(9000, 15000)), -2),
                "wire",
                "DM",
                country_of[cp],
            )

    dormant_set = set(inj["dormant"])
    for a in acct_ids:  # legitimate lumpy behaviour that shares features with the typologies
        r = rng.random()
        if a in dormant_set:
            continue
        if r < 0.10:  # property / car / inheritance: big credit, partly spent within days
            day = int(rng.integers(10, days - 12))
            amt = float(rng.uniform(8000, 60000))
            buf.add(T0(day), rng.choice(dom_ext), a, round(amt, 2), "wire")
            buf.add(
                T0(day + rng.uniform(1, 6)),
                a,
                rng.choice(dom_ext),
                round(amt * rng.uniform(0.6, 0.97), 2),
                "wire",
            )
        elif r < 0.13:  # party / wedding: gifts to many people within days
            day = int(rng.integers(10, days - 10))
            for _ in range(int(rng.integers(9, 15))):
                buf.add(
                    T0(day + rng.uniform(0, 3)),
                    a,
                    rng.choice(dom_ext),
                    round(float(rng.uniform(100, 400)), -1),
                    "p2p",
                )
        elif r < 0.19:  # occasional larger cash deposits (sale of an item, savings)
            for _ in range(int(rng.integers(1, 4))):
                buf.add(
                    T0(rng.uniform(10, days - 1)),
                    "CASH",
                    a,
                    float(round(rng.uniform(2500, 9500), -1)),
                    "cash",
                )

    # injected typologies
    labels: list[dict] = []

    def lab(entity: str, typ: str, evasive: bool, group: str = "") -> None:
        labels.append({"entity": entity, "typology": typ, "evasive": evasive, "group": group})

    def T(day: float) -> int:
        return int(t0 + day * DAY)

    for i, a in enumerate(inj["structuring"]):
        ev = i % 4 == 3
        lab(a, "structuring", ev)
        for ep in range(int(rng.integers(1, 3))):
            d0 = rng.uniform(15 + ep * 100, 100 + ep * 100)
            if ev and i % 8 == 3:  # only three deposits in the window
                n, lo, hi, span = 3, 8200, 9800, 6
            elif ev:  # amounts under the lower band
                n, lo, hi, span = 5, 6500, 7800, 6
            else:
                n, lo, hi, span = int(rng.integers(4, 8)), 8000, 9900, 5
            for _ in range(n):
                buf.add(
                    T(d0 + rng.uniform(0, span)),
                    "CASH",
                    a,
                    float(round(rng.uniform(lo, hi), -1)),
                    "cash",
                )
    for i, a in enumerate(inj["passthrough"]):
        ev = i % 4 == 3
        lab(a, "passthrough", ev)
        d0 = rng.uniform(15, days - 30)
        for j in range(int(rng.integers(3, 6))):
            amt = round(float(rng.uniform(6000, 30000)), 2)
            t = T(d0 + j * rng.uniform(1, 3))
            buf.add(t, rng.choice(dom_ext), a, amt, "wire")
            lag = rng.uniform(55, 70) if ev else rng.uniform(2, 36)
            buf.add(
                t + int(lag * 3600),
                a,
                rng.choice(dom_ext),
                round(amt * rng.uniform(0.95, 0.99), 2),
                "wire",
            )
    for typ in ("fan_in", "fan_out"):
        for i, a in enumerate(inj[typ]):
            ev = i % 4 == 3
            lab(a, typ, ev)
            d0 = rng.uniform(20, days - 30)
            n = int(rng.integers(7, 10)) if ev else int(rng.integers(12, 26))
            span = 8 if ev else 4
            amts = rng.uniform(700, 1600, n).round(2) if ev else rng.uniform(500, 1500, n).round(2)
            peers = rng.choice(dom_ext, n, replace=False)
            tot = float(amts.sum())
            if typ == "fan_in":
                for v, p in zip(amts, peers, strict=True):
                    buf.add(T(d0 + rng.uniform(0, span)), p, a, float(v), "p2p")
                buf.add(
                    T(d0 + span + rng.uniform(0.5, 3)),
                    a,
                    rng.choice(dom_ext),
                    round(tot * 0.95, 2),
                    "wire",
                )
            else:
                buf.add(
                    T(d0 - rng.uniform(0.5, 3)),
                    rng.choice(dom_ext),
                    a,
                    round(tot * 1.02, 2),
                    "wire",
                )
                for v, p in zip(amts, peers, strict=True):
                    buf.add(T(d0 + rng.uniform(0, span)), a, p, float(v), "p2p")
    for r_i, ring in enumerate(rings):
        ev = r_i % 3 == 2
        for a in ring:
            lab(a, "roundtrip", ev, f"ring{r_i}")
        amt = float(rng.uniform(15000, 80000))
        d0 = rng.uniform(15, days - 70)
        step = rng.uniform(5, 8) if ev else rng.uniform(0.5, 2.2)
        t = d0
        for lap in range(2):
            for h in range(len(ring)):
                amt *= rng.uniform(0.985, 0.995)
                buf.add(T(t), ring[h], ring[(h + 1) % len(ring)], round(amt, 2), "wire")
                t += step
            t += 15 if lap == 0 and not ev else 0
    for i, a in enumerate(inj["dormant"]):
        ev = i % 4 == 3
        lab(a, "dormant", ev)
        wake = rng.uniform(min(165, days * 0.6), days - 15)
        if ev:  # silent for only ~70 days: below the 90-day line
            wake = rng.uniform(100, 110)
        tot = float(rng.uniform(30000, 100000))
        n = int(rng.integers(6, 11))
        for j in range(n):
            t = T(wake + rng.uniform(0, 10))
            if j % 2 == 0:
                buf.add(
                    t,
                    rng.choice(dom_ext),
                    a,
                    round(tot / n * 1.5 * rng.uniform(0.8, 1.2), 2),
                    "wire",
                )
            else:
                buf.add(
                    t, a, rng.choice(dom_ext), round(tot / n * rng.uniform(0.8, 1.2), 2), "wire"
                )
    for i, a in enumerate(inj["high_risk"]):
        ev = i % 4 == 3
        lab(a, "high_risk", ev)
        d0 = rng.uniform(15, days - 30)
        n = 3 if ev else int(rng.integers(3, 6))
        lo, hi = (3000, 5000) if ev else (6000, 18000)
        for _ in range(n):
            cp = rng.choice(hr_ext)
            buf.add(
                T(d0 + rng.uniform(0, 20)),
                a,
                cp,
                float(round(rng.uniform(lo, hi), -2)),
                "wire",
                "DM",
                country_of[cp],
            )

    # assemble
    c = buf.cols
    txns = pd.DataFrame(c)
    txns["ts"] = pd.to_datetime(txns["ts"].astype("int64"), unit="s")
    txns = txns.sort_values("ts", kind="stable").reset_index(drop=True)
    txns.insert(0, "txn_id", np.arange(1, len(txns) + 1))
    txns["amount"] = txns["amount"].astype(float).round(2)
    txns = txns.rename(columns={"src_c": "src_country", "dst_c": "dst_country"})
    txns["is_cross_border"] = txns["src_country"] != txns["dst_country"]
    txns["src_is_entity"] = txns["src"].str.startswith("A")
    txns["dst_is_entity"] = txns["dst"].str.startswith("A")
    accounts = pd.DataFrame({"entity": acct_ids, "segment": [segment[a] for a in acct_ids]})
    accounts["country"] = "DM"
    labels_df = pd.DataFrame(labels, columns=["entity", "typology", "evasive", "group"])
    return txns, accounts, labels_df
