"""Audit analyses computed from scenario alerts, labels and model scores."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import scenarios as S
from . import sampling as smp

TYPOLOGIES = [
    "structuring",
    "passthrough",
    "fan_in",
    "fan_out",
    "roundtrip",
    "dormant",
    "high_risk",
]
SWEEPS = {
    "TM-01-STRUCT": ("min_count", [2, 3, 4, 5, 6, 7]),
    "TM-02-PASSTHRU": ("window_hours", [12, 24, 36, 48, 60, 72, 96]),
    "TM-03-FAN": ("min_counterparties", [5, 6, 8, 10, 12, 15, 20]),
    "TM-04-ROUNDTRIP": ("max_days", [3, 5, 7, 10, 14, 21, 30]),
    "TM-05-DORMANT": ("dormant_days", [30, 45, 60, 75, 90, 120, 150]),
    "TM-06-HIGHRISK": ("min_total", [5000, 10000, 15000, 20000, 30000, 40000, 60000]),
}


def has_typologies(labels: pd.DataFrame) -> bool:
    """False for datasets (e.g. IBM AMLworld) whose labels carry no injected typology."""
    return bool(labels["typology"].isin(TYPOLOGIES).any())


def own_entities(labels: pd.DataFrame, module) -> set:
    """Labelled entities a scenario is meant to catch; every labelled entity if untyped."""
    if not has_typologies(labels):
        return set(labels["entity"])
    return set(labels.loc[labels["typology"].isin(module.TYPOLOGIES), "entity"])


def entity_sets(alerts: pd.DataFrame) -> dict[str, set]:
    return {sid: set(g["entity"]) for sid, g in alerts.groupby("scenario_id")}


def productivity(alerts: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    sus = set(labels["entity"])
    rows = []
    for m in S.MODULES:
        a = alerts[alerts["scenario_id"] == m.SCENARIO_ID]
        ents = set(a["entity"])
        own = own_entities(labels, m)
        rows.append({
            "scenario": m.SCENARIO_ID, "alerts": len(a), "entities": len(ents),
            "true_positive": len(ents & sus), "conversion": len(ents & sus) / len(ents) if ents else np.nan,
            "own_typology_hits": len(ents & own), "own_precision": len(ents & own) / len(ents) if ents else np.nan,
            "own_recall": len(ents & own) / len(own) if own else np.nan,
        })  # fmt: skip
    return pd.DataFrame(rows)


def overlap(sets: dict[str, set], order: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    jac = pd.DataFrame(index=order, columns=order, dtype=float)
    cnt = pd.DataFrame(index=order, columns=order, dtype=int)
    for a in order:
        for b in order:
            sa, sb = sets.get(a, set()), sets.get(b, set())
            cnt.loc[a, b] = len(sa & sb)
            jac.loc[a, b] = len(sa & sb) / len(sa | sb) if sa | sb else 0.0
    return jac, cnt


def coverage(
    sets: dict[str, set], labels: pd.DataFrame, order: list[str]
) -> dict[str, pd.DataFrame]:
    """Typology x scenario detection rate (all, obvious variants, evasive variants)."""
    out = {}
    for name, sel in (
        ("all", slice(None)),
        ("obvious", ~labels["evasive"]),
        ("evasive", labels["evasive"]),
    ):
        sub = labels if name == "all" else labels[sel]
        m = pd.DataFrame(index=TYPOLOGIES, columns=[*order, "ANY"], dtype=float)
        for t in TYPOLOGIES:
            ents = set(sub.loc[sub["typology"] == t, "entity"])
            for sid in order:
                m.loc[t, sid] = len(ents & sets.get(sid, set())) / len(ents) if ents else np.nan
            union = set().union(*sets.values()) if sets else set()
            m.loc[t, "ANY"] = len(ents & union) / len(ents) if ents else np.nan
        out[name] = m
    return out


def sensitivity(
    con, cfg: dict, labels: pd.DataFrame, population: set | None = None
) -> pd.DataFrame:
    sus = set(labels["entity"])
    rows = []
    for m in S.MODULES:
        key, values = SWEEPS[m.SCENARIO_ID]
        own = own_entities(labels, m)
        current = cfg["scenarios"][S.CONFIG_KEY[m.SCENARIO_ID]][key]
        for v in sorted({*values, current}):
            c = S.with_override(cfg, m.SCENARIO_ID, key, v)
            ents = set(S.run_scenario(con, c, m.SCENARIO_ID)["entity"])
            if population is not None:
                ents &= population
            tp, hit = len(ents & sus), len(ents & own)
            prec = tp / len(ents) if ents else 0.0
            rec = hit / len(own) if own else 0.0
            rows.append({
                "scenario": m.SCENARIO_ID, "parameter": key, "value": v, "is_current": v == current,
                "entities": len(ents), "true_positive": tp, "own_hits": hit, "precision": prec,
                "own_recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            })  # fmt: skip
    return pd.DataFrame(rows)


def atl_btl(
    scores: pd.DataFrame,
    labels: pd.DataFrame,
    sets: dict[str, set],
    seed: int,
    n_btl: int = 200,
    n_atl: int = 100,
    conf: float = 0.95,
) -> dict:
    """Portfolio and per-scenario ATL/BTL tests. Reviewer outcomes come from the labels."""
    y = scores.set_index("entity")["is_suspicious"]
    atl = scores[scores["rule_alert"] == 1]
    btl = scores[scores["rule_alert"] == 0].copy()
    truth_missed = int(btl["is_suspicious"].sum())
    out: dict = {"n_atl": len(atl), "n_btl": len(btl), "truth_btl_positives": truth_missed,
                 "truth_total_positives": int(y.sum()), "conf": conf}  # fmt: skip

    # random sample of BTL
    ids = smp.draw_sample(btl["entity"], n_btl, seed)
    k = int(y.loc[ids].sum())
    out["random"] = {**smp.btl_estimate(len(btl), len(ids), k, conf), "sample": len(ids)}

    # stratified by risk score within BTL: top decile / next 30% / rest
    btl["rank"] = btl["risk_score"].rank(ascending=False, method="first") / len(btl)
    btl["stratum"] = np.where(
        btl["rank"] <= 0.10, "top10", np.where(btl["rank"] <= 0.40, "mid30", "rest60")
    )
    alloc = {
        "top10": int(n_btl * 0.4),
        "mid30": int(n_btl * 0.35),
        "rest60": n_btl - int(n_btl * 0.4) - int(n_btl * 0.35),
    }
    smpl = smp.stratified_draw(btl, "stratum", alloc, seed)
    strata = []
    for name in alloc:
        pop = int((btl["stratum"] == name).sum())
        sub = smpl[smpl["stratum"] == name]
        strata.append({"stratum": name, "pop": pop, "n": len(sub), "k": int(sub["is_suspicious"].sum()),
                       "truth": int(btl.loc[btl["stratum"] == name, "is_suspicious"].sum())})  # fmt: skip
    out["stratified"] = {**smp.stratified_estimate(strata, conf), "strata": strata}

    # ATL: how many alerted entities are confirmed
    atl_ids = smp.draw_sample(atl["entity"], n_atl, seed + 1)
    kk = int(y.loc[atl_ids].sum())
    lo, hi = smp.clopper_pearson(kk, len(atl_ids), conf)
    out["atl"] = {"n": len(atl_ids), "k": kk, "precision": kk / len(atl_ids), "lo": lo, "hi": hi,
                  "truth_precision": float(atl["is_suspicious"].mean())}  # fmt: skip

    # repeated random BTL draws: does the interval cover the truth?
    cover = 0
    reps = 500
    for r in range(reps):
        ids_r = smp.draw_sample(btl["entity"], n_btl, 1000 + r)
        est = smp.btl_estimate(len(btl), len(ids_r), int(y.loc[ids_r].sum()), conf)
        cover += est["est_lo"] <= truth_missed <= est["est_hi"]
    out["coverage_check"] = {"reps": reps, "covered": cover / reps}

    # per scenario: sample entities NOT alerted by the scenario, look for its typologies
    per = []
    for m in S.MODULES:
        alerted = sets.get(m.SCENARIO_ID, set())
        pool = scores.loc[~scores["entity"].isin(alerted), "entity"]
        own = own_entities(labels, m)
        truth_n = len(own - alerted)
        n = min(150, len(pool))
        ids_s = smp.draw_sample(pool, n, seed + 7)
        ks = sum(e in own for e in ids_s)
        est = smp.btl_estimate(len(pool), n, ks, conf)
        per.append({"scenario": m.SCENARIO_ID, "btl_population": len(pool), "sample": n, "found": ks,
                    "est_missed": est["est"], "est_lo": est["est_lo"], "est_hi": est["est_hi"],
                    "truth_missed": truth_n,
                    "n_to_detect_at_truth": smp.detection_sample_size(truth_n / len(pool)) if truth_n else None,
                    })  # fmt: skip
    out["per_scenario"] = pd.DataFrame(per)

    # targeted yield: does the model point at the BTL positives?
    top = btl.sort_values("risk_score", ascending=False).head(n_btl)
    out["targeted"] = {"n": len(top), "found": int(top["is_suspicious"].sum()),
                       "random_expected": truth_missed * len(top) / len(btl)}  # fmt: skip
    return out


def gain_curve(score: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-score, kind="stable")
    return np.arange(1, len(y) + 1), np.cumsum(y[order]) / y.sum()
