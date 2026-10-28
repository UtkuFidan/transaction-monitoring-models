"""End-to-end evaluation of rules and ML on one dataset.

Design (see ``windows.py``): fit on an early window with the train/validation entities, report on
a later window with held-out test entities. Optuna tunes the three boosting libraries and the
rule thresholds on the early window only.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from . import model as M
from . import rule_tuning as RT
from . import scenarios as S
from .audit.analysis import own_entities
from .data import connect
from .features import FEATURES, GROUPS, build_features
from .windows import Design, assign_groups, split_windows, window_labels


def _prepare(
    txns: pd.DataFrame, labels: pd.DataFrame, cfg: dict, design: Design, cache: Path | None
):
    labels_w = window_labels(txns, labels)
    con = connect(txns)
    if cache is not None and (cache / "alerts.parquet").exists():
        alerts = pd.read_parquet(cache / "alerts.parquet")
        feats = pd.read_parquet(cache / "features.parquet")
    else:
        alerts = S.run_all(con, cfg)
        feats = build_features(con, txns, cfg, alerts)
        if cache is not None:
            cache.mkdir(parents=True, exist_ok=True)
            alerts.to_parquet(cache / "alerts.parquet", index=False)
            feats.to_parquet(cache / "features.parquet", index=False)
    feats["group"] = assign_groups(feats["entity"], design.fractions, design.seed).to_numpy()
    feats["y"] = feats["entity"].isin(set(labels_w["entity"])).astype(int)
    n_scn = alerts.groupby("entity")["scenario_id"].nunique()
    feats["n_scenarios"] = feats["entity"].map(n_scn).fillna(0).astype(int)
    return {"con": con, "alerts": alerts, "feats": feats, "labels": labels_w, "txns": txns}


def _rule_entities(con, cfg: dict, pool: set) -> tuple[set, dict[str, set]]:
    per = {
        m.SCENARIO_ID: set(S.run_scenario(con, cfg, m.SCENARIO_ID)["entity"]) & pool
        for m in S.MODULES
    }
    return set().union(*per.values()), per


def _rule_summary(ents: set, test: pd.DataFrame) -> dict:
    flag = test["entity"].isin(ents).to_numpy()
    y = test["y"].to_numpy()
    hit = int(y[flag].sum())
    return {
        "alerted_entities": int(flag.sum()), "share_of_entities": float(flag.mean()),
        "recall": hit / int(y.sum()) if y.sum() else float("nan"),
        "precision": hit / int(flag.sum()) if flag.sum() else float("nan"),
        "pr_auc_binary": float(average_precision_score(y, flag)),
    }  # fmt: skip


def run_experiment(
    txns: pd.DataFrame,
    labels: pd.DataFrame,
    cfg: dict,
    design: Design,
    *,
    tune_seconds: float = 360,
    rule_seconds: float = 60,
    lean_frac: float = 0.25,
    tuned: dict | None = None,
    n_boot: int = 500,
    log: Callable[[str], None] = print,
    cache: Path | None = None,
    stop_after: str | None = None,
) -> dict:
    """Run the whole design. ``cache`` keeps window features; ``stop_after`` ("models" or
    "rules") returns the tuned parameters found so far, so long searches can run in steps."""
    t_start = time.time()
    seed = design.seed
    early, late = split_windows(txns, design.cut)
    log(f"windows: early {len(early):,} rows, late {len(late):,} rows")
    log("early window: scenarios and features")
    A = _prepare(early, labels, cfg, design, None if cache is None else cache / "early")
    log("late window: scenarios and features")
    B = (
        A
        if design.cut is None
        else _prepare(late, labels, cfg, design, None if cache is None else cache / "late")
    )

    fa, fb = A["feats"], B["feats"]
    tr, va = fa[fa["group"] == "train"], fa[fa["group"] == "val"]
    te = fb[fb["group"] == "test"].reset_index(drop=True)
    fit_df = fa[fa["group"].isin(["train", "val"])]
    x_tr, y_tr = tr[FEATURES].to_numpy(), tr["y"].to_numpy()
    x_va, y_va = va[FEATURES].to_numpy(), va["y"].to_numpy()
    x_fit, y_fit = fit_df[FEATURES].to_numpy(), fit_df["y"].to_numpy()
    x_te, y_te = te[FEATURES].to_numpy(), te["y"].to_numpy()
    pops = {
        g: {"entities": len(d), "positives": int(d["y"].sum())}
        for g, d in (("train", tr), ("val", va), ("test", te))
    }
    log(f"populations: {pops}")

    # feature ablation: default LightGBM, train -> validation
    ablation = []
    for name, cols in GROUPS.items():
        idx = [FEATURES.index(c) for c in cols]
        p = M.fit_predict("lightgbm", {"n_estimators": 300, "learning_rate": 0.05}, x_tr[:, idx], y_tr,
                          x_va[:, idx], seed)  # fmt: skip
        ablation.append({"features": name, "n_features": len(cols),
                         "val_pr_auc": float(average_precision_score(y_va, p))})  # fmt: skip
        log(f"ablation {name}: {ablation[-1]['val_pr_auc']:.4f}")

    # hyper-parameter search
    tuned = tuned or {}
    model_params = tuned.setdefault("models", {})
    search = tuned.setdefault("search", {})
    if not model_params:
        for lib in M.LIBS:
            res = M.tune(lib, x_tr, y_tr, x_va, y_va, tune_seconds, seed, log=log)
            model_params[lib] = res["best_params"]
            search[lib] = {k: v for k, v in res.items() if k != "best_params"}

    if stop_after == "models":
        return {"tuned": {"models": model_params, "search": search}}
    val_pred = {
        lib: M.fit_predict(lib, p, x_tr, y_tr, x_va, seed) for lib, p in model_params.items()
    }
    val_pred["blend"] = np.mean([M.rank01(v) for v in val_pred.values()], axis=0)
    val_ap = {k: float(average_precision_score(y_va, v)) for k, v in val_pred.items()}
    selected = max(val_ap, key=val_ap.get)
    log(f"validation PR-AUC {val_ap}; selected {selected}")

    # final fits on train + validation, scored once on the held-out test entities
    models, test_pred = {}, {}
    for lib, p in model_params.items():
        models[lib] = M.make_model(lib, p, seed).fit(x_fit, y_fit)
        test_pred[lib] = models[lib].predict_proba(x_te)[:, 1]
    test_pred["blend"] = np.mean([M.rank01(v) for v in test_pred.values()], axis=0)
    default_lgbm = M.fit_predict("lightgbm", {}, x_fit, y_fit, x_te, seed)
    gbm_te = test_pred[selected]
    gbm_va = val_pred[selected]

    # isolation forest and the blended risk score (weights picked on validation)
    iso_fit = np.log1p(np.clip(x_tr, 0, None))
    iso_va = M.isolation_scores(iso_fit, np.log1p(np.clip(x_va, 0, None)), seed)
    iso_te = M.isolation_scores(iso_fit, np.log1p(np.clip(x_te, 0, None)), seed)
    w, w_val_ap = M.tune_blend(y_va, gbm_va, iso_va, va["n_scenarios"].to_numpy())
    risk = M.blend_score(w, gbm_te, iso_te, te["n_scenarios"].to_numpy())

    # rule thresholds: tune on the early window's tuning entities, apply to the late window
    pool_tune = set(fit_df["entity"])
    pool_test = set(te["entity"])
    if "rules" not in tuned:
        tuned["rules"] = {}
        for frac_name, frac in (("same_workload", 1.0), ("lean", lean_frac)):
            res = [
                RT.tune_scenario(A["con"], cfg, m, pool_tune, A["labels"], frac, rule_seconds, seed, log)
                for m in S.MODULES
            ]  # fmt: skip
            tuned["rules"][frac_name] = res
    if stop_after == "rules":
        return {"tuned": {"models": model_params, "search": search, "rules": tuned["rules"]}}
    rule_cfgs = {"untuned": cfg} | {k: RT.apply(cfg, v) for k, v in tuned["rules"].items()}
    rule_eval, rule_sets, rule_per_scn = {}, {}, {}
    for name, c in rule_cfgs.items():
        ents, per = _rule_entities(B["con"], c, pool_test)
        rule_sets[name], rule_per_scn[name] = ents, per
        rule_eval[name] = _rule_summary(ents, te)
        log(f"rules {name}: {rule_eval[name]}")

    flags = {k: te["entity"].isin(v).astype(int).to_numpy() for k, v in rule_sets.items()}
    boot = M.bootstrap(
        y_te,
        {"gradient_boosting": gbm_te, "combined": risk, "isolation_forest": iso_te},
        flags, n_boot, seed,
    )  # fmt: skip

    def pr(s):
        return float(average_precision_score(y_te, s))

    pr_auc = {"rules_binary": rule_eval["untuned"]["pr_auc_binary"],
              "rules_tuned": rule_eval["same_workload"]["pr_auc_binary"],
              "isolation_forest": pr(iso_te), "gradient_boosting": pr(gbm_te), "combined": pr(risk),
              "lightgbm_default": pr(default_lgbm), **{lib: pr(p) for lib, p in test_pred.items()}}  # fmt: skip
    eq = {}
    for name in rule_cfgs:
        n_rules = rule_eval[name]["alerted_entities"]
        rec = rule_eval[name]["recall"]
        k_g, k_c = M.top_k_for_recall(gbm_te, y_te, rec), M.top_k_for_recall(risk, y_te, rec)
        eq[name] = {
            "target_recall": rec, "rules_alerts": n_rules, "gbm_alerts": k_g, "combined_alerts": k_c,
            "reduction_gbm": 1 - k_g / n_rules if n_rules else None,
            "reduction_combined": 1 - k_c / n_rules if n_rules else None,
            "gbm_recall_at_rules_budget": M.recall_at_k(gbm_te, y_te, n_rules),
            "reduction_gbm_ci": boot["reduction_ci"].get(f"gradient_boosting|{name}"),
        }  # fmt: skip
    work = {
        str(t): {"gbm": M.top_k_for_recall(gbm_te, y_te, t), "combined": M.top_k_for_recall(risk, y_te, t),
                 "isolation_forest": M.top_k_for_recall(iso_te, y_te, t)}
        for t in (0.5, 0.8, 0.9, 0.95)
    }  # fmt: skip

    # like-for-like with a plain random 5-fold out-of-fold split inside the early window
    oof = np.zeros(len(y_fit))
    sel_lib = selected if selected in model_params else max(model_params, key=lambda k: val_ap[k])
    for a_idx, b_idx in StratifiedKFold(5, shuffle=True, random_state=seed).split(x_fit, y_fit):
        oof[b_idx] = M.fit_predict(sel_lib, model_params[sel_lib], x_fit[a_idx], y_fit[a_idx],
                                   x_fit[b_idx], seed)  # fmt: skip

    # importance and reason codes from the tuned LightGBM
    pi = permutation_importance(
        M.make_model("lightgbm", model_params["lightgbm"], seed).fit(x_tr, y_tr), x_va, y_va,
        scoring="average_precision", n_repeats=3, random_state=seed,
    )  # fmt: skip
    imp = pd.DataFrame(
        {"feature": FEATURES, "permutation_importance": pi.importances_mean}
    ).sort_values("permutation_importance", ascending=False)
    codes = M.reason_codes(models["lightgbm"], te, fit_df)

    scores = te[["entity"]].copy()
    scores["is_suspicious"] = y_te
    scores["gbm_score"] = gbm_te
    scores["iso_score"] = iso_te
    scores["n_scenarios"] = te["n_scenarios"].to_numpy()
    scores["rule_alert"] = flags["untuned"]
    scores["rule_alert_tuned"] = flags["same_workload"]
    scores["risk_score"] = risk
    scores["reason_codes"] = codes

    # per-typology recall on the test entities
    lab = B["labels"]
    lab = lab[lab["entity"].isin(pool_test)]
    top_k = {
        n: set(scores.nlargest(rule_eval[n]["alerted_entities"], "gbm_score")["entity"])
        for n in rule_cfgs
    }
    rows = []
    for typ, g in lab.groupby("typology"):
        ents = set(g["entity"])
        row = {"typology": typ, "entities": len(ents)}
        for n in rule_cfgs:
            row[f"rules_{n}"] = len(ents & rule_sets[n]) / len(ents)
            row[f"model_{n}_budget"] = len(ents & top_k[n]) / len(ents)
        rows.append(row)
    per_typology = pd.DataFrame(rows).sort_values("entities", ascending=False)

    per_scenario = []
    for name in rule_cfgs:
        for m in S.MODULES:
            ents = rule_per_scn[name][m.SCENARIO_ID]
            own = own_entities(B["labels"], m) & pool_test
            per_scenario.append({
                "config": name, "scenario": m.SCENARIO_ID, "entities": len(ents),
                "precision": len(ents & set(lab["entity"])) / len(ents) if ents else None,
                "own_recall": len(ents & own) / len(own) if own else None,
            })  # fmt: skip

    metrics = {
        "n_entities": len(te), "n_positives": int(y_te.sum()), "prevalence": float(y_te.mean()),
        "design": {"cut": None if design.cut is None else str(design.cut),
                   "fractions": design.fractions, "seed": seed, "populations": pops,
                   "early_rows": len(early), "late_rows": len(late)},
        "feature_ablation": ablation, "n_features": len(FEATURES),
        "tuning": {lib: {**search.get(lib, {}), "val_pr_auc": val_ap[lib], "test_pr_auc": pr_auc[lib]}
                   for lib in model_params},
        "validation_pr_auc": val_ap, "selected_model": selected,
        "blend_weights": {"gbm": w[0], "isolation": w[1], "rules": w[2], "val_pr_auc": w_val_ap},
        "pr_auc": pr_auc,
        "roc_auc": {"isolation_forest": M.evaluate_scores(y_te, iso_te)["roc_auc"],
                    "gradient_boosting": M.evaluate_scores(y_te, gbm_te)["roc_auc"],
                    "combined": M.evaluate_scores(y_te, risk)["roc_auc"]},
        "pr_auc_ci": boot["pr_auc_ci"],
        "random_split_oof_pr_auc": float(average_precision_score(y_fit, oof)),
        "rules": rule_eval, "equal_recall": eq["untuned"], "equal_recall_by_rules": eq,
        "workload_at_recall": work, "per_scenario": per_scenario,
        "per_typology": per_typology.round(4).to_dict("records"),
        "seconds": round(time.time() - t_start),
    }  # fmt: skip
    return {
        "scores": scores, "metrics": metrics, "importance": imp, "per_typology": per_typology,
        "tuned": {"models": model_params, "search": search, "rules": tuned["rules"]},
        "late": B, "pool_test": pool_test, "alerts": B["alerts"], "rule_cfgs": rule_cfgs,
    }  # fmt: skip


def save(res: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    res["scores"].to_parquet(out / "entity_scores.parquet", index=False)
    res["importance"].to_csv(out / "permutation_importance.csv", index=False)
    res["per_typology"].to_csv(out / "per_typology_recall.csv", index=False)
    (out / "model_metrics.json").write_text(json.dumps(res["metrics"], indent=2, default=str))
    (out / "tuned_params.json").write_text(json.dumps(res["tuned"], indent=2, default=str))
