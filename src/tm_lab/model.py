"""ML layer: tuned gradient boosting (LightGBM, XGBoost, CatBoost), Isolation Forest, risk score.

Hyper-parameters are searched with Optuna on a validation split of entities that the final test
entities never touch. Metrics come from :func:`evaluate_scores`, with bootstrap intervals.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

from .features import FEATURES

LIBS = ("lightgbm", "xgboost", "catboost")
W_GRID = [(g, i, 1 - g - i) for g in np.arange(0, 1.01, 0.1) for i in np.arange(0, 1.01 - g, 0.1)]


def top_k_for_recall(score: np.ndarray, y: np.ndarray, target: float) -> int:
    """Smallest number of top-scored entities whose recall reaches ``target``."""
    order = np.argsort(-score, kind="stable")
    hits = np.cumsum(y[order])
    need = int(np.ceil(target * y.sum() - 1e-9))
    return int(np.searchsorted(hits, need) + 1) if need > 0 else 0


def recall_at_k(score: np.ndarray, y: np.ndarray, k: int) -> float:
    order = np.argsort(-score, kind="stable")[:k]
    return float(y[order].sum() / y.sum()) if y.sum() else float("nan")


def make_model(lib: str, params: dict, seed: int = 42, n_jobs: int = -1):
    """Build a classifier for ``lib``; imports are local so the package imports without them."""
    if lib == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(**params, random_state=seed, n_jobs=n_jobs, verbosity=-1)
    if lib == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            **params, tree_method="hist", random_state=seed, n_jobs=n_jobs, verbosity=0
        )
    if lib == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            **params, random_seed=seed, thread_count=n_jobs, verbose=0, allow_writing_files=False
        )
    raise ValueError(f"unknown library {lib!r}")


def suggest(trial, lib: str) -> dict:
    """Optuna search space per library."""
    f, i = trial.suggest_float, trial.suggest_int
    common = {"scale_pos_weight": f("scale_pos_weight", 1.0, 60.0, log=True)}
    if lib == "lightgbm":
        return {
            **common, "n_estimators": i("n_estimators", 100, 700),
            "learning_rate": f("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": i("num_leaves", 8, 256, log=True),
            "min_child_samples": i("min_child_samples", 5, 200, log=True),
            "subsample": f("subsample", 0.5, 1.0), "subsample_freq": 1,
            "colsample_bytree": f("colsample_bytree", 0.4, 1.0),
            "reg_alpha": f("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": f("reg_lambda", 1e-8, 10.0, log=True),
        }  # fmt: skip
    if lib == "xgboost":
        return {
            **common, "n_estimators": i("n_estimators", 100, 600),
            "learning_rate": f("learning_rate", 0.01, 0.2, log=True),
            "max_depth": i("max_depth", 3, 10),
            "min_child_weight": f("min_child_weight", 1.0, 50.0, log=True),
            "subsample": f("subsample", 0.5, 1.0),
            "colsample_bytree": f("colsample_bytree", 0.4, 1.0),
            "gamma": f("gamma", 1e-8, 5.0, log=True),
            "reg_alpha": f("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": f("reg_lambda", 1e-8, 10.0, log=True),
        }  # fmt: skip
    if lib == "catboost":
        return {
            **common, "iterations": i("iterations", 100, 500),
            "learning_rate": f("learning_rate", 0.02, 0.3, log=True),
            "depth": i("depth", 4, 8), "l2_leaf_reg": f("l2_leaf_reg", 1.0, 20.0, log=True),
            "random_strength": f("random_strength", 0.1, 5.0, log=True),
            "bagging_temperature": f("bagging_temperature", 0.0, 3.0),
            "border_count": i("border_count", 64, 254),
        }  # fmt: skip
    raise ValueError(f"unknown library {lib!r}")


def downsample_negatives(
    x: np.ndarray, y: np.ndarray, max_rows: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Keep every positive and a random share of negatives so that at most ``max_rows`` remain."""
    if len(y) <= max_rows:
        return x, y
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    keep = rng.choice(neg, size=max(max_rows - len(pos), 1), replace=False)
    idx = np.sort(np.concatenate([pos, keep]))
    return x[idx], y[idx]


def tune(
    lib: str,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    y_va: np.ndarray,
    seconds: float,
    seed: int = 42,
    max_train_rows: int = 150_000,
    log: Callable[[str], None] | None = None,
) -> dict:
    """TPE search maximising validation PR-AUC within a wall-clock budget.

    Negatives are down-sampled while searching (every positive is kept); validation is always
    scored on the full validation split.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    xs, ys = downsample_negatives(x_tr, y_tr, max_train_rows, seed)
    t0 = time.time()

    def objective(trial) -> float:
        params = suggest(trial, lib)
        m = make_model(lib, params, seed).fit(xs, ys)
        return float(average_precision_score(y_va, m.predict_proba(x_va)[:, 1]))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, timeout=seconds, gc_after_trial=True)
    if log:
        log(
            f"{lib}: {len(study.trials)} trials in {time.time() - t0:.0f}s, best {study.best_value:.4f}"
        )
    return {
        "library": lib, "best_params": study.best_params, "val_pr_auc_tuning": study.best_value,
        "n_trials": len(study.trials), "seconds": round(time.time() - t0, 1),
    }  # fmt: skip


def fit_predict(lib: str, params: dict, x_fit, y_fit, x_pred, seed: int = 42) -> np.ndarray:
    m = make_model(lib, params, seed).fit(x_fit, y_fit)
    return m.predict_proba(x_pred)[:, 1]


def rank01(s: np.ndarray) -> np.ndarray:
    return pd.Series(s).rank(pct=True, method="average").to_numpy()


def log_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.log1p(df[FEATURES].clip(lower=0).to_numpy())


def isolation_scores(x_fit: np.ndarray, x_pred: np.ndarray, seed: int = 42) -> np.ndarray:
    iso = IsolationForest(n_estimators=300, random_state=seed, n_jobs=-1).fit(x_fit)
    return -iso.score_samples(x_pred)


def tune_blend(
    y: np.ndarray, gbm: np.ndarray, iso: np.ndarray, rules: np.ndarray
) -> tuple[tuple[float, float, float], float]:
    """Pick (gbm, isolation, rules) weights on the simplex grid by validation PR-AUC."""
    parts = (rank01(gbm), rank01(iso), np.minimum(rules / 3.0, 1.0))
    best, best_w = -1.0, W_GRID[0]
    for w in W_GRID:
        v = average_precision_score(y, sum(wi * p for wi, p in zip(w, parts, strict=True)))
        if v > best:
            best, best_w = v, w
    return tuple(float(round(w, 2)) for w in best_w), float(best)


def blend_score(w, gbm, iso, rules) -> np.ndarray:
    return w[0] * rank01(gbm) + w[1] * rank01(iso) + w[2] * np.minimum(rules / 3.0, 1.0)


def bootstrap(
    y: np.ndarray,
    scores: dict[str, np.ndarray],
    rule_flags: dict[str, np.ndarray],
    n_boot: int = 500,
    seed: int = 42,
) -> dict:
    """Entity-level bootstrap: PR-AUC per score and review reduction at the rules' recall.

    For every resample the rules' recall is recomputed and each score must reach it, so the
    interval covers both sides of the comparison. Returns 95% percentile intervals.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    ap = {k: [] for k in scores}
    red = {(k, r): [] for k in scores for r in rule_flags}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.sum() == 0:
            continue
        for k, s in scores.items():
            ap[k].append(average_precision_score(yb, s[idx]))
        for r, flag in rule_flags.items():
            fb = flag[idx]
            n_rules = int(fb.sum())
            rec = float(yb[fb == 1].sum() / yb.sum())
            if n_rules == 0 or rec == 0:
                continue
            for k, s in scores.items():
                red[(k, r)].append(1 - top_k_for_recall(s[idx], yb, rec) / n_rules)

    def ci(v):
        a = np.asarray(v)
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) else None

    return {
        "pr_auc_ci": {k: ci(v) for k, v in ap.items()},
        "reduction_ci": {f"{k}|{r}": ci(v) for (k, r), v in red.items()},
    }


def evaluate_scores(y: np.ndarray, score: np.ndarray) -> dict:
    return {
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
    }


def reason_codes(model, x: pd.DataFrame, ref: pd.DataFrame, top: int = 3) -> list[str]:
    """Top positive TreeSHAP contributions per entity, with the entity's percentile on the feature.

    ``model`` is a fitted LightGBM classifier; ``ref`` supplies the percentile reference.
    """
    contrib = model.predict(x[FEATURES].to_numpy(), pred_contrib=True)[:, :-1]
    pct = np.column_stack(
        [np.searchsorted(np.sort(ref[c].to_numpy()), x[c].to_numpy()) / len(ref) for c in FEATURES]
    )
    order = np.argsort(-contrib, axis=1)[:, :top]
    return [
        "; ".join(
            f"{FEATURES[j]} (p{int(round(pct[i, j] * 100))})" for j in row if contrib[i, j] > 0
        )
        for i, row in enumerate(order)
    ]


def quiet() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
