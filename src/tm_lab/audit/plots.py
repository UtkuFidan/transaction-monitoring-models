"""Figures for the README and the audit report (matplotlib, static PNGs)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .analysis import gain_curve  # noqa: E402

BLUE, ORANGE, GREY, RED, GREEN = "#1f5fa8", "#d9822b", "#8a8f98", "#b3261e", "#2e7d4f"
plt.rcParams.update({
    "figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "font.size": 9, "axes.titlesize": 10,
})  # fmt: skip


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path.name


def fig_sensitivity(sens, path: Path) -> str:
    ids = list(dict.fromkeys(sens["scenario"]))
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, sid in zip(axes.ravel(), ids, strict=False):
        d = sens[sens["scenario"] == sid].sort_values("value")
        ax.plot(d["entities"], d["true_positive"], "-o", color=BLUE, ms=4)
        cur = d[d["is_current"]].iloc[0]
        ax.plot(cur["entities"], cur["true_positive"], "o", color=ORANGE, ms=9, zorder=5,
                label=f"current ({d['parameter'].iloc[0]}={cur['value']:g})")  # fmt: skip
        for _, r in d.iterrows():
            ax.annotate(
                f"{r['value']:g}",
                (r["entities"], r["true_positive"]),
                fontsize=7,
                xytext=(3, -9),
                textcoords="offset points",
                color=GREY,
            )
        ax.set_xlim(0, max(d["entities"].max(), 1) * 1.12)
        ax.set_ylim(0, d["true_positive"].max() * 1.2 + 1)
        ax.set_title(sid)
        ax.set_xlabel("alerted entities")
        ax.set_ylabel("true positives")
        ax.legend(loc="upper left", fontsize=7)
    fig.suptitle("Threshold sensitivity: alerts vs true positives (labels = parameter value)")
    return _save(fig, path)


def _heat(ax, m, title, fmt="{:.0%}"):
    im = ax.imshow(m.to_numpy(dtype=float), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(
        range(m.shape[1]), [str(c).replace("TM-0", "S") for c in m.columns], rotation=45, ha="right"
    )
    ax.set_yticks(range(m.shape[0]), list(m.index))
    ax.grid(False)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.iloc[i, j]
            if v == v:
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8,
                        color="white" if v > 0.6 else "black")  # fmt: skip
    ax.set_title(title)
    return im


def fig_coverage(cov, path: Path) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    _heat(axes[0], cov["obvious"], "Typology x scenario: obvious variants")
    _heat(axes[1], cov["evasive"], "Typology x scenario: evasive variants (near the threshold)")
    return _save(fig, path)


def fig_overlap(jac, path: Path) -> str:
    fig, ax = plt.subplots(figsize=(5.5, 4.4))
    _heat(ax, jac, "Scenario overlap (Jaccard of alerted entities)", "{:.2f}")
    ax.set_yticks(range(jac.shape[0]), [str(i).replace("TM-0", "S") for i in jac.index])
    return _save(fig, path)


def fig_productivity(prod, path: Path) -> str:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(prod))
    ax.bar(x, prod["entities"], color=GREY, label="alerted entities")
    ax.bar(x, prod["true_positive"], color=BLUE, label="confirmed suspicious")
    for i, r in prod.iterrows():
        if r["entities"]:
            ax.text(i, r["entities"] + 1, f"{r['conversion']:.0%}", ha="center", fontsize=8)
    ax.set_xticks(x, [s.replace("TM-0", "S") for s in prod["scenario"]], rotation=20)
    ax.set_ylabel("entities")
    ax.set_title("Per-scenario productivity (label = alert-to-case conversion)")
    ax.legend()
    return _save(fig, path)


def fig_model_gain(scores, path: Path) -> str:
    y = scores["is_suspicious"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for col, name, c in (("risk_score", "combined risk score", BLUE), ("gbm_score", "gradient boosting", ORANGE),
                         ("iso_score", "isolation forest", GREY)):  # fmt: skip
        n, rec = gain_curve(scores[col].to_numpy(), y)
        ax.plot(n, rec, label=name, color=c)
    r = scores[scores["rule_alert"] == 1]
    ax.plot(
        len(r),
        r["is_suspicious"].sum() / y.sum(),
        "s",
        color=RED,
        ms=8,
        label="rule scenarios (as run)",
    )
    if "rule_alert_tuned" in scores:
        rt = scores[scores["rule_alert_tuned"] == 1]
        ax.plot(len(rt), rt["is_suspicious"].sum() / y.sum(), "D", color=GREEN, ms=7,
                label="rule scenarios (tuned, same volume)")  # fmt: skip
    ax.set_xlim(0, max(600, int(2 * len(r))))
    ax.set_xlabel("entities reviewed (ranked by score)")
    ax.set_ylabel("recall of suspicious entities")
    ax.set_title("Review workload vs recall (held-out entities)")
    ax.legend(loc="lower right")
    return _save(fig, path)


def fig_btl(atl, path: Path) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    r, s = atl["random"], atl["stratified"]
    rows = [("random sample", r["est"], r["est_lo"], r["est_hi"]),
            ("stratified by risk score", s["est"], s["est_lo"], s["est_hi"])]  # fmt: skip
    for i, (_, est, lo, hi) in enumerate(rows):
        ax.errorbar(est, i, xerr=[[est - lo], [hi - est]], fmt="o", color=BLUE, capsize=4)
    ax.axvline(atl["truth_btl_positives"], color=RED, ls="--", label="true BTL positives")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("suspicious entities below the line (estimate, 95% interval)")
    ax.set_title("BTL test: estimate vs known truth")
    ax.legend()
    return _save(fig, path)
