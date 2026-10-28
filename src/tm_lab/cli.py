"""Command line: ``tm-lab generate | fetch-ibm | convert-saml | run-scenarios | evaluate | audit``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import scenarios
from .audit.analysis import own_entities
from .data import connect, load_config, load_dataset, save_dataset
from .generator import generate
from .windows import Design, median_cut, split_windows, window_labels


def _cmd_generate(a) -> None:
    txns, accounts, labels = generate(a.accounts, a.days, a.seed, a.start)
    save_dataset(Path(a.out), txns, accounts, labels)
    print(
        f"{len(txns):,} transactions, {len(accounts):,} accounts, {len(labels)} labelled entities -> {a.out}"
    )


def _cmd_fetch_ibm(a) -> None:
    from . import ibm

    csv = Path(a.raw) / ibm.MEMBER
    if not csv.exists():
        csv = ibm.fetch(Path(a.raw))
    txns, accounts, labels = ibm.convert(csv, a.days)
    save_dataset(Path(a.out), txns, accounts, labels)
    print(
        f"{len(txns):,} transactions, {len(accounts):,} accounts, {len(labels):,} suspicious -> {a.out}"
    )


def _cmd_run(a) -> None:
    txns, accounts, labels = load_dataset(a.data)
    cfg = load_config(a.config)
    con = connect(txns, accounts)
    alerts = scenarios.run_all(con, cfg)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    alerts.to_parquet(out / "alerts.parquet", index=False)
    sus = set(labels["entity"])
    rows = []
    for m in scenarios.MODULES:
        ents = set(alerts.loc[alerts["scenario_id"] == m.SCENARIO_ID, "entity"])
        own = own_entities(labels, m)
        rows.append({
            "scenario": m.SCENARIO_ID, "alerts": int((alerts["scenario_id"] == m.SCENARIO_ID).sum()),
            "entities": len(ents), "precision_any": round(len(ents & sus) / len(ents), 3) if ents else None,
            "recall_any": round(len(ents & sus) / len(sus), 3),
            "precision_own": round(len(ents & own) / len(ents), 3) if ents else None,
            "recall_own": round(len(ents & own) / len(own), 3) if own else None,
        })  # fmt: skip
    every = set(alerts["entity"])
    rows.append({
        "scenario": "ANY", "alerts": len(alerts), "entities": len(every),
        "precision_any": round(len(every & sus) / len(every), 3) if every else None,
        "recall_any": round(len(every & sus) / len(sus), 3),
        "precision_own": None, "recall_own": None,
    })  # fmt: skip
    table = pd.DataFrame(rows)
    table.to_csv(out / "scenario_metrics.csv", index=False)
    print(table.to_string(index=False))


def _cmd_convert_saml(a) -> None:
    from . import saml

    csv = Path(a.raw) / saml.MEMBER
    txns, accounts, labels = saml.convert(csv, a.account_frac, a.seed)
    save_dataset(Path(a.out), txns, accounts, labels)
    print(
        f"{len(txns):,} transactions, {len(accounts):,} accounts, {len(labels):,} labelled pairs -> {a.out}"
    )


def _design(a, txns) -> Design:
    cut = None if a.window == "none" else median_cut(txns)
    fractions = (0.5, 0.2, 0.3) if a.window == "none" else (0.6, 0.2, 0.2)
    return Design(cut=cut, fractions=fractions, seed=a.seed)


def _cmd_evaluate(a) -> None:
    from .experiment import run_experiment, save

    txns, _, labels = load_dataset(a.data)
    cfg = load_config(a.config)
    tuned = json.loads(Path(a.params).read_text()) if a.params else None
    out = Path(a.out)
    res = run_experiment(
        txns, labels, cfg, _design(a, txns), tune_seconds=a.tune_seconds,
        rule_seconds=a.rule_seconds, tuned=tuned, n_boot=a.bootstrap, cache=out / "cache",
        stop_after=None if a.stage == "all" else a.stage,
    )  # fmt: skip
    if a.stage != "all":
        out.mkdir(parents=True, exist_ok=True)
        (out / "tuned_params.json").write_text(json.dumps(res["tuned"], indent=2, default=str))
        print(f"stage {a.stage} done -> {out / 'tuned_params.json'}")
        return
    save(res, out)
    res["alerts"].to_parquet(out / "alerts.parquet", index=False)
    print(
        json.dumps(
            {k: v for k, v in res["metrics"].items() if k != "per_typology"}, indent=2, default=str
        )
    )


def _cmd_audit(a) -> None:
    from .audit import run_audit

    txns, _, labels = load_dataset(a.data)
    cfg = load_config(a.config)
    out = Path(a.out)
    scores = pd.read_parquet(out / "entity_scores.parquet")
    metrics = json.loads((out / "model_metrics.json").read_text())
    cut = metrics["design"]["cut"]
    _, late = split_windows(txns, None if cut is None else pd.Timestamp(cut))
    pop = set(scores["entity"])
    lab = window_labels(late, labels)
    lab = lab[lab["entity"].isin(pop)].reset_index(drop=True)
    alerts = pd.read_parquet(out / "alerts.parquet")
    alerts = alerts[alerts["entity"].isin(pop)].reset_index(drop=True)
    res = run_audit(
        connect(late),
        cfg,
        lab,
        scores,
        alerts,
        metrics,
        a.seed,
        Path(a.reports),
        Path(a.docs),
        pop,
        a.source,
    )
    print(f"report: {Path(a.reports) / 'tm_audit_report.md'}  findings: {len(res['findings'])}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="tm-lab", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="create a seeded synthetic dataset")
    g.add_argument("--accounts", type=int, default=2500)
    g.add_argument("--days", type=int, default=270)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--start", default="2025-04-01")
    g.add_argument("--out", default="data/sample")
    g.set_defaults(fn=_cmd_generate)
    f = sub.add_parser(
        "fetch-ibm", help="download IBM AMLworld HI-Small (public, no login) and convert"
    )
    f.add_argument("--raw", default="data/raw/ibm")
    f.add_argument("--out", default="data/raw/ibm_lab")
    f.add_argument("--days", type=int, default=None, help="keep only the first N days")
    f.set_defaults(fn=_cmd_fetch_ibm)
    c = sub.add_parser(
        "convert-saml", help="convert a hand-downloaded SAML-D CSV to the lab schema"
    )
    c.add_argument("--raw", default="data/raw/saml")
    c.add_argument("--out", default="data/raw/saml_lab")
    c.add_argument("--account-frac", type=float, default=1.0, help="keep this share of accounts")
    c.add_argument("--seed", type=int, default=42)
    c.set_defaults(fn=_cmd_convert_saml)
    for name, fn, help_ in (("run-scenarios", _cmd_run, "run rule scenarios on the whole period"),
                            ("evaluate", _cmd_evaluate, "tune and evaluate rules and models"),
                            ("audit", _cmd_audit, "ATL/BTL, sensitivity, coverage and report")):  # fmt: skip
        s = sub.add_parser(name, help=help_)
        s.add_argument("--data", default="data/sample")
        s.add_argument("--config", default=None, help="thresholds YAML (default: packaged)")
        s.add_argument("--out", default="outputs")
        s.add_argument("--seed", type=int, default=42)
        if name == "evaluate":
            s.add_argument("--window", choices=["median", "none"], default="median",
                           help="'median': fit early, test late; 'none': one window")  # fmt: skip
            s.add_argument(
                "--tune-seconds", type=float, default=360, help="Optuna budget per library"
            )
            s.add_argument(
                "--rule-seconds", type=float, default=60, help="Optuna budget per scenario"
            )
            s.add_argument(
                "--params", default=None, help="tuned_params.json to replay (skips tuning)"
            )
            s.add_argument("--bootstrap", type=int, default=500)
            s.add_argument("--stage", choices=["models", "rules", "all"], default="all",
                           help="run the long searches in steps; replay with --params")  # fmt: skip
        if name == "audit":
            s.add_argument("--source", choices=["synthetic", "ibm", "saml"], default="synthetic")
            s.add_argument("--reports", default="reports")
            s.add_argument("--docs", default="docs")
        s.set_defaults(fn=fn)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
