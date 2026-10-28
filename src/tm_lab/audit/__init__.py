"""Audit of transaction-monitoring thresholds: ATL/BTL, sensitivity, coverage, report."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from .. import scenarios as S
from . import analysis, plots, report


def run_audit(
    con,
    cfg: dict,
    labels: pd.DataFrame,
    scores: pd.DataFrame,
    alerts: pd.DataFrame,
    metrics: dict,
    seed: int,
    report_dir: Path,
    fig_dir: Path,
    population: set | None = None,
    source: str | None = None,
) -> dict:
    order = [m.SCENARIO_ID for m in S.MODULES]
    sets = analysis.entity_sets(alerts)
    typed = analysis.has_typologies(labels)
    res = {
        "cfg": cfg, "seed": seed, "source": source or ("synthetic" if typed else "ibm"), "metrics": metrics, "sets": sets, "typed": typed,
        "productivity": analysis.productivity(alerts, labels),
        "coverage": analysis.coverage(sets, labels, order),
        "sensitivity": analysis.sensitivity(con, cfg, labels, population),
        "atl": analysis.atl_btl(scores, labels, sets, seed),
    }  # fmt: skip
    res["jaccard"], res["overlap_counts"] = analysis.overlap(sets, order)
    fig_dir.mkdir(parents=True, exist_ok=True)
    res["figures"] = [
        plots.fig_sensitivity(res["sensitivity"], fig_dir / "threshold_sensitivity.png"),
        *([plots.fig_coverage(res["coverage"], fig_dir / "coverage_matrix.png")] if typed else []),
        plots.fig_overlap(res["jaccard"], fig_dir / "scenario_overlap.png"),
        plots.fig_productivity(res["productivity"], fig_dir / "scenario_productivity.png"),
        plots.fig_model_gain(scores, fig_dir / "model_gain.png"),
        plots.fig_btl(res["atl"], fig_dir / "btl_estimate.png"),
    ]
    res["findings"] = report.build_findings(res)
    report_dir.mkdir(parents=True, exist_ok=True)
    fig_rel = Path(os.path.relpath(fig_dir, report_dir)).as_posix()
    (report_dir / "tm_audit_report.md").write_text(report.render(res, res["findings"], fig_rel))
    summary = {
        "model": metrics,
        "productivity": res["productivity"].to_dict("records"),
        "atl": {k: v for k, v in res["atl"].items() if k != "per_scenario"},
        "per_scenario_btl": res["atl"]["per_scenario"].to_dict("records"),
        **(
            {
                "coverage_evasive": res["coverage"]["evasive"].round(3).to_dict(),
                "coverage_obvious": res["coverage"]["obvious"].round(3).to_dict(),
            }
            if typed
            else {}
        ),
        "findings": [{k: f[k] for k in ("id", "title", "rating")} for f in res["findings"]],
    }
    (report_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return res
