"""Build the audit-style findings report from computed results (no hard-coded numbers)."""

from __future__ import annotations

import pandas as pd

from .. import scenarios as S

SHORT = {m.SCENARIO_ID: m.__doc__.split("\n")[0].replace('"""', "").strip() for m in S.MODULES}


def _pct(x: float) -> str:
    return f"{x:.0%}"


def _md(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in fmt:
                v = fmt[c](v)
            elif isinstance(v, float):
                v = "-" if v != v else f"{v:.2f}"
            cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_findings(res: dict) -> list[dict]:
    prod, cov, sens, atl, m = (
        res["productivity"],
        res["coverage"],
        res["sensitivity"],
        res["atl"],
        res["metrics"],
    )
    F: list[dict] = []

    live = prod[prod["entities"] > 0]
    weak = live[live["conversion"] < 0.5].sort_values("conversion")
    tot_alerts, tot_tp = int(live["entities"].sum()), int(live["true_positive"].sum())
    if len(weak):
        worst = weak.iloc[0]
        rating = "High" if worst["conversion"] < 0.25 else "Medium"
        obs = (
            f"{len(weak)} of {len(live)} scenarios convert fewer than half of their alerted entities into "
            "confirmed suspicious activity: "
            + ", ".join(
                f"{r.scenario} {_pct(r.conversion)} ({r.true_positive}/{r.entities})"
                for r in weak.itertuples()
            )
            + f". Across all scenarios {tot_alerts} entity-alerts yielded {tot_tp} confirmed ({_pct(tot_tp / tot_alerts)})."
        )
        F.append({"title": "Low alert-to-case conversion in selected scenarios", "rating": rating,
                  "observation": obs,
                  "risk": "Analyst capacity is consumed by false positives; the institution cannot show that the "
                          "threshold is a justified trade-off between workload and detection.",
                  "recommendation": "Segment the scenario (e.g. merchants, cash-intensive businesses) with documented "
                                    "rationale, re-tune against the sensitivity curves in section 3, and track conversion "
                                    "monthly per scenario."})  # fmt: skip
    else:
        F.append({"title": "Scenario productivity", "rating": "Low",
                  "observation": f"All scenarios convert at least half of their alerts ({_pct(tot_tp / tot_alerts)} overall).",
                  "risk": "No material risk identified.", "recommendation": "Keep monitoring conversion monthly."})  # fmt: skip

    r, s = atl["random"], atl["stratified"]
    share = atl["truth_btl_positives"] / atl["truth_total_positives"]
    rating = "High" if share >= 0.20 else "Medium" if share >= 0.10 else "Low"
    F.append({
        "title": "Suspicious activity below the line (portfolio BTL test)", "rating": rating,
        "observation": (
            f"Below-the-line population: {atl['n_btl']} entities with no alert. A random sample of {r['sample']} contained "
            f"{r['k']} suspicious entities, extrapolating to about {r['est']:.0f} missed entities "
            f"({atl['conf']:.0%} interval {r['est_lo']:.0f} to {r['est_hi']:.0f}). A sample stratified by risk score "
            f"estimates {s['est']:.0f} ({s['est_lo']:.0f} to {s['est_hi']:.0f}). The labels show the true figure is "
            f"{atl['truth_btl_positives']}, i.e. {_pct(share)} of all {atl['truth_total_positives']} suspicious entities "
            f"are not alerted. Over {atl['coverage_check']['reps']} repeated random draws the interval contained the truth "
            f"{_pct(atl['coverage_check']['covered'])} of the time."),
        "risk": "Suspicious activity that no scenario flags is invisible to the first line and to regulators' "
                "reviews; a low BTL rate in a sample is not evidence of absence when prevalence is small.",
        "recommendation": "Run a BTL sample each cycle with a stratified design (more weight near the line), "
                          "document sample sizes and exceptions, and feed confirmed misses into scenario tuning."})  # fmt: skip

    if res["typed"]:
        ev, ob = cov["evasive"]["ANY"], cov["obvious"]["ANY"]
        gaps = ev[ev < 0.5]
        rating = "High" if (ev < 0.25).any() else "Medium" if len(gaps) else "Low"
        F.append({
            "title": "Typology coverage gaps for near-threshold (evasive) behaviour", "rating": rating,
            "observation": (
                "Detection rate by any scenario, obvious vs evasive variant: "
                + "; ".join(f"{t} {_pct(ob[t])} vs {_pct(ev[t])}" for t in ev.index)
                + ". Evasive variants are built to sit just under the default thresholds (for example 3 instead of 4 "
                  "deposits, deposits below 80% of the reporting line, a 55-70 hour forward instead of 48, a 70-day "
                  "instead of 90-day silence)."),
            "risk": "Customers who know or learn the thresholds can stay below them; static thresholds create a "
                    "predictable blind spot.",
            "recommendation": "Add behavioural or model-based challenger scenarios for the weakest typologies, vary "
                              "thresholds by segment, and include near-threshold cases in the BTL test design."})  # fmt: skip

    items, worse = [], 0
    recall_basis = "the scenario's own typology" if res["typed"] else "all labelled entities"
    for sid, d in sens.groupby("scenario"):
        cur = d[d["is_current"]].iloc[0]
        best = d.sort_values(["f1", "value"], ascending=[False, True]).iloc[0]
        if best["f1"] > cur["f1"] + 0.05:
            worse += 1
            items.append(f"{sid} {cur['parameter']}={cur['value']:g} (F1 {cur['f1']:.2f}, {cur['entities']} entities, "
                         f"{cur['true_positive']} TP) vs {best['value']:g} (F1 {best['f1']:.2f}, {best['entities']} entities, "
                         f"{best['true_positive']} TP)")  # fmt: skip
    F.append({
        "title": "Threshold calibration against sensitivity curves",
        "rating": "Medium" if worse else "Low",
        "observation": ("Settings where a different value improves F1 (precision on any suspicious entity, "
                        f"recall on {recall_basis}) by more than 0.05: " + ("; ".join(items) + "."
                        if items else "none; all current values are within 0.05 F1 of the best value on the grid.")
                        + " The sweep varies one parameter at a time and is evaluated on this dataset only."),
        "risk": "Without documented sensitivity analysis a threshold cannot be defended to a regulator, and "
                "may sit on a steep part of the curve where small changes move alert volume sharply.",
        "recommendation": "Keep the curves with each threshold decision, re-run after material data or product "
                          "changes, and record who approved the chosen operating point and why."})  # fmt: skip

    jac = res["jaccard"]
    ids = list(jac.index)
    pairs = [
        (a, b, jac.loc[a, b])
        for i, a in enumerate(ids)
        for b in ids[i + 1 :]
        if jac.loc[a, b] >= 0.15
    ]
    sets = res["sets"]
    uniq = {
        sid: len(sets.get(sid, set()) - set().union(*(v for k, v in sets.items() if k != sid)))
        for sid in ids
    }
    F.append({
        "title": "Scenario overlap and unique contribution", "rating": "Low" if not pairs else "Medium",
        "observation": ("Pairs with Jaccard >= 0.15: " + (", ".join(f"{a}/{b} {v:.2f}" for a, b, v in pairs) if pairs else "none")
                        + ". Entities alerted by only that scenario: "
                        + ", ".join(f"{sid} {n}" for sid, n in uniq.items()) + "."),
        "risk": "Heavy overlap means duplicate work and overstated coverage; scenarios with no unique entities may be "
                "redundant, while unique-only scenarios carry the diversity of the programme.",
        "recommendation": "Consolidate alerts per entity before case creation and review scenarios whose entire output "
                          "is covered by others."})  # fmt: skip

    eq = m["equal_recall"]
    ci = m["pr_auc_ci"]["gradient_boosting"]
    red_ci = eq.get("reduction_gbm_ci")
    F.append({
        "title": "ML scoring as a challenger to rule prioritisation", "rating": "Medium",
        "observation": (
            f"On held-out entities in the later window, PR-AUC is {m['pr_auc']['gradient_boosting']:.2f} for the tuned "
            f"gradient-boosting model (95% bootstrap interval {ci[0]:.2f} to {ci[1]:.2f}), "
            f"{m['pr_auc']['isolation_forest']:.2f} for the isolation forest, {m['pr_auc']['combined']:.2f} for the "
            f"combined score and {m['pr_auc']['rules_binary']:.2f} for the rules as run (prevalence {m['prevalence']:.1%}). "
            f"At the rules' recall of {_pct(eq['target_recall'])} gradient boosting needs {eq['gbm_alerts']} reviews "
            f"instead of {eq['rules_alerts']} ({_pct(eq['reduction_gbm'])} fewer"
            + (f", interval {_pct(red_ci[0])} to {_pct(red_ci[1])}" if red_ci else "")
            + f"); the combined score needs {eq['combined_alerts']} ({_pct(eq['reduction_combined'])} fewer). Reviewing the top "
            f"{atl['targeted']['n']} below-the-line entities by risk score finds {atl['targeted']['found']} suspicious entities "
            f"against {atl['targeted']['random_expected']:.1f} expected from random selection."),
        "risk": "The model is trained on labels that, in a real bank, come from past investigations and inherit the "
                "blind spots of the rules; on simulated data the numbers are optimistic. Unexplained scores are not "
                "acceptable to model-risk or audit functions.",
        "recommendation": "Deploy as a challenger and prioritisation layer first (not as a replacement), keep reason "
                          "codes with every score, monitor drift, and validate on out-of-time data before any "
                          "threshold is relaxed."})  # fmt: skip

    ru, rt, rl = m["rules"]["untuned"], m["rules"]["same_workload"], m["rules"]["lean"]
    gain = rt["recall"] - ru["recall"]
    F.append({
        "title": "Rule thresholds tuned on a tuning split versus current thresholds",
        "rating": "Medium" if gain > 0.02 or rt["precision"] > 1.2 * ru["precision"] else "Low",
        "observation": (
            f"Thresholds were optimised on the earlier window's tuning entities for maximum recall within each "
            f"scenario's current alert volume, then applied unchanged to the later window's held-out entities. Current "
            f"thresholds alert {ru['alerted_entities']} entities (recall {_pct(ru['recall'])}, precision "
            f"{_pct(ru['precision'])}). Tuned at the same volume: {rt['alerted_entities']} entities, recall "
            f"{_pct(rt['recall'])}, precision {_pct(rt['precision'])}. Tuned to a quarter of the volume: "
            f"{rl['alerted_entities']} entities, recall {_pct(rl['recall'])}, precision {_pct(rl['precision'])}."),
        "risk": "Thresholds set by judgement and left unreviewed carry unexplained alert volume; thresholds fitted on one "
                "period can decay when behaviour shifts.",
        "recommendation": "Run threshold tuning on a tuning split with a stated alert budget at each model "
                          "governance cycle, confirm on a later holdout as done here, and document the chosen operating "
                          "point."})  # fmt: skip

    ps = atl["per_scenario"]
    inconclusive = ps[(ps["found"] == 0) & (ps["truth_missed"] > 0)]
    F.append({
        "title": "Per-scenario BTL tests are under-powered at typical sample sizes",
        "rating": "Medium" if len(inconclusive) else "Low",
        "observation": (
            f"With about {int(ps['sample'].median())} entities sampled per scenario, {len(inconclusive)} of "
            f"{(ps['truth_missed'] > 0).sum()} scenarios with misses found no exception"
            + (": " + ", ".join(f"{r.scenario} (true misses {r.truth_missed}, needs n >= {r.n_to_detect_at_truth})"
                                for r in inconclusive.itertuples()) if len(inconclusive) else "")
            + "."),
        "risk": "A clean BTL sample can create false assurance when the typology is rare.",
        "recommendation": "State the tolerable miss rate up front and size samples with the zero-failure formula, or "
                          "use risk-stratified or targeted sampling for rare typologies."})  # fmt: skip

    for i, f in enumerate(F, 1):
        f["id"] = f"F-{i:02d}"
    return F


def render(res: dict, findings: list[dict], fig_rel: str = "../docs") -> str:
    m, atl, cfg = res["metrics"], res["atl"], res["cfg"]
    prod = res["productivity"].copy()
    desc = {
        "synthetic": "Data are synthetic; reviewer outcomes are taken from the injected labels, so every figure below is "
        "reproducible and none is an observation about a real institution.",
        "ibm": "Data are the IBM AMLworld HI-Small simulation; reviewer outcomes are its laundering flags (an account "
        "counts as suspicious if it touches a flagged transaction in the window). None of this is an observation "
        "about a real institution.",
        "saml": "Data are the SAML-D simulation (Oztas et al., 2023); reviewer outcomes are its laundering flags (an "
        "account counts as suspicious if it touches a flagged transaction in the window). None of this is an "
        "observation about a real institution.",
    }[res["source"]]
    d = m["design"]
    window = (
        f"the later window (from {d['cut'][:10]}), held-out test entities only"
        if d["cut"]
        else "the single observation window, held-out test entities only"
    )
    L = ["# Transaction-monitoring audit report", "",
         f"Generated by `tm-lab audit` from the actual run (seed {res['seed']}). " + desc, "",
         "## 1. Scope and approach", "",
         f"- Population: {m['n_entities']:,} entities in {window}, {m['n_positives']} labelled suspicious ({m['prevalence']:.1%}).",
         f"- Scenarios in scope: {', '.join(SHORT)}. Reporting threshold {cfg['reporting_threshold']:,.0f}.",
         "- Procedures: per-scenario productivity, scenario overlap, "
         + ("typology coverage, " if res["typed"] else "")
         + "one-at-a-time threshold sensitivity, above-the-line (ATL) and below-the-line (BTL) sampling, and ML challenger comparison.", "",
         "## 2. Summary of findings", "", "| ID | Rating | Finding |", "|---|---|---|"]  # fmt: skip
    L += [f"| {f['id']} | {f['rating']} | {f['title']} |" for f in findings]
    L += ["", "## 3. Detailed findings", ""]
    for f in findings:
        L += [f"### {f['id']} {f['title']} (rating: {f['rating']})", "",
              f"**Observation.** {f['observation']}", "", f"**Risk.** {f['risk']}", "",
              f"**Recommendation.** {f['recommendation']}", ""]  # fmt: skip
    pc = prod.copy()
    pc["conversion"] = pc["conversion"].map(lambda v: "-" if v != v else _pct(v))
    pc["own_precision"] = pc["own_precision"].map(lambda v: "-" if v != v else _pct(v))
    pc["own_recall"] = pc["own_recall"].map(lambda v: "-" if v != v else _pct(v))
    L += [
        "## 4. Supporting tables",
        "",
        "### 4.1 Scenario productivity",
        "",
        _md(pc),
        "",
    ]
    if res["typed"]:
        cov = res["coverage"]["all"].copy()
        cov = cov.map(lambda v: "-" if v != v else _pct(v))
        cov.insert(0, "typology", cov.index)
        L += ["### 4.2 Coverage: share of each typology detected by each scenario", "",
              _md(cov.reset_index(drop=True)), ""]  # fmt: skip
    L += ["### 4.3 Threshold sensitivity", ""]
    sens = res["sensitivity"].copy()
    sens["current"] = sens["is_current"].map({True: "yes", False: ""})
    L += [_md(sens[["scenario", "parameter", "value", "current", "entities", "true_positive", "precision", "own_recall", "f1"]],
              {"value": lambda v: f"{v:g}"}), "", "### 4.4 Per-scenario BTL tests", ""]  # fmt: skip
    ps = atl["per_scenario"].copy()
    L += [_md(ps, {"est_missed": lambda v: f"{v:.1f}", "est_lo": lambda v: f"{v:.1f}", "est_hi": lambda v: f"{v:.1f}",
                   "n_to_detect_at_truth": lambda v: "-" if v is None or v != v else int(v)}), ""]  # fmt: skip
    L += ["### 4.5 Figures", ""]
    L += [f"![{n}]({fig_rel}/{n})" for n in res["figures"]]
    L += ["", "---", "", "By Utku Fidan · utkufidan.com", ""]
    return "\n".join(L)
