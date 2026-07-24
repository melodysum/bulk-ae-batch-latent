#!/usr/bin/env python3
"""Baseline for the progressor time axis (README §3.7).

Question: in uncorrected log-CPM, how strongly does a published TB signature
score track days-to-diagnosis, within each cohort?

This is the number any learned representation has to beat. If an encoder does
not improve on a mean of sixteen genes, it has no reason to exist here.

Design decisions, all forced by the design audit:
  * progressors only          - the control groups are not comparable (§3.3)
  * days > 0                  - negative and zero times are prevalent disease (§3.7)
  * donor-level aggregation   - 2.03 samples/donor in GSE79362; sample-level
                                correlation would count the same person twice
  * two ranges reported       - full, and the [91, 730] overlap window, because
                                the cohorts sample different parts of the
                                timeline (§3.7.3)

Signature definitions are taken verbatim from the sibling DE analysis
(tb_analysis.R) so the two repositories cannot silently diverge.

    python scripts/run_baseline.py --expr-dir data/expr --out results/baseline_timeaxis.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tbbatch import timeaxis  # noqa: E402

SIGNATURES = {
    "Zak16": ["GBP5", "BATF2", "FCGR1B", "SCARF1", "TRAV27", "ISG15", "ANKRD22",
              "ETV7", "SERPING1", "SAMD9L", "IFIT2", "IFIT3", "IFI44L",
              "CXCL10", "HERC5", "OAS1"],
    "Eleven_gene": ["GBP5", "BATF2", "FCGR1B", "ANKRD22", "ETV7", "SERPING1",
                    "SAMD9L", "IFI44L", "CXCL10", "HERC5", "OAS1"],
}
RISK4_NUM = ["GBP5", "SEPTIN4"]
RISK4_DEN = ["CDO1", "TRAV27"]

WINDOW = (91.0, 730.0)


def mean_score(expr: pd.DataFrame, genes: list[str]) -> tuple[pd.Series, list[str]]:
    present = [g for g in genes if g in expr.index]
    missing = [g for g in genes if g not in expr.index]
    return expr.loc[present].mean(axis=0), missing


def risk4_ratio(expr: pd.DataFrame) -> tuple[pd.Series | None, list[str]]:
    need = RISK4_NUM + RISK4_DEN
    missing = [g for g in need if g not in expr.index]
    if missing:
        return None, missing
    num = expr.loc[RISK4_NUM].sum(axis=0)
    den = expr.loc[RISK4_DEN].sum(axis=0)
    return num / den, []


def spearman_ci(x: np.ndarray, y: np.ndarray, alpha: float = 0.05):
    """Fisher-z confidence interval for Spearman's rho."""
    n = len(x)
    rho, p = stats.spearmanr(x, y)
    if n < 5 or not np.isfinite(rho):
        return rho, p, (np.nan, np.nan)
    se = 1.0 / np.sqrt(n - 3)
    z = np.arctanh(np.clip(rho, -0.999999, 0.999999))
    crit = stats.norm.ppf(1 - alpha / 2)
    return rho, p, (np.tanh(z - crit * se), np.tanh(z + crit * se))


def build(study: str, expr_dir: Path, meta_dir: Path) -> pd.DataFrame:
    expr = pd.read_csv(expr_dir / f"{study}_logCPM_csv.gz", index_col=0)
    meta = pd.read_csv(meta_dir / f"{study}_meta.csv")

    parsed = timeaxis.parse_series(meta.TimeToTB)
    meta["days_to_tb"] = parsed.days
    meta["time_kind"] = parsed.kind
    meta["donor_id"] = study + "::" + meta.PatientID.astype(str)
    meta["study"] = study

    keep = (meta.Progression.eq("Positive")
            & meta.time_kind.eq("parsed")
            & meta.days_to_tb.gt(0)
            & meta.sample_id.isin(expr.columns))
    meta = meta[keep].copy()
    expr = expr[meta.sample_id.tolist()]

    for name, genes in SIGNATURES.items():
        s, miss = mean_score(expr, genes)
        meta[name] = s.values
        if miss:
            print(f"  {study} {name}: missing {miss}")
    r4, miss = risk4_ratio(expr)
    if r4 is not None:
        meta["RISK4_ratio"] = r4.values
    elif miss:
        print(f"  {study} RISK4_ratio: missing {miss} -> skipped")
    return meta


def donor_level(df: pd.DataFrame, score: str) -> pd.DataFrame:
    """One row per donor. Repeated samples are averaged rather than dropped:
    with 2.03 samples/donor in GSE79362, dropping would waste half the data,
    and keeping them would violate independence."""
    return df.groupby("donor_id").agg(
        days_to_tb=("days_to_tb", "mean"),
        score=(score, "mean"),
        n_samples=("sample_id", "size"),
    ).reset_index()


def within_between(df: pd.DataFrame, score: str) -> dict:
    """Split the association into a within-person and a between-person part.

    Within: centre both variables inside each donor, so every donor acts as
    its own control and individual baseline expression cancels. Only donors
    observed at more than one distinct time contribute.
    Between: one averaged row per donor.
    """
    lon = df.groupby("donor_id").filter(lambda d: d.days_to_tb.nunique() > 1)
    out = {"n_longitudinal_donors": int(lon.donor_id.nunique()),
           "n_longitudinal_samples": int(len(lon))}
    if len(lon) >= 5:
        w = lon.copy()
        w["d_c"] = w.days_to_tb - w.groupby("donor_id").days_to_tb.transform("mean")
        w["s_c"] = w[score] - w.groupby("donor_id")[score].transform("mean")
        r, p = stats.spearmanr(w.d_c, w.s_c)
        out["within_rho"], out["within_p"] = float(r), float(p)
    b = donor_level(df, score)
    r, p = stats.spearmanr(b.days_to_tb, b.score)
    out["between_rho"], out["between_p"] = float(r), float(p)
    out["n_donors"] = int(len(b))
    return out


def site_heterogeneity(df: pd.DataFrame, score: str, site_col: str = "GeographicalRegion") -> dict:
    """Cochran's Q on Fisher-z transformed correlations, one per site.

    A pooled correlation near zero can mean 'no signal' or 'sites pointing in
    opposite directions and cancelling'. Those call for completely different
    responses, so they must be distinguished before anything is concluded.
    """
    per, zs, ws = [], [], []
    for site, d in df.groupby(site_col):
        b = donor_level(d, score)
        if len(b) < 5:
            continue
        r, p = stats.spearmanr(b.days_to_tb, b.score)
        z = np.arctanh(np.clip(r, -0.999, 0.999))
        per.append({"site": site, "n_donors": len(b), "rho": float(r), "p": float(p)})
        zs.append(z); ws.append(len(b) - 3)
    if len(per) < 2:
        return {"per_site": per}
    zs, ws = np.array(zs), np.array(ws)
    zbar = float((ws * zs).sum() / ws.sum())
    Q = float((ws * (zs - zbar) ** 2).sum()); dfree = len(zs) - 1
    return {"per_site": per, "pooled_rho": float(np.tanh(zbar)),
            "Q": Q, "df": dfree, "p_heterogeneity": float(1 - stats.chi2.cdf(Q, dfree)),
            "I2_percent": float(max(0.0, (Q - dfree) / Q) * 100) if Q > 0 else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expr-dir", default="data/expr")
    ap.add_argument("--meta-dir", default="data/metadata")
    ap.add_argument("--out", default="results/baseline_timeaxis.md")
    a = ap.parse_args()

    expr_dir, meta_dir = Path(a.expr_dir), Path(a.meta_dir)
    frames = {s: build(s, expr_dir, meta_dir) for s in ("GSE79362", "GSE94438")}

    scores = [c for c in ("Zak16", "Eleven_gene", "RISK4_ratio")
              if all(c in f.columns for f in frames.values())]

    L = ["# Baseline: signature score vs time-to-diagnosis", "",
         "Uncorrected log-CPM, progressors only, days > 0, aggregated to one row",
         "per donor. Negative Spearman rho = score rises as diagnosis approaches,",
         "which is the direction the signature literature predicts.", ""]

    for label, lo, hi in [("Full range", 0.0, np.inf),
                          (f"Overlap window [{WINDOW[0]:.0f}, {WINDOW[1]:.0f}] days", *WINDOW)]:
        L += [f"## {label}", "",
              "| cohort | signature | donors | rho | 95% CI | p |",
              "|---|---|---|---|---|---|"]
        for study, f in frames.items():
            w = f[(f.days_to_tb >= lo) & (f.days_to_tb <= hi)]
            for sc in scores:
                d = donor_level(w, sc)
                rho, p, (lo_ci, hi_ci) = spearman_ci(d.days_to_tb.values, d.score.values)
                L.append(f"| {study} | {sc} | {len(d)} | {rho:+.3f} | "
                         f"[{lo_ci:+.2f}, {hi_ci:+.2f}] | {p:.3g} |")
        L.append("")

    primary = "Zak16"
    L += ["## Within-donor vs between-donor (Zak16)", "",
          "Repeated sampling is a leakage hazard elsewhere in this project, but on",
          "this axis it supplies a within-person contrast that cancels individual",
          "baseline expression.", "",
          "| cohort | longitudinal donors | within rho | within p | between rho | between p |",
          "|---|---|---|---|---|---|"]
    for study, f in frames.items():
        wb = within_between(f, primary)
        wr = f"{wb['within_rho']:+.3f}" if "within_rho" in wb else "n/a"
        wp = f"{wb['within_p']:.3g}" if "within_p" in wb else "n/a"
        L.append(f"| {study} | {wb['n_longitudinal_donors']} | {wr} | {wp} | "
                 f"{wb['between_rho']:+.3f} | {wb['between_p']:.3g} |")
    L.append("")

    L += ["## Site decomposition within GSE94438 (Zak16)", "",
          "A pooled rho near zero has two very different explanations: no signal, or",
          "sites disagreeing in direction and cancelling.", "",
          "| site | donors | rho | p |", "|---|---|---|---|"]
    het = site_heterogeneity(frames["GSE94438"], primary)
    for r in het["per_site"]:
        L.append(f"| {r['site']} | {r['n_donors']} | {r['rho']:+.3f} | {r['p']:.3g} |")
    if "Q" in het:
        L += ["",
              f"Cochran's Q = {het['Q']:.2f}, df = {het['df']}, "
              f"p = {het['p_heterogeneity']:.3f}, I^2 = {het['I2_percent']:.0f}%.",
              "",
              ("Heterogeneity is supported: the sites differ by more than sampling noise."
               if het["p_heterogeneity"] < 0.05 else
               "Heterogeneity is not established at this sample size.")]
    L.append("")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L))
    print("\n".join(L))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
