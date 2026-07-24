#!/usr/bin/env python3
"""Time-supervised representation learning on the progressor axis.

Section 5 of the README showed that batch correction cannot recover the
GSE94438 signature-time association, because the failure is heterogeneity in
the direction of the relationship rather than in its offset. An adversarial
objective targets the offset. This script tests the alternative: give the
encoder the time axis as an explicit supervised objective and see whether a
learned representation beats a fixed published signature.

Arms, in increasing order of capacity:

    A  Zak16 signature       fixed 16-gene mean, no fitting at all
    B  Ridge on HVGs         linear supervision, no representation learning
    C  AE latent + ridge     unsupervised representation, then a linear head
    D  AE + joint time head  representation and head trained together

Evaluation:

    within-cohort   donor-grouped 5-fold CV inside GSE79362, the cohort where
                    the association exists
    cross-cohort    leave-one-study-out, both directions

Metric is Spearman rho between predicted and observed days-to-diagnosis,
computed at donor level. The target to beat is arm A.

Leakage controls: gene standardisation and highly-variable-gene selection are
fitted on training folds only; donors never cross a fold boundary; the encoder
is pretrained without labels so the 108 labelled donors are never used to fit
the ~10^5 encoder parameters directly.

    python scripts/run_time_encoder.py --expr-dir data/expr --seeds 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tbbatch import timeaxis  # noqa: E402

ZAK16 = ["GBP5", "BATF2", "FCGR1B", "SCARF1", "TRAV27", "ISG15", "ANKRD22",
         "ETV7", "SERPING1", "SAMD9L", "IFIT2", "IFIT3", "IFI44L", "CXCL10",
         "HERC5", "OAS1"]
N_HVG = 2000
LATENT = 16


# --------------------------------------------------------------------------
# data

def load_cohort(study: str, expr_dir: Path, meta_dir: Path):
    expr = pd.read_csv(expr_dir / f"{study}_logCPM_csv.gz", index_col=0)
    meta = pd.read_csv(meta_dir / f"{study}_meta.csv")
    p = timeaxis.parse_series(meta.TimeToTB)
    meta["days"], meta["kind"] = p.days, p.kind
    meta["donor"] = study + "::" + meta.PatientID.astype(str)
    meta["study"] = study
    meta = meta[meta.sample_id.isin(expr.columns)].copy()
    return expr[meta.sample_id.tolist()], meta


def build(expr_dir: Path, meta_dir: Path):
    e79, m79 = load_cohort("GSE79362", expr_dir, meta_dir)
    e94, m94 = load_cohort("GSE94438", expr_dir, meta_dir)
    shared = sorted(set(e79.index) & set(e94.index))

    # unlabelled pool: every sample from both cohorts
    X_all = pd.concat([e79.loc[shared], e94.loc[shared]], axis=1).T
    meta_all = pd.concat([m79, m94], ignore_index=True).set_index("sample_id")
    meta_all = meta_all.loc[X_all.index]

    # labelled subset: progressors with a usable positive time
    lab = meta_all[(meta_all.Progression == "Positive")
                   & (meta_all.kind == "parsed")
                   & (meta_all.days > 0)]
    return X_all, meta_all, lab, shared


# --------------------------------------------------------------------------
# fold-local preprocessing

class Prep:
    """Standardise genes and pick HVGs. Fitted on training rows only."""

    def __init__(self, n_hvg: int = N_HVG):
        self.n_hvg = n_hvg

    def fit(self, X: np.ndarray):
        self.mu_ = X.mean(0)
        self.sd_ = X.std(0) + 1e-8
        Z = (X - self.mu_) / self.sd_
        self.hvg_ = np.argsort(-X.var(0))[: self.n_hvg]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mu_) / self.sd_)[:, self.hvg_]

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# --------------------------------------------------------------------------
# models

class AE(nn.Module):
    def __init__(self, d_in: int, latent: int = LATENT, hidden: int = 256, p_drop: float = 0.2):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hidden, latent),
        )
        self.dec = nn.Sequential(
            nn.Linear(latent, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, d_in),
        )

    def forward(self, x):
        z = self.enc(x)
        return z, self.dec(z)


class TimeHead(nn.Module):
    def __init__(self, latent: int = LATENT, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, z):
        return self.net(z).squeeze(-1)


def pretrain_ae(X: np.ndarray, epochs: int = 150, seed: int = 0, lr: float = 1e-3) -> AE:
    """Unsupervised. No labels are used, so the whole training pool contributes."""
    torch.manual_seed(seed)
    ae = AE(X.shape[1])
    opt = torch.optim.Adam(ae.parameters(), lr=lr, weight_decay=1e-5)
    Xt = torch.tensor(X, dtype=torch.float32)
    n = len(Xt)
    bs = min(64, n)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            _, xh = ae(Xt[idx])
            loss = ((xh - Xt[idx]) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            opt.step()
    return ae.eval()


def train_joint(X_lab: np.ndarray, y: np.ndarray, X_pool: np.ndarray,
                epochs: int = 300, seed: int = 0, lam_time: float = 1.0) -> tuple[AE, TimeHead]:
    """Reconstruction on the unlabelled pool, time regression on the labelled subset."""
    torch.manual_seed(seed)
    ae, head = AE(X_lab.shape[1]), TimeHead()
    opt = torch.optim.Adam(list(ae.parameters()) + list(head.parameters()),
                           lr=1e-3, weight_decay=1e-4)
    XL = torch.tensor(X_lab, dtype=torch.float32)
    YL = torch.tensor(y, dtype=torch.float32)
    XP = torch.tensor(X_pool, dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        idx = torch.randperm(len(XP))[:64]
        _, xh = ae(XP[idx])
        l_rec = ((xh - XP[idx]) ** 2).mean()
        z, _ = ae(XL)
        l_time = ((head(z) - YL) ** 2).mean()
        (l_rec + lam_time * l_time).backward()
        torch.nn.utils.clip_grad_norm_(list(ae.parameters()) + list(head.parameters()), 1.0)
        opt.step()
    return ae.eval(), head.eval()


# --------------------------------------------------------------------------
# evaluation

def donor_rho(pred: np.ndarray, y: np.ndarray, donors: np.ndarray) -> float:
    """Aggregate to one row per donor, then Spearman."""
    d = pd.DataFrame({"p": pred, "y": y, "donor": donors}).groupby("donor").mean()
    if len(d) < 5 or d.p.nunique() < 2:
        return np.nan
    return float(stats.spearmanr(d.p, d.y)[0])


def zak_score(expr_T: pd.DataFrame, genes: list[str]) -> np.ndarray:
    present = [g for g in genes if g in expr_T.columns]
    # signature is high near diagnosis, so negate to make it a time predictor
    return -expr_T[present].mean(axis=1).to_numpy()


def run_split(Xtr, ytr, Xte, yte, Xpool_tr, seed: int) -> dict:
    prep = Prep().fit(Xpool_tr)
    Atr, Ate, Apool = prep.transform(Xtr), prep.transform(Xte), prep.transform(Xpool_tr)
    out = {}

    ridge = RidgeCV(alphas=np.logspace(-1, 4, 20)).fit(Atr, ytr)
    out["B_ridge"] = ridge.predict(Ate)

    ae = pretrain_ae(Apool, seed=seed)
    with torch.no_grad():
        ztr = ae.enc(torch.tensor(Atr, dtype=torch.float32)).numpy()
        zte = ae.enc(torch.tensor(Ate, dtype=torch.float32)).numpy()
    out["C_ae_ridge"] = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(ztr, ytr).predict(zte)

    ae2, head = train_joint(Atr, ytr, Apool, seed=seed)
    with torch.no_grad():
        z2 = ae2.enc(torch.tensor(Ate, dtype=torch.float32))
        out["D_ae_timehead"] = head(z2).numpy()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expr-dir", default="data/expr")
    ap.add_argument("--meta-dir", default="data/metadata")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="results/time_encoder.md")
    a = ap.parse_args()

    X_all, meta_all, lab, shared = build(Path(a.expr_dir), Path(a.meta_dir))
    print(f"pool {X_all.shape[0]} samples x {len(shared)} shared genes")
    print(f"labelled {len(lab)} samples / {lab.donor.nunique()} donors\n")

    Xlab = X_all.loc[lab.index]
    y = lab.days.to_numpy(float)
    donors = lab.donor.to_numpy()
    study = lab.study.to_numpy()

    rows = []

    # ---- arm A: fixed signature, no fitting -------------------------------
    for s in ("GSE79362", "GSE94438"):
        m = study == s
        rows.append({"evaluation": f"within {s}", "arm": "A Zak16 (no fitting)",
                     "rho": donor_rho(zak_score(Xlab[m], ZAK16), y[m], donors[m]),
                     "sd": np.nan, "n_donors": len(set(donors[m]))})

    # ---- within-cohort, donor-grouped CV on GSE79362 ----------------------
    m = study == "GSE79362"
    Xw, yw, dw = Xlab[m].to_numpy(), y[m], donors[m]
    pool_meta = meta_all[meta_all.study == "GSE79362"]
    preds = {k: np.zeros((a.seeds, len(yw))) for k in ("B_ridge", "C_ae_ridge", "D_ae_timehead")}
    for seed in range(a.seeds):
        gkf = GroupKFold(n_splits=5)
        for tr, te in gkf.split(Xw, yw, groups=dw):
            # the unlabelled pool must exclude every donor in the test fold,
            # otherwise the encoder has seen the held-out people's expression
            held = set(dw[te])
            keep = pool_meta.index[~pool_meta.donor.isin(held)]
            pool_w = X_all.loc[keep].to_numpy()
            r = run_split(Xw[tr], yw[tr], Xw[te], yw[te], pool_w, seed)
            for k, v in r.items():
                preds[k][seed, te] = v
    for k, P in preds.items():
        rr = [donor_rho(P[s], yw, dw) for s in range(a.seeds)]
        rows.append({"evaluation": "within GSE79362 (donor-grouped CV)", "arm": k,
                     "rho": float(np.nanmean(rr)), "sd": float(np.nanstd(rr)),
                     "n_donors": len(set(dw))})

    # ---- cross-cohort, leave-one-study-out --------------------------------
    for tr_s, te_s in [("GSE94438", "GSE79362"), ("GSE79362", "GSE94438")]:
        mtr, mte = study == tr_s, study == te_s
        pool_tr = X_all.loc[meta_all.study == tr_s].to_numpy()
        acc = {k: [] for k in ("B_ridge", "C_ae_ridge", "D_ae_timehead")}
        for seed in range(a.seeds):
            r = run_split(Xlab[mtr].to_numpy(), y[mtr], Xlab[mte].to_numpy(),
                          y[mte], pool_tr, seed)
            for k, v in r.items():
                acc[k].append(donor_rho(v, y[mte], donors[mte]))
        for k, v in acc.items():
            rows.append({"evaluation": f"train {tr_s} -> test {te_s}", "arm": k,
                         "rho": float(np.nanmean(v)), "sd": float(np.nanstd(v)),
                         "n_donors": len(set(donors[mte]))})
        rows.append({"evaluation": f"train {tr_s} -> test {te_s}",
                     "arm": "A Zak16 (no fitting)",
                     "rho": donor_rho(zak_score(Xlab[mte], ZAK16), y[mte], donors[mte]),
                     "sd": np.nan, "n_donors": len(set(donors[mte]))})

    df = pd.DataFrame(rows)

    L = ["# Time-supervised encoder on the progressor axis", "",
         f"Pool: {X_all.shape[0]} samples, {len(shared)} shared genes. "
         f"Labelled subset: {len(lab)} samples from {lab.donor.nunique()} donors.",
         f"Seeds: {a.seeds}. Metric: donor-level Spearman rho between predicted and "
         "observed days-to-diagnosis. Higher is better; a fixed signature (arm A) "
         "requires no fitting and so transfers by construction.", ""]
    for ev in df.evaluation.unique():
        sub = df[df.evaluation == ev].sort_values("arm")
        L += [f"## {ev}", "", "| arm | donors | rho | sd over seeds |", "|---|---|---|---|"]
        for _, r in sub.iterrows():
            sd = "" if np.isnan(r.sd) else f"{r.sd:.3f}"
            rho = "n/a" if np.isnan(r.rho) else f"{r.rho:+.3f}"
            L.append(f"| {r.arm} | {r.n_donors} | {rho} | {sd} |")
        L.append("")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L))
    print("\n".join(L))
    df.to_csv(out.with_suffix(".csv"), index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
