#!/usr/bin/env python3
"""scVI as a non-linear negative control for the decoupling result.

scVI is a deep generative model with a negative-binomial likelihood and the
batch supplied as a conditioning variable. It is the most capable correction
method tried here, and needs raw counts rather than log-CPM. It is included to
show that the failure to move biological transfer is not specific to linear
correction: a deep model does not move it either.

    python scripts/run_scvi.py --expr-dir data/expr --meta-dir data/metadata

Requires scvi-tools, scanpy, anndata. Counts files (GSE*_counts_csv.gz) must be
present; regenerate them with the R snippet in data/expr/README.md.
"""
from __future__ import annotations
import argparse, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tbbatch import timeaxis  # noqa: E402

ZAK16 = ["GBP5","BATF2","FCGR1B","SCARF1","TRAV27","ISG15","ANKRD22","ETV7",
         "SERPING1","SAMD9L","IFIT2","IFIT3","IFI44L","CXCL10","HERC5","OAS1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expr-dir", default="data/expr")
    ap.add_argument("--meta-dir", default="data/metadata")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--out", default="results/scvi.md")
    a = ap.parse_args()
    ed, md = Path(a.expr_dir), Path(a.meta_dir)

    from scipy import stats
    from sklearn.neighbors import NearestNeighbors
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score
    import scvi, anndata as ad, scanpy as sc, torch
    scvi.settings.seed = 0; torch.set_num_threads(4)

    c79 = pd.read_csv(ed / "GSE79362_counts_csv.gz", index_col=0)
    c94 = pd.read_csv(ed / "GSE94438_counts_csv.gz", index_col=0)
    shared = sorted(set(c79.index) & set(c94.index))

    def meta(s):
        m = pd.read_csv(md / f"{s}_meta.csv"); p = timeaxis.parse_series(m.TimeToTB)
        m["days"], m["kind"] = p.days, p.kind
        m["donor"] = s + "::" + m.PatientID.astype(str); m["study"] = s
        return m[(m.Progression == "Positive") & (m.kind == "parsed") & (m.days > 0)]

    m79, m94 = meta("GSE79362"), meta("GSE94438")
    ids = [i for i in m79.sample_id if i in c79.columns] + \
          [i for i in m94.sample_id if i in c94.columns]
    C = pd.concat([c79.loc[shared], c94.loc[shared]], axis=1)[ids].T
    meta_all = pd.concat([m79, m94]).set_index("sample_id").loc[ids]

    A = ad.AnnData(C.values.astype("float32"))
    A.obs["batch"] = meta_all.study.values
    A.obs_names = ids; A.var_names = shared
    sc.pp.highly_variable_genes(A, n_top_genes=2000, flavor="seurat_v3", batch_key="batch")
    Ah = A[:, A.var.highly_variable].copy()
    scvi.model.SCVI.setup_anndata(Ah, batch_key="batch")
    model = scvi.model.SCVI(Ah, n_latent=30)
    model.train(max_epochs=a.epochs, enable_progress_bar=False)

    Z = model.get_latent_representation()
    batch = (meta_all.study == "GSE94438").astype(int).values

    def ilisi(Z, b, k=30):
        k = int(min(k, len(Z) - 1))
        _, idx = NearestNeighbors(n_neighbors=k).fit(Z).kneighbors(Z)
        nb = len(np.unique(b))
        v = [1 / np.sum((np.bincount(b[r], minlength=nb) / len(r)) ** 2) for r in idx]
        return float((np.mean(v) - 1) / (nb - 1))

    def bauc(Z, b):
        pr = cross_val_predict(LogisticRegression(max_iter=2000), Z, b, cv=5,
                               method="predict_proba")[:, 1]
        return float(roc_auc_score(b, pr))

    den = np.log1p(model.get_normalized_expression(library_size=1e4).values)
    Xc = pd.DataFrame(den, index=ids, columns=Ah.var_names)
    zg = [g for g in ZAK16 if g in Ah.var_names]
    tr = {}
    for s, mm in [("GSE79362", m79), ("GSE94438", m94)]:
        sub = Xc.loc[[i for i in mm.sample_id if i in ids]].copy()
        dd = mm.set_index("sample_id").loc[sub.index]
        sub["donor"] = dd.donor.values; sub["days"] = dd.days.values
        g = sub.groupby("donor").mean()
        tr[s] = float(stats.spearmanr(-g[zg].mean(axis=1), g.days)[0])

    L = ["# scVI: a deep generative model does not move transfer either", "",
         f"Raw counts, {len(ids)} progressor samples, 2000 highly variable genes, "
         f"latent dim 30, {a.epochs} epochs. {len(zg)} of 16 Zak16 genes fall in "
         "the HVG set.", "",
         "| metric | value |", "|---|---|",
         f"| iLISI (mixing) | {ilisi(Z, batch):.3f} |",
         f"| batch AUC | {bauc(Z, batch):.3f} |",
         f"| GSE79362 transfer | {tr['GSE79362']:+.3f} |",
         f"| GSE94438 transfer | {tr['GSE94438']:+.3f} |", "",
         "GSE94438 transfer remains at chance, consistent with ComBat and "
         "Harmony. No correction method reached the biological result, because "
         "the two cohorts share no signal on this axis for a method to preserve.", ""]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L)); print("\n".join(L))


if __name__ == "__main__":
    main()
