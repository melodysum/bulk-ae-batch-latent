"""
Bulk transcriptomics autoencoder + latent-space batch/biology diagnostics.

Design goals (given small-n / high-p bulk RNA-seq across cohorts):
  1. A *plain* AE first (no batch correction) so we can SEE how much batch
     leaks into the latent space -- that leakage is what motivates the
     discriminator / triplet extensions.
  2. Every claim about "batch vs biological signal" is QUANTIFIED, not just
     eyeballed on a 2D projection:
        - linear-probe accuracy (logistic regression on latent):
              batch acc  -> want ~ chance (good mixing)
              biology acc-> want high     (signal preserved)
        - silhouette by batch (want low) and by biology (want high)
  3. A LINEAR BASELINE (PCA, and PCA after limma-style removeBatchEffect)
     is computed on the same probes. If the AE latent cannot beat the linear
     baseline on batch-mixing, adding adversarial/triplet losses is patching
     the wrong architecture.

Swap `make_synthetic()` for your real loader (see load_real() stub) to run on
GSE79362 / GSE94438.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 0
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)


# --------------------------------------------------------------------------- #
# 1. Data                                                                      #
# --------------------------------------------------------------------------- #
def make_synthetic(n_per_batch=(220, 180), n_genes=2000, n_bio_genes=120,
                   n_batch_genes=300, batch_shift=1.8, bio_shift=1.2,
                   prog_frac=0.28):
    """Two cohorts, binary biology label (e.g. progressor vs non-progressor).

    Key realism knobs mirroring a TB cross-cohort setting:
      - biology is the WEAKER, sparser effect (n_bio_genes small, bio_shift low)
      - batch is the STRONGER, broader effect (n_batch_genes large, batch_shift high)
      - biology is imbalanced across batches (prog_frac differs per cohort) so
        batch and biology are partially CONFOUNDED -- the hard, realistic case.
    """
    Xs, batch, bio = [], [], []
    prog_fracs = [prog_frac, prog_frac * 0.55]          # imbalance across cohorts
    bio_genes   = rng.choice(n_genes, n_bio_genes, replace=False)
    for b, (n, pf) in enumerate(zip(n_per_batch, prog_fracs)):
        # baseline log-expression
        base = rng.normal(4.0, 1.0, size=(n, n_genes))
        # batch effect: additive shift on a broad, batch-specific gene set
        bg = rng.choice(n_genes, n_batch_genes, replace=False)
        base[:, bg] += (1 if b == 0 else -1) * batch_shift * rng.uniform(0.5, 1.5, len(bg))
        base += rng.normal(0, 0.15 * (b + 1), size=(n, n_genes))   # heteroscedastic noise
        # biology label + effect (same direction in both batches = the signal we want)
        y = (rng.uniform(size=n) < pf).astype(int)
        base[y == 1][:, bio_genes] += bio_shift            # (indexing below, kept explicit)
        base[np.ix_(y == 1, bio_genes)] += bio_shift
        Xs.append(np.maximum(base, 0)); batch.append(np.full(n, b)); bio.append(y)
    X = np.vstack(Xs)                                      # log-space "expression"
    return X, np.concatenate(batch), np.concatenate(bio)


def load_real():
    """Stub for GSE79362 / GSE94438.

    Return (X, batch, bio) where
      X     : (n_samples, n_genes) log-normalised expression (log1p CPM / VST),
              already restricted to the 14,128 shared genes,
      batch : cohort id per sample (0 = GSE79362, 1 = GSE94438),
      bio   : biological label per sample (progressor=1 / non-progressor=0,
              or active=1 / latent=0 -- your choice of contrast).
    NOTE: keep repeated-measures grouping in mind. For probing, ideally split
    by SUBJECT not by sample to avoid leakage across timepoints.
    """
    raise NotImplementedError("Plug in your merged GEO matrix + metadata here.")


def preprocess(X, n_hvg=1000):
    """HVG selection (variance) + per-gene z-score. Returns float32 array + hvg idx.

    HVG is deliberate: on 14k genes with a few hundred samples, dropping to the
    top ~1-2k most-variable genes cuts overfitting and speeds training. Fit HVG
    on the FULL matrix here; if you later add a held-out cohort, fit on train.
    """
    v = X.var(axis=0)
    hvg = np.argsort(v)[::-1][:n_hvg]
    Xh = X[:, hvg]
    Xz = (Xh - Xh.mean(0)) / (Xh.std(0) + 1e-8)
    return Xz.astype(np.float32), hvg


# --------------------------------------------------------------------------- #
# 2. Autoencoder                                                               #
# --------------------------------------------------------------------------- #
class AE(nn.Module):
    def __init__(self, n_in, hidden=256, latent=16, p_drop=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_in, hidden), nn.BatchNorm1d(hidden), nn.LeakyReLU(0.2),
            nn.Dropout(p_drop),
            nn.Linear(hidden, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden), nn.BatchNorm1d(hidden), nn.LeakyReLU(0.2),
            nn.Dropout(p_drop),
            nn.Linear(hidden, n_in),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def train_ae(Xz, latent=16, epochs=200, bs=64, lr=1e-3, wd=1e-4):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.tensor(Xz, device=dev)
    model = AE(Xz.shape[1], latent=latent).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)  # wd = L2 reg
    lossf = nn.MSELoss()
    n = X.shape[0]
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n, device=dev); tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = X[idx]
            xh, _ = model(xb)
            loss = lossf(xh, xb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        if (ep + 1) % 50 == 0:
            print(f"  epoch {ep+1:3d}  recon MSE {tot/n:.4f}")
    model.eval()
    with torch.no_grad():
        _, Z = model(X)
    return Z.cpu().numpy(), model


# --------------------------------------------------------------------------- #
# 3. Latent diagnostics (the actual "observe batch vs biological signal")      #
# --------------------------------------------------------------------------- #
def probe(Z, labels, groups=None):
    """Linear-probe accuracy: how linearly decodable is `labels` from Z.

    Higher = that factor is strongly encoded in the latent space.
    We report accuracy AND the majority-class baseline so imbalance is visible.
    """
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    acc = cross_val_score(Z, labels, clf, cv=cv, scoring="balanced_accuracy") \
        if False else cross_val_score(clf, Z, labels, cv=cv, scoring="balanced_accuracy")
    chance = 1.0 / len(np.unique(labels))
    return acc.mean(), acc.std(), chance


def sil(Z, labels):
    return silhouette_score(Z, labels)


def diagnostics(name, Z, batch, bio):
    bacc, bstd, bch = probe(Z, batch)
    yacc, ystd, ych = probe(Z, bio)
    print(f"\n[{name}]  latent dim = {Z.shape[1]}")
    print(f"  BATCH  balanced-probe acc = {bacc:.3f} ± {bstd:.3f}  (chance {bch:.2f})"
          f"   -> want LOW  | silhouette(batch)   = {sil(Z, batch):+.3f} (want ~0/neg)")
    print(f"  BIO    balanced-probe acc = {yacc:.3f} ± {ystd:.3f}  (chance {ych:.2f})"
          f"   -> want HIGH | silhouette(biology) = {sil(Z, bio):+.3f} (want positive)")
    return dict(batch_acc=bacc, bio_acc=yacc,
                batch_sil=sil(Z, batch), bio_sil=sil(Z, bio))


def linear_baselines(Xz, batch, bio, k=16):
    """PCA, and PCA after a limma-style removeBatchEffect (regress out batch)."""
    pca = PCA(k, random_state=SEED).fit_transform(Xz)
    # removeBatchEffect: subtract per-batch mean per gene, then PCA
    Xc = Xz.copy()
    for b in np.unique(batch):
        m = Xc[batch == b].mean(0)
        Xc[batch == b] -= m
    pca_rbe = PCA(k, random_state=SEED).fit_transform(Xc)
    return diagnostics("PCA (raw)", pca, batch, bio), \
           diagnostics("PCA (removeBatchEffect)", pca_rbe, batch, bio)


# --------------------------------------------------------------------------- #
# 4. Plot latent (2D PCA of the latent) coloured by batch and by biology       #
# --------------------------------------------------------------------------- #
def plot_latent(Z, batch, bio, path, title):
    p2 = PCA(2, random_state=SEED).fit_transform(Z)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for a, lab, name, cmap in [(ax[0], batch, "batch / cohort", "coolwarm"),
                               (ax[1], bio, "biology label", "viridis")]:
        for v in np.unique(lab):
            m = lab == v
            a.scatter(p2[m, 0], p2[m, 1], s=14, alpha=0.7,
                      label=f"{name.split('/')[0].strip()}={v}")
        a.set_title(f"latent coloured by {name}"); a.set_xlabel("latent-PC1")
        a.set_ylabel("latent-PC2"); a.legend(frameon=False, fontsize=8)
    fig.suptitle(title); fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("Generating synthetic cross-cohort bulk data ...")
    X, batch, bio = make_synthetic()
    print(f"  X {X.shape} | batches {np.bincount(batch)} | "
          f"bio per batch: {[int(bio[batch==b].sum()) for b in (0,1)]} progressors")

    Xz, hvg = preprocess(X, n_hvg=1000)

    print("\nTraining plain autoencoder (no batch correction) ...")
    Z, model = train_ae(Xz, latent=16, epochs=200)

    ae_stats = diagnostics("AE latent (no correction)", Z, batch, bio)
    lin_raw, lin_rbe = linear_baselines(Xz, batch, bio, k=16)

    plot_latent(Z, batch, bio, "ae_latent.png",
                "Plain AE latent -- batch still visible => motivates discriminator/triplet")

    print("\n=== summary (lower batch_acc = better mixing; higher bio_acc = signal kept) ===")
    for nm, d in [("AE (plain)", ae_stats), ("PCA raw", lin_raw),
                  ("PCA+removeBatchEffect", lin_rbe)]:
        print(f"  {nm:24s} batch_acc={d['batch_acc']:.3f}  bio_acc={d['bio_acc']:.3f}")
