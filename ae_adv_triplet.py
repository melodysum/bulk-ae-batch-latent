"""
Extension of ae_bulk.py: AE + batch discriminator (gradient reversal / DANN)
+ cross-batch triplet loss. Reuses the SAME data + diagnostics so results are
directly comparable to the plain AE.

Combined objective:
    L = L_recon                      (keep it an autoencoder)
      + lambda_adv  * L_adv(GRL)     (make batch UN-decodable from z)
      + lambda_trip * L_triplet      (keep same-biology-across-batch close)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ae_bulk import (make_synthetic, preprocess, diagnostics, linear_baselines,
                     plot_latent, SEED)

torch.manual_seed(SEED)
np.random.seed(SEED)


# --- gradient reversal layer -------------------------------------------------
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lamb):
        ctx.lamb = lamb
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lamb * g, None


def grad_reverse(x, lamb):
    return GradReverse.apply(x, lamb)


# --- model: encoder / decoder / batch discriminator --------------------------
class AEAdv(nn.Module):
    def __init__(self, n_in, n_batch, hidden=256, latent=16, p=0.1):
        super().__init__()
        # NOTE: LayerNorm, not BatchNorm. BatchNorm normalises using minibatch
        # statistics -> if a minibatch is batch-imbalanced it re-injects batch
        # structure and fights the discriminator. LayerNorm is per-sample = safe.
        self.enc = nn.Sequential(
            nn.Linear(n_in, hidden), nn.LayerNorm(hidden), nn.LeakyReLU(0.2),
            nn.Dropout(p), nn.Linear(hidden, latent))
        self.dec = nn.Sequential(
            nn.Linear(latent, hidden), nn.LayerNorm(hidden), nn.LeakyReLU(0.2),
            nn.Dropout(p), nn.Linear(hidden, n_in))
        self.disc = nn.Sequential(                       # batch discriminator D
            nn.Linear(latent, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, n_batch))

    def encode(self, x):
        return self.enc(x)

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z


# --- cross-batch batch-hard triplet loss -------------------------------------
def triplet_loss(z, bio, batch, margin=0.3):
    z = F.normalize(z, dim=1)                            # unit sphere -> dist in [0,2]
    """Online batch-hard triplets with a cross-batch preference for positives.

    For each anchor i:
      hardest positive = FARTHEST same-bio sample, preferring a DIFFERENT batch
                         (this is what forces batch-invariance of biology).
      hardest negative = NEAREST different-bio sample.
    Only anchors that have >=1 valid positive and negative contribute.
    """
    d = torch.cdist(z, z)                                # pairwise distances
    same_bio  = bio[:, None] == bio[None, :]
    diff_bio  = ~same_bio
    diff_batch = batch[:, None] != batch[None, :]
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)

    # positive mask: same biology, not self; add big bonus to cross-batch pairs
    pos_mask = same_bio & ~eye
    # prefer cross-batch positives: inflate their distance so "hardest" picks them
    pos_d = d + (pos_mask & diff_batch).float() * 1e3
    pos_d = pos_d.masked_fill(~pos_mask, -1.0)
    hardest_pos = pos_d.max(1).values                    # -1 where no positive
    # undo the +1e3 bonus for the loss value
    hardest_pos = torch.where(hardest_pos > 5e2, hardest_pos - 1e3, hardest_pos)

    neg_d = d.masked_fill(~diff_bio, float("inf"))
    hardest_neg = neg_d.min(1).values                    # inf where no negative

    valid = (hardest_pos >= 0) & torch.isfinite(hardest_neg)
    if valid.sum() == 0:
        return z.new_tensor(0.0)
    loss = F.relu(hardest_pos[valid] - hardest_neg[valid] + margin)
    return loss.mean()


def train_adv(Xz, bio, batch, latent=16, epochs=300, bs=128,
              lr=1e-3, wd=1e-4, lam_adv_max=1.5, lam_trip=0.5,
              warmup=40, k_disc=1):
    """Two-optimizer adversarial recipe (more stable than one-shot GRL):
      - opt_D updates ONLY the discriminator to classify batch from z.detach()
      - opt_G updates encoder+decoder to (a) reconstruct, (b) satisfy triplets,
        (c) CONFUSE D by pushing its output toward uniform (batch-invariant z).
    k_disc>1 keeps D near-optimal so the confusion signal is meaningful.
    lambda_adv is ramped in so reconstruction stabilises first.
    """
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.tensor(Xz, device=dev)
    y = torch.tensor(bio, device=dev)
    b = torch.tensor(batch, device=dev)
    n_batch = int(b.max().item()) + 1

    model = AEAdv(Xz.shape[1], n_batch, latent=latent).to(dev)
    opt_G = torch.optim.Adam(list(model.enc.parameters()) +
                             list(model.dec.parameters()), lr=lr, weight_decay=wd)
    opt_D = torch.optim.Adam(model.disc.parameters(), lr=lr)
    n = X.shape[0]
    log_uniform = -np.log(n_batch)                       # target: D outputs uniform

    for ep in range(epochs):
        prog = max(0.0, (ep - warmup) / max(1, epochs - warmup))
        lam_adv = lam_adv_max * (2.0 / (1.0 + np.exp(-5 * prog)) - 1.0)
        model.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb, bb = X[idx], y[idx], b[idx]

            # (1) train discriminator on frozen encoder output
            for _ in range(k_disc):
                with torch.no_grad():
                    z_det = model.encode(xb)
                d_loss = F.cross_entropy(model.disc(z_det), bb)
                opt_D.zero_grad(); d_loss.backward(); opt_D.step()

            # (2) train enc+dec: reconstruct + triplet + confuse D
            xh, z = model(xb)
            l_rec = F.mse_loss(xh, xb)
            l_tri = triplet_loss(z, yb, bb)
            logp = F.log_softmax(model.disc(z), dim=1)    # D held fixed here
            l_conf = -(logp.mean(1)).mean() * 0 + F.kl_div(
                logp, torch.full_like(logp, 1.0 / n_batch),
                reduction="batchmean")                    # KL(uniform || D) -> uniform
            loss = l_rec + lam_trip * l_tri + lam_adv * l_conf
            opt_G.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(                # tame adversarial spikes
                list(model.enc.parameters()) + list(model.dec.parameters()), 5.0)
            opt_G.step()

        if (ep + 1) % 75 == 0:
            print(f"  ep {ep+1:3d} lam_adv={lam_adv:.2f} rec={l_rec.item():.3f} "
                  f"D_acc≈{(model.disc(z.detach()).argmax(1)==bb).float().mean():.2f} "
                  f"tri={l_tri.item():.3f}")
    model.eval()
    with torch.no_grad():
        Z = model.encode(X)
    return Z.cpu().numpy()


if __name__ == "__main__":
    X, batch, bio = make_synthetic()
    Xz, _ = preprocess(X, n_hvg=1000)

    print("Training AE + batch discriminator (GRL) + cross-batch triplet ...")
    Z = train_adv(Xz, bio, batch)

    diagnostics("AE + discriminator + triplet", Z, batch, bio)
    plot_latent(Z, batch, bio, "ae_adv_latent.png",
                "AE + batch discriminator + triplet -- batch should now be mixed")
    print("\n(compare batch_acc against the plain-AE run: it should drop toward chance)")
