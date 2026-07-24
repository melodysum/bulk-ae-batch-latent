# Time-supervised encoder on the progressor axis

Pool: 789 samples, 15,264 shared genes. Labelled subset: 168 samples from 108 donors.
Metric: donor-level Spearman rho between predicted and observed days-to-diagnosis.
Positive is better. Arm A requires no fitting, so it transfers by construction.
Neural arms: 2 seeds, standard deviation over seeds in the last column.

## within GSE79362

| arm | donors | rho | sd |
|---|---|---|---|
| A  Zak16, no fitting | 33 | +0.417 |  |

## within GSE94438

| arm | donors | rho | sd |
|---|---|---|---|
| A  Zak16, no fitting | 75 | +0.022 |  |

## within GSE79362 (donor-grouped CV)

| arm | donors | rho | sd |
|---|---|---|---|
| B  Ridge on 2000 HVGs | 33 | +0.331 |  |
| C  AE latent + ridge head | 33 | +0.194 | 0.021 |
| D  AE + joint time head | 33 | +0.317 | 0.067 |

## train GSE94438 -> test GSE79362

| arm | donors | rho | sd |
|---|---|---|---|
| A  Zak16, no fitting | 33 | +0.417 |  |
| B  Ridge on 2000 HVGs | 33 | -0.110 |  |
| C  AE latent + ridge head | 33 | -0.290 | 0.010 |
| D  AE + joint time head | 33 | +0.007 | 0.024 |

## train GSE79362 -> test GSE94438

| arm | donors | rho | sd |
|---|---|---|---|
| A  Zak16, no fitting | 75 | +0.022 |  |
| B  Ridge on 2000 HVGs | 75 | -0.094 |  |
| C  AE latent + ridge head | 75 | -0.079 | 0.012 |
| D  AE + joint time head | 75 | -0.101 | 0.006 |
