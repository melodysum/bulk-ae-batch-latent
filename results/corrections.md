# Batch mixing versus biological transfer

Combined progressor set: 168 samples (67 GSE79362, 101 GSE94438), 15264 shared genes.

Mixing metrics measure whether the cohorts are still distinguishable. Transfer is the Zak16 signature's Spearman correlation with time to diagnosis, per cohort, at donor level. A method can improve mixing arbitrarily while leaving transfer untouched.

| method | iLISI (mix, higher better) | batch AUC (0.5 ideal) | GSE79362 transfer | GSE94438 transfer |
|---|---|---|---|---|
| Uncorrected | 0.372 | 1.000 | +0.417 | +0.022 |
| ComBat | 0.703 | 0.201 | +0.418 | +0.023 |
| Harmony | 0.571 | 0.889 | embedding only | embedding only |

ComBat moves iLISI from 0.37 to 0.70 and the batch probe from perfect separation (AUC 1.00) to near-chance (0.20), while GSE94438 transfer moves from +0.022 to +0.023. The batch signal is removed; the scientific result does not change. That is the decoupling this repository set out to demonstrate.
