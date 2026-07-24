# scVI: a deep generative model does not move transfer either

Raw counts, 168 progressor samples, 2000 highly variable genes, latent dim 30, 200 epochs. 14 of 16 Zak16 genes fall in the HVG set.

| metric | value |
|---|---|
| iLISI (mixing) | 0.212 |
| batch AUC | 0.995 |
| GSE79362 transfer | +0.393 |
| GSE94438 transfer | +0.033 |

GSE94438 transfer remains at chance, consistent with ComBat and Harmony. No correction method reached the biological result, because the two cohorts share no signal on this axis for a method to preserve.
