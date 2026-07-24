# Where the cross-cohort signature failure lives

Progressors only, one row per donor. GSE79362: 33 donors. GSE94438: 75 donors. Shared genes: 15264.

## 1. Per-gene time-correlation does not reproduce

Correlating the two gene-wise correlation vectors gives Spearman -0.101 across 15264 genes. The 200 genes most strongly associated with time in GSE79362 have mean |rho| = 0.496 there and 0.099 in GSE94438, with sign agreement 32%.

The failure is not in how the signature aggregates genes. The individual genes do not carry the same time information in the two cohorts.

## 2. The Zak16 genes individually

| gene | GSE79362 | GSE94438 |
|---|---|---|
| GBP5 | -0.482 | -0.094 |
| BATF2 | -0.401 | +0.003 |
| SCARF1 | -0.441 | -0.135 |
| TRAV27 | +0.244 | -0.047 |
| ISG15 | -0.189 | +0.059 |
| ANKRD22 | -0.533 | -0.011 |
| ETV7 | -0.421 | -0.099 |
| SERPING1 | -0.334 | +0.020 |
| SAMD9L | -0.376 | -0.021 |
| IFIT2 | -0.350 | -0.048 |
| IFIT3 | -0.408 | -0.008 |
| IFI44L | -0.260 | +0.001 |
| CXCL10 | -0.160 | -0.047 |
| HERC5 | -0.328 | +0.012 |
| OAS1 | -0.318 | +0.031 |

Every gene that tracks time in GSE79362 collapses toward zero in GSE94438. The signature's interferon programme is specific to the first cohort's progression.

## 3. Matched time windows do not rescue reproduction

If GSE94438 simply sampled too early, restricting both cohorts to an early window would restore agreement. It does not.

| window (days) | GSE79362 Zak16-time (n) | GSE94438 Zak16-time (n) |
|---|---|---|
| [0, 180] | n/a (4) | +0.186 (21) |
| [0, 270] | +0.213 (14) | -0.094 (31) |
| [0, 365] | +0.186 (23) | +0.032 (40) |
| [365, 730] | +0.217 (9) | -0.326 (33) |

Restricting to a shared early window if anything worsens gene-level agreement, so timeline shift is not the explanation.

## 4. GSE94438 has a time signal, in different genes

The genes most associated with time in GSE94438:

| gene | GSE94438 rho | GSE79362 rho |
|---|---|---|
| TPM4 | -0.423 | +0.022 |
| MYZAP | -0.415 | -0.108 |
| YTHDF2P1 | -0.404 | +0.261 |
| MROH6 | +0.402 | -0.180 |
| SPAG1 | -0.385 | -0.215 |
| SYAP1 | -0.368 | +0.055 |
| SLC22A18 | +0.361 | +0.064 |
| CLP1 | -0.355 | +0.118 |
| SDAD1P2 | -0.352 | +0.016 |
| SGCD | -0.352 | -0.212 |

These are not the interferon genes of the published signature. Each cohort has a progression signal; the signals occupy different gene programmes. This is consistent with different progression biology in an adolescent latent-infection cohort versus adult household contacts, and it is not something batch correction can reach: there is no shared signal to preserve.
