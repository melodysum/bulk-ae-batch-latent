# Baseline: signature score vs time-to-diagnosis

Uncorrected log-CPM, progressors only, days > 0, aggregated to one row
per donor. Negative Spearman rho = score rises as diagnosis approaches,
which is the direction the signature literature predicts.

## Full range

| cohort | signature | donors | rho | 95% CI | p |
|---|---|---|---|---|---|
| GSE79362 | Zak16 | 33 | -0.449 | [-0.69, -0.13] | 0.00875 |
| GSE79362 | Eleven_gene | 33 | -0.426 | [-0.67, -0.10] | 0.0135 |
| GSE79362 | RISK4_ratio | 33 | +0.281 | [-0.07, +0.57] | 0.113 |
| GSE94438 | Zak16 | 75 | -0.022 | [-0.25, +0.21] | 0.852 |
| GSE94438 | Eleven_gene | 75 | -0.022 | [-0.25, +0.21] | 0.85 |
| GSE94438 | RISK4_ratio | 75 | -0.000 | [-0.23, +0.23] | 0.998 |

## Overlap window [91, 730] days

| cohort | signature | donors | rho | 95% CI | p |
|---|---|---|---|---|---|
| GSE79362 | Zak16 | 32 | -0.423 | [-0.67, -0.09] | 0.016 |
| GSE79362 | Eleven_gene | 32 | -0.421 | [-0.67, -0.09] | 0.0163 |
| GSE79362 | RISK4_ratio | 32 | +0.209 | [-0.15, +0.52] | 0.25 |
| GSE94438 | Zak16 | 73 | -0.022 | [-0.25, +0.21] | 0.851 |
| GSE94438 | Eleven_gene | 73 | -0.025 | [-0.25, +0.21] | 0.835 |
| GSE94438 | RISK4_ratio | 73 | +0.020 | [-0.21, +0.25] | 0.865 |

## Within-donor vs between-donor (Zak16)

Repeated sampling is a leakage hazard elsewhere in this project, but on
this axis it supplies a within-person contrast that cancels individual
baseline expression.

| cohort | longitudinal donors | within rho | within p | between rho | between p |
|---|---|---|---|---|---|
| GSE79362 | 19 | -0.493 | 0.000175 | -0.449 | 0.00875 |
| GSE94438 | 0 | n/a | n/a | -0.022 | 0.852 |

## Site decomposition within GSE94438 (Zak16)

A pooled rho near zero has two very different explanations: no signal, or
sites disagreeing in direction and cancelling.

| site | donors | rho | p |
|---|---|---|---|
| Ethiopia | 11 | -0.563 | 0.0714 |
| South Africa | 39 | +0.196 | 0.231 |
| The Gambia | 25 | -0.279 | 0.176 |

Cochran's Q = 6.21, df = 2, p = 0.045, I^2 = 68%.

Heterogeneity is supported: the sites differ by more than sampling noise.
