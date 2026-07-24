# Does site correction recover the GSE94438 time association?

Progressors only, days > 0. 101 samples, 75 donors. Target to beat: the GSE79362 estimate of rho = -0.449.

## Pooled association after each correction

| method | pooled rho | p | moved toward target? |
|---|---|---|---|
| raw | -0.022 | 0.852 | — |
| site-wise centering | -0.020 | 0.864 | +0.002 |
| ComBat | -0.009 | 0.937 | +0.013 |

Distance still to cover: 0.427. None of the corrections closes any part of it.

## Per-site correlations are unchanged

| site | donors | raw | site-wise centering | ComBat |
|---|---|---|---|---|
| Ethiopia | 11 | -0.563 | -0.563 | -0.563 |
| South Africa | 39 | +0.196 | +0.196 | +0.192 |
| The Gambia | 25 | -0.279 | -0.279 | -0.274 |

## Why: rank correlation is invariant under per-site monotone maps

Applying an arbitrary affine transform to one site's scores:

| transform | rho |
|---|---|
| `1.0*score + 0.0` | +0.196420 |
| `1.0*score + -5.0` | +0.196420 |
| `3.7*score + 2.1` | +0.196420 |
| `0.2*score + 100.0` | +0.196420 |

Identical to six decimal places. Centering subtracts a per-site constant; ComBat applies a per-site location and scale adjustment. Both are monotone within site, so neither can alter a within-site rank correlation — and once per-site offsets are gone, the pooled value is essentially fixed by the within-site values. **The result is a theorem, not an experiment.**

## Caveat on the site story

South Africa's +0.196 has a bootstrap 95% CI of [-0.121, +0.482], which crosses zero. Leave-one-out is stable (+0.162 to +0.258), so it is not one donor's doing, but **'the sites point in opposite directions' remains suggestive rather than established.** What is established is narrower and sufficient: the pooled association is absent, and per-site offsets are not the reason.

Sampling windows are also comparable across sites (medians 244 / 335 / 335 days), so timeline shift does not explain the site differences either.

## Consequence for the modelling plan

An adversarial encoder optimises site-indistinguishability. The cheapest way to reach that objective is to remove per-site location and scale — which is what ComBat did, and it changed nothing. For an encoder to help here it would have to learn a transform that is **non-monotone within site**, and nothing in the adversarial objective encourages that.

So the batch-removal framing is the wrong tool for this particular failure. If the encoder is to earn its place, it should be given the time axis as an explicit objective (a time-regression head, or a triplet defined on time), evaluated under leave-one-study-out — not asked to make cohorts indistinguishable and hoped to preserve biology as a side effect.
