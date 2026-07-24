# Bulk Transcriptomics Autoencoder: Batch vs Biological Signal in Latent Space

**English** | [中文](#中文说明)

A PyTorch autoencoder for bulk RNA-seq that **quantifies** how much of the latent space is taken up by batch effect versus biological signal, and then attempts cross-cohort alignment with a **batch discriminator (adversarial)** and a **triplet loss**.

Two things make this repo different from most batch-correction code:

1. It **audits the study design before training anything.** Whether adversarial correction is even *possible* on a given dataset collection is a property of the design, not of the method — and it is decidable from metadata alone. `scripts/run_audit.py` answers that question first, on real cohort metadata, in one command.
2. It **honestly documents how hard adversarial training is to tune and the exact hyperparameters that make it diverge** (see §6). In the small-n / high-p bulk setting, whether a method is *reproducible* and *worth the added complexity* matters more than the method itself.

> **Status.** The design-audit and split layer runs on real metadata from GSE79362 and GSE94438 and is fully tested. The model layer currently runs on synthetic data; the supervised cross-cohort arm turns out to be **unavailable by design** — the two cohorts share no supervised label axis, verified in §3.3. §3.6 sets out the sharper question they can answer instead.

---

## 1. Files

### Data layer — runs on real cohort metadata

| File | Contents |
|---|---|
| `configs/studies.yaml` | Study registry. Column mappings, label-axis definitions, and which experiments each axis can support. Adding a cohort means adding a block here, not editing code. |
| `src/tbbatch/metadata.py` | Schema adapters onto a canonical frame. Namespaces donor IDs by study; `pool()` **raises** rather than concatenating cohorts whose negative classes mean different things. |
| `src/tbbatch/audit.py` | Cramér's V with Bergsma correction (batch↔label association), repeated-measures structure, Monte-Carlo donor-leakage estimation. |
| `src/tbbatch/timeaxis.py` | Parses `TimeToTB` into days across cohorts. Handles unit mismatch, the non-`NA` `"---"` sentinel, and post-diagnosis negative times; returns an explicit `kind` per row rather than coercing. Also donor-level power and within-donor spread (§3.7.1–2). |
| `src/tbbatch/splits.py` | Leave-one-study-out (outer) and donor-grouped stratified k-fold (inner), plus a batch-balanced minibatch sampler. Every split is asserted before it is returned; a leaking split **raises** rather than quietly scoring. |
| `scripts/run_audit.py` | One command → `results/design_audit.md`. |
| `scripts/run_site_correction.py` | Tests whether site centering or ComBat recovers the GSE94438 association, with the invariance argument for why not. → `results/site_correction.md` |
| `scripts/run_baseline.py` | Signature score vs time-to-diagnosis on real expression: per-cohort correlation, within- vs between-donor split, and Cochran's Q site heterogeneity. → `results/baseline_timeaxis.md` |
| `data/metadata/` | Sample-level metadata for both cohorts (no expression values), so the audit is reproducible without a Bioconductor install. See its README for provenance. |
| `tests/test_splits.py` | 10 tests. Three construct leaking splits deliberately and assert the guards fire. |

### Model layer — currently synthetic

| File | Contents |
|---|---|
| `ae_bulk.py` | Data (synthetic demo + real-data hook), preprocessing, plain autoencoder, **latent diagnostics** (linear probe + silhouette), **linear baselines** (PCA / removeBatchEffect), latent visualization |
| `ae_adv_triplet.py` | Extends `ae_bulk.py` with a **batch discriminator (two-optimizer adversarial)** + **cross-batch triplet loss**, reusing the same diagnostics for a direct comparison |

## 2. Quick start

```bash
pip install -r requirements.txt

# 1. Audit the design first — no count matrices needed
python scripts/run_audit.py --registry configs/studies.yaml --root .
cat results/design_audit.md

# 2. Verify the leakage guards actually fire
python -m pytest tests/ -q

# 3. Model layer (synthetic data)
python ae_bulk.py          # plain AE + two linear baselines + diagnostics + latent plot
python ae_adv_triplet.py   # adversarial + triplet version, compared against the plain AE
```

To use your own data, fill in the `load_real()` stub in `ae_bulk.py`, returning `(X, batch, bio)` — `X` is a log-normalized expression matrix (log1p CPM / VST, restricted to shared genes), `batch` is the cohort id, and `bio` is the biological label. **Take the sample ordering and the split indices from the data layer** (`tbbatch.splits`) rather than splitting inside the model script; §3 explains why that matters here specifically.

---

## 3. Design audit: what the metadata says before any model runs

Everything in this section is a property of the study design. None of it requires expression values, and all of it constrains what the model layer is allowed to claim. Regenerate with `scripts/run_audit.py`; full output in [`results/design_audit.md`](results/design_audit.md).

Cohorts: **GSE79362** (ACS adolescent cohort, Zak et al. 2016) and **GSE94438** (GC6-74 household contacts, Suliman et al. 2018).

### 3.1 Batch is *not* meaningfully confounded with label — good news, and a correction

Cramér's V between the technical variable and the biological label:

| Association | Cramér's V | Reading |
|---|---|---|
| study × label (pooled, n=783) | **0.072** | negligible |
| site × label (within GSE94438, n=428) | **0.069** | negligible |

| study | label=0 | label=1 | n | positive rate |
|---|---|---|---|---|
| GSE79362 | 245 | 110 | 355 | 0.310 |
| GSE94438 | 327 | 101 | 428 | 0.236 |

This matters because adversarial batch removal has a hard precondition: if batch determined the label, the data would contain no counter-examples, and *any* method that erases batch would necessarily erase biology with it. **That failure mode does not apply here.** The design contains the counter-examples the method needs.

> **Correction to an earlier version of this README.** A previous revision justified reaching for adversarial correction partly on the grounds that "batch is confounded with the biological label — the progressor fraction differs between cohorts." Measured, the difference is 0.310 vs 0.236, Cramér's V = 0.072. **That justification does not hold for these two cohorts** and has been removed. The remaining justifications in §5.2 — non-linear batch structure, and wanting a reusable encoder — are unaffected.

### 3.2 The real leakage risk is donor-level, and it is severe

| study | samples | donors | donors w/ repeats | max/donor | effective-N inflation |
|---|---|---|---|---|---|
| GSE79362 | 355 | **144** | 105 | 6 | **2.47×** |
| GSE94438 | 428 | 334 | 79 | 4 | 1.28× |

Monte-Carlo over 2000 random 80/20 **sample-level** splits, counting donors that land on both sides:

| study | mean donors leaked | min | max | P(clean split) |
|---|---|---|---|---|
| GSE79362 | 47.7 | 35 | 60 | **0.000** |
| GSE94438 | 27.8 | 14 | 42 | **0.000** |

Not one clean split in 2000 draws, for either cohort. On GSE79362 a random split is not *risky*, it is **guaranteed** to put the same person's timepoints on both sides, and the effective sample size is 144 rather than 355.

This is the same repeated-measures structure that, in the [sibling DE analysis](https://github.com/melodysum/TB-Whole-Blood-Transcriptomics-GSE79362-GSE94438), collapsed the strict DEG count from 30 to 9 once `duplicateCorrelation` (ρ ≈ 0.31) accounted for it. `splits.py` carries that same constraint into the deep-learning protocol, where it is much easier to forget.

### 3.3 `TBStatus` cannot support a pooled transfer experiment — and why that is a *label* problem

| study | negative class | n |
|---|---|---|
| GSE79362 | LTBI | 245 |
| GSE94438 | household contact | 327 |

The two negative classes are **disjoint**. `study` therefore determines, with certainty, what "control" *means*. A pooled classifier can be right for the wrong reason, and — this is the important part — **no amount of adversarial batch removal can fix it, because the confound lives in the label definition, not in the expression values.** The encoder has nothing to erase.

`metadata.pool()` refuses this axis by default rather than concatenating silently.

**The `Progression` column does not rescue this — verified, not assumed.** Both cohorts also carry a `Progression` column, which looks like a shared axis. It is not. Inside each cohort it is *exactly collinear* with `TBStatus` (every off-diagonal count is zero):

| | Negative | Positive |
|---|---|---|
| GSE79362 `Progression` | 245 | 110 |
| GSE79362 `TBStatus` | LTBI **245** | PTB **110** |
| GSE94438 `Progression` | 327 | 101 |
| GSE94438 `TBStatus` | Control **327** | PTB **101** |

`Progression` is a **relabelling**, carrying no information `TBStatus` does not already have — and it is a actively hazardous one, because it renames two *different* control populations to the same string `"Negative"`. Any check that compares class labels as text would see two matching axes and pool them. The underlying populations are exactly as different as before.

`audit.independent_axes()` detects this automatically rather than trusting the registry annotation, on the principle that **a guard which depends on correct manual labelling is not a guard.**

**Conclusion: these two cohorts have no shared supervised axis.** That is a property of the study designs — GC6-74 recruited household contacts, ACS recruited an adolescent LTBI cohort — and no relabelling, and no batch-correction method, can manufacture one.

What the cohorts *do* support is set out in §3.6.

### 3.4 Protocol

- **Outer:** leave-one-study-out. The only split where the test batch was never seen.
- **Inner:** donor-grouped stratified 5-fold, on the training pool only.
- **Minibatches:** batch-balanced across cohorts — this both stabilizes the discriminator and closes the BatchNorm leakage path described in §6.
- **Enforcement:** assertions, not conventions. Validated splits from the current metadata:

```
LOSO[hold=GSE79362]  train=428  test=355   passed
LOSO[hold=GSE94438]  train=355  test=428   passed
donorCV[fold=0..4] (GSE79362)  ~115/29 donors  passed
```

### 3.5 A note on the sibling repo's AUC numbers

The companion DE analysis reports published TB signatures at AUC ≈ 0.77 in GSE79362 and ≈ 0.69 in GSE94438. **Those are not a cross-cohort transfer measurement.** They are the same signatures scored *within each cohort separately*, against different comparators (PTB vs LTBI, and PTB vs household contact). The gap reflects the difficulty of the two contrasts, not transfer loss.

The genuine cross-cohort quantities from that analysis are the concordance figures — gene-level logFC Spearman ρ = 0.641, Hallmark NES ρ = 0.815, over 14,128 shared genes. **A transfer baseline does not yet exist for these cohorts**; establishing one is what the LOSO protocol above is for.

### 3.6 What these cohorts can actually answer

With no shared supervised axis, the naive project ("does adversarial correction improve cross-cohort transfer AUC?") is not answerable here. The question that *is* answerable is sharper, and arguably more useful:

> **Cross-cohort transfer loss has at least two independent sources — technical batch effect, and comparator shift (what "control" means). Batch-correction methods address only the first. Can the two be told apart, and does correcting the first move the second at all?**

The audit already supplies the setup, and it makes a falsifiable prediction:

- batch↔label association is **negligible** (§3.1, V = 0.072), so batch confounding is *not* the obstacle;
- the comparator shift is **total** (§3.3), disjoint control populations;
- therefore a method that removes batch information should improve **batch-mixing metrics** (probe accuracy, iLISI, kBET) **while leaving transfer performance essentially unmoved.**

If that prediction holds, it is a clean demonstration of the failure mode this repo was built to expose: *the PCA looks better and nothing has actually improved* — but now with a mechanism attached rather than a warning. If it fails, that is more interesting still.

This costs no extra data and no extra code. It is the same encoder, the same splits, the same metrics — only the claim changes, from one the data cannot support to one it can.


### 3.7 `TimeToTB`: the one axis the cohorts genuinely share — after three traps

`TimeToTB` turns out to be the shared axis that `Progression` was not. It is not usable as delivered. Raw values, both cohorts:

```
GSE79362  chr  "642 Day(s)"  "0 Day(s)"  "---"  "-91 Day(s)"  "-253 Day(s)"  NA
GSE94438  chr  "22 month(s)"  "7 month(s)"  "3 month(s)"  NA
```

**Trap 1 — different units.** Days in one cohort, months in the other. Concatenating without conversion compresses one timescale by ~30×. `timeaxis.parse_series()` normalises both to days (30.4375 d/month); the observed ranges then overlap properly (GSE94438's 3–24 months = 91–730 d, against GSE79362's 0–642 d).

**Trap 2 — a sentinel that is not `NA`.** GSE79362 uses the literal string `"---"`. `is.na()` reports **FALSE** for it, so the missingness count of 91/355 taken from `is.na()` is an **undercount** by however many `"---"` rows exist. Every parsed row is returned with an explicit `kind` (`parsed` / `missing` / `sentinel` / `unparseable`) rather than being coerced.

**Trap 3 — negative times.** `"-91 Day(s)"`, `"-253 Day(s)"`. Under the "time from sampling to diagnosis" reading, these are samples drawn **after** diagnosis — prevalent disease, possibly on treatment. That is a different biological state, not an early progression signal, and pooling it with pre-diagnosis samples would put the strongest disease signal in the group meant to represent the earliest.

**How trap 2 actually bit — including once, here.** The `"---"` rows are not scattered. They align exactly with a class:

| `TimeToTB == "---"` | Negative | Positive |
|---|---|---|
| FALSE (a real value) | **0** | 98 |
| TRUE (`"---"`) | **166** | 0 |
| `NA` | 79 | 12 |

So true missingness in GSE79362 is **257/355**, not the 91/355 that `is.na()` reports — a **2.8× undercount**.

An earlier revision of this section read the `is.na()` crosstab and concluded that "166 negatives carry a follow-up time, therefore `(TimeToTB, Progression)` is right-censored survival data." **That was wrong.** All 166 are the sentinel; they hold no value at all. No non-progressor in either cohort has a time. There are no censoring times, hence no risk set, hence **no pooled survival model** — and the reason is the opposite of the one first given.

`timeaxis.audit_series()` classifies those rows correctly and reports `negatives_with_time = 0`. The error came from reasoning off the raw R output instead of running the parser that exists for exactly this. A guard you route around is not a guard; `tests/test_splits.py::test_sentinels_are_not_counted_as_censoring_times` reproduces the real 166/79/98/12 structure so the conclusion cannot silently regress.

**The corrected structure is simpler and better.** Both cohorts populate `TimeToTB` for **progressors only** — 98 in GSE79362, 101 in GSE94438. Symmetric, no censoring, no asymmetry to reconcile.

#### 3.7.1 Donor counts change what this axis is for

Those are sample counts. The independent units are people, and the count survives a cleaning cascade. Run on the real metadata:

| step | GSE79362 (samples/donors) | GSE94438 (samples/donors) |
|---|---|---|
| all samples | 355 / 144 | 434 / 334 |
| progressors | 110 / 40 | 101 / **75** |
| + parseable time | 98 / 33 | 101 / 75 |
| + drop days < 0 (post-diagnosis) | 85 / 33 | 101 / 75 |
| + drop days = 0 (day of diagnosis) | **67 / 33** | **101 / 75** |

Two things to note. GSE79362 loses 7 donors at the parsing step — those donors have progressor samples but no usable time. And GSE94438 has **75** progressor donors, not the 76 an earlier count reported: in R, `x[cond]` returns `NA` rows wherever `cond` is `NA`, and with 6 `NA` progression labels `unique()` counted `NA` as a donor.

Final: **168 samples, 108 donors.**

Power to detect a monotone association with time-to-diagnosis (α = 0.05, two-sided):

| true ρ | n=33 (GSE79362) | n=75 (GSE94438) | n=108 (pooled) |
|---|---|---|---|
| 0.3 | 0.40 | 0.75 | 0.89 |
| 0.4 | 0.64 | 0.95 | 0.99 |
| 0.5 | 0.85 | 1.00 | 1.00 |
| 0.6 | 0.97 | 1.00 | 1.00 |

**So this axis is an evaluation set, not a training set.** 116 people against ~14,000 genes cannot support fitting an encoder; it comfortably supports *testing* one for moderate effects. That fixes the architecture:

1. train the encoder **unsupervised on all 783 samples / 478 donors** — labels are not needed, so the full collection contributes;
2. evaluate the resulting latent on the progressor time axis, cross-cohort.

The two arms use different data for different jobs, and neither is asked to do something its n cannot support.

#### 3.7.2 Repeated measures — an asset in one cohort, an illusion in the other

Everywhere else in this audit, repeated sampling is a leakage hazard (§3.2). On this axis it can invert: a person observed at several distances from diagnosis supplies a **within-person contrast**, which removes individual baseline expression as a nuisance and is far more powerful than comparing different people.

Both cohorts appear to offer this. Only one does:

| cohort | donors with >1 progressor sample | donors with >1 **distinct time** | median within-donor spread |
|---|---|---|---|
| GSE79362 | 19 | **19** | **360 days** |
| GSE94438 | 20 | **0** | — |

Every one of GSE94438's 20 multi-sample donors has a within-donor time spread of **exactly 0 days**. Those are replicates at a single timepoint, not longitudinal sampling.

An earlier version of `donor_time_structure()` tested only `count > 1` and reported `supports_within_donor_design = True` for GSE94438. That was wrong, and it is the kind of wrong that survives review: the sample counts look longitudinal, and nothing errors. The helper now requires a non-zero spread, and `tests/test_splits.py::test_same_timepoint_replicates_are_not_a_longitudinal_design` locks it.

The practical consequence: the two cohorts are not interchangeable halves of one experiment. **GSE79362 (33 donors, 19 longitudinal, 360-day median spread) is where a within-donor mixed model belongs; GSE94438 (75 donors, cross-sectional) is the between-person replication.** Design the split accordingly, and keep donor grouping — the within-person contrast is only valid if the folds respect it.

#### 3.7.3 A third kind of shift: the cohorts sample different parts of the timeline

Distinct from batch effect (§3.1) and comparator shift (§3.3), and equally beyond the reach of batch correction:

| cohort | n | min | Q1 | median | Q3 | max |
|---|---|---|---|---|---|---|
| GSE79362 | 67 | 4 | 191 | **274** | 436 | 894 |
| GSE94438 | 101 | 91 | 213 | **426** | 639 | 730 |

GSE94438's samples sit systematically **further from diagnosis** — median 426 days against 274, and its window never opens closer than 91 days, while GSE79362 reaches down to 4.

This matters because TB progression signal is known to strengthen as diagnosis approaches; that proximity dependence is the central finding of the signature literature. So a model trained on one cohort and tested on the other faces covariate shift **on the target variable itself**. Transfer could fail for that reason alone, with batch effect entirely innocent.

Unlike the other two shifts, this one is fixable by design. Restricting both cohorts to the overlapping window [91, 730] days:

| cohort | samples / donors | median |
|---|---|---|
| GSE79362 | 54 / 32 | 282 d |
| GSE94438 | 96 / 73 | 411 d |
| **total** | **150 / 105** | — |

The medians still differ, so window restriction alone does not equalise the cohorts, but it removes the non-overlapping tails and makes the comparison interpretable. **Report both the full-range and window-restricted results** — the gap between them is itself a measurement of how much apparent transfer loss is timeline shift rather than technical batch.

**What does work: the progressor-only window.** Restricted to progressors, "days from sampling to TB diagnosis" means the same thing in both cohorts — and because the control group is excluded entirely, **the disjoint-control confound of §3.3 disappears.** n ≈ 199 before cleaning (98 + 101), fewer after sentinels and negative times are removed.

That is the second real experiment this collection supports, alongside §3.6. It is small, but it is the only cross-cohort supervised target here that is not confounded by design.

---

## 3A. Corrections log

Every claim below was made, checked against the data, and found wrong. They are kept rather than quietly edited out, because the sequence *is* the analysis: each retraction narrowed what this dataset collection can support, and the final design is what survived.

| # | Claim as made | Check that falsified it | What it actually was | Code change |
|---|---|---|---|---|
| 1 | Published-signature AUC 0.77 → 0.69 is a **cross-cohort transfer loss** baseline | Read `signature_AUC_summary.csv` | Same signatures scored *within* each cohort separately, against different comparators. Difference reflects contrast difficulty. No transfer experiment exists. | §3.5 added |
| 2 | Adversarial correction is warranted because **batch is confounded with the label** in these cohorts | Cramér's V on real metadata | 0.310 vs 0.236, **V = 0.072**, negligible. The doom scenario does not apply; the design contains the counter-examples the method needs. | `audit.confounding_report()`; justification struck from §5.2 |
| 3 | The disjoint-control problem is solved by switching to the **`Progression` axis** | `table(Progression, TBStatus)` | Exactly collinear inside both cohorts — a pure relabelling. Worse, it renames two *different* control populations to the same string `"Negative"`, so a text-matching guard would pool them. | `audit.independent_axes()` — detects collinearity empirically instead of trusting the registry |
| 4 | 166 GSE79362 negatives carry follow-up times, so this is **right-censored survival data** | `sum(TimeToTB == "---")` | All 166 are the sentinel — no value at all. Neither cohort has censoring times. True missingness 257/355, not 91/355. | `timeaxis` sentinel handling; regression test on the real 166/79/98/12 structure |
| 5 | GSE94438 has **76** progressor donors | `nunique()` on parsed metadata | 75. In R, `x[cond]` returns `NA` rows wherever `cond` is `NA`; with 6 `NA` progression labels, `unique()` counted `NA` as a donor. | §3.7.1 recomputed from the CSV |
| 6 | GSE94438's 1.33 samples/donor still **supports a within-person contrast** | within-donor time spread | All 20 multi-sample donors have a spread of exactly 0 days — replicates at one timepoint, not longitudinal. | `donor_time_structure()` now requires non-zero spread; regression test added |

Four of these (2, 4, 5, 6) share one shape: **a summary statistic was trusted where the underlying rows were not inspected.** Errors 4 and 6 are the sharper pair, because in each case the tool that catches the mistake had already been written — and in 6 the tool itself encoded the same shallow test (`count > 1`) that the prose did.

The methodological residue is the design rule the repo now follows: *a guard that depends on correct manual annotation, or that can be routed around by eyeballing raw output, is not a guard.* Hence `independent_axes()` measures collinearity rather than reading a config field, and `audit_series()` reports a `kind` per row rather than a missingness count.

---

## 3B. First results on real expression data

`scripts/run_baseline.py` -> [`results/baseline_timeaxis.md`](results/baseline_timeaxis.md). Uncorrected log-CPM, progressors only, days > 0, one row per donor. Signature definitions taken verbatim from the [sibling DE analysis](https://github.com/melodysum/TB-Whole-Blood-Transcriptomics-GSE79362-GSE94438) so the two repositories cannot silently diverge. Shared gene space after intersection: **15,264**.

A negative rho means the score rises as diagnosis approaches — the direction the signature literature predicts.

### The relationship holds in one cohort and vanishes in the other

| cohort | signature | donors | rho | 95% CI | p |
|---|---|---|---|---|---|
| GSE79362 | Zak16 | 33 | **-0.449** | [-0.69, -0.13] | 0.009 |
| GSE79362 | Eleven_gene | 33 | -0.426 | [-0.67, -0.10] | 0.014 |
| GSE94438 | Zak16 | 75 | **-0.022** | [-0.25, +0.21] | 0.85 |
| GSE94438 | Eleven_gene | 75 | -0.025 | [-0.25, +0.21] | 0.85 |

Restricting both cohorts to the [91, 730] day overlap window barely moves either number (-0.423 and -0.022), so **timeline shift (§3.7.3) does not explain the discrepancy.**

The GSE94438 null is not merely underpowered: at 75 donors the confidence interval excludes any association stronger than |rho| = 0.25.

### The within-person contrast is stronger, as §3.7.2 predicted

| cohort | longitudinal donors | within rho | p | between rho | p |
|---|---|---|---|---|---|
| GSE79362 | 19 | **-0.493** | **0.00018** | -0.449 | 0.009 |
| GSE94438 | **0** | n/a | n/a | -0.022 | 0.85 |

Using each donor as their own control gives a stronger association from *fewer* people, and a p-value two orders of magnitude smaller. GSE94438 contributes zero longitudinal donors — its repeats are same-timepoint replicates (§3.7.2), so this analysis is unavailable there by construction.

### The GSE94438 null is sites cancelling, not absence of signal

| site | donors | rho | p |
|---|---|---|---|
| Ethiopia | 11 | **-0.563** | 0.071 |
| The Gambia | 25 | **-0.279** | 0.18 |
| South Africa | 39 | **+0.196** | 0.23 |
| pooled | 75 | -0.022 | 0.85 |

Cochran's Q = 6.21, df = 2, **p = 0.045**, **I² = 68%**.

No single site reaches significance on its own, but they differ by more than sampling noise, and the largest site points the *opposite way*. The pooled near-zero is a cancellation.

**This is the finding the project was missing.** It converts the vague goal "does batch correction help?" into a quantitative target:

> GSE94438's signature-time association is destroyed by between-site heterogeneity. Does site correction recover it, and does the recovered value approach the GSE79362 estimate of -0.45?

Success and failure are both legible in advance. And it sharpens §3.6: site here is not additive noise — it **inverts the sign of a biological relationship**, which is a far more demanding test than improving a mixing metric.

### Caveats

- Site strata are small (11–39 donors) and the decomposition was not pre-registered. Q = 6.21 at p = 0.045 is suggestive, not decisive; it justifies the next experiment rather than concluding one.
- `FCGR1B` is absent from the GSE94438 matrix, so its Zak16 and Eleven_gene scores use one gene fewer. Same limitation as the sibling analysis.
- The shared gene count (15,264) exceeds the sibling repo's 14,128 because filtering thresholds differ (`>= 20 samples with CPM > 1` here vs a group-size rule there). Neither is wrong; they are not interchangeable.
- RISK4 is a ratio rather than a mean and behaves differently (+0.281 in GSE79362, wrong direction, n.s.). Not pursued further here.

---

## 3C. Site correction does not recover the signal — and provably cannot

`scripts/run_site_correction.py` → [`results/site_correction.md`](results/site_correction.md). Run before writing any encoder, on the reasoning that if a one-line linear correction already works then a neural method must beat it, and if no correction can work then the batch-removal framing itself is wrong.

| method | pooled rho | p | movement |
|---|---|---|---|
| raw | −0.022 | 0.85 | — |
| site-wise gene centering | −0.020 | 0.86 | +0.002 |
| ComBat (site as batch) | −0.009 | 0.94 | +0.013 |
| **GSE79362 target** | **−0.449** | | **0.427 to cover** |

Nothing moves. And the per-site correlations are untouched:

| site | donors | raw | centering | ComBat |
|---|---|---|---|---|
| Ethiopia | 11 | −0.563 | −0.563 | −0.563 |
| South Africa | 39 | +0.196 | +0.196 | +0.192 |
| The Gambia | 25 | −0.279 | −0.279 | −0.274 |

### The reason is a theorem, not a result

Spearman correlation is invariant under any monotone transform. Applying arbitrary affine maps to one site's scores:

| transform | rho |
|---|---|
| `1.0*score + 0.0` | +0.196420 |
| `3.7*score + 2.1` | +0.196420 |
| `0.2*score + 100.0` | +0.196420 |

Identical to six decimal places. **Centering subtracts a per-site constant; ComBat applies a per-site location and scale adjustment. Both are monotone within site, so neither can alter a within-site rank correlation** — and once per-site offsets are gone, the pooled value is essentially fixed by the within-site values.

The failure in GSE94438 is heterogeneous **slopes**, not heterogeneous **offsets**. Batch correction addresses offsets.

### This changes the modelling plan

An adversarial encoder optimises site-indistinguishability. The cheapest route to that objective is removing per-site location and scale — exactly what ComBat did, and it changed nothing. For an encoder to help here it would have to learn a transform that is **non-monotone within site**, and nothing in the adversarial objective encourages that.

So for this failure, the batch-removal framing is the wrong tool — not underpowered, wrong. If the encoder is to earn its place it should receive the time axis as an **explicit objective** (a time-regression head, or a triplet defined on time distance), evaluated under leave-one-study-out. That is a different architecture from the one this repo started with, and §3B/§3C are the reason for the change.

### What this does and does not establish

- **Established:** the pooled association is absent in GSE94438, and per-site offsets are not the reason. No location/scale correction can change that.
- **Not established:** that the sites genuinely point in opposite directions. South Africa's +0.196 has a bootstrap 95% CI of [−0.121, +0.482], crossing zero. Leave-one-out is stable (+0.162 to +0.258), so it is not one donor's doing — but three sites at n = 11–39 cannot settle it.
- **Ruled out as explanations:** timeline shift (site medians 244 / 335 / 335 days are comparable) and single-donor outliers.
- **Still open:** whether the GSE94438 null reflects real biological heterogeneity between settings, a signature that does not transfer, or insufficient signal at these sample sizes. Distinguishing those needs more cohorts, not a better model.

---

## 4. Design principles (from first principles)

The cross-cohort bulk RNA-seq setting is a textbook **small-n / high-p** problem: two cohorts, ~14k shared genes, and — per §3.2 — only 478 truly independent individuals behind 783 samples. Three design constraints follow:

1. **A deep AE overfits easily, and just as easily memorizes batch and treats it as "signal."** So the AE is not the only tool — it's an object to be diagnosed and constrained (built-in HVG selection + L2 regularization + dropout).
2. **"Observing batch / biological signal" must be quantified, not eyeballed on UMAP/PCA.** 2D projections cluster deceptively. Two metrics are used here:
   - **linear-probe accuracy** (logistic regression on the latent): lower batch probe = better mixing, higher biology probe = signal retained;
   - **silhouette**: lower batch silhouette is better, higher biology silhouette is better.
3. **There must be a linear baseline for comparison.** If the AE latent can't beat a one-line `removeBatchEffect` on batch mixing, then adversarial / triplet is patching the wrong architecture.

A fourth constraint comes out of §3: **the probe itself must be split by donor.** With 2.47 samples per person in GSE79362, an ungrouped probe is scoring on people it has already seen and will overstate biological decodability.

---

## 5. Results (synthetic data)

### 5.1 A plain AE does NOT remove batch on its own

A naive autoencoder happily encodes batch as the dominant compressible structure in the latent. In the figure below: on the left (colored by cohort) the two cohorts separate almost completely; on the right (colored by biology) the picture is much less clean.

![plain AE latent](ae_latent.png)

### 5.2 Quantified comparison

| Representation | batch probe (↓ better) | bio probe (↑ better) | Training stability | Notes |
|---|---|---|---|---|
| Plain AE latent | 1.00 (sil +0.22) | 1.00 | stable | batch encoded verbatim, no correction |
| PCA(raw) | 1.00 (sil +0.63) | 1.00 | deterministic | no correction, batch dominates |
| **PCA + removeBatchEffect** | **0.37** (sil +0.00) | 1.00 | deterministic | linear baseline — wins outright here |
| AE + discriminator + triplet | see diary below | — | hyperparameter-dependent | needs careful tuning, verify with an independent probe |

> Biology is trivially separable in the synthetic data, so the bio probe is 1.00 almost everywhere; on real TB data this column is where the discrimination actually shows up.

**Key takeaway: in this setting, a plain AE loses to a one-line `removeBatchEffect` on batch mixing.** So the batch discriminator and triplet loss are not decoration — they have to earn their place. They earn it where **linear methods fail**:

- batch effect is **non-linear** (linear regression can't subtract it out);
- you want a **reusable encoder** that can project future cohorts into the same latent.

~~batch is confounded with the biological label~~ — this third justification appeared in an earlier revision and **has been withdrawn**: measured on the real cohorts it is negligible (§3.1). Keeping it would have been a claim the data does not support.

---

## 6. Adversarial Training: A Debugging Diary

Adversarial batch correction is notoriously finicky to tune. Three attempts are recorded here honestly, rather than tuning the demo into a clean win — because "how easily it breaks" is itself a result you need.

### Attempt 1: single-optimizer GRL — `batch_acc` didn't drop (0.997), `tri` was 0 from the start

This isn't a bug; it exposes two problems:
1. The batch effect in the synthetic data is strong and low-rank, so a weak-λ GRL can't push it out;
2. Biology is too easily separable in the synthetic data, so the triplet satisfies its margin immediately and does nothing (on real TB data, where biological signal is weak, the triplet actually starts to matter).

So I switched to a more stable, standard recipe: **two-optimizer alternating updates** (train the discriminator D to competence first, then let the encoder fool it), **removed the BatchNorm in the encoder** (it leaks batch statistics), and increased λ_adv.

### Attempt 2: two-optimizer + λ_adv=8 — diverged again (rec blew up to 6.1, batch silhouette 0.97)

`λ_adv=8` + `k_disc=5` let the discriminator always win; the confusion gradient became so large it blew up reconstruction. The triplet oscillated between 0 and 88 because `z` wasn't normalized and the distance scale ran away.

**This is the real face of adversarial training: it walks a min-max tightrope, and one over-heavy hyperparameter makes it diverge.**

Three standard stabilizations: drop λ_adv to 1.5, set `k_disc=1`, add gradient clipping; and compute the triplet on **L2-normalized `z`** (bounding distances to [0,2], the metric-learning convention).

### Attempt 3: after stabilization — training is stable, but it reveals a deeper trap

During training `D_acc` falls from 0.88 to 0.50 (the discriminator is fooled to chance), reconstruction is stable (~0.76), and the triplet now works (bio silhouette 0.59).

**But — note this key phenomenon: the discriminator was fooled to chance during training, yet a fresh logistic-regression probe fit afterwards still gives `batch_acc` = 1.00.**

This isn't a failure; it's a famous and deep trap of adversarial batch correction: **fooling one discriminator ≠ removing batch information from the representation.** The encoder merely found an encoding that *that particular D* can't exploit, while an independent probe recovers batch just fine (a 16-dim latent has plenty of room to hide it). This is exactly why the field moved to stricter metrics like **iLISI / kBET** and to combining adversarial training with explicit distribution matching (MMD).

### Field notes (directly usable lessons)

- **Don't use BatchNorm in the encoder — use LayerNorm.** BN normalizes within a minibatch, so when a minibatch is cohort-imbalanced it re-injects batch statistics into `z` and fights the discriminator — the sneakiest pitfall. The batch-balanced sampler in `splits.py` attacks the same problem from the sampling side.
- **Ramp λ_adv from 0 with a sigmoid** (DANN schedule) so reconstruction stabilizes first; keep `k_disc` at 1–2 with gradient clipping.
- **L2-normalize `z` before the triplet**, otherwise the distance scale runs away and the loss oscillates.
- **The convergence criterion is NOT "D_acc dropped to chance" — it's an independent probe + iLISI.** Don't trust the loss curve.

---

## 7. Next steps

### Batch discriminator

The principle is a min-max between the encoder and a discriminator D. The recommended **two-optimizer** scheme is more controllable than a one-shot GRL:
- `opt_D` updates only D, classifying batch from `z.detach()`;
- `opt_G` updates encoder+decoder, pushing D's output toward uniform (batch becomes indistinguishable).

Full implementation in `train_adv()` in `ae_adv_triplet.py`.

### Triplet loss — the heart is "cross-batch positives"

An ordinary triplet only pulls same-class samples together; the step that matters for cross-cohort alignment is: **the anchor's positive is preferentially drawn from the same class in a *different* cohort**. This writes "same biology, different batch → pull together" directly into the objective. See `triplet_loss()`.

Two practical issues:
- **Progressors are rare → many minibatches contain no positives**, and the triplet silently no-ops. Use **PK-sampling** (P classes × K samples per minibatch, the batch-hard standard from Hermans et al.) or fall back to online semi-hard. The batch-balanced sampler in `splits.py` is the first half of this.
- **`z` must be L2-normalized** before computing distances.

### Recommended path (from first principles — not the shortest, but the most robust)

1. **Unblock the label axis.** Export `Progression` / `TimeToTB` and fill in `configs/studies.yaml`. Nothing supervised is meaningful until the two cohorts are on a shared contrast (§3.3).
2. **Run the two baselines first**: plain AE and `removeBatchEffect`, quantified with the probe + iLISI/kBET, under leave-one-study-out. **If the linear method already mixes batch well without losing biology, stop there** — for cross-cohort reproducibility, simpler is more defensible. A negative result here is a real result.
3. **Only if the linear method fails** should you reach for supervised alignment. And there, **try an MMD penalty before the adversarial route** — it's deterministic, no min-max tightrope, and far more stable at small n:

```python
import torch

def mmd_rbf(za, zb, sigmas=(1., 2., 4., 8.)):
    """Distributional gap between batch a and batch b in the latent; penalize it in the loss."""
    def k(x, y):
        d = torch.cdist(x, y) ** 2
        return sum(torch.exp(-d / (2 * s ** 2)) for s in sigmas)
    return k(za, za).mean() + k(zb, zb).mean() - 2 * k(za, zb).mean()

# L = recon + lam_trip * triplet + lam_mmd * mmd_rbf(z[batch == 0], z[batch == 1])
```

MMD + triplet (distribution alignment + biological-structure preservation) is usually more reproducible than adversarial + triplet, and easier to explain to a reviewer. Keep the adversarial route as the heavy weapon for when MMD can't push batch down either.

4. **Fix the evaluation protocol**: an independent logistic probe (batch↓ / bio↑) + scib's iLISI / cLISI + kBET, always under the LOSO / donor-grouped splits from `tbbatch.splits`.

### Optional: fit the count distribution more faithfully

Swap the reconstruction loss from MSE to a **negative-binomial (NB) likelihood**, modeling raw counts directly (the scVI approach), which respects the mean-variance relationship of bulk counts better.

---

## Roadmap

| Stage | Status |
|---|---|
| Study registry + schema adapters | done |
| Confounding & leakage audit on real metadata | done |
| LOSO + donor-grouped splitters, assertion-enforced | done |
| Test suite (leakage & parsing guards verified to fire) | done — 21/21 |
| `Progression` label axis | **resolved: redundant** — no shared supervised axis exists (§3.3) |
| Count matrices exported and intersected (15,264 shared genes) | done |
| **Baseline: signature score vs time-to-diagnosis (§3B)** | **done** |
| **Site correction test — negative, with an invariance proof (§3C)** | **done** |
| Count matrices wired to `load_real()` | pending |
| Baselines (uncorrected / ComBat / PCA / AE) under LOSO | pending |
| Adversarial + triplet on real data, with ablations | **deprioritised** — see §3C |
| Time-supervised encoder (regression head / time-triplet) under LOSO | **next** |
| Decomposition: batch effect vs comparator shift (§3.6) | **next** |
| Progressor-only time-to-diagnosis axis (§3.7) | **resolved on real metadata: 168 samples / 108 donors** — evaluation set, not training set |
| Unsupervised encoder on all 478 donors, evaluated on §3.7 axis | **next** |

---

## References

- Ganin & Lempitsky, *Domain-Adversarial Training of Neural Networks* (GRL / DANN)
- Hermans et al., *In Defense of the Triplet Loss for Person Re-Identification* (batch-hard mining)
- Luecken et al., *Benchmarking atlas-level data integration* (iLISI / kBET / scIB metrics)
- Lopez et al., *scVI* (NB likelihood decoder)
- Bergsma, *A bias-correction for Cramér's V and Tschuprow's T*
- Zak et al. 2016, *Lancet* (GSE79362); Suliman et al. 2018, *AJRCCM* (GSE94438)

---
---

<a name="中文说明"></a>

**[English](#bulk-transcriptomics-autoencoder-batch-vs-biological-signal-in-latent-space)** | 中文

# Bulk Transcriptomics Autoencoder:Latent Space 中的 Batch 与 Biological Signal

用 PyTorch 建立一个 bulk RNA-seq 的 autoencoder,**量化**观察 latent space 里 batch effect 和生物信号各自占多少,并在此基础上尝试用 **batch discriminator(对抗)** 与 **triplet loss** 做 cross-cohort 对齐。

这个仓库和大多数 batch correction 代码不一样的地方有两点:

1. **它在训练任何模型之前先审计实验设计。** 一批数据集到底**能不能**做对抗式校正,是设计本身的性质,不是方法的性质——而且**只看 metadata 就能判定**。`scripts/run_audit.py` 用一条命令、在真实队列 metadata 上先回答这个问题。
2. **它如实记录了对抗训练有多难调、在哪些超参下会崩**(见第六节)。因为在 small-n / high-p 的 bulk 场景里,方法能不能复现、值不值得上,比方法本身更重要。

> **当前状态。** 设计审计与划分层已在 GSE79362、GSE94438 的真实 metadata 上跑通并有测试覆盖。模型层目前仍跑合成数据;有监督的跨队列实验被证实**在设计上就不可得**——两个队列不存在共享的有监督 label 轴,验证过程见 3.3 节。3.6 节给出它们能回答的那个更锋利的问题。

---

## 一、文件说明

### 数据层——跑在真实队列 metadata 上

| 文件 | 内容 |
|---|---|
| `configs/studies.yaml` | 研究注册表。列名映射、label 轴定义、以及每个轴能支撑哪类实验。新增队列只需加一个 block,不用改代码。 |
| `src/tbbatch/metadata.py` | Schema adapter,统一到规范化表。donor ID 加 study 前缀防跨队列碰撞;当两个队列的负类含义不同时,`pool()` **直接抛异常**而非默默拼接。 |
| `src/tbbatch/audit.py` | Cramér's V(含 Bergsma 校正,衡量 batch↔label 关联)、重复测量结构、donor 泄漏的蒙特卡洛估计。 |
| `src/tbbatch/timeaxis.py` | 把 `TimeToTB` 跨队列解析为天。处理单位不一致、非 `NA` 的 `"---"` 缺失标记、以及确诊后的负数时间;每行显式返回 `kind`,不做强制转换。另含 donor 级功效计算与个体内跨度(3.7.1–2 节)。 |
| `src/tbbatch/splits.py` | Leave-one-study-out(外层)+ donor-grouped 分层 k-fold(内层),另含 batch-balanced minibatch 采样器。每个 split 返回前都跑断言;**泄漏就 raise**,而不是默默给你一个好看的数字。 |
| `scripts/run_audit.py` | 一条命令 → `results/design_audit.md`。 |
| `scripts/run_site_correction.py` | 检验站点中心化 / ComBat 能否恢复 GSE94438 的关联,并给出「为何不能」的不变性论证。→ `results/site_correction.md` |
| `scripts/run_baseline.py` | 真实表达上的 signature 得分 vs 到确诊时间:分队列相关、个体内/个体间分解、站点异质性 Cochran's Q。→ `results/baseline_timeaxis.md` |
| `data/metadata/` | 两个队列的样本级 metadata(**不含表达值**),使审计无需安装 Bioconductor 即可复现。来源见其 README。 |
| `tests/test_splits.py` | 10 个测试。其中三个**故意构造泄漏的 split**,验证守卫真的会拦。 |

### 模型层——目前为合成数据

| 文件 | 内容 |
|---|---|
| `ae_bulk.py` | 数据(合成 demo + 真实数据接口)、preprocessing、plain autoencoder、**latent 诊断**(线性探针 + silhouette)、**线性 baseline**(PCA / removeBatchEffect)、latent 可视化 |
| `ae_adv_triplet.py` | 在 `ae_bulk.py` 基础上加 **batch discriminator(双优化器对抗)** + **跨 batch triplet loss**,复用同一套诊断以便直接对照 |

## 二、快速开始

```bash
pip install -r requirements.txt

# 1. 先审计设计——不需要 count 矩阵
python scripts/run_audit.py --registry configs/studies.yaml --root .
cat results/design_audit.md

# 2. 验证防泄漏守卫真的会触发
python -m pytest tests/ -q

# 3. 模型层(合成数据)
python ae_bulk.py          # plain AE + 两条线性 baseline + 诊断 + latent 图
python ae_adv_triplet.py   # 对抗 + triplet 版本,与 plain AE 对照
```

接自己的数据:填 `ae_bulk.py` 里的 `load_real()` 桩,返回 `(X, batch, bio)`——`X` 为已 log-normalized 的表达矩阵(log1p CPM / VST,限制到共享基因),`batch` 为 cohort id,`bio` 为生物标签。**样本顺序与划分索引请从数据层 `tbbatch.splits` 取**,不要在模型脚本里自己划分;第三节解释了在这批数据上为什么这一点尤其要紧。

---

## 三、设计审计:训练之前,metadata 已经说了什么

本节全部是实验设计的性质,不需要任何表达值,而且它约束了模型层**被允许宣称什么**。用 `scripts/run_audit.py` 重新生成,完整输出见 [`results/design_audit.md`](results/design_audit.md)。

队列:**GSE79362**(ACS 青少年队列,Zak et al. 2016)与 **GSE94438**(GC6-74 家庭密接,Suliman et al. 2018)。

### 3.1 Batch 与 label 并未实质混杂——这是好消息,同时是一处更正

技术变量与生物标签之间的 Cramér's V:

| 关联 | Cramér's V | 判读 |
|---|---|---|
| study × label(合并,n=783) | **0.072** | negligible |
| site × label(94438 内部,n=428) | **0.069** | negligible |

| study | label=0 | label=1 | n | 阳性率 |
|---|---|---|---|---|
| GSE79362 | 245 | 110 | 355 | 0.310 |
| GSE94438 | 327 | 101 | 428 | 0.236 |

这一点很关键,因为对抗式去 batch 有一个硬前提:**如果 batch 决定了 label,数据里就没有反例**,那么任何能抹掉 batch 的方法都必然连生物信号一起抹掉。**这个失败模式在这里不成立。** 数据设计里有方法所需要的反例。

> **对本 README 早期版本的更正。** 之前的版本把「batch 与生物标签混杂——两个队列 progressor 比例不同」列为采用对抗方法的理由之一。实测:0.310 vs 0.236,Cramér's V = 0.072。**这条理由在这两个队列上不成立**,已删除。第 5.2 节其余两条理由(batch 非线性、需要可复用 encoder)不受影响。

### 3.2 真正的泄漏风险在 donor 层,而且很严重

| study | 样本数 | donor 数 | 有重复的 donor | 单人最多 | 有效样本量虚增 |
|---|---|---|---|---|---|
| GSE79362 | 355 | **144** | 105 | 6 | **2.47×** |
| GSE94438 | 428 | 334 | 79 | 4 | 1.28× |

对 2000 次随机 80/20 **样本级**划分做蒙特卡洛,统计落在两侧的 donor 数:

| study | 平均泄漏 donor 数 | 最少 | 最多 | P(干净划分) |
|---|---|---|---|---|
| GSE79362 | 47.7 | 35 | 60 | **0.000** |
| GSE94438 | 27.8 | 14 | 42 | **0.000** |

两个队列各 2000 次抽样,**没有一次干净**。在 GSE79362 上,随机划分不是「有风险」,而是**必然**把同一个人的不同时间点分到两侧,有效样本量是 144 而不是 355。

这正是[姊妹 DE 分析仓库](https://github.com/melodysum/TB-Whole-Blood-Transcriptomics-GSE79362-GSE94438)里,用 `duplicateCorrelation`(ρ ≈ 0.31)把重复测量纳入模型后,strict DEG 从 30 塌到 9 的同一个结构。`splits.py` 把同一约束搬进了深度学习流程——在那里更容易被忘掉。

### 3.3 `TBStatus` 无法支撑合并迁移实验——而且这是**标签**问题

| study | 负类 | n |
|---|---|---|
| GSE79362 | LTBI | 245 |
| GSE94438 | household contact | 327 |

两个负类**完全不重叠**。因此 `study` 百分之百决定了「control」**是什么意思**。合并训练的分类器可以「答案对、理由错」,而且——这是关键——**再强的对抗去 batch 也修不好,因为混杂在标签定义里,不在表达值里。** encoder 根本没有可抹的东西。

`metadata.pool()` 默认拒绝这个轴,而不是默默拼起来。

**`Progression` 列救不了这一点——这是验证过的,不是假设。** 两个队列都有 `Progression` 列,看上去像是共享的轴。它不是。在每个队列**内部**,它与 `TBStatus` **完全共线**(交叉表 off-diagonal 全为 0):

| | Negative | Positive |
|---|---|---|
| GSE79362 `Progression` | 245 | 110 |
| GSE79362 `TBStatus` | LTBI **245** | PTB **110** |
| GSE94438 `Progression` | 327 | 101 |
| GSE94438 `TBStatus` | Control **327** | PTB **101** |

`Progression` 是一次**改名**,不携带任何 `TBStatus` 之外的信息——而且是一次有危险的改名,因为它把两个**不同的**对照人群统一命名为 `"Negative"`。任何以字符串比对 label 的检查都会认为两个轴一致并把它们合并,而底层人群的差异丝毫未变。

`audit.independent_axes()` 会自动检出这种情况,而不依赖注册表里的人工标注——原则是:**依赖人工正确标注才生效的守卫,不算守卫。**

**结论:这两个队列不存在共享的有监督轴。** 这是实验设计本身的性质——GC6-74 招募的是家庭密接,ACS 招募的是青少年 LTBI 队列——任何改名、任何 batch correction 方法都造不出一个来。

这两个队列**能**支撑什么,见 3.6 节。

### 3.4 协议

- **外层:** leave-one-study-out。唯一一种测试批次从未被见过的划分。
- **内层:** donor-grouped 分层 5-fold,只在训练池内部做。
- **Minibatch:** 跨队列均衡采样——既稳定判别器,也堵住第六节说的 BatchNorm 泄漏路径。
- **执行方式:** 靠断言,不靠自觉。当前 metadata 下已验证的划分:

```
LOSO[hold=GSE79362]  train=428  test=355   passed
LOSO[hold=GSE94438]  train=355  test=428   passed
donorCV[fold=0..4] (GSE79362)  约 115/29 donors  passed
```

### 3.5 关于姊妹仓库那组 AUC 数字的说明

配套的 DE 分析报告了已发表 TB signature 在 GSE79362 上 AUC ≈ 0.77、在 GSE94438 上 ≈ 0.69。**这不是跨队列迁移的测量结果。** 那是同一组 signature 在**每个队列内部各自独立**评分,而且对照组不同(PTB vs LTBI,与 PTB vs 家庭密接)。差距反映的是两个对比本身的难度,不是迁移损失。

该分析里真正的跨队列量是一致性指标——基因层 logFC Spearman ρ = 0.641,Hallmark NES ρ = 0.815,基于 14,128 个共享基因。**这两个队列的迁移基线目前并不存在**,上面那套 LOSO 协议正是为了把它建起来。

### 3.6 这两个队列真正能回答什么

既然不存在共享的有监督轴,那个朴素的课题(「对抗校正能不能提高跨队列迁移 AUC?」)在这里无法回答。但**能**回答的那个问题更锋利,也更有价值:

> **跨队列迁移损失至少有两个独立来源——技术性 batch effect,以及对照组漂移(「control」指什么)。batch correction 方法只处理前者。这两者能不能被分开?修好前者会不会让后者动一点?**

审计结果已经把台子搭好了,而且给出一个**可证伪的预测**:

- batch↔label 关联 **negligible**(3.1 节,V = 0.072),所以混杂**不是**障碍;
- 对照组漂移是**彻底的**(3.3 节),两个对照人群完全不重叠;
- 因此,一个能去掉 batch 信息的方法应当改善**batch mixing 指标**(探针准确率、iLISI、kBET),**同时让迁移性能基本不动。**

如果这个预测成立,它就是对本仓库一开始想揭示的那个失败模式的一次干净演示:*PCA 更好看了,而什么都没有真正改善*——但这次是带机制的,不再只是一句警告。如果预测被推翻,那更有意思。

这不需要额外数据、不需要额外代码。同一个 encoder、同一套划分、同一批指标——变的只是**宣称**:从数据支撑不了的那个,换成数据支撑得了的那个。


### 3.7 `TimeToTB`:队列真正共享的那个轴——但要先拆三颗雷

`TimeToTB` 才是 `Progression` 没能提供的那个共享轴。但它原样不可用。两个队列的原始值:

```
GSE79362  chr  "642 Day(s)"  "0 Day(s)"  "---"  "-91 Day(s)"  "-253 Day(s)"  NA
GSE94438  chr  "22 month(s)"  "7 month(s)"  "3 month(s)"  NA
```

**雷 1——单位不同。** 一个队列用天,另一个用月。不换算直接拼接会把其中一条时间轴压缩约 30 倍。`timeaxis.parse_series()` 统一归一到天(30.4375 天/月),归一后区间才正确重叠(GSE94438 的 3–24 月 = 91–730 天,对 GSE79362 的 0–642 天)。

**雷 2——一个不是 `NA` 的缺失标记。** GSE79362 用字面字符串 `"---"`。`is.na()` 对它返回 **FALSE**,所以由 `is.na()` 得到的 91/355 缺失数是**低估**,低估的量等于 `"---"` 的行数。解析结果对每一行显式返回 `kind`(`parsed` / `missing` / `sentinel` / `unparseable`),而不是强制转换。

**雷 3——负数时间。** `"-91 Day(s)"`、`"-253 Day(s)"`。按「采样到确诊的时间」这个读法,它们是**确诊之后**采的血——现患疾病,可能已在治疗。那是另一种生物学状态,不是早期进展信号;把它和确诊前样本混在一起,等于把最强的疾病信号放进了本应代表最早期的那一组。

**雷 2 实际是怎么炸的——包括在这份文档里炸过一次。** `"---"` 不是零散分布的,它与类别完全对齐:

| `TimeToTB == "---"` | Negative | Positive |
|---|---|---|
| FALSE(有真实值) | **0** | 98 |
| TRUE(`"---"`) | **166** | 0 |
| `NA` | 79 | 12 |

所以 GSE79362 的真实缺失是 **257/355**,而不是 `is.na()` 报告的 91/355——**低估 2.8 倍**。

本节的早期版本读了 `is.na()` 交叉表,得出「166 个阴性带随访时间,所以 `(TimeToTB, Progression)` 是右删失生存数据」。**这是错的。** 那 166 行全是 sentinel,根本没有值。两个队列里没有任何一个非进展者有时间值。不存在删失时间,因而不存在 risk set,因而**合并生存模型不可得**——但理由和最初给出的正好相反。

`timeaxis.audit_series()` 会把这些行正确归类,报告 `negatives_with_time = 0`。错误的来源是**拿 R 的原始输出推理,而没有跑那个正是为此而写的解析器**。绕得过去的守卫不算守卫;`tests/test_splits.py::test_sentinels_are_not_counted_as_censoring_times` 复现了真实的 166/79/98/12 结构,使这个结论不会悄悄倒退。

**更正后的结构更简单,也更好。** 两个队列都只对 **progressor** 填充 `TimeToTB`——GSE79362 有 98 个,GSE94438 有 101 个。对称、无删失、无需调和的不一致。

#### 3.7.1 Donor 数改变了这个轴的用途

上面是样本数。独立单位是人,而且这个数要经过一层清洗才定得下来。在真实 metadata 上跑出来:

| 步骤 | GSE79362(样本/donor) | GSE94438(样本/donor) |
|---|---|---|
| 全部样本 | 355 / 144 | 434 / 334 |
| progressor | 110 / 40 | 101 / **75** |
| + 时间可解析 | 98 / 33 | 101 / 75 |
| + 剔除 days < 0(确诊后) | 85 / 33 | 101 / 75 |
| + 剔除 days = 0(确诊当天) | **67 / 33** | **101 / 75** |

两点值得注意。GSE79362 在解析这一步丢掉 7 个 donor——这些人有 progressor 样本但没有可用时间。以及 GSE94438 的 progressor donor 是 **75**,不是早先数出的 76:R 里 `x[cond]` 在 `cond` 为 `NA` 处返回 `NA` 行,而该队列有 6 个 `NA` 的 progression 标签,`unique()` 把 `NA` 当成了一个 donor。

最终:**168 样本,108 donor。**

检出与「到确诊时间」单调关联的功效(α = 0.05,双侧):

| 真实 ρ | n=33(GSE79362) | n=75(GSE94438) | n=108(合并) |
|---|---|---|---|
| 0.3 | 0.40 | 0.75 | 0.89 |
| 0.4 | 0.64 | 0.95 | 0.99 |
| 0.5 | 0.85 | 1.00 | 1.00 |
| 0.6 | 0.97 | 1.00 | 1.00 |

**所以这个轴是评估集,不是训练集。** 116 个人对约 14,000 个基因,撑不起拟合一个 encoder;但用来**检验**一个 encoder、在中等效应量下是够用的。这就把架构定死了:

1. encoder 在**全部 783 样本 / 478 donor 上无监督训练**——不需要标签,所以整批数据都能出力;
2. 用得到的 latent 在 progressor 时间轴上做跨队列评估。

两条臂用不同数据干不同的活,谁都没被要求做它样本量撑不起的事。

#### 3.7.2 重复测量:在一个队列里是资产,在另一个里是幻觉

在这份审计的其他地方,重复采样都是泄漏风险(3.2 节)。在这个轴上它可以反过来:同一个人在距确诊不同远近处被观测,提供的是**个体内对比**,把个体基线表达作为干扰项消掉,功效远高于跨个体比较。

两个队列看上去都提供了这一点。实际只有一个:

| 队列 | 有 >1 个 progressor 样本的 donor | 有 >1 个**不同时间点**的 donor | 个体内跨度中位数 |
|---|---|---|---|
| GSE79362 | 19 | **19** | **360 天** |
| GSE94438 | 20 | **0** | — |

GSE94438 那 20 个多样本 donor,**每一个的个体内时间跨度都恰好是 0 天**。它们是单一时间点的重复样本,不是纵向采样。

`donor_time_structure()` 的早期版本只检查 `count > 1`,对 GSE94438 报告 `supports_within_donor_design = True`。这是错的,而且是那种**能躲过审阅**的错:样本计数看起来就像纵向数据,而且不报任何错。现在该函数要求跨度非零,`tests/test_splits.py::test_same_timepoint_replicates_are_not_a_longitudinal_design` 把它锁住。

实际后果:两个队列不是同一个实验的可互换两半。**GSE79362(33 donor,19 个纵向,跨度中位数 360 天)是个体内混合模型该待的地方;GSE94438(75 donor,横断面)是跨个体的重复验证。** 划分要按此设计,且 donor 分组必须保留——个体内对比只有在 fold 尊重它时才成立。

#### 3.7.3 第三种漂移:两个队列采样于时间轴的不同区段

它区别于 batch effect(3.1 节)和对照组漂移(3.3 节),而且同样不是 batch correction 够得着的:

| 队列 | n | 最小 | Q1 | 中位数 | Q3 | 最大 |
|---|---|---|---|---|---|---|
| GSE79362 | 67 | 4 | 191 | **274** | 436 | 894 |
| GSE94438 | 101 | 91 | 213 | **426** | 639 | 730 |

GSE94438 的样本系统性地**离确诊更远**——中位数 426 天对 274 天,而且它的窗口最近只到 91 天,GSE79362 却能低到 4 天。

这一点要紧,是因为 TB 进展信号已知会随确诊临近而增强;这种「距离依赖」正是 signature 文献的核心发现。所以在一个队列上训练、到另一个队列上测试,面对的是**目标变量本身的协变量漂移**。迁移可能仅因这个原因就失败,而 batch effect 完全无辜。

与另外两种漂移不同,这一种可以靠设计消解。把两个队列都限制到重叠窗口 [91, 730] 天:

| 队列 | 样本 / donor | 中位数 |
|---|---|---|
| GSE79362 | 54 / 32 | 282 天 |
| GSE94438 | 96 / 73 | 411 天 |
| **合计** | **150 / 105** | — |

中位数仍有差距,所以只做窗口限制并不能让两个队列等价,但它切掉了不重叠的尾部,使比较可解释。**全区间和窗口限制两套结果都要报**——两者之差本身就是在测量:表观迁移损失里有多少是时间轴漂移、而非技术性 batch。

**真正可行的是:progressor-only 窗口。** 限定在 progressor 内部,「采样到确诊的天数」在两个队列里含义相同——而且由于对照组被完全排除,**3.3 节那个负类不重叠的混杂随之消失。** 清洗前 n ≈ 199(98 + 101),去掉 sentinel 与负数时间后更少。

这是本数据集支持的第二个真实验,与 3.6 节并列。样本量小,但它是这里唯一一个**不被设计本身混杂**的跨队列有监督目标。

---

## 3A. 更正记录

以下每一条都曾被提出、经数据检验、被证伪。保留而不是悄悄删掉,因为这个序列**本身就是分析过程**:每一次撤回都收窄了这批数据能支撑的范围,最终的设计是活下来的那部分。

| # | 曾经的论断 | 证伪它的检验 | 实际情况 | 代码改动 |
|---|---|---|---|---|
| 1 | 已发表 signature 的 AUC 0.77 → 0.69 是**跨队列迁移损失**基线 | 读 `signature_AUC_summary.csv` | 同一组 signature 在每个队列**内部**各自评分,对照组不同。差距反映对比难度。根本不存在迁移实验。 | 新增 3.5 节 |
| 2 | 该上对抗方法,因为这批队列里 **batch 与 label 混杂** | 在真实 metadata 上算 Cramér's V | 0.310 vs 0.236,**V = 0.072**,negligible。那个失败模式不成立;设计里有方法所需的反例。 | `audit.confounding_report()`;5.2 节删去该理由 |
| 3 | 换到 **`Progression` 轴**即可解决负类不重叠问题 | `table(Progression, TBStatus)` | 在两个队列内部完全共线——纯改名。更糟的是它把两个**不同的**对照人群统一叫作 `"Negative"`,任何字符串比对的守卫都会放行合并。 | `audit.independent_axes()`——实测共线性,不再信任注册表声明 |
| 4 | GSE79362 有 166 个阴性带随访时间,所以这是**右删失生存数据** | `sum(TimeToTB == "---")` | 那 166 个全是 sentinel,根本没有值。两个队列都没有删失时间。真实缺失 257/355,而非 91/355。 | `timeaxis` 的 sentinel 处理;针对真实 166/79/98/12 结构的回归测试 |
| 5 | GSE94438 有 **76** 个 progressor donor | 在解析后的 metadata 上 `nunique()` | 是 75。R 里 `x[cond]` 在 `cond` 为 `NA` 处返回 `NA` 行;该队列有 6 个 `NA` 标签,`unique()` 把 `NA` 当成一个 donor。 | 3.7.1 节改为从 CSV 重算 |
| 6 | GSE94438 每人 1.33 个样本,仍**支持个体内对比** | 个体内时间跨度 | 20 个多样本 donor 的跨度全部恰好为 0 天——单时间点重复,不是纵向。 | `donor_time_structure()` 改为要求跨度非零;新增回归测试 |

其中四条(2、4、5、6)是同一个形状:**信任了汇总统计量,而没有去看底层的行。** 第 4 和第 6 条更尖锐,因为两次能抓住错误的工具都已经写好了——而第 6 条里,工具本身就编码了和文字相同的那个浅层判据(`count > 1`)。

沉淀下来的方法论规则,也是本仓库现在遵循的:*依赖人工正确标注、或者能被「看一眼原始输出」绕过去的守卫,不算守卫。* 因此 `independent_axes()` 实测共线性而不读 config 字段,`audit_series()` 逐行返回 `kind` 而不是一个缺失计数。

---

## 3B. 真实表达数据上的第一批结果

`scripts/run_baseline.py` → [`results/baseline_timeaxis.md`](results/baseline_timeaxis.md)。未校正 log-CPM,仅 progressor,days > 0,每个 donor 一行。signature 定义直接取自[姊妹 DE 分析仓库](https://github.com/melodysum/TB-Whole-Blood-Transcriptomics-GSE79362-GSE94438),避免两个仓库悄悄分叉。取交集后的共享基因空间:**15,264**。

rho 为负表示得分随确诊临近而升高——正是 signature 文献预测的方向。

### 关系在一个队列成立,在另一个队列消失

| 队列 | signature | donor | rho | 95% CI | p |
|---|---|---|---|---|---|
| GSE79362 | Zak16 | 33 | **-0.449** | [-0.69, -0.13] | 0.009 |
| GSE79362 | Eleven_gene | 33 | -0.426 | [-0.67, -0.10] | 0.014 |
| GSE94438 | Zak16 | 75 | **-0.022** | [-0.25, +0.21] | 0.85 |
| GSE94438 | Eleven_gene | 75 | -0.025 | [-0.25, +0.21] | 0.85 |

把两个队列都限制到 [91, 730] 天重叠窗口后,两个数字几乎不动(-0.423 与 -0.022),因此**时间轴漂移(3.7.3 节)解释不了这个差异。**

GSE94438 的零结果也不只是功效不足:在 75 个 donor 下,置信区间排除了任何强于 |rho| = 0.25 的关联。

### 个体内对比更强,与 3.7.2 节的预测一致

| 队列 | 纵向 donor | 个体内 rho | p | 个体间 rho | p |
|---|---|---|---|---|---|
| GSE79362 | 19 | **-0.493** | **0.00018** | -0.449 | 0.009 |
| GSE94438 | **0** | 不可得 | 不可得 | -0.022 | 0.85 |

让每个人做自己的对照,用**更少**的人得到了**更强**的关联,p 值小两个数量级。GSE94438 贡献 0 个纵向 donor——它的重复是单时间点复制(3.7.2 节),所以这项分析在那里从构造上就不可得。

### GSE94438 的零结果是站点相互抵消,不是没有信号

| 站点 | donor | rho | p |
|---|---|---|---|
| Ethiopia | 11 | **-0.563** | 0.071 |
| The Gambia | 25 | **-0.279** | 0.18 |
| South Africa | 39 | **+0.196** | 0.23 |
| 合并 | 75 | -0.022 | 0.85 |

Cochran's Q = 6.21,df = 2,**p = 0.045**,**I² = 68%**。

单个站点都没达到显著,但它们的差异超出抽样噪声可解释的范围,而且**最大的站点指向相反方向**。合并后的近零值是一次抵消。

**这正是这个项目此前缺的那个发现。** 它把「batch correction 有没有用」这个含糊目标,换成了一个定量靶标:

> GSE94438 的 signature–时间关联被站点间异质性摧毁了。站点校正能否把它恢复?恢复后的值能否接近 GSE79362 的 -0.45?

成功和失败都事先可读。而且它使 3.6 节更锋利:这里的站点不是加性噪声,它**反转了一个生物学关系的符号**——这比改善某个 mixing 指标是严苛得多的检验。

### 需要声明的限制

- 站点分层样本小(11–39 个 donor),且该分解不是预注册的。Q = 6.21、p = 0.045 是提示性的,不是定论;它证成的是下一个实验,而不是一个结论。
- `FCGR1B` 不在 GSE94438 矩阵中,因此该队列的 Zak16 与 Eleven_gene 少用一个基因。与姊妹分析同一限制。
- 共享基因数(15,264)高于姊妹仓库的 14,128,因为过滤阈值不同(这里是 `>= 20 个样本 CPM > 1`,那边用的是组大小规则)。两者都不算错,但不可互换。
- RISK4 是比值而非均值,行为不同(GSE79362 中 +0.281,方向相反,不显著)。此处不再深入。

---

## 3C. 站点校正恢复不了信号——而且可以证明它不可能恢复

`scripts/run_site_correction.py` → [`results/site_correction.md`](results/site_correction.md)。在写任何 encoder 之前先跑,理由是:如果一行线性校正就有效,神经方法就必须打赢它;如果任何校正都无效,那么 batch-removal 这个框架本身就是错的。

| 方法 | 合并 rho | p | 移动量 |
|---|---|---|---|
| 未校正 | −0.022 | 0.85 | — |
| 站点内基因中心化 | −0.020 | 0.86 | +0.002 |
| ComBat(站点为 batch) | −0.009 | 0.94 | +0.013 |
| **GSE79362 靶标** | **−0.449** | | **还差 0.427** |

纹丝不动。而且各站点的相关系数完全没变:

| 站点 | donor | 未校正 | 中心化 | ComBat |
|---|---|---|---|---|
| Ethiopia | 11 | −0.563 | −0.563 | −0.563 |
| South Africa | 39 | +0.196 | +0.196 | +0.192 |
| The Gambia | 25 | −0.279 | −0.279 | −0.274 |

### 原因是一条定理,不是一个实验结果

Spearman 相关在任意单调变换下不变。对某一个站点的得分施加任意仿射变换:

| 变换 | rho |
|---|---|
| `1.0*score + 0.0` | +0.196420 |
| `3.7*score + 2.1` | +0.196420 |
| `0.2*score + 100.0` | +0.196420 |

小数点后六位完全一致。**中心化是减去一个站点内常数;ComBat 是站点内的位置与尺度调整。二者在站点内都是单调的,所以都无法改变站点内的秩相关**——而一旦站点间偏移被消除,合并值基本上就由各站点内的值决定了。

GSE94438 的失败是**斜率**异质,不是**偏移**异质。而 batch correction 处理的是偏移。

### 这改变了建模方案

对抗式 encoder 优化的是「站点不可分辨」。达成这个目标最省力的路径就是消除站点内的位置和尺度——正是 ComBat 做的事,而它什么也没改变。要让 encoder 在这里起作用,它必须学到一个**在站点内非单调**的变换,而对抗目标里没有任何东西鼓励这一点。

所以对于这个失败,batch-removal 框架是**错的工具**——不是功效不足,是不对路。如果 encoder 要证明自己有存在价值,应该把时间轴作为**显式目标**交给它(时间回归头,或定义在时间距离上的 triplet),并在 leave-one-study-out 下评估。这与本仓库最初的架构不同,而 3B / 3C 就是改变的理由。

### 这确立了什么,没确立什么

- **已确立:** GSE94438 的合并关联确实不存在,而且站点间偏移不是原因。任何位置/尺度校正都改变不了这一点。
- **未确立:** 各站点真的方向相反。South Africa 的 +0.196 自助法 95% CI 为 [−0.121, +0.482],跨过零。留一法稳定(+0.162 至 +0.258),所以不是某一个 donor 造成的——但 n = 11–39 的三个站点定不了这件事。
- **已排除的解释:** 时间轴漂移(各站点中位数 244 / 335 / 335 天,可比)与单个 donor 的离群影响。
- **仍然开放:** GSE94438 的零结果究竟反映不同地区真实的生物学异质性、signature 不迁移、还是在这个样本量下信号不足。区分这三者需要更多队列,不是更好的模型。

---

## 四、设计理念(第一性原理)

bulk RNA-seq 跨队列场景是典型的 **small-n / high-p**:两个 cohort、~14k 共享基因,而且按 3.2 节,783 个样本背后只有 478 个真正独立的个体。这决定了三条设计约束:

1. **深度 AE 极易过拟合,也极易把 batch 直接背下来当「信号」。** 所以 AE 不是唯一工具,而是一个被诊断、被约束的探索对象——内建 HVG 筛选 + L2 正则 + Dropout。
2. **「观察 batch / biological signal」必须量化,不能只看 UMAP/PCA。** 2D 投影的簇很会骗人。这里用两个指标:
   - **线性探针准确率**(在 latent 上跑 logistic 回归):batch 探针越低越好(混得开),bio 探针越高越好(信号还在);
   - **silhouette**:batch silhouette 越低越好,bio silhouette 越高越好。
3. **必须有线性 baseline 作对照。** 如果 AE 的 latent 在 batch mixing 上打不过一行 `removeBatchEffect`,那对抗 / triplet 就是在给一个错的架构打补丁。

第三节还带出第四条约束:**探针本身必须按 donor 划分。** GSE79362 平均每人 2.47 个样本,不分组的探针是在已经见过的人身上评分,会让「生物可解码性」虚高。

---

## 五、结果(合成数据)

### 5.1 Plain AE 根本不会「自动」去 batch

一个朴素 autoencoder 会很乐意把 batch 当成主要可压缩结构编码进 latent。下图:左边按 cohort 着色两个队列几乎完全分开,右边按生物标签反而没那么干净。

![plain AE latent](ae_latent.png)

### 5.2 量化对照

| 表示 | batch 探针(↓好) | bio 探针(↑好) | 训练稳定性 | 备注 |
|---|---|---|---|---|
| Plain AE latent | 1.00(sil +0.22) | 1.00 | 稳 | batch 被原样编码,未做任何校正 |
| PCA(raw) | 1.00(sil +0.63) | 1.00 | 确定性 | 无校正,batch 主导 |
| **PCA + removeBatchEffect** | **0.37**(sil +0.00) | 1.00 | 确定性 | 线性 baseline,在这个设定里直接赢 |
| AE + discriminator + triplet | 见下方实录 | — | 依超参 | 需仔细调,且要用独立探针验证 |

> 合成数据里生物信号本身太易分,所以多数方案 bio 探针都是 1.00;真实 TB 数据里这一列才有区分度。

**核心结论:在这个设定下,朴素 AE 在 batch mixing 上打不过一行 `removeBatchEffect`。** 因此 batch discriminator 和 triplet loss 不是锦上添花,它们要证明自己存在的价值——价值在于以下**线性方法失效**的场景:

- batch effect 是**非线性**的(线性回归扣不掉);
- 你想要一个**可复用的 encoder**,把未来的新 cohort 直接投到同一 latent。

~~batch 与生物标签混杂~~ ——这第三条理由出现在早期版本中,现已**撤回**:在真实队列上实测为 negligible(3.1 节)。留着它就是在做数据不支持的宣称。

---

## 六、对抗训练踩坑实录

对抗式 batch correction 在实践中出了名的难调。这里如实记录三次尝试,而不是靠调参数把 demo 做漂亮——因为「它有多容易崩」本身就是你需要知道的结论。

### 尝试 1:单优化器 GRL —— `batch_acc` 没掉(0.997),`tri` 一开始就为 0

这不是 bug,它恰好暴露了两个问题:
1. 合成数据里的 batch effect 又强又低秩,弱 λ 的 GRL 推不动;
2. 生物信号在合成数据里太容易分,triplet 一上来就满足了 margin 所以失效(真实 TB 数据里生物信号弱,triplet 才会真正起作用)。

于是改用更稳的标准配方重做:**双优化器交替更新**(先把判别器 D 训到位,再让 encoder 去骗它)、**去掉 encoder 里会泄漏 batch 统计量的 BatchNorm**、加大 λ_adv。

### 尝试 2:双优化器 + λ_adv=8 —— 又崩了(rec 爆到 6.1,batch silhouette 0.97)

`λ_adv=8` + `k_disc=5` 让判别器永远赢,confusion 梯度过大,直接把重构炸掉;triplet 在 0↔88 之间震荡,是因为 `z` 没归一化、距离尺度失控。

**这就是对抗训练的真实面貌:它在 min-max 的钢丝上,超参一重就发散。**

三个标准稳定化修正:λ_adv 降到 1.5、`k_disc=1`、加梯度裁剪;triplet 改在 **L2 归一化的 z** 上算(把距离限制在 [0,2],度量学习的惯例)。

### 尝试 3:稳定化后 —— 训练稳了,但揭示了一个更深的陷阱

训练过程中 D_acc 从 0.88 掉到 0.50(判别器被骗到随机水平),重构稳定(~0.76),triplet 也生效了(bio silhouette 0.59)。

**但——注意这个关键现象:训练时把那个判别器骗到 chance 了,可事后拿一个全新的 logistic 回归探针去测,`batch_acc` 依然 = 1.00。**

这不是失败,而是对抗式 batch correction 一个著名且深刻的陷阱:**骗过某一个判别器 ≠ 把 batch 信息从表示里去掉。** encoder 只是找到了让「那个特定 D」抓不到的编码方式,而一个独立探针照样能把 batch 挖出来(16 维 latent 空间大得很,藏得下)。这正是领域后来转向用 **iLISI / kBET** 这类更严格的指标、并把对抗与显式分布匹配(MMD)结合起来的原因。

### 踩坑清单(直接可用的经验)

- **encoder 别用 BatchNorm,用 LayerNorm。** BN 在 minibatch 内归一化,当一个 batch 里两个 cohort 比例失衡时会把 batch 统计量重新注入 `z`,和判别器对着干——最隐蔽的坑。`splits.py` 里的 batch-balanced 采样器是从采样端解决同一个问题。
- **λ_adv 用 sigmoid 从 0 缓升**(DANN schedule),让重构先稳住;`k_disc` 保持 1~2,配梯度裁剪。
- **triplet 前先 L2 归一化 `z`**,否则距离尺度失控、loss 震荡。
- **收敛判据不是「D_acc 掉到 chance」,而是独立探针 + iLISI。** 别信 loss 曲线。

---

## 七、下一步

### Batch discriminator

原理是 encoder 与判别器 D 的 min-max。推荐**双优化器**方案(比一次性 GRL 更可控):
- `opt_D` 只更新 D,用 `z.detach()` 分类 batch;
- `opt_G` 更新 encoder+decoder,让 D 的输出趋向 uniform(batch 不可分)。

完整实现见 `ae_adv_triplet.py` 的 `train_adv()`。

### Triplet loss —— 灵魂是「跨 batch 取 positive」

普通 triplet 只让同类靠近;对 cross-cohort 有用的那一步是:**anchor 的 positive 优先取自另一个 cohort 的同类样本**。这等价于把「同生物、不同 batch → 拉近」写进目标。见 `triplet_loss()`。

两个现实问题:
- **progressor 稀有 → 很多 minibatch 里没有正样本**,triplet 静默失效。用 **PK-sampling**(每个 minibatch 保证 P 个类 × K 个样本,Hermans et al. batch-hard 标配)或退化到 online semi-hard。`splits.py` 的 batch-balanced 采样器是这件事的前半截。
- **`z` 必须 L2 归一化**再算距离。

### 推荐路径(第一性原理,不是最短但最稳)

1. **先解锁 label 轴。** 导出 `Progression` / `TimeToTB`,填进 `configs/studies.yaml`。在两个队列落到共享对比之前,任何有监督结果都没有意义(3.3 节)。
2. **先跑两条 baseline**:plain AE 和 `removeBatchEffect`,在 leave-one-study-out 下用探针 + iLISI/kBET 量化。**如果线性方法已经 batch 混得好、生物又没丢,就到此为止**——cross-cohort reproducibility 越简单越可辩护。**这里出一个否定结果,同样是真结果。**
3. **只有线性方法失效时**,才上有监督的对齐。这时**优先试 MMD 惩罚而不是对抗**——它是确定性的、没有 min-max 钢丝,在 small-n 上稳得多:

```python
import torch

def mmd_rbf(za, zb, sigmas=(1., 2., 4., 8.)):
    """batch a 与 batch b 在 latent 里的分布差异;加进 loss 惩罚它。"""
    def k(x, y):
        d = torch.cdist(x, y) ** 2
        return sum(torch.exp(-d / (2 * s ** 2)) for s in sigmas)
    return k(za, za).mean() + k(zb, zb).mean() - 2 * k(za, zb).mean()

# L = recon + lam_trip * triplet + lam_mmd * mmd_rbf(z[batch == 0], z[batch == 1])
```

MMD + triplet(分布对齐 + 生物结构保持)通常比对抗 + triplet 更容易复现,也更好向审稿人解释。对抗留作「MMD 也压不下去」时的重武器。

4. **评估协议固定**:独立 logistic 探针(batch↓ / bio↑)+ scib 的 iLISI / cLISI + kBET,并且**始终跑在 `tbbatch.splits` 给出的 LOSO / donor-grouped 划分下**。

### 可选:更贴合 count 分布

把重构损失从 MSE 换成**负二项(NB)似然**、直接在 raw counts 上建模(scVI 的做法),对 bulk count 的均值-方差关系更忠实。

---

## 路线图

| 阶段 | 状态 |
|---|---|
| 研究注册表 + schema adapter | 完成 |
| 真实 metadata 上的混杂与泄漏审计 | 完成 |
| LOSO + donor-grouped 划分器,断言强制 | 完成 |
| 测试套件(已验证泄漏与解析守卫会触发) | 完成 — 21/21 |
| `Progression` label 轴 | **已查清:冗余** — 不存在共享有监督轴(3.3 节) |
| Count 矩阵已导出并取交集(15,264 共享基因) | 完成 |
| **Baseline:signature 得分 vs 到确诊时间(3B 节)** | **完成** |
| **站点校正检验——否定结果,附不变性证明(3C 节)** | **完成** |
| Count 矩阵接入 `load_real()` | 待办 |
| LOSO 下的 baseline(未校正 / ComBat / PCA / AE) | 待办 |
| 真实数据上的对抗 + triplet,含 ablation | **降级** — 见 3C 节 |
| 时间监督 encoder(回归头 / 时间 triplet),LOSO 下评估 | **下一步** |
| 分解:batch effect vs 对照组漂移(3.6 节) | **下一步** |
| Progressor-only 到确诊时间轴(3.7 节) | **已在真实 metadata 上定案:168 样本 / 108 donor**——评估集,非训练集 |
| 在全部 478 donor 上无监督训练 encoder,在 3.7 轴上评估 | **下一步** |

---

## 参考

- Ganin & Lempitsky, *Domain-Adversarial Training of Neural Networks*(GRL / DANN)
- Hermans et al., *In Defense of the Triplet Loss for Person Re-Identification*(batch-hard mining)
- Luecken et al., *Benchmarking atlas-level data integration*(iLISI / kBET / scIB 指标)
- Lopez et al., *scVI*(NB likelihood decoder)
- Bergsma, *A bias-correction for Cramér's V and Tschuprow's T*
- Zak et al. 2016, *Lancet*(GSE79362);Suliman et al. 2018, *AJRCCM*(GSE94438)
