# Design audit

Generated from cohort metadata before any model is trained. 
Everything below is a property of the study design, not of a method.

## 1. Cohort inventory

| study    |   samples |   donors |   donors w/ repeats |   max/donor | eff. N inflation   | control class     |
|:---------|----------:|---------:|--------------------:|------------:|:-------------------|:------------------|
| GSE79362 |       355 |      144 |                 105 |           6 | 2.465x             | LTBI              |
| GSE94438 |       428 |      334 |                  79 |           4 | 1.281x             | household contact |

## 2. Label-axis compatibility

**Blocked.** Axis `tbstatus` has disjoint negative classes across cohorts: {'GSE79362': 'LTBI', 'GSE94438': 'household contact'}.

`study` therefore determines the *meaning* of the negative class. A pooled classifier can reach the right answer for the wrong reason, and no amount of adversarial batch removal fixes it, because the confound is in the label definition rather than in the expression values. This axis is usable for within-study analysis and for unsupervised representation work only.

## 3. Batch/label confounding

Cramer's V between the technical variable and the label. Low values mean the design contains the counter-examples an adversarial model needs.

### study x label (pooled)

| study    |   label=0 |   label=1 |   n |   pos_rate |
|:---------|----------:|----------:|----:|-----------:|
| GSE79362 |       245 |       110 | 355 |      0.31  |
| GSE94438 |       327 |       101 | 428 |      0.236 |

Cramer's V = **0.0716** (negligible) - batch carries almost no label information

### site x label (GSE94438)

| site         |   label=0 |   label=1 |   n |   pos_rate |
|:-------------|----------:|----------:|----:|-----------:|
| Ethiopia     |        29 |        16 |  45 |      0.356 |
| South Africa |       164 |        46 | 210 |      0.219 |
| The Gambia   |       134 |        39 | 173 |      0.225 |

Cramer's V = **0.0685** (negligible) - batch carries almost no label information

## 4. Donor leakage under naive splitting

Monte-Carlo over 2000 random 80/20 sample-level splits: how many donors land on both sides.

| study    |   mean_donors_leaked |   min_donors_leaked |   max_donors_leaked |   p_clean_split |   n_sim |
|:---------|---------------------:|--------------------:|--------------------:|----------------:|--------:|
| GSE79362 |                47.74 |                  35 |                  60 |               0 |    2000 |
| GSE94438 |                27.78 |                  14 |                  42 |               0 |    2000 |

## 5. Protocol selected

- Outer: leave-one-study-out **(representation metrics only on this axis)**
- Inner: donor-grouped stratified 5-fold, on the training pool only
- Mini-batches: batch-balanced sampling across cohorts
- Every split validated by assertion; a leaking split raises rather than scores

### Split validation

- `LOSO[hold=GSE79362]` train=428 test=355 - passed
- `LOSO[hold=GSE94438]` train=355 test=428 - passed
- `donorCV[fold=0]` (GSE79362) train_donors=115 test_donors=29 - passed
- `donorCV[fold=1]` (GSE79362) train_donors=116 test_donors=28 - passed
- `donorCV[fold=2]` (GSE79362) train_donors=115 test_donors=29 - passed
- `donorCV[fold=3]` (GSE79362) train_donors=115 test_donors=29 - passed
- `donorCV[fold=4]` (GSE79362) train_donors=115 test_donors=29 - passed
