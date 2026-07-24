"""Design audit: is this dataset collection even correctable?

Adversarial batch correction has a hard precondition. If the batch variable
determines the label, no method can separate technical from biological
variance, because the data contain no counter-examples. This module measures
that precondition BEFORE any model is trained, and measures the second failure
mode (donor leakage) that random splitting hides.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# Cohen-style reading of Cramer's V for the batch-vs-label association.
# The bands are conventional, not sacred; what matters is reporting the number.
_BANDS = [
    (0.10, "negligible", "batch carries almost no label information"),
    (0.30, "weak", "correctable; monitor for over-correction"),
    (0.50, "moderate", "risky; adversarial removal may erode signal"),
    (1.01, "strong", "confounded; do not attempt adversarial correction"),
]


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Bias-corrected Cramer's V for two categorical vectors."""
    ct = pd.crosstab(a, b)
    if min(ct.shape) < 2:
        return 0.0
    chi2 = chi2_contingency(ct)[0]
    n = int(ct.values.sum())
    phi2 = chi2 / n
    r, k = ct.shape
    # Bergsma correction, keeps small tables from reading high by chance
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2c / denom)) if denom > 0 else 0.0


def interpret(v: float) -> tuple[str, str]:
    for thresh, tag, note in _BANDS:
        if v < thresh:
            return tag, note
    return _BANDS[-1][1], _BANDS[-1][2]


def confounding_table(df: pd.DataFrame, batch_col: str, label_col: str = "label") -> pd.DataFrame:
    """Contingency table with per-batch positive rate."""
    ct = pd.crosstab(df[batch_col], df[label_col])
    ct.columns = [f"label={c}" for c in ct.columns]
    ct["n"] = ct.sum(axis=1)
    pos = [c for c in ct.columns if c.endswith("=1")]
    if pos:
        ct["pos_rate"] = (ct[pos[0]] / ct["n"]).round(3)
    return ct


def confounding_report(df: pd.DataFrame, batch_col: str, label_col: str = "label") -> dict:
    v = cramers_v(df[batch_col], df[label_col])
    tag, note = interpret(v)
    return {
        "batch": batch_col,
        "cramers_v": round(v, 4),
        "severity": tag,
        "note": note,
        "table": confounding_table(df, batch_col, label_col),
    }


def axes_are_collinear(a: pd.Series, b: pd.Series) -> bool:
    """True iff two label columns are a pure relabelling of one another.

    Each level of `a` maps to exactly one level of `b` and vice versa, so the
    second column carries no information the first does not already have.

    This exists because a relabelled axis is an actively dangerous thing to
    trust: renaming two *different* control populations to the same string
    ("LTBI" and "household contact" both -> "Negative") makes a disjoint-control
    confound invisible to any check that compares class labels as text, while
    leaving the underlying populations exactly as different as before.
    """
    ct = pd.crosstab(a, b)
    if ct.empty:
        return False
    return bool((ct.gt(0).sum(axis=1) == 1).all() and (ct.gt(0).sum(axis=0) == 1).all())


def independent_axes(df: pd.DataFrame, candidates: list[str],
                     by: str = "study") -> dict:
    """Which of `candidates` are genuinely distinct label axes.

    Collinearity is assessed WITHIN each stratum of `by`, because two columns
    can be collinear inside every cohort while differing across cohorts, which
    is precisely the case that looks like a new axis and is not one.
    """
    import itertools

    redundant: dict[str, str] = {}
    present = [c for c in candidates if c in df.columns]
    for a, b in itertools.combinations(present, 2):
        per_stratum = []
        for _, g in df.groupby(by):
            sub = g[[a, b]].dropna()
            if len(sub):
                per_stratum.append(axes_are_collinear(sub[a], sub[b]))
        if per_stratum and all(per_stratum):
            redundant[b] = a
    return {
        "independent": [c for c in present if c not in redundant],
        "redundant": redundant,
    }


def donor_structure(df: pd.DataFrame) -> dict:
    """Repeated-measures structure. The gap between n_samples and n_donors is
    the factor by which naive sample-level splitting inflates effective N."""
    per = df.groupby("donor_id").size()
    return {
        "n_samples": int(len(df)),
        "n_donors": int(per.nunique() if per.empty else len(per)),
        "donors_with_repeats": int((per > 1).sum()),
        "max_samples_per_donor": int(per.max()),
        "mean_samples_per_donor": round(float(per.mean()), 3),
        "effective_n_inflation": round(len(df) / len(per), 3),
    }


def leakage_probability(df: pd.DataFrame, test_frac: float = 0.2,
                        n_sim: int = 2000, seed: int = 0) -> dict:
    """Monte-Carlo: how bad is a naive random sample-level split?

    Reports the number of donors that end up on BOTH sides. Anything above
    zero means the test set contains people the model has already seen.
    """
    rng = np.random.default_rng(seed)
    donors = df["donor_id"].to_numpy()
    n = len(donors)
    cut = int(round((1 - test_frac) * n))
    shared = np.empty(n_sim, dtype=int)
    for i in range(n_sim):
        idx = rng.permutation(n)
        shared[i] = len(set(donors[idx[:cut]]) & set(donors[idx[cut:]]))
    return {
        "mean_donors_leaked": round(float(shared.mean()), 2),
        "min_donors_leaked": int(shared.min()),
        "max_donors_leaked": int(shared.max()),
        "p_clean_split": round(float((shared == 0).mean()), 6),
        "n_sim": n_sim,
    }
