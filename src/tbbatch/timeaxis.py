"""Parsing `TimeToTB` into a usable time axis.

The column arrives as free text and is not safe to coerce naively. Across the
two cohorts it carries four distinct hazards:

1. **Different units.** GSE79362 records ``"642 Day(s)"``; GSE94438 records
   ``"22 month(s)"``. Concatenating without conversion silently compresses one
   cohort's timescale by ~30x.
2. **A sentinel that is not NA.** GSE79362 uses the literal string ``"---"``.
   R's ``is.na()`` reports FALSE for it, so any missingness count taken from
   ``is.na()`` alone is an undercount.
3. **Negative values.** GSE79362 contains ``"-91 Day(s)"``, ``"-253 Day(s)"``.
   Under the "time from sampling to TB diagnosis" reading these are samples
   drawn *after* diagnosis - a different biological state (prevalent disease,
   possibly on treatment), not an early progression signal.
4. **Sentinels that align exactly with a class.** In GSE79362 all 166 ``"---"``
   rows are non-progressors, and every real value belongs to a progressor. An
   ``is.na()``-based tally therefore reports "166 negatives carry a time" and
   invites the conclusion that these are censoring times and the data are
   right-censored survival. They are not: those rows hold no value at all.
   Both cohorts populate the column for progressors only. `audit_series`
   classifies sentinels correctly and does not make this mistake, but only if
   it is actually run on the data instead of reasoning from a raw crosstab.

Nothing here guesses. `parse_series` returns the parsed value plus an explicit
`kind` for every row, so downstream code chooses what to keep.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

DAYS_PER_MONTH = 30.4375  # mean Gregorian month

# Strings that mean "no value" while not being NA.
SENTINELS = {"---", "--", "-", "", "NA", "N/A", "n/a", "NaN", "unknown", "Unknown"}

_PAT = re.compile(
    r"^\s*(?P<sign>[+-]?)\s*(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>day|month|week|year)s?\s*\(?s?\)?\s*$",
    re.IGNORECASE,
)

_TO_DAYS = {"day": 1.0, "week": 7.0, "month": DAYS_PER_MONTH, "year": 365.25}


@dataclass(frozen=True)
class ParsedTime:
    days: float | None
    kind: str  # parsed | missing | sentinel | unparseable


def parse_one(value) -> ParsedTime:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ParsedTime(None, "missing")
    s = str(value).strip()
    if s in SENTINELS:
        return ParsedTime(None, "sentinel")
    m = _PAT.match(s)
    if not m:
        return ParsedTime(None, "unparseable")
    days = float(m["num"]) * _TO_DAYS[m["unit"].lower()]
    if m["sign"] == "-":
        days = -days
    return ParsedTime(days, "parsed")


def parse_series(s: pd.Series) -> pd.DataFrame:
    """Vectorised wrapper. Returns columns `days` and `kind`, index preserved."""
    out = [parse_one(v) for v in s]
    return pd.DataFrame(
        {"days": [p.days for p in out], "kind": [p.kind for p in out]},
        index=s.index,
    )


def audit_series(s: pd.Series, event: pd.Series | None = None) -> dict:
    """Structural report for one cohort's time column.

    `event` is the progression indicator, if available. Its interaction with
    the time column is what distinguishes a regression target from censored
    survival data.
    """
    p = parse_series(s)
    rep: dict = {
        "n": int(len(s)),
        "kind_counts": p.kind.value_counts().to_dict(),
        "is_na_would_report": int(s.isna().sum()),
        "true_missing": int((p.kind != "parsed").sum()),
        "undercount_by_sentinels": int((p.kind == "sentinel").sum()),
    }
    ok = p.days.dropna()
    if len(ok):
        rep["days_min"] = float(ok.min())
        rep["days_max"] = float(ok.max())
        rep["days_median"] = float(ok.median())
        rep["n_negative"] = int((ok < 0).sum())
        rep["n_zero"] = int((ok == 0).sum())
    if event is not None:
        has_time = p.kind == "parsed"
        rep["time_by_event"] = (
            pd.crosstab(has_time.rename("has_time"), event.rename("event"))
            .to_dict()
        )
        pos = event.astype(str).str.lower().isin({"positive", "1", "true", "yes"})
        rep["negatives_with_time"] = int((has_time & ~pos).sum())
        rep["censored_structure_likely"] = bool(rep["negatives_with_time"] > 0)
    return rep


def progressor_window(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    max_days: float | None = None,
    drop_post_diagnosis: bool = True,
) -> pd.DataFrame:
    """The one axis the two cohorts genuinely share.

    Restricting to progressors removes the control group entirely, and with it
    the disjoint-control-definition confound documented in the design audit:
    "days from sampling to TB diagnosis" means the same thing in both cohorts,
    whereas "control" does not.

    Adds `days_to_tb`. Rows without a parseable time are dropped. Samples drawn
    after diagnosis (negative times) are dropped by default.
    """
    p = parse_series(df[time_col])
    out = df.copy()
    out["days_to_tb"] = p.days
    out["time_kind"] = p.kind

    pos = out[event_col].astype(str).str.lower().isin({"positive", "1", "true", "yes"})
    out = out[pos & (out.time_kind == "parsed")].copy()

    if drop_post_diagnosis:
        out = out[out.days_to_tb >= 0].copy()
    if max_days is not None:
        out = out[out.days_to_tb <= max_days].copy()
    return out


def power_spearman(n: int, rho: float, alpha: float = 0.05) -> float:
    """Power to detect a monotone association at donor-level n.

    Included because the progressor time axis has 116 independent people
    behind 211 samples, and that number decides whether the axis can be
    trained on or only evaluated on. Fisher z approximation.
    """
    from scipy import stats

    if n < 4:
        return float("nan")
    z = np.arctanh(rho) * np.sqrt(n - 3)
    crit = stats.norm.ppf(1 - alpha / 2)
    return float(stats.norm.cdf(z - crit) + stats.norm.cdf(-z - crit))


def donor_time_structure(df: pd.DataFrame, donor_col: str = "donor_id",
                         time_col: str = "days_to_tb") -> dict:
    """Within-donor spread on the time axis.

    On this axis repeated sampling is an asset rather than a leakage hazard:
    a donor observed at several distances from diagnosis supplies a
    within-person contrast, which removes individual baseline expression as a
    nuisance. `n_donors_multi` and `median_within_donor_range_days` say how
    much of that is actually available.
    """
    g = df.groupby(donor_col)[time_col]
    per = g.size()
    rng = (g.max() - g.min()).loc[per > 1]
    # A donor with several samples at the SAME time contributes replicates, not
    # a longitudinal contrast. GSE94438 has 20 such donors, every one with a
    # spread of exactly 0 days. Counting them as longitudinal would claim a
    # within-person design the data cannot support, so spread is required.
    n_spread = int((rng > 0).sum())
    return {
        "n_samples": int(len(df)),
        "n_donors": int(per.size),
        "samples_per_donor": round(float(per.mean()), 3),
        "n_donors_multi": int((per > 1).sum()),
        "n_donors_with_time_spread": n_spread,
        "median_within_donor_range_days": (
            float(rng[rng > 0].median()) if n_spread else None
        ),
        "supports_within_donor_design": bool(n_spread >= 10),
    }
