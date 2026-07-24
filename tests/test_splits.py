"""The guards are only worth having if they fire. These tests construct
leaking splits on purpose and assert that the code refuses them."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tbbatch import audit, metadata, splits  # noqa: E402


def toy(n_donors=40, per_donor=3, studies=("A", "B")):
    rows = []
    for s in studies:
        for d in range(n_donors):
            for t in range(per_donor):
                rows.append({
                    "sample_id": f"{s}_{d}_{t}",
                    "donor_id": f"{s}::D{d}",
                    "study": s,
                    "site": s,
                    "age": 20 + d % 10,
                    "sex": "M" if d % 2 else "F",
                    "lib_size": 1e7,
                    "label": int(d % 3 == 0),
                })
    return pd.DataFrame(rows)


def test_loso_holds_out_whole_study():
    df = toy()
    got = list(splits.leave_one_study_out(df))
    assert len(got) == 2
    for sp in got:
        assert not set(df.study.iloc[sp.train_idx]) & set(df.study.iloc[sp.test_idx])


def test_donor_cv_never_shares_a_donor():
    df = toy()
    for sp in splits.donor_grouped_kfold(df, n_splits=5):
        tr = set(df.donor_id.iloc[sp.train_idx])
        te = set(df.donor_id.iloc[sp.test_idx])
        assert not tr & te


def test_hand_built_leaking_split_is_rejected():
    df = toy(n_donors=10, per_donor=3, studies=("A",))
    tr = np.arange(0, 20)
    te = np.arange(18, 30)          # rows 18,19 belong to a donor also in train
    with pytest.raises(splits.LeakageError):
        splits._validate(df, tr, te, "deliberate", check_study=False)


def test_single_class_side_is_rejected():
    df = toy(n_donors=10, per_donor=2, studies=("A",))
    pos = df.index[df.label == 1].to_numpy()
    neg = df.index[df.label == 0].to_numpy()
    with pytest.raises(splits.LeakageError):
        splits._validate(df, neg, pos, "single-class", check_study=False)


def test_pool_refuses_disjoint_control_definitions():
    reg = metadata.load_registry(Path(__file__).parents[1] / "configs/studies.yaml")
    frames = {"GSE79362": pd.DataFrame(), "GSE94438": pd.DataFrame()}
    with pytest.raises(metadata.LabelAxisError):
        metadata.pool(frames, reg, "tbstatus")


def test_progression_axis_carries_the_same_disjoint_controls():
    """`Progression` is populated and technically usable, but pooling on it is
    still refused: renaming LTBI and household-contact to 'Negative' does not
    make them the same population."""
    reg = metadata.load_registry(Path(__file__).parents[1] / "configs/studies.yaml")
    axis = reg["GSE79362"].axes["progression"]
    assert axis.ready
    assert axis.status == "REDUNDANT_WITH_TBSTATUS"
    frames = {"GSE79362": pd.DataFrame(), "GSE94438": pd.DataFrame()}
    with pytest.raises(metadata.LabelAxisError):
        metadata.pool(frames, reg, "progression")


def test_donor_namespacing_prevents_cross_study_collision():
    """Two cohorts can independently use the ID '01_0524'."""
    a = pd.DataFrame({"donor_id": ["01_0524"], "study": ["A"]})
    b = pd.DataFrame({"donor_id": ["01_0524"], "study": ["B"]})
    a["donor_id"] = a.study + "::" + a.donor_id
    b["donor_id"] = b.study + "::" + b.donor_id
    assert set(a.donor_id) & set(b.donor_id) == set()


def test_batch_balanced_minibatches_are_balanced():
    df = toy(n_donors=30, per_donor=2).reset_index(drop=True)
    for idx in splits.batch_balanced_batches(df, batch_size=20):
        counts = df.study.iloc[idx].value_counts()
        assert counts.nunique() == 1


def test_cramers_v_extremes():
    n = 200
    perfect = pd.DataFrame({"b": ["X"] * n + ["Y"] * n, "l": [1] * n + [0] * n})
    assert audit.cramers_v(perfect.b, perfect.l) > 0.95
    rng = np.random.default_rng(0)
    indep = pd.DataFrame({"b": rng.choice(["X", "Y"], 2 * n),
                          "l": rng.choice([0, 1], 2 * n)})
    assert audit.cramers_v(indep.b, indep.l) < 0.15


def test_leakage_probability_zero_when_one_sample_per_donor():
    df = toy(n_donors=100, per_donor=1, studies=("A",))
    assert audit.leakage_probability(df, n_sim=200)["mean_donors_leaked"] == 0.0


def test_relabelled_axis_is_detected_as_redundant():
    """`Progression` in GSE79362/GSE94438 is a pure relabelling of `TBStatus`,
    and it renames two different control populations to the same string. The
    guard must catch this without relying on how the registry annotated it."""
    def cohort(study, n_neg, n_pos, neg_status):
        return pd.DataFrame({
            "study": study,
            "Progression": ["Negative"] * n_neg + ["Positive"] * n_pos,
            "TBStatus": [neg_status] * n_neg + ["PTB"] * n_pos,
        })
    df = pd.concat([cohort("GSE79362", 245, 110, "LTBI"),
                    cohort("GSE94438", 327, 101, "Control")], ignore_index=True)
    res = audit.independent_axes(df, ["TBStatus", "Progression"])
    assert res["redundant"] == {"Progression": "TBStatus"}
    assert res["independent"] == ["TBStatus"]


def test_genuinely_independent_axis_is_not_flagged():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "study": ["A"] * 100 + ["B"] * 100,
        "TBStatus": rng.choice(["PTB", "LTBI"], 200),
        "Sex": rng.choice(["M", "F"], 200),
    })
    res = audit.independent_axes(df, ["TBStatus", "Sex"])
    assert res["redundant"] == {}


# --- time axis -------------------------------------------------------------

from tbbatch import timeaxis  # noqa: E402


def test_units_are_harmonised_across_cohorts():
    """GSE79362 records days, GSE94438 records months. Concatenating without
    conversion would compress one cohort's timescale by ~30x."""
    assert timeaxis.parse_one("642 Day(s)").days == 642.0
    assert abs(timeaxis.parse_one("22 month(s)").days - 669.625) < 1e-6
    assert abs(timeaxis.parse_one("1 month(s)").days - 30.4375) < 1e-6


def test_sentinel_is_not_treated_as_a_value():
    """'---' is a string, so R's is.na() reports FALSE for it. Counting
    missingness from is.na() alone undercounts."""
    p = timeaxis.parse_one("---")
    assert p.days is None and p.kind == "sentinel"
    s = pd.Series(["10 Day(s)", "---", None])
    rep = timeaxis.audit_series(s)
    assert rep["is_na_would_report"] == 1      # what is.na() sees
    assert rep["true_missing"] == 2            # what is actually missing
    assert rep["undercount_by_sentinels"] == 1


def test_negative_times_are_parsed_but_flagged_not_silently_dropped():
    assert timeaxis.parse_one("-91 Day(s)").days == -91.0
    rep = timeaxis.audit_series(pd.Series(["-91 Day(s)", "10 Day(s)"]))
    assert rep["n_negative"] == 1


def test_progressor_window_drops_controls_and_post_diagnosis_samples():
    df = pd.DataFrame({
        "TimeToTB": ["100 Day(s)", "-50 Day(s)", "200 Day(s)", "300 Day(s)", "---"],
        "Progression": ["Positive", "Positive", "Negative", "Positive", "Positive"],
    })
    out = timeaxis.progressor_window(df, "TimeToTB", "Progression")
    assert list(out.days_to_tb) == [100.0, 300.0]   # control, post-dx, sentinel all gone


def test_sentinels_are_not_counted_as_censoring_times():
    """Regression test for a real analysis error.

    In GSE79362 all 166 '---' rows are non-progressors. An is.na()-based tally
    reports '166 negatives carry a time', which invites the (wrong) conclusion
    that these are censoring times and the data are right-censored survival.
    They hold no value at all. The parser must classify them as sentinels so
    that negatives_with_time is 0, not 166."""
    df = pd.DataFrame({
        "TimeToTB": ["---"] * 166 + [None] * 79 + ["100 Day(s)"] * 98 + [None] * 12,
        "Progression": ["Negative"] * 245 + ["Positive"] * 110,
    })
    rep = timeaxis.audit_series(df.TimeToTB, df.Progression)
    assert rep["is_na_would_report"] == 91          # what R's is.na() sees
    assert rep["true_missing"] == 257               # 91 + 166 sentinels
    assert rep["undercount_by_sentinels"] == 166
    assert rep["negatives_with_time"] == 0          # the claim that was wrong
    assert rep["censored_structure_likely"] is False


def test_censoring_structure_still_detected_when_genuinely_present():
    """The detector must not simply always return False."""
    df = pd.DataFrame({
        "TimeToTB": ["100 Day(s)"] * 3 + [None],
        "Progression": ["Positive", "Negative", "Negative", "Positive"],
    })
    rep = timeaxis.audit_series(df.TimeToTB, df.Progression)
    assert rep["negatives_with_time"] == 2
    assert rep["censored_structure_likely"] is True


def test_power_matches_the_donor_counts_in_the_readme():
    """116 donors, not 211 samples. The distinction decides whether the
    progressor time axis can be trained on or only evaluated on."""
    assert abs(timeaxis.power_spearman(40, 0.4) - 0.73) < 0.02
    assert abs(timeaxis.power_spearman(76, 0.4) - 0.95) < 0.02
    assert abs(timeaxis.power_spearman(116, 0.3) - 0.91) < 0.02


def test_within_donor_structure_distinguishes_the_two_cohorts():
    """GSE79362 (2.75 samples/progressor) supports a within-person contrast;
    GSE94438 (1.33) mostly does not."""
    rich = pd.DataFrame({"donor_id": np.repeat([f"d{i}" for i in range(20)], 3),
                         "days_to_tb": np.tile([600., 400., 200.], 20)})
    sparse = pd.DataFrame({"donor_id": [f"e{i}" for i in range(20)],
                           "days_to_tb": np.linspace(100, 700, 20)})
    r = timeaxis.donor_time_structure(rich)
    s = timeaxis.donor_time_structure(sparse)
    assert r["supports_within_donor_design"] is True
    assert r["median_within_donor_range_days"] == 400.0
    assert s["supports_within_donor_design"] is False
    assert s["median_within_donor_range_days"] is None


def test_same_timepoint_replicates_are_not_a_longitudinal_design():
    """Regression test. GSE94438 has 20 progressor donors with more than one
    sample, but every one of them has a within-donor time spread of exactly 0:
    they are replicates at a single timepoint, not longitudinal sampling. An
    earlier version flagged count>1 as sufficient and wrongly reported that
    GSE94438 supported a within-person contrast."""
    reps = pd.DataFrame({"donor_id": np.repeat([f"r{i}" for i in range(20)], 2),
                         "days_to_tb": np.repeat(np.linspace(100, 700, 20), 2)})
    out = timeaxis.donor_time_structure(reps)
    assert out["n_donors_multi"] == 20            # they do have repeats
    assert out["n_donors_with_time_spread"] == 0  # but none across time
    assert out["supports_within_donor_design"] is False
    assert out["median_within_donor_range_days"] is None
