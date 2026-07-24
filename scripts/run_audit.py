#!/usr/bin/env python3
"""Run the design audit on real cohort metadata and emit a markdown report.

    python scripts/run_audit.py --registry configs/studies.yaml --root .

Produces results/design_audit.md. No count matrices required: everything here
is decidable from metadata alone, which is why it runs first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tbbatch import audit, metadata, splits  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="configs/studies.yaml")
    ap.add_argument("--root", default=".")
    ap.add_argument("--axis", default="tbstatus")
    ap.add_argument("--out", default="results/design_audit.md")
    args = ap.parse_args()

    reg = metadata.load_registry(args.registry)
    L: list[str] = ["# Design audit", ""]
    L.append("Generated from cohort metadata before any model is trained. ")
    L.append("Everything below is a property of the study design, not of a method.")
    L.append("")

    frames = {}
    L.append("## 1. Cohort inventory")
    L.append("")
    rows = []
    for name, st in reg.items():
        df = metadata.load_study_metadata(st, args.root)
        lab = metadata.attach_label(df, st.axes[args.axis])
        frames[name] = lab
        d = audit.donor_structure(lab)
        rows.append({
            "study": name,
            "samples": d["n_samples"],
            "donors": d["n_donors"],
            "donors w/ repeats": d["donors_with_repeats"],
            "max/donor": d["max_samples_per_donor"],
            "eff. N inflation": f"{d['effective_n_inflation']}x",
            "control class": st.axes[args.axis].control_definition,
        })
    L.append(pd.DataFrame(rows).to_markdown(index=False))
    L.append("")

    L.append("## 2. Label-axis compatibility")
    L.append("")
    controls = {n: reg[n].axes[args.axis].control_definition for n in reg}
    if len(set(controls.values())) > 1:
        L.append(f"**Blocked.** Axis `{args.axis}` has disjoint negative classes across "
                 f"cohorts: {controls}.")
        L.append("")
        L.append("`study` therefore determines the *meaning* of the negative class. A pooled "
                 "classifier can reach the right answer for the wrong reason, and no amount of "
                 "adversarial batch removal fixes it, because the confound is in the label "
                 "definition rather than in the expression values. This axis is usable for "
                 "within-study analysis and for unsupervised representation work only.")
        pooled_ok = False
    else:
        L.append(f"Axis `{args.axis}` shares a control definition across cohorts. Pooling permitted.")
        pooled_ok = True
    L.append("")

    pooled = metadata.pool(frames, reg, args.axis, allow_disjoint_controls=True)

    L.append("## 3. Batch/label confounding")
    L.append("")
    L.append("Cramer's V between the technical variable and the label. Low values mean the "
             "design contains the counter-examples an adversarial model needs.")
    L.append("")
    for col, scope, frame in [
        ("study", "pooled", pooled),
        ("site", "GSE94438", frames["GSE94438"]),
    ]:
        if frame[col].nunique() < 2:
            continue
        rep = audit.confounding_report(frame, col)
        L.append(f"### {col} x label ({scope})")
        L.append("")
        L.append(rep["table"].to_markdown())
        L.append("")
        L.append(f"Cramer's V = **{rep['cramers_v']}** ({rep['severity']}) - {rep['note']}")
        L.append("")

    L.append("## 4. Donor leakage under naive splitting")
    L.append("")
    L.append("Monte-Carlo over 2000 random 80/20 sample-level splits: how many donors land "
             "on both sides.")
    L.append("")
    rows = []
    for name, f in frames.items():
        lk = audit.leakage_probability(f)
        rows.append({"study": name, **lk})
    L.append(pd.DataFrame(rows).to_markdown(index=False))
    L.append("")

    L.append("## 5. Protocol selected")
    L.append("")
    if pooled_ok:
        L.append("- Outer: leave-one-study-out")
    else:
        L.append("- Outer: leave-one-study-out **(representation metrics only on this axis)**")
    L.append("- Inner: donor-grouped stratified 5-fold, on the training pool only")
    L.append("- Mini-batches: batch-balanced sampling across cohorts")
    L.append("- Every split validated by assertion; a leaking split raises rather than scores")
    L.append("")

    L.append("### Split validation")
    L.append("")
    for sp in splits.leave_one_study_out(pooled):
        L.append(f"- `{sp.name}` train={sp.meta['n_train']} test={sp.meta['n_test']} - passed")
    pool_df = pooled[pooled.study == "GSE79362"].reset_index(drop=True)
    for sp in splits.donor_grouped_kfold(pool_df, n_splits=5):
        L.append(f"- `{sp.name}` (GSE79362) train_donors={sp.meta['n_train_donors']} "
                 f"test_donors={sp.meta['n_test_donors']} - passed")
    L.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
