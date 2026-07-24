"""Splitters that cannot silently leak.

Two nested constraints:
  outer  - leave-one-study-out : tests transfer across cohorts
  inner  - donor-grouped folds : stops a person appearing on both sides

Every split is validated before it is returned. A leaking split raises rather
than quietly producing an optimistic number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


class LeakageError(AssertionError):
    """A split placed the same donor or study on both sides."""


@dataclass(frozen=True)
class Split:
    name: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.train_idx) + len(self.test_idx)


def _validate(df: pd.DataFrame, tr: np.ndarray, te: np.ndarray,
              name: str, check_study: bool) -> None:
    if len(tr) == 0 or len(te) == 0:
        raise LeakageError(f"{name}: empty side (train={len(tr)}, test={len(te)})")
    if set(tr) & set(te):
        raise LeakageError(f"{name}: overlapping row indices")

    shared_donors = set(df.donor_id.iloc[tr]) & set(df.donor_id.iloc[te])
    if shared_donors:
        raise LeakageError(
            f"{name}: {len(shared_donors)} donor(s) on both sides, "
            f"e.g. {sorted(shared_donors)[:3]}"
        )
    if check_study:
        shared_studies = set(df.study.iloc[tr]) & set(df.study.iloc[te])
        if shared_studies:
            raise LeakageError(f"{name}: study on both sides: {shared_studies}")

    if "label" in df.columns:
        for side, idx in (("train", tr), ("test", te)):
            if df.label.iloc[idx].nunique() < 2:
                raise LeakageError(f"{name}: {side} side is single-class")


def leave_one_study_out(df: pd.DataFrame) -> Iterator[Split]:
    """Outer loop. Train on all cohorts but one, test on the held-out cohort.

    This is the only split that answers "does the representation transfer",
    because it is the only one where the test batch was never seen.
    """
    studies = sorted(df.study.unique())
    if len(studies) < 2:
        raise ValueError("leave-one-study-out needs >= 2 studies")
    pos = np.arange(len(df))
    for held in studies:
        mask = (df.study == held).to_numpy()
        tr, te = pos[~mask], pos[mask]
        _validate(df, tr, te, f"LOSO[hold={held}]", check_study=True)
        yield Split(
            name=f"LOSO[hold={held}]",
            train_idx=tr,
            test_idx=te,
            meta={"held_out_study": held,
                  "n_train": int(len(tr)), "n_test": int(len(te))},
        )


def donor_grouped_kfold(df: pd.DataFrame, n_splits: int = 5,
                        seed: int = 0) -> Iterator[Split]:
    """Inner loop. Stratified on label, grouped on donor.

    Grouping on donor is what makes the number honest for a longitudinal
    cohort: with ~2.5 samples per person, ungrouped CV is scoring the model
    on people it already trained on.
    """
    pos = np.arange(len(df))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(
        sgkf.split(pos, df.label.to_numpy(), groups=df.donor_id.to_numpy())
    ):
        _validate(df, tr, te, f"donorCV[fold={k}]", check_study=False)
        yield Split(
            name=f"donorCV[fold={k}]",
            train_idx=tr,
            test_idx=te,
            meta={"fold": k,
                  "n_train_donors": int(df.donor_id.iloc[tr].nunique()),
                  "n_test_donors": int(df.donor_id.iloc[te].nunique())},
        )


def nested(df: pd.DataFrame, n_inner: int = 5, seed: int = 0) -> Iterator[tuple[Split, list[Split]]]:
    """Full protocol: LOSO outside, donor-grouped CV inside the training pool.

    Hyperparameters are chosen on the inner folds only. The held-out study is
    touched exactly once, at the end.
    """
    for outer in leave_one_study_out(df):
        pool = df.iloc[outer.train_idx].reset_index(drop=True)
        inner = list(donor_grouped_kfold(pool, n_splits=n_inner, seed=seed))
        yield outer, inner


def batch_balanced_batches(df: pd.DataFrame, batch_size: int,
                           batch_col: str = "study", seed: int = 0) -> Iterator[np.ndarray]:
    """Mini-batches with equal representation per technical batch.

    Matters for adversarial training: an unbalanced mini-batch lets the
    discriminator win on marginal frequency alone, and lets any BatchNorm
    layer leak batch identity through its running statistics.
    """
    rng = np.random.default_rng(seed)
    groups = {b: df.index[df[batch_col] == b].to_numpy() for b in df[batch_col].unique()}
    per = max(1, batch_size // len(groups))
    pools = {b: rng.permutation(idx).tolist() for b, idx in groups.items()}
    while all(len(p) >= per for p in pools.values()):
        take = []
        for b in pools:
            take.extend(pools[b][:per])
            pools[b] = pools[b][per:]
        yield np.array(take)
