"""Schema harmonisation across cohorts.

Each cohort ships its own colData column names and, more importantly, its own
class semantics. This module maps them onto a single canonical frame and
refuses to silently paper over semantic mismatches.

Canonical columns
-----------------
sample_id, donor_id, study, site, age, sex, lib_size, label, label_axis
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

CANONICAL = ["sample_id", "donor_id", "study", "site", "age", "sex", "lib_size"]


class LabelAxisError(ValueError):
    """Raised when studies are pooled on an axis they do not actually share."""


@dataclass(frozen=True)
class LabelAxis:
    name: str
    column: str
    positive: list[str] | None
    negative: list[str] | None
    control_definition: str
    status: str = "READY"

    @property
    def ready(self) -> bool:
        return self.positive is not None and self.negative is not None


@dataclass(frozen=True)
class Study:
    name: str
    cohort: str
    metadata_csv: str
    columns: dict[str, str]
    axes: dict[str, LabelAxis]
    longitudinal: bool
    timepoint_column: str | None


def load_registry(path: str | Path) -> dict[str, Study]:
    cfg = yaml.safe_load(Path(path).read_text())
    out: dict[str, Study] = {}
    for name, block in cfg["studies"].items():
        axes = {
            axis_name: LabelAxis(
                name=axis_name,
                column=spec["column"],
                positive=spec.get("positive"),
                negative=spec.get("negative"),
                control_definition=spec["control_definition"],
                status=spec.get("status", "READY"),
            )
            for axis_name, spec in block["label_axes"].items()
        }
        out[name] = Study(
            name=name,
            cohort=block["cohort"],
            metadata_csv=block["metadata_csv"],
            columns=block["columns"],
            axes=axes,
            longitudinal=block.get("longitudinal", False),
            timepoint_column=block.get("timepoint_column"),
        )
    return out


def load_study_metadata(study: Study, root: str | Path = ".") -> pd.DataFrame:
    """Read one cohort's metadata and rename to canonical columns."""
    df = pd.read_csv(Path(root) / study.metadata_csv)
    rename = {src: dst for dst, src in study.columns.items() if src in df.columns}
    out = df.rename(columns=rename).copy()
    out["study"] = study.name

    missing = [c for c in CANONICAL if c not in out.columns]
    if missing:
        raise KeyError(f"{study.name}: canonical columns absent after rename: {missing}")

    # Donor IDs are only unique within a cohort ("01_0524" style collides
    # easily). Namespace them so a pooled frame can never merge two people.
    out["donor_id"] = out["study"] + "::" + out["donor_id"].astype(str)
    return out


def attach_label(df: pd.DataFrame, axis: LabelAxis) -> pd.DataFrame:
    """Binarise one label axis. Rows outside the declared classes are dropped."""
    if not axis.ready:
        raise LabelAxisError(
            f"axis '{axis.name}' has status {axis.status}: positive/negative "
            "class values are not defined in the registry yet."
        )
    if axis.column not in df.columns:
        raise KeyError(f"label column '{axis.column}' not present")

    out = df.copy()
    out["label"] = pd.NA
    out.loc[out[axis.column].isin(axis.positive), "label"] = 1
    out.loc[out[axis.column].isin(axis.negative), "label"] = 0
    out = out[out["label"].notna()].copy()
    out["label"] = out["label"].astype(int)
    out["label_axis"] = axis.name
    return out


def pool(
    frames: dict[str, pd.DataFrame],
    registry: dict[str, Study],
    axis_name: str,
    allow_disjoint_controls: bool = False,
) -> pd.DataFrame:
    """Concatenate cohorts on a shared label axis.

    Refuses by default when the negative classes are disjoint across studies,
    because then `study` alone determines what "control" means and any pooled
    classifier can exploit it.
    """
    controls = {s: registry[s].axes[axis_name].control_definition for s in frames}
    if len(set(controls.values())) > 1 and not allow_disjoint_controls:
        raise LabelAxisError(
            f"axis '{axis_name}' has study-specific control definitions "
            f"{controls}. Pooling would confound `study` with the meaning of "
            "the negative class. Pass allow_disjoint_controls=True only if "
            "you are doing unsupervised representation work."
        )
    return pd.concat(frames.values(), ignore_index=True)
