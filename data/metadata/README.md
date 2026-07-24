# Metadata provenance

These two CSVs contain **sample-level metadata only — no expression values.**
They are committed so that `scripts/run_audit.py` is reproducible without a
Bioconductor install or a large download.

| File | Source |
|---|---|
| `GSE79362_sample_qc.csv` | `curatedTBData::GSE79362`, `colData` of `assay_reprocess_hg38` |
| `GSE94438_sample_qc.csv` | `curatedTBData::GSE94438`, `colData` of `assay_reprocess_hg38` |

Both derive from publicly deposited GEO series (GSE79362, Zak et al. 2016;
GSE94438, Suliman et al. 2018) via the `curatedTBData` Bioconductor package.
Columns retained: `sample_id`, `TBStatus`, `GeographicalRegion`, `PatientID`,
`Age`, `Gender`, `library_size`, `low_lib`.

`PatientID` is the donor identifier used for repeated-measures grouping. It is
already pseudonymous in the source deposit; `metadata.py` further namespaces it
with the study accession so that identifiers cannot collide across cohorts.

## Not committed

Count matrices are not included (size, and they are better fetched from the
canonical source). To regenerate:

```r
library(curatedTBData)
obj <- curatedTBData("GSE79362", dry.run = FALSE, curated.only = FALSE)[["GSE79362"]]
counts <- as.matrix(experiments(obj)[["assay_reprocess_hg38"]])
```

## Missing for the supervised transfer experiment

The `Progression` and `TimeToTB` columns exist in the source `colData` of both
cohorts but were not exported into these QC tables. They are the only label
axis the two cohorts genuinely share (see `results/design_audit.md` §2), so the
supervised arm is blocked until they are added:

```r
write.csv(meta[, c("sample_id","Progression","TimeToTB","MeasurementTime")],
          "data/metadata/GSE79362_progression.csv", row.names = FALSE)
```
