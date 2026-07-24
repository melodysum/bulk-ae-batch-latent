# Expression matrices (not committed)

`scripts/run_baseline.py` expects two files here:

```
GSE79362_logCPM_csv.gz    18,608 genes x 355 samples
GSE94438_logCPM_csv.gz    16,196 genes x 434 samples
```

They are gitignored: ~15 MB each, and better regenerated from the canonical
source than mirrored. Intersecting them gives the 15,264-gene shared space
used throughout.

## Regenerate

```r
suppressPackageStartupMessages({
  library(MultiAssayExperiment); library(curatedTBData); library(edgeR)
})

export_counts <- function(study, outdir = "data/expr") {
  dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
  obj  <- curatedTBData(study, dry.run = FALSE, curated.only = FALSE)[[study]]
  cnt  <- as.matrix(experiments(obj)[["assay_reprocess_hg38"]])
  cnt  <- rowsum(cnt, group = rownames(cnt), reorder = FALSE)
  keep <- rowSums(cpm(cnt) > 1) >= 20
  lcpm <- cpm(cnt[keep, , drop = FALSE], log = TRUE, prior.count = 1)
  gz <- gzfile(file.path(outdir, paste0(study, "_logCPM_csv.gz")), "w")
  write.csv(round(lcpm, 3), gz); close(gz)
  cat(study, ": genes =", nrow(lcpm), " samples =", ncol(lcpm), "\n")
}

export_counts("GSE79362")
export_counts("GSE94438")
```

The `>= 20` filter is this project's choice and differs from the sibling DE
analysis, which used a group-size rule. That is why the shared gene count is
15,264 here and 14,128 there.

## Raw counts (for scVI)

`scripts/run_scvi.py` needs the raw count matrices, not log-CPM:

```
GSE79362_counts_csv.gz
GSE94438_counts_csv.gz
```

Regenerate with the same R session, writing raw `cnt[keep, ]` instead of `lcpm`.

## GSE107994 (third cohort)

```
GSE107994_logCPM_csv.gz    16,543 genes x 175 samples
GSE107994_counts_csv.gz    (for scVI, optional)
```

UK adult LTBI contacts (Leicester). Used for the discrimination question only
(section 7.0): 9 progressor donors, 49 non-progressor donors. No continuous
TimeToTB field. Regenerate with the same R snippet, study = "GSE107994".
