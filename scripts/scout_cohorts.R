#!/usr/bin/env Rscript
# Screen every curatedTBData study for a usable progression time axis.
# Supports the Limitations claim that continuous-time progression cohorts are
# exhausted within this collection.
suppressPackageStartupMessages({
  library(curatedTBData); library(MultiAssayExperiment)
})
data(DataSummary, package = "curatedTBData")

rows <- list()
for (gse in DataSummary$GEOAccession) {
  r <- tryCatch({
    obj <- curatedTBData(gse, dry.run = FALSE, curated.only = TRUE)[[gse]]
    cd  <- as.data.frame(colData(obj))
    has_prog <- "Progression" %in% colnames(cd)
    has_ttt  <- "TimeToTB" %in% colnames(cd)
    n_prog   <- if (has_prog) sum(cd$Progression == "Positive", na.rm = TRUE) else 0L
    data.frame(GSE = gse, n = nrow(cd), Progression = has_prog,
               TimeToTB = has_ttt, n_progressor = n_prog)
  }, error = function(e) data.frame(GSE = gse, n = NA, Progression = NA,
                                    TimeToTB = NA, n_progressor = NA))
  rows[[gse]] <- r
}
out <- do.call(rbind, rows)
usable <- subset(out, Progression & TimeToTB & n_progressor >= 20)
cat("Studies with progression label + continuous TimeToTB + >=20 progressors:\n")
print(usable[, c("GSE", "n", "n_progressor")])
write.csv(out, "results/cohort_screen.csv", row.names = FALSE)
