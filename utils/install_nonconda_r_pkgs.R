#!/usr/bin/env Rscript

options(warn=1)
options(repos=structure(c(CRAN="https://cloud.r-project.org/")))
for (package_name in c("fuzzyRankTests", "bapred")) {
    if (!(package_name %in% installed.packages()))
        install.packages(package_name)
}
