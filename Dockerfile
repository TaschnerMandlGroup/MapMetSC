# Docker inheritance
FROM rocker/rstudio:4.4.0

# who maintains this image
LABEL maintainer daria lazic 
LABEL version mapmetscv2

RUN apt-get -y update \
    && apt-get install -y --no-install-recommends apt-utils \
    && apt-get install -y --no-install-recommends zlib1g-dev libglpk-dev libmagick++-dev libfftw3-dev libxml2-dev libxt-dev curl libcairo2-dev libproj-dev libgdal-dev libudunits2-dev libarchive-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/ilists/*

RUN R -e 'install.packages(c("rmarkdown", "pheatmap", "viridis", "BiocManager", "devtools", "tiff", \
                             "ggrepel", "patchwork", "mclust", "RColorBrewer", "uwot", "Seurat", \
                             "SeuratObject", "cowplot", "ggridges", "gridGraphics", "scales", "Matrix"))'
RUN R -e 'BiocManager::install(c("CATALYST", "scater", "dittoSeq", "tidyverse", "lisaClust", "imcRtools", \
                                 "cytomapper", "ComplexHeatmap", "BiocParallel", "SingleR", "diffcyt", \
                                 "BioQC", "edgeR", "stringr", "circlize", "paletteer", "tidyHeatmap", "bruceR", \
                                 "FactoMineR", "factoextra", "ggpubr", "rstatix", "gridExtra", "gtools", "forcats", "openxlsx"))'
RUN R -e 'devtools::install_github("stuchly/Rphenoannoy@8b81e2e7fb0599f45070e2cba1b28ac219b7c472")'