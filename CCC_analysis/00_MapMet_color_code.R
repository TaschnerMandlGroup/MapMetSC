#!/usr/bin/env Rscript

# load R packages

library("Seurat")
library("SingleCellExperiment")
library("tidyverse")
library("ComplexHeatmap")
library("stringr")
library("tidyverse")
library("CellChat")
library("monocle3")
library("SpatialExperiment")
library("SingleR")
library("magrittr")
library("RColorBrewer")
library("DElegate")
library("dplyr")
library("ggalluvial")
library("NMF")
library("clusterProfiler")
library("org.Hs.eg.db")
library("DOSE")
library("meshes")


# scRNA-seq color code

PATIENT_GROUPS = c("I" = "Control",
                   "II" = "MYCNamp",
                   "III" = "ATRXmut",
                   "IV" = "MYCNwt/ATRXwt")

# UMAP color code
COLOR_CODE_RNA_v3 = c("NB (8)" = "palegreen4",
                      "T (5)" = "#cc0000",   
                      "T (6)" = "darkred", 
                      "T (9)" = "#f44336",   
                      "M (1)" = "#1f78b4", 
                      "M (2)" = "#66a8de",    
                      "M (10)" = "#39BEB1")

COLOR_CODE_RNA_v3_EXT = c("NB (8)" = "palegreen4",
                      "T (5)" = "#cc0000",   
                      "T (6)" = "darkred", 
                      "T (9)" = "#f44336",
                      "T (18)" = "#ac2020",
                      "NK (4)" = "#0B5394",
                      "M (1)" = "#1f78b4", 
                      "M (2)" = "#66a8de",    
                      "M (10)" = "#39BEB1",
                      "M (15)"= "#124e77",
                      "B (19)" = "#992669",  
                      "B (3)" = "#fc58a9",   
                      "B (7)" = "#9e3b75",   
                      "B (11)" = "#7f0d71",   
                      "B (16)" = "#8a1bca",   
                      "pDC (12)" = "#15acbc",   
                      "E (13)" = "#113556",   
                      "SC (14)" = "#7f5b22",  
                      "SC (17)" = "#997439",
                      "SC (20)" = "#573807",  
                      "other (21)" = "#000000")


COLOR_CODE_RNA_v2 = c("NB (8)" = "#e31a1c",
                   "T (5)" = "#1f78b4",   
                   "T (6)" = "#1f78b4", 
                   "T (9)" = "#39BEB1",   
                   "T (18)" = "#1f78b4",  
                   "NK (4)" = "#a6cee3",   
                   "B (3)" = "#33a02c",   
                   "B (7)" = "#33a02c",   
                   "B (11)" = "#33a02c",   
                   "B (16)" = "#33a02c",   
                   "B (19)" = "#33a02c",  
                   "M (1)" = "#ee6363", 
                   "M (2)" = "#ff7f00",    
                   "M (10)" = "#ff7f00",    
                   "M (15)"= "#ff7f00", 
                   "pDC (12)" = "#fdbf6f",   
                   "E (13)" = "#b15928",   
                   "SC (14)" = "#6a3d9a",  
                   "SC (17)" = "#6a3d9a",
                   "SC (20)" = "#6a3d9a",  
                   "other (21)" = "#000000")

COLOR_CODE_RNA = c("NB (0)" = "#8B0000", 
               "NB (5)" = "#e31a1c",
               "NB (4)" = "#e31a1c",      
               "NB (3)" = "#e31a1c",     
               "NB (2)" = "#e31a1c",     
               "NB (1)" = "#e31a1c", 
               "NB (8)" = "#e31a1c",
               "T (5)" = "#1f78b4",   
               "T (6)" = "#1f78b4", 
               "T (9)" = "#39BEB1",   
               "T (18)" = "#1f78b4",  
               "NK (4)" = "#a6cee3",   
               "B (3)" = "#33a02c",   
               "B (7)" = "#33a02c",   
               "B (11)" = "#33a02c",   
               "B (16)" = "#33a02c",   
               "B (19)" = "#33a02c",  
               "M (1)" = "#ee6363", 
               "M (2)" = "#ff7f00",    
               "M (10)" = "#ff7f00",    
               "M (15)"= "#ff7f00", 
               "pDC (12)" = "#fdbf6f",   
               "E (13)" = "#b15928",   
               "SC (14)" = "#6a3d9a",  
               "SC (17)" = "#6a3d9a",
               "SC (20)" = "#6a3d9a",  
               "other (21)" = "#000000")

GROUP_NAMES_LONG = c(
  C = "control",
  M = "MYCN amplified",
  A = "ATRXmut",
  S = "sporadic"
)

CELL_TYPE_ABBREVIATIONS <- c(
  "T" = "T cell",
  NK  = "natural killer cell",
  B   = "B cell",
  M   = "myeloid cell",
  pDC = "plasmacytoid dendritic cell",
  E   = "erythroid lineage cell",
  SC  = "hematopoietic precursor cell",
  NB  = "neuroblastoma cell"
)

COMMON_SAMPLES_IMC_RNA <- c("2016_4503", # Group II
                            "2018_1404", # Group II
                            "2019_5754") # Group II

HIGHLIGHT_PATIENTS <- c("2005_1702" = "plain",
                        "2006_2684" = "plain",
                        "2014_0102" = "plain",
                        "2016_1853" = "plain",
                        "2016_2950" = "plain",
                        "2016_3924" = "plain",
                        "2016_4503" = "bold",
                        "2018_1404" = "bold",
                        "2018_1625" = "plain",
                        "2018_4252" = "plain",
                        "2018_6056" = "plain",
                        "2019_2495" = "plain",
                        "2019_5022" = "plain",
                        "2019_5754" = "bold",
                        "2020_1288" = "plain",
                        "2020_1667" = "plain")

HIGHLIGHT_PATIENTS_TUMOR_CELLS <- c("2005_1702" = "plain",
                                    "2006_2684" = "plain",
                                    "2016_3924" = "plain",
                                    "2016_4503" = "bold",
                                    "2018_1404" = "bold",
                                    "2018_1625" = "plain",
                                    "2018_6056" = "plain",
                                    "2019_2495" = "plain",
                                    "2019_5022" = "plain",
                                    "2019_5754" = "bold",
                                    "2020_1667" = "plain")

# IMC color code

COLOR_CODE_IMC <- c("fibroblast/endothel"=rgb(212,89,67, maxColorValue = 255),
                  "schwann cell"=rgb(156,76,24, maxColorValue = 255),
                  "MSC"=rgb(174,129,37, maxColorValue = 255),
                  "HPC"=rgb(212,143,75, maxColorValue = 255),
                  "HSPC"=rgb(253,180,98, maxColorValue = 255),
                  "Ki67+ CXCR4+ cell"=rgb(254,215,0, maxColorValue = 255),
                  "LIN- Ki67+"=rgb(254,217,166, maxColorValue = 255),
                  "B cell"=rgb(168,128,250, maxColorValue = 255),
                  "GZMB+ S100B- DC/NK"=rgb(28,146,255, maxColorValue = 255),
                  "GZMB+ S100B+ DC/NK"=rgb(0,255,255, maxColorValue = 255),
                  "CD8+ naive T cell"=rgb(188,128,189, maxColorValue = 255),
                  "DN GZMB+ T cell"=rgb(190,174,212, maxColorValue = 255),
                  "CD8+ S100B+ T cell"=rgb(253,218,236, maxColorValue = 255),
                  "CD8+ GZMB+ PD1lo T cell"=rgb(251,154,153, maxColorValue = 255),
                  "CD4+ naive T cell"=rgb(242,3,137, maxColorValue = 255),
                  "DN T cell"=rgb(251,128,114, maxColorValue = 255),
                  "CD4+ PD1+ T cell"=rgb(254,142,175, maxColorValue = 255),
                  "CD8+ PDL1lo T cell"=rgb(240,0,240, maxColorValue = 255),
                  "CD4+ S100B+ T cell"=rgb(227,26,28, maxColorValue = 255),
                  "dense T cell region"=rgb(145,2,144, maxColorValue = 255),
                  "CD14+ PDL1- MO"=rgb(104,180,238, maxColorValue = 255),
                  "CD14+ PDL1+ MO"=rgb(51,180,180, maxColorValue = 255),
                  "CD14- PDL1- MO/DC"=rgb(39,130,187, maxColorValue = 255),
                  "CD14- PDL1+ MO/DC"=rgb(166,206,227, maxColorValue = 255),
                  "monoblast-like"=rgb(128,177,211, maxColorValue = 255),
                  "myelocyte"=rgb(217,217,217, maxColorValue = 255),          
                  "myeloblast"=rgb(2,0,138, maxColorValue = 255),
                  "neutrophil"=rgb(113,113,113, maxColorValue = 255),
                  "pDC"=rgb(212,241,240, maxColorValue = 255),
                  "band cell"=rgb(169,189,210, maxColorValue = 255),
                  "CD24+ marker-lo TC"=rgb(230,245,201, maxColorValue = 255),
                  "early SYM-like TC"=rgb(204,235,197, maxColorValue = 255),
                  "bridge-like TC"=rgb(198,255,72, maxColorValue = 255),
                  "GATA3hi TC"=rgb(65,208,57, maxColorValue = 255),
                  "CD24- marker-lo TC"=rgb(60,119,97, maxColorValue = 255),
                  "CHGAhi TC"=rgb(102,166,30, maxColorValue = 255),
                  "Ki67hi TC"=rgb(149,144,89, maxColorValue = 255),
                  "CXCR4hi TC"=rgb(210,205,126, maxColorValue = 255),
                  "CD44+ mes-like TC"=rgb(27,158,119, maxColorValue = 255),
                  "GD2lo TC"=rgb(102,194,165, maxColorValue = 255),
                  "neural progenitor"=rgb(238,118,0, maxColorValue = 255),
                  "other"=rgb(255,255,209, maxColorValue = 255))
