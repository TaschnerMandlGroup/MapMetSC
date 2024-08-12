<img src="https://github.com/TaschnerMandlGroup/MapMetSC/blob/main/docs/img/logo.png" align="right" alt="Logo" width="55" />

# MapMetSC
[comment]: <> (repo-specific shields will work once the repo is online)
![Suggestions Welcome](https://img.shields.io/badge/suggestions-welcome-green)

_This repository is currently under development._

This repository provides scripts to reproduce figures published in Lazic et al (in preparation). In brief, we spatially and temporally mapped primary and metastatic neuroblastoma by multi-modal high-plex imaging in combination with scRNA-seq datasets. The data analysis worflow in Lazic et al. is split into two parts:
- [MapMetIP](https://github.com/TaschnerMandlGroup/MapMetIP): `python` based image-processing pipeline 
- [MapMetSC](https://github.com/TaschnerMandlGroup/MapMetSC): `R` based single-cell analysis worfklow 

## Data 

Single-cell data can either be obtained by downloading raw images from Zenodo [(MapMetIP_FullDataset.zip)](10.5281/zenodo.13220634) and processing these using [MapMetIP](https://github.com/TaschnerMandlGroup/MapMetIP) or by directly downloading MapMetIP-processed data from Zenodo [(MapMetIP_ProcessedDataset.zip)](10.5281/zenodo.13220634). For download, replace path/to/extract/directory with the absolute path to the directory, where the data should be stored.

 ```bash
wget -P <path/to/extract/directory> https://zenodo.org/records/13220635/files/MapMetIP_ProcessedDataset.zip
unzip <path/to/extract/directory>/MapMetIP_ProcessedDataset.zip -d <path/to/extract/directory>
rm <path/to/extract/directory>/MapMetIP_ProcessedDataset.zip
 ```
Alternatively, the R object containing already MapMetSC-processed single cell data with all annotations (cell types, metaclusters, spatial neighborhoods, etc.) can be downloaded from Zenodo [(MapMet_final_scobject.rds)](10.5281/zenodo.13220634).

## Usage
  
Clone the repository via
 ```bash
 git clone https://github.com/TaschnerMandlGroup/MapMetSC.git
 ```
For reproducibility, we provide a docker image. Pull the docker image using:
 ```bash
 docker pull lazdaria/mapmetsc:v1.0
 ```
 and run a container from that image:
 ```bash
docker run -p 8787:8787 -e PASSWORD=mapmetsc -v <path/to/MapMetSC>:/home/rstudio/MapMetSC -v <path/to/extracted/singlecelldata>:/mnt/data lazdaria/mapmetsc:v1.0
 ```
 An RStudio server session can then be accessed via your browser at `localhost:8787` with the `username: rstudio` and `password: mapmetsc`. Due to a bug fix for the `testInteractions` function in later versions of imcRtools, we provide another docker image ([lazdaria/mapmetsc_spatial:v1.0](https://hub.docker.com/repository/docker/lazdaria/mapmetsc_spatial/general)) for [spatial analysis](https://github.com/TaschnerMandlGroup/MapMetSC/tree/main/analysis/10_spatial_analysis.Rmd).

To reproduce results from Lazic et al., proceed with the provided [RMD files](https://github.com/TaschnerMandlGroup/MapMetSC/tree/main/analysis). Alternatively, already rendered [html files](https://github.com/TaschnerMandlGroup/MapMetSC/tree/main/docs) are provided to demonstrate each step of the pipeline. 

 ### Cell-cell communication (CCC) analysis
 
To reproduce CCC analysis results on public single-cell RNA-sequencing data from [Fetahu et al.](10.5281/zenodo.7707614), as described in Lazic et al., we provide a separate docker image. Pull the image from docker hub via:
 ```bash
 docker image pull swernig/mapmet_paper:v1.2
```
Afterwards, proceed with RMD files in the folder [CCC_analysis](https://github.com/TaschnerMandlGroup/MapMetSC/tree/main/CCC_analysis).
  
## Contributors

- [Daria Lazic](https://github.com/LazDaria)
- [Sara Wernig-Zorc](https://github.com/sarawernig)
- [Simon Gutwein](https://github.com/SimonBon/)

## Funding

This work was funded by the Austrian Science Fund (FWF#I4162 and FWF#35841), the Vienna Science and Technology Fund (WWTF; LS18-111), the Swiss Government Excellence Scholarship and the St. Anna Kinderkrebsforschung e.V.

