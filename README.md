<img src="https://github.com/TaschnerMandlGroup/MapMetSC/blob/main/docs/img/logo.png" align="right" alt="Logo" width="55" />

# MapMetSC
[comment]: <> (repo-specific shields will work once the repo is online)
![Suggestions Welcome](https://img.shields.io/badge/suggestions-welcome-green)

_This repository is currently under development._

This repository provides scripts to reproduce figures published in Lazic et al (in preparation). In brief, we spatially and temporally mapped primary and metastatic neuroblastoma by multi-modal high-plex imaging in combination with scRNA-seq datasets. The data analysis worflow in Lazic et al. is split into two parts:
- [MapMetIP](https://github.com/TaschnerMandlGroup/MapMetIP): `python` based image-processing pipeline 
- [MapMetSC](https://github.com/TaschnerMandlGroup/MapMetSC): `R` based single-cell analysis worfklow 

## Data 
To use `MapMetSC`, please refer to and first proceed with [MapMetIP](https://github.com/TaschnerMandlGroup/MapMetIP) to download files, process images and extract **single-cell data**.

## Usage
  
Clone the repository via
 ```bash
 git clone https://github.com/TaschnerMandlGroup/MapMetSC.git
 ```
For reproducibility, build a docker image from the Dockerfile:
 ```bash
 docker build -t mapmet_sc .
 ```
 and run a container from that image:
 ```bash
docker run -p 8787:8787 -e PASSWORD=mapmetsc -v <path/to/MapMetSC>:/home/rstudio/MapMetSC -v <path/to/extracted/singlecelldata>:/mnt/data mapmet_sc
 ```
 An RStudio server session can then be accessed via your browser at `localhost:8787` with the `username: rstudio` and `password: mapmetsc`.

 ### Cell-cell communication (CCC) analysis
 
 To reproduce CCC analysis results on public single-cell RNA-sequencing data from [Fetahu et al.](10.5281/zenodo.7707614), as described in Lazic et al., we provide a separate docker image. Pull the image from docker hub via:
 ```bash
 docker image pull swernig/mapmet_paper:v1.2
```
Afterwards, proceed with RMD files the folder [CCC_analysis](https://github.com/TaschnerMandlGroup/MapMetSC/tree/main/CCC_analysis).
  
## Contributors

- [Daria Lazic](https://github.com/LazDaria)
- [Sara Wernig-Zorc](https://github.com/sarawernig)
- [Simon Gutwein](https://github.com/SimonBon/)

## Funding

This work was funded by the Austrian Science Fund (FWF#I4162 and FWF#35841), the Vienna Science and Technology Fund (WWTF; LS18-111), the Swiss Government Excellence Scholarship and the St. Anna Kinderkrebsforschung e.V.

