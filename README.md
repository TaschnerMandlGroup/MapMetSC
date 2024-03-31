<img src="https://github.com/TaschnerMandlGroup/MapMetSC/blob/main/docs/img/logo.png" align="right" alt="Logo" width="55" />

# MapMetSC
[comment]: <> (repo-specific shields will work once the repo is online)
![Suggestions Welcome](https://img.shields.io/badge/suggestions-welcome-green)

This repository provides scripts to reproduce figures published in [Lazic et al.](coming/soon). In brief, we spatially and temporally mapped primary and metastatic neuroblastoma by multi-modal high-plex imaging and combination with scRNA-seq datasets. The data analysis worflow in Lazic et al. is split into a python-based image-processing pipeline [MapMetIP](https://github.com/TaschnerMandlGroup/MapMetIP) and an R-based single-cell analysis worfklow MapMetSC. 

## Data 
To use `MapMetSC`, start with [MapMetIP](https://github.com/TaschnerMandlGroup/MapMetIP) to download files, process images and extract single-cell data.

## Usage
  
To be able to run the provided Rmd scripts, clone the repository via
 ```bash
 git clone https://github.com/TaschnerMandlGroup/MapMetSC.git
 ```
For reproducibility, we provide a docker image. Pull the docker image using:
 ```bash
 docker pull lazdaria/mapmetsc
 ```
 and run the container:
 ```bash
docker run -p 8787:8787 -e PASSWORD=mapmetsc -v <path/to/MapMetSC:/home/rstudio/MapMetSC -v <path/to/singlecelldata>:/mnt/data
 ```
 An RStudio server session can then be accessed via your browser at `localhost:8787` with the `username: rstudio` and `password: mapmetsc`.
  
## Contributors

- [Daria Lazic](https://github.com/LazDaria)
- [Sara Wernig-Zorc](https://github.com/sarawernig)
- [Simon Gutwein](https://github.com/SimonBon/)

## Funding

This work was funded by the Austrian Science Fund (FWF#I4162 and FWF#35841), the Vienna Science and Technology Fund (WWTF; LS18-111), the Swiss Government Excellence Scholarship and the St. Anna Kinderkrebsforschung e.V.

