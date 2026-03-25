# `ABoVE-Shrubs`: modelling vegetation height and plant functional types in northern and western Alaska with commercial VHR imagery
`ABoVE-Shrubs` refers to a project to map and examine change in Arctic shrub structure in western Alaska. 

It is funded by the NASA Terrestrial Ecology Program as part of the [NASA Arctic/Boreal Vulnerability Experiment](https://above.nasa.gov/). The portion of the project supported by this repository involves the development of segmentation and regression deep learning models applied to very-high-resolution (VHR) spaceborne imagery. These models are run in `singularity`, an open source container platform that ensures the portability and reproducability of our workflow to map Arctic/Boreal land cover and canopy height from VHR imagery.

[![DOI](https://zenodo.org/badge/627911660.svg)](https://zenodo.org/badge/latestdoi/627911660)
![CI Workflow](https://github.com/nasa-nccs-hpda/above-shrubs/actions/workflows/ci.yml/badge.svg)
![CI to DockerHub Dev](https://github.com/nasa-nccs-hpda/above-shrubs/actions/workflows/dockerhub-dev.yml/badge.svg)
![CI to DockerHub Prod](https://github.com/nasa-nccs-hpda/above-shrubs/actions/workflows/dockerhub.yml/badge.svg)
![Code style: PEP8](https://github.com/nasa-nccs-hpda/above-shrubs/actions/workflows/lint.yml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Coverage Status](https://coveralls.io/repos/github/nasa-nccs-hpda/above-shrubs/badge.svg?branch=main)](https://coveralls.io/github/nasa-nccs-hpda/above-shrubs?branch=main)


## Getting Started

1. SSH to ADAPT Login

```bash
ssh adaptlogin.nccs.nasa.gov
```

2. SSH to GPU Login

```bash
ssh gpulogin1
```

3. Clone above-shrubs repository

Clone the github 

```bash
git clone https://github.com/nasa-nccs-hpda/above-shrubs
```

## Downloading Singularity Container

This project has two containers: 
  - The `Development` container has all of the Python dependencies, with the exception
of the PIP installable modules of this repository. You will need to export the path to this repository in order
to pull the latest changes.

  - The `Production` container has the latest stable release with a PIP installable
module of this package. No need to export any paths to the PYTHONPATH environment variable.

If you are working on the NASA Goddard NCCS Explore system, we have built a default container under:
/explore/nobackup/projects/ilab/containers/above-shrubs.tif.

### Development container

```bash
module load singularity;
mkdir -p /lscratch/$USER/container
singularity build --sandbox /lscratch/$USER/container/above-shrubs docker://nasanccs/above-shrubs:dev
```

### Production container

```bash
module load singularity;
mkdir -p /lscratch/$USER/container
singularity build --sandbox /lscratch/$USER/container/above-shrubs docker://nasanccs/above-shrubs:latest
```


## Authors

- Melanie J. Frost, melanie.frost@nasa.gov
- Jordan Alexis Caraballo-Vega, jordan.a.caraballo-vega@nasa.gov
- Matthew Macander, mmacander@abrinc.com
- Paul M. Montesano, paul.m.montesano@nasa.gov
