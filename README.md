# Alaska’s boreal-tundra resolved with vegetation height from commercial imagery and a foundation model

A reconfiguration of vegetation structure patterns in the boreal forest-tundra ecotone and Low Arctic tundra is underway, driven by shrubification, tree expansion, permafrost thaw, and disturbance regimes linked to a rapid warming. The short-statured vegetation in these regions presents challenges for resolving the heterogeneity of plant functional types, even when very high resolution spaceborne imagery is used with earth observation artificial intelligence (AIEO) models. In this study, we resolve height variation of high northern latitude plant functional types by fine-tuning an AIEO model built for global applications using a large variety of domain-specific airborne training and validation data, and field observations. 

Funded by the NASA Terrestrial Ecology Program as part of the [NASA Arctic/Boreal Vulnerability Experiment](https://above.nasa.gov/). 

The portion of the project supported by this repository includes notebooks for anaysis code for the development of segmentation and regression deep learning models applied to very-high-resolution (VHR) spaceborne imagery. These models are run in `singularity`, an open source container platform that ensures the portability and reproducability of our workflow to map Arctic/Boreal land cover and canopy height from VHR imagery.

[![DOI](https://zenodo.org/badge/627911660.svg)](https://zenodo.org/badge/latestdoi/627911660)
![CI Workflow](https://github.com/nasa-nccs-hpda/above-shrubs/actions/workflows/ci.yml/badge.svg)
![CI to DockerHub Dev](https://github.com/nasa-nccs-hpda/above-shrubs/actions/workflows/dockerhub-dev.yml/badge.svg)
![CI to DockerHub Prod](https://github.com/nasa-nccs-hpda/above-shrubs/actions/workflows/dockerhub.yml/badge.svg)
![Code style: PEP8](https://github.com/nasa-nccs-hpda/above-shrubs/actions/workflows/lint.yml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Coverage Status](https://coveralls.io/repos/github/nasa-nccs-hpda/above-shrubs/badge.svg?branch=main)](https://coveralls.io/github/nasa-nccs-hpda/above-shrubs?branch=main)

| Mapped and summarized results | Boreal-tundra focus area with input VHR extents |
| :--------------------: | :--------------------: |
| <img width="450" height="450" alt="FIGURE_03_map_bar_donut_cavm100km_alaska_chm_002m_height_class_area_with_ci_002m_mos_2018_212_deltayr8_months78_50m_FIG_03" src="https://github.com/user-attachments/assets/bf6b5407-4e82-4ea1-9961-436d71500e9d" /> | <img width="450" height="350" alt="SUPP_FIGURE_01_alaska_map_TTE_dinov3_training_2026-02-09" src="https://github.com/user-attachments/assets/f2cacc00-0544-4ff3-9f6b-58e0ffb234ae" /> |

## Authors
- Paul M. Montesano, paul.m.montesano@nasa.gov
- Melanie J. Frost, melanie.frost@nasa.gov
- Jordan Alexis Caraballo-Vega, jordan.a.caraballo-vega@nasa.gov
- Matthew Macander, mmacander@abrinc.com

