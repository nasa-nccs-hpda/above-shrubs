# above-shrubs

Development of segmentation and regression deep learning models for ABoVE Shrubs project.

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

This project has two containers. The Development container has all of the Python dependencies, with the exception
of the PIP installable modules of this repository. You will need to export the path to this repository in order
to pull the latest changes.

The second container is the production container which containers the latest stable release with a PIP installable
module of this package. No need to export any paths to the PYTHONPATH environment variable.

If you are working on the NCCS Explore system, we have built a default container under:
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

## 2. CHM Regression

Run the pipeline based on the specific steps you might need. For example:

### 2.1. Setup

Converts data into numpy arrays for the dataloader to process.

```bash
singularity exec --env PYTHONPATH="$NOBACKUP/development/above-shrubs" --nv -B $NOBACKUP,/lscratch,/explore/nobackup/people,/explore/nobackup/projects /lscratch/$USER/container/above-shrubs python $NOBACKUP/development/above-shrubs/above_shrubs/view/chm_pipeline_cnn.py -c /explore/nobackup/people/jacaraba/development/above-shrubs/projects/chm_cnn/configs/above_shrubs_cnn_v1.yaml -s setup
```

### 2.2. Preprocessing

Modifications to tiles and rebinning of heights.

```bash
singularity exec --env PYTHONPATH="$NOBACKUP/development/above-shrubs" --nv -B $NOBACKUP,/lscratch,/explore/nobackup/people,/explore/nobackup/projects /lscratch/$USER/container/above-shrubs python $NOBACKUP/development/above-shrubs/above_shrubs/view/chm_pipeline_cnn.py -c /explore/nobackup/people/jacaraba/development/above-shrubs/projects/chm_cnn/configs/above_shrubs_cnn_v1.yaml -s preprocess
```

### 2.3. Training

Training of the deep learning model.

```bash
singularity exec --env PYTHONPATH="$NOBACKUP/development/above-shrubs" --nv -B $NOBACKUP,/lscratch,/explore/nobackup/people,/explore/nobackup/projects /lscratch/$USER/container/above-shrubs python $NOBACKUP/development/above-shrubs/above_shrubs/view/chm_pipeline_cnn.py -c /explore/nobackup/people/jacaraba/development/above-shrubs/projects/chm_cnn/configs/above_shrubs_cnn_v1.yaml -s train
```

To turn this into a full slurm submission command:

```bash
sbatch --mem-per-cpu=10240 -G4 -c40 -t05-00:00:00 -J cunet-v1.1.1 --wrap="singularity exec --env PYTHONPATH=/explore/nobackup/people/jacaraba/development/above-shrubs:/explore/nobackup/people/jacaraba/development/tensorflow-caney --nv -B $NOBACKUP,/explore/nobackup/people,/explore/nobackup/projects /explore/nobackup/projects/ilab/containers/above-shrubs.2023.07 python /explore/nobackup/people/jacaraba/development/above-shrubs/above_shrubs/view/chm_pipeline_cnn.py -c /explore/nobackup/projects/above/misc/ABoVE_Shrubs/development/configs/above_shrubs_chm_custom_unet.yaml -s train"
```

### 2.4. Inference

Inference as part of a single system:

```bash
singularity exec --env PYTHONPATH="$NOBACKUP/development/above-shrubs" --nv -B $NOBACKUP,/lscratch,/explore/nobackup/people,/explore/nobackup/projects /explore/nobackup/projects/ilab/containers/above-shrubs.2023.07 python $NOBACKUP/development/above-shrubs/above_shrubs/view/chm_pipeline_cnn.py -c /explore/nobackup/people/jacaraba/development/above-shrubs/projects/chm_cnn/configs/above_shrubs_cnn_v1.yaml -s predict
```

Parallel inference across the cluster using Slurm:

```bash
for i in {1..20}; do sbatch scripts/slurm_predict.sh; done
```

### 2.5. Validate

```bash
singularity exec --env PYTHONPATH="$NOBACKUP/development/above-shrubs:$NOBACKUP/development/tensorflow-caney" --nv -B /explore/nobackup/projects,$NOBACKUP,/lscratch,/explore/nobackup/people /lscratch/$USER/container/tensorflow-caney python $NOBACKUP/development/above-shrubs/above_shrubs/view/chm_pipeline_cnn.py -c /explore/nobackup/people/jacaraba/development/above-shrubs/projects/chm_cnn/configs/above_shrubs_cnn_v1.yaml -s validate
```

### 2.6. All Steps

```bash
singularity exec --env PYTHONPATH="$NOBACKUP/development/above-shrubs" --nv -B $NOBACKUP,/lscratch,/explore/nobackup/people,/explore/nobackup/projects /lscratch/$USER/container/above-shrubs python $NOBACKUP/development/above-shrubs/above_shrubs/view/chm_pipeline_cnn.py -c /explore/nobackup/people/jacaraba/development/above-shrubs/projects/chm_cnn/configs/above_shrubs_cnn_v1.yaml -s setup preprocess train predict validate
```

## 3. Land Cover Segmentation

Run the pipeline based on the specific steps you might need. For example:

### 3.1. Preprocessing

Generate training and validation tiles.

```bash
singularity exec --env PYTHONPATH="$NOBACKUP/development/above-shrubs" --nv -B $NOBACKUP,/lscratch,/explore/nobackup/people,/explore/nobackup/projects /lscratch/$USER/container/above-shrubs python $NOBACKUP/development/above-shrubs/above_shrubs/view/landcover_pipeline_cnn.py -c $NOBACKUP/development/above-shrubs/projects/landcover_cnn/above_shrubs_cnn_v1.yaml -d $NOBACKUP/development/above-shrubs/projects/landcover_cnn/above_shrubs_cnn_v1.csv -s preprocess
```

Adding CHM to the input bands:

```bash
singularity exec --env PYTHONPATH="$NOBACKUP/development/above-shrubs:$NOBACKUP/development/tensorflow-caney" --nv -B $NOBACKUP,/lscratch,/explore/nobackup/people,/explore/nobackup/projects /lscratch/$USER/container/above-shrubs python $NOBACKUP/development/above-shrubs/above_shrubs/view/landcover_pipeline_cnn.py -c $NOBACKUP/development/above-shrubs/projects/landcover_cnn/above_shrubs_cnn_v1.1_chm.yaml -d $NOBACKUP/development/above-shrubs/projects/landcover_cnn/above_shrubs_cnn_v1.1_chm.csv -s preprocess
```

### 3.2. Train

Training of the model.

```bash
singularity exec --env PYTHONPATH="$NOBACKUP/development/above-shrubs" --nv -B $NOBACKUP,/lscratch,/explore/nobackup/people,/explore/nobackup/projects /lscratch/$USER/container/above-shrubs python $NOBACKUP/development/above-shrubs/above_shrubs/view/landcover_pipeline_cnn.py -c $NOBACKUP/development/above-shrubs/projects/landcover_cnn/above_shrubs_cnn_v1.yaml -d $NOBACKUP/development/above-shrubs/projects/landcover_cnn/above_shrubs_cnn_v1.csv -s train
```

### 3.2. Prediction

```bash
singularity exec --env PYTHONPATH="$NOBACKUP/development/above-shrubs" --nv -B $NOBACKUP,/lscratch,/explore/nobackup/people,/explore/nobackup/projects /lscratch/$USER/container/above-shrubs python $NOBACKUP/development/above-shrubs/above_shrubs/view/landcover_pipeline_cnn.py -c $NOBACKUP/development/above-shrubs/projects/landcover_cnn/above_shrubs_cnn_v1.yaml -d $NOBACKUP/development/above-shrubs/projects/landcover_cnn/above_shrubs_cnn_v1.csv -s predict
```

### 2.6. All Steps

```bash
singularity exec --env PYTHONPATH="$NOBACKUP/development/above-shrubs" --nv -B $NOBACKUP,/lscratch,/explore/nobackup/people,/explore/nobackup/projects /lscratch/$USER/container/above-shrubs python $NOBACKUP/development/above-shrubs/above_shrubs/view/landcover_pipeline_cnn.py -c $NOBACKUP/development/above-shrubs/projects/landcover_cnn/above_shrubs_cnn_v1.yaml -d $NOBACKUP/development/above-shrubs/projects/landcover_cnn/above_shrubs_cnn_v1.csv -s preprocess train predict
```

## Authors

- Jordan Alexis Caraballo-Vega, jordan.a.caraballo-vega@nasa.gov
- Matthew Macander, mmacander@abrinc.com
- Paul M. Montesano, paul.m.montesano@nasa.gov
