# above-shrubs

Development of regression deep learning models for ABoVE Shrubs project.

## 1. Downloading Singularity Container

### Development container

```bash
module load singularity;
mkdir -p /lscratch/$USER/container
singularity build --sandbox /lscratch/$USER/container/above-shrubs docker://nasanccs/above-shrubs:dev
```

### Production container

```bash
module load singularity;
singularity build --sandbox /lscratch/$USER/container/above-shrubs docker://nasanccs/above-shrubs:latest
```

## 2. Run the pipeline based on the specific you might need. For example:

### 2.1. Setup

```bash
singularity exec --env PYTHONPATH="$NOBACKUP/development/above-shrubs" --nv -B $NOBACKUP,/lscratch,/explore/nobackup/people /lscratch/$USER/container/above-shrubs python $NOBACKUP/development/above-shrubs/above_shrubs/view/chm_pipeline_cnn.py -c /explore/nobackup/people/jacaraba/development/above-shrubs/projects/chm_cnn/configs/above_shrubs_cnn_v1.yaml -s setup
```

### 2.2. Preprocessing

```bash
singularity exec --env PYTHONPATH="$NOBACKUP/development/above-shrubs" --nv -B $NOBACKUP,/lscratch,/explore/nobackup/people /lscratch/$USER/container/above-shrubs python $NOBACKUP/development/above-shrubs/above_shrubs/view/chm_pipeline_cnn.py -c /explore/nobackup/people/jacaraba/development/above-shrubs/projects/chm_cnn/configs/above_shrubs_cnn_v1.yaml -s preprocess
```

### 2.3. Training

```bash
singularity exec --env PYTHONPATH="$NOBACKUP/development/above-shrubs" --nv -B $NOBACKUP,/lscratch,/explore/nobackup/people /lscratch/$USER/container/above-shrubs python $NOBACKUP/development/above-shrubs/above_shrubs/view/chm_pipeline_cnn.py -c /explore/nobackup/people/jacaraba/development/above-shrubs/projects/chm_cnn/configs/above_shrubs_cnn_v1.yaml -s train
```

### 2.4. Inference

```bash
singularity exec --env PYTHONPATH="$NOBACKUP/development/above-shrubs" --nv -B $NOBACKUP,/lscratch,/explore/nobackup/people /lscratch/$USER/container/above-shrubs python $NOBACKUP/development/above-shrubs/above_shrubs/view/chm_pipeline_cnn.py -c /explore/nobackup/people/jacaraba/development/above-shrubs/projects/chm_cnn/configs/above_shrubs_cnn_v1.yaml -s predict
```

### 2.5. Validate

```bash
singularity exec --env PYTHONPATH="$NOBACKUP/development/above-shrubs:$NOBACKUP/development/tensorflow-caney" --nv -B /explore/nobackup/projects/ilab,/explore/nobackup/projects/3sl,$NOBACKUP,/lscratch,/explore/nobackup/people /lscratch/$USER/container/tensorflow-caney python $NOBACKUP/development/above-shrubs/above_shrubs/view/chm_pipeline_cnn.py -c /explore/nobackup/people/jacaraba/development/above-shrubs/projects/chm_cnn/configs/above_shrubs_cnn_v1.yaml -s validate
```

### 2.6. All Steps

```bash
singularity exec --env PYTHONPATH="$NOBACKUP/development/above-shrubs" --nv -B $NOBACKUP,/lscratch,/explore/nobackup/people /lscratch/$USER/container/above-shrubs python $NOBACKUP/development/above-shrubs/above_shrubs/view/chm_pipeline_cnn.py -c /explore/nobackup/people/jacaraba/development/above-shrubs/projects/chm_cnn/configs/above_shrubs_cnn_v1.yaml -s setup preprocess train predict validate
```

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
