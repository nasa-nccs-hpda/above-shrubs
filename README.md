# above-shrubs

Development of regression deep learning models for ABoVE Shrubs project.

## Downloading Singularity Container

### Production container

```bash
module load singularity;
singularity build --sandbox /lscratch/$USER/container/above-shrubs docker://nasanccs/above-shrubs:latest
```

### Development container

1. Dowload base container with dependencies

```bash
module load singularity;
singularity build --sandbox /lscratch/$USER/container/above-shrubs docker://nasanccs/above-shrubs:dev
```

2. Run the pipeline based on the specific you might need. For example:

2.1. Preprocessing

```bash
singularity exec --env PYTHONPATH="$NOBACKUP/development/above-shrubs:$NOBACKUP/development/tensorflow-caney" --nv -B /explore/nobackup/projects/ilab,/explore/nobackup/projects/3sl,$NOBACKUP,/lscratch,/explore/nobackup/people /lscratch/$USER/container/tensorflow-caney python $NOBACKUP/development/above-shrubs/above_shrubs/view/chm_pipeline_cnn.py
```

2.2. Training

```bash
```

2.3. Inference

```bash
```

2.4. Training and Inference

```bash
```

### Using Slurm

TBD

