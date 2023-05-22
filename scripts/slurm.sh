#!/bin/bash
#SBATCH --job-name "ʕ•ᴥ•ʔ"
#SBATCH --time=05-00:00:0
#SBATCH -N 1
#SBATCH --mail-user=jordan.a.caraballo-vega@nasa.gov
#SBATCH --mail-type=ALL

# Environment variables
CONTAINER_PATH="REPLACE_WITH_CONTAINER_PATH"
CONFIG_PATH="/explore/nobackup/people/jacaraba/development/above-shrubs/projects/chm_cnn/configs/above_shrubs_cnn_v1.yaml"

# 1. Load singularity module
module load singularity

# 2. Execute software, one of setup, preprocess, train, predict, validate
srun -n 1 singularity exec --env PYTHONPATH="$NOBACKUP/development/above-shrubs" --nv \
     -B $NOBACKUP,/lscratch,/explore/nobackup/people /lscratch/$USER/container/above-shrubs \
     python $NOBACKUP/development/above-shrubs/above_shrubs/view/chm_pipeline_cnn.py \
     -c /explore/nobackup/people/jacaraba/development/above-shrubs/projects/chm_cnn/configs/above_shrubs_cnn_v1.yaml \
     -s preprocess