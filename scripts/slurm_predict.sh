#!/bin/bash
#SBATCH --job-name "above-chm"
#SBATCH --time=05-00:00:0
#SBATCH -G1
#SBATCH -c10
#SBATCH --mem-per-cpu=10240
#SBATCH --export=ALL
#SBATCH --mail-user=jordan.a.caraballo-vega@nasa.gov
#SBATCH --mail-type=ALL

# You can call this file with the following command
# for i in {1..20}; do sbatch slurm_predict.sh; done

# Environment variables
CONTAINER_PATH="REPLACE_WITH_CONTAINER_PATH"
CONFIG_PATH="/explore/nobackup/people/jacaraba/development/above-shrubs/projects/chm_cnn/configs/above_shrubs_cnn_v1.yaml"

# 1. Load singularity module
module load singularity

# 2. Execute software, one of setup, preprocess, train, predict, validate
srun -n 1 singularity exec --env PYTHONPATH="$NOBACKUP/development/above-shrubs" --nv \
     -B $NOBACKUP,/lscratch,/explore/nobackup/people /lscratch/$USER/container/above-shrubs \
     python $NOBACKUP/development/above-shrubs/above_shrubs/view/chm_pipeline_cnn.py \
     -c $CONFIG_PATH \
     -s predict

