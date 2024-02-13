#!/bin/bash
#SBATCH --job-name "above-landcover"
#SBATCH --time=05-00:00:00
#SBATCH -G 1
#SBATCH -c 10
#SBATCH -q ilab
#SBATCH --mem-per-cpu=18G
#SBATCH --export=ALL
#SBATCH --mail-user=jordan.a.caraballo-vega@nasa.gov
#SBATCH --mail-type=ALL

# You can call this file with the following command
# for i in {1..64}; do sbatch slurm_predict_landcover.sh; done

# Environment variables
CONTAINER_PATH="/explore/nobackup/projects/ilab/containers/vhr-cloudmask.sif"
CONFIG_PATH="/explore/nobackup/people/jacaraba/development/above-shrubs/projects/landcover_cnn/above_shrubs_cnn_v1.yaml"

# 1. Load singularity module
module load singularity

# 2. Execute software, one of setup, preprocess, train, predict, validate
srun -n 1 singularity exec -B /explore/nobackup/projects,$NOBACKUP,/explore/nobackup/people \
    --env PYTHONPATH="/explore/nobackup/people/jacaraba/development/above-shrubs" \
    --nv $CONTAINER_PATH \
    python /explore/nobackup/people/jacaraba/development/above-shrubs/above_shrubs/view/landcover_pipeline_cnn.py \
    -c $CONFIG_PATH \
    -s predict
