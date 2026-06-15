#!/bin/bash
#SBATCH --job-name=reprML
#SBATCH --output=reprML-glue-alg-%j.out
#SBATCH --error=reprML-glue-alg-%j.err
#SBATCH --time=03:29:00 # Approx. 3 1/2 hrs
# SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=all
#SBATCH --mail-user=sari@cl.uni-heidelberg.de

# Define (input) variables
if (! -z $1); then # Add input
    NOISE="-alg"
fi

ENV="repr"

# Activate the conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV

# Monitor GPU usage in background
(
    while true; do
        echo "== GPU Status: $(date) =="
        nvidia-smi --query-gpu=index,utilization.memory,utilization.gpu --format=csv
        sleep 30
    done
) > gpu_monitor.log &
MONITOR_PID=$!#

# Set tooling determinism
export CUBLAS_WORKSPACE_CONFIG=:4096:8 # Reproducible matrix operations

# Define script and config variables
declare -a CONFIGS=("$PWD/config/train-glue"${NOISE}".yaml")
SCRIPT="trainer.py"
echo "Running $script with the following configurations: ${CONFIGS[*]}"

python "$SCRIPT" --config "${CONFIGS[@]}"