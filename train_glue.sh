#!/bin/bash
#SBATCH --job-name=reprML
#SBATCH --output=reprML-glue-%j.out
#SBATCH --error=reprML-glue-%j.err
#SBATCH --time=03:29:00 # Approx. 3 1/2 hrs
# SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=all
#SBATCH --mail-user=sari@cl.uni-heidelberg.de

# Save input variables

DETERM=""
ENV="repr"

# Activate the conda environment
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate $ENV
source /home/hd/hd_hd/hd_ea226/research-project/.env/bin/activate

# if (! -z $DETERM); then
#     # Enable deterministic Cuda BLAS operations
#     echo "Enabling deterministic CuBLAS"
#     export CUBLAS_WORKSPACE_CONFIG=:4096:8
#     # export CUBLAS_WORKSPACE_CONFIG=:16:8
# fi

# python - <<'EOF'
# import os
# print(os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
# EOF

# Monitor GPU usage in background
(
    while true; do
        echo "== GPU Status: $(date) =="
        nvidia-smi --query-gpu=index,utilization.memory,utilization.gpu --format=csv
        sleep 30
    done
) > gpu_monitor.log &
MONITOR_PID=$!#

# Define script and config variables
declare -a CONFIGS=("$PWD/config/train-glue"${DETERM}".yaml")
SCRIPT="trainer.py"
echo "Running $script with the following configurations: ${CONFIGS[*]}"

python "$SCRIPT" --config "${CONFIGS[@]}"