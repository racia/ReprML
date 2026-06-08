#!/bin/bash
#SBATCH --job-name=reprML
#SBATCH --output=reprML-squad-imp.out
#SBATCH --error=reprML-squad-imp.err
# SBATCH --time=08:29:00 # Approx. 3 1/2 hrs
# SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=all
#SBATCH --mail-user=sari@cl.uni-heidelberg.de

# Save input variables
if (! -z $1); then
    NOISE="-imp"
fi

ENV="repr"

# Activate the conda environment
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate $ENV
source /home/hd/hd_hd/hd_ea226/research-project/.env/bin/activate

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
declare -a CONFIGS=("$PWD/config/train-squad"${NOISE}".yaml")
SCRIPT="trainer.py"
echo "Running $script with the following configurations: ${CONFIGS[*]}"

python "$SCRIPT" --config "${CONFIGS[@]}"