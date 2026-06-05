#!/bin/bash
#SBATCH --job-name=reprML
#SBATCH --output=reprML-squad.out
#SBATCH --error=reprML-squad.err
#SBATCH --time=00:29:00
# SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=all
#SBATCH --mail-user=sari@cl.uni-heidelberg.de

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
declare -a CONFIGS=("$PWD/config/train-squad.yaml")
SCRIPT="trainer.py"
echo "Running $script with the following configurations: ${CONFIGS[*]}"

python "$SCRIPT" --config "${CONFIGS[@]}"

