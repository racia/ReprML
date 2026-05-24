#!/bin/bash
#SBATCH --job-name=reprML
#SBATCH --output=reprML.out
#SBATCH --error=reprML.err
#SBATCH --time=02:29:00
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

python trainer.py --dataset_name squad --task_name plain_text --model_name roberta-base --num_seeds 2

