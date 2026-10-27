#!/bin/bash
#SBATCH --job-name=nuscenes_vlm_mini
#SBATCH --partition=mtech
#SBATCH --qos=mtech
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --exclude=cn07
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --error=outputs/slurm-%j.err

set -e

echo "=========================================================="
echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "=========================================================="

mkdir -p outputs

# Activate Conda environment
source /csehome/m25csa031/miniconda3/etc/profile.d/conda.sh
conda activate vit

cd /csehome/m25csa031/nuscenes_vlm_phase1

export HF_HOME=/scratch/m25csa031/.cache/huggingface
export PYTHONUNBUFFERED=1

# Display GPU status
nvidia-smi

echo "=== Step 1: Run Inference on v1.0-mini (Qwen2.5-VL-3B) ==="
python src/infer.py \
    --config configs/config.yaml \
    --version v1.0-mini \
    --model qwen2_5_vl_3b

echo "=== Step 2: Run Evaluation on Predictions ==="
python src/evaluation.py \
    --config configs/config.yaml \
    --predictions outputs/predictions/predictions_v1.0-mini_qwen2_5_vl_3b.jsonl

echo "=========================================================="
echo "Job completed successfully at $(date)"
echo "=========================================================="
