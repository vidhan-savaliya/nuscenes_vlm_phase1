#!/bin/bash
# =============================================================================
# E2C-VLM Phase 1: Causality-Aware Model Compression (Stage 1)
# Paper: Development of Edge Intelligence-based Energy-Efficient Causal VLM
#        for Robust Autonomous Systems
# Authors: Ayanabha Ghosh, Vidhan Pravinbhai Savaliya, Debasis Das (IIT Jodhpur)
# =============================================================================
#SBATCH --job-name=e2c_vlm_phase1
#SBATCH --partition=mtech
#SBATCH --qos=mtech
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --exclude=cn07
#SBATCH --output=outputs/slurm-e2c-%j.out
#SBATCH --error=outputs/slurm-e2c-%j.err

set -e

echo "========================================================================"
echo "E2C-VLM Stage 1 Experiment started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "========================================================================"

mkdir -p outputs/manifests outputs/predictions outputs/metrics outputs/reports

# Activate Conda environment
source /csehome/m25csa031/miniconda3/etc/profile.d/conda.sh
conda activate vit

PYTHON_BIN="/csehome/m25csa031/miniconda3/envs/vit/bin/python"

cd /csehome/m25csa031/nuscenes_vlm_phase1

export HF_HOME=/scratch/m25csa031/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

# Display GPU status
nvidia-smi

CONFIG="configs/config.yaml"
VERSION="v1.0-mini"
MODEL="qwen2_5_vl_3b"

echo "------------------------------------------------------------------------"
echo "Step 1: Baseline Inference (Unpruned VLM, rho = 0.0)"
echo "------------------------------------------------------------------------"
$PYTHON_BIN src/infer.py \
    --config $CONFIG \
    --version $VERSION \
    --model $MODEL \
    --no-causal-pruning

$PYTHON_BIN src/evaluation.py \
    --config $CONFIG \
    --version $VERSION \
    --predictions outputs/predictions/predictions_${VERSION}_${MODEL}.jsonl

echo "------------------------------------------------------------------------"
echo "Step 2: Heuristic Pruning Baseline (FastV Attention, rho = 0.50)"
echo "------------------------------------------------------------------------"
$PYTHON_BIN src/infer.py \
    --config $CONFIG \
    --version $VERSION \
    --model $MODEL \
    --causal-pruning \
    --pruning-method fastv_attention \
    --pruning-ratio 0.50

$PYTHON_BIN src/evaluation.py \
    --config $CONFIG \
    --version $VERSION \
    --predictions outputs/predictions/predictions_${VERSION}_${MODEL}_fastv_attention_p50.jsonl

echo "------------------------------------------------------------------------"
echo "Step 3: Proposed Model (E2C-VLM Causal Token Pruning, rho = 0.50, Eq. 1)"
echo "------------------------------------------------------------------------"
$PYTHON_BIN src/infer.py \
    --config $CONFIG \
    --version $VERSION \
    --model $MODEL \
    --causal-pruning \
    --pruning-method causal \
    --pruning-ratio 0.50 \
    --intervention-mode group \
    --group-size 16

$PYTHON_BIN src/evaluation.py \
    --config $CONFIG \
    --version $VERSION \
    --predictions outputs/predictions/predictions_${VERSION}_${MODEL}_causal_p50.jsonl

echo "------------------------------------------------------------------------"
echo "Step 4: High-Compression Study (E2C-VLM Causal Token Pruning, rho = 0.70)"
echo "------------------------------------------------------------------------"
$PYTHON_BIN src/infer.py \
    --config $CONFIG \
    --version $VERSION \
    --model $MODEL \
    --causal-pruning \
    --pruning-method causal \
    --pruning-ratio 0.70 \
    --intervention-mode group \
    --group-size 16

$PYTHON_BIN src/evaluation.py \
    --config $CONFIG \
    --version $VERSION \
    --predictions outputs/predictions/predictions_${VERSION}_${MODEL}_causal_p70.jsonl

echo "------------------------------------------------------------------------"
echo "Step 5: Comparative Evaluation Benchmark Report"
echo "------------------------------------------------------------------------"
$PYTHON_BIN src/evaluation.py \
    --compare \
    outputs/metrics/metrics_predictions_${VERSION}_${MODEL}.json \
    outputs/metrics/metrics_predictions_${VERSION}_${MODEL}_fastv_attention_p50.json \
    outputs/metrics/metrics_predictions_${VERSION}_${MODEL}_causal_p50.json \
    outputs/metrics/metrics_predictions_${VERSION}_${MODEL}_causal_p70.json

echo "========================================================================"
echo "E2C-VLM Phase 1 execution completed successfully at $(date)"
echo "========================================================================"
