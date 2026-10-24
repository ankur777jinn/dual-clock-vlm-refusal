#!/usr/bin/env bash
# ==============================================================================
# IIT Roorkee MFS Lab - vitallab3 Execution Launcher
# Hardware: NVIDIA RTX 6000 Ada Generation (48 GB VRAM)
# Assigned GPU: GPU 2 (CUDA_VISIBLE_DEVICES=2)
# Environment: emr (Python 3.10, PyTorch 2.14+cu130)
# Judge: Local pre-cached Meta-Llama-3.1-8B-Instruct (Zero API keys needed!)
# ==============================================================================

set -euo pipefail

# 1. Hardware & Memory configuration
export CUDA_VISIBLE_DEVICES=2
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.7"

# 2. Local Judge Model (Pre-cached on vitallab3!)
JUDGE_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"

echo "================================================================="
echo " Starting Dual-Clock VLM Refusal Experiment on vitallab3 (GPU 2)"
echo " GPU: RTX 6000 Ada 48GB | Target: Qwen2.5-VL-7B-Instruct (bf16)"
echo " Judge: Local ${JUDGE_MODEL} (No Claude API required!)"
echo "================================================================="

mkdir -p outputs logs

# Step 0: Pre-check & Grid test
echo "--> [Step 0/5] Running grid verification test..."
python -m pytest tests/test_config.py -v

# Step 1: Phase 1 Diagnostics
echo "--> [Step 1/5] Running Phase 1 Diagnostics (L_read & L_harm Clocks)..."
python scripts/run_phase1.py \
    --data-dir data \
    --advbench-path data/advbench_520.txt \
    --alpaca-path data/alpaca_520.txt \
    --output-dir outputs/phase1 \
    --device cuda 2>&1 | tee logs/phase1.log

# Extract changepoints
L_READ=$(python -c "import json; d=json.load(open('outputs/phase1/phase1_results.json')); print(d['L_read'])")
L_HARM=$(python -c "import json; d=json.load(open('outputs/phase1/phase1_results.json')); print(d['L_harm'])")
echo "Phase 1 Complete: L_read = ${L_READ}, L_harm = ${L_HARM}"

# Step 2: Pilot Validation Sweep
echo "--> [Step 2/5] Running Validation Split Sweep (Pilot Variance)..."
python scripts/run_phase2.py \
    --data-dir data \
    --output-dir outputs/validation_sweep \
    --split validation \
    --L-read "${L_READ}" \
    --L-harm "${L_HARM}" \
    --run-nonsafety \
    --device cuda 2>&1 | tee logs/validation.log

# Step 3: Confirmatory Phase 2 Grid on Test Split (with Local Judge)
echo "--> [Step 3/5] Running Confirmatory 55-Cell Grid on Test Split..."
python scripts/run_phase2.py \
    --data-dir data \
    --output-dir outputs/phase2_test \
    --split test \
    --L-read "${L_READ}" \
    --L-harm "${L_HARM}" \
    --run-nonsafety \
    --judge \
    --judge-model "${JUDGE_MODEL}" \
    --device cuda 2>&1 | tee logs/phase2_test.log

# Step 4: Controls C1-C5, Baselines B1-B5 & Capability
echo "--> [Step 4/5] Running Controls & Capability Benchmarks..."
python scripts/run_controls.py \
    --data-dir data \
    --output-dir outputs/controls \
    --src-layer "${L_HARM}" \
    --dst-layer "${L_READ}" \
    --run-capability \
    --device cuda 2>&1 | tee logs/controls.log

# Step 5: Confirmatory Statistical Analysis & Gatekeeping Chain
echo "--> [Step 5/5] Running Mixed-Effects Model & Decision Tree..."
python scripts/run_analysis.py \
    --grid-results outputs/phase2_test/grid_results_test.parquet \
    --phase1-results outputs/phase1/phase1_results.json \
    --controls-results outputs/controls/controls_and_baselines_results.json \
    --output-dir outputs/final_verdict 2>&1 | tee logs/analysis.log

echo "================================================================="
echo " All experiments complete! Final Paper Verdict:"
cat outputs/final_verdict/confirmatory_analysis_summary.json
echo "================================================================="
