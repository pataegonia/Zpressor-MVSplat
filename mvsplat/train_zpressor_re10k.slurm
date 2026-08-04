#!/usr/bin/env bash
#SBATCH -J mvsplat-zpressor-train
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v8
#SBATCH -t 6-00:00:00
#SBATCH -o slurm-mvsplat-zpressor-%A.out
#SBATCH -e slurm-mvsplat-zpressor-%A.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -n "${REPO_DIR:-}" ]]; then
    REPO_DIR="${REPO_DIR}"
elif [[ -d "${SUBMIT_DIR}/src" && -d "${SUBMIT_DIR}/config" ]]; then
    REPO_DIR="${SUBMIT_DIR}"
else
    REPO_DIR="${SUBMIT_DIR}/mvsplat"
fi
CONDA_ENV_NAME="${CONDA_ENV_NAME:-hisplat}"
DATASET_ROOT="${DATASET_ROOT:-/data3/local_datasets/re10k}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/re10k_zpressor_ctx12_cluster6}"
UNIMATCH_CKPT="${UNIMATCH_CKPT:-checkpoints/gmdepth-scale1-resumeflowthings-scannet-5d9d7964.pth}"

CONTEXT_VIEWS="${CONTEXT_VIEWS:-12}"
TARGET_VIEWS="${TARGET_VIEWS:-8}"
CLUSTER_NUM="${CLUSTER_NUM:-6}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_STEPS="${MAX_STEPS:-100000}"
SAVE_EVERY="${SAVE_EVERY:-20000}"
WANDB_MODE_VALUE="${WANDB_MODE_VALUE:-disabled}"
WANDB_PROJECT="${WANDB_PROJECT:-mvsplat-zpressor}"

# false: standard training, initialized from the official UniMatch backbone.
# true: initialize the backbone randomly as well as the newly added ZPressor.
TRAIN_BACKBONE_FROM_SCRATCH="${TRAIN_BACKBONE_FROM_SCRATCH:-false}"

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
    echo "ERROR: Conda not found."
    exit 1
fi
conda activate "${CONDA_ENV_NAME}"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/../ZPressor${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_MODE="${WANDB_MODE_VALUE}"
export SLURM_JOB_NAME=bash

cd "${REPO_DIR}"

[[ -d "${DATASET_ROOT}" ]] || {
    echo "ERROR: missing dataset: ${DATASET_ROOT}"
    exit 1
}

if (( CLUSTER_NUM > CONTEXT_VIEWS )); then
    echo "ERROR: CLUSTER_NUM (${CLUSTER_NUM}) must be <= CONTEXT_VIEWS (${CONTEXT_VIEWS})."
    exit 1
fi

if [[ "${TRAIN_BACKBONE_FROM_SCRATCH}" == "true" ]]; then
    BACKBONE_ARG="model.encoder.unimatch_weights_path=null"
else
    [[ -f "${UNIMATCH_CKPT}" ]] || {
        echo "ERROR: missing UniMatch checkpoint: ${UNIMATCH_CKPT}"
        exit 1
    }
    BACKBONE_ARG="model.encoder.unimatch_weights_path=${UNIMATCH_CKPT}"
fi

echo "Training MVSplat+ZPressor from a new model"
echo "dataset=${DATASET_ROOT} context=${CONTEXT_VIEWS} target=${TARGET_VIEWS} cluster=${CLUSTER_NUM}"
echo "output=${OUTPUT_DIR} random_backbone=${TRAIN_BACKBONE_FROM_SCRATCH}"

srun python -m src.main +experiment=re10k \
    mode=train \
    "dataset.roots=[${DATASET_ROOT}]" \
    "dataset.view_sampler.num_context_views=${CONTEXT_VIEWS}" \
    "dataset.view_sampler.num_target_views=${TARGET_VIEWS}" \
    "data_loader.train.batch_size=${BATCH_SIZE}" \
    "data_loader.train.num_workers=${NUM_WORKERS}" \
    data_loader.train.persistent_workers=true \
    model.encoder.use_cluster=true \
    "model.encoder.cluster_num=${CLUSTER_NUM}" \
    model.encoder.num_heads=1 \
    model.encoder.num_layers=6 \
    "${BACKBONE_ARG}" \
    checkpointing.load=null \
    checkpointing.resume=false \
    "checkpointing.every_n_train_steps=${SAVE_EVERY}" \
    "trainer.max_steps=${MAX_STEPS}" \
    trainer.num_nodes=1 \
    "wandb.mode=${WANDB_MODE_VALUE}" \
    "wandb.project=${WANDB_PROJECT}" \
    wandb.name=re10k-zpressor \
    "output_dir=${OUTPUT_DIR}"

echo "Training outputs: ${OUTPUT_DIR}"
