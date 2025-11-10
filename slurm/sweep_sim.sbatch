#!/bin/bash
#SBATCH --job-name=sim_sweep
#SBATCH --partition=single
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0:30:00
#SBATCH --array=0-0%1           # real range set by submit_sweep_sim
#SBATCH --export=NONE
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

# Project root, because submit_sweep_sim called sbatch from there
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
module purge
module load devel/miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate seqdecon
export PYTHONUNBUFFERED=1
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"
export TMPDIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"

python simulations/sweep_simulations.py --task "${SLURM_ARRAY_TASK_ID}" --defer-index