#!/bin/bash
#SBATCH --job-name=seqdecon_array
#SBATCH --partition=gpu-single          # confirm with sinfo
#SBATCH --gres=gpu:A100:1                # or A40/H200 if appropriate
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --array=1                     # <-- grid size
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

module purge
module load devel/miniconda/3
conda activate seqdecon

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
mkdir -p logs

python slurm_sweep_training.py
