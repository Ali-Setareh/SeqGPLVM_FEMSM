#!/bin/bash
#SBATCH --job-name=seqdecon_array
#SBATCH --partition=gpu-single
#SBATCH --gres=gpu:A100:1          # or A40:1 / H200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --array=1-24                  # set to 1-48 when you’re ready
#SBATCH --export=NONE              # don't inherit a messy login env
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Clean env, load conda, activate env INSIDE the job
module purge
module load devel/miniforge || module load devel/miniconda/3

# --- make conda activation nounset-safe ---
set +u
# define MKL vars so the activate.d hook can't error under -u
export MKL_INTERFACE_LAYER=${MKL_INTERFACE_LAYER-}
export MKL_THREADING_LAYER=${MKL_THREADING_LAYER-}
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate seqdecon
set -u
# ------------------------------------------

# (optional) persistent place for results + progress heartbeats
export FINAL_ROOT="$SLURM_SUBMIT_DIR/results"
mkdir -p "$FINAL_ROOT/progress"

# make Python output stream line-by-line
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1

# Run the sweep (1 task = first combo). Use stdbuf for live-ish logs.
stdbuf -oL -eL python experiments/sweep_training_seqgplvm.py
