#!/bin/bash
#SBATCH --job-name=seqdecon_array
#SBATCH --partition=cpu-single
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --array=1-12                  # set to 1-48 when you’re ready
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
# ensure project root is importable
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"
# run as a module so sys.path[0] is the project root
stdbuf -oL -eL python -m experiments.sweep_pipeline_seqgplvm
