#!/bin/bash
#SBATCH --job-name=sim_sweep
#SBATCH --partition=single          # <- change if your CPU partition is named differently
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --array=0-0%1               # <- wrapper will override this with the real range
#SBATCH --export=NONE               # don't inherit a messy login env
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Clean env, load conda, activate env INSIDE the job (same pattern you use)
module purge
module load devel/miniforge || module load devel/miniconda/3

# --- make conda activation nounset-safe (matches your template) ---
set +u
export MKL_INTERFACE_LAYER=${MKL_INTERFACE_LAYER-}
export MKL_THREADING_LAYER=${MKL_THREADING_LAYER-}
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate seqdecon                 # <- your working env name
set -u
# ------------------------------------------------------------------

# Thread + output hygiene
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

# One config per array task; index updates deferred to a post job
python sweep_simulations.py --task "${SLURM_ARRAY_TASK_ID}" --defer-index
