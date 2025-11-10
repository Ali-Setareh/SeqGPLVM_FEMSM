#!/bin/bash
#SBATCH --job-name=rebuild_index
#SBATCH --partition=single
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --export=NONE

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load devel/miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate seqdecon
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

python -c "from utils.runs import rebuild_index; rebuild_index()"