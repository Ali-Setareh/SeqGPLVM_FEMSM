#!/bin/bash
#SBATCH --job-name=rebuild_index
#SBATCH --partition=single          # <- change if needed
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --export=NONE
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module purge
module load devel/miniforge || module load devel/miniconda/3

set +u
export MKL_INTERFACE_LAYER=${MKL_INTERFACE_LAYER-}
export MKL_THREADING_LAYER=${MKL_THREADING_LAYER-}
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate seqdecon               # <- same env
set -u

export PYTHONUNBUFFERED=1
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

python - <<'PY'
from utils.runs import rebuild_index
out = rebuild_index(".", dgp="blackwell_yamauchi")
print(f"Index rebuilt at: {out}")
PY
