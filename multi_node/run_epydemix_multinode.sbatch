#!/bin/bash
# Submit unchanged for two nodes, or override the node count at submission:
#   sbatch --nodes=1 run_epydemix_multinode.sbatch
#   sbatch --nodes=2 run_epydemix_multinode.sbatch

#SBATCH --account=training2637
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --time=00:30:00
#SBATCH --partition=develgpus
# Adjust/remove the GPU request to match your cluster. Epydemix simulation is
# CPU-only; the final sbi training stage uses one visible GPU when available.
#SBATCH --gres=gpu:1
#SBATCH --output=epydemix-%j.out
#SBATCH --error=epydemix-%j.err

set -euo pipefail

# Activate your environment here if it is not already active at submission.
# Example used by the original sbi_intro setup:
# source ../sbi_env/activate.sh
# Or:

source /p/project1/training2637/.env/activate.sh

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Slurm runs a spooled copy of this batch file, so BASH_SOURCE[0] points into
# /var/spool rather than to the directory containing the submitted script.
# Resolve the companion Python file from the directory where sbatch was run.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-${PWD}}"
PYTHON_SCRIPT="${PYTHON_SCRIPT:-${SUBMIT_DIR}/epydemix_multinode_sbi.py}"
NUM_SIMULATIONS="${NUM_SIMULATIONS:-50000}"
SEED="${SEED:-2026}"
TRAIN_MODE="${TRAIN_MODE:-both}"
POSTERIOR_SAMPLES="${POSTERIOR_SAMPLES:-10000}"
RUN_DIR="${RUN_DIR:-${PWD}/results/nodes${SLURM_NNODES}_job${SLURM_JOB_ID}}"

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
    echo "ERROR: Python program not found: ${PYTHON_SCRIPT}" >&2
    echo "Place epydemix_multinode_sbi.py in the directory from which you run sbatch," >&2
    echo "or submit with an absolute path:" >&2
    echo "  PYTHON_SCRIPT=epydemix_multinode_sbi.py sbatch run_epydemix_multinode.sbatch" >&2
    exit 2
fi

mkdir -p "${RUN_DIR}"

echo "Job:              ${SLURM_JOB_ID}"
echo "Nodes:            ${SLURM_NNODES}"
echo "CPUs per node:    ${SLURM_CPUS_PER_TASK}"
echo "Simulations:      ${NUM_SIMULATIONS}"
echo "Output directory: ${RUN_DIR}"
echo "Python program:   ${PYTHON_SCRIPT}"
echo "Python:           $(command -v python)"
python - <<'PY'
import epydemix, sbi, torch
print("epydemix:", epydemix.__version__)
print("sbi:", getattr(sbi, "__version__", "unknown"))
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
PY

# Prove that the simulation step can start exactly one task on each allocated
# node. Keep this file with the results as an audit trail for the comparison.
srun \
    --nodes="${SLURM_NNODES}" \
    --ntasks="${SLURM_NNODES}" \
    --ntasks-per-node=1 \
    hostname | sort -u | tee "${RUN_DIR}/simulation_nodes.txt"

observed_nodes="$(wc -l < "${RUN_DIR}/simulation_nodes.txt" | tr -d '[:space:]')"
if [[ "${observed_nodes}" -ne "${SLURM_NNODES}" ]]; then
    echo "ERROR: requested ${SLURM_NNODES} nodes but the preflight reached ${observed_nodes}." >&2
    exit 3
fi
echo "Verified ${observed_nodes} distinct simulation nodes."

simulation_started="$(date +%s)"

# One Slurm task per node. Inside each task, the Python program starts a local
# process pool with SLURM_CPUS_PER_TASK workers. No Ray or sbi patch is used.
srun \
    --nodes="${SLURM_NNODES}" \
    --ntasks="${SLURM_NNODES}" \
    --ntasks-per-node=1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    python -u "${PYTHON_SCRIPT}" simulate \
        --num-simulations "${NUM_SIMULATIONS}" \
        --output-dir "${RUN_DIR}" \
        --seed "${SEED}"

simulation_finished="$(date +%s)"
simulation_wall_seconds="$((simulation_finished - simulation_started))"
echo "${simulation_wall_seconds}" > "${RUN_DIR}/slurm_simulation_wall_seconds.txt"
echo "Distributed simulation wall time: ${simulation_wall_seconds}s"

# srun above is a barrier: all node shards are complete before this starts.
# This stage runs once, merges (theta, x), and hands them to unmodified sbi.
srun \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    python -u "${PYTHON_SCRIPT}" train \
        --input-dir "${RUN_DIR}" \
        --train-mode "${TRAIN_MODE}" \
        --posterior-samples "${POSTERIOR_SAMPLES}"

echo "Finished. Compare slurm_simulation_wall_seconds.txt between the one- and two-node runs."

