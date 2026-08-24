#!/bin/bash
# Run from the directory containing all three files:
#   sbatch --nodes=1 run_ray_epydemix.sbatch
#   sbatch --nodes=2 run_ray_epydemix.sbatch

#SBATCH --account=training2637
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --time=00:30:00
#SBATCH --partition=develgpus
# Epydemix uses CPUs. Only the final sbi training needs one GPU; adjust this
# directive to the syntax and minimum GPU allocation of your partition.
#SBATCH --gres=gpu:1
#SBATCH --output=ray-epydemix-%j.out
#SBATCH --error=ray-epydemix-%j.err

set -euo pipefail

# Activate the same environment on every node before starting Ray.
 source /p/project1/training2637/.env/activate.sh
# Or, if using the original repository setup:
# source ../sbi_env/activate.sh

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

# ParaStation does not necessarily perform PATH lookup for an srun executable.
# Pass the absolute Python path to srun and invoke Ray's documented console
# entry point as a module function; this does not require a venv/bin/ray file.
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
RAY_CLI_CODE='from ray.scripts.scripts import main; main()'
if ! "${PYTHON_BIN}" -c 'import ray, ray.scripts.scripts' >/dev/null 2>&1; then
    echo "ERROR: Python cannot import Ray: ${PYTHON_BIN}" >&2
    echo "Activate the intended venv in this batch script, then install Ray with:" >&2
    echo "  ${PYTHON_BIN} -m pip install 'ray==2.56.1'" >&2
    exit 2
fi

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-${PWD}}"
PYTHON_SCRIPT="${PYTHON_SCRIPT:-${SUBMIT_DIR}/epydemix_ray_sbi.py}"
PATCH_SCRIPT="${PATCH_SCRIPT:-${SUBMIT_DIR}/sbi_ray_patch.py}"
NUM_SIMULATIONS="${NUM_SIMULATIONS:-50000}"
SIMULATION_BATCH_SIZE="${SIMULATION_BATCH_SIZE:-}"
SEED="${SEED:-2026}"
TRAIN_MODE="${TRAIN_MODE:-both}"
POSTERIOR_SAMPLES="${POSTERIOR_SAMPLES:-10000}"
SKIP_TRAINING="${SKIP_TRAINING:-0}"
RUN_DIR="${RUN_DIR:-${SUBMIT_DIR}/results/ray_nodes${SLURM_NNODES}_job${SLURM_JOB_ID}}"

for required_file in "${PYTHON_SCRIPT}" "${PATCH_SCRIPT}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "ERROR: required file not found: ${required_file}" >&2
        echo "Place all three files in the directory where you run sbatch." >&2
        exit 2
    fi
done
mkdir -p "${RUN_DIR}"

mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
head_node="${nodes[0]}"
read -ra head_addresses <<< "$(srun --nodes=1 --ntasks=1 -w "${head_node}" hostname --ip-address)"
head_node_ip="${head_addresses[0]}"
for address in "${head_addresses[@]}"; do
    if [[ "${address}" == *.* ]]; then
        head_node_ip="${address}"
        break
    fi
done
# Ray reserves 10002-19999 for worker processes by default. Keep the head/GCS
# port in a disjoint range; the job-derived value reduces collisions when
# several users run Ray allocations concurrently.
ray_port="${RAY_PORT:-$((20000 + SLURM_JOB_ID % 10000))}"
export RAY_ADDRESS="${head_node_ip}:${ray_port}"

echo "Job:                 ${SLURM_JOB_ID}"
echo "Allocated nodes:     ${nodes[*]}"
echo "Ray head:            ${head_node} (${RAY_ADDRESS})"
echo "CPUs per Ray node:   ${SLURM_CPUS_PER_TASK}"
echo "Simulations:         ${NUM_SIMULATIONS}"
echo "Simulation batch:    ${SIMULATION_BATCH_SIZE:-automatic}"
echo "Output directory:    ${RUN_DIR}"
echo "Python:              ${PYTHON_BIN}"
echo "Ray CLI:             ${PYTHON_BIN} (ray.scripts.scripts)"
"${PYTHON_BIN}" - <<'PY'
import epydemix, ray, sbi, torch
print("epydemix:", getattr(epydemix, "__version__", "unknown"))
print("ray:", ray.__version__)
print("sbi:", getattr(sbi, "__version__", "unknown"))
print("torch:", torch.__version__)
print("CUDA available to driver:", torch.cuda.is_available())
PY

ray_step_pids=()
cleanup() {
    set +e
    for pid in "${ray_step_pids[@]}"; do
        kill "${pid}" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup EXIT

echo "Starting Ray head on ${head_node}"
srun --nodes=1 --ntasks=1 -w "${head_node}" \
    "${PYTHON_BIN}" -c "${RAY_CLI_CODE}" start --head \
        --node-ip-address="${head_node_ip}" \
        --port="${ray_port}" \
        --num-cpus="${SLURM_CPUS_PER_TASK}" \
        --num-gpus=0 \
        --include-dashboard=false \
        --block &
ray_step_pids+=("$!")

head_ready=0
for _ in $(seq 1 30); do
    if "${PYTHON_BIN}" -c "${RAY_CLI_CODE}" \
        status --address="${RAY_ADDRESS}" >/dev/null 2>&1; then
        head_ready=1
        break
    fi
    sleep 2
done
if [[ "${head_ready}" -ne 1 ]]; then
    echo "ERROR: Ray head did not become ready at ${RAY_ADDRESS}" >&2
    exit 3
fi

for ((i = 1; i < ${#nodes[@]}; i++)); do
    worker_node="${nodes[$i]}"
    echo "Starting Ray worker on ${worker_node}"
    srun --nodes=1 --ntasks=1 -w "${worker_node}" \
        "${PYTHON_BIN}" -c "${RAY_CLI_CODE}" \
        start --address="${RAY_ADDRESS}" \
            --num-cpus="${SLURM_CPUS_PER_TASK}" \
            --num-gpus=0 \
            --block &
    ray_step_pids+=("$!")
done

driver_args=(
    --num-simulations "${NUM_SIMULATIONS}"
    --output-dir "${RUN_DIR}"
    --seed "${SEED}"
    --expected-ray-nodes "${SLURM_NNODES}"
    --train-mode "${TRAIN_MODE}"
    --posterior-samples "${POSTERIOR_SAMPLES}"
)
if [[ -n "${SIMULATION_BATCH_SIZE}" ]]; then
    driver_args+=(--simulation-batch-size "${SIMULATION_BATCH_SIZE}")
fi
if [[ "${SKIP_TRAINING}" == "1" ]]; then
    driver_args+=(--skip-training)
fi

# A Slurm batch script runs on the first allocated node, which is also the Ray
# head selected above. The driver runs once; Ray schedules simulation batches
# across the head and every joined worker node.
cd "${SUBMIT_DIR}"
"${PYTHON_BIN}" -u "${PYTHON_SCRIPT}" "${driver_args[@]}"

echo "Ray task execution audit:"
"${PYTHON_BIN}" - "${RUN_DIR}/run_summary.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))
print(json.dumps(summary["simulation_worker_processes_by_node"], indent=2))
PY
echo "Finished. Compare simulation_elapsed_seconds in the one- and two-node summaries."

