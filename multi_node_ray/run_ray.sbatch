#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --time=00:40:00
#SBATCH --partition=develgpus
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/%j.out
#SBATCH --error=outputs/%j.err

export NCCL_SOCKET_IFNAME="ib0"
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

source /p/project1/training2506/sbi_env/activate.sh

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export SLURM_GPUS_PER_TASK=4

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

echo $nodes

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

# Initialize head
echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}"  --block &
sleep 40
echo "------------------------------------"

# Initialize all of the workers
worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    #export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}"  --block &
    sleep 20
done

python -u pyross_example_script.py
echo -e "\nFinished"
