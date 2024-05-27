#!/bin/bash
#SBATCH --account=training2405
#SBATCH --nodes=2
##SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=0:30:00
#SBATCH --partition=gpus
#SBATCH --output=outputs/%j.out
#SBATCH --error=outputs/%j.err

set -x
#export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_SOCKET_IFNAME="ib0"
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
# # so processes know who to talk to
# MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
# # Allow communication over InfiniBand cells.
# MASTER_ADDR="${MASTER_ADDR}i"
# # Get IP for hostname.
# export MASTER_ADDR="$(nslookup "$MASTER_ADDR" | grep -oP '(?<=Address: ).*')"
# export MASTER_PORT=7010
# export GPUS_PER_NODE=4

#export SLURM_GPUS_PER_TASK=4

#source /p/project/atmlaml/bazarova1/memiliflow/sc_venv_template/activate.sh
#source /p/project/loki/bazarova1/sc_venv_memilio/activate.sh
source /p/project/training2405/sc_venv_sbi/activate.sh

# __doc_head_address_start__
#export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
# Getting the node names
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
# __doc_head_address_end__

# __doc_head_ray_start__
#port=7010
port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}"  --block &
# __doc_head_ray_end__

# __doc_worker_ray_start__
# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

echo "------------------------------------"

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

#export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

for ((i = 1; i <= worker_num; i++)); do
    #export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}"  --block &
    sleep 5
done
# __doc_worker_ray_end__

# __doc_script_start__
# ray/doc/source/cluster/doc_code/simple-trainer.py
#python -u simple-trainer.py "$SLURM_CPUS_PER_TASK"
#export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

#srun --overlap 
#python -u run_sbi_joblib1.py
python -u pyross_example_script.py
#python -u run_sbi_multinode.py
#python -u simple-trainer-2.py
