#!/bin/bash

#SBATCH --job-name=mini_unimax
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=96

NUM_NODES=1

# SLURM Node configuration
nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Node IP: $head_node_ip"

mkdir -p logs
# Environment setup
export LOGLEVEL=INFO
export NCCL_BUFFSIZE=2097152
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME="eth0,en,eth,em,bond"
export UCX_IB_PCI_RELAXED_ORDERING=on \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
NCCL_IB_PCI_RELAXED_ORDERING=1 \
NCCL_TOPO_FILE=/opt/microsoft/ndv5-topo.xml \
UCX_NET_DEVICES=eth0 \
NCCL_DEBUG=INFO \
NCCL_IGNORE_DISABLED_P2P=1 \
OMPI_MCA_coll_hcoll_enable=0

source /opt/hpcx-v2.18-gcc-mlnx_ofed-ubuntu22.04-cuda12-x86_64/hpcx-init.sh

# Azure Slurm MASTER Addr & Ports
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export HF_HOME="/anf/tmp"
export LM_WORK="/nfsdata/languageAI"
export TRITON_CACHE_DIR="/tmp"
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=7200
export NCCL_DEBUG=INFO

srun -l \
    --container-workdir="$PWD" \
    --container-mounts="$PWD:$PWD,/anf:/anf,/nfsdata:/nfsdata" \
    --container-image="../docker_images/nanotron_v2.sqsh" \
    --output=logs/%x-%j.err \
    --error=logs/%x-%j.log \
    torchrun \
    --nnodes="$NUM_NODES" \
    --nproc_per_node=8 \
    --rdzv_id=101 \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$head_node_ip:29500" \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    ./run_train_ok.py --config-file examples/config_smollm_qwen_1b_unimax.yaml
