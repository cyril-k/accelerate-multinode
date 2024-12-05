# accelerate-multinode

Run it on a Slurm cluster with the following command:

`sbatch --exclusive run.sh`

Contents of `run.sh`:
``` bash
#!/bin/bash
#SBATCH --job-name=test-1
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=100
#SBATCH --mem=1000GB
#SBATCH --output=logs/run-%j.out
#SBATCH --error=logs/run-%j.err

LOG_PATH=./logs/run-$SLURM_JOB_ID.log

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=20
export NCCL_DEBUG=INFO


_hf_cache_dir_mt="/root/.cache/huggingface:/root/.cache/huggingface"

CONTAINER_IMAGE='docker://ghcr.io#cyril-k/accelerate-multinode:latest'
CONTAINER_NAME="accelerate-multinode-${SLURM_JOB_ID}"
CONTAINER_MOUNTS="${_hf_cache_dir_mt}"
CONTAINER_WORK_DIR="/workspace"

head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)


GPUS_PER_NODE=8
NNODES=2
NODE_RANK=$SLURM_PROCID
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
MASTER_PORT=29500

echo "START TIME: $(date)"
echo "MASTER_ADDR ${MASTER_ADDR}"

SCRIPT="/workspace/complete_nlp_example.py"
SCRIPT_ARGS="--mixed_precision fp16 \
    --output_dir /workspace/output"

CMD="$SCRIPT $SCRIPT_ARGS"
LAUNCHER="accelerate launch \
--num_processes $WORLD_SIZE \
--num_machines $NNODES \
--machine_rank $NODE_RANK \
--rdzv_backend c10d \
--main_process_ip $head_node_ip \
--main_process_port 29500 \
--multi_gpu \
--dynamo_backend no \
--mixed_precision bf16"

srun \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --container-name="${CONTAINER_NAME}" \
    --container-workdir="${CONTAINER_WORK_DIR}" \
    --export=all \
    bash -c "$LAUNCHER $CMD" 2>&1 | tee $LOG_PATH

echo "END TIME: $(date)"
```