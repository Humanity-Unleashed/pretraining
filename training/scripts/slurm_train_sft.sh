#!/bin/bash
#SBATCH --job-name=sft_test
#SBATCH --account=<YOUR ACCOUNT>
#SBATCH --output=<YOUR LOGS>
#SBATCH --export=ALL
#SBATCH --nodes=1  
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2    
#SBATCH --cpus-per-task=24
#SBATCH --time=20:00:00
#SBATCH --mem=250GB
#SBATCH --partition=<YOUR PARTITION>

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export GPUS_PER_NODE=2
export NNODES=$SLURM_NNODES
export NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT_CANDIDATES=(6000 6001 6002 6003)
# check if master port candidate is already being used; if not, use it as master port
for MPC in ${MASTER_PORT_CANDIDATES[@]}; do
    NUM_LISTENING_PROCESSES=$(lsof -Pi :${MPC} -sTCP:LISTEN | wc -l)
    if test $NUM_LISTENING_PROCESSES -eq 0; then
        MASTER_PORT=${MPC}
        export MASTER_PORT=${MPC}
        echo "Setting master port to ${MASTER_PORT}."
        break
    fi
done
if [ -z ${MASTER_PORT+x} ]; then
    echo "Could not find an available master port. Exiting."
    exit
fi

# Custom environment
source ~/.bashrc
conda deactivate
conda activate humun

module load cuda
module load cudnn
module load gcc/12.2.0-fasrc01

deepspeed --module humun_econ_transformer.train_sft \
   --max_len 512 \
   --dataset Open-Orca/OpenOrca \
   --input_key question \
   --output_key response \
   --apply_chat_template \
   --train_batch_size 256 \
   --micro_train_batch_size 8 \
   --max_samples 500000 \
   --pretrain Qwen/Qwen2-1.5B \
   --save_path ./checkpoint/qwen_2_sft \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 3 \
   --max_epochs 1 \
   --packing_samples \
   --bf16 \
   --flash_attn \
   --learning_rate 5e-6 \
   --gradient_checkpointing \
