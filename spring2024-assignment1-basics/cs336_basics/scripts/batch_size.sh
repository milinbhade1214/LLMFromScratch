#!/bin/bash
#SBATCH --job-name=batch_size_sweep
#SBATCH --output=logs/batch_size_%A_%a.out
#SBATCH --error=logs/batch_size_%A_%a.err
#SBATCH --partition=med_24h_4gpu
#SBATCH --time=24:00:00

# Define the range of batch sizes
batch_sizes=(1 2 4 8 16 32 64 128 256 512 1024)

# Loop through each batch size
for batch_size in "${batch_sizes[@]}"
do
    echo "Running training with batch size: $batch_size"
    python cs336_basics/train.py \
        --dataset_name='ts' \
        --context_length=256 \
        --batch_size=$batch_size \
        --vocab_size=10000 \
        --d_model=512 \
        --d_ff=2048 \
        --attn_pdrop=0.0 \
        --resid_pdrop=0.0 \
        --num_layers=4 \
        --num_heads=16 \
        --lr_max=0.001 \
        --total_iters=10000 \
        --wandb_project='cs336_basics' \
        --wandb_run_name="tinystories_batchsize_${batch_size}" \
        --wandb_logging=True \
        --eval_iters=1
done
