#!/bin/bash

# Define the range of learning rates
learning_rates=(0.001 0.0005 0.0001 0.00005 0.00001)

# Loop through each learning rate
for lr in "${learning_rates[@]}"
do
    echo "Running training with learning rate: $lr"
    python cs336_basics/train.py \
        --dataset_name='ts' \
        --context_length=256 \
        --batch_size=128 \
        --vocab_size=10000 \
        --d_model=512 \
        --d_ff=2048 \
        --attn_pdrop=0.0 \
        --resid_pdrop=0.0 \
        --num_layers=4 \
        --num_heads=16 \
        --lr_max=$lr \
        --total_iters=10000 \
        --wandb_project='cs336_basics' \
        --wandb_run_name="tinystories_lr_${lr}" \
        --wandb_logging=True
done
