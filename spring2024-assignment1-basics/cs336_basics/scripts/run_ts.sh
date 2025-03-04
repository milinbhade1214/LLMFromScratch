#!/bin/bash

# baseline
python cs336_basics/train.py \
    --dataset_name='ts' \
    --context_length=256 \
    --batch_size=64 \
    --vocab_size=10000 \
    --d_model=512 \
    --d_ff=2048 \
    --attn_pdrop=0.0 \
    --resid_pdrop=0.0 \
    --num_layers=4 \
    --num_heads=16 \
    --lr_max=0.0005 \
    --total_iters=20000 \
    --wandb_project='cs336_basics' \
    --wandb_run_name="leaderboard" \
    --wandb_logging=True
