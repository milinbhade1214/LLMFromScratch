#!/bin/bash

# baseline
python cs336_basics/train.py \
    --dataset_name='owt' \
    --context_length=256 \
    --batch_size=64 \
    --vocab_size=50304 \
    --d_model=768 \
    --d_ff=3072 \
    --attn_pdrop=0.0 \
    --resid_pdrop=0.0 \
    --num_layers=12 \
    --num_heads=12 \
    --lr_max=0.0005 \
    --total_iters=20000 \
    --wandb_project='cs336_basics' \
    --wandb_run_name="leaderboard" \
    --wandb_logging=True
