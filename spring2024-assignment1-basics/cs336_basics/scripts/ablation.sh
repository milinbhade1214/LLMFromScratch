#!/bin/bash

# baseline
# python cs336_basics/train.py \
#     --dataset_name='tinystories' \
#     --context_length=256 \
#     --batch_size=128 \
#     --vocab_size=10000 \
#     --d_model=512 \
#     --d_ff=2048 \
#     --attn_pdrop=0.0 \
#     --resid_pdrop=0.0 \
#     --num_layers=4 \
#     --num_heads=16 \
#     --lr_max=0.001 \
#     --total_iters=10000 \
#     --wandb_project='cs336_basics' \
#     --wandb_run_name="tinystories_ablation_baseline" \
#     --wandb_logging=True

# no_rmsnorm=True
# python cs336_basics/train.py \
#     --dataset_name='tinystories' \
#     --context_length=256 \
#     --batch_size=128 \
#     --vocab_size=10000 \
#     --d_model=512 \
#     --d_ff=2048 \
#     --attn_pdrop=0.0 \
#     --resid_pdrop=0.0 \
#     --num_layers=4 \
#     --num_heads=16 \
#     --lr_max=0.001 \
#     --total_iters=10000 \
#     --wandb_project='cs336_basics' \
#     --wandb_run_name="tinystories_ablation_no_rmsnorm" \
#     --wandb_logging=True \
#     --no_rmsnorm=True

# parallel_layers=True
# python cs336_basics/train.py \
#     --dataset_name='tinystories' \
#     --context_length=256 \
#     --batch_size=128 \
#     --vocab_size=10000 \
#     --d_model=512 \
#     --d_ff=2048 \
#     --attn_pdrop=0.0 \
#     --resid_pdrop=0.0 \
#     --num_layers=4 \
#     --num_heads=16 \
#     --lr_max=0.001 \
#     --total_iters=10000 \
#     --wandb_project='cs336_basics' \
#     --wandb_run_name="tinystories_ablation_parallel_layers" \
#     --wandb_logging=True \
#     --parallel_layers=True

# post_norm=True
python cs336_basics/train.py \
    --dataset_name='tinystories' \
    --context_length=256 \
    --batch_size=128 \
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
    --wandb_run_name="tinystories_ablation_post_norm" \
    --wandb_logging=True \
    --post_norm=True
