#!/bin/bash

python cs336_systems/profile_memory.py \
    --wandb_run_name "profiling_2p7b_memory_forward_mixed" \
    --d_model  1280 \
    --d_ff  5120 \
    --num_layers 36 \
    --num_heads 20 \
    --only_forward=True \
    --mixed_precision=True
