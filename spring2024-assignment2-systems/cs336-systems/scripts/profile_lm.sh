#!/bin/bash


python cs336-systems/cs336_systems/benchmarking_lm.py \
    --wandb_run_name "benchmarking_small" \
    --d_model 768 \
    --d_ff 3072 \
    --num_layers 12 \
    --num_heads 12

python cs336-systems/cs336_systems/benchmarking_lm.py \
    --wandb_run_name "benchmarking_medium" \
    --d_model 1024 \
    --d_ff 4096 \
    --num_layers 24 \
    --num_heads 16

python cs336-systems/cs336_systems/benchmarking_lm.py \
    --wandb_run_name "benchmarking_large" \
    --d_model 1280 \
    --d_ff 5120 \
    --num_layers 36 \
    --num_heads 20

python cs336-systems/cs336_systems/benchmarking_lm.py \
    --wandb_run_name "benchmarking_xl" \
    --d_model 1600 \
    --d_ff 6400 \
    --num_layers 48 \
    --num_heads 25

python cs336-systems/cs336_systems/benchmarking_lm.py \
    --wandb_run_name "benchmarking_2p7b" \
    --d_model 2560 \
    --d_ff 10240 \
    --num_layers 32 \
    --num_heads 32