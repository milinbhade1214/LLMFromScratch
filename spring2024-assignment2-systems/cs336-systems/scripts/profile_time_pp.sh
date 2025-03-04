#!/bin/bash


python cs336_systems/profile_time_pytorchprof.py \
    --context_length=128 \
    --batch_size=16 \
    --vocab_size=10000 \
    --d_model=1280 \
    --d_ff=5120 \
    --attn_pdrop=0.0 \
    --residual_pdrop=0.0 \
    --num_layers=36 \
    --num_heads=20 \
    --enable_backward=False \
    --warmup_steps=3 \
    --num_steps=10 \
    --profile_memory=False