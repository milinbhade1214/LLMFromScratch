#!/bin/bash


python cs336_systems/profile_time.py \
    --context_length=128 \
    --batch_size=16 \
    --vocab_size=10000 \
    --d_model=768 \
    --d_ff=3072 \
    --attn_pdrop=0.0 \
    --residual_pdrop=0.0 \
    --num_layers=12 \
    --num_heads=12 \
    --enable_backward=True \
    --warmup_steps=3 \
    --num_steps=10 \
    --profile_memory=False