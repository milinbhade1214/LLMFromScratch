#!/bin/bash


# baseline
python cs336_basics/sample.py \
    --context_length=256 \
    --batch_size=1 \
    --vocab_size=10000 \
    --d_model=512 \
    --d_ff=2048 \
    --attn_pdrop=0.0 \
    --resid_pdrop=0.0 \
    --num_layers=4 \
    --num_heads=16 \

