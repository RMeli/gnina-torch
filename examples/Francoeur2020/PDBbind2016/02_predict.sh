#!/bin/bash

python -m gnina.inference \
    ../data/types/ref2_crystal_test0.types \
    default2018 \
    out/checkpoint_100.pt \
    --data_root ../data/ \
    --batch_size 1 \
    --label_pos 0 \
    --affinity_pos 1 \
    --no_roc_auc \
    --out_dir out
