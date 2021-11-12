#!/bin/bash

nohup \
python -m gnina.training \
    ../data/types/ref_uff_train0.types \
    --testfile ../data/types/ref_uff_test0.types \
    --data_root ../data/ \
    --model default2018 \
    --batch_size 50 \
    --iterations 200 \
    --test_every 5 \
    --checkpoint_every 5 \
    --balanced \
    --stratify_receptor \
    --label_pos 0 \
    --affinity_pos 1 \
    --silent \
    --out_dir out &
