#!/bin/bash

checkpoint="checkpoint_750.pt"

for i in 1 2 3 4 5 6
do
nohup \
python -m gnina.inference \
    ../data/types/ref_uff_test0.types \
    default2018 \
    out${i}/${checkpoint} \
    --data_root ../data/ \
    --batch_size 1 \
    --label_pos 0 \
    --affinity_pos 1 \
    --no_roc_auc \
    --out_dir out${i} &
done
