#!/bin/bash

epochs=500

# PDBbind Docked
for i in 1 2 3
do
nohup \
python -m gnina.training \
    ../data/types/ref_uff_train0.types \
    --testfile ../data/types/ref_uff_test0.types \
    --data_root ../data/ \
    --model default2018 \
    --batch_size 50 \
    --iterations ${epochs} \
    --test_every 2 \
    --checkpoint_every 10 \
    --balanced \
    --stratify_receptor \
    --label_pos 0 \
    --affinity_pos 1 \
    --silent \
    --seed ${i} \
    --out_dir out${i} &
done
