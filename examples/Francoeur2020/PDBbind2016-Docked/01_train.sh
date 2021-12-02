#!/bin/bash

# PDBbind Docked

epochs=1000
test_every=5


# GPU: 0
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
    --test_every ${test_every} \
    --checkpoint_every 50 \
    --num_checkpoints 10 \
    --balanced \
    --stratify_receptor \
    --label_pos 0 \
    --affinity_pos 1 \
    --scale_affinity_loss 0.05 \
    --silent \
    --seed ${i} \
    --gpu cuda:0 \
    --out_dir out${i} &
done


# GPU: 1
for i in 4 5 6
do
nohup \
python -m gnina.training \
    ../data/types/ref_uff_train0.types \
    --testfile ../data/types/ref_uff_test0.types \
    --data_root ../data/ \
    --model default2018 \
    --batch_size 50 \
    --iterations ${epochs} \
    --test_every ${test_every} \
    --checkpoint_every 50 \
    --num_checkpoints 10 \
    --balanced \
    --stratify_receptor \
    --label_pos 0 \
    --affinity_pos 1 \
    --scale_affinity_loss 0.05 \
    --silent \
    --seed ${i} \
    --gpu cuda:1 \
    --out_dir out${i} &
done
