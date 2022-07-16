#!/bin/bash

# PDBbind Crystal
# No balancing because all poses are good poses (crystal poses)
# No receptor stratification

epochs=750
test_every=5

# GPU 0
for i in 1 2 3
do
nohup \
python -m gninatorch.training \
    ../data/types/ref2_crystal_train0.types \
    --testfile ../data/types/ref2_crystal_test0.types \
    --data_root ../data/ \
    --model default2018 \
    --batch_size 50 \
    --iterations ${epochs} \
    --test_every ${test_every} \
    --checkpoint_every 50 \
    --label_pos 0 \
    --affinity_pos 1 \
    --no_roc_auc \
    --silent \
    --seed ${i} \
    --gpu cuda:0 \
    --out_dir out${i} &
done

# GPU 1
for i in 4 5 6
do
nohup \
python -m gninatorch.training \
    ../data/types/ref2_crystal_train0.types \
    --testfile ../data/types/ref2_crystal_test0.types \
    --data_root ../data/ \
    --model default2018 \
    --batch_size 50 \
    --iterations ${epochs} \
    --test_every ${test_every} \
    --checkpoint_every 50 \
    --label_pos 0 \
    --affinity_pos 1 \
    --no_roc_auc \
    --silent \
    --seed ${i} \
    --gpu cuda:1 \
    --out_dir out${i} &
done
