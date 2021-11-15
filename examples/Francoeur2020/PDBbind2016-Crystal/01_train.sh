#!/bin/bash

epochs=500

# PDBbind Crystal
# No balancing because all poses are good poses (crystal poses)
# No receptor stratification
for i in 1 2 3
do
nohup \
python -m gnina.training \
    ../data/types/ref2_crystal_train0.types \
    --testfile ../data/types/ref2_crystal_test0.types \
    --data_root ../data/ \
    --model default2018 \
    --batch_size 50 \
    --iterations ${epochs} \
    --test_every 2 \
    --checkpoint_every 10 \
    --label_pos 0 \
    --affinity_pos 1 \
    --no_roc_auc \
    --silent \
    --seed ${i} \
    --out_dir out${i} &
done
